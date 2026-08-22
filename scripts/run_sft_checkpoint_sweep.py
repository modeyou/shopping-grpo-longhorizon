#!/usr/bin/env python3
"""Merge and evaluate LoRA checkpoints on one frozen multi-turn dev subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "gap-ask-enabled",
    "gap-ask-disabled",
    "complete-ask-enabled",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_assets(assets: Path) -> tuple[int, str]:
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "shopping-sft-checkpoint-sweep-v1":
        raise ValueError("unsupported checkpoint sweep asset manifest")
    if manifest.get("reward_contract") != "shopsimulator-reward-v4":
        raise ValueError("checkpoint sweep must use Reward v4 assets")
    if manifest.get("final_evaluation_used") is not False:
        raise ValueError("checkpoint sweep must not use final evaluation tasks")

    task_sets = {}
    for name in ("tasks", "gap_openings", "complete_openings"):
        path = assets / f"{name}.jsonl"
        rows = read_jsonl(path)
        task_ids = [int(row["task_id"]) for row in rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"duplicate task_id in {path}")
        expected_sha = manifest["subset_sha256"][name]
        if sha256(path) != expected_sha:
            raise ValueError(f"asset SHA256 mismatch: {path}")
        task_sets[name] = set(task_ids)

    if not (
        task_sets["tasks"]
        == task_sets["gap_openings"]
        == task_sets["complete_openings"]
    ):
        raise ValueError("checkpoint sweep task/opening sets do not match")
    return len(task_sets["tasks"]), sha256(manifest_path)


def build_merge_command(
    *, python: str, base_model: Path, adapter: Path, output: Path
) -> list[str]:
    return [
        python,
        str(ROOT / "scripts/merge_lora_adapter.py"),
        "--base-model",
        str(base_model),
        "--adapter",
        str(adapter),
        "--output",
        str(output),
        "--bf16",
    ]


def build_actor_command(
    *,
    vllm_bin: Path,
    model: Path,
    model_name: str,
    host: str,
    port: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        str(vllm_bin),
        "serve",
        str(model),
        "--served-model-name",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--language-model-only",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
        "--trust-remote-code",
    ]


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def validate_existing_merge(merged: Path, adapter: Path) -> bool:
    if not merged.exists():
        return False
    if not any(merged.iterdir()):
        return False
    manifest_path = merged / "merge_manifest.json"
    weights = list(merged.glob("*.safetensors"))
    if not manifest_path.is_file() or not weights:
        raise ValueError(f"incomplete existing merged model: {merged}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = Path(manifest["source"]["adapter"]).resolve()
    if recorded != adapter.resolve():
        raise ValueError(
            f"merged model adapter mismatch: {recorded} != {adapter.resolve()}"
        )
    return True


def wait_for_actor(
    process: subprocess.Popen,
    *,
    url: str,
    model_name: str,
    timeout: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Actor exited with {process.returncode}; inspect {log_path}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.load(response)
            names = [item["id"] for item in payload["data"]]
            if model_name in names:
                return
        except Exception:  # Actor connection failures are expected while loading.
            pass
        time.sleep(2)
    raise TimeoutError(f"Actor readiness timeout; inspect {log_path}")


def stop_actors(processes: list[subprocess.Popen], timeout: int = 60) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        if process.poll() is None:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Actor PID {process.pid} did not stop after SIGTERM"
                ) from exc


def checkpoint_result(output: Path, expected_tasks: int, model_name: str) -> dict:
    result = {}
    for condition in CONDITIONS:
        condition_root = output / condition
        trajectories = condition_root / "trajectories.jsonl"
        summary_path = condition_root / "summary.json"
        rows = read_jsonl(trajectories)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(rows) != expected_tasks:
            raise ValueError(f"unexpected trajectory count: {trajectories}")
        if summary.get("completed_tasks") != expected_tasks:
            raise ValueError(f"incomplete checkpoint evaluation: {summary_path}")
        if summary.get("missing_tasks") != []:
            raise ValueError(f"missing checkpoint sweep tasks: {summary_path}")
        protocol = summary["protocol"]
        if protocol.get("model") != model_name:
            raise ValueError(f"model mismatch in {summary_path}")
        if protocol.get("reward_contract") != "shopsimulator-reward-v4":
            raise ValueError(f"Reward contract mismatch in {summary_path}")
        clarification = summary["clarification"]
        result[condition] = {
            "strict_successes": summary["strict_successes"],
            "strict_success_rate": summary["strict_success_rate"],
            "done_tasks": summary["done_tasks"],
            "reward_valid_tasks": summary["reward_valid_tasks"],
            "mean_final_reward": summary["mean_final_reward"],
            "guard_rejections": summary["context_projection"]["guard_rejections"],
            "asked_tasks": clarification["asked_tasks"],
            "grounded_questions": clarification["grounded_questions"],
            "complete_unnecessary_ask_tasks": clarification[
                "complete_unnecessary_ask_tasks"
            ],
            "trajectory_sha256": sha256(trajectories),
            "summary_sha256": sha256(summary_path),
        }
    result["derived"] = {
        "strict_successes": sum(
            result[condition]["strict_successes"] for condition in CONDITIONS
        ),
        "strict_success_rate": sum(
            result[condition]["strict_successes"] for condition in CONDITIONS
        )
        / (expected_tasks * len(CONDITIONS)),
        "gap_clarification_strict_gain": (
            result["gap-ask-enabled"]["strict_success_rate"]
            - result["gap-ask-disabled"]["strict_success_rate"]
        ),
        "complete_unnecessary_ask_rate": (
            result["complete-ask-enabled"]["complete_unnecessary_ask_tasks"]
            / expected_tasks
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--actor-log-root", type=Path, required=True)
    parser.add_argument("--vllm-bin", type=Path, required=True)
    parser.add_argument("--actor-host", default="127.0.0.1")
    parser.add_argument(
        "--actor-ports", type=int, nargs="+", default=[18102, 18103, 18104, 18105]
    )
    parser.add_argument("--gpu-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.actor_ports) != len(args.gpu_indices):
        raise SystemExit("--actor-ports and --gpu-indices must have equal lengths")
    if len(set(args.actor_ports)) != len(args.actor_ports):
        raise SystemExit("Actor ports must be unique")
    if len(set(args.gpu_indices)) != len(args.gpu_indices):
        raise SystemExit("GPU indices must be unique")
    if len(set(args.checkpoints)) != len(args.checkpoints):
        raise SystemExit("checkpoints must be unique")

    expected_tasks, asset_manifest_sha = validate_assets(args.assets.resolve())
    if not args.base_model.resolve().is_dir():
        raise SystemExit(f"base model directory is missing: {args.base_model}")
    if not args.vllm_bin.resolve().is_file():
        raise SystemExit(f"vLLM executable is missing: {args.vllm_bin}")
    for step in args.checkpoints:
        adapter = args.sft_root.resolve() / f"checkpoint-{step}"
        if not (adapter / "adapter_model.safetensors").is_file():
            raise SystemExit(f"adapter is missing: {adapter}")
    occupied = [
        port for port in args.actor_ports if not port_is_free(args.actor_host, port)
    ]
    if occupied:
        raise SystemExit(f"Actor ports are already occupied: {occupied}")

    required_environment = (
        "SHOPSIM_BASE_URL",
        "SHOPPER_BASE_URL",
        "SHOPPER_API_KEY",
        "SHOPPER_MODEL",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"required environment is missing: {missing}")

    plan = {
        "checkpoints": args.checkpoints,
        "expected_tasks_per_condition": expected_tasks,
        "conditions": CONDITIONS,
        "asset_manifest_sha256": asset_manifest_sha,
        "actor_ports": args.actor_ports,
        "gpu_indices": args.gpu_indices,
        "final_evaluation_used": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        print("SFT CHECKPOINT SWEEP PREFLIGHT PASSED")
        return

    args.merged_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.actor_log_root.mkdir(parents=True, exist_ok=True)
    completed = {}

    for step in args.checkpoints:
        label = f"checkpoint-{step}"
        adapter = args.sft_root.resolve() / label
        merged = args.merged_root.resolve() / label
        output = args.output_root.resolve() / label
        actor_log_dir = args.actor_log_root.resolve() / label
        model_name = f"qwen35-2b-sft-step{step}"

        merge_command = build_merge_command(
            python=sys.executable,
            base_model=args.base_model.resolve(),
            adapter=adapter,
            output=merged,
        )
        if args.dry_run:
            print(shlex.join(merge_command))
            continue
        if validate_existing_merge(merged, adapter):
            print(f"reuse verified merged model: {merged}", flush=True)
        else:
            if merged.exists() and any(merged.iterdir()):
                raise SystemExit(f"refusing incomplete merged output: {merged}")
            merge_environment = os.environ.copy()
            merge_environment["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(merge_command, check=True, env=merge_environment)

        actor_log_dir.mkdir(parents=True, exist_ok=True)
        processes = []
        log_handles = []
        try:
            actor_urls = []
            for gpu, port in zip(args.gpu_indices, args.actor_ports, strict=True):
                actor_url = f"http://{args.actor_host}:{port}/v1"
                actor_urls.append(actor_url)
                log_path = actor_log_dir / f"gpu{gpu}-port{port}.log"
                pid_path = actor_log_dir / f"gpu{gpu}-port{port}.pid"
                if log_path.exists() or pid_path.exists():
                    raise RuntimeError(f"refusing existing Actor artifact: {log_path}")
                command = build_actor_command(
                    vllm_bin=args.vllm_bin.resolve(),
                    model=merged,
                    model_name=model_name,
                    host=args.actor_host,
                    port=port,
                    max_model_len=args.max_model_len,
                    max_num_seqs=args.max_num_seqs,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                log_handle = log_path.open("wb")
                log_handles.append(log_handle)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
                processes.append(process)
                pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
                wait_for_actor(
                    process,
                    url=f"{actor_url}/models",
                    model_name=model_name,
                    timeout=args.startup_timeout,
                    log_path=log_path,
                )
                print(
                    f"Actor ready: step={step} gpu={gpu} port={port} pid={process.pid}",
                    flush=True,
                )

            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": sys.executable,
                    "MULTITURN_ASSET_DIR": str(args.assets.resolve()),
                    "MULTITURN_SHARDS": str(len(actor_urls)),
                    "LLM_BASE_URLS": ",".join(actor_urls),
                    "LLM_API_KEY": "EMPTY",
                    "SERVED_MODEL_NAME": model_name,
                    "EVAL_OUTPUT_DIR": str(output),
                }
            )
            environment.pop("MULTITURN_LIMIT", None)
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/evaluate_multiturn_parallel.sh"),
                    f"sft-sweep-step{step}",
                ],
                check=True,
                env=environment,
            )
            completed[label] = checkpoint_result(output, expected_tasks, model_name)
        finally:
            stop_actors(processes)
            for handle in log_handles:
                handle.close()

        audit = {
            "schema_version": "shopping-sft-checkpoint-sweep-results-v1",
            "reward_contract": "shopsimulator-reward-v4",
            "asset_manifest_sha256": asset_manifest_sha,
            "expected_tasks_per_condition": expected_tasks,
            "final_evaluation_used": False,
            "completed": completed,
        }
        audit_path = args.output_root / "sweep_results.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"checkpoint {step} accepted; audit={audit_path}", flush=True)

    if args.dry_run:
        print("SFT CHECKPOINT SWEEP DRY RUN PASSED")
    else:
        print("SFT CHECKPOINT SWEEP COMPLETED", flush=True)


if __name__ == "__main__":
    main()
