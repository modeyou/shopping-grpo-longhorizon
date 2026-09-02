#!/usr/bin/env python3
"""Evaluate one exported model on frozen multi-turn development or final assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

try:
    from scripts.run_sft_checkpoint_sweep import (
        CONDITIONS,
        build_actor_command,
        checkpoint_result,
        read_jsonl,
        sha256,
        stop_actors,
        validate_assets,
        validate_shopper_api,
        wait_for_actor,
    )
except ModuleNotFoundError:
    from run_sft_checkpoint_sweep import (  # type: ignore[no-redef]
        CONDITIONS,
        build_actor_command,
        checkpoint_result,
        read_jsonl,
        sha256,
        stop_actors,
        validate_assets,
        validate_shopper_api,
        wait_for_actor,
    )


ROOT = Path(__file__).resolve().parents[1]


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def validate_exported_model(model: Path) -> dict:
    if not model.is_dir():
        raise ValueError(f"exported model directory is missing: {model}")
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"exported model config is missing: {config}")
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"exported model weights are missing: {model}")
    return {
        "config_sha256": sha256(config),
        "weight_files": [
            {"name": path.name, "bytes": path.stat().st_size} for path in weights
        ],
    }


def validate_source_checkpoint(source: Path) -> dict:
    if not source.is_dir() or not source.name.startswith("global_step_"):
        raise ValueError(f"invalid veRL global-step checkpoint: {source}")
    actor = source / "actor"
    if not actor.is_dir() or not any(actor.iterdir()):
        raise ValueError(f"actor checkpoint is missing or empty: {actor}")
    try:
        step = int(source.name.removeprefix("global_step_"))
    except ValueError as exc:
        raise ValueError(f"invalid checkpoint step: {source.name}") from exc
    tracker = source.parent / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError(f"checkpoint tracker is missing: {tracker}")
    recorded_step = int(tracker.read_text(encoding="utf-8").strip())
    if recorded_step < step:
        raise ValueError(
            f"checkpoint tracker is behind source step: {recorded_step} < {step}"
        )
    return {
        "path": str(source),
        "actor_path": str(actor),
        "step": step,
        "latest_checkpointed_iteration": recorded_step,
    }


def validate_final_assets(assets: Path) -> tuple[int, str]:
    def normalized_sha(path: Path) -> str:
        payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(payload).hexdigest()

    manifest_path = assets / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"final asset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "shopping-multiturn-final-subset-v1":
        raise ValueError("unsupported final asset manifest")
    if manifest.get("evaluation_role") != "final":
        raise ValueError("final assets must declare evaluation_role=final")
    if manifest.get("final_evaluation_used") is not True:
        raise ValueError("final assets must declare final_evaluation_used=true")
    if manifest.get("reward_contract") != "shopsimulator-reward-v4":
        raise ValueError("final evaluation must use Reward v4 assets")
    if int(manifest.get("task_count") or 0) != 200:
        raise ValueError("the frozen final evaluation must contain 200 tasks")

    task_sets = {}
    for name in ("tasks", "gap_openings", "complete_openings"):
        path = assets / f"{name}.jsonl"
        rows = read_jsonl(path)
        task_ids = [int(row["task_id"]) for row in rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"duplicate task_id in {path}")
        if normalized_sha(path) != manifest["subset_sha256"][name]:
            raise ValueError(f"final asset SHA256 mismatch: {path}")
        task_sets[name] = set(task_ids)
    if not (
        task_sets["tasks"]
        == task_sets["gap_openings"]
        == task_sets["complete_openings"]
    ):
        raise ValueError("final task/opening sets do not match")

    conditions_path = assets / "conditions.jsonl"
    if normalized_sha(conditions_path) != manifest["subset_sha256"]["conditions"]:
        raise ValueError(f"final asset SHA256 mismatch: {conditions_path}")
    condition_rows = read_jsonl(conditions_path)
    expected_conditions = set(CONDITIONS)
    by_task = {task_id: set() for task_id in task_sets["tasks"]}
    for row in condition_rows:
        task_id = int(row["task_id"])
        if task_id not in by_task:
            raise ValueError(f"condition references unknown final task: {task_id}")
        by_task[task_id].add(str(row["condition"]))
    if any(values != expected_conditions for values in by_task.values()):
        raise ValueError("each final task must have the frozen G+/G-/C+ conditions")
    return len(task_sets["tasks"]), normalized_sha(manifest_path)


def validate_shopsimulator(base_url: str, task_id: int) -> None:
    from shopping_grpo.environment.client import ShopAgentEnv

    env = ShopAgentEnv(base_url=base_url, timeout=60, multiturn=True)
    try:
        result = env.reset(task_id, initial_request="")
    finally:
        env.release()
    if result.get("environment_version") != "shopsimulator-environment-v2.1":
        raise RuntimeError("ShopSimulator environment version mismatch")
    if result.get("reward_version") != "shopsimulator-reward-v4":
        raise RuntimeError("ShopSimulator reward version mismatch")
    print("ShopSimulator Environment v2.1 / Reward v4 preflight passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument(
        "--evaluation-role", choices=("dev", "final"), default="dev"
    )
    parser.add_argument("--assets", type=Path, required=True)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partial shards only when the saved run plan is identical.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.actor_ports) != len(args.gpu_indices):
        raise SystemExit("--actor-ports and --gpu-indices must have equal lengths")
    if len(set(args.actor_ports)) != len(args.actor_ports):
        raise SystemExit("Actor ports must be unique")
    if len(set(args.gpu_indices)) != len(args.gpu_indices):
        raise SystemExit("GPU indices must be unique")

    model = args.model.resolve()
    source_checkpoint = (
        args.source_checkpoint.resolve() if args.source_checkpoint else None
    )
    assets = args.assets.resolve()
    output_root = args.output_root.resolve()
    actor_log_root = args.actor_log_root.resolve()
    model_audit = validate_exported_model(model)
    if args.evaluation_role == "dev" and source_checkpoint is None:
        raise SystemExit("--source-checkpoint is required for a dev RL evaluation")
    source_audit = (
        validate_source_checkpoint(source_checkpoint)
        if source_checkpoint is not None
        else None
    )
    if args.evaluation_role == "final":
        expected_tasks, asset_manifest_sha = validate_final_assets(assets)
    else:
        expected_tasks, asset_manifest_sha = validate_assets(assets)
    if not args.vllm_bin.resolve().is_file():
        raise SystemExit(f"vLLM executable is missing: {args.vllm_bin}")

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
        "schema_version": (
            "shopping-final-model-evaluation-v1"
            if args.evaluation_role == "final"
            else "shopping-standalone-checkpoint-evaluation-v1"
        ),
        "evaluation_role": args.evaluation_role,
        "model": str(model),
        "model_name": args.model_name,
        "model_audit": model_audit,
        "source_checkpoint": source_audit,
        "reward_contract": "shopsimulator-reward-v4",
        "expected_tasks_per_condition": expected_tasks,
        "conditions": CONDITIONS,
        "asset_manifest_sha256": asset_manifest_sha,
        "actor_ports": args.actor_ports,
        "gpu_indices": args.gpu_indices,
        "final_evaluation_used": args.evaluation_role == "final",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        task_rows = read_jsonl(assets / "tasks.jsonl")
        validate_shopsimulator(
            os.environ["SHOPSIM_BASE_URL"], int(task_rows[0]["task_id"])
        )
        validate_shopper_api(
            base_url=os.environ["SHOPPER_BASE_URL"],
            api_key=os.environ["SHOPPER_API_KEY"],
            model=os.environ["SHOPPER_MODEL"],
        )
    if args.preflight_only:
        print("STANDALONE CHECKPOINT EVALUATION PREFLIGHT PASSED")
        return
    if args.dry_run:
        for gpu, port in zip(args.gpu_indices, args.actor_ports, strict=True):
            command = build_actor_command(
                vllm_bin=args.vllm_bin.resolve(),
                model=model,
                model_name=args.model_name,
                host=args.actor_host,
                port=port,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            print(f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(command), flush=True)
        print("STANDALONE CHECKPOINT EVALUATION DRY RUN PASSED")
        return

    output_nonempty = output_root.exists() and any(output_root.iterdir())
    logs_nonempty = actor_log_root.exists() and any(actor_log_root.iterdir())
    if (output_nonempty or logs_nonempty) and not args.resume:
        raise SystemExit(
            "evaluation output/logs must be new or empty; pass --resume for an identical partial run"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    actor_log_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "evaluation_plan.json"
    if args.resume:
        if not plan_path.is_file():
            raise SystemExit(f"resume plan is missing: {plan_path}")
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise SystemExit("resume plan does not match the current model/assets/config")
    else:
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    processes = []
    log_handles = []
    try:
        actor_urls = []
        for gpu, port in zip(args.gpu_indices, args.actor_ports, strict=True):
            actor_url = f"http://{args.actor_host}:{port}/v1"
            actor_urls.append(actor_url)
            log_path = actor_log_root / f"gpu{gpu}-port{port}.log"
            pid_path = actor_log_root / f"gpu{gpu}-port{port}.pid"
            command = build_actor_command(
                vllm_bin=args.vllm_bin.resolve(),
                model=model,
                model_name=args.model_name,
                host=args.actor_host,
                port=port,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            log_handle = log_path.open("ab" if args.resume else "xb")
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
                model_name=args.model_name,
                timeout=args.startup_timeout,
                log_path=log_path,
            )
            print(
                f"Actor ready: gpu={gpu} port={port} pid={process.pid}",
                flush=True,
            )

        environment = os.environ.copy()
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "MULTITURN_ASSET_DIR": str(assets),
                "MULTITURN_SHARDS": str(len(actor_urls)),
                "LLM_BASE_URLS": ",".join(actor_urls),
                "LLM_API_KEY": "EMPTY",
                "SERVED_MODEL_NAME": args.model_name,
                "EVAL_OUTPUT_DIR": str(output_root),
            }
        )
        environment.pop("MULTITURN_LIMIT", None)
        subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/evaluate_multiturn_parallel.sh"),
                args.model_name,
            ],
            check=True,
            env=environment,
        )
        result = checkpoint_result(output_root, expected_tasks, args.model_name)
        audit = {**plan, "result": result}
        audit_path = output_root / "evaluation_results.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"standalone checkpoint accepted; audit={audit_path}", flush=True)
        print("STANDALONE CHECKPOINT EVALUATION COMPLETED", flush=True)
    finally:
        stop_actors(processes)
        for handle in log_handles:
            handle.close()


if __name__ == "__main__":
    main()
