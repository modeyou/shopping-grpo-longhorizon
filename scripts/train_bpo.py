#!/usr/bin/env python3
"""Run the repository's independent formal full-BPO v1 recipe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from shopping_grpo.training.grpo.data_manifest import validate_grpo_data_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bpo.yaml"
AGENT_CONFIG = ROOT / "configs/bpo_agent_loop.yaml"
TOOL_CONFIG = ROOT / "configs/tools.json"
DATA_ENVIRONMENT_MANIFEST = ROOT / "data/environment-v4.json"
BPO_RUNTIME_MANIFEST = ROOT / "data/environment-bpo-v1.json"
DATA_MANIFEST = ROOT / "data/grpo/formal-v2/manifest.json"
TRAIN_DATA = ROOT / "data/grpo/formal-v2/multiturn-train.parquet"
VALIDATION_DATA = ROOT / "data/grpo/formal-v2/multiturn-validation.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=VALIDATION_DATA)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--shopper-model", default="deepseek-v4-flash-0731")
    parser.add_argument("--shopper-base-url", default=os.environ.get("SHOPPER_BASE_URL"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", default="shopping-agent-bpo-v1")
    parser.add_argument("--logger", choices=("console", "swanlab"), default="console")
    parser.add_argument("--seed", type=int, default=20260824)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("hydra_overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _weights_exist(model):
    return any(
        (model / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _file(path, description):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{description} is missing: {resolved}")
    return resolved


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _overrides(args):
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    logger = "[console,swanlab]" if args.logger == "swanlab" else "[console]"
    return [
        f"trainer.logger={logger}",
        f"trainer.experiment_name={args.experiment_name}",
        *extra,
    ]


def build(args):
    model = Path(args.model).expanduser().resolve()
    if not model.is_dir() or not (model / "config.json").is_file() or not _weights_exist(model):
        raise SystemExit(f"BPO model directory is incomplete: {model}")
    train_data = _file(args.train_data, "BPO train parquet")
    val_data = _file(args.val_data, "BPO validation parquet")
    manifest = _file(args.data_manifest, "BPO data manifest")
    try:
        validate_grpo_data_manifest(
            manifest,
            train_data=train_data,
            validation_data=val_data,
            environment_manifest=DATA_ENVIRONMENT_MANIFEST,
            root=ROOT,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid BPO data manifest: {exc}") from exc
    output = Path(args.output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"BPO output directory must be new or empty: {output}")
    key = os.environ.get("SHOPPER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and (not args.shopper_base_url or not key):
        raise SystemExit("BPO requires --shopper-base-url and SHOPPER_API_KEY")
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("BPO SwanLab logging requires SWANLAB_API_KEY")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "SHOPPING_BPO_ROOT": str(ROOT),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(BPO_RUNTIME_MANIFEST),
            "SHOP_REWARD_VERSION": "shopsimulator-reward-v4",
            "SHOPPING_REWARD_SHAPING_PROFILE": "none",
            "SHOPPING_LENGTH_SHAPING_ENABLE": "false",
            "SHOPPER_MODEL": str(args.shopper_model),
            "SHOPPER_BASE_URL": str(args.shopper_base_url or ""),
            "SHOPPER_API_KEY": key or "",
            "SHOPSIM_BASE_URL": str(args.env_url),
            "BPO_SEED": str(args.seed),
            "GRPO_SEED": str(args.seed),
            "BPO_MODEL_PATH": str(model),
            "BPO_TRAIN_FILE": str(train_data),
            "BPO_VAL_FILE": str(val_data),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "BPO_OUTPUT_DIR": str(output),
            "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(
                output / "training_diagnostics.jsonl"
            ),
            "SHOPPING_TOOL_CONFIG": str(TOOL_CONFIG),
            "SHOPPING_BPO_DATA_MANIFEST": str(manifest),
            "GRPO_CONFIG_NAME": "bpo",
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {"SWANLAB_MODE": "online", "SWANLAB_LOG_DIR": str(output / "swanlab")}
        )
    command = [
        sys.executable,
        "-m",
        "shopping_grpo.training.bpo.entrypoint",
        f"--config-path={CONFIG.parent}",
        f"--config-name={CONFIG.stem}",
        *_overrides(args),
    ]
    audit = {
        "algorithm": "full-bpo-v1",
        "command": command,
        "model": str(model),
        "train_data": str(train_data),
        "validation_data": str(val_data),
        "data_manifest": str(manifest),
        "output": str(output),
        "seed": args.seed,
        "shopper_model": args.shopper_model,
        "shopper_base_url": args.shopper_base_url,
        "logger": args.logger,
        "reward_version": "shopsimulator-reward-v4",
        "reward_profile": "none",
    }
    return command, environment, audit


def write_contract(environment, audit):
    input_paths = {
        "train_data": audit["train_data"],
        "validation_data": audit["validation_data"],
        "data_manifest": audit["data_manifest"],
        "bpo_config": CONFIG,
        "bpo_agent_config": AGENT_CONFIG,
        "tool_config": TOOL_CONFIG,
        "data_environment_manifest": DATA_ENVIRONMENT_MANIFEST,
        "bpo_runtime_manifest": BPO_RUNTIME_MANIFEST,
        "model_config": Path(audit["model"]) / "config.json",
    }
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT
    )
    contract = {
        "schema_version": "shopping-bpo-run-contract-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "dirty": bool(status),
        },
        "launch": audit,
        "frozen_method": {
            "branch_count": 1,
            "sibling_count": 4,
            "return_budget": 4,
            "branch_selection": "maximum_exact_entropy",
            "entropy_state": "action-boundary-first-token",
            "rollout_audit": "exact-tree-v1",
            "upstream_lambda": 0.95,
            "ppo_clip": 0.2,
            "gpu_memory_utilization": 0.45,
            "max_num_seqs": 8,
            "minimum_free_gpu_memory_gib": 20.0,
            "sparse_cuda_mapping": "physical-to-logical-v1",
            "use_fused_kernels": False,
            "use_liger": True,
            "use_remove_padding": True,
        },
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
    }
    destination = Path(environment["BPO_OUTPUT_DIR"]) / "run_contract.json"
    destination.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main():
    args = parse_args()
    command, environment, audit = build(args)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    Path(environment["BPO_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    preflight = [
        sys.executable,
        "-m",
        "scripts.check_bpo_runtime",
        *_overrides(args),
    ]
    status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if status:
        raise SystemExit(status)
    if args.preflight_only:
        print("BPO runtime preflight-only passed")
        return
    print(f"BPO run contract written: {write_contract(environment, audit)}")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
