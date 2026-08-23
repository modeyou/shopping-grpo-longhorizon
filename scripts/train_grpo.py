#!/usr/bin/env python3
"""Run the repository's single supported Shopping Agent GRPO recipe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from shopping_grpo.training.grpo.data_manifest import validate_grpo_data_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/environment-v4.json"
DEFAULT_MODEL = ROOT / "outputs/models/sft-merged"
DEFAULT_TRAIN_DATA = ROOT / "data/grpo/formal-v2/multiturn-train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/grpo/formal-v2/multiturn-validation.parquet"


DEFAULT_DATA_MANIFEST = ROOT / "data/grpo/formal-v2/manifest.json"


def _model_has_weights(path: Path) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--shopper-model", default="deepseek-v4-flash-0731")
    parser.add_argument("--shopper-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument(
        "--logger",
        choices=("console", "swanlab"),
        default="console",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("GRPO_SEED", "20260823")),
    )
    parser.add_argument(
        "--reward-profile",
        choices=("none", "bounded-v1"),
        default=os.environ.get("SHOPPING_REWARD_SHAPING_PROFILE", "none"),
        help="training-only reward profile; Reward v4 itself is never changed",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=DEFAULT_DATA_MANIFEST,
        help="已验收的 Reward v4 GRPO 数据 manifest",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved veRL command without running preflight or training",
    )
    execution.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the full GRPO runtime preflight without loading the model or training",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args()


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def _hydra_overrides(args: argparse.Namespace) -> list[str]:
    logger_override = (
        "trainer.logger=[console,swanlab]"
        if args.logger == "swanlab"
        else "trainer.logger=[console]"
    )
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return [
        logger_override,
        f"trainer.experiment_name={args.experiment_name}",
        *extra,
    ]


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(
            "model directory has no supported weight file or sharded index: "
            f"{model}"
        )
    train_data = _validated_path(args.train_data, "train parquet")
    val_data = _validated_path(args.val_data, "validation parquet")
    config = _validated_path(args.config, "GRPO example config")
    data_manifest = args.data_manifest.expanduser().resolve()
    if not data_manifest.is_file():
        raise SystemExit(f"GRPO data manifest is missing: {data_manifest}")
    try:
        validate_grpo_data_manifest(
            data_manifest,
            train_data=train_data,
            validation_data=val_data,
            environment_manifest=DEFAULT_MANIFEST,
            root=ROOT,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid GRPO data manifest: {exc}") from exc
    output = args.output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        if any(output.iterdir()):
            raise SystemExit(f"output directory must be new or empty: {output}")
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")
    shopper_api_key = os.environ.get("SHOPPER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if (not args.shopper_base_url or not shopper_api_key) and not args.dry_run:
        raise SystemExit(
            "multi-turn GRPO requires --shopper-base-url and SHOPPER_API_KEY "
            "(OPENAI_API_KEY is accepted as a fallback)"
        )

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            # veRL's colocated FSDP workers select devices by local rank. Ray's
            # default per-actor CUDA_VISIBLE_DEVICES rewrite makes every rank see
            # a one-device namespace and can map all ranks onto cuda:0.
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "SHOP_REWARD_VERSION": "shopsimulator-reward-v4",
            "GRPO_SEED": str(args.seed),
            "SHOPPING_REWARD_SHAPING_PROFILE": str(args.reward_profile),
            "SHOPPER_MODEL": str(args.shopper_model),
            "SHOPPER_BASE_URL": str(args.shopper_base_url or ""),
            "SHOPPER_API_KEY": shopper_api_key or "",
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "GRPO_OUTPUT_DIR": str(output),
            "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(
                output / "training_diagnostics.jsonl"
            ),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "GRPO_CONFIG_NAME": config.stem,
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    environment["SHOPPING_GRPO_DATA_MANIFEST"] = str(data_manifest)
    overrides = _hydra_overrides(args)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *overrides,
    ]
    return command, environment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_contract(audit: dict, environment: dict[str, str]) -> Path:
    """Persist a secret-free, machine-verifiable GRPO launch contract."""
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
    )
    git_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
    )
    input_paths = {
        "train_data": Path(environment["GRPO_TRAIN_FILE"]),
        "validation_data": Path(environment["GRPO_VAL_FILE"]),
        "grpo_config": Path(audit["config"]),
        "agent_loop_config": Path(environment["SHOPPING_AGENT_LOOP_CONFIG"]),
        "tool_config": Path(environment["SHOPPING_TOOL_CONFIG"]),
        "environment_manifest": Path(environment["SHOPPING_ENV_MANIFEST"]),
        "model_config": Path(environment["GRPO_MODEL_PATH"]) / "config.json",
    }
    input_paths["data_manifest"] = Path(
        environment["SHOPPING_GRPO_DATA_MANIFEST"]
    )
    contract = {
        "schema_version": "shopping-grpo-run-contract-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_commit,
            "dirty": bool(git_status),
            "diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        },
        "launch": audit,
        "runtime_contract": {
            "environment_version": environment["SHOPPING_ENVIRONMENT_VERSION"],
            "reward_version": environment["SHOP_REWARD_VERSION"],
            "reward_profile": environment["SHOPPING_REWARD_SHAPING_PROFILE"],
            "seed": int(environment["GRPO_SEED"]),
            "shopper_model": environment["SHOPPER_MODEL"],
            "shopper_base_url": environment["SHOPPER_BASE_URL"],
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for name, path in input_paths.items()
        },
    }
    destination = Path(environment["GRPO_OUTPUT_DIR"]) / "run_contract.json"
    destination.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    args = parse_args()
    command, environment = build_command(args)
    audit = {
        "command": command,
        "model": environment["GRPO_MODEL_PATH"],
        "train_data": environment["GRPO_TRAIN_FILE"],
        "val_data": environment["GRPO_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["GRPO_OUTPUT_DIR"],
        "logger": args.logger,
        "reward_profile": environment["SHOPPING_REWARD_SHAPING_PROFILE"],
        "seed": int(environment["GRPO_SEED"]),
        "shopper_model": environment["SHOPPER_MODEL"],
        "shopper_base_url": environment["SHOPPER_BASE_URL"],
        "config": str(args.config.resolve()),
    }
    audit["data_manifest"] = environment["SHOPPING_GRPO_DATA_MANIFEST"]
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    Path(environment["GRPO_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    overrides = _hydra_overrides(args)
    preflight = [
        sys.executable,
        str(ROOT / "scripts/check_grpo_runtime.py"),
        *overrides,
    ]
    preflight_status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if preflight_status:
        raise SystemExit(preflight_status)
    if args.preflight_only:
        print("GRPO runtime preflight-only passed")
        return
    contract_path = write_run_contract(audit, environment)
    print(f"GRPO run contract written: {contract_path}")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
