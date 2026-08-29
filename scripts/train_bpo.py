#!/usr/bin/env python3
"""Run the repository's independent formal CARL-BPO v1 recipe."""

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
from shopping_grpo.training.bpo.step0_validation import (
    build_validation_contract,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bpo.yaml"
AGENT_CONFIG = ROOT / "configs/bpo_agent_loop.yaml"
TOOL_CONFIG = ROOT / "configs/tools.json"
DATA_ENVIRONMENT_MANIFEST = ROOT / "data/environment-v4.json"
BPO_RUNTIME_MANIFEST = ROOT / "data/environment-bpo-v1.json"
DATA_MANIFEST = ROOT / "data/grpo/formal-v2/manifest.json"
TRAIN_DATA = ROOT / "data/grpo/formal-v2/multiturn-train.parquet"
VALIDATION_DATA = ROOT / "data/grpo/formal-v2/multiturn-validation.parquet"
STEP0_CACHE_ROOT = ROOT / "outputs/bpo/step0-validation-cache"


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
    parser.add_argument(
        "--experiment-name", default="carl-bpo-v1-step500-r4000-seed20260823"
    )
    parser.add_argument("--logger", choices=("console", "swanlab"), default="swanlab")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--diagnostic-steps",
        type=int,
        choices=(1,),
        help=(
            "Run one real parameter-update step with formal data and all "
            "gradient/delta gates enabled. This is a diagnostic run, not a "
            "formal checkpoint."
        ),
    )
    parser.add_argument(
        "--step0-cache-dir",
        type=Path,
        default=STEP0_CACHE_ROOT,
        help="Shared content-addressed cache for deterministic step-0 validation.",
    )
    parser.add_argument(
        "--refresh-step0-validation",
        action="store_true",
        help="Recompute and atomically replace the matching step-0 cache entry.",
    )
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

def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _step0_validation_contract(args, *, model, val_data, manifest):
    model_files = {
        f"model/{path.name}": path
        for path in sorted(model.iterdir())
        if path.is_file()
    }
    if not model_files:
        raise SystemExit(f"BPO model directory has no files: {model}")
    runtime_inputs = {
        "validation_data": val_data,
        "data_manifest": manifest,
        "bpo_config": CONFIG,
        "bpo_agent_config": AGENT_CONFIG,
        "tool_config": TOOL_CONFIG,
        "data_environment_manifest": DATA_ENVIRONMENT_MANIFEST,
        "bpo_runtime_manifest": BPO_RUNTIME_MANIFEST,
        "verl_dynamic_sampling_patch": (
            ROOT / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"
        ),
        "bpo_agent_loop": (
            ROOT / "src/shopping_grpo/training/bpo/agent_loop.py"
        ),
        "bpo_advantage": ROOT / "src/shopping_grpo/training/bpo/advantage.py",
        "bpo_branching": ROOT / "src/shopping_grpo/training/bpo/branching.py",
        "bpo_reward": ROOT / "src/shopping_grpo/training/bpo/reward.py",
        "bpo_step0_validation": (
            ROOT / "src/shopping_grpo/training/bpo/step0_validation.py"
        ),
        "bpo_runtime": ROOT / "src/shopping_grpo/training/bpo/runtime.py",
        "bpo_fused_ppo_gradient_patch": (
            ROOT / "src/shopping_grpo/training/bpo/fused_ppo_grad_patch.py"
        ),
        "grpo_agent_loop": (
            ROOT / "src/shopping_grpo/training/grpo/adapter/agent_loop.py"
        ),
        "grpo_session": (
            ROOT / "src/shopping_grpo/training/grpo/adapter/session.py"
        ),
        "grpo_runtime": (
            ROOT / "src/shopping_grpo/training/grpo/adapter/runtime.py"
        ),
        "grpo_dynamic_sampling": (
            ROOT / "src/shopping_grpo/training/grpo/dynamic_sampling.py"
        ),
        **model_files,
    }
    try:
        return build_validation_contract(
            root=ROOT,
            git_commit=_git_commit(),
            inputs=runtime_inputs,
            settings={
                "algorithm": "carl-bpo-v1",
                "environment_url": str(args.env_url),
                "reward_profile": "none",
                "reward_version": "shopsimulator-reward-v4",
                "seed": int(args.seed),
                "shopper_base_url": str(args.shopper_base_url or ""),
                "shopper_model": str(args.shopper_model),
                "validation_sampling": "deterministic-n1",
                "hydra_overrides": list(args.hydra_overrides),
                "diagnostic_steps": args.diagnostic_steps,
            },
        )
    except ValueError as exc:
        raise SystemExit(f"invalid BPO step-0 validation contract: {exc}") from exc


def _overrides(args):
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    logger = "[console,swanlab]" if args.logger == "swanlab" else "[console]"
    overrides = [
        f"trainer.logger={logger}",
        f"trainer.experiment_name={args.experiment_name}",
    ]
    if args.diagnostic_steps is not None:
        forbidden = (
            "trainer.total_training_steps=",
            "trainer.val_before_train=",
            "trainer.save_freq=",
            "trainer.test_freq=",
        )
        conflicts = [
            value for value in extra if str(value).startswith(forbidden)
        ]
        if conflicts:
            raise SystemExit(
                "--diagnostic-steps owns trainer step/save/test overrides: "
                + ", ".join(conflicts)
            )
        overrides.extend(
            [
                f"trainer.total_training_steps={args.diagnostic_steps}",
                "trainer.val_before_train=false",
                "trainer.save_freq=-1",
                "trainer.test_freq=-1",
            ]
        )
    overrides.extend(extra)
    return overrides


def validate_launcher_owned_ray(environ=None):
    """Reject attachment to a Ray head that lacks the launcher-owned runtime."""
    environment = os.environ if environ is None else environ
    address = str(environment.get("RAY_ADDRESS", "")).strip()
    if address:
        raise SystemExit(
            "CARL-BPO requires a launcher-owned local Ray runtime; stop the "
            "manually started Ray head and unset RAY_ADDRESS"
        )


def build(args):
    validate_launcher_owned_ray()
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

    step0_contract = _step0_validation_contract(
        args, model=model, val_data=val_data, manifest=manifest
    )
    step0_contract_sha256 = validate_contract(step0_contract)
    step0_cache_dir = Path(
        getattr(args, "step0_cache_dir", STEP0_CACHE_ROOT)
    ).expanduser().resolve()
    if step0_cache_dir.exists() and not step0_cache_dir.is_dir():
        raise SystemExit(
            f"BPO step-0 cache directory is not a directory: {step0_cache_dir}"
        )
    step0_cache_path = step0_cache_dir / f"{step0_contract_sha256}.json"

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "SHOPPING_BPO_ROOT": str(ROOT),
            "SHOPPING_BPO_REQUIRE_PARAMETER_UPDATE": "1",
            "SHOPPING_BPO_DIAGNOSTIC_STEPS": (
                str(args.diagnostic_steps)
                if args.diagnostic_steps is not None
                else ""
            ),
            "SHOPPING_BPO_SCHEDULER_HORIZON": "500",
            "SHOPPING_BPO_WARMUP_STEPS": "10",
            "SHOPPING_BPO_MIN_LR_RATIO": "0.1",
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
            "SHOPPING_BPO_STEP0_CACHE_PATH": str(step0_cache_path),
            "SHOPPING_BPO_STEP0_CONTRACT_SHA256": step0_contract_sha256,
            "SHOPPING_BPO_STEP0_CONTRACT_JSON": json.dumps(
                step0_contract, ensure_ascii=False, sort_keys=True
            ),
            "SHOPPING_BPO_STEP0_REFRESH": (
                "1" if getattr(args, "refresh_step0_validation", False) else "0"
            ),
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
        "algorithm": "carl-bpo-v1",
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
        "execution_mode": (
            "diagnostic" if args.diagnostic_steps is not None else "formal"
        ),
        "diagnostic_steps": args.diagnostic_steps,
        "step0_validation": {
            "cache_path": str(step0_cache_path),
            "contract_sha256": step0_contract_sha256,
            "refresh": bool(getattr(args, "refresh_step0_validation", False)),
        },
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
        "bpo_fused_ppo_gradient_patch": (
            ROOT / "src/shopping_grpo/training/bpo/fused_ppo_grad_patch.py"
        ),
        "bpo_agent_loop": ROOT / "src/shopping_grpo/training/bpo/agent_loop.py",
        "bpo_advantage": ROOT / "src/shopping_grpo/training/bpo/advantage.py",
        "bpo_branching": ROOT / "src/shopping_grpo/training/bpo/branching.py",
        "bpo_reward": ROOT / "src/shopping_grpo/training/bpo/reward.py",
        "bpo_runtime": ROOT / "src/shopping_grpo/training/bpo/runtime.py",
        "grpo_dynamic_sampling": (
            ROOT / "src/shopping_grpo/training/grpo/dynamic_sampling.py"
        ),
        "verl_dynamic_sampling_patch": (
            ROOT / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"
        ),
        "model_config": Path(audit["model"]) / "config.json",
    }
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT
    )
    step0_contract = json.loads(environment["SHOPPING_BPO_STEP0_CONTRACT_JSON"])
    step0_contract_sha256 = validate_contract(step0_contract)
    step0_contract_path = (
        Path(environment["BPO_OUTPUT_DIR"]) / "step0_validation_contract.json"
    )
    step0_contract_path.write_text(
        json.dumps(step0_contract, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    contract = {
        "schema_version": "shopping-carl-bpo-run-contract-v1",
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
            "branch_selection": "stage_target_then_exact_entropy",
            "group_schedule": ["root", "local"],
            "local_stage_schedule": (
                ["product"] * 8 + ["option"] * 7 + ["search_recovery"] * 5
            ),
            "branch_candidate_policy": "exclude-final-action-v1",
            "entropy_state": "action-boundary-first-token",
            "rollout_audit": "exact-tree-v1",
            "upstream_lambda": 0.0,
            "ppo_clip": 0.2,
            "gpu_memory_utilization": 0.45,
            "max_num_seqs": 8,
            "minimum_free_gpu_memory_gib": 20.0,
            "sparse_cuda_mapping": "physical-to-logical-v1",
            "use_fused_kernels": True,
            "use_liger": True,
            "use_remove_padding": True,
            "actor_calculate_entropy": False,
            "dynamic_target_prompts": 2,
            "dynamic_minimum_accepted_prompts": 2,
            "dynamic_require_full_batch": True,
            "dynamic_soft_warning_generation_batches": 10,
            "dynamic_max_generation_batches": 30,
            "dataloader_num_workers": 0,
            "tolerant_xml_parameter_parser": True,
            "fused_ppo_input_gradient_backport": "ctx-needs-input-grad-v1",
            "optimizer_update_audit": "first-step-nonzero-gradient-and-delta-v1",
            "scheduler": "cosine",
            "scheduler_horizon": 500,
            "warmup_steps": 10,
            "minimum_lr_ratio": 0.1,
            "effective_tree_budget": 1000,
            "effective_return_budget": 4000,
            "trees_per_optimizer_step": 2,
            "returns_per_optimizer_step": 8,
            "maximum_optimizer_steps": 500,
            "budget_checkpoint_returns": [80, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000],
            "checkpoint_steps": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "validation_steps": [0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        },
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "step0_validation": {
            "contract_path": str(step0_contract_path),
            "contract_sha256": step0_contract_sha256,
            "cache_path": environment["SHOPPING_BPO_STEP0_CACHE_PATH"],
            "refresh": environment["SHOPPING_BPO_STEP0_REFRESH"] == "1",
            "reuse_policy": "exact-contract-sha256-v1",
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
    # CPU preflight deliberately exercises a synthetic BPO tree.  It shares
    # the diagnostics hook with training, so remove only that launcher-owned
    # file before the real process starts to keep audits unambiguous.
    diagnostics_path = Path(environment["SHOPPING_GRPO_DIAGNOSTICS_PATH"])
    diagnostics_path.unlink(missing_ok=True)
    print(f"BPO run contract written: {write_contract(environment, audit)}")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
