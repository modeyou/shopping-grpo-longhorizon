#!/usr/bin/env python3
"""Launch one arm of the frozen formal GRPO A/B decision gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/grpo/formal-v2/multiturn-train.parquet"
VALIDATION = ROOT / "data/grpo/formal-v2/multiturn-validation.parquet"
MANIFEST = ROOT / "data/grpo/formal-v2/manifest.json"
PROJECT = "shopping-multiturn-agentic"
SCHEDULER_HORIZON = 500
DEFAULT_STAGE_END = 50

ARM_CONTRACTS = {
    "a": {
        "reward_profile": "none",
        "experiment_name": "grpo-a-native-v4",
    },
    "b": {
        "reward_profile": "bounded-v1",
        "experiment_name": "grpo-b-bounded-v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_CONTRACTS), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shopper-model", default="deepseek-v4-flash-0731")
    parser.add_argument(
        "--shopper-base-url",
        default=os.environ.get("SHOPPER_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--stage-end", type=int, default=DEFAULT_STAGE_END)
    parser.add_argument("--experiment-name")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--logger", choices=("console", "swanlab"), default="swanlab")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--dry-run", action="store_true")
    execution.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    if args.stage_end <= 0 or args.stage_end > SCHEDULER_HORIZON:
        raise SystemExit(
            f"--stage-end must be in [1, {SCHEDULER_HORIZON}], got {args.stage_end}"
        )
    arm = ARM_CONTRACTS[args.arm]
    experiment_name = args.experiment_name or arm["experiment_name"]
    command = [
        sys.executable,
        str(ROOT / "scripts/train_grpo.py"),
        "--model",
        str(args.model),
        "--train-data",
        str(TRAIN),
        "--val-data",
        str(VALIDATION),
        "--data-manifest",
        str(MANIFEST),
        "--output",
        str(args.output),
        "--experiment-name",
        experiment_name,
        "--logger",
        args.logger,
        "--reward-profile",
        arm["reward_profile"],
        "--shopper-model",
        args.shopper_model,
        "--env-url",
        args.env_url,
        "--seed",
        str(args.seed),
    ]
    if args.shopper_base_url:
        command.extend(("--shopper-base-url", args.shopper_base_url))
    if args.resume_from_checkpoint is not None:
        command.extend(("--resume-from-checkpoint", str(args.resume_from_checkpoint)))
    if args.dry_run:
        command.append("--dry-run")
    elif args.preflight_only:
        command.append("--preflight-only")
    command.extend(
        (
            "--",
            "trainer.n_gpus_per_node=4",
            f"trainer.total_training_steps={args.stage_end}",
            f"shopping_scheduler.total_training_steps={SCHEDULER_HORIZON}",
            "trainer.save_freq=25",
            f"trainer.test_freq={args.stage_end}",
            "trainer.val_before_train=false",
            f"trainer.project_name={PROJECT}",
            "actor_rollout_ref.model.use_remove_padding=true",
            "actor_rollout_ref.model.use_fused_kernels=true",
            "actor_rollout_ref.model.use_liger=true",
            "actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            "data.dataloader_num_workers=0",
        )
    )
    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=os.environ.copy()))


if __name__ == "__main__":
    main()
