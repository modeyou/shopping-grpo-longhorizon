#!/usr/bin/env python3
'''Launch the single frozen 500-update native Reward-v4 GRPO run.'''

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'data/grpo/formal-v2/multiturn-train.parquet'
VALIDATION = ROOT / 'data/grpo/formal-v2/multiturn-validation.parquet'
MANIFEST = ROOT / 'data/grpo/formal-v2/manifest.json'
PROJECT = 'shopping-multiturn-agentic'
TOTAL_UPDATES = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shopper-model', default='deepseek-v4-flash-0731')
    parser.add_argument(
        '--shopper-base-url',
        default=os.environ.get('SHOPPER_BASE_URL') or os.environ.get('OPENAI_BASE_URL'),
    )
    parser.add_argument('--env-url', default='http://127.0.0.1:5700')
    parser.add_argument('--seed', type=int, default=20260823)
    parser.add_argument('--experiment-name')
    parser.add_argument('--resume-from-checkpoint', type=Path)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--dry-run', action='store_true')
    execution.add_argument('--preflight-only', action='store_true')
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_name = (
        args.experiment_name or f'grpo-native-v4-500-s{args.seed}-{stamp}'
    )
    command = [
        sys.executable,
        str(ROOT / 'scripts/train_grpo.py'),
        '--model', str(args.model),
        '--train-data', str(TRAIN),
        '--val-data', str(VALIDATION),
        '--data-manifest', str(MANIFEST),
        '--output', str(args.output),
        '--experiment-name', experiment_name,
        '--require-clean-git',
        '--logger', 'swanlab',
        '--reward-profile', 'none',
        '--shopper-model', args.shopper_model,
        '--env-url', args.env_url,
        '--seed', str(args.seed),
    ]
    if args.shopper_base_url:
        command.extend(('--shopper-base-url', args.shopper_base_url))
    if args.resume_from_checkpoint is not None:
        command.extend(('--resume-from-checkpoint', str(args.resume_from_checkpoint)))
    if args.dry_run:
        command.append('--dry-run')
    elif args.preflight_only:
        command.append('--preflight-only')
    command.extend(
        (
            '--',
            'trainer.n_gpus_per_node=4',
            f'trainer.total_training_steps={TOTAL_UPDATES}',
            'trainer.save_freq=25',
            'trainer.test_freq=50',
            'trainer.val_before_train=true',
            f'trainer.project_name={PROJECT}',
            'actor_rollout_ref.model.use_remove_padding=true',
            'actor_rollout_ref.model.use_fused_kernels=true',
            'actor_rollout_ref.model.use_liger=true',
            'actor_rollout_ref.model.fused_kernel_options.impl_backend=torch',
            'actor_rollout_ref.model.override_config.attn_implementation=sdpa',
            'actor_rollout_ref.actor.optim.lr=1e-6',
            'actor_rollout_ref.actor.optim.lr_warmup_steps=-1',
            'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03',
            'actor_rollout_ref.actor.optim.lr_scheduler_type=constant',
            'data.dataloader_num_workers=0',
        )
    )
    return command


def main() -> None:
    args = parse_args()
    raise SystemExit(
        subprocess.call(build_command(args), cwd=ROOT, env=os.environ.copy())
    )


if __name__ == '__main__':
    main()
