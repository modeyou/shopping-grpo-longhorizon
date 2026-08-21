#!/usr/bin/env python3
"""Freeze a Reward-reachable multi-turn evaluation benchmark."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV_ROOT = ROOT / "environments/ShopSimulator/shop_env"
sys.path.insert(0, str(SHOP_ENV_ROOT))

from shopping_grpo.multiturn.benchmark import freeze_curated_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--products",
        type=Path,
        default=Path(
            "environments/ShopSimulator/shop_env/data/"
            "fine_items_eval_train_all.json.gz"
        ),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/multiturn/tasks/evaluation.jsonl"),
    )
    parser.add_argument(
        "--reserve",
        type=Path,
        default=Path("data/multiturn/tasks/reserve.jsonl"),
    )
    parser.add_argument(
        "--split-metadata",
        type=Path,
        default=Path("data/multiturn/tasks/metadata.json"),
    )
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=Path("data/environment.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/multiturn/evaluation-v1"),
    )
    parser.add_argument(
        "--exclude-tasks",
        type=Path,
        action="append",
        default=[],
        help="Task manifest(s) that replacements must not overlap.",
    )
    parser.add_argument(
        "--reward-version",
        choices=("shopsimulator-reward-v3", "shopsimulator-reward-v4"),
        default="shopsimulator-reward-v3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = freeze_curated_evaluation(
        product_data_path=args.products,
        candidates_path=args.candidates,
        reserve_path=args.reserve,
        split_metadata_path=args.split_metadata,
        environment_manifest_path=args.environment_manifest,
        output_dir=args.output_dir,
        exclusion_paths=args.exclude_tasks,
        reward_version=args.reward_version,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
