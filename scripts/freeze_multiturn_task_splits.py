#!/usr/bin/env python3
"""Freeze project-owned, task-disjoint multi-turn data pools."""

import argparse
import json
from pathlib import Path

from shopping_grpo.multiturn.splits import freeze_task_splits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUSIONS = (
    ROOT / "data/reference/sft-v1/train.jsonl",
    ROOT / "data/reference/sft-v1/validation.jsonl",
    ROOT / "data/sft_pure_v4/all.jsonl",
    ROOT / "data/reference/grpo-v1/train.jsonl",
    ROOT / "data/reference/grpo-v1/validation.jsonl",
    ROOT / "data/evaluation/tasks.jsonl",
    ROOT / "src/shopping_grpo/resources/blind_final_task_ids.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product-data",
        type=Path,
        default=ROOT / (
            "environments/ShopSimulator/shop_env/data/"
            "fine_items_eval_train_all.json.gz"
        ),
    )
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=ROOT / "data/environment.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/multiturn/tasks",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        type=Path,
        help="JSONL/JSON task source to exclude; repeatable. Defaults to all reference datasets.",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--sft-candidates", type=int, default=3000)
    parser.add_argument("--grpo-train", type=int, default=5000)
    parser.add_argument("--grpo-validation", type=int, default=500)
    parser.add_argument("--evaluation", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = freeze_task_splits(
        product_data_path=args.product_data,
        environment_manifest_path=args.environment_manifest,
        exclusion_paths=args.exclude or DEFAULT_EXCLUSIONS,
        output_dir=args.output_dir,
        split_sizes={
            "evaluation": args.evaluation,
            "sft_candidates": args.sft_candidates,
            "grpo_validation": args.grpo_validation,
            "grpo_train": args.grpo_train,
        },
        seed=args.seed,
    )
    print(json.dumps({
        "schema_version": metadata["schema_version"],
        "goal_count": metadata["environment"]["goal_count"],
        "excluded": metadata["exclusions"]["unique_task_ids"],
        "splits": {
            name: detail["tasks"]
            for name, detail in metadata["splits"].items()
        },
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
