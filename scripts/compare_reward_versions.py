#!/usr/bin/env python3
"""Gold-replay the same frozen tasks under Reward v3 and v4."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
sys.path.append(str(SHOP_ENV))

from shopping_grpo.evaluation.artifacts import (
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.multiturn.benchmark import (
    audit_gold_task_version,
    load_products,
    read_task_ids,
)


COMPARISON_VERSION = "shopping-reward-v3-v4-comparison-v1"
V3 = "shopsimulator-reward-v3"
V4 = "shopsimulator-reward-v4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--products",
        type=Path,
        default=SHOP_ENV / "data/fine_items_eval_train_all.json.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _status(value: bool) -> str:
    return "eligible" if value else "rejected"


def main():
    args = parse_args()
    task_ids = read_task_ids(args.tasks)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        task_ids = task_ids[: args.limit]
    products = load_products(args.products)
    outside = [
        task_id
        for task_id in task_ids
        if task_id < 0 or task_id >= len(products)
    ]
    if outside:
        raise SystemExit(f"task IDs outside product data: {outside[:10]}")

    rows = []
    eligibility_transitions = Counter()
    reward_transitions = Counter()
    v3_reasons = Counter()
    v4_reasons = Counter()
    gained = []
    lost = []
    price_recovered = []
    for index, task_id in enumerate(task_ids, start=1):
        product = products[task_id]
        v3 = audit_gold_task_version(
            product, task_id, reward_version=V3
        )
        v4 = audit_gold_task_version(
            product, task_id, reward_version=V4
        )
        transition = f"{_status(v3['eligible'])}_to_{_status(v4['eligible'])}"
        reward_transition = (
            f"{v3['audit'].get('reward_type')} -> "
            f"{v4['audit'].get('reward_type')}"
        )
        eligibility_transitions[transition] += 1
        reward_transitions[reward_transition] += 1
        v3_reasons.update(v3["reasons"])
        v4_reasons.update(v4["reasons"])
        if not v3["eligible"] and v4["eligible"]:
            gained.append(task_id)
        if v3["eligible"] and not v4["eligible"]:
            lost.append(task_id)
        if (
            "explicit_price_not_compiled" in v3["reasons"]
            and "explicit_price_not_compiled" not in v4["reasons"]
        ):
            price_recovered.append(task_id)
        rows.append(
            {
                "schema_version": COMPARISON_VERSION,
                "task_id": task_id,
                "eligibility_transition": transition,
                "reward_type_transition": reward_transition,
                "v3": v3,
                "v4": v4,
            }
        )
        print(f"compare {index}/{len(task_ids)} task={task_id}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "comparison.jsonl"
    summary_path = args.output_dir / "summary.json"
    manifest_path = args.output_dir / "manifest.json"
    summary = {
        "schema_version": COMPARISON_VERSION,
        "task_count": len(task_ids),
        "eligibility_transitions": dict(
            sorted(eligibility_transitions.items())
        ),
        "reward_type_transitions": dict(sorted(reward_transitions.items())),
        "v3_reject_reasons": dict(sorted(v3_reasons.items())),
        "v4_reject_reasons": dict(sorted(v4_reasons.items())),
        "v4_gained_task_ids": gained,
        "v4_lost_task_ids": lost,
        "v4_recovered_price_task_ids": price_recovered,
        "net_eligible_change": len(gained) - len(lost),
    }
    write_jsonl_atomic(rows_path, rows, force=args.force)
    write_json_atomic(summary_path, summary, force=args.force)
    manifest = {
        "schema_version": COMPARISON_VERSION,
        "tasks": str(args.tasks),
        "tasks_sha256": sha256_file(args.tasks),
        "products": str(args.products),
        "products_sha256": sha256_file(args.products),
        "reward_versions": [V3, V4],
        "task_count": len(task_ids),
        "artifacts": {
            rows_path.name: sha256_file(rows_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    write_json_atomic(manifest_path, manifest, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
