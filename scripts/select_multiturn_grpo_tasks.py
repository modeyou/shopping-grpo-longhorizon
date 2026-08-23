#!/usr/bin/env python3
"""Select the frozen active GRPO train/validation tasks from one reservoir."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


SHOP_ENV_ROOT = (
    Path(__file__).resolve().parents[1] / "environments/ShopSimulator/shop_env"
)
sys.path.insert(0, str(SHOP_ENV_ROOT))

from shopping_grpo.evaluation.artifacts import (
    load_unique_task_ids,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.training.grpo.data_manifest import (
    REWARD_VERSION,
    SELECTION_SCHEMA,
    repo_relative,
    sha256_file,
)
from shopping_grpo.multiturn.benchmark import (
    audit_gold_task_version,
    load_products,
)
from shopping_grpo.multiturn.splits import decompressed_sha256


ROOT = Path(__file__).resolve().parents[1]


def parse_exclusion(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--exclude must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("--exclude must use non-empty LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reservoir",
        type=Path,
        default=ROOT / "data/multiturn/tasks/grpo_train.jsonl",
    )
    parser.add_argument("--exclude", action="append", type=parse_exclusion, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--expected-reservoir-sha256")
    parser.add_argument(
        "--products",
        type=Path,
        default=ROOT / (
            "environments/ShopSimulator/shop_env/data/"
            "fine_items_eval_train_all.json.gz"
        ),
    )
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=ROOT / "data/environment-v4.json",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reservoir = args.reservoir.resolve()
    reservoir_hash = sha256_file(reservoir)
    if (
        args.expected_reservoir_sha256
        and reservoir_hash != args.expected_reservoir_sha256
    ):
        raise SystemExit("GRPO reservoir hash mismatch")
    reservoir_ids = load_unique_task_ids(reservoir)

    exclusions = []
    excluded_ids: set[int] = set()
    labels = set()
    for label, raw_path in args.exclude:
        if label in labels:
            raise SystemExit(f"duplicate exclusion label: {label}")
        labels.add(label)
        path = raw_path.resolve()
        ids = load_unique_task_ids(path)
        overlap = sorted(reservoir_ids & ids)
        excluded_ids.update(ids)
        exclusions.append(
            {
                "label": label,
                "path": repo_relative(path, ROOT),
                "sha256": sha256_file(path),
                "tasks": len(ids),
                "reservoir_overlap_count": len(overlap),
                "reservoir_overlap_task_ids": overlap,
            }
        )

    environment_path = args.environment_manifest.resolve()
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if (environment.get("reward") or {}).get("version") != REWARD_VERSION:
        raise SystemExit(f"environment manifest must use {REWARD_VERSION}")
    products_path = args.products.resolve()
    product_hash = decompressed_sha256(products_path)
    if product_hash != environment.get("product_data_sha256"):
        raise SystemExit(
            "decompressed product data hash does not match environment manifest"
        )
    products = load_products(products_path)

    def rank(task_id: int) -> tuple[bytes, int]:
        return (
            hashlib.sha256(f"{int(args.seed)}:{task_id}".encode("utf-8")).digest(),
            task_id,
        )

    candidates = sorted(reservoir_ids - excluded_ids, key=rank)
    outside = [task_id for task_id in candidates if not 0 <= task_id < len(products)]
    if outside:
        raise SystemExit(f"task IDs outside product data: {outside[:10]}")
    reward_audits = [
        audit_gold_task_version(
            products[task_id], task_id, reward_version=REWARD_VERSION
        )
        for task_id in candidates
    ]
    reachable = [row["task_id"] for row in reward_audits if row["eligible"]]
    required = args.validation_count + args.train_count
    if len(reachable) < required:
        raise SystemExit(
            f"only {len(reachable)} Reward v4 reachable tasks for {required} requested"
        )
    selected = {
        "validation": reachable[: args.validation_count],
        "train": reachable[
            args.validation_count : args.validation_count + args.train_count
        ],
        "unused": reachable[required:],
    }
    train_path = output / "train-tasks.jsonl"
    validation_path = output / "validation-tasks.jsonl"
    write_jsonl_atomic(train_path, ({"task_id": value} for value in selected["train"]))
    write_jsonl_atomic(
        validation_path,
        ({"task_id": value} for value in selected["validation"]),
    )
    reward_audit_path = output / "reward-audit.jsonl"
    write_jsonl_atomic(reward_audit_path, reward_audits)
    selected_ids = set(selected["train"]) | set(selected["validation"])
    rejection_reasons = Counter(
        reason
        for row in reward_audits
        if not row["eligible"]
        for reason in row.get("reasons") or []
    )
    manifest = {
        "schema_version": SELECTION_SCHEMA,
        "status": "frozen",
        "source_reservoir": {
            "path": repo_relative(reservoir, ROOT),
            "sha256": reservoir_hash,
            "tasks": len(reservoir_ids),
        },
        "selection": {
            "seed": args.seed,
            "method": (
                "sha256(seed:task_id) ascending, then retain Reward v4 reachable tasks"
            ),
            "split_order": ["validation", "train", "unused"],
        },
        "reward_contract": {
            "version": REWARD_VERSION,
            "environment_manifest": {
                "path": repo_relative(environment_path, ROOT),
                "sha256": sha256_file(environment_path),
            },
            "product_data": {
                "path": repo_relative(products_path, ROOT),
                "compressed_sha256": sha256_file(products_path),
                "decompressed_sha256": product_hash,
            },
            "audit": {
                "path": repo_relative(reward_audit_path, ROOT),
                "sha256": sha256_file(reward_audit_path),
                "rows": len(reward_audits),
                "reachable_tasks": len(reachable),
                "rejected_tasks": len(reward_audits) - len(reachable),
                "rejection_reasons": dict(sorted(rejection_reasons.items())),
            },
        },
        "splits": {
            "validation": {
                "path": repo_relative(validation_path, ROOT),
                "sha256": sha256_file(validation_path),
                "tasks": len(selected["validation"]),
                "task_ids": selected["validation"],
            },
            "train": {
                "path": repo_relative(train_path, ROOT),
                "sha256": sha256_file(train_path),
                "tasks": len(selected["train"]),
                "task_ids": selected["train"],
            },
            "unused": {
                "tasks": len(selected["unused"]),
                "task_ids": selected["unused"],
            },
        },
        "exclusions": exclusions,
        "audit": {
            "post_exclusion_candidate_tasks": len(candidates),
            "reward_reachable_tasks": len(reachable),
            "reward_rejected_tasks": len(reward_audits) - len(reachable),
            "all_selected_tasks_reward_reachable": all(
                row["eligible"]
                for row in reward_audits
                if row["task_id"] in selected_ids
            ),
            "train_validation_overlap_count": 0,
            "selected_exclusion_overlap_count": len(selected_ids & excluded_ids),
        },
    }
    manifest_path = output / "selection-manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "reservoir_tasks": len(reservoir_ids),
                "train_tasks": len(selected["train"]),
                "validation_tasks": len(selected["validation"]),
                "unused_tasks": len(selected["unused"]),
                "reward_rejected_tasks": len(reward_audits) - len(reachable),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
