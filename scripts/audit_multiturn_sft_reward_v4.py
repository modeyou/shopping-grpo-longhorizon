#!/usr/bin/env python3
"""Build immutable Reward v4 candidate pools from stored v3 Teacher raw data."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from shopping_grpo.collection.multiturn_sft_v4 import (
    POOL_SCHEMA_VERSION,
    audit_source,
    content_hash,
    cross_policy_overlaps,
    read_jsonl,
    sha256_file,
    task_ids,
)
from shopping_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic


SOURCES = {
    "complete-no-ask-v1": "complete",
    "composite-replay-v1": "composite",
    "autonomous-gap-v1": "autonomous",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-raw", type=Path, required=True)
    parser.add_argument("--composite-raw", type=Path, required=True)
    parser.add_argument("--autonomous-raw", type=Path, required=True)
    parser.add_argument(
        "--products",
        type=Path,
        default=SHOP_ENV / "data/fine_items_eval_train_all.json.gz",
    )
    parser.add_argument(
        "--allowed-tasks",
        type=Path,
        default=ROOT / "data/multiturn/tasks/sft_candidates.jsonl",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=Path,
        action="append",
        default=[
            ROOT / "data/multiturn/evaluation-dev-v2/tasks.jsonl",
            ROOT / "data/multiturn/evaluation-v2/tasks.jsonl",
            ROOT / "data/multiturn/tasks/grpo_validation.jsonl",
            ROOT / "data/multiturn/tasks/grpo_train.jsonl",
            ROOT / "data/evaluation/tasks.jsonl",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_products(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    if not isinstance(products, list):
        raise ValueError("product data must be a JSON list")
    return products


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")

    inputs = {
        "complete-no-ask-v1": args.complete_raw.resolve(),
        "composite-replay-v1": args.composite_raw.resolve(),
        "autonomous-gap-v1": args.autonomous_raw.resolve(),
    }
    for path in [*inputs.values(), args.products, args.allowed_tasks, *args.exclude_tasks]:
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")

    allowed = task_ids(args.allowed_tasks)
    excluded_by_source = {
        str(path): task_ids(path) for path in args.exclude_tasks
    }
    excluded = set().union(*excluded_by_source.values())
    overlap = sorted(allowed & excluded)
    if overlap:
        raise SystemExit(
            "sft_candidates overlap excluded tasks: "
            f"{overlap[:20]}"
        )

    products = load_products(args.products)
    output.mkdir(parents=True, exist_ok=True)
    pools = {}
    source_summaries = {}
    artifact_paths = {}
    for policy, raw_path in inputs.items():
        rows, audits, summary = audit_source(
            raw_path=raw_path,
            expected_policy=policy,
            products=products,
            allowed_task_ids=allowed,
            excluded_task_ids=excluded,
        )
        short_name = SOURCES[policy]
        pool_path = output / f"{short_name}.jsonl"
        audit_path = output / f"{short_name}-audit.jsonl"
        write_jsonl_atomic(pool_path, rows)
        write_jsonl_atomic(audit_path, audits)
        pools[policy] = rows
        source_summaries[policy] = summary
        artifact_paths[f"{short_name}_pool"] = pool_path
        artifact_paths[f"{short_name}_audit"] = audit_path

    overlaps = cross_policy_overlaps(pools)
    manifest = {
        "schema_version": POOL_SCHEMA_VERSION,
        "environment": "shopsimulator-environment-v2.1",
        "source_reward": "shopsimulator-reward-v3",
        "audit_reward": "shopsimulator-reward-v4",
        "selection": "strict-v3-and-v4-gold-intersection",
        "inputs": {
            policy: {
                "path": str(path),
                "rows": len(read_jsonl(path)),
                "sha256": sha256_file(path),
            }
            for policy, path in inputs.items()
        },
        "products": {
            "path": str(args.products.resolve()),
            "sha256": sha256_file(args.products),
            "rows": len(products),
        },
        "allowed_tasks": {
            "path": str(args.allowed_tasks.resolve()),
            "sha256": sha256_file(args.allowed_tasks),
            "tasks": len(allowed),
        },
        "excluded_tasks": {
            str(path): {
                "sha256": sha256_file(path),
                "tasks": len(excluded_by_source[str(path)]),
            }
            for path in args.exclude_tasks
        },
        "source_summaries": source_summaries,
        "cross_policy_overlaps": overlaps,
        "artifacts": {
            name: {
                "path": str(path),
                "rows": len(read_jsonl(path)),
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
        "pool_content_hashes": {
            policy: [content_hash(row) for row in rows]
            for policy, rows in pools.items()
        },
    }
    manifest_path = output / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_summaries": source_summaries,
                "cross_policy_overlap_counts": {
                    key: value["tasks"] for key, value in overlaps.items()
                },
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
