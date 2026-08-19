#!/usr/bin/env python3
"""Export a deterministic ShopSimulator source pool for LLM task generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.personalization.source import (
    DEFAULT_PRODUCT_DATA,
    load_source_tasks,
    read_task_ids,
    select_source_tasks,
    write_jsonl,
)
from shopping_grpo.personalization.schema import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-pool", type=Path, default=Path("data/grpo/train.jsonl"))
    parser.add_argument("--product-data", type=Path, default=DEFAULT_PRODUCT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> int:
    args = parse_args()
    pool = read_task_ids(args.task_pool)
    selected = select_source_tasks(pool, count=args.count, seed=args.seed)
    rows = load_source_tasks(selected, product_data=args.product_data)
    output = args.output_dir / "source_tasks.jsonl"
    write_jsonl(output, rows)
    manifest = {
        "schema_version": "personalized-source-export-v1",
        "task_pool": _portable_path(args.task_pool),
        "product_data": _portable_path(args.product_data),
        "count": len(rows),
        "seed": args.seed,
        "selected_task_ids": selected,
        "selected_hash": stable_hash(selected),
        "source_rows_hash": stable_hash(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"exported {len(rows)} source tasks -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
