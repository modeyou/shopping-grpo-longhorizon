#!/usr/bin/env python3
"""Build veRL parquet prompts from frozen multi-turn gap/complete openings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.rollout import MULTITURN_SYSTEM_PROMPT
from shopping_grpo.multiturn.tasks import validate_task_row
from shopping_grpo.training.grpo.adapter.runtime import MULTITURN_HARNESS_VERSION


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task_ids(rows, label):
    values = [int(row["task_id"]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate task_id values")
    return set(values)


def build_record(opening, mode, split, index):
    validate_task_row(opening)
    audit = opening.get("opening_audit") or {}
    omitted_facts = list(audit.get("omitted_facts") or []) if mode == "gap" else []
    if mode == "gap" and not omitted_facts:
        raise ValueError(f"gap task {opening['task_id']} has no omitted facts")
    return {
        "data_source": "shopsimulator-multiturn",
        "prompt": [
            {"role": "system", "content": MULTITURN_SYSTEM_PROMPT},
            {"role": "user", "content": opening["initial_request"]},
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "split": split,
            "index": int(index),
            "task_id": int(opening["task_id"]),
            "harness_version": MULTITURN_HARNESS_VERSION,
            "interaction_mode": mode,
            "initial_request": opening["initial_request"],
            "source_goal_hash": opening["source_goal_hash"],
            "opening_audit": {"omitted_facts": omitted_facts},
            "interaction_kwargs": {
                "name": "shopsimulator-multiturn",
                "task_id": int(opening["task_id"]),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--gap-openings", type=Path, required=True)
    parser.add_argument("--complete-openings", type=Path)
    parser.add_argument("--exclude-tasks", type=Path, action="append", default=[])
    parser.add_argument("--mode", choices=("gap", "complete", "mixed"), default="mixed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to build veRL parquet data") from exc

    selected_ids = task_ids(read_jsonl(args.tasks), "task manifest")
    excluded_ids = set()
    for path in args.exclude_tasks:
        excluded_ids.update(task_ids(read_jsonl(path), f"excluded manifest {path}"))
    overlap = sorted(selected_ids & excluded_ids)
    if overlap:
        raise SystemExit(f"GRPO tasks overlap an excluded manifest: {overlap[:10]}")

    gap = read_jsonl(args.gap_openings)
    complete = read_jsonl(args.complete_openings) if args.complete_openings else []
    if task_ids(gap, "gap openings") != selected_ids:
        raise SystemExit("gap opening task IDs do not exactly match --tasks")
    if complete and task_ids(complete, "complete openings") != selected_ids:
        raise SystemExit("complete opening task IDs do not exactly match --tasks")
    if complete:
        complete_by_id = {int(row["task_id"]): row for row in complete}
        for row in gap:
            peer = complete_by_id[int(row["task_id"])]
            if row.get("source_goal_hash") != peer.get("source_goal_hash"):
                raise SystemExit(f"opening source hash mismatch for task {row['task_id']}")
    if args.mode in {"complete", "mixed"} and not complete:
        raise SystemExit("--complete-openings is required for complete or mixed mode")
    selected = []
    if args.mode in {"gap", "mixed"}:
        selected.extend((row, "gap") for row in gap)
    if args.mode in {"complete", "mixed"}:
        selected.extend((row, "complete") for row in complete)
    rows = [build_record(row, mode, args.split, index) for index, (row, mode) in enumerate(selected)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"wrote {args.output} rows={len(rows)} mode={args.mode}")


if __name__ == "__main__":
    main()
