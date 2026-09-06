#!/usr/bin/env python3
"""Filter frozen tasks, Rubrics, and trajectories to the Judge review set."""

from __future__ import annotations

import argparse
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.blind_guard import guard_declared_final_tasks
from shopping_grpo.evaluation.manifest import sha256_file


MANIFEST_VERSION = "shopping-judge-calibration-inputs-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        help="LABEL=JSONL; repeat for every actor under calibration",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-blind-final", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = str(value).partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("--trajectory must use LABEL=JSONL")
    return label.strip(), Path(raw_path)


def _case_ids(path: Path) -> list[int]:
    task_ids = [int(row["task_id"]) for row in iter_jsonl(path)]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("calibration cases must contain unique task IDs")
    return task_ids


def _filter(path: Path, task_ids: list[int]) -> list[dict]:
    wanted = set(task_ids)
    indexed = {}
    for row in iter_jsonl(path):
        task_id = int(row["task_id"])
        if task_id in wanted:
            if task_id in indexed:
                raise ValueError(f"{path} repeats task_id={task_id}")
            indexed[task_id] = row
    missing = [task_id for task_id in task_ids if task_id not in indexed]
    if missing:
        raise ValueError(f"{path} lacks calibration tasks: {missing}")
    return [indexed[task_id] for task_id in task_ids]


def main():
    args = parse_args()
    guard_declared_final_tasks(args.tasks, allowed=args.allow_blind_final)
    task_ids = _case_ids(args.cases)
    specs = [_spec(value) for value in args.trajectory]
    if len({label for label, _ in specs}) != len(specs):
        raise SystemExit("trajectory labels must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tasks": args.output_dir / "tasks.jsonl",
        "rubrics": args.output_dir / "rubrics.jsonl",
    }
    write_jsonl_atomic(
        outputs["tasks"], _filter(args.tasks, task_ids), force=args.force
    )
    write_jsonl_atomic(
        outputs["rubrics"], _filter(args.rubrics, task_ids), force=args.force
    )
    trajectory_outputs = {}
    for label, path in specs:
        output = args.output_dir / label / "trajectories.jsonl"
        write_jsonl_atomic(output, _filter(path, task_ids), force=args.force)
        trajectory_outputs[label] = {
            "path": str(output),
            "sha256": sha256_file(output),
        }
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "case_count": len(task_ids),
        "task_ids": task_ids,
        "cases": {"path": str(args.cases), "sha256": sha256_file(args.cases)},
        "tasks": {
            "path": str(outputs["tasks"]),
            "sha256": sha256_file(outputs["tasks"]),
        },
        "rubrics": {
            "path": str(outputs["rubrics"]),
            "sha256": sha256_file(outputs["rubrics"]),
        },
        "trajectories": trajectory_outputs,
    }
    write_json_atomic(
        args.output_dir / "manifest.json", manifest, force=args.force
    )
    print(
        f"prepared {len(task_ids)} calibration tasks for "
        f"{len(specs)} actors under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
