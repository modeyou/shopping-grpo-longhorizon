#!/usr/bin/env python3
"""Build the frozen G+/G-/C+ Base/SFT/GRPO comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_json_atomic
from shopping_grpo.evaluation.comparison import (
    MULTITURN_CONDITIONS,
    compare_multiturn_evaluation_grid,
)


def _task_ids(path: Path) -> list[int]:
    values = [int(row["task_id"]) for row in iter_jsonl(path)]
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate task IDs")
    return values


def _run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=ROOT")
    label, root = value.split("=", 1)
    if not label.strip() or not root.strip():
        raise argparse.ArgumentTypeError("--run must be LABEL=ROOT")
    return label.strip(), Path(root)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tasks", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=_run_spec,
        required=True,
        help="LABEL=ROOT; ROOT must contain CONDITION/evaluations.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    actors = {}
    for label, root in args.run:
        if label in actors:
            raise SystemExit(f"duplicate actor label: {label}")
        actors[label] = {
            condition: list(
                iter_jsonl(root / condition / "evaluations.jsonl")
            )
            for condition in MULTITURN_CONDITIONS
        }
    result = compare_multiturn_evaluation_grid(
        expected_task_ids=_task_ids(args.expected_tasks),
        actors=actors,
    )
    write_json_atomic(args.output, result, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
