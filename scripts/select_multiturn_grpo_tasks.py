#!/usr/bin/env python3
"""Select the frozen active GRPO train/validation tasks from one reservoir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    load_unique_task_ids,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.training.grpo.data_manifest import (
    SELECTION_SCHEMA,
    deterministic_active_split,
    repo_relative,
    sha256_file,
)


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

    selected = deterministic_active_split(
        reservoir_ids,
        excluded_ids,
        seed=args.seed,
        train_count=args.train_count,
        validation_count=args.validation_count,
    )
    train_path = output / "train-tasks.jsonl"
    validation_path = output / "validation-tasks.jsonl"
    write_jsonl_atomic(train_path, ({"task_id": value} for value in selected["train"]))
    write_jsonl_atomic(
        validation_path,
        ({"task_id": value} for value in selected["validation"]),
    )
    selected_ids = set(selected["train"]) | set(selected["validation"])
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
            "method": "sha256(seed:task_id) ascending",
            "split_order": ["validation", "train", "unused"],
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
            "eligible_reservoir_tasks": len(reservoir_ids - excluded_ids),
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
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
