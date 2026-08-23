#!/usr/bin/env python3
"""Publish accepted Reward v4 GRPO parquet data from frozen active tasks/openings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

try:
    from scripts.prepare_multiturn_grpo_dataset import (
        build_record,
        read_jsonl,
        task_ids,
    )
except ModuleNotFoundError:
    from prepare_multiturn_grpo_dataset import build_record, read_jsonl, task_ids
from shopping_grpo.evaluation.artifacts import (
    load_unique_task_ids,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.training.grpo.data_manifest import (
    DATASET_SCHEMA,
    REWARD_VERSION,
    SELECTION_SCHEMA,
    repo_relative,
    resolve_repo_path,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def artifact(path: Path, *, rows: int | None = None, tasks: int | None = None) -> dict:
    detail = {"path": repo_relative(path, ROOT), "sha256": sha256_file(path)}
    if rows is not None:
        detail["rows"] = int(rows)
    if tasks is not None:
        detail["tasks"] = int(tasks)
    return detail


def validate_selection(path: Path) -> tuple[dict, Path, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SELECTION_SCHEMA:
        raise SystemExit(f"selection manifest must use {SELECTION_SCHEMA}")
    if manifest.get("status") != "frozen":
        raise SystemExit("selection manifest status must be frozen")
    reservoir = manifest.get("source_reservoir") or {}
    reservoir_path = resolve_repo_path(str(reservoir.get("path", "")), ROOT)
    if sha256_file(reservoir_path) != reservoir.get("sha256"):
        raise SystemExit("selection reservoir hash mismatch")
    splits = manifest.get("splits") or {}
    train_path = resolve_repo_path(str((splits.get("train") or {}).get("path", "")), ROOT)
    validation_path = resolve_repo_path(
        str((splits.get("validation") or {}).get("path", "")), ROOT
    )
    for label, task_path in (("train", train_path), ("validation", validation_path)):
        detail = splits[label]
        if sha256_file(task_path) != detail.get("sha256"):
            raise SystemExit(f"selection {label} task hash mismatch")
        ids = load_unique_task_ids(task_path)
        if len(ids) != int(detail.get("tasks", -1)):
            raise SystemExit(f"selection {label} task count mismatch")
    for detail in manifest.get("exclusions") or []:
        exclusion_path = resolve_repo_path(str(detail.get("path", "")), ROOT)
        if sha256_file(exclusion_path) != detail.get("sha256"):
            raise SystemExit(f"selection exclusion hash mismatch: {detail.get('label')}")
    if (manifest.get("audit") or {}).get("selected_exclusion_overlap_count") != 0:
        raise SystemExit("selection manifest contains excluded active tasks")
    return manifest, train_path, validation_path


def validate_openings(task_path: Path, gap_path: Path, complete_path: Path) -> tuple[list, list]:
    selected = load_unique_task_ids(task_path)
    gap = read_jsonl(gap_path)
    complete = read_jsonl(complete_path)
    if task_ids(gap, "gap openings") != selected:
        raise SystemExit(f"gap openings do not exactly cover {task_path}")
    if task_ids(complete, "complete openings") != selected:
        raise SystemExit(f"complete openings do not exactly cover {task_path}")
    complete_by_id = {int(row["task_id"]): row for row in complete}
    for row in gap:
        peer = complete_by_id[int(row["task_id"])]
        if row.get("source_goal_hash") != peer.get("source_goal_hash"):
            raise SystemExit(f"opening source hash mismatch for task {row['task_id']}")
    return gap, complete


def write_parquet_atomic(path: Path, rows: list[dict]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to finalize GRPO data") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--train-gap-openings", type=Path, required=True)
    parser.add_argument("--train-complete-openings", type=Path, required=True)
    parser.add_argument("--validation-gap-openings", type=Path, required=True)
    parser.add_argument("--validation-complete-openings", type=Path, required=True)
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=ROOT / "data/environment-v4.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/grpo/formal-v2",
    )
    args = parser.parse_args()

    selection_path = args.selection_manifest.resolve()
    selection, train_task_source, validation_task_source = validate_selection(
        selection_path
    )
    environment_path = args.environment_manifest.resolve()
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if (environment.get("reward") or {}).get("version") != REWARD_VERSION:
        raise SystemExit(f"environment manifest must use {REWARD_VERSION}")

    train_gap, train_complete = validate_openings(
        train_task_source,
        args.train_gap_openings.resolve(),
        args.train_complete_openings.resolve(),
    )
    validation_gap, validation_complete = validate_openings(
        validation_task_source,
        args.validation_gap_openings.resolve(),
        args.validation_complete_openings.resolve(),
    )
    output = args.output_dir.resolve()
    targets = {
        "train_tasks": output / "multiturn-train-tasks.jsonl",
        "validation_tasks": output / "multiturn-validation-tasks.jsonl",
        "train_gap": output / "multiturn-train-gap-openings.jsonl",
        "train_complete": output / "multiturn-train-complete-openings.jsonl",
        "validation_gap": output / "multiturn-validation-gap-openings.jsonl",
        "validation_complete": output / "multiturn-validation-complete-openings.jsonl",
        "train": output / "multiturn-train.parquet",
        "validation": output / "multiturn-validation.parquet",
        "manifest": output / "manifest.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise SystemExit("refusing existing formal GRPO artifacts: " + ", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)

    train_task_rows = [{"task_id": value} for value in sorted(load_unique_task_ids(train_task_source))]
    validation_task_rows = [
        {"task_id": value}
        for value in sorted(load_unique_task_ids(validation_task_source))
    ]
    write_jsonl_atomic(targets["train_tasks"], train_task_rows)
    write_jsonl_atomic(targets["validation_tasks"], validation_task_rows)
    write_jsonl_atomic(targets["train_gap"], train_gap)
    write_jsonl_atomic(targets["train_complete"], train_complete)
    write_jsonl_atomic(targets["validation_gap"], validation_gap)
    write_jsonl_atomic(targets["validation_complete"], validation_complete)

    train_records = [
        build_record(row, mode, "train", index)
        for index, (row, mode) in enumerate(
            [(row, "gap") for row in train_gap]
            + [(row, "complete") for row in train_complete]
        )
    ]
    validation_records = [
        build_record(row, mode, "validation", index)
        for index, (row, mode) in enumerate(
            [(row, "gap") for row in validation_gap]
            + [(row, "complete") for row in validation_complete]
        )
    ]
    write_parquet_atomic(targets["train"], train_records)
    write_parquet_atomic(targets["validation"], validation_records)

    train_ids = load_unique_task_ids(targets["train_tasks"])
    validation_ids = load_unique_task_ids(targets["validation_tasks"])
    excluded_ids: set[int] = set()
    for detail in selection.get("exclusions") or []:
        excluded_ids.update(
            load_unique_task_ids(resolve_repo_path(detail["path"], ROOT))
        )
    opening_models = sorted(
        {
            str(row.get("opening_model"))
            for row in train_gap + validation_gap
            if row.get("opening_model")
        }
    )
    opening_prompt_hashes = sorted(
        {
            str(row.get("opening_prompt_hash"))
            for row in train_gap + validation_gap
            if row.get("opening_prompt_hash")
        }
    )
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "status": "accepted",
        "reward_version": REWARD_VERSION,
        "environment": artifact(environment_path),
        "selection_source": artifact(selection_path),
        "source_reservoir": selection["source_reservoir"],
        "selection_method": selection["selection"],
        "selection": {
            "train": artifact(targets["train_tasks"], tasks=len(train_ids)),
            "validation": artifact(
                targets["validation_tasks"], tasks=len(validation_ids)
            ),
            "unused_tasks": int(selection["splits"]["unused"]["tasks"]),
        },
        "opening_generation": {
            "models": opening_models,
            "prompt_hashes": opening_prompt_hashes,
            "temperature": 0.0,
            "thinking": False,
        },
        "openings": {
            "train_gap": artifact(targets["train_gap"], rows=len(train_gap), tasks=len(train_ids)),
            "train_complete": artifact(targets["train_complete"], rows=len(train_complete), tasks=len(train_ids)),
            "validation_gap": artifact(targets["validation_gap"], rows=len(validation_gap), tasks=len(validation_ids)),
            "validation_complete": artifact(targets["validation_complete"], rows=len(validation_complete), tasks=len(validation_ids)),
        },
        "artifacts": {
            "train": artifact(targets["train"], rows=len(train_records), tasks=len(train_ids)),
            "validation": artifact(
                targets["validation"], rows=len(validation_records), tasks=len(validation_ids)
            ),
        },
        "exclusions": selection.get("exclusions") or [],
        "audit": {
            "train_validation_overlap_count": len(train_ids & validation_ids),
            "selected_exclusion_overlap_count": len((train_ids | validation_ids) & excluded_ids),
            "train_mode_counts": {"gap": len(train_gap), "complete": len(train_complete)},
            "validation_mode_counts": {
                "gap": len(validation_gap),
                "complete": len(validation_complete),
            },
        },
    }
    write_json_atomic(targets["manifest"], manifest)
    print(
        json.dumps(
            {
                "manifest": str(targets["manifest"]),
                "train_tasks": len(train_ids),
                "train_rows": len(train_records),
                "validation_tasks": len(validation_ids),
                "validation_rows": len(validation_records),
                "reward_version": REWARD_VERSION,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
