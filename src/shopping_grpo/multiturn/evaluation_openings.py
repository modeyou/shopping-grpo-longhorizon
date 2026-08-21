"""Freeze paired gap/complete openings for multi-turn evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    index_jsonl,
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.multiturn.tasks import (
    build_task_row,
    source_goal_hash,
    validate_task_row,
)


EVALUATION_OPENING_SCHEMA = "shopping-multiturn-evaluation-opening-v1"
CONDITION_SCHEMA = "shopping-multiturn-evaluation-condition-v1"
COMPLETE_OPENING_POLICY = "private-instruction-full-v1"
COMPLETE_OPENING_POLICY_HASH = hashlib.sha256(
    COMPLETE_OPENING_POLICY.encode("utf-8")
).hexdigest()
CONDITIONS = (
    ("gap-ask-enabled", "gap", True),
    ("gap-ask-disabled", "gap", False),
    ("complete-ask-enabled", "complete", True),
)


def _task_ids(path: str | Path) -> list[int]:
    result = [int(row["task_id"]) for row in iter_jsonl(path)]
    if len(result) != len(set(result)):
        raise ValueError("task manifest contains duplicate task IDs")
    return result


def freeze_evaluation_openings(
    *,
    task_manifest: str | Path,
    gap_openings: str | Path,
    output_dir: str | Path,
    env_factory,
    base_url: str = "http://127.0.0.1:5700",
) -> dict:
    """Validate frozen gap openings and publish complete/condition artifacts."""

    task_manifest = Path(task_manifest)
    gap_openings = Path(gap_openings)
    output_dir = Path(output_dir)
    task_ids = _task_ids(task_manifest)
    expected = set(task_ids)
    gaps = index_jsonl(gap_openings, key="task_id", allowed_keys=expected)
    missing = sorted(expected - set(gaps))
    if missing:
        raise ValueError(f"gap openings are incomplete: {missing[:10]}")

    complete_rows = []
    condition_rows = []
    for task_id in task_ids:
        gap = validate_task_row(gaps[task_id])
        audit = gap.get("opening_audit") or {}
        if not audit.get("omitted_dimensions") or not audit.get("omitted_facts"):
            raise ValueError(f"task {task_id} gap opening lacks an audit")
        with env_factory(base_url=base_url, multiturn=True) as env:
            env.reset(task_id)
            context = env.shopper_context
        actual_hash = source_goal_hash(context)
        if gap.get("source_goal_hash") != actual_hash:
            raise ValueError(f"task {task_id} source goal changed")
        complete = build_task_row(
            task_id,
            context["instruction_full"],
            context,
            COMPLETE_OPENING_POLICY,
            COMPLETE_OPENING_POLICY_HASH,
        )
        complete["evaluation_opening_schema"] = EVALUATION_OPENING_SCHEMA
        complete_rows.append(complete)
        for condition, opening_kind, ask_enabled in CONDITIONS:
            condition_rows.append({
                "schema_version": CONDITION_SCHEMA,
                "task_id": task_id,
                "condition": condition,
                "opening_kind": opening_kind,
                "ask_shopper_enabled": ask_enabled,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "complete_openings.jsonl"
    conditions_path = output_dir / "conditions.jsonl"
    metadata_path = output_dir / "opening_metadata.json"
    write_jsonl_atomic(complete_path, complete_rows)
    write_jsonl_atomic(conditions_path, condition_rows)
    metadata = {
        "schema_version": EVALUATION_OPENING_SCHEMA,
        "task_count": len(task_ids),
        "condition_count": len(condition_rows),
        "complete_opening_policy": COMPLETE_OPENING_POLICY,
        "inputs": {
            "task_manifest": str(task_manifest),
            "task_manifest_sha256": sha256_file(task_manifest),
            "gap_openings": str(gap_openings),
            "gap_openings_sha256": sha256_file(gap_openings),
        },
        "outputs": {
            "complete_openings": str(complete_path),
            "complete_openings_sha256": sha256_file(complete_path),
            "conditions": str(conditions_path),
            "conditions_sha256": sha256_file(conditions_path),
        },
        "validation": {
            "all_tasks_have_gap_openings": True,
            "all_source_goal_hashes_match": True,
            "gap_conditions_share_one_opening": True,
            "conditions_per_task": len(CONDITIONS),
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata
