#!/usr/bin/env python3
"""Apply small, human-reviewed semantic corrections to frozen Rubrics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    index_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.contracts import validate_rubric_bundle
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.evaluation.rubric import stable_hash


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def apply_corrections(bundle: dict, correction_set: dict) -> dict:
    result = deepcopy(bundle)
    by_id = {rubric["rubric_id"]: rubric for rubric in result["rubrics"]}
    for correction in correction_set["corrections"]:
        rubric_id = str(correction["rubric_id"])
        if rubric_id not in by_id:
            raise ValueError(f"task {result['task_id']} lacks {rubric_id}")
        semantics = by_id[rubric_id].get("candidate_semantics") or []
        option_semantics = [
            value
            for value in semantics
            if value.get("constraint_type") == "option_component"
        ]
        if len(option_semantics) != 1:
            raise ValueError(
                f"task {result['task_id']} {rubric_id} must have one option semantic"
            )
        expected = option_semantics[0].get("expected_value") or {}
        old_component = str(correction["old_component"])
        if expected.get("component") != old_component:
            raise ValueError(
                f"task {result['task_id']} {rubric_id} old_component mismatch"
            )
        expected["component"] = str(correction["component"])
        by_id[rubric_id]["selection_reason"] = str(correction_set["reason"])
    for rubric in result["rubrics"]:
        rubric["rubric_source"] = "human_review_revision"
        rubric["data_sources"] = sorted(
            set(rubric.get("data_sources") or []) | {"human_review"}
        )
    result["review"] = {"status": "human_approved"}
    result["generation"] = {
        **result["generation"],
        "curator_version": "human-semantic-correction-v1",
        "candidate_hash": stable_hash(result["rubrics"]),
    }
    return validate_rubric_bundle(result, expected_task_id=int(result["task_id"]))


def main():
    args = parse_args()
    rubrics = index_jsonl(args.rubrics, key="task_id")
    corrections = index_jsonl(args.corrections, key="task_id")
    missing = sorted(set(corrections) - set(rubrics))
    if missing:
        raise SystemExit(f"correction tasks missing from Rubrics: {missing}")
    rows = []
    for task_id, bundle in rubrics.items():
        if task_id in corrections:
            bundle = apply_corrections(bundle, corrections[task_id])
        rows.append(bundle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rubrics.jsonl"
    write_jsonl_atomic(output, rows, force=args.force)
    manifest = {
        "schema_version": "shopping-rubric-semantic-corrections-manifest-v1",
        "rubric_count": len(rows),
        "corrected_task_ids": sorted(int(value) for value in corrections),
        "base_rubrics_sha256": sha256_file(args.rubrics),
        "corrections_sha256": sha256_file(args.corrections),
        "rubrics_sha256": sha256_file(output),
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest, force=args.force)
    print(f"materialized {len(rows)} Rubrics; corrected_tasks={len(corrections)}")


if __name__ == "__main__":
    main()
