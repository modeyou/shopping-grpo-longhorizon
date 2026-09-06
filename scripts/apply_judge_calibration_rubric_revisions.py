#!/usr/bin/env python3
"""Materialize human-approved Rubric revisions for the Judge calibration set."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    index_jsonl,
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.contracts import (
    RUBRIC_SCHEMA_VERSION,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.evaluation.rubric import (
    GENERIC_ACCEPTANCE_CRITERIA,
    TASK_FACTS_VERSION,
    build_query_evidence_anchors,
    stable_hash,
)


REVISION_SCHEMA_VERSION = "shopping-human-rubric-revision-v1"
OUTPUT_RUBRIC_VERSION = "shopping-multiturn-rubric-calibration-v1"
MANIFEST_VERSION = "shopping-calibration-rubric-manifest-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--base-rubrics", type=Path, required=True)
    parser.add_argument("--revisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-all-base-tasks",
        action="store_true",
        help="emit every base Rubric instead of only the calibration cases",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_revisions(path: Path) -> dict[int, dict]:
    revisions = index_jsonl(path, key="task_id")
    for task_id, revision in revisions.items():
        if revision.get("schema_version") != REVISION_SCHEMA_VERSION:
            raise ValueError(f"task {task_id} has an unsupported revision schema")
        if revision.get("rubric_version") != OUTPUT_RUBRIC_VERSION:
            raise ValueError(f"task {task_id} has an unexpected rubric version")
        requirements = revision.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            raise ValueError(f"task {task_id} revision has no requirements")
    return {int(task_id): value for task_id, value in revisions.items()}


def _revision_task_ids(cases_path: Path) -> list[int]:
    task_ids = []
    for case in iter_jsonl(cases_path):
        if case.get("expected_rubric_version") == OUTPUT_RUBRIC_VERSION:
            task_ids.append(int(case["task_id"]))
    return task_ids


def materialize_revision(base: Mapping, revision: Mapping) -> dict:
    task_id = int(base["task_id"])
    if int(revision["task_id"]) != task_id:
        raise ValueError("revision task does not match base Rubric")
    query = str(revision.get("query_override") or base["query"]).strip()
    anchors = build_query_evidence_anchors(query)
    anchors_by_id = {item["anchor_id"]: item for item in anchors}
    rubrics = []
    candidate_index = 0
    for rubric_index, requirement in enumerate(revision["requirements"], start=1):
        anchor_ids = [str(value) for value in requirement["query_anchor_ids"]]
        unknown = sorted(set(anchor_ids) - set(anchors_by_id))
        if unknown:
            raise ValueError(f"task {task_id} references unknown anchors: {unknown}")
        semantics = []
        candidate_ids = []
        for raw_semantic in requirement["semantics"]:
            candidate_index += 1
            candidate_id = f"h{candidate_index:04d}"
            candidate_ids.append(candidate_id)
            semantics.append({"candidate_id": candidate_id, **dict(raw_semantic)})
        rubrics.append(
            {
                "rubric_id": f"r{rubric_index:04d}",
                "rubric_source": "human_review_revision",
                "candidate_ids": candidate_ids,
                "candidate_semantics": semantics,
                "description": str(requirement["description"]).strip(),
                "acceptance_criteria": GENERIC_ACCEPTANCE_CRITERIA,
                "hardness": str(requirement["hardness"]),
                "hardness_source": "human_review_confirmation",
                "query_spans": [
                    {
                        "text": anchors_by_id[anchor_id]["text"],
                        "start": anchors_by_id[anchor_id]["start"],
                        "end": anchors_by_id[anchor_id]["end"],
                    }
                    for anchor_id in anchor_ids
                ],
                "query_anchor_ids": anchor_ids,
                "data_sources": ["human_review", "query"],
                "selection_reason": str(requirement["selection_reason"]).strip(),
            }
        )
    task_facts = {
        "schema_version": TASK_FACTS_VERSION,
        "task_id": task_id,
        "query": query,
    }
    bundle = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_version": OUTPUT_RUBRIC_VERSION,
        "task_id": task_id,
        "query": query,
        "generation": {
            "curator_version": "human-rubric-review-v1",
            "curator_model": "human",
            "curator_prompt_version": "manual-review-2026-09-06",
            "task_data_hash": stable_hash(task_facts),
            "query_hash": stable_hash(query),
            "candidate_hash": stable_hash(revision["requirements"]),
        },
        "review": {"status": "human_approved"},
        "rubrics": rubrics,
    }
    return validate_rubric_bundle(bundle, expected_task_id=task_id)


def main():
    args = parse_args()
    base = index_jsonl(args.base_rubrics, key="task_id")
    revisions = _load_revisions(args.revisions)
    expected_revisions = _revision_task_ids(args.cases)
    if set(revisions) != set(expected_revisions):
        raise SystemExit(
            "revision tasks must exactly match cases requiring a revised Rubric: "
            f"expected={sorted(expected_revisions)} actual={sorted(revisions)}"
        )
    case_ids = [int(case["task_id"]) for case in iter_jsonl(args.cases)]
    output_ids = list(base) if args.include_all_base_tasks else case_ids
    missing = sorted(set(case_ids) - set(base))
    if missing:
        raise SystemExit(f"base Rubrics lack calibration tasks: {missing}")
    rows = []
    for task_id in output_ids:
        row = base[task_id]
        if task_id in revisions:
            row = materialize_revision(row, revisions[task_id])
        else:
            row = validate_rubric_bundle(row, expected_task_id=task_id)
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rubrics.jsonl"
    write_jsonl_atomic(output, rows, force=args.force)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "rubric_count": len(rows),
        "includes_all_base_tasks": bool(args.include_all_base_tasks),
        "revised_task_ids": sorted(revisions),
        "base_rubrics_sha256": sha256_file(args.base_rubrics),
        "cases_sha256": sha256_file(args.cases),
        "revisions_sha256": sha256_file(args.revisions),
        "rubrics_sha256": sha256_file(output),
    }
    write_json_atomic(
        args.output_dir / "manifest.json", manifest, force=args.force
    )
    print(
        f"materialized {len(rows)} Rubrics; "
        f"human_revised={len(revisions)}"
    )


if __name__ == "__main__":
    main()
