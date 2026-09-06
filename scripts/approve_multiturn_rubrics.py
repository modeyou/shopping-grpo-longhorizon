#!/usr/bin/env python3
"""Approve an exactly reviewed Rubric JSONL and bind approval to its hash."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.contracts import (
    RUBRIC_APPROVAL_SCHEMA_VERSION,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.manifest import sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--confirm-reviewed",
        action="store_true",
        help="Attest that every Rubric was reviewed against its Query.",
    )
    return parser.parse_args()


def approve_bundles(
    rows: list[dict], *, reviewer: str, reviewed_at: str
) -> list[dict]:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer must be non-empty")
    approved = []
    seen_task_ids = set()
    for raw in rows:
        bundle = validate_rubric_bundle(raw)
        task_id = bundle["task_id"]
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id {task_id}")
        seen_task_ids.add(task_id)
        unresolved = [
            item["rubric_id"]
            for item in bundle["rubrics"]
            if item["hardness"] == "needs_review"
        ]
        if unresolved:
            raise ValueError(
                f"task {task_id} has unresolved needs_review Rubrics: {unresolved}"
            )
        result = deepcopy(bundle)
        result["review"] = {
            "status": "approved",
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
        }
        approved.append(validate_rubric_bundle(result, require_approved=True))
    if not approved:
        raise ValueError("rubrics must contain at least one task")
    return approved


def main():
    args = parse_args()
    if not args.confirm_reviewed:
        raise SystemExit(
            "refusing to approve without --confirm-reviewed; inspect every Rubric first"
        )
    reviewed_at = datetime.now(timezone.utc).isoformat()
    rows = list(iter_jsonl(args.rubrics))
    approved = approve_bundles(
        rows, reviewer=args.reviewer, reviewed_at=reviewed_at
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    approved_path = args.output_dir / "rubrics.approved.jsonl"
    manifest_path = args.output_dir / "approval_manifest.json"
    write_jsonl_atomic(approved_path, approved)
    manifest = {
        "schema_version": RUBRIC_APPROVAL_SCHEMA_VERSION,
        "reviewer": args.reviewer.strip(),
        "approved_at": reviewed_at,
        "task_count": len(approved),
        "source_rubrics_sha256": sha256_file(args.rubrics),
        "approved_rubrics_sha256": sha256_file(approved_path),
    }
    write_json_atomic(manifest_path, manifest)
    print(f"approved {len(approved)} Rubric bundles: {approved_path}")
    print(f"approval manifest: {manifest_path}")


if __name__ == "__main__":
    main()
