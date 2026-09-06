#!/usr/bin/env python3
"""Build the frozen G+/G-/C+ Base/SFT/GRPO comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import iter_jsonl, load_json, write_json_atomic
from shopping_grpo.evaluation.blind_guard import guard_declared_final_tasks
from shopping_grpo.evaluation.comparison import (
    MULTITURN_CONDITIONS,
    compare_evaluation_runs,
    compare_multiturn_evaluation_grid,
)
from shopping_grpo.evaluation.manifest import sha256_file


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
    parser.add_argument("--allow-blind-final", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _comparison_fingerprint(manifest: dict, *, expected_tasks_sha256: str) -> dict:
    """Return the inputs that must be identical for a fair model comparison."""

    task_manifest = manifest.get("task_manifest")
    protocol = manifest.get("protocol")
    judge = manifest.get("judge")
    outputs = manifest.get("outputs")
    environment = manifest.get("environment")
    run_plan = manifest.get("run_plan")
    if not all(
        isinstance(value, dict)
        for value in (
            task_manifest,
            protocol,
            judge,
            outputs,
            environment,
            run_plan,
        )
    ):
        raise ValueError("run manifest is missing a required object")
    if task_manifest.get("sha256") != expected_tasks_sha256:
        raise ValueError("run manifest task hash does not match --expected-tasks")
    rubrics_sha256 = outputs.get("rubrics_sha256")
    if not isinstance(rubrics_sha256, str) or len(rubrics_sha256) != 64:
        raise ValueError("run manifest is missing rubrics_sha256")
    request_sha256 = outputs.get("judge_requests_sha256")
    if not isinstance(request_sha256, str) or len(request_sha256) != 64:
        raise ValueError("run manifest is missing judge_requests_sha256")
    judge_execution = run_plan.get("judge")
    if not isinstance(judge_execution, dict):
        raise ValueError("run manifest is missing frozen judge execution config")
    return {
        "task_manifest_sha256": expected_tasks_sha256,
        "rubrics_sha256": rubrics_sha256,
        "rubric_schema": (manifest.get("artifact_schemas") or {}).get("rubric"),
        "judge": judge,
        "judge_execution": judge_execution,
        "trajectory_judge_prompt_version": (
            (manifest.get("prompt_versions") or {}).get("trajectory_judge")
        ),
        "reward_versions": environment.get("reward_versions"),
        "composite_score": protocol.get("composite_score"),
    }


def _load_verified_condition(
    root: Path,
    condition: str,
    *,
    expected_tasks_sha256: str,
) -> tuple[list[dict], dict]:
    output_dir = root / condition
    evaluations_path = output_dir / "evaluations.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("condition") != condition:
        raise ValueError(f"{manifest_path}: condition does not match {condition}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{manifest_path}: outputs must be an object")
    if outputs.get("evaluations_sha256") != sha256_file(evaluations_path):
        raise ValueError(f"{manifest_path}: evaluations hash mismatch")
    return list(iter_jsonl(evaluations_path)), _comparison_fingerprint(
        manifest,
        expected_tasks_sha256=expected_tasks_sha256,
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=MULTITURN_CONDITIONS,
        help="Condition to compare; omit only for the full G+/G-/C+ grid.",
    )


def main():
    args = parse_args()
    guard_declared_final_tasks(
        args.expected_tasks,
        allowed=args.allow_blind_final,
    )
    expected_tasks_sha256 = sha256_file(args.expected_tasks)
    conditions = tuple(args.condition or MULTITURN_CONDITIONS)
    actors = {}
    fingerprints = {}
    for label, root in args.run:
        if label in actors:
            raise SystemExit(f"duplicate actor label: {label}")
        actors[label] = {}
        for condition in conditions:
            evaluations, fingerprint = _load_verified_condition(
                root,
                condition,
                expected_tasks_sha256=expected_tasks_sha256,
            )
            actors[label][condition] = evaluations
            fingerprints[f"{label}/{condition}"] = fingerprint
    reference_key, reference = next(iter(fingerprints.items()))
    mismatches = {
        key: fingerprint
        for key, fingerprint in fingerprints.items()
        if fingerprint != reference
    }
    if mismatches:
        raise SystemExit(
            "comparison input manifests disagree with "
            f"{reference_key}: {sorted(mismatches)}"
        )
    expected_task_ids = _task_ids(args.expected_tasks)
    if conditions == MULTITURN_CONDITIONS:
        result = compare_multiturn_evaluation_grid(
            expected_task_ids=expected_task_ids,
            actors=actors,
        )
    else:
        result = {
            "schema_version": "shopping-partial-multiturn-evaluation-v1",
            "expected_tasks": len(expected_task_ids),
            "actors": list(actors),
            "conditions": list(conditions),
            "condition_effects_by_actor": None,
            "model_progression_by_condition": {
                condition: compare_evaluation_runs(
                    expected_task_ids=expected_task_ids,
                    runs={
                        label: condition_runs[condition]
                        for label, condition_runs in actors.items()
                    },
                )
                for condition in conditions
            },
            "composite_score": None,
        }
    result["comparison_protocol"] = reference
    write_json_atomic(args.output, result, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
