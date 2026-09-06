#!/usr/bin/env python3
"""Audit Judge outputs against a versioned, partially automated review set."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    index_jsonl,
    iter_jsonl,
    write_json_atomic,
)
from shopping_grpo.evaluation.manifest import canonical_json_sha256
from shopping_grpo.evaluation.prompts import (
    build_trajectory_judge_messages,
    deterministic_judge_facts,
)


CASE_VERSION = "shopping-judge-calibration-case-v1"
REPORT_VERSION = "shopping-judge-calibration-report-v2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=DIR containing preprocessed.jsonl and judges.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = str(value).partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("--run must use LABEL=DIR")
    return label.strip(), Path(raw_path)


def _load_cases(path: Path) -> list[dict]:
    cases = list(iter_jsonl(path))
    seen = set()
    for index, case in enumerate(cases, start=1):
        if case.get("schema_version") != CASE_VERSION:
            raise ValueError(f"case {index} has unsupported schema_version")
        task_id = int(case["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate calibration task {task_id}")
        seen.add(task_id)
        if case.get("disposition") not in {
            "model_issue",
            "evaluator_error",
            "mixed_review",
        }:
            raise ValueError(f"task {task_id} has invalid disposition")
        if not isinstance(case.get("actors"), Mapping):
            raise ValueError(f"task {task_id} actors must be an object")
        if not isinstance(case.get("manual_review_required"), bool):
            raise ValueError(
                f"task {task_id} manual_review_required must be boolean"
            )
        human_review = case.get("human_review")
        if human_review is not None and not isinstance(human_review, Mapping):
            raise ValueError(f"task {task_id} human_review must be an object")
        if case["manual_review_required"] and human_review is not None:
            raise ValueError(
                f"task {task_id} cannot be pending and human-reviewed"
            )
    if not cases:
        raise ValueError("calibration set must not be empty")
    return cases


def _fact_index(facts: Mapping, name: str) -> dict[str, str]:
    return {
        str(item["rubric_id"]): str(item["status"])
        for item in facts.get(name) or []
    }


def _judge_statuses(judge_result: Mapping) -> dict[str, str]:
    return {
        str(item["rubric_id"]): str(item["status"])
        for item in judge_result.get("rubric_assessments") or []
    }


def _without_trajectory_id(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_trajectory_id(child)
            for key, child in value.items()
            if key != "trajectory_id"
        }
    if isinstance(value, list):
        return [_without_trajectory_id(child) for child in value]
    return deepcopy(value)


def audit_calibration(
    *,
    cases: list[Mapping],
    rubrics: Mapping[int, Mapping],
    runs: Mapping[str, Mapping[str, Mapping[int, Mapping]]],
) -> dict:
    assertions = []
    manual_tasks = []
    human_reviewed_tasks = []
    rubric_revision_tasks = []

    def record(task_id: int, check: str, passed: bool, detail: str) -> None:
        assertions.append(
            {
                "task_id": task_id,
                "check": check,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    for case in cases:
        task_id = int(case["task_id"])
        rubric = rubrics.get(task_id)
        if rubric is None:
            raise ValueError(f"missing rubric for calibration task {task_id}")
        if case["manual_review_required"]:
            manual_tasks.append(task_id)
        if case.get("human_review") is not None:
            human_reviewed_tasks.append(task_id)
        expected_rubric_version = case.get("expected_rubric_version")
        if (
            expected_rubric_version is not None
            and rubric.get("rubric_version") != expected_rubric_version
        ):
            rubric_revision_tasks.append(task_id)
        for actor, expected in case["actors"].items():
            if actor not in runs:
                raise ValueError(f"calibration task {task_id} needs run {actor!r}")
            preprocessed = runs[actor]["preprocessed"].get(task_id)
            judge_row = runs[actor]["judges"].get(task_id)
            if preprocessed is None or judge_row is None:
                raise ValueError(f"run {actor!r} lacks calibration task {task_id}")
            normalized = preprocessed["normalized_trajectory"]
            facts = deterministic_judge_facts(
                normalized=normalized,
                rubric_bundle=rubric,
            )
            judge = judge_row["judge_result"]
            judge_status = _judge_statuses(judge)
            for rubric_id, expected_status in (
                expected.get("rubric_statuses") or {}
            ).items():
                record(
                    task_id,
                    f"{actor}.rubric_statuses.{rubric_id}",
                    judge_status.get(rubric_id) == expected_status,
                    (
                        f"expected={expected_status} "
                        f"actual={judge_status.get(rubric_id)}"
                    ),
                )
            for fact_name in ("price_checks", "option_checks"):
                actual = _fact_index(facts, fact_name)
                for rubric_id, expected_status in (
                    expected.get(fact_name) or {}
                ).items():
                    fact_passed = actual.get(rubric_id) == expected_status
                    record(
                        task_id,
                        f"{actor}.{fact_name}.{rubric_id}.fact",
                        fact_passed,
                        f"expected={expected_status} actual={actual.get(rubric_id)}",
                    )
                    required_judge = {
                        "pass": "satisfied",
                        "fail": "violated",
                    }.get(expected_status)
                    if required_judge:
                        record(
                            task_id,
                            f"{actor}.{fact_name}.{rubric_id}.judge",
                            judge_status.get(rubric_id) == required_judge,
                            (
                                f"expected={required_judge} "
                                f"actual={judge_status.get(rubric_id)}"
                            ),
                        )
            if "termination_reason" in expected:
                actual_reason = preprocessed["deterministic_metrics"][
                    "reward_and_outcome"
                ].get("termination_reason")
                record(
                    task_id,
                    f"{actor}.termination_reason",
                    actual_reason == expected["termination_reason"],
                    (
                        f"expected={expected['termination_reason']} "
                        f"actual={actual_reason}"
                    ),
                )
            errors = judge.get("errors") or {}
            labels = {errors.get("primary"), *(errors.get("secondary") or [])}
            for forbidden in expected.get("forbidden_errors") or []:
                record(
                    task_id,
                    f"{actor}.forbidden_error.{forbidden}",
                    forbidden not in labels,
                    f"judge_errors={sorted(str(label) for label in labels if label)}",
                )
        pair = case.get("pair")
        if pair:
            left_label, right_label = pair["actors"]
            left = runs[left_label]
            right = runs[right_label]
            left_pre = left["preprocessed"][task_id]
            right_pre = right["preprocessed"][task_id]
            left_messages = build_trajectory_judge_messages(
                normalized=left_pre["normalized_trajectory"],
                rubric_bundle=rubric,
                deterministic_metrics=left_pre["deterministic_metrics"],
            )
            right_messages = build_trajectory_judge_messages(
                normalized=right_pre["normalized_trajectory"],
                rubric_bundle=rubric,
                deterministic_metrics=right_pre["deterministic_metrics"],
            )
            if pair.get("semantic_request_equal"):
                record(
                    task_id,
                    "pair.semantic_request_equal",
                    left_messages == right_messages,
                    (
                        f"left={canonical_json_sha256(left_messages)} "
                        f"right={canonical_json_sha256(right_messages)}"
                    ),
                )
            if pair.get("judge_output_equal"):
                left_judge = _without_trajectory_id(
                    left["judges"][task_id]["judge_result"]
                )
                right_judge = _without_trajectory_id(
                    right["judges"][task_id]["judge_result"]
                )
                record(
                    task_id,
                    "pair.judge_output_equal",
                    left_judge == right_judge,
                    (
                        f"left={canonical_json_sha256(left_judge)} "
                        f"right={canonical_json_sha256(right_judge)}"
                    ),
                )

    failures = [item for item in assertions if not item["passed"]]
    return {
        "schema_version": REPORT_VERSION,
        "case_count": len(cases),
        "automated_assertion_count": len(assertions),
        "automated_pass_count": len(assertions) - len(failures),
        "automated_failure_count": len(failures),
        "manual_review_task_count": len(set(manual_tasks)),
        "manual_review_task_ids": sorted(set(manual_tasks)),
        "human_reviewed_task_count": len(set(human_reviewed_tasks)),
        "human_reviewed_task_ids": sorted(set(human_reviewed_tasks)),
        "rubric_revision_task_count": len(set(rubric_revision_tasks)),
        "rubric_revision_task_ids": sorted(set(rubric_revision_tasks)),
        "release_ready": (
            not failures and not manual_tasks and not rubric_revision_tasks
        ),
        "assertions": assertions,
    }


def main():
    args = parse_args()
    cases = _load_cases(args.cases)
    rubrics = index_jsonl(args.rubrics, key="task_id")
    run_specs = [_run_spec(value) for value in args.run]
    if len({label for label, _ in run_specs}) != len(run_specs):
        raise SystemExit("--run labels must be unique")
    runs = {}
    for label, root in run_specs:
        runs[label] = {
            "preprocessed": index_jsonl(
                root / "preprocessed.jsonl",
                key="task_id",
            ),
            "judges": index_jsonl(
                root / "judges.jsonl", key="task_id"
            ),
        }
    report = audit_calibration(cases=cases, rubrics=rubrics, runs=runs)
    write_json_atomic(args.output, report, force=args.force)
    print(
        f"calibration cases={report['case_count']} "
        f"assertions={report['automated_assertion_count']} "
        f"failures={report['automated_failure_count']} "
        f"manual_pending={report['manual_review_task_count']} "
        f"rubric_revision_pending={report['rubric_revision_task_count']}"
    )
    if report["automated_failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
