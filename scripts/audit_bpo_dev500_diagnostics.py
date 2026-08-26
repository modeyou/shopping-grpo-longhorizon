#!/usr/bin/env python3
"""Diagnose a BPO dev500 result against an existing paired SFT baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from shopping_grpo.evaluation.summary import REWARD_V4, _is_strict_success


CONDITIONS = (
    "gap-ask-enabled",
    "gap-ask-disabled",
    "complete-ask-enabled",
)
VALIDATION_STEPS = (10, 50, 100, 150, 200)
VALIDATION_KEYS = (
    "strict_success_rate",
    "purchase_success_rate",
    "mean_reward",
    "terminal_utility_mean",
    "done_rate",
    "reward_valid_rate",
    "sampling_invalid_rate",
    "infrastructure_invalid_rate",
    "reward_unverifiable_rate",
)
STEP_PATTERN = re.compile(
    r"(?:training/global_step|global_step|step)['\"]?\s*[:=]\s*(\d+)"
)
METRIC_PATTERN = re.compile(
    r"['\"]?val-shopping/summary/([A-Za-z0-9_]+)['\"]?\s*[:=]\s*"
    r"(?:np\.(?:float16|float32|float64)\()?([-+]?\d+(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?)"
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reward_detail(row: dict) -> dict:
    detail = ((row.get("terminal_result") or {}).get("reward_detail") or {})
    return detail if isinstance(detail, dict) else {}


def _index(rows: list[dict], expected: set[int], condition: str, label: str) -> dict:
    indexed = {}
    for row in rows:
        task_id = int(row["task_id"])
        if task_id not in expected:
            raise ValueError(f"{label} contains unexpected task_id {task_id}")
        if task_id in indexed:
            raise ValueError(f"{label} contains duplicate task_id {task_id}")
        if row.get("interaction_mode") != condition:
            raise ValueError(
                f"{label} task {task_id} condition mismatch: "
                f"{row.get('interaction_mode')!r}"
            )
        detail = _reward_detail(row)
        if detail.get("reward_version") != REWARD_V4:
            raise ValueError(f"{label} task {task_id} did not use Reward v4")
        indexed[task_id] = row
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"{label} task set mismatch; missing={missing[:10]}")
    return indexed


def _question_count(row: dict) -> int:
    return len(row.get("shopper_questions") or [])


def _guard_count(row: dict) -> int:
    return len(row.get("blocked_tool_calls") or [])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome(row: dict) -> dict:
    detail = _reward_detail(row)
    terminal = row.get("terminal_result") or {}
    return {
        "strict_success": bool(_is_strict_success(row)),
        "reward_type": str(detail.get("reward_type") or "unknown"),
        "reward_valid": bool(detail.get("reward_valid")),
        "done": bool(row.get("done") or terminal.get("done")),
        "termination_reason": str(
            detail.get("termination_reason")
            or terminal.get("termination_reason")
            or row.get("termination_reason")
            or "unknown"
        ),
        "final_reward": float(row.get("final_reward", 0.0)),
        "question_count": _question_count(row),
        "guard_count": _guard_count(row),
    }


def compare_condition(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    expected_task_ids: list[int],
    condition: str,
) -> dict:
    expected = set(expected_task_ids)
    baseline = _index(baseline_rows, expected, condition, f"baseline/{condition}")
    candidate = _index(candidate_rows, expected, condition, f"candidate/{condition}")
    strict_transitions = Counter()
    reward_type_transitions = Counter()
    strict_gains = []
    strict_losses = []
    guard_increased = []
    guard_decreased = []
    question_increased = []
    question_decreased = []
    reward_deltas = []
    strict_flip_records = []

    for task_id in expected_task_ids:
        left = baseline[task_id]
        right = candidate[task_id]
        left_success = _is_strict_success(left)
        right_success = _is_strict_success(right)
        transition = (
            f"{'success' if left_success else 'failure'}_to_"
            f"{'success' if right_success else 'failure'}"
        )
        strict_transitions[transition] += 1
        if not left_success and right_success:
            strict_gains.append(task_id)
        elif left_success and not right_success:
            strict_losses.append(task_id)

        if left_success != right_success:
            left_outcome = _outcome(left)
            right_outcome = _outcome(right)
            strict_flip_records.append(
                {
                    "task_id": task_id,
                    "direction": "gain" if right_success else "loss",
                    "baseline": left_outcome,
                    "candidate": right_outcome,
                    "question_delta": (
                        right_outcome["question_count"]
                        - left_outcome["question_count"]
                    ),
                    "guard_delta": (
                        right_outcome["guard_count"]
                        - left_outcome["guard_count"]
                    ),
                    "reward_delta": (
                        right_outcome["final_reward"]
                        - left_outcome["final_reward"]
                    ),
                }
            )

        left_detail = _reward_detail(left)
        right_detail = _reward_detail(right)
        reward_type_transitions[
            f"{left_detail.get('reward_type', 'unknown')} -> "
            f"{right_detail.get('reward_type', 'unknown')}"
        ] += 1
        guard_delta = _guard_count(right) - _guard_count(left)
        if guard_delta > 0:
            guard_increased.append(task_id)
        elif guard_delta < 0:
            guard_decreased.append(task_id)
        question_delta = _question_count(right) - _question_count(left)
        if question_delta > 0:
            question_increased.append(task_id)
        elif question_delta < 0:
            question_decreased.append(task_id)
        reward_deltas.append(
            float(right.get("final_reward", 0.0))
            - float(left.get("final_reward", 0.0))
        )

    baseline_guards = sum(_guard_count(row) for row in baseline.values())
    candidate_guards = sum(_guard_count(row) for row in candidate.values())
    baseline_questions = sum(_question_count(row) for row in baseline.values())
    candidate_questions = sum(_question_count(row) for row in candidate.values())
    return {
        "paired_tasks": len(expected_task_ids),
        "strict_transitions": dict(sorted(strict_transitions.items())),
        "strict_gains": strict_gains,
        "strict_losses": strict_losses,
        "strict_flip_records": strict_flip_records,
        "strict_net": len(strict_gains) - len(strict_losses),
        "mean_reward_delta": sum(reward_deltas) / len(reward_deltas),
        "baseline_guards": baseline_guards,
        "candidate_guards": candidate_guards,
        "guard_delta": candidate_guards - baseline_guards,
        "guard_increased_task_ids": guard_increased,
        "guard_decreased_task_ids": guard_decreased,
        "baseline_questions": baseline_questions,
        "candidate_questions": candidate_questions,
        "question_delta": candidate_questions - baseline_questions,
        "question_increased_task_ids": question_increased,
        "question_decreased_task_ids": question_decreased,
        "reward_type_transitions": dict(
            reward_type_transitions.most_common()
        ),
    }


def parse_validation_curve(log_text: str) -> dict[int, dict[str, float]]:
    curve: dict[int, dict[str, float]] = {}
    current_step = None
    for line in log_text.splitlines():
        step_match = STEP_PATTERN.search(line)
        if step_match:
            current_step = int(step_match.group(1))
        metrics = {
            name: float(value)
            for name, value in METRIC_PATTERN.findall(line)
            if name in VALIDATION_KEYS
        }
        if metrics and current_step in VALIDATION_STEPS:
            curve.setdefault(current_step, {}).update(metrics)
    return curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.assets.resolve() / "tasks.jsonl")
    expected_task_ids = [int(row["task_id"]) for row in tasks]
    if len(expected_task_ids) != 500 or len(set(expected_task_ids)) != 500:
        raise SystemExit("expected one frozen dev500 task set")

    comparisons = {}
    for condition in CONDITIONS:
        comparisons[condition] = compare_condition(
            read_jsonl(
                args.baseline_root.resolve() / condition / "trajectories.jsonl"
            ),
            read_jsonl(
                args.candidate_root.resolve() / condition / "trajectories.jsonl"
            ),
            expected_task_ids,
            condition,
        )

    curve = {}
    if args.training_log is not None:
        curve = parse_validation_curve(
            args.training_log.resolve().read_text(
                encoding="utf-8", errors="replace"
            )
        )
    baseline_paths = {
        condition: (
            args.baseline_root.resolve() / condition / "trajectories.jsonl"
        )
        for condition in CONDITIONS
    }
    candidate_paths = {
        condition: (
            args.candidate_root.resolve() / condition / "trajectories.jsonl"
        )
        for condition in CONDITIONS
    }
    result = {
        "schema_version": "shopping-bpo-dev500-diagnostics-v1",
        "input_audit": {
            "tasks": {
                "path": str((args.assets.resolve() / "tasks.jsonl")),
                "sha256": _sha256(args.assets.resolve() / "tasks.jsonl"),
                "rows": len(expected_task_ids),
            },
            "baseline": {
                condition: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for condition, path in baseline_paths.items()
            },
            "candidate": {
                condition: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for condition, path in candidate_paths.items()
            },
        },
        "validation_curve": {str(step): curve.get(step, {}) for step in VALIDATION_STEPS},
        "paired_comparison": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("VALIDATION CURVE")
    print("step\tstrict\tpurchase\tmean_reward\tdone\treward_valid\tinvalid")
    for step in VALIDATION_STEPS:
        metrics = curve.get(step, {})
        print(
            f"{step}\t{metrics.get('strict_success_rate', float('nan')):.4f}\t"
            f"{metrics.get('purchase_success_rate', float('nan')):.4f}\t"
            f"{metrics.get('mean_reward', float('nan')):.4f}\t"
            f"{metrics.get('done_rate', float('nan')):.4f}\t"
            f"{metrics.get('reward_valid_rate', float('nan')):.4f}\t"
            f"{metrics.get('sampling_invalid_rate', float('nan')):.4f}"
        )

    print("\nPAIRED DEV500")
    print("condition\tgains\tlosses\tnet\tguard_delta\tquestion_delta\treward_delta")
    for condition in CONDITIONS:
        item = comparisons[condition]
        print(
            f"{condition}\t{len(item['strict_gains'])}\t"
            f"{len(item['strict_losses'])}\t{item['strict_net']:+d}\t"
            f"{item['guard_delta']:+d}\t{item['question_delta']:+d}\t"
            f"{item['mean_reward_delta']:+.4f}"
        )
        for direction in ("gain", "loss"):
            flips = [
                row
                for row in item["strict_flip_records"]
                if row["direction"] == direction
            ]
            question_changes = Counter(
                "up" if row["question_delta"] > 0 else
                "down" if row["question_delta"] < 0 else "same"
                for row in flips
            )
            guard_changes = Counter(
                "up" if row["guard_delta"] > 0 else
                "down" if row["guard_delta"] < 0 else "same"
                for row in flips
            )
            print(
                f"  {direction}: questions={dict(question_changes)} "
                f"guards={dict(guard_changes)}"
            )
    print(f"\noutput: {args.output.resolve()}")
    print("BPO DEV500 OFFLINE DIAGNOSTICS ACCEPTED")


if __name__ == "__main__":
    main()
