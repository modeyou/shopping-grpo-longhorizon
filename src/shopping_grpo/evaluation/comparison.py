"""Paired Baseline/SFT/GRPO diagnostics without a composite score."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import combinations

from shopping_grpo.evaluation.contracts import CONTRACT_VERSION, JUDGE_DIMENSIONS
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION


COMPARISON_SCHEMA_VERSION = "shopping-paired-model-comparison-v2"
MULTITURN_GRID_SCHEMA_VERSION = "shopping-multiturn-evaluation-grid-v1"
MULTITURN_CONDITIONS = (
    "gap-ask-enabled",
    "gap-ask-disabled",
    "complete-ask-enabled",
)


def _index_run(
    *,
    label: str,
    evaluations: Iterable[Mapping],
    expected: set[int],
) -> dict[int, Mapping]:
    result = {}
    for record in evaluations:
        if record.get("schema_version") != EVALUATION_RESULT_VERSION:
            raise ValueError(f"{label} contains an unsupported evaluation schema")
        task_id = int(record["task_id"])
        if task_id not in expected:
            raise ValueError(f"{label} contains unexpected task_id {task_id}")
        if task_id in result:
            raise ValueError(f"{label} contains duplicate task_id {task_id}")
        result[task_id] = record
    return result


def _hard_violation_count(record: Mapping) -> int | None:
    quality = record["trajectory_quality"]
    if quality.get("judge_status") != "valid":
        return None
    rubric = record["requirement_rubric"]
    hardness = {
        item["rubric_id"]: item["hardness"]
        for item in rubric["rubrics"]
    }
    return sum(
        assessment["status"] == "violated"
        and hardness.get(assessment["rubric_id"]) == "hard"
        for assessment in rubric["assessments"]
    )


def _dimension_score(record: Mapping, name: str) -> int | None:
    quality = record["trajectory_quality"]
    if quality.get("judge_status") != "valid":
        return None
    return int(quality["dimension_scores"][name]["score"])


def _delta_summary(deltas: list[float], *, lower_is_better: bool) -> dict:
    if lower_is_better:
        improved = sum(delta < 0 for delta in deltas)
        worsened = sum(delta > 0 for delta in deltas)
    else:
        improved = sum(delta > 0 for delta in deltas)
        worsened = sum(delta < 0 for delta in deltas)
    return {
        "paired_tasks": len(deltas),
        "mean_delta_target_minus_source": (
            sum(deltas) / len(deltas) if deltas else 0.0
        ),
        "improved_tasks": improved,
        "unchanged_tasks": sum(delta == 0 for delta in deltas),
        "worsened_tasks": worsened,
    }


def _pairwise(
    source_label: str,
    source: Mapping[int, Mapping],
    target_label: str,
    target: Mapping[int, Mapping],
) -> dict:
    paired_ids = sorted(set(source) & set(target))
    strict_transitions = Counter()
    reward_type_transitions = Counter()
    disagreement_transitions = Counter()
    hard_deltas = []
    dimension_deltas = {name: [] for name in JUDGE_DIMENSIONS}
    tool_step_deltas = []
    shop_step_deltas = []
    guard_deltas = []
    duplicate_action_deltas = []
    question_count_deltas = []
    grounded_question_deltas = []
    post_answer_action_deltas = []

    for task_id in paired_ids:
        left = source[task_id]
        right = target[task_id]
        left_reward = left["reward_and_terminal"]["metrics"]
        right_reward = right["reward_and_terminal"]["metrics"]
        left_success = bool(left_reward.get("strict_gold_success"))
        right_success = bool(right_reward.get("strict_gold_success"))
        strict_transitions[
            f"{'success' if left_success else 'failure'}_to_"
            f"{'success' if right_success else 'failure'}"
        ] += 1
        reward_type_transitions[
            f"{left_reward.get('reward_type', 'unknown')} -> "
            f"{right_reward.get('reward_type', 'unknown')}"
        ] += 1

        left_disagreement = bool(
            left["requirement_rubric"]["reward_rubric_disagreement"]
        )
        right_disagreement = bool(
            right["requirement_rubric"]["reward_rubric_disagreement"]
        )
        disagreement_transitions[
            f"{'disagreement' if left_disagreement else 'aligned'}_to_"
            f"{'disagreement' if right_disagreement else 'aligned'}"
        ] += 1

        left_hard = _hard_violation_count(left)
        right_hard = _hard_violation_count(right)
        if left_hard is not None and right_hard is not None:
            hard_deltas.append(right_hard - left_hard)
        for name in JUDGE_DIMENSIONS:
            left_score = _dimension_score(left, name)
            right_score = _dimension_score(right, name)
            if left_score is not None and right_score is not None:
                dimension_deltas[name].append(right_score - left_score)

        left_deterministic = left["deterministic"]
        right_deterministic = right["deterministic"]
        tool_step_deltas.append(
            right_deterministic["actions_and_efficiency"][
                "executed_tool_steps"
            ]
            - left_deterministic["actions_and_efficiency"][
                "executed_tool_steps"
            ]
        )
        shop_step_deltas.append(
            right_deterministic["actions_and_efficiency"][
                "executed_shop_steps"
            ]
            - left_deterministic["actions_and_efficiency"][
                "executed_shop_steps"
            ]
        )
        guard_deltas.append(
            right_deterministic["legality"]["guard_rejection_count"]
            - left_deterministic["legality"]["guard_rejection_count"]
        )
        duplicate_action_deltas.append(
            right_deterministic["repetition"][
                "duplicate_canonical_action_count"
            ]
            - left_deterministic["repetition"][
                "duplicate_canonical_action_count"
            ]
        )
        left_clarification = (left.get("clarification") or {}).get(
            "deterministic"
        ) or {}
        right_clarification = (right.get("clarification") or {}).get(
            "deterministic"
        ) or {}
        question_count_deltas.append(
            int(right_clarification.get("question_count") or 0)
            - int(left_clarification.get("question_count") or 0)
        )
        grounded_question_deltas.append(
            int(right_clarification.get("grounded_question_count") or 0)
            - int(left_clarification.get("grounded_question_count") or 0)
        )
        post_answer_action_deltas.append(
            int(bool(right_clarification.get("auditable_post_answer_action")))
            - int(bool(left_clarification.get("auditable_post_answer_action")))
        )

    return {
        "source": source_label,
        "target": target_label,
        "paired_tasks": len(paired_ids),
        "paired_task_ids": paired_ids,
        "reward_and_terminal": {
            "strict_success_transitions": dict(
                sorted(strict_transitions.items())
            ),
            "reward_type_transitions": dict(
                sorted(reward_type_transitions.items())
            ),
        },
        "requirement_rubric": {
            "hard_violation_delta": _delta_summary(
                hard_deltas,
                lower_is_better=True,
            ),
            "reward_rubric_disagreement_transitions": dict(
                sorted(disagreement_transitions.items())
            ),
        },
        "trajectory_quality": {
            name: _delta_summary(deltas, lower_is_better=False)
            for name, deltas in dimension_deltas.items()
        },
        "clarification": {
            "question_count": _delta_summary(
                question_count_deltas, lower_is_better=False
            ),
            "grounded_question_count": _delta_summary(
                grounded_question_deltas, lower_is_better=False
            ),
            "auditable_post_answer_action": _delta_summary(
                post_answer_action_deltas, lower_is_better=False
            ),
        },
        "deterministic": {
            "executed_tool_steps": _delta_summary(
                tool_step_deltas,
                lower_is_better=True,
            ),
            "executed_shop_steps": _delta_summary(
                shop_step_deltas,
                lower_is_better=True,
            ),
            "guard_rejections": _delta_summary(
                guard_deltas,
                lower_is_better=True,
            ),
            "duplicate_canonical_actions": _delta_summary(
                duplicate_action_deltas,
                lower_is_better=True,
            ),
        },
    }

def compare_evaluation_runs(
    *,
    expected_task_ids: Iterable[int],
    runs: Mapping[str, Iterable[Mapping]],
) -> dict:
    """Compare each model pair on identical task IDs, section by section."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    if len(runs) < 2:
        raise ValueError("at least two model runs are required")
    expected_set = set(expected)
    indexed = {
        str(label): _index_run(
            label=str(label),
            evaluations=evaluations,
            expected=expected_set,
        )
        for label, evaluations in runs.items()
    }
    labels = list(indexed)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "expected_tasks": len(expected),
        "models": {
            label: {
                "completed_evaluations": len(indexed[label]),
                "missing_task_ids": sorted(
                    expected_set - set(indexed[label])
                ),
            }
            for label in labels
        },
        "pairwise": {
            f"{source}_to_{target}": _pairwise(
                source,
                indexed[source],
                target,
                indexed[target],
            )
            for source, target in combinations(labels, 2)
        },
    }


def _success(record: Mapping, field: str) -> bool:
    return bool(record["reward_and_terminal"]["metrics"].get(field))


def _condition_effect(
    *,
    expected: set[int],
    gap_enabled: Mapping[int, Mapping],
    gap_disabled: Mapping[int, Mapping],
    complete_enabled: Mapping[int, Mapping],
) -> dict:
    paired_gap_ids = sorted(set(gap_enabled) & set(gap_disabled))
    strict_transitions = Counter()
    purchase_transitions = Counter()
    gained = []
    lost = []
    for task_id in paired_gap_ids:
        disabled_strict = _success(
            gap_disabled[task_id], "strict_gold_success"
        )
        enabled_strict = _success(
            gap_enabled[task_id], "strict_gold_success"
        )
        strict_transitions[
            f"{'success' if disabled_strict else 'failure'}_to_"
            f"{'success' if enabled_strict else 'failure'}"
        ] += 1
        if not disabled_strict and enabled_strict:
            gained.append(task_id)
        if disabled_strict and not enabled_strict:
            lost.append(task_id)

        disabled_purchase = _success(
            gap_disabled[task_id], "purchase_success"
        )
        enabled_purchase = _success(
            gap_enabled[task_id], "purchase_success"
        )
        purchase_transitions[
            f"{'success' if disabled_purchase else 'failure'}_to_"
            f"{'success' if enabled_purchase else 'failure'}"
        ] += 1

    gap_enabled_successes = sum(
        _success(record, "strict_gold_success")
        for record in gap_enabled.values()
    )
    gap_disabled_successes = sum(
        _success(record, "strict_gold_success")
        for record in gap_disabled.values()
    )
    grounded_asked_tasks = []
    gap_no_ask_tasks = []
    for task_id, record in gap_enabled.items():
        clarification = (record.get("clarification") or {}).get(
            "deterministic"
        ) or {}
        if clarification.get("question_count") and clarification.get(
            "all_questions_grounded"
        ):
            grounded_asked_tasks.append(task_id)
        if clarification.get("gap_no_ask"):
            gap_no_ask_tasks.append(task_id)

    unnecessary_ask_tasks = []
    for task_id, record in complete_enabled.items():
        clarification = (record.get("clarification") or {}).get(
            "deterministic"
        ) or {}
        if clarification.get("complete_unnecessary_ask"):
            unnecessary_ask_tasks.append(task_id)

    denominator = len(expected)
    return {
        "expected_tasks": denominator,
        "paired_gap_tasks": len(paired_gap_ids),
        "paired_gap_task_ids": paired_gap_ids,
        "strict_success": {
            "gap_ask_enabled": gap_enabled_successes,
            "gap_ask_disabled": gap_disabled_successes,
            "rate_delta_enabled_minus_disabled": (
                (gap_enabled_successes - gap_disabled_successes) / denominator
                if denominator
                else 0.0
            ),
            "transitions_disabled_to_enabled": dict(
                sorted(strict_transitions.items())
            ),
            "gained_task_ids": gained,
            "lost_task_ids": lost,
            "net_gained_tasks": len(gained) - len(lost),
        },
        "purchase_success": {
            "transitions_disabled_to_enabled": dict(
                sorted(purchase_transitions.items())
            )
        },
        "clarification": {
            "grounded_asked_tasks": len(grounded_asked_tasks),
            "grounded_asked_task_ids": sorted(grounded_asked_tasks),
            "gap_no_ask_tasks": len(gap_no_ask_tasks),
            "gap_no_ask_task_ids": sorted(gap_no_ask_tasks),
            "complete_unnecessary_ask_tasks": len(unnecessary_ask_tasks),
            "complete_unnecessary_ask_task_ids": sorted(
                unnecessary_ask_tasks
            ),
            "complete_unnecessary_ask_rate": (
                len(unnecessary_ask_tasks) / denominator
                if denominator
                else 0.0
            ),
        },
    }


def compare_multiturn_evaluation_grid(
    *,
    expected_task_ids: Iterable[int],
    actors: Mapping[str, Mapping[str, Iterable[Mapping]]],
) -> dict:
    """Compare Base/SFT/GRPO across G+, G- and C+ without a total score."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected_task_ids contains duplicates")
    if not actors:
        raise ValueError("at least one actor is required")
    expected_set = set(expected)
    indexed = {}
    for actor_label, condition_runs in actors.items():
        missing_conditions = set(MULTITURN_CONDITIONS) - set(condition_runs)
        unexpected_conditions = set(condition_runs) - set(MULTITURN_CONDITIONS)
        if missing_conditions or unexpected_conditions:
            raise ValueError(
                f"{actor_label} condition mismatch: "
                f"missing={sorted(missing_conditions)} "
                f"unexpected={sorted(unexpected_conditions)}"
            )
        indexed[str(actor_label)] = {}
        for condition in MULTITURN_CONDITIONS:
            run = _index_run(
                label=f"{actor_label}/{condition}",
                evaluations=condition_runs[condition],
                expected=expected_set,
            )
            for task_id, record in run.items():
                clarification = record.get("clarification") or {}
                deterministic = clarification.get("deterministic") or {}
                actual = str(
                    deterministic.get("interaction_mode") or ""
                )
                if actual != condition:
                    raise ValueError(
                        f"{actor_label}/{condition} task {task_id} has "
                        f"interaction_mode={actual!r}"
                    )
            indexed[str(actor_label)][condition] = run

    by_actor = {
        actor: _condition_effect(
            expected=expected_set,
            gap_enabled=runs["gap-ask-enabled"],
            gap_disabled=runs["gap-ask-disabled"],
            complete_enabled=runs["complete-ask-enabled"],
        )
        for actor, runs in indexed.items()
    }
    by_condition = {}
    if len(indexed) >= 2:
        for condition in MULTITURN_CONDITIONS:
            by_condition[condition] = compare_evaluation_runs(
                expected_task_ids=expected,
                runs={
                    actor: runs[condition].values()
                    for actor, runs in indexed.items()
                },
            )
    return {
        "schema_version": MULTITURN_GRID_SCHEMA_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "expected_tasks": len(expected),
        "actors": list(indexed),
        "conditions": list(MULTITURN_CONDITIONS),
        "condition_effects_by_actor": by_actor,
        "model_progression_by_condition": by_condition,
        "composite_score": None,
    }
