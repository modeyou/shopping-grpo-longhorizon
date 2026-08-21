from copy import deepcopy

import pytest

from shopping_grpo.evaluation.comparison import (
    compare_multiturn_evaluation_grid,
)
from shopping_grpo.evaluation.contracts import CONTRACT_VERSION
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION


def _record(task_id, condition, *, success, questions=0, grounded=False):
    return {
        "schema_version": EVALUATION_RESULT_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "task_id": task_id,
        "reward_and_terminal": {
            "metrics": {
                "strict_gold_success": success,
                "purchase_success": success,
                "reward_type": "gold_purchase" if success else "wrong_purchase",
            }
        },
        "requirement_rubric": {
            "rubrics": [],
            "assessments": [],
            "reward_rubric_disagreement": False,
        },
        "trajectory_quality": {
            "judge_status": "not_judged",
            "dimension_scores": {},
            "errors": {"primary": None, "secondary": []},
        },
        "clarification": {
            "deterministic": {
                "interaction_mode": condition,
                "question_count": questions,
                "grounded_question_count": questions if grounded else 0,
                "all_questions_grounded": bool(questions and grounded),
                "gap_no_ask": condition.startswith("gap-") and questions == 0,
                "complete_unnecessary_ask": (
                    condition == "complete-ask-enabled" and questions > 0
                ),
                "auditable_post_answer_action": questions > 0,
            },
            "judge_assessment": {},
        },
        "deterministic": {
            "actions_and_efficiency": {
                "executed_tool_steps": 1,
                "executed_shop_steps": 1,
            },
            "repetition": {"duplicate_canonical_action_count": 0},
            "legality": {"guard_rejection_count": 0},
            "context": {},
            "validity": {},
        },
    }


def _actor(success_enabled):
    return {
        "gap-ask-enabled": [
            _record(1, "gap-ask-enabled", success=success_enabled, questions=1, grounded=True),
            _record(2, "gap-ask-enabled", success=False, questions=0),
        ],
        "gap-ask-disabled": [
            _record(1, "gap-ask-disabled", success=False),
            _record(2, "gap-ask-disabled", success=False),
        ],
        "complete-ask-enabled": [
            _record(1, "complete-ask-enabled", success=True, questions=0),
            _record(2, "complete-ask-enabled", success=True, questions=1),
        ],
    }


def test_grid_reports_causal_gain_and_overasking_without_total_score():
    grid = compare_multiturn_evaluation_grid(
        expected_task_ids=[1, 2],
        actors={"base": _actor(False), "sft": _actor(True)},
    )

    sft = grid["condition_effects_by_actor"]["sft"]
    assert sft["strict_success"]["gained_task_ids"] == [1]
    assert sft["strict_success"]["net_gained_tasks"] == 1
    assert sft["clarification"]["complete_unnecessary_ask_task_ids"] == [2]
    assert grid["model_progression_by_condition"]["gap-ask-enabled"]
    assert grid["composite_score"] is None


def test_grid_rejects_condition_label_mismatch():
    actor = _actor(True)
    bad = deepcopy(actor)
    bad["gap-ask-enabled"][0]["clarification"]["deterministic"][
        "interaction_mode"
    ] = "gap-ask-disabled"
    with pytest.raises(ValueError, match="interaction_mode"):
        compare_multiturn_evaluation_grid(
            expected_task_ids=[1, 2], actors={"bad": bad}
        )
