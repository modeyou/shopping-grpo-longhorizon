import json

import pytest

from scripts.audit_bpo_dev500_diagnostics import (
    compare_condition,
    parse_validation_curve,
)


def trajectory(
    task_id,
    *,
    condition="gap-ask-enabled",
    success=False,
    guards=0,
    questions=0,
    reward=0.0,
):
    reward_type = "gold_purchase" if success else "wrong_purchase"
    return {
        "task_id": task_id,
        "interaction_mode": condition,
        "status": "done",
        "done": True,
        "final_reward": reward,
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "reward_type": reward_type,
                "reward_valid": True,
                "purchase_success": success,
                "termination_reason": reward_type,
            },
        },
        "blocked_tool_calls": [{"reason": "guard"}] * guards,
        "shopper_questions": [{"question": "q"}] * questions,
    }


def test_compare_condition_reports_paired_flips_and_behavior_deltas():
    baseline = [
        trajectory(1, success=False, guards=0, questions=1, reward=0.0),
        trajectory(2, success=True, guards=1, questions=1, reward=1.0),
    ]
    candidate = [
        trajectory(1, success=True, guards=2, questions=0, reward=1.0),
        trajectory(2, success=False, guards=0, questions=2, reward=0.0),
    ]

    result = compare_condition(
        baseline, candidate, [1, 2], "gap-ask-enabled"
    )

    assert result["strict_gains"] == [1]
    assert result["strict_losses"] == [2]
    assert result["strict_net"] == 0
    assert result["guard_delta"] == 1
    assert result["question_delta"] == 0
    assert result["mean_reward_delta"] == 0.0
    assert result["strict_flip_records"] == [
        {
            "task_id": 1,
            "direction": "gain",
            "baseline": {
                "strict_success": False,
                "reward_type": "wrong_purchase",
                "reward_valid": True,
                "done": True,
                "termination_reason": "wrong_purchase",
                "final_reward": 0.0,
                "question_count": 1,
                "guard_count": 0,
            },
            "candidate": {
                "strict_success": True,
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "done": True,
                "termination_reason": "gold_purchase",
                "final_reward": 1.0,
                "question_count": 0,
                "guard_count": 2,
            },
            "question_delta": -1,
            "guard_delta": 2,
            "reward_delta": 1.0,
        },
        {
            "task_id": 2,
            "direction": "loss",
            "baseline": {
                "strict_success": True,
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "done": True,
                "termination_reason": "gold_purchase",
                "final_reward": 1.0,
                "question_count": 1,
                "guard_count": 1,
            },
            "candidate": {
                "strict_success": False,
                "reward_type": "wrong_purchase",
                "reward_valid": True,
                "done": True,
                "termination_reason": "wrong_purchase",
                "final_reward": 0.0,
                "question_count": 2,
                "guard_count": 0,
            },
            "question_delta": 1,
            "guard_delta": -1,
            "reward_delta": -1.0,
        },
    ]


def test_compare_condition_rejects_reward_v3():
    baseline = [trajectory(1)]
    candidate = [trajectory(1)]
    candidate[0]["terminal_result"]["reward_detail"]["reward_version"] = (
        "shopsimulator-reward-v3"
    )

    with pytest.raises(ValueError, match="Reward v4"):
        compare_condition(baseline, candidate, [1], "gap-ask-enabled")


def test_parse_validation_curve_supports_console_and_dict_formats():
    log = "\n".join(
        [
            "step:10 - val-shopping/summary/strict_success_rate:0.61 "
            "- val-shopping/summary/purchase_success_rate:0.63",
            "(TaskRunner) {'training/global_step': 50, "
            "'val-shopping/summary/strict_success_rate': np.float64(0.65), "
            "'val-shopping/summary/mean_reward': 0.55}",
        ]
    )

    curve = parse_validation_curve(log)

    assert curve[10]["strict_success_rate"] == 0.61
    assert curve[10]["purchase_success_rate"] == 0.63
    assert curve[50]["strict_success_rate"] == 0.65
    assert curve[50]["mean_reward"] == 0.55
