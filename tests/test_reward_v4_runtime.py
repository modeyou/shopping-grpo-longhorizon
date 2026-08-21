from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    reward_breakdown,
    validate_reward,
)


def _gate():
    return {
        "status": "pass",
        "passed": True,
        "verifiable": True,
        "comparator": "test",
        "source_field": "test",
    }


def test_grpo_runtime_accepts_and_minimizes_reward_v4():
    detail = validate_reward(
        {
            "reward_version": "shopsimulator-reward-v4",
            "reward_type": "gold_purchase",
            "reward_valid": True,
            "termination_reason": "gold_purchase",
            "target_asin_match": True,
            "hard_gates": {
                "category:0": _gate(),
                "core_function:0": _gate(),
            },
            "weighted_score": 1.0,
            "evidence_coverage": 1.0,
            "constraint_scores": {
                "category:0": 1.0,
                "core_function:0": 1.0,
                "option:0": 1.0,
                "price:0": 1.0,
            },
            "terminal_utility": 1.0,
            "purchase_success": True,
            "sampling_invalid": False,
        }
    )

    assert detail["reward_version"] == "shopsimulator-reward-v4"
    assert detail["constraint_scores"]["price:0"] == 1.0

    state = make_runtime_state(task_id=1, max_steps=35)
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": 1.0,
            "reward_version": detail["reward_version"],
            "reward_type": detail["reward_type"],
            "reward_valid": detail["reward_valid"],
            "reward_detail": detail,
        }
    )
    breakdown = reward_breakdown(state)

    assert breakdown["total"] == 1.0
    assert breakdown["r_type"] == 1.0
    assert breakdown["r_option"] == 1.0
    assert breakdown["core_function_score"] == 1.0
