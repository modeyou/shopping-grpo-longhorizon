from __future__ import annotations

import pytest

from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_bpo_tree_metrics,
    aggregate_shopping_metrics,
    swanlab_key_metrics,
)


def _reward(total):
    return {
        "full": float(total > 0),
        "strict": float(total > 0),
        "native": total,
        "semantic": total,
        "total": total,
        "efficiency": 0.0,
        "penalty_overlong": 0.0,
        "penalty_unfinished": 0.0,
        "penalty_repeat": 0.0,
        "repeat_action_rate": 0.0,
        "r_type": float(total > 0),
        "r_att": float(total > 0),
        "r_option": float(total > 0),
        "r_price": float(total > 0),
        "terminal_utility": total,
        "purchase_success": bool(total > 0),
        "sampling_invalid": False,
    }


def _shopping(total, *, bpo_metrics=None):
    row = {
        "reward": _reward(total),
        "steps": 3,
        "done": True,
        "termination_reason": "done",
        "infrastructure_invalid": False,
        "reward_unverifiable": False,
        "reward_valid": True,
        "shopper_questions": 1,
        "shopper_rejections": 0,
    }
    if bpo_metrics is not None:
        row["bpo_metrics"] = dict(bpo_metrics)
    return row


def test_compact_swanlab_summary_selects_the_decision_metrics():
    metrics = aggregate_shopping_metrics([_shopping(1.0), _shopping(0.0)])

    assert metrics["summary/strict_success_rate"] == 0.5
    assert metrics["summary/purchase_success_rate"] == 0.5
    assert metrics["summary/mean_reward"] == 0.5
    assert metrics["summary/done_rate"] == 1.0
    assert len(swanlab_key_metrics(metrics)) == 11


def test_bpo_tree_metrics_capture_branch_diversity_and_return_spread():
    summaries = {
        "tree": {
            "bpo_branch_relative_position": 0.5,
            "bpo_branch_entropy": 2.0,
            "bpo_backbone_action_count": 3,
            "bpo_branch_prefix_steps": 1,
            "bpo_branch_prefix_shopper_calls": 1,
            "bpo_branch_prefix_environment_transitions": 0,
            "bpo_unique_branch_action_count": 4,
            "bpo_unique_tool_sequence_count": 3,
        }
    }

    metrics = aggregate_bpo_tree_metrics(
        summaries,
        [{"uid": "tree", "rewards": [0.0, 0.0, 1.0, 1.0]}],
    )

    assert metrics["bpo_branch/relative_position_mean"] == 0.5
    assert metrics["bpo_diversity/unique_branch_actions_mean"] == 4.0
    assert metrics["bpo_return/sibling_std_mean"] == 0.5
    assert metrics["bpo_return/sibling_range_mean"] == 1.0
    assert metrics["bpo_return/sibling_unique_count_mean"] == 2.0


def test_shopping_aggregation_logs_attached_bpo_metrics():
    tree_metrics = {
        "bpo_branch/relative_position_mean": 0.25,
        "bpo_return/sibling_std_mean": 0.4,
    }

    metrics = aggregate_shopping_metrics(
        [_shopping(1.0, bpo_metrics=tree_metrics) for _ in range(4)]
    )

    assert metrics["bpo_branch/relative_position_mean"] == 0.25
    assert metrics["bpo_return/sibling_std_mean"] == pytest.approx(0.4)


def test_shopping_aggregation_rejects_partially_missing_bpo_metrics():
    with pytest.raises(ValueError, match="missing from some trajectories"):
        aggregate_shopping_metrics(
            [_shopping(1.0, bpo_metrics={"bpo/x": 1.0}), _shopping(0.0)]
        )
