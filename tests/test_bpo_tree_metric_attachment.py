from types import SimpleNamespace

import pytest


def test_agent_loop_attaches_tree_metrics_to_all_siblings():
    pytest.importorskip("verl")
    from shopping_grpo.training.bpo.agent_loop import attach_bpo_tree_metrics

    outputs = []
    for sibling, env_idx, reward in zip(
        range(4), (0, 0, 1, 2), (0.0, 0.0, 1.0, 1.0), strict=True
    ):
        shopping = {
            "reward": {"total": reward},
            "actions": [{"tool": f"tool-{sibling}"}],
            "termination_reason": "done",
        }
        outputs.append(
            SimpleNamespace(
                reward_score=reward,
                extra_fields={
                    "shopping": shopping,
                    "bpo_group_id": "tree",
                    "bpo_sibling_index": sibling,
                    "bpo_branch_action": 1,
                    "bpo_branch_entropy": 2.0,
                    "bpo_action_token_starts": [0, 2],
                    "bpo_return_budget": 4,
                    "bpo_env_idx": env_idx,
                    "bpo_branch_prefix_sha256": "same-prefix",
                    "bpo_branch_action_sha256": f"action-{sibling}",
                    "bpo_backbone_action_count": 3,
                    "bpo_branch_relative_position": 0.5,
                    "bpo_branch_prefix_steps": 1,
                    "bpo_branch_prefix_shopper_calls": 0,
                    "bpo_branch_prefix_environment_transitions": 1,
                },
            )
        )

    metrics = attach_bpo_tree_metrics(outputs)

    assert metrics["bpo_branch/relative_position_mean"] == 0.5
    assert metrics["bpo_diversity/unique_branch_actions_mean"] == 4.0
    assert metrics["bpo_return/sibling_std_mean"] == 0.5
    assert all(
        output.extra_fields["shopping"]["bpo_metrics"] == metrics
        for output in outputs
    )
