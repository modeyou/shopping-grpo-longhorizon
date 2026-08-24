import pytest
import torch

from shopping_grpo.training.bpo.advantage import compute_bpo_advantage


def test_bpo_loo_advantage_and_upstream_action_weighting():
    rewards = torch.tensor(
        [[0.0, 0.0, score, 0.0] for score in (1.0, 0.0, -1.0, 2.0)]
    )
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]] * 4)
    metadata = {
        "bpo_group_id": ["g"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 2]] * 4,
    }
    advantages, returns = compute_bpo_advantage(
        rewards,
        mask,
        metadata=metadata,
        sibling_count=4,
        upstream_lambda=0.5,
    )
    expected = torch.tensor([2 / 3, -2 / 3, -2.0, 2.0])
    assert torch.allclose(advantages[:, 2], expected)
    assert torch.allclose(advantages[:, 0], expected * 0.5)
    assert torch.all(advantages[:, 1] == 0)
    assert torch.equal(advantages, returns)


def test_bpo_rejects_incomplete_sibling_group():
    metadata = {
        "bpo_group_id": ["g"] * 3,
        "bpo_sibling_index": [0, 1, 2],
        "bpo_branch_action": [0] * 3,
        "bpo_action_token_starts": [[0]] * 3,
    }
    with pytest.raises(ValueError, match="siblings"):
        compute_bpo_advantage(
            torch.zeros(3, 2),
            torch.ones(3, 2),
            metadata=metadata,
            sibling_count=4,
        )
