import pytest
import torch

from shopping_grpo.training.bpo.advantage import compute_bpo_advantage


def test_sibling_loo_advantage_and_upstream_propagation():
    rewards = torch.tensor([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
    ])
    mask = torch.tensor([[1, 1, 1, 0]] * 4)
    metadata = {
        "bpo_group_id": ["g"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 2]] * 4,
    }
    advantages, returns = compute_bpo_advantage(
        rewards, mask, metadata=metadata, upstream_lambda=0.5
    )
    expected_scalar = torch.tensor([2 / 3, -2 / 3, -2.0, 2.0])
    assert torch.allclose(advantages[:, 0], expected_scalar * 0.5)
    assert torch.allclose(advantages[:, 2], expected_scalar)
    assert torch.all(advantages[:, 3] == 0)
    assert torch.equal(returns, advantages)


def test_bpo_rejects_incomplete_sibling_group():
    with pytest.raises(ValueError, match="must contain siblings"):
        compute_bpo_advantage(
            torch.zeros((3, 1)),
            torch.ones((3, 1)),
            metadata={
                "bpo_group_id": ["g"] * 3,
                "bpo_sibling_index": [0, 1, 2],
                "bpo_branch_action": [0] * 3,
                "bpo_action_token_starts": [[0]] * 3,
            },
        )
