import pytest
import torch

from shopping_grpo.training.bpo.advantage import (
    build_bpo_policy_weights,
    summarize_bpo_actor_batch,
)


def test_root_and_local_weights_encode_equal_sequence_loss_mass():
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 4
        + [[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]] * 4
    )
    metadata = {
        "bpo_group_id": ["root"] * 4 + ["local"] * 4,
        "bpo_group_type": ["root"] * 4 + ["local"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3] * 2,
        "bpo_branch_action": [-1] * 4 + [1] * 4,
        "bpo_action_token_starts": [[0]] * 4 + [[0, 2, 4]] * 4,
        "bpo_action_token_ends": [[6]] * 4 + [[2, 4, 6]] * 4,
        "bpo_action_metadata_valid": [True] * 8,
    }

    weights = build_bpo_policy_weights(
        mask,
        metadata=metadata,
        sibling_count=4,
        dtype=torch.float32,
    )
    support = weights.ne(0)

    assert torch.all(weights[mask == 0] == 0)
    assert torch.all(support[:4] == mask[:4].bool())
    assert torch.all(support[4:, :2] == 0)
    assert torch.all(support[4:, 2:4] == mask[4:, 2:4].bool())
    assert torch.all(support[4:, 4:] == 0)

    row_means = (weights * support).sum(-1) / support.sum(-1)
    assert torch.isclose(row_means[:4].mean(), torch.tensor(1.0))
    assert torch.isclose(row_means[4:].mean(), torch.tensor(1.0))


def test_weights_reproduce_root_local_action_mean_under_verl_sequence_mean():
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 4
        + [[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]] * 4
    )
    metadata = {
        "bpo_group_id": ["root"] * 4 + ["local"] * 4,
        "bpo_group_type": ["root"] * 4 + ["local"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3] * 2,
        "bpo_branch_action": [-1] * 4 + [1] * 4,
        "bpo_action_token_starts": [[0]] * 4 + [[0, 2, 4]] * 4,
        "bpo_action_token_ends": [[6]] * 4 + [[2, 4, 6]] * 4,
        "bpo_action_metadata_valid": [True] * 8,
    }
    scalar_advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    weights = build_bpo_policy_weights(
        mask,
        metadata=metadata,
        sibling_count=4,
        dtype=torch.float32,
    )
    support = weights.ne(0)
    weighted_advantages = scalar_advantages[:, None] * weights
    native_sequence_token_mean = (
        weighted_advantages.sum(dim=-1) / support.sum(dim=-1)
    ).mean()
    expected = 0.5 * scalar_advantages[:4].mean() + 0.5 * scalar_advantages[4:].mean()

    assert torch.isclose(native_sequence_token_mean, expected)


def test_root_actions_receive_equal_mass_despite_different_token_lengths():
    mask = torch.ones((4, 6))
    metadata = {
        "bpo_group_id": ["root"] * 4,
        "bpo_group_type": ["root"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [-1] * 4,
        "bpo_action_token_starts": [[0, 1]] * 4,
        "bpo_action_token_ends": [[1, 6]] * 4,
        "bpo_action_metadata_valid": [True] * 4,
    }

    weights = build_bpo_policy_weights(mask, metadata=metadata, sibling_count=4)

    first_action_mass = weights[:, :1].sum(dim=-1)
    second_action_mass = weights[:, 1:].sum(dim=-1)
    assert torch.allclose(first_action_mass, second_action_mass)


def test_bpo_actor_diagnostics_reject_policy_weight_outside_actor_mask():
    shape = (1, 2)
    mask = torch.tensor([[1.0, 0.0]])
    policy_weights = torch.ones(shape)
    zeros = torch.zeros(shape)

    with pytest.raises(ValueError, match="outside the actor response mask"):
        summarize_bpo_actor_batch(zeros, mask, zeros, zeros, policy_weights)


def test_bpo_weights_reject_group_without_actor_token_support():
    mask = torch.tensor([[1.0, 0.0]] * 4)
    metadata = {
        "bpo_group_id": ["local"] * 4,
        "bpo_group_type": ["local"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 1]] * 4,
        "bpo_action_token_ends": [[1, 2]] * 4,
        "bpo_action_metadata_valid": [True] * 4,
    }

    with pytest.raises(ValueError, match="no actor-token policy support"):
        build_bpo_policy_weights(
            mask,
            metadata=metadata,
            sibling_count=4,
            dtype=torch.float32,
        )
