import pytest
import torch

from shopping_grpo.training.bpo.advantage import (
    build_bpo_policy_weights,
    summarize_bpo_actor_batch,
)


def test_root_and_local_weights_balance_effective_actor_tokens_only():
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 4
        + [[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]] * 4
    )
    metadata = {
        "bpo_group_id": ["root"] * 4 + ["local"] * 4,
        "bpo_group_type": ["root"] * 4 + ["local"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3] * 2,
        "bpo_branch_action": [0] * 4 + [1] * 4,
        "bpo_action_token_starts": [[0]] * 4 + [[0, 2, 4]] * 4,
    }

    weights = build_bpo_policy_weights(
        mask,
        metadata=metadata,
        sibling_count=4,
        dtype=torch.float32,
    )

    assert torch.all(weights[mask == 0] == 0)
    root_mass = weights[:4].sum()
    local_mass = weights[4:].sum()
    assert torch.isclose(root_mass, local_mass)
    assert torch.isclose(root_mass, torch.tensor(10.0))


def test_bpo_actor_diagnostics_reject_policy_weight_outside_actor_mask():
    shape = (1, 2)
    mask = torch.tensor([[1.0, 0.0]])
    policy_weights = torch.ones(shape)
    zeros = torch.zeros(shape)

    with pytest.raises(ValueError, match="outside the actor response mask"):
        summarize_bpo_actor_batch(
            zeros,
            mask,
            zeros,
            zeros,
            policy_weights,
        )


def test_bpo_weights_reject_group_without_actor_token_support():
    mask = torch.tensor([[1.0, 0.0]] * 4)
    metadata = {
        "bpo_group_id": ["local"] * 4,
        "bpo_group_type": ["local"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 1]] * 4,
    }

    with pytest.raises(ValueError, match="no actor-token policy support"):
        build_bpo_policy_weights(
            mask,
            metadata=metadata,
            sibling_count=4,
            dtype=torch.float32,
        )
