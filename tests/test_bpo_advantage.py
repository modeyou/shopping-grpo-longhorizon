import pytest
import torch
from omegaconf import OmegaConf

from scripts.check_bpo_runtime import validate_bpo_runtime_hooks
from shopping_grpo.training.bpo.advantage import (
    audit_bpo_rollout_batch,
    compute_bpo_advantage,
)


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
    assert torch.isclose(advantages[:, 2].sum(), torch.tensor(0.0))
    assert torch.allclose(advantages[:, 0], expected * 0.5)
    assert torch.all(advantages[:, 1] == 0)
    assert torch.equal(advantages, returns)


def test_bpo_outcome_reward_is_not_dropped_by_actor_response_mask():
    rewards = torch.zeros((4, 4), dtype=torch.float32)
    rewards[:, 3] = torch.tensor([1.0, 0.0, -1.0, 2.0])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)
    metadata = {
        "bpo_group_id": ["g"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 2]] * 4,
    }

    advantages, _ = compute_bpo_advantage(
        rewards,
        mask,
        metadata=metadata,
        sibling_count=4,
        upstream_lambda=0.5,
    )

    expected = torch.tensor([2 / 3, -2 / 3, -2.0, 2.0])
    assert torch.allclose(advantages[:, 2], expected)
    assert torch.all(advantages[:, 1] == 0)
    assert torch.all(advantages[:, 3] == 0)


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


def test_bpo_rejects_reward_flat_sibling_group():
    metadata = {
        "bpo_group_id": ["g"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 1]] * 4,
    }
    with pytest.raises(ValueError, match="no reward variation"):
        compute_bpo_advantage(
            torch.zeros(4, 3),
            torch.ones(4, 3),
            metadata=metadata,
            sibling_count=4,
        )


def test_bpo_rejects_non_finite_sibling_reward():
    rewards = torch.zeros(4, 3)
    rewards[2, 2] = torch.nan
    metadata = {
        "bpo_group_id": ["g"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 1]] * 4,
    }
    with pytest.raises(ValueError, match="non-finite rewards"):
        compute_bpo_advantage(
            rewards,
            torch.ones_like(rewards),
            metadata=metadata,
            sibling_count=4,
        )


def test_bpo_rollout_batch_audit_rejects_prefix_drift():
    responses = torch.tensor(
        [[10, 11, 20 + row, 0] for row in range(4)], dtype=torch.long
    )
    mask = torch.tensor([[1, 1, 1, 0]] * 4)
    metadata = {
        "bpo_group_id": ["g"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [1] * 4,
        "bpo_action_token_starts": [[0, 2]] * 4,
        "bpo_branch_entropy": [2.0] * 4,
        "bpo_return_budget": [4] * 4,
        "bpo_env_idx": [0, 0, 1, 2],
        "bpo_branch_prefix_sha256": ["same"] * 4,
    }
    audits = audit_bpo_rollout_batch(
        torch.tensor([[1, 2]] * 4),
        responses,
        mask,
        metadata=metadata,
        sibling_count=4,
    )
    assert audits[0]["prefix_tokens"] == 2
    responses[2, 1] = 99
    with pytest.raises(ValueError, match="token prefixes"):
        audit_bpo_rollout_batch(
            torch.tensor([[1, 2]] * 4),
            responses,
            mask,
            metadata=metadata,
            sibling_count=4,
        )


def test_real_verl_dispatcher_accepts_bpo_on_cpu():
    config = OmegaConf.create(
        {
            "critic": {"enable": False},
            "algorithm": {
                "use_kl_in_reward": False,
                "bpo": {
                    "sibling_count": 4,
                    "upstream_lambda": 0.95,
                }
            },
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
        }
    )
    validate_bpo_runtime_hooks(config, validate_official_config=False)
