import pytest
import torch

from scripts.check_bpo_runtime import validate_bpo_runtime_hooks
from shopping_grpo.training.bpo.advantage import (
    audit_bpo_rollout_batch,
    build_bpo_policy_weights,
    compute_bpo_advantage,
    summarize_bpo_actor_batch,
)


def _local_metadata(rows, response_length, starts):
    return {
        "bpo_group_id": ["g"] * rows,
        "bpo_group_type": ["local"] * rows,
        "bpo_sibling_index": list(range(rows)),
        "bpo_branch_action": [len(starts) - 1] * rows,
        "bpo_action_token_starts": [list(starts) for _ in range(rows)],
        "bpo_action_token_ends": [
            [*starts[1:], response_length] for _ in range(rows)
        ],
        "bpo_action_metadata_valid": [True] * rows,
    }


def _local_audit_metadata(*, backbone_actions=3, relative_position=0.5):
    metadata = _local_metadata(4, 4, [0, 2])
    metadata.update(
        {
            "bpo_branch_entropy": [2.0] * 4,
            "bpo_return_budget": [4] * 4,
            "bpo_env_idx": [0, 1, 2, 3],
            "bpo_branch_prefix_sha256": ["same"] * 4,
            "bpo_backbone_action_count": [backbone_actions] * 4,
            "bpo_branch_relative_position": [relative_position] * 4,
            "bpo_branch_semantic_action_sha256": ["a", "a", "b", "b"],
            "bpo_branch_semantic_valid": [True] * 4,
        }
    )
    return metadata


def test_carl_loo_advantage_updates_only_the_local_branch_action():
    rewards = torch.tensor(
        [[0.0, 0.0, score, 0.0] for score in (1.0, 0.0, -1.0, 2.0)]
    )
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]] * 4)
    metadata = _local_metadata(4, 4, [0, 2])

    advantages, returns = compute_bpo_advantage(
        rewards, mask, metadata=metadata, sibling_count=4
    )

    expected = torch.tensor([2 / 3, -2 / 3, -2.0, 2.0])
    assert torch.allclose(advantages[:, 2], expected)
    assert torch.allclose(advantages[:, 3], expected)
    assert torch.all(advantages[:, :2] == 0)
    assert torch.equal(advantages, returns)


def test_bpo_outcome_reward_is_not_dropped_by_actor_response_mask():
    rewards = torch.zeros((4, 4), dtype=torch.float32)
    rewards[:, 3] = torch.tensor([1.0, 0.0, -1.0, 2.0])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)

    advantages, _ = compute_bpo_advantage(
        rewards,
        mask,
        metadata=_local_metadata(4, 4, [0, 2]),
        sibling_count=4,
    )

    assert torch.allclose(
        advantages[:, 2], torch.tensor([2 / 3, -2 / 3, -2.0, 2.0])
    )
    assert torch.all(advantages[:, [0, 1, 3]] == 0)


def test_bpo_actor_diagnostics_expose_exact_policy_support():
    rewards = torch.zeros((4, 5), dtype=torch.float32)
    rewards[:, 4] = torch.tensor([1.0, 0.0, -1.0, 2.0])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]] * 4)
    metadata = _local_metadata(4, 5, [0, 2])

    advantages, returns, internals = compute_bpo_advantage(
        rewards,
        mask,
        metadata=metadata,
        sibling_count=4,
        return_diagnostics=True,
    )
    policy_weights = build_bpo_policy_weights(
        mask, metadata=metadata, sibling_count=4, dtype=rewards.dtype
    )
    diagnostics = summarize_bpo_actor_batch(
        rewards, mask, advantages, returns, policy_weights
    )

    assert torch.equal(policy_weights, internals["policy_weights"])
    assert diagnostics["response_mask_total_tokens"] == 12
    assert diagnostics["policy_mask_nonzero_tokens"] == 4
    assert diagnostics["advantages_nonzero_tokens"] == 4
    assert diagnostics["all_finite"] is True


def test_bpo_rejects_incomplete_sibling_group():
    with pytest.raises(ValueError, match="siblings"):
        compute_bpo_advantage(
            torch.zeros(3, 2),
            torch.ones(3, 2),
            metadata=_local_metadata(3, 2, [0]),
            sibling_count=4,
        )


def test_bpo_rejects_reward_flat_sibling_group():
    with pytest.raises(ValueError, match="no reward variation"):
        compute_bpo_advantage(
            torch.zeros(4, 3),
            torch.ones(4, 3),
            metadata=_local_metadata(4, 3, [0, 1]),
            sibling_count=4,
        )


def test_bpo_rejects_non_finite_sibling_reward():
    rewards = torch.zeros(4, 3)
    rewards[2, 2] = torch.nan
    with pytest.raises(ValueError, match="non-finite rewards"):
        compute_bpo_advantage(
            rewards,
            torch.ones_like(rewards),
            metadata=_local_metadata(4, 3, [0, 1]),
            sibling_count=4,
        )


def test_bpo_rollout_batch_audit_rejects_prefix_drift():
    responses = torch.tensor(
        [[10, 11, 20 + row, 0] for row in range(4)], dtype=torch.long
    )
    mask = torch.tensor([[1, 1, 1, 0]] * 4)
    metadata = _local_audit_metadata()

    audits = audit_bpo_rollout_batch(
        torch.tensor([[1, 2]] * 4),
        responses,
        mask,
        metadata=metadata,
        sibling_count=4,
    )
    assert audits[0]["prefix_tokens"] == 2
    assert audits[0]["action_count"] == 4

    responses[2, 1] = 99
    with pytest.raises(ValueError, match="token prefixes"):
        audit_bpo_rollout_batch(
            torch.tensor([[1, 2]] * 4),
            responses,
            mask,
            metadata=metadata,
            sibling_count=4,
        )


def test_bpo_rollout_batch_audit_rejects_final_action_branch():
    with pytest.raises(ValueError, match="precede the final action"):
        audit_bpo_rollout_batch(
            torch.tensor([[1, 2]] * 4),
            torch.tensor([[10, 11, 20, 0]] * 4),
            torch.tensor([[1, 1, 1, 0]] * 4),
            metadata=_local_audit_metadata(
                backbone_actions=2, relative_position=1.0
            ),
            sibling_count=4,
        )


def test_root_rollout_audit_and_policy_weights_allow_independent_action_counts():
    responses = torch.tensor([[10, 11, 12, 13, 14, 15]] * 4)
    mask = torch.ones((4, 6), dtype=torch.float32)
    starts = [[0, 3], [0, 2, 4], [0], [0, 1, 3, 5]]
    ends = [[3, 6], [2, 4, 6], [6], [1, 3, 5, 6]]
    metadata = {
        "bpo_group_id": ["root"] * 4,
        "bpo_group_type": ["root"] * 4,
        "bpo_sibling_index": [0, 1, 2, 3],
        "bpo_branch_action": [-1] * 4,
        "bpo_action_token_starts": starts,
        "bpo_action_token_ends": ends,
        "bpo_action_metadata_valid": [True] * 4,
        "bpo_branch_entropy": [0.0] * 4,
        "bpo_return_budget": [4] * 4,
        "bpo_env_idx": [-1] * 4,
        "bpo_branch_prefix_sha256": ["same-prompt"] * 4,
        "bpo_backbone_action_count": [2, 3, 1, 4],
        "bpo_branch_relative_position": [-1.0] * 4,
        "bpo_branch_semantic_action_sha256": [""] * 4,
        "bpo_branch_semantic_valid": [False] * 4,
    }

    audits = audit_bpo_rollout_batch(
        torch.tensor([[1, 2]] * 4),
        responses,
        mask,
        metadata=metadata,
        sibling_count=4,
    )
    weights = build_bpo_policy_weights(
        mask, metadata=metadata, sibling_count=4, dtype=torch.float32
    )

    assert audits[0]["backbone_action_counts"] == [2, 3, 1, 4]
    assert audits[0]["action_count"] == 10
    assert torch.all(weights > 0)
    for row, row_starts, row_ends in zip(range(4), starts, ends, strict=True):
        action_masses = [
            weights[row, start:end].sum()
            for start, end in zip(row_starts, row_ends, strict=True)
        ]
        assert torch.allclose(
            torch.stack(action_masses),
            torch.full((len(action_masses),), action_masses[0]),
        )


def test_real_verl_dispatcher_accepts_bpo_on_cpu():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    config = OmegaConf.create(
        {
            "critic": {"enable": False},
            "algorithm": {
                "use_kl_in_reward": False,
                "bpo": {"sibling_count": 4},
            },
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                    "optim": {
                        "lr": 1e-6,
                        "lr_warmup_steps_ratio": 0.0,
                        "total_training_steps": 500,
                        "weight_decay": 0.01,
                        "lr_warmup_steps": 10,
                        "betas": [0.9, 0.999],
                        "clip_grad": 1.0,
                        "optimizer": "AdamW",
                        "optimizer_impl": "torch.optim",
                        "min_lr_ratio": 0.1,
                        "lr_scheduler_type": "cosine",
                        "num_cycles": 0.5,
                        "zero_indexed_step": True,
                    },
                }
            },
        }
    )
    validate_bpo_runtime_hooks(config, validate_official_config=False)
