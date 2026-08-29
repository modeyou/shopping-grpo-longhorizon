"""CARL-BPO Root episode and Local suffix-only leave-one-out advantages."""

from __future__ import annotations

from collections import defaultdict


def audit_bpo_rollout_batch(
    prompts,
    responses,
    response_mask,
    *,
    metadata,
    sibling_count=4,
):
    """Validate the reconstructed rollout tree before computing any gradient."""
    batch_size, response_length = responses.shape
    if prompts.shape[0] != batch_size:
        raise ValueError("BPO prompts and responses batch sizes must match")
    if tuple(response_mask.shape) != (batch_size, response_length):
        raise ValueError("BPO responses and response mask shapes must match")
    groups = _validate_metadata(metadata, batch_size, sibling_count)
    extra_required = {
        "bpo_branch_entropy",
        "bpo_return_budget",
        "bpo_env_idx",
        "bpo_branch_prefix_sha256",
        "bpo_backbone_action_count",
        "bpo_branch_relative_position",
    }
    missing = extra_required - set(metadata)
    if missing:
        raise ValueError(f"BPO rollout audit metadata is missing: {sorted(missing)}")

    audits = []
    for group_id, rows in groups.items():
        ordered = sorted(rows, key=lambda row: int(metadata["bpo_sibling_index"][row]))
        group_type = str(
            metadata.get("bpo_group_type", ["local"] * batch_size)[ordered[0]]
        )
        if group_type not in {"root", "local"}:
            raise ValueError(f"BPO group {group_id!r} has unknown group type")
        branch_actions = {int(metadata["bpo_branch_action"][row]) for row in ordered}
        entropies = {float(metadata["bpo_branch_entropy"][row]) for row in ordered}
        prefix_hashes = {
            str(metadata["bpo_branch_prefix_sha256"][row]) for row in ordered
        }
        budgets = {int(metadata["bpo_return_budget"][row]) for row in ordered}
        backbone_action_counts = {
            int(metadata["bpo_backbone_action_count"][row]) for row in ordered
        }
        relative_positions = {
            float(metadata["bpo_branch_relative_position"][row]) for row in ordered
        }
        if (
            len(branch_actions) != 1
            or len(entropies) != 1
            or len(prefix_hashes) != 1
            or len(backbone_action_counts) != 1
            or len(relative_positions) != 1
        ):
            raise ValueError(f"BPO group {group_id!r} does not share one branch state")
        if budgets != {int(sibling_count)}:
            raise ValueError(f"BPO group {group_id!r} has an invalid return budget")

        env_indices = [int(metadata["bpo_env_idx"][row]) for row in ordered]
        if min(env_indices) < 0:
            raise ValueError(f"BPO group {group_id!r} has invalid environment leases")
        if group_type == "local" and len(set(env_indices[1:])) != sibling_count - 1:
            raise ValueError(f"BPO group {group_id!r} clone leases are not isolated")

        prompt_rows = [
            tuple(int(value) for value in prompts[row].reshape(-1)) for row in ordered
        ]
        if len(set(prompt_rows)) != 1:
            raise ValueError(f"BPO group {group_id!r} prompts are not identical")
        if group_type == "root":
            audits.append(
                {
                    "group_id": group_id,
                    "group_type": group_type,
                    "branch_action": -1,
                    "branch_entropy": 0.0,
                    "backbone_action_count": 1,
                    "branch_relative_position": 0.0,
                    "prefix_tokens": 0,
                    "env_indices": env_indices,
                }
            )
            continue

        branch_action = next(iter(branch_actions))
        backbone_action_count = next(iter(backbone_action_counts))
        if backbone_action_count < 2 or branch_action >= backbone_action_count - 1:
            raise ValueError(
                f"BPO group {group_id!r} branch boundary must precede the final action"
            )
        expected_relative_position = branch_action / (backbone_action_count - 1)
        relative_position = next(iter(relative_positions))
        if abs(relative_position - expected_relative_position) > 1e-8:
            raise ValueError(
                f"BPO group {group_id!r} has inconsistent branch position metadata"
            )
        prefixes = []
        prefix_masks = []
        prefix_starts = []
        for row in ordered:
            starts = [int(value) for value in metadata["bpo_action_token_starts"][row]]
            if branch_action < 0 or branch_action >= len(starts):
                raise ValueError(f"BPO group {group_id!r} has an invalid branch action")
            if starts != sorted(starts) or starts[0] < 0 or starts[-1] >= response_length:
                raise ValueError(f"BPO group {group_id!r} has invalid action boundaries")
            branch_start = starts[branch_action]
            prefixes.append(tuple(int(value) for value in responses[row, :branch_start]))
            prefix_masks.append(
                tuple(int(value) for value in response_mask[row, :branch_start])
            )
            prefix_starts.append(tuple(starts[: branch_action + 1]))
        if len(set(prefixes)) != 1 or len(set(prefix_masks)) != 1:
            raise ValueError(f"BPO group {group_id!r} token prefixes are not identical")
        if len(set(prefix_starts)) != 1:
            raise ValueError(f"BPO group {group_id!r} action prefixes are not identical")
        audits.append(
            {
                "group_id": group_id,
                "group_type": group_type,
                "branch_action": branch_action,
                "branch_entropy": next(iter(entropies)),
                "backbone_action_count": backbone_action_count,
                "branch_relative_position": relative_position,
                "prefix_tokens": len(prefixes[0]),
                "env_indices": env_indices,
            }
        )
    return audits


def _validate_metadata(metadata, batch_size, sibling_count):
    required = {
        "bpo_group_id", "bpo_sibling_index", "bpo_branch_action",
        "bpo_action_token_starts",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"BPO rollout metadata is missing: {sorted(missing)}")
    groups = defaultdict(list)
    for row in range(batch_size):
        groups[str(metadata["bpo_group_id"][row])].append(row)
    for group_id, rows in groups.items():
        siblings = sorted(int(metadata["bpo_sibling_index"][row]) for row in rows)
        if len(rows) != sibling_count or siblings != list(range(sibling_count)):
            raise ValueError(
                f"BPO group {group_id!r} must contain siblings 0..{sibling_count - 1}"
            )
    return groups


def build_bpo_policy_weights(
    response_mask,
    *,
    metadata,
    sibling_count=4,
    upstream_lambda=0.0,
    dtype=None,
):
    """Build token weights before veRL's actor loss/remove-padding path."""
    import torch

    if response_mask.ndim != 2:
        raise ValueError("BPO response mask must be a two-dimensional tensor")
    if sibling_count < 2:
        raise ValueError("BPO sibling_count must be at least two")
    if not 0.0 <= float(upstream_lambda) <= 1.0:
        raise ValueError("BPO upstream_lambda must be in [0, 1]")

    batch_size, response_length = response_mask.shape
    _validate_metadata(metadata, batch_size, sibling_count)
    weight_dtype = dtype if dtype is not None else response_mask.dtype
    weights = torch.zeros(
        (batch_size, response_length),
        dtype=weight_dtype,
        device=response_mask.device,
    )
    group_rows = defaultdict(list)
    for row in range(batch_size):
        group_rows[str(metadata["bpo_group_id"][row])].append(row)
    for row in range(batch_size):
        group_type = str(metadata.get("bpo_group_type", ["local"] * batch_size)[row])
        if group_type == "root":
            weights[row] = response_mask[row]
            continue
        if group_type != "local":
            raise ValueError(f"unknown BPO group type: {group_type!r}")
        branch_action = int(metadata["bpo_branch_action"][row])
        starts = [int(value) for value in metadata["bpo_action_token_starts"][row]]
        if not starts or branch_action < 0 or branch_action >= len(starts):
            raise ValueError("invalid BPO action boundary metadata")
        if (
            starts != sorted(starts)
            or starts[0] < 0
            or starts[-1] >= response_length
        ):
            raise ValueError("BPO action boundaries are not valid response offsets")
        for action_index, start in enumerate(starts):
            end = (
                starts[action_index + 1]
                if action_index + 1 < len(starts)
                else response_length
            )
            if action_index >= branch_action:
                weights[row, start:end] = 1.0

    # The upstream actor loss is token-mean.  Normalize each Root/Local group
    # to equal mass without changing token-level PPO weighting within a group.
    active_counts = {
        group_id: int(sum(weights[row].ne(0).sum().item() for row in rows))
        for group_id, rows in group_rows.items()
    }
    total_active = sum(active_counts.values())
    group_count = len(active_counts)
    if total_active and group_count:
        for group_id, rows in group_rows.items():
            count = active_counts[group_id]
            if count:
                weights[rows] *= float(total_active) / float(group_count * count)
    return weights


def summarize_bpo_actor_batch(
    token_level_rewards,
    response_mask,
    advantages,
    returns,
    policy_weights,
):
    """Return JSON-safe actor-side support and advantage diagnostics."""
    import torch

    tensors = {
        "token_level_rewards": token_level_rewards,
        "response_mask": response_mask,
        "advantages": advantages,
        "returns": returns,
        "policy_weights": policy_weights,
    }
    shape = tuple(int(value) for value in response_mask.shape)
    if any(
        tuple(int(value) for value in tensor.shape) != shape
        for tensor in tensors.values()
    ):
        raise ValueError("BPO actor diagnostics tensors must have equal shapes")

    def finite(tensor):
        return bool(torch.isfinite(tensor.detach().float()).all().item())

    mask = response_mask.detach()
    policy = policy_weights.detach()
    policy_mask = (policy != 0) & (mask != 0)
    advantage_float = advantages.detach().float()
    reward_float = token_level_rewards.detach().float()
    return_float = returns.detach().float()
    row_counts = mask.ne(0).sum(dim=-1).detach().cpu().tolist()
    return {
        "batch_size": shape[0],
        "response_length": shape[1],
        "response_mask_total_tokens": int(mask.ne(0).sum().item()),
        "response_mask_nonzero_rows": int(
            (mask.ne(0).sum(dim=-1) > 0).sum().item()
        ),
        "response_mask_row_min": int(min(row_counts, default=0)),
        "response_mask_row_max": int(max(row_counts, default=0)),
        "response_mask_row_mean": (
            float(sum(row_counts) / len(row_counts)) if row_counts else 0.0
        ),
        "policy_weight_nonzero_tokens": int(policy.ne(0).sum().item()),
        "policy_mask_nonzero_tokens": int(policy_mask.sum().item()),
        "policy_mask_weight_sum": float((policy * mask).float().sum().item()),
        "token_reward_abs_sum": float(reward_float.abs().sum().item()),
        "advantages_nonzero_tokens": int(advantages.ne(0).sum().item()),
        "advantages_abs_sum": float(advantage_float.abs().sum().item()),
        "advantages_abs_max": (
            float(advantage_float.abs().max().item())
            if advantage_float.numel()
            else 0.0
        ),
        "returns_abs_sum": float(return_float.abs().sum().item()),
        "all_finite": all(finite(tensor) for tensor in tensors.values()),
    }


def compute_bpo_advantage(
    token_level_rewards,
    response_mask,
    *,
    metadata,
    sibling_count=4,
    upstream_lambda=0.0,
    return_diagnostics=False,
):
    """Compute LOO returns with Root-wide and Local suffix-only policy support."""
    import torch

    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("reward and response mask shapes must match")
    if sibling_count < 2:
        raise ValueError("BPO sibling_count must be at least two")
    if not 0.0 <= float(upstream_lambda) <= 1.0:
        raise ValueError("BPO upstream_lambda must be in [0, 1]")
    batch_size, response_length = token_level_rewards.shape
    groups = _validate_metadata(metadata, batch_size, sibling_count)
    # Outcome rewards can live on tool/environment tokens whose actor mask is
    # zero. The mask controls policy-gradient placement, not trajectory return.
    scores = token_level_rewards.sum(dim=-1)
    scalar_advantages = torch.zeros_like(scores)
    with torch.no_grad():
        for rows in groups.values():
            group_scores = scores[rows]
            if not torch.isfinite(group_scores).all():
                raise ValueError("BPO sibling group contains non-finite rewards")
            if torch.max(group_scores) - torch.min(group_scores) <= 1e-8:
                raise ValueError(
                    "BPO sibling group has no reward variation after reward alignment"
                )
            total = group_scores.sum()
            for local_index, row in enumerate(rows):
                sibling_mean = (total - group_scores[local_index]) / (sibling_count - 1)
                scalar_advantages[row] = scores[row] - sibling_mean

        weights = build_bpo_policy_weights(
            response_mask,
            metadata=metadata,
            sibling_count=sibling_count,
            upstream_lambda=upstream_lambda,
            dtype=token_level_rewards.dtype,
        )
        advantages = scalar_advantages.unsqueeze(-1) * weights * response_mask
    if return_diagnostics:
        return advantages, advantages.clone(), {
            "policy_weights": weights,
            "scalar_advantages": scalar_advantages,
            "scores": scores,
        }
    return advantages, advantages.clone()
