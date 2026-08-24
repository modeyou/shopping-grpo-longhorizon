"""Sibling-local leave-one-out advantages with upstream BPO propagation."""

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
    }
    missing = extra_required - set(metadata)
    if missing:
        raise ValueError(f"BPO rollout audit metadata is missing: {sorted(missing)}")

    audits = []
    for group_id, rows in groups.items():
        ordered = sorted(rows, key=lambda row: int(metadata["bpo_sibling_index"][row]))
        branch_actions = {int(metadata["bpo_branch_action"][row]) for row in ordered}
        entropies = {float(metadata["bpo_branch_entropy"][row]) for row in ordered}
        prefix_hashes = {
            str(metadata["bpo_branch_prefix_sha256"][row]) for row in ordered
        }
        budgets = {int(metadata["bpo_return_budget"][row]) for row in ordered}
        if len(branch_actions) != 1 or len(entropies) != 1 or len(prefix_hashes) != 1:
            raise ValueError(f"BPO group {group_id!r} does not share one branch state")
        if budgets != {int(sibling_count)}:
            raise ValueError(f"BPO group {group_id!r} has an invalid return budget")

        env_indices = [int(metadata["bpo_env_idx"][row]) for row in ordered]
        if min(env_indices) < 0 or len(set(env_indices[1:])) != sibling_count - 1:
            raise ValueError(f"BPO group {group_id!r} clone leases are not isolated")

        branch_action = next(iter(branch_actions))
        prompt_rows = [
            tuple(int(value) for value in prompts[row].reshape(-1)) for row in ordered
        ]
        if len(set(prompt_rows)) != 1:
            raise ValueError(f"BPO group {group_id!r} prompts are not identical")
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
                "branch_action": branch_action,
                "branch_entropy": next(iter(entropies)),
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


def compute_bpo_advantage(
    token_level_rewards,
    response_mask,
    *,
    metadata,
    sibling_count=4,
    upstream_lambda=0.95,
):
    """Compute local LOO sibling advantages and propagate them to shared actions."""
    import torch

    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("reward and response mask shapes must match")
    if sibling_count < 2:
        raise ValueError("BPO sibling_count must be at least two")
    if not 0.0 <= float(upstream_lambda) <= 1.0:
        raise ValueError("BPO upstream_lambda must be in [0, 1]")
    batch_size, response_length = token_level_rewards.shape
    groups = _validate_metadata(metadata, batch_size, sibling_count)
    scores = (token_level_rewards * response_mask).sum(dim=-1)
    scalar_advantages = torch.zeros_like(scores)
    with torch.no_grad():
        for rows in groups.values():
            group_scores = scores[rows]
            total = group_scores.sum()
            for local_index, row in enumerate(rows):
                sibling_mean = (total - group_scores[local_index]) / (sibling_count - 1)
                scalar_advantages[row] = scores[row] - sibling_mean

        weights = torch.zeros_like(response_mask, dtype=token_level_rewards.dtype)
        for row in range(batch_size):
            branch_action = int(metadata["bpo_branch_action"][row])
            starts = [int(value) for value in metadata["bpo_action_token_starts"][row]]
            if not starts or branch_action < 0 or branch_action >= len(starts):
                raise ValueError("invalid BPO action boundary metadata")
            if starts != sorted(starts) or starts[0] < 0 or starts[-1] >= response_length:
                raise ValueError("BPO action token starts are not valid response offsets")
            for action_index, start in enumerate(starts):
                end = (
                    starts[action_index + 1]
                    if action_index + 1 < len(starts)
                    else response_length
                )
                distance = max(branch_action - action_index, 0)
                weights[row, start:end] = float(upstream_lambda) ** distance
        advantages = scalar_advantages.unsqueeze(-1) * weights * response_mask
    return advantages, advantages.clone()
