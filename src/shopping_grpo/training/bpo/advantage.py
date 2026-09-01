"""CARL-BPO v3 Root/Local leave-one-out credit and action-balanced weights."""

from __future__ import annotations

from collections import defaultdict


def _row_action_spans(response_mask, starts, ends, *, group_id):
    """Validate exact action boundaries and return actor-token spans."""
    response_length = int(response_mask.shape[-1])
    starts = [int(value) for value in starts]
    ends = [int(value) for value in ends]
    if not starts or len(starts) != len(ends):
        raise ValueError(f"BPO group {group_id!r} has incomplete action boundaries")
    if starts != sorted(set(starts)) or starts[0] < 0:
        raise ValueError(f"BPO group {group_id!r} has invalid action starts")
    if ends[:-1] != starts[1:] or ends[-1] > response_length:
        raise ValueError(f"BPO group {group_id!r} has inconsistent action ends")
    spans = []
    for start, end in zip(starts, ends, strict=True):
        if start >= end:
            raise ValueError(f"BPO group {group_id!r} has an empty action span")
        actor_tokens = response_mask[start:end].ne(0)
        if not bool(actor_tokens.any().item()):
            raise ValueError(
                f"BPO group {group_id!r} action has no actor-token policy support"
            )
        spans.append((start, end, actor_tokens))
    return spans


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
        "bpo_action_token_ends",
        "bpo_action_metadata_valid",
        "bpo_branch_semantic_action_sha256",
        "bpo_branch_semantic_valid",
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
        if (
            len(branch_actions) != 1
            or len(entropies) != 1
            or len(prefix_hashes) != 1
        ):
            raise ValueError(f"BPO group {group_id!r} does not share one branch state")
        if budgets != {int(sibling_count)}:
            raise ValueError(f"BPO group {group_id!r} has an invalid return budget")

        if not all(bool(metadata["bpo_action_metadata_valid"][row]) for row in ordered):
            raise ValueError(f"BPO group {group_id!r} has invalid action metadata")
        env_indices = [int(metadata["bpo_env_idx"][row]) for row in ordered]
        if group_type == "root":
            if set(env_indices) != {-1}:
                raise ValueError(
                    f"BPO Root group {group_id!r} must use the non-applicable lease sentinel"
                )
        elif min(env_indices) < 0 or len(set(env_indices)) != sibling_count:
            raise ValueError(f"BPO group {group_id!r} clone leases are not isolated")

        prompt_rows = [
            tuple(int(value) for value in prompts[row].reshape(-1)) for row in ordered
        ]
        if len(set(prompt_rows)) != 1:
            raise ValueError(f"BPO group {group_id!r} prompts are not identical")
        if group_type == "root":
            if branch_actions != {-1} or entropies != {0.0}:
                raise ValueError(f"BPO Root group {group_id!r} has branch metadata")
            action_counts = []
            for row in ordered:
                spans = _row_action_spans(
                    response_mask[row],
                    metadata["bpo_action_token_starts"][row],
                    metadata["bpo_action_token_ends"][row],
                    group_id=group_id,
                )
                action_count = int(metadata["bpo_backbone_action_count"][row])
                if action_count != len(spans):
                    raise ValueError(
                        f"BPO Root group {group_id!r} action count is inconsistent"
                    )
                action_counts.append(action_count)
            audits.append(
                {
                    "group_id": group_id,
                    "group_type": group_type,
                    "branch_action": -1,
                    "branch_entropy": 0.0,
                    "backbone_action_counts": action_counts,
                    "action_count": sum(action_counts),
                    "branch_relative_position": -1.0,
                    "prefix_tokens": 0,
                    "env_indices": env_indices,
                }
            )
            continue

        branch_action = next(iter(branch_actions))
        backbone_action_counts = {
            int(metadata["bpo_backbone_action_count"][row]) for row in ordered
        }
        relative_positions = {
            float(metadata["bpo_branch_relative_position"][row]) for row in ordered
        }
        if len(backbone_action_counts) != 1 or len(relative_positions) != 1:
            raise ValueError(f"BPO group {group_id!r} does not share one branch state")
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
            ends = [int(value) for value in metadata["bpo_action_token_ends"][row]]
            _row_action_spans(
                response_mask[row], starts, ends, group_id=group_id
            )
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
        semantic_valid = [
            bool(metadata["bpo_branch_semantic_valid"][row]) for row in ordered
        ]
        semantic_hashes = [
            str(metadata["bpo_branch_semantic_action_sha256"][row])
            for row in ordered
        ]
        if (
            not all(semantic_valid)
            or not all(semantic_hashes)
            or len(set(semantic_hashes)) < 2
        ):
            raise ValueError(
                f"BPO Local group {group_id!r} lacks semantic action diversity"
            )
        audits.append(
            {
                "group_id": group_id,
                "group_type": group_type,
                "branch_action": branch_action,
                "branch_entropy": next(iter(entropies)),
                "backbone_action_count": backbone_action_count,
                "branch_relative_position": relative_position,
                "prefix_tokens": len(prefixes[0]),
                "unique_semantic_actions": len(set(semantic_hashes)),
                "action_count": int(sibling_count),
                "env_indices": env_indices,
            }
        )
    return audits


def _validate_metadata(metadata, batch_size, sibling_count):
    required = {
        "bpo_group_id",
        "bpo_group_type",
        "bpo_sibling_index",
        "bpo_branch_action",
        "bpo_action_token_starts",
        "bpo_action_token_ends",
        "bpo_action_metadata_valid",
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
    dtype=None,
):
    """Encode the exact v3 action mean in veRL's sequence-token mean loss.

    veRL averages each response over ``response_mask`` and then averages
    responses.  The positive weights below make that objective equal to an
    action mean inside each Root/Local group and give every present group type
    equal total mass.  The runtime replaces the actor loss mask with the
    non-zero support of these weights, so Local prefix/suffix tokens are absent
    from both numerator and denominator.
    """
    import torch

    if response_mask.ndim != 2:
        raise ValueError("BPO response mask must be a two-dimensional tensor")
    if sibling_count < 2:
        raise ValueError("BPO sibling_count must be at least two")

    batch_size, response_length = response_mask.shape
    _validate_metadata(metadata, batch_size, sibling_count)
    weight_dtype = dtype if dtype is not None else response_mask.dtype
    actor_mask = response_mask.ne(0)
    weights = torch.zeros(
        (batch_size, response_length),
        dtype=weight_dtype,
        device=response_mask.device,
    )
    group_rows = defaultdict(list)
    group_types = {}
    for row in range(batch_size):
        group_id = str(metadata["bpo_group_id"][row])
        group_type = str(metadata.get("bpo_group_type", ["local"] * batch_size)[row])
        if group_type not in {"root", "local"}:
            raise ValueError(f"unknown BPO group type: {group_type!r}")
        group_rows[group_id].append(row)
        previous = group_types.setdefault(group_id, group_type)
        if previous != group_type:
            raise ValueError(f"BPO group {group_id!r} has inconsistent group type")

    groups_by_type = defaultdict(list)
    for group_id, group_type in group_types.items():
        groups_by_type[group_type].append(group_id)
    type_mass = 1.0 / len(groups_by_type)

    for group_type, group_ids in groups_by_type.items():
        group_mass = type_mass / len(group_ids)
        for group_id in group_ids:
            rows = group_rows[group_id]
            units = []
            row_support_counts = {}
            for row in rows:
                if not bool(metadata.get("bpo_action_metadata_valid", [True] * batch_size)[row]):
                    raise ValueError(f"BPO group {group_id!r} has invalid action metadata")
                starts = metadata["bpo_action_token_starts"][row]
                row_ends = metadata["bpo_action_token_ends"][row]
                spans = _row_action_spans(
                    response_mask[row], starts, row_ends, group_id=group_id
                )
                selected = spans
                if group_type == "local":
                    branch_action = int(metadata["bpo_branch_action"][row])
                    if branch_action < 0 or branch_action >= len(spans):
                        raise ValueError("invalid BPO branch action metadata")
                    selected = [spans[branch_action]]
                row_support_counts[row] = sum(
                    int(actor_tokens.sum().item())
                    for _, _, actor_tokens in selected
                )
                for start, end, actor_tokens in selected:
                    units.append((row, start, end, actor_tokens))
            if not units:
                raise ValueError(
                    f"BPO group {group_id!r} has no actor-token policy support"
                )
            unit_coefficient = float(batch_size) * group_mass / len(units)
            for row, start, end, actor_tokens in units:
                action_token_count = int(actor_tokens.sum().item())
                coefficient = (
                    unit_coefficient
                    * row_support_counts[row]
                    / action_token_count
                )
                span_weights = weights[row, start:end]
                span_weights[actor_tokens] = coefficient
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
    if bool(((policy != 0) & (mask == 0)).any().item()):
        raise ValueError("BPO policy weights extend outside the actor response mask")
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


def summarize_bpo_token_mass(
    response_mask,
    *,
    metadata,
    sibling_count=4,
):
    """Return exact actor-token mass for every selected group/sibling/action."""
    batch_size = int(response_mask.shape[0])
    groups = _validate_metadata(metadata, batch_size, sibling_count)
    result = []
    for group_id, rows in groups.items():
        ordered = sorted(
            rows, key=lambda row: int(metadata["bpo_sibling_index"][row])
        )
        group_types = {str(metadata["bpo_group_type"][row]) for row in ordered}
        if len(group_types) != 1:
            raise ValueError(f"BPO group {group_id!r} has inconsistent group type")
        group_type = next(iter(group_types))
        siblings = []
        for row in ordered:
            spans = _row_action_spans(
                response_mask[row],
                metadata["bpo_action_token_starts"][row],
                metadata["bpo_action_token_ends"][row],
                group_id=group_id,
            )
            branch_action = int(metadata["bpo_branch_action"][row])
            actions = []
            for action_index, (start, end, actor_tokens) in enumerate(spans):
                selected = group_type == "root" or action_index == branch_action
                actions.append(
                    {
                        "action_index": action_index,
                        "start": start,
                        "end": end,
                        "span_tokens": end - start,
                        "actor_tokens": int(actor_tokens.sum().item()),
                        "selected_for_policy": selected,
                    }
                )
            siblings.append(
                {
                    "sibling_index": int(metadata["bpo_sibling_index"][row]),
                    "actions": actions,
                    "selected_actor_tokens": sum(
                        action["actor_tokens"]
                        for action in actions
                        if action["selected_for_policy"]
                    ),
                }
            )
        result.append(
            {
                "group_id": str(group_id),
                "group_type": group_type,
                "siblings": siblings,
                "selected_actor_tokens": sum(
                    sibling["selected_actor_tokens"] for sibling in siblings
                ),
            }
        )
    return {"groups": result}


def compute_bpo_advantage(
    token_level_rewards,
    response_mask,
    *,
    metadata,
    sibling_count=4,
    return_diagnostics=False,
):
    """Compute LOO values with Root-action and Local-branch-action support."""
    import torch

    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("reward and response mask shapes must match")
    if sibling_count < 2:
        raise ValueError("BPO sibling_count must be at least two")
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
            dtype=token_level_rewards.dtype,
        )
        advantages = scalar_advantages.unsqueeze(-1) * weights * response_mask
        policy_mask = weights.ne(0).to(response_mask.dtype)
    if return_diagnostics:
        return advantages, advantages.clone(), {
            "policy_weights": weights,
            "policy_mask": policy_mask,
            "scalar_advantages": scalar_advantages,
            "scores": scores,
        }
    return advantages, advantages.clone()
