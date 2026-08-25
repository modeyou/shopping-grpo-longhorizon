"""Pure helpers for selecting one maximum-entropy BPO action boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BranchCandidate:
    action_index: int
    token_offset: int
    entropy: float
    snapshot_id: str
    payload: object = None


def full_vocabulary_entropy(logprobs):
    """Return entropy after validating complete normalized log-probabilities."""
    values = [float(value) for value in logprobs]
    if not values or any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError(
            "full-vocabulary logprobs must be non-empty and exclude NaN/+Inf"
        )
    # A masked or impossible token is represented by log(p) == -Inf.  Its
    # Shannon contribution is defined by continuity as 0 * log(0) == 0;
    # multiplying the IEEE values directly would instead manufacture NaN.
    probabilities = [
        0.0 if value == -math.inf else math.exp(value) for value in values
    ]
    probability_mass = sum(probabilities)
    if not 0.999 <= probability_mass <= 1.001:
        raise ValueError("full-vocabulary probabilities must sum to one")
    return -sum(
        probability * logprob
        for probability, logprob in zip(probabilities, values, strict=True)
        if probability > 0.0
    )


def first_decision_token(tokens):
    """Skip fixed chat/XML protocol pieces and return first semantic offset."""
    protocol = {
        "", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
        "<function>", "</function>", "<think>", "</think>", "assistant",
        "analysis", "final", "commentary", "{", "[", "(", ":", chr(34),
    }
    for index, token in enumerate(tokens):
        normalized = str(token).replace("▁", "").replace("Ġ", "").strip()
        if normalized and normalized not in protocol and not normalized.startswith("<|"):
            return index
    raise ValueError("assistant action contains no semantic decision token")


def select_branch_candidate(candidates):
    """Choose maximum entropy, breaking exact ties toward earliest action."""
    values = list(candidates)
    if not values:
        raise ValueError("at least one BPO branch candidate is required")
    if any(not math.isfinite(float(item.entropy)) for item in values):
        raise ValueError("BPO branch entropy must be finite")
    return min(values, key=lambda item: (-float(item.entropy), int(item.action_index)))


def retain_branch_candidates(candidates, *, limit=2):
    """Keep the best candidates needed to exclude a later terminal action.

    BPO branches before an action and the paper excludes the trajectory's final
    action from the candidate set.  Until the backbone terminates, retaining the
    two best entropy candidates is sufficient: if the best candidate becomes
    the final action, the second-best candidate is the correct choice.
    """
    values = list(candidates)
    if int(limit) < 1:
        raise ValueError("BPO retained candidate limit must be positive")
    if any(not math.isfinite(float(item.entropy)) for item in values):
        raise ValueError("BPO branch entropy must be finite")
    return sorted(
        values,
        key=lambda item: (-float(item.entropy), int(item.action_index)),
    )[: int(limit)]


def select_nonterminal_branch_candidate(candidates, *, action_count):
    """Select the maximum-entropy boundary while excluding the final action."""
    count = int(action_count)
    if count < 2:
        raise ValueError("BPO backbone needs at least two actions to branch")
    eligible = [
        item for item in candidates if int(item.action_index) < count - 1
    ]
    if not eligible:
        raise ValueError("BPO backbone has no non-terminal branch boundary")
    return select_branch_candidate(eligible)


def validate_tree_outputs(outputs, *, sibling_count=4):
    """Fail before PPO if restored siblings do not share one exact prefix/state."""
    values = list(outputs)
    if len(values) != int(sibling_count):
        raise ValueError("BPO tree must contain exactly K sibling outputs")
    metadata = [value.extra_fields for value in values]
    expected_indices = list(range(int(sibling_count)))
    actual_indices = [int(item.get("bpo_sibling_index", -1)) for item in metadata]
    if actual_indices != expected_indices:
        raise ValueError("BPO tree siblings must be ordered 0..K-1")

    invariant_keys = (
        "bpo_group_id",
        "bpo_branch_action",
        "bpo_branch_entropy",
        "bpo_return_budget",
        "bpo_branch_prefix_sha256",
        "bpo_backbone_action_count",
        "bpo_branch_relative_position",
    )
    for key in invariant_keys:
        if len({str(item.get(key)) for item in metadata}) != 1:
            raise ValueError(f"BPO siblings disagree on {key}")
    if int(metadata[0]["bpo_return_budget"]) != int(sibling_count):
        raise ValueError("BPO return budget must equal sibling count for M=1")

    env_indices = [int(item.get("bpo_env_idx", -1)) for item in metadata]
    clone_env_indices = env_indices[1:]
    if min(env_indices) < 0 or len(set(clone_env_indices)) != int(sibling_count) - 1:
        raise ValueError("BPO clone siblings must use K-1 distinct leases")

    branch_action = int(metadata[0]["bpo_branch_action"])
    backbone_action_count = int(metadata[0]["bpo_backbone_action_count"])
    if backbone_action_count < 2 or branch_action >= backbone_action_count - 1:
        raise ValueError("BPO branch boundary must precede the final action")
    prompts = [tuple(output.prompt_ids) for output in values]
    if len(set(prompts)) != 1:
        raise ValueError("BPO siblings do not share one original prompt")
    shared_prefixes = []
    shared_masks = []
    for output, item in zip(values, metadata, strict=True):
        starts = [int(value) for value in item.get("bpo_action_token_starts", [])]
        if branch_action < 0 or branch_action >= len(starts):
            raise ValueError("BPO branch action is outside action boundaries")
        branch_start = starts[branch_action]
        shared_prefixes.append(tuple(output.response_ids[:branch_start]))
        shared_masks.append(tuple(output.response_mask[:branch_start]))
    if len(set(shared_prefixes)) != 1 or len(set(shared_masks)) != 1:
        raise ValueError("BPO sibling token prefixes differ before the branch boundary")
    return {
        "group_id": str(metadata[0]["bpo_group_id"]),
        "branch_action": branch_action,
        "branch_entropy": float(metadata[0]["bpo_branch_entropy"]),
        "prefix_tokens": len(shared_prefixes[0]),
        "env_indices": env_indices,
    }
