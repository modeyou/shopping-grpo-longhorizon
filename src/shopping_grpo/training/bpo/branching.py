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
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("full-vocabulary logprobs must be finite and non-empty")
    probabilities = [math.exp(value) for value in values]
    probability_mass = sum(probabilities)
    if not 0.999 <= probability_mass <= 1.001:
        raise ValueError("full-vocabulary probabilities must sum to one")
    return -sum(
        probability * logprob
        for probability, logprob in zip(probabilities, values, strict=True)
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
