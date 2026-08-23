"""Decision-boundary scoring and deterministic BPO branch selection."""

from dataclasses import dataclass
import math

PROTOCOL_ONLY_PIECES = frozenset({
    "<", ">", "/", "_", "tool", "call", "function", "assistant",
    "<tool_call>", "</tool_call>", "<function=", "arguments", "argument",
    "parameter", "parameters", "name", "=", ":", "{", "}", '"', "'",
})

WHITESPACE_MARKERS = ("▁", "Ġ", "Ċ", "ĉ")


def shannon_entropy(log_probs):
    """Compute entropy from a complete vocabulary log-probability vector."""
    values = [float(value) for value in log_probs]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("entropy requires finite full-vocabulary log probabilities")
    probabilities = [math.exp(value) for value in values]
    mass = sum(probabilities)
    if not 0.999 <= mass <= 1.001:
        raise ValueError("entropy probe must contain the complete normalized vocabulary")
    return -sum(p * value for p, value in zip(probabilities, values))


def first_decision_token(token_pieces):
    for index, raw_piece in enumerate(token_pieces):
        piece = str(raw_piece)
        for marker in WHITESPACE_MARKERS:
            piece = piece.replace(marker, "")
        piece = piece.strip()
        if piece and piece.casefold() not in PROTOCOL_ONLY_PIECES:
            if not all(character in "<>/_=:-{}[]" for character in piece):
                return index
    raise ValueError("assistant action has no semantic decision token")


@dataclass(frozen=True)
class BranchCandidate:
    action_index: int
    token_offset: int
    entropy: float
    snapshot_id: str
    payload: object

    def __post_init__(self):
        if self.action_index < 0 or self.token_offset < 0:
            raise ValueError("branch indices must be non-negative")
        if not math.isfinite(self.entropy) or self.entropy < 0:
            raise ValueError("branch entropy must be finite and non-negative")
        if not self.snapshot_id:
            raise ValueError("branch snapshot_id is required")


def select_branch_candidate(candidates):
    """Select maximum entropy, breaking ties by the earliest boundary."""
    values = list(candidates)
    if not values:
        raise ValueError("BPO requires at least one valid decision boundary")
    return min(
        values,
        key=lambda item: (-item.entropy, item.action_index, item.token_offset),
    )
