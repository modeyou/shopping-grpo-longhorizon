import math

import pytest

from shopping_grpo.training.bpo.branching import (
    BranchCandidate,
    first_decision_token,
    full_vocabulary_entropy,
    select_branch_candidate,
)
from shopping_grpo.training.bpo.runtime import sibling_group_starts


def test_full_vocabulary_entropy_requires_normalized_distribution():
    assert full_vocabulary_entropy([math.log(0.25)] * 4) == pytest.approx(math.log(4))
    assert full_vocabulary_entropy([math.log(0.5), math.log(0.5), -math.inf]) == (
        pytest.approx(math.log(2))
    )
    with pytest.raises(ValueError, match="sum to one"):
        full_vocabulary_entropy([math.log(0.2), math.log(0.2)])
    with pytest.raises(ValueError, match="NaN/\\+Inf"):
        full_vocabulary_entropy([0.0, math.nan])
    with pytest.raises(ValueError, match="NaN/\\+Inf"):
        full_vocabulary_entropy([0.0, math.inf])


def test_first_decision_token_skips_protocol_tokens():
    assert first_decision_token(["<|im_start|>", "assistant", " ", "search_products"]) == 3
    with pytest.raises(ValueError, match="no semantic"):
        first_decision_token(["<|im_start|>", "assistant", " "])


def test_maximum_entropy_uses_earliest_action_as_tie_break():
    later = BranchCandidate(2, 0, 1.2, "later")
    earlier = BranchCandidate(1, 0, 1.2, "earlier")
    lower = BranchCandidate(0, 0, 0.8, "lower")
    assert select_branch_candidate([later, lower, earlier]) is earlier


def test_sibling_groups_must_be_contiguous_repeats():
    assert sibling_group_starts([7, 7, 7, 7, 9, 9, 9, 9], 4) == [0, 4]
    with pytest.raises(ValueError, match="contiguous"):
        sibling_group_starts([7, 7, 9, 7], 4)
