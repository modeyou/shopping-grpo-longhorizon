import math

import pytest

from shopping_grpo.training.bpo.branching import (
    BranchCandidate,
    first_decision_token,
    select_branch_candidate,
    shannon_entropy,
)
from shopping_grpo.training.bpo.runtime import sibling_group_starts


def test_exact_entropy_requires_complete_normalized_distribution():
    entropy = shannon_entropy([math.log(0.25), math.log(0.75)])
    assert entropy == pytest.approx(0.5623351446)
    with pytest.raises(ValueError, match="complete normalized vocabulary"):
        shannon_entropy([math.log(0.2), math.log(0.2)])


def test_first_decision_token_skips_protocol_fragments():
    assert first_decision_token(["<tool_call>", "<function=", "search_products"]) == 2
    assert first_decision_token(["Ċ", "▁<tool_call>", "Ġ=", "click"]) == 3
    with pytest.raises(ValueError, match="no semantic decision token"):
        first_decision_token(["<", "/", ">", "_"])


def test_branch_selection_uses_entropy_then_earliest_boundary():
    late = BranchCandidate(3, 2, 4.0, "late", None)
    early = BranchCandidate(1, 5, 4.0, "early", None)
    lower = BranchCandidate(0, 0, 3.9, "lower", None)
    assert select_branch_candidate([late, lower, early]) is early


def test_worker_layout_preserves_each_contiguous_sibling_group():
    assert sibling_group_starts([10, 10, 10, 10, 20, 20, 20, 20], 4) == [0, 4]
    with pytest.raises(ValueError, match="contiguous repeats"):
        sibling_group_starts([10, 20, 10, 20], 4)
