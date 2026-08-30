import math
from types import SimpleNamespace

import pytest

from shopping_grpo.training.bpo.branching import (
    BranchCandidate,
    first_decision_token,
    full_vocabulary_entropy,
    retain_branch_candidates,
    select_branch_candidate,
    select_nonterminal_branch_candidate,
    validate_tree_outputs,
)
from shopping_grpo.training.bpo.agent_loop import ShoppingBPOAgentLoop
from shopping_grpo.training.bpo.runtime import sibling_group_starts
from shopping_grpo.training.grpo.dynamic_sampling import build_carl_group_assignments


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


def test_retained_candidates_allow_final_action_to_be_excluded():
    early = BranchCandidate(0, 0, 0.8, "early")
    middle = BranchCandidate(1, 0, 1.2, "middle")
    final = BranchCandidate(2, 0, 2.0, "final")
    retained = retain_branch_candidates([early, middle, final], limit=2)
    assert retained == [final, middle]
    assert select_nonterminal_branch_candidate(retained, action_count=3) is middle
    with pytest.raises(ValueError, match="at least two actions"):
        select_nonterminal_branch_candidate([final], action_count=1)


def test_sibling_groups_must_be_contiguous_repeats():
    assert sibling_group_starts([7, 7, 7, 7, 9, 9, 9, 9], 4) == [0, 4]
    with pytest.raises(ValueError, match="contiguous"):
        sibling_group_starts([7, 7, 9, 7], 4)


def test_carl_group_roles_are_frozen_before_worker_sharding():
    assert build_carl_group_assignments(2, ["root", "local"]) == (
        "root",
        "local",
    )
    with pytest.raises(ValueError, match="exactly one Root and one Local"):
        build_carl_group_assignments(1, ["root", "local"])
    with pytest.raises(ValueError, match="group schedule"):
        build_carl_group_assignments(2, ["root", "root"])


@pytest.mark.parametrize(
    ("tool_name", "stage"),
    [
        ("search_products", "search_strategy"),
        ("next_page", "search_strategy"),
        ("select_option", "option"),
        ("open_product", "product"),
        ("buy_now", "excluded"),
    ],
)
def test_local_stage_uses_the_boundary_action_contract(tool_name, stage):
    assert ShoppingBPOAgentLoop._stage_for_tool_calls(
        [SimpleNamespace(name=tool_name)]
    ) == stage


def test_tree_outputs_share_exact_prefix_and_isolated_clone_leases():
    from types import SimpleNamespace

    outputs = []
    for sibling, env_idx, suffix in zip(
        range(4), (0, 0, 1, 2), ((20, 21), (30, 31), (40, 41), (50, 51)), strict=True
    ):
        outputs.append(
            SimpleNamespace(
                prompt_ids=[1, 2],
                response_ids=[10, 11, *suffix],
                response_mask=[1, 1, 1, 1],
                extra_fields={
                    "bpo_group_id": "tree",
                    "bpo_sibling_index": sibling,
                    "bpo_branch_action": 1,
                    "bpo_branch_entropy": 2.5,
                    "bpo_action_token_starts": [0, 2],
                    "bpo_return_budget": 4,
                    "bpo_env_idx": env_idx,
                    "bpo_branch_prefix_sha256": "same",
                    "bpo_backbone_action_count": 3,
                    "bpo_branch_relative_position": 0.5,
                },
            )
        )
    audit = validate_tree_outputs(outputs)
    assert audit["prefix_tokens"] == 2
    assert audit["env_indices"] == [0, 0, 1, 2]
    outputs[3].response_ids[1] = 99
    with pytest.raises(ValueError, match="token prefixes"):
        validate_tree_outputs(outputs)
