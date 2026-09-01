from types import SimpleNamespace

import pytest

from scripts.check_bpo_runtime import validate_bpo_tree_contracts
from shopping_grpo.training.bpo.branching import validate_tree_outputs


def test_root_tree_allows_independent_action_counts_and_uses_lease_sentinel():
    outputs = []
    for sibling, action_count in enumerate((2, 3, 5, 4)):
        outputs.append(
            SimpleNamespace(
                prompt_ids=[1, 2],
                response_ids=[10 + sibling],
                response_mask=[1],
                extra_fields={
                    "bpo_group_id": "root-tree",
                    "bpo_group_type": "root",
                    "bpo_sibling_index": sibling,
                    "bpo_branch_action": -1,
                    "bpo_branch_entropy": 0.0,
                    "bpo_return_budget": 4,
                    "bpo_env_idx": -1,
                    "bpo_branch_prefix_sha256": "same-prompt",
                    "bpo_backbone_action_count": action_count,
                    "bpo_branch_relative_position": -1.0,
                },
            )
        )

    audit = validate_tree_outputs(outputs)

    assert audit["group_type"] == "root"
    assert audit["backbone_action_counts"] == [2, 3, 5, 4]
    assert audit["env_indices"] == [-1, -1, -1, -1]


def test_local_tree_still_requires_one_backbone_action_count():
    outputs = []
    for sibling, env_idx, action_count in zip(
        range(4), (0, 1, 2, 3), (3, 3, 4, 3), strict=True
    ):
        outputs.append(
            SimpleNamespace(
                prompt_ids=[1, 2],
                response_ids=[10, 11, 20 + sibling],
                response_mask=[1, 1, 1],
                extra_fields={
                    "bpo_group_id": "local-tree",
                    "bpo_group_type": "local",
                    "bpo_sibling_index": sibling,
                    "bpo_branch_action": 1,
                    "bpo_branch_entropy": 2.5,
                    "bpo_action_token_starts": [0, 2],
                    "bpo_return_budget": 4,
                    "bpo_env_idx": env_idx,
                    "bpo_branch_prefix_sha256": "same",
                    "bpo_backbone_action_count": action_count,
                    "bpo_branch_relative_position": 0.5,
                },
            )
        )

    with pytest.raises(ValueError, match="Local siblings disagree"):
        validate_tree_outputs(outputs)


def test_tree_rejects_mixed_root_and_local_group_types():
    outputs = []
    for sibling in range(4):
        outputs.append(
            SimpleNamespace(
                prompt_ids=[1, 2],
                response_ids=[10 + sibling],
                response_mask=[1],
                extra_fields={
                    "bpo_group_id": "mixed-tree",
                    "bpo_group_type": "local" if sibling == 3 else "root",
                    "bpo_sibling_index": sibling,
                    "bpo_branch_action": -1,
                    "bpo_branch_entropy": 0.0,
                    "bpo_return_budget": 4,
                    "bpo_env_idx": -1,
                    "bpo_branch_prefix_sha256": "same-prompt",
                    "bpo_backbone_action_count": 2,
                    "bpo_branch_relative_position": -1.0,
                },
            )
        )

    with pytest.raises(ValueError, match="bpo_group_type"):
        validate_tree_outputs(outputs)


def test_preflight_exercises_root_and_local_tree_contracts(capsys):
    validate_bpo_tree_contracts()

    output = capsys.readouterr().out
    assert "Root/Local tree contracts preflight passed" in output
    assert '"root_action_counts": [1, 2, 2, 3]' in output
