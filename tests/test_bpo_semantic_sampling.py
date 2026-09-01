from shopping_grpo.training.grpo.dynamic_sampling import (
    select_reward_varying_groups,
)


def _select(*, hashes, valid=None):
    return select_reward_varying_groups(
        ["local"] * 4,
        [1.25, 1.25, 0.0, 0.0],
        group_types=["local"] * 4,
        purchase_success=[True, True, False, False],
        action_metadata_valid=[True] * 4,
        semantic_action_hashes=hashes,
        semantic_action_valid=valid if valid is not None else [True] * 4,
    )


def test_local_return_contrast_requires_two_semantic_actions():
    indices, stats = _select(hashes=["same"] * 4)

    assert indices == []
    assert stats["kept_group_count"] == 0
    assert stats["insufficient_semantic_action_diversity_group_count"] == 1
    assert stats["groups"][0]["drop_reason"] == (
        "insufficient_semantic_action_diversity"
    )


def test_local_two_semantic_actions_are_trainable():
    indices, stats = _select(hashes=["product-a", "product-a", "product-b", "product-b"])

    assert indices == [0, 1, 2, 3]
    assert stats["kept_group_count"] == 1
    assert stats["groups"][0]["unique_semantic_action_count"] == 2
    assert stats["groups"][0]["semantic_action_sample_counts"] == {
        "product-a": 2,
        "product-b": 2,
    }


def test_invalid_local_semantic_action_is_dropped_without_exception():
    indices, stats = _select(
        hashes=["product-a", "", "product-b", "product-b"],
        valid=[True, False, True, True],
    )

    assert indices == []
    assert stats["semantic_action_invalid_group_count"] == 1
    assert stats["groups"][0]["drop_reason"] == "semantic_action_invalid"


def test_empty_local_semantic_hash_is_dropped_without_exception():
    indices, stats = _select(
        hashes=["product-a", "", "product-b", "product-b"],
        valid=[True] * 4,
    )

    assert indices == []
    assert stats["semantic_action_invalid_group_count"] == 1
    assert stats["groups"][0]["drop_reason"] == "semantic_action_invalid"


def test_root_group_does_not_require_local_semantic_metadata():
    indices, stats = select_reward_varying_groups(
        ["root"] * 4,
        [1.25, 1.25, 0.0, 0.0],
        group_types=["root"] * 4,
        purchase_success=[True, True, False, False],
        action_metadata_valid=[True] * 4,
        semantic_action_hashes=[""] * 4,
        semantic_action_valid=[False] * 4,
    )

    assert indices == [0, 1, 2, 3]
    assert stats["kept_group_count"] == 1
