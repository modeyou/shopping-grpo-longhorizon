import gzip
import hashlib
import json
from pathlib import Path

import pytest

from shopping_grpo.multiturn.splits import (
    SPLIT_SCHEMA_VERSION,
    build_task_splits,
    count_shop_goals,
    decompressed_sha256,
    freeze_task_splits,
    validate_disjoint_splits,
)


def _write_jsonl(path, task_ids):
    path.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )


def _product_fixture(path):
    products = [
        {
            "asin": "one",
            "instructions": [
                {"attributes": ["a"]},
                {"attributes": []},
            ],
        },
        {
            "asin": "one",
            "instructions": [{"attributes": ["duplicate"]}],
        },
        {
            "asin": "two",
            "instructions": [
                {"attributes": ["b"]},
                {"attributes": ["c"]},
            ],
        },
    ]
    raw = json.dumps(products, ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _environment_manifest(path, product_hash):
    manifest = {
        "manifest_version": "shopping-environment-manifest-v1",
        "environment_version": "shopsimulator-environment-v2.1",
        "shopsimulator_commit": "a" * 40,
        "product_data_sha256": product_hash,
        "search": {
            "version": "shopsimulator-multifield-bm25-v2",
            "page_size": 20,
        },
        "reward": {"version": "shopsimulator-reward-v3"},
        "observation_version": "shopping-observation-v2",
        "tool_version": "shopping-tools-v2",
        "max_steps": 35,
        "seed": 1,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_build_task_splits_is_deterministic_disjoint_and_excluded():
    sizes = {
        "evaluation": 3,
        "sft_candidates": 5,
        "grpo_validation": 2,
        "grpo_train": 7,
    }
    first = build_task_splits(
        goal_count=25,
        excluded_task_ids={1, 4, 9},
        split_sizes=sizes,
        seed=17,
    )
    second = build_task_splits(
        goal_count=25,
        excluded_task_ids={1, 4, 9},
        split_sizes=sizes,
        seed=17,
    )
    assert first == second
    assert {name: len(first[name]) for name in sizes} == sizes
    assert len(first["reserve"]) == 5
    validate_disjoint_splits(first, {1, 4, 9})


def test_build_task_splits_rejects_out_of_range_exclusions():
    with pytest.raises(ValueError, match="outside the goal space"):
        build_task_splits(
            goal_count=10,
            excluded_task_ids={10},
            split_sizes={
                "evaluation": 1,
                "sft_candidates": 1,
                "grpo_validation": 1,
                "grpo_train": 1,
            },
            seed=1,
        )


def test_count_and_hash_shop_goals_from_compressed_product_data(tmp_path):
    products = tmp_path / "products.json.gz"
    expected_hash = _product_fixture(products)
    assert decompressed_sha256(products) == expected_hash
    assert count_shop_goals(products) == 3


def test_freeze_task_splits_is_idempotent_and_refuses_changes(tmp_path):
    products = tmp_path / "products.json.gz"
    product_hash = _product_fixture(products)
    manifest = tmp_path / "environment.json"
    _environment_manifest(manifest, product_hash)
    excluded = tmp_path / "excluded.jsonl"
    _write_jsonl(excluded, [0])
    output = tmp_path / "tasks"
    sizes = {
        "evaluation": 1,
        "sft_candidates": 1,
        "grpo_validation": 1,
        "grpo_train": 1,
    }

    # The fixture has only three goals, so use a second fixture with five goals.
    with gzip.open(products, "wt", encoding="utf-8") as handle:
        json.dump([
            {"asin": str(index), "instructions": [{"attributes": ["x"]}]}
            for index in range(6)
        ], handle)
    product_hash = decompressed_sha256(products)
    _environment_manifest(manifest, product_hash)

    first = freeze_task_splits(
        product_data_path=products,
        environment_manifest_path=manifest,
        exclusion_paths=[excluded],
        output_dir=output,
        split_sizes=sizes,
        seed=3,
    )
    second = freeze_task_splits(
        product_data_path=products,
        environment_manifest_path=manifest,
        exclusion_paths=[excluded],
        output_dir=output,
        split_sizes=sizes,
        seed=3,
    )
    assert first == second
    assert first["schema_version"] == SPLIT_SCHEMA_VERSION
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8")) == first

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_task_splits(
            product_data_path=products,
            environment_manifest_path=manifest,
            exclusion_paths=[excluded],
            output_dir=output,
            split_sizes=sizes,
            seed=4,
        )
