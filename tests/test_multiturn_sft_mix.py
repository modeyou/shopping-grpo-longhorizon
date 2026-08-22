import json

import pytest

from shopping_grpo.collection.multiturn_sft_mix import (
    POLICY_ORDER,
    allocate_row_quotas,
    membership_patterns,
    select_disjoint_rows,
    split_selected,
)
from shopping_grpo.evaluation.artifacts import (
    ArtifactError,
    guard_task_id_disjoint,
    load_unique_task_ids,
)


def _item(task_id, policy, schema_variant=None, assistant_tokens=100):
    return {
        "row": {
            "task_id": task_id,
            "trajectory_id": f"trajectory-{policy}-{task_id}",
            "teacher_policy": policy,
            "source_goal_hash": f"goal-{task_id}",
            "schema_variant": schema_variant or "multiturn-shop-tools-v1",
        },
        "input_tokens": assistant_tokens * 4,
        "assistant_tokens": assistant_tokens,
    }


def test_token_targets_are_converted_to_row_quotas():
    quotas = allocate_row_quotas(
        total_rows=64,
        token_ratios={
            "complete-no-ask-v1": 0.5,
            "composite-replay-v1": 0.3,
            "autonomous-gap-v1": 0.2,
        },
        average_assistant_tokens={
            "complete-no-ask-v1": 100,
            "composite-replay-v1": 200,
            "autonomous-gap-v1": 400,
        },
    )

    assert sum(quotas.values()) == 64
    expected_tokens = {
        policy: quotas[policy] * average
        for policy, average in {
            "complete-no-ask-v1": 100,
            "composite-replay-v1": 200,
            "autonomous-gap-v1": 400,
        }.items()
    }
    total = sum(expected_tokens.values())
    assert abs(expected_tokens["complete-no-ask-v1"] / total - 0.5) < 0.03
    assert abs(expected_tokens["composite-replay-v1"] / total - 0.3) < 0.03
    assert abs(expected_tokens["autonomous-gap-v1"] / total - 0.2) < 0.03


def test_selection_is_disjoint_across_overlapping_policy_pools():
    complete = "complete-no-ask-v1"
    composite = "composite-replay-v1"
    autonomous = "autonomous-gap-v1"
    pools = {
        complete: [
            _item(
                task_id,
                complete,
                schema_variant=(
                    "complete-ask-available-v1"
                    if task_id % 2 == 0
                    else "complete-shop-tools-v1"
                ),
                assistant_tokens=80 + task_id,
            )
            for task_id in range(1, 101)
        ],
        composite: [
            _item(task_id, composite, assistant_tokens=120 + task_id)
            for task_id in range(50, 151)
        ],
        autonomous: [
            _item(task_id, autonomous, assistant_tokens=160 + task_id)
            for task_id in range(100, 181)
        ],
    }
    quotas = {complete: 6, composite: 4, autonomous: 2}

    selected = select_disjoint_rows(pools=pools, quotas=quotas, seed=42)
    selected_ids = [
        item["row"]["task_id"]
        for policy in POLICY_ORDER
        for item in selected[policy]
    ]
    selected_hashes = [
        item["row"]["source_goal_hash"]
        for policy in POLICY_ORDER
        for item in selected[policy]
    ]

    assert {policy: len(rows) for policy, rows in selected.items()} == quotas
    assert len(selected_ids) == len(set(selected_ids))
    assert len(selected_hashes) == len(set(selected_hashes))
    complete_variants = {
        item["row"]["schema_variant"] for item in selected[complete]
    }
    assert complete_variants == {
        "complete-ask-available-v1",
        "complete-shop-tools-v1",
    }

    train, validation = split_selected(
        selected, validation_ratio=0.2, seed=42
    )
    assert len(train) + len(validation) == sum(quotas.values())
    assert {item["row"]["task_id"] for item in train}.isdisjoint(
        item["row"]["task_id"] for item in validation
    )


def test_membership_patterns_count_unique_tasks():
    patterns = membership_patterns(
        {
            "complete-no-ask-v1": [{"task_id": 1}, {"task_id": 2}],
            "composite-replay-v1": [{"task_id": 2}, {"task_id": 3}],
            "autonomous-gap-v1": [{"task_id": 2}],
        }
    )

    assert patterns == {
        "complete-no-ask-v1": 1,
        "complete-no-ask-v1+composite-replay-v1+autonomous-gap-v1": 1,
        "composite-replay-v1": 1,
    }


def _write_task_ids(path, task_ids):
    path.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )


def test_final_evaluation_task_guard_accepts_disjoint_inputs(tmp_path):
    evaluation = tmp_path / "evaluation.jsonl"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_task_ids(evaluation, [10, 20])
    _write_task_ids(train, [1, 2])
    _write_task_ids(validation, [3])

    assert load_unique_task_ids(evaluation) == {10, 20}
    audit = guard_task_id_disjoint([train, validation], evaluation)
    assert audit["overlap_count"] == 0
    assert audit["candidate_rows"] == 3


def test_final_evaluation_task_guard_rejects_overlap(tmp_path):
    evaluation = tmp_path / "evaluation.jsonl"
    train = tmp_path / "train.jsonl"
    _write_task_ids(evaluation, [10, 20])
    _write_task_ids(train, [1, 20])

    with pytest.raises(ArtifactError, match="overlaps final evaluation"):
        guard_task_id_disjoint([train], evaluation)