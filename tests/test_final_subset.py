import json
from pathlib import Path

import pytest

from scripts.freeze_multiturn_final_subset import build_subset, select_task_ids


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_hash_selection_is_deterministic_and_order_independent():
    first = select_task_ids([1, 2, 3, 4], size=2, seed="frozen")
    second = select_task_ids([4, 3, 2, 1], size=2, seed="frozen")
    assert first == second
    assert len(first) == 2


def test_build_subset_preserves_source_order_and_three_conditions(tmp_path):
    tasks = [{"task_id": task_id} for task_id in range(1, 6)]
    _write_jsonl(tmp_path / "tasks.jsonl", tasks)
    _write_jsonl(tmp_path / "gap_openings.jsonl", tasks)
    _write_jsonl(tmp_path / "complete_openings.jsonl", tasks)
    _write_jsonl(tmp_path / "reward_audit.jsonl", tasks)
    _write_jsonl(
        tmp_path / "conditions.jsonl",
        [
            {"task_id": task_id, "condition": condition}
            for task_id in range(1, 6)
            for condition in (
                "gap-ask-enabled",
                "gap-ask-disabled",
                "complete-ask-enabled",
            )
        ],
    )

    files, manifest = build_subset(tmp_path, size=3, seed="frozen")
    selected = [
        json.loads(line)["task_id"]
        for line in files["tasks.jsonl"].decode().splitlines()
    ]
    assert selected == sorted(selected)
    assert manifest["task_count"] == 3
    assert manifest["condition_count"] == 9
    assert manifest["final_evaluation_used"] is True
    assert manifest["selection"]["result_blind"] is True


def test_subset_rejects_missing_condition(tmp_path):
    rows = [{"task_id": 1}]
    for name in (
        "tasks.jsonl",
        "gap_openings.jsonl",
        "complete_openings.jsonl",
        "reward_audit.jsonl",
    ):
        _write_jsonl(tmp_path / name, rows)
    _write_jsonl(
        tmp_path / "conditions.jsonl",
        [{"task_id": 1, "condition": "gap-ask-enabled"}],
    )
    with pytest.raises(ValueError, match="three rows"):
        build_subset(tmp_path, size=1, seed="frozen")


def test_repository_final200_is_frozen_and_disjoint():
    root = Path(__file__).resolve().parents[1]
    final = root / "data/multiturn/final-200-v1"
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    final_ids = {
        int(json.loads(line)["task_id"])
        for line in (final / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    excluded_ids = set()
    for path in (
        root / "data/multiturn/evaluation-dev-v2/tasks.jsonl",
        root / "data/multiturn/tasks/grpo_train.jsonl",
        root / "data/multiturn/tasks/grpo_validation.jsonl",
        root / "data/multiturn/tasks/sft_candidates.jsonl",
    ):
        excluded_ids.update(
            int(json.loads(line)["task_id"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    assert len(final_ids) == manifest["task_count"] == 200
    assert manifest["selection"]["result_blind"] is True
    assert not final_ids & excluded_ids
