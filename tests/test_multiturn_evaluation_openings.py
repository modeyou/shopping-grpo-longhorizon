import json
from pathlib import Path

import pytest

from shopping_grpo.multiturn.evaluation_openings import (
    CONDITION_SCHEMA,
    freeze_evaluation_openings,
)
from shopping_grpo.multiturn.tasks import build_task_row


class Env:
    def __init__(self, **kwargs):
        assert kwargs["multiturn"] is True
        self.shopper_context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def reset(self, task_id):
        self.shopper_context = {
            "instruction_full": f"task {task_id} complete request",
            "goal_options": [f"option {task_id}"],
        }
        return {"instruction": ""}


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gap(task_id):
    context = {
        "instruction_full": f"task {task_id} complete request",
        "goal_options": [f"option {task_id}"],
    }
    return build_task_row(
        task_id,
        f"task {task_id} gap request",
        context,
        "teacher",
        "a" * 64,
        omitted_dimensions=["budget"],
        omitted_facts=[f"task {task_id} complete request"],
    )


def test_freezes_complete_openings_and_three_condition_mappings(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    gaps = tmp_path / "gaps.jsonl"
    output = tmp_path / "frozen"
    _write(tasks, [{"task_id": 2}, {"task_id": 5}])
    _write(gaps, [_gap(2), _gap(5)])

    metadata = freeze_evaluation_openings(
        task_manifest=tasks,
        gap_openings=gaps,
        output_dir=output,
        env_factory=Env,
    )

    complete = [
        json.loads(line)
        for line in (output / "complete_openings.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    conditions = [
        json.loads(line)
        for line in (output / "conditions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["task_id"] for row in complete] == [2, 5]
    assert complete[0]["initial_request"] == "task 2 complete request"
    assert len(conditions) == 6
    assert all(row["schema_version"] == CONDITION_SCHEMA for row in conditions)
    task_two = [row for row in conditions if row["task_id"] == 2]
    assert [row["opening_kind"] for row in task_two] == [
        "gap", "gap", "complete"
    ]
    assert metadata["validation"]["conditions_per_task"] == 3


def test_rejects_incomplete_gap_openings(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    gaps = tmp_path / "gaps.jsonl"
    _write(tasks, [{"task_id": 2}, {"task_id": 5}])
    _write(gaps, [_gap(2)])

    with pytest.raises(ValueError, match="incomplete"):
        freeze_evaluation_openings(
            task_manifest=tasks,
            gap_openings=gaps,
            output_dir=tmp_path / "frozen",
            env_factory=Env,
        )
