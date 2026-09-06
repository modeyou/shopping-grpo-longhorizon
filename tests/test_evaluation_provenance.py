import hashlib
import json

import pytest

from shopping_grpo.evaluation.artifacts import ArtifactError
from shopping_grpo.evaluation.blind_guard import guard_declared_final_tasks
from shopping_grpo.evaluation.manifest import canonical_json_sha256


def test_canonical_json_hash_is_order_independent():
    assert canonical_json_sha256({"b": [2, 1], "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": [2, 1]}
    )


def test_declared_final_requires_explicit_opt_in(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"task_id": 7}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "evaluation_role": "final",
                "task_count": 1,
                "selection": {"result_blind": True},
                "subset_sha256": {
                    "tasks": hashlib.sha256(tasks.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="allow-blind-final"):
        guard_declared_final_tasks(tasks, allowed=False)
    guard_declared_final_tasks(tasks, allowed=True)
