import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "data/multiturn/evaluation-v1"


def _ids(path):
    return [
        int(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_curated_multiturn_evaluation_asset_is_frozen_and_disjoint():
    tasks_path = ASSET / "tasks.jsonl"
    audit_path = ASSET / "reward_audit.jsonl"
    metadata_path = ASSET / "metadata.json"
    tasks_bytes = tasks_path.read_bytes()
    audit_bytes = audit_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_ids = _ids(tasks_path)

    assert metadata["schema_version"] == "shopping-multiturn-evaluation-v1"
    assert len(task_ids) == 500
    assert len(set(task_ids)) == 500
    assert metadata["task_count"] == 500
    assert metadata["task_sha256"] == hashlib.sha256(tasks_bytes).hexdigest()
    assert metadata["audit_sha256"] == hashlib.sha256(audit_bytes).hexdigest()
    assert metadata["selection"]["source_rejected"] == 194
    assert metadata["selection"]["replacement_tasks"] == 194

    training_ids = set()
    for name in ("sft_candidates", "grpo_validation", "grpo_train"):
        training_ids.update(_ids(ROOT / f"data/multiturn/tasks/{name}.jsonl"))
    assert not set(task_ids) & training_ids

    audits = {
        int(row["task_id"]): row
        for row in (
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    assert all(audits[task_id]["eligible"] for task_id in task_ids)


def test_development_and_formal_multiturn_evaluations_are_disjoint():
    formal = set(_ids(ROOT / "data/multiturn/evaluation-v1/tasks.jsonl"))
    development_path = ROOT / "data/multiturn/evaluation-dev-v1/tasks.jsonl"
    development = _ids(development_path)
    metadata = json.loads(
        (development_path.parent / "metadata.json").read_text(encoding="utf-8")
    )

    assert len(development) == len(set(development)) == 500
    assert not formal & set(development)
    assert metadata["selection"]["source_rejected"] == 190
    assert metadata["validation"][
        "selected_tasks_disjoint_from_exclusions"
    ] is True
