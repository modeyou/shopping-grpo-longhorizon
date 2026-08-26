import hashlib

import pytest

from scripts.run_standalone_checkpoint_evaluation import (
    validate_exported_model,
    validate_source_checkpoint,
)
from scripts.audit_standalone_checkpoint_evaluation import summarize


def test_validate_exported_model_records_config_and_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"model_type": "qwen3_5"}\n', encoding="utf-8")
    weight = model / "model.safetensors"
    weight.write_bytes(b"weights")

    audit = validate_exported_model(model)

    assert audit == {
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "weight_files": [{"name": "model.safetensors", "bytes": 7}],
    }


def test_validate_exported_model_rejects_missing_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="weights are missing"):
        validate_exported_model(model)


def test_validate_source_checkpoint_binds_step_and_tracker(tmp_path):
    run = tmp_path / "run"
    source = run / "global_step_200"
    actor = source / "actor"
    actor.mkdir(parents=True)
    (actor / "world_size_4_rank_0.pt").write_bytes(b"checkpoint")
    (run / "latest_checkpointed_iteration.txt").write_text(
        "200\n", encoding="utf-8"
    )

    audit = validate_source_checkpoint(source)

    assert audit["path"] == str(source)
    assert audit["actor_path"] == str(actor)
    assert audit["step"] == 200
    assert audit["latest_checkpointed_iteration"] == 200


def test_validate_source_checkpoint_rejects_stale_tracker(tmp_path):
    run = tmp_path / "run"
    source = run / "global_step_200"
    actor = source / "actor"
    actor.mkdir(parents=True)
    (actor / "rank.pt").write_bytes(b"checkpoint")
    (run / "latest_checkpointed_iteration.txt").write_text(
        "150\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="tracker is behind"):
        validate_source_checkpoint(source)


def test_summarize_uses_all_three_panels():
    result = {
        "gap-ask-enabled": {
            "strict_successes": 350,
            "strict_success_rate": 0.7,
            "mean_final_reward": 0.6,
            "done_tasks": 490,
            "reward_valid_tasks": 495,
            "guard_rejections": 2,
            "complete_unnecessary_ask_tasks": 0,
        },
        "gap-ask-disabled": {
            "strict_successes": 250,
            "strict_success_rate": 0.5,
            "mean_final_reward": 0.5,
            "done_tasks": 480,
            "reward_valid_tasks": 485,
            "guard_rejections": 3,
            "complete_unnecessary_ask_tasks": 0,
        },
        "complete-ask-enabled": {
            "strict_successes": 375,
            "strict_success_rate": 0.75,
            "mean_final_reward": 0.7,
            "done_tasks": 500,
            "reward_valid_tasks": 500,
            "guard_rejections": 1,
            "complete_unnecessary_ask_tasks": 450,
        },
    }

    values = summarize(result, 500)

    assert values["total"] == 0.65
    assert values["gap_gain"] == pytest.approx(0.2)
    assert values["unnecessary_ask"] == 0.9
    assert values["mean_reward"] == pytest.approx(0.6)
    assert values["done"] == 1470
    assert values["reward_valid"] == 1480
    assert values["guards"] == 6
