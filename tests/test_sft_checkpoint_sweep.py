import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_sft_checkpoint_sweep import (
    build_actor_command,
    build_merge_command,
    checkpoint_result,
    shopper_infrastructure_errors,
    validate_assets,
    validate_shopper_api,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_assets_binds_all_three_views_to_one_task_set(tmp_path):
    rows = [{"task_id": 3}, {"task_id": 7}]
    hashes = {}
    for name in ("tasks", "gap_openings", "complete_openings"):
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        hashes[name] = _sha256(path)
    manifest = {
        "schema_version": "shopping-sft-checkpoint-sweep-v1",
        "reward_contract": "shopsimulator-reward-v4",
        "final_evaluation_used": False,
        "subset_sha256": hashes,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    count, manifest_sha = validate_assets(tmp_path)

    assert count == 2
    assert manifest_sha == _sha256(tmp_path / "manifest.json")


def test_validate_assets_rejects_final_evaluation_marker(tmp_path):
    for name in ("tasks", "gap_openings", "complete_openings"):
        _write_jsonl(tmp_path / f"{name}.jsonl", [{"task_id": 3}])
    manifest = {
        "schema_version": "shopping-sft-checkpoint-sweep-v1",
        "reward_contract": "shopsimulator-reward-v4",
        "final_evaluation_used": True,
        "subset_sha256": {
            name: _sha256(tmp_path / f"{name}.jsonl")
            for name in ("tasks", "gap_openings", "complete_openings")
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must not use final evaluation"):
        validate_assets(tmp_path)


def test_commands_keep_merge_on_cpu_and_actor_single_gpu_external():
    merge = build_merge_command(
        python="python",
        base_model=Path("base"),
        adapter=Path("adapter"),
        output=Path("merged"),
    )
    actor = build_actor_command(
        vllm_bin=Path("vllm"),
        model=Path("merged"),
        model_name="candidate",
        host="127.0.0.1",
        port=18102,
        max_model_len=24576,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
    )

    assert merge[-1] == "--bf16"
    assert "--output" in merge
    assert actor[:3] == ["vllm", "serve", "merged"]
    assert actor[actor.index("--port") + 1] == "18102"
    assert actor[actor.index("--served-model-name") + 1] == "candidate"
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(actor)


def test_validate_shopper_api_makes_authenticated_chat_request():
    response = io.BytesIO(
        json.dumps(
            {"choices": [{"message": {"content": "OK"}}]}
        ).encode()
    )

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        validate_shopper_api(
            base_url="https://shopper.example.test/v1",
            api_key="secret",
            model="shopper-model",
        )

    request = urlopen.call_args.args[0]
    assert request.full_url == (
        "https://shopper.example.test/v1/chat/completions"
    )
    assert request.get_header("Authorization") == "Bearer secret"


def test_shopper_infrastructure_errors_detects_forbidden_only():
    rows = [
        {"task_id": 1, "error": {"message": "HTTP Error 403: Forbidden"}},
        {"task_id": 2, "error": {"message": "ContextBudgetError"}},
        {"task_id": 3, "error": None},
    ]

    assert [
        row["task_id"] for row in shopper_infrastructure_errors(rows)
    ] == [1]


def test_checkpoint_result_rejects_shopper_infrastructure_failures(
    tmp_path,
):
    for condition in (
        "gap-ask-enabled",
        "gap-ask-disabled",
        "complete-ask-enabled",
    ):
        root = tmp_path / condition
        root.mkdir()
        _write_jsonl(
            root / "trajectories.jsonl",
            [
                {
                    "task_id": 3,
                    "error": {"message": "HTTP Error 403: Forbidden"},
                }
            ],
        )
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "completed_tasks": 1,
                    "missing_tasks": [],
                    "protocol": {
                        "model": "candidate",
                        "reward_contract": "shopsimulator-reward-v4",
                    },
                    "clarification": {},
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Shopper infrastructure errors"):
        checkpoint_result(tmp_path, 1, "candidate")
