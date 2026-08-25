import json

import pytest

from scripts.audit_bpo_formal_run import audit


def _write_run(tmp_path, *, returns=400, branches=300):
    output = tmp_path / "run"
    output.mkdir(parents=True)
    contract = {
        "schema_version": "shopping-bpo-run-contract-v1",
        "launch": {"reward_profile": "none", "seed": 20260823, "logger": "swanlab"},
        "frozen_method": {
            "effective_tree_budget": 100,
            "effective_return_budget": 400,
            "maximum_optimizer_steps": 100,
            "scheduler": "cosine",
            "scheduler_horizon": 500,
            "warmup_steps": 10,
            "minimum_lr_ratio": 0.1,
        },
    }
    (output / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    metrics = {
        "training/optimizer_updated": 1,
        "bpo_budget/effective_trees_total": 100,
        "bpo_budget/effective_returns_total": returns,
        "bpo_budget/effective_return_target": 400,
        "rollout/generated_response_tokens_total": 1234,
        "bpo_cost/backbone_rollouts_total": 100,
        "bpo_cost/branch_rollouts_total": branches,
        "bpo_cost/environment_transitions_total": 500,
        "bpo_cost/shopper_api_calls_total": 20,
    }
    event = {"event": "optimizer_step", "global_step": 50, "metrics": metrics}
    (output / "training_diagnostics.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    (output / "global_step_25").mkdir()
    (output / "global_step_50").mkdir()
    log = tmp_path / "run.log"
    log.write_text(
        "BPO scheduler contract:\n"
        "BPO optimizer update audit:\n"
        "BPO tree audit passed:\n",
        encoding="utf-8",
    )
    return output, log


def test_formal_audit_accepts_closed_r400_contract(tmp_path):
    output, log = _write_run(tmp_path)
    result = audit(output, log)
    assert result["status"] == "accepted"
    assert result["effective_returns"] == 400


def test_formal_audit_rejects_budget_or_tree_cost_mismatch(tmp_path):
    output, log = _write_run(tmp_path, returns=396)
    with pytest.raises(ValueError, match="effective_returns_total"):
        audit(output, log)

    output, log = _write_run(tmp_path / "second", branches=299)
    with pytest.raises(ValueError, match="M=1,K=4"):
        audit(output, log)
