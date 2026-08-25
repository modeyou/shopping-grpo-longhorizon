import json

import pytest

from scripts.audit_bpo_formal_run import audit


def _write_run(tmp_path, *, returns=1600, branches=1200):
    output = tmp_path / "run"
    output.mkdir(parents=True)
    contract = {
        "schema_version": "shopping-bpo-run-contract-v1",
        "launch": {"reward_profile": "none", "seed": 20260823, "logger": "swanlab"},
        "frozen_method": {
            "effective_tree_budget": 400,
            "effective_return_budget": 1600,
            "trees_per_optimizer_step": 2,
            "returns_per_optimizer_step": 8,
            "maximum_optimizer_steps": 200,
            "checkpoint_steps": [10, 25, 50, 75, 100, 125, 150, 175, 200],
            "validation_steps": [0, 10, 50, 100, 150, 200],
            "scheduler": "cosine",
            "scheduler_horizon": 500,
            "warmup_steps": 10,
            "minimum_lr_ratio": 0.1,
        },
    }
    (output / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    metrics = {
        "training/optimizer_updated": 1,
        "bpo_budget/effective_trees_total": 400,
        "bpo_budget/effective_returns_total": returns,
        "bpo_budget/effective_return_target": 1600,
        "bpo_batch/trees": 2,
        "bpo_batch/sibling_returns": 8,
        "bpo_batch/full_batch": 1,
        "bpo_sampling/candidate_batches": 2,
        "bpo_sampling/seconds_to_first_tree": 3.0,
        "bpo_sampling/seconds_to_full_batch": 7.0,
        "rollout/generated_response_tokens_total": 1234,
        "bpo_cost/backbone_rollouts_total": 400,
        "bpo_cost/branch_rollouts_total": branches,
        "bpo_cost/environment_transitions_total": 500,
        "bpo_cost/shopper_api_calls_total": 20,
    }
    events = [
        {"event": "optimizer_step", "global_step": step, "metrics": metrics}
        for step in range(1, 201)
    ]
    event = events[0]
    (output / "training_diagnostics.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    with (output / "training_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
        for event in events[1:]:
            handle.write(json.dumps(event) + chr(10))
    for step in [10, *range(25, 201, 25)]:
        (output / f"global_step_{step}").mkdir()
    log = tmp_path / "run.log"
    log.write_text(
        "BPO scheduler contract:\n"
        "BPO optimizer update audit:\n"
        "BPO tree audit passed:\n",
        encoding="utf-8",
    )
    return output, log


def test_formal_audit_accepts_closed_r1600_contract(tmp_path):
    output, log = _write_run(tmp_path)
    result = audit(output, log)
    assert result["status"] == "accepted"
    assert result["effective_returns"] == 1600


def test_formal_audit_rejects_budget_or_tree_cost_mismatch(tmp_path):
    output, log = _write_run(tmp_path, returns=1592)
    with pytest.raises(ValueError, match="effective_returns_total"):
        audit(output, log)

    output, log = _write_run(tmp_path / "second", branches=1199)
    with pytest.raises(ValueError, match="K=4"):
        audit(output, log)
