import json
from pathlib import Path

import pytest

from scripts.audit_bpo_formal_run import audit
from shopping_grpo.training.grpo.dynamic_sampling import select_carl_local_stage_target
from shopping_grpo.training.bpo.step0_validation import (
    build_validation_contract,
    freeze_validation_cache,
)



def _write_run(tmp_path, *, returns=4000, branches=3000):
    output = tmp_path / "run"
    output.mkdir(parents=True)
    step0_input = tmp_path / "step0-validation-input.parquet"
    step0_input.write_bytes(b"frozen validation fixture")
    step0_contract = build_validation_contract(
        root=tmp_path,
        git_commit="a" * 40,
        inputs={"validation_data": step0_input},
        settings={"validation_sampling": "deterministic-n1"},
    )
    step0_contract_path = output / "step0_validation_contract.json"
    step0_contract_path.write_text(
        json.dumps(step0_contract), encoding="utf-8"
    )
    step0_cache_path = output / "step0_validation_cache.json"
    freeze_validation_cache(
        step0_cache_path,
        contract_sha256_value=step0_contract["contract_sha256"],
        metrics={
            "val-shopping/summary/strict_success_rate": 0.6,
            "val-shopping/summary/purchase_success_rate": 0.6,
            "val-shopping/summary/mean_reward": 0.5,
            "val-shopping/summary/terminal_utility_mean": 0.5,
            "val-shopping/summary/done_rate": 0.9,
            "val-shopping/summary/sampling_invalid_rate": 0.0,
        },
    )
    contract = {
        "schema_version": "shopping-carl-bpo-run-contract-v1",
        "launch": {
            "algorithm": "carl-bpo-v2",
            "reward_profile": "none",
            "seed": 20260823,
            "logger": "swanlab",
        },
        "frozen_method": {
            "effective_tree_budget": 1000,
            "effective_return_budget": 4000,
            "group_schedule": ["root", "local"],
            "local_stage_weights": {
                "product": 8,
                "option": 7,
                "search_strategy": 5,
            },
            "candidate_selector": "goal-priority-reservoir-v2",
            "trees_per_optimizer_step": 2,
            "returns_per_optimizer_step": 8,
            "maximum_optimizer_steps": 500,
            "checkpoint_steps": list(range(25, 501, 25)),
            "validation_steps": [0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "scheduler": "cosine",
            "scheduler_horizon": 500,
            "warmup_steps": 10,
            "minimum_lr_ratio": 0.1,
            "optimizer_update_audit": (
                "startup-hard-gate-first-gradient-and-positive-lr-delta-v1"
            ),
        },
        "step0_validation": {
            "reuse_policy": "exact-contract-sha256-v1",
            "contract_path": str(step0_contract_path),
            "contract_sha256": step0_contract["contract_sha256"],
            "cache_path": str(step0_cache_path),
        },
    }
    (output / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    metrics = {
        "training/optimizer_updated": 1,
        "bpo_budget/effective_trees_total": 1000,
        "bpo_budget/effective_returns_total": returns,
        "bpo_budget/effective_return_target": 4000,
        "bpo_batch/trees": 2,
        "bpo_batch/root_groups": 1,
        "bpo_batch/local_groups": 1,
        "bpo_batch/sibling_returns": 8,
        "bpo_batch/full_batch": 1,
        "bpo_sampling/candidate_batches": 2,
        "bpo_sampling/seconds_to_first_tree": 3.0,
        "bpo_sampling/seconds_to_full_batch": 7.0,
        "rollout/generated_response_tokens_total": 1234,
        "bpo_cost/backbone_rollouts_total": 1000,
        "bpo_cost/branch_rollouts_total": branches,
        "bpo_cost/environment_transitions_total": 500,
        "bpo_cost/shopper_api_calls_total": 20,
        "summary/strict_success_rate": 0.6,
        "summary/purchase_success_rate": 0.6,
        "summary/mean_reward": 0.5,
        "summary/sampling_invalid_rate": 0.0,
        "summary/infrastructure_invalid_rate": 0.0,
        "summary/reward_unverifiable_rate": 0.0,
        "reward/train_return_mean": 0.5,
        "bpo_group/root_count": 1.0,
        "bpo_group/local_count": 1.0,
        "bpo_stage/product_count": 1.0,
        "bpo_stage/option_count": 1.0,
        "bpo_stage/search_strategy_count": 0.0,
        "bpo_stage/fallback_count": 0.0,
        "bpo_stage/unavailable_count": 0.0,
        "bpo_branch/relative_position_mean": 0.5,
        "bpo_branch/entropy_mean": 2.0,
        "bpo_diversity/unique_branch_actions_mean": 3.0,
        "bpo_diversity/unique_tool_sequences_mean": 2.0,
        "bpo_return/sibling_std_mean": 0.25,
        "bpo_return/sibling_range_mean": 0.5,
        "group/completion_contrast": 1.0,
        "group/gold_contrast": 0.0,
        "group/failure_utility_contrast": 1.0,
        "group/effective_ratio": 1.0,
        "carl_sampling/selected_goal_groups": 1.0,
        "carl_sampling/selected_failure_groups": 1.0,
        "carl_sampling/reservoir_replacements": 0.0,
        "carl_sampling/local_stage_mismatch_groups": 0.0,
    }
    events = []
    for step in range(1, 501):
        stage, _ = select_carl_local_stage_target(step - 1)
        events.extend(
            [
                {
                    "event": "optimizer_selection",
                    "global_step": step,
                    "local_stage_target": stage,
                    "selected_groups": {
                        "root": {"local_stage": "root"},
                        "local": {"local_stage": stage},
                    },
                },
                {"event": "optimizer_step", "global_step": step, "metrics": metrics},
            ]
        )
    event = events[0]
    (output / "training_diagnostics.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    with (output / "training_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
        for event in events[1:]:
            handle.write(json.dumps(event) + chr(10))
    for step in range(25, 501, 25):
        (output / f"global_step_{step}").mkdir()
    log = tmp_path / "run.log"
    log.write_text(
        "BPO scheduler contract:\n"
        "BPO optimizer update audit:\n"
        "BPO tree audit passed:\n"
        "BPO step-0 validation cache frozen:\n",
        encoding="utf-8",
    )
    return output, log


def test_formal_audit_accepts_closed_r4000_contract(tmp_path):
    output, log = _write_run(tmp_path)
    result = audit(output, log)
    assert result["status"] == "accepted"
    assert result["effective_returns"] == 4000


def test_formal_audit_rejects_nonblocking_optimizer_contract(tmp_path):
    output, log = _write_run(tmp_path)
    contract_path = output / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["frozen_method"]["optimizer_update_audit"] = (
        "first-step-nonzero-gradient-and-delta-v1"
    )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer_update_audit"):
        audit(output, log)


def test_formal_audit_rejects_budget_or_tree_cost_mismatch(tmp_path):
    output, log = _write_run(tmp_path, returns=3992)
    with pytest.raises(ValueError, match="effective_returns_total"):
        audit(output, log)

    output, log = _write_run(tmp_path / "second", branches=1199)
    with pytest.raises(ValueError, match="K=4"):
        audit(output, log)


def test_formal_audit_rejects_a_missing_step0_cache(tmp_path):
    output, log = _write_run(tmp_path)
    contract = json.loads((output / "run_contract.json").read_text(encoding="utf-8"))
    cache = contract["step0_validation"]["cache_path"]
    Path(cache).unlink()
    with pytest.raises(ValueError, match="step-0 artifact is missing"):
        audit(output, log)


def test_formal_audit_rejects_a_missing_25_step_checkpoint(tmp_path):
    output, log = _write_run(tmp_path)
    (output / "global_step_275").rmdir()
    with pytest.raises(ValueError, match="missing every-25-step checkpoints: 275"):
        audit(output, log)
