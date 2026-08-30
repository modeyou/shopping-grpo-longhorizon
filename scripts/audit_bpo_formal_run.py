#!/usr/bin/env python3
"""Accept a completed CARL-BPO run only when its frozen contracts close."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from shopping_grpo.training.bpo.step0_validation import (
    load_validation_cache,
    validate_contract,
)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    return parser.parse_args()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(output: Path, log: Path) -> dict:
    output = output.expanduser().resolve()
    log = log.expanduser().resolve()
    contract_path = output / "run_contract.json"
    diagnostics_path = output / "training_diagnostics.jsonl"
    for path in (contract_path, diagnostics_path, log):
        if not path.is_file():
            raise ValueError(f"required BPO artifact is missing: {path}")

    contract = _json(contract_path)
    if contract.get("schema_version") != "shopping-carl-bpo-run-contract-v1":
        raise ValueError("unexpected CARL-BPO run contract schema")
    method = contract.get("frozen_method") or {}
    expected_method = {
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
    }
    for name, expected in expected_method.items():
        if method.get(name) != expected:
            raise ValueError(f"formal BPO contract mismatch: {name}")
    launch = contract.get("launch") or {}
    if launch.get("algorithm") != "carl-bpo-v2":
        raise ValueError("CARL-BPO algorithm mismatch")
    if launch.get("reward_profile") != "none":
        raise ValueError("CARL-BPO must use Reward v4 with an explicit train return")
    if int(launch.get("seed", -1)) != 20260823:
        raise ValueError("CARL-BPO seed mismatch")
    if launch.get("logger") != "swanlab":
        raise ValueError("formal BPO must use SwanLab")


    step0 = contract.get("step0_validation") or {}
    if step0.get("reuse_policy") != "exact-contract-sha256-v1":
        raise ValueError("formal BPO step-0 reuse policy is missing")
    step0_contract_path = Path(str(step0.get("contract_path") or "")).resolve()
    step0_cache_path = Path(str(step0.get("cache_path") or "")).resolve()
    for path in (step0_contract_path, step0_cache_path):
        if not path.is_file():
            raise ValueError(f"required BPO step-0 artifact is missing: {path}")
    step0_contract = _json(step0_contract_path)
    step0_sha256 = validate_contract(step0_contract)
    if step0_sha256 != step0.get("contract_sha256"):
        raise ValueError("formal BPO step-0 contract hash mismatch")
    step0_metrics = load_validation_cache(
        step0_cache_path,
        expected_contract_sha256=step0_sha256,
    )
    required_step0_metrics = {
        "val-shopping/summary/strict_success_rate",
        "val-shopping/summary/purchase_success_rate",
        "val-shopping/summary/mean_reward",
        "val-shopping/summary/terminal_utility_mean",
        "val-shopping/summary/done_rate",
        "val-shopping/summary/sampling_invalid_rate",
    }
    missing_step0_metrics = required_step0_metrics.difference(step0_metrics or {})
    if missing_step0_metrics:
        raise ValueError(
            "formal BPO step-0 cache is missing key SwanLab metrics: "
            + ", ".join(sorted(missing_step0_metrics))
        )
    events = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    optimizer_events = [event for event in events if event.get("event") == "optimizer_step"]
    selection_events = [
        event for event in events if event.get("event") == "optimizer_selection"
    ]
    if not optimizer_events:
        raise ValueError("BPO run has no optimizer-step diagnostics")
    if len(optimizer_events) != 500:
        raise ValueError("CARL-BPO formal run must complete exactly 500 global steps")
    if len(selection_events) != 500:
        raise ValueError("CARL-BPO formal run must record 500 optimizer selections")
    if [int(event.get("global_step", -1)) for event in optimizer_events] != list(
        range(1, 501)
    ):
        raise ValueError("CARL-BPO optimizer diagnostics must cover global steps 1..500")
    if any(
        int((event.get("metrics") or {}).get("training/optimizer_updated", 0)) != 1
        for event in optimizer_events
    ):
        raise ValueError("BPO diagnostics contain a non-update optimizer event")

    final_metrics = optimizer_events[-1]["metrics"]
    required_final = {
        "bpo_budget/effective_trees_total": 1000,
        "bpo_budget/effective_returns_total": 4000,
        "bpo_budget/effective_return_target": 4000,
    }
    for name, expected in required_final.items():
        if int(final_metrics.get(name, -1)) != expected:
            raise ValueError(f"BPO final budget mismatch: {name}")
    generated_tokens = int(final_metrics.get("rollout/generated_response_tokens_total", 0))
    backbone = int(final_metrics.get("bpo_cost/backbone_rollouts_total", 0))
    branches = int(final_metrics.get("bpo_cost/branch_rollouts_total", 0))
    transitions = int(final_metrics.get("bpo_cost/environment_transitions_total", 0))
    shopper_calls = int(final_metrics.get("bpo_cost/shopper_api_calls_total", -1))
    if generated_tokens <= 0 or backbone <= 0 or transitions <= 0 or shopper_calls < 0:
        raise ValueError("BPO cumulative cost metrics are missing or invalid")
    if branches != 3 * backbone:
        raise ValueError("BPO generated tree cost does not satisfy K=4")
    required_training_metrics = {
        "summary/strict_success_rate",
        "summary/purchase_success_rate",
        "summary/mean_reward",
        "reward/train_return_mean",
        "summary/sampling_invalid_rate",
        "summary/infrastructure_invalid_rate",
        "summary/reward_unverifiable_rate",
        "bpo_branch/relative_position_mean",
        "bpo_branch/entropy_mean",
        "bpo_group/root_count",
        "bpo_group/local_count",
        "bpo_stage/product_count",
        "bpo_stage/option_count",
        "bpo_stage/search_strategy_count",
        "bpo_stage/fallback_count",
        "bpo_stage/unavailable_count",
        "bpo_diversity/unique_branch_actions_mean",
        "bpo_diversity/unique_tool_sequences_mean",
        "bpo_return/sibling_std_mean",
        "bpo_return/sibling_range_mean",
        "group/completion_contrast",
        "group/gold_contrast",
        "group/failure_utility_contrast",
        "group/effective_ratio",
        "carl_sampling/selected_goal_groups",
        "carl_sampling/selected_failure_groups",
        "carl_sampling/reservoir_replacements",
        "carl_sampling/local_stage_mismatch_groups",
    }

    for event in optimizer_events:
        metrics = event.get("metrics") or {}
        if int(metrics.get("bpo_batch/trees", -1)) != 2:
            raise ValueError("CARL-BPO formal run contains a non-2-group optimizer batch")
        if int(metrics.get("bpo_batch/root_groups", -1)) != 1:
            raise ValueError("CARL-BPO formal run requires one Root group per optimizer batch")
        if int(metrics.get("bpo_batch/local_groups", -1)) != 1:
            raise ValueError("CARL-BPO formal run requires one Local group per optimizer batch")
        if int(metrics.get("bpo_batch/sibling_returns", -1)) != 8:
            raise ValueError("BPO formal run contains a non-R8 optimizer batch")
        if int(metrics.get("bpo_batch/full_batch", -1)) != 1:
            raise ValueError("BPO formal run contains an incomplete optimizer batch")
        missing_metrics = required_training_metrics.difference(metrics)
        if missing_metrics:
            raise ValueError(
                "BPO optimizer diagnostics are missing key SwanLab metrics: "
                + ", ".join(sorted(missing_metrics))
            )
        if any(
            not math.isfinite(float(metrics[name]))
            for name in required_training_metrics
        ):
            raise ValueError("BPO key SwanLab metrics must all be finite")

        candidate_batches = int(metrics.get("bpo_sampling/candidate_batches", 0))
        first_tree = float(metrics.get("bpo_sampling/seconds_to_first_tree", -1))
        full_batch = float(metrics.get("bpo_sampling/seconds_to_full_batch", -1))
        if (
            candidate_batches < 1
            or not math.isfinite(first_tree)
            or not math.isfinite(full_batch)
            or not 0 <= first_tree <= full_batch
        ):
            raise ValueError("BPO full-batch acquisition metrics are invalid")

    selected_stage_counts = {"product": 0, "option": 0, "search_strategy": 0}
    for event in selection_events:
        selected = event.get("selected_groups") or {}
        if set(selected) != {"root", "local"}:
            raise ValueError("CARL-BPO selection must contain one Root and one Local")
        local_stage = str((selected["local"] or {}).get("local_stage"))
        target = str(event.get("local_stage_target"))
        if local_stage != target or local_stage not in selected_stage_counts:
            raise ValueError("CARL-BPO selected Local group violates its stage target")
        selected_stage_counts[local_stage] += 1
    if selected_stage_counts != {
        "product": 200,
        "option": 175,
        "search_strategy": 125,
    }:
        raise ValueError("CARL-BPO final Local stage coverage is not 40/35/25")

    log_text = log.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "BPO scheduler contract:",
        "BPO optimizer update audit:",
        "BPO tree audit passed:",
        "BPO step-0 validation cache",
    ):
        if marker not in log_text:
            raise ValueError(f"formal BPO log is missing marker: {marker}")
    checkpoints = sorted(
        path for path in output.glob("global_step_*") if path.is_dir()
    )
    checkpoint_steps = {
        int(path.name.removeprefix("global_step_")) for path in checkpoints
    }
    missing_checkpoint_steps = set(range(25, 501, 25)).difference(checkpoint_steps)
    if missing_checkpoint_steps:
        raise ValueError(
            "CARL-BPO is missing every-25-step checkpoints: "
            + ", ".join(str(step) for step in sorted(missing_checkpoint_steps))
        )

    return {
        "status": "accepted",
        "optimizer_steps": len(optimizer_events),
        "effective_trees": 1000,
        "effective_returns": 4000,
        "generated_response_tokens": generated_tokens,
        "backbone_rollouts": backbone,
        "branch_rollouts": branches,
        "environment_transitions": transitions,
        "shopper_api_calls": shopper_calls,
        "checkpoints": [str(path) for path in checkpoints],
        "step0_contract_sha256": step0_sha256,
        "step0_cache": str(step0_cache_path),
    }


def main():
    args = parse_args()
    try:
        result = audit(args.output, args.log)
    except ValueError as exc:
        raise SystemExit(f"BPO FORMAL AUDIT FAILED: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("CARL-BPO-N500-R4000 FORMAL RUN ACCEPTED")


if __name__ == "__main__":
    main()
