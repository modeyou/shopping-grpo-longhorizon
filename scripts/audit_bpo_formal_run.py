#!/usr/bin/env python3
"""Accept a completed BPO-N-R400 run only when its frozen contracts close."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    if contract.get("schema_version") != "shopping-bpo-run-contract-v1":
        raise ValueError("unexpected BPO run contract schema")
    method = contract.get("frozen_method") or {}
    expected_method = {
        "effective_tree_budget": 100,
        "effective_return_budget": 400,
        "maximum_optimizer_steps": 100,
        "scheduler": "cosine",
        "scheduler_horizon": 500,
        "warmup_steps": 10,
        "minimum_lr_ratio": 0.1,
    }
    for name, expected in expected_method.items():
        if method.get(name) != expected:
            raise ValueError(f"formal BPO contract mismatch: {name}")
    launch = contract.get("launch") or {}
    if launch.get("reward_profile") != "none":
        raise ValueError("BPO-N-R400 must use native Reward v4")
    if int(launch.get("seed", -1)) != 20260823:
        raise ValueError("BPO-N-R400 seed mismatch")
    if launch.get("logger") != "swanlab":
        raise ValueError("formal BPO must use SwanLab")

    events = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    optimizer_events = [event for event in events if event.get("event") == "optimizer_step"]
    if not optimizer_events:
        raise ValueError("BPO run has no optimizer-step diagnostics")
    if len(optimizer_events) > 100:
        raise ValueError("BPO exceeded its optimizer-step safety ceiling")
    if any(
        int((event.get("metrics") or {}).get("training/optimizer_updated", 0)) != 1
        for event in optimizer_events
    ):
        raise ValueError("BPO diagnostics contain a non-update optimizer event")

    final_metrics = optimizer_events[-1]["metrics"]
    required_final = {
        "bpo_budget/effective_trees_total": 100,
        "bpo_budget/effective_returns_total": 400,
        "bpo_budget/effective_return_target": 400,
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
        raise ValueError("BPO generated tree cost does not satisfy M=1,K=4")

    log_text = log.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "BPO scheduler contract:",
        "BPO optimizer update audit:",
        "BPO tree audit passed:",
    ):
        if marker not in log_text:
            raise ValueError(f"formal BPO log is missing marker: {marker}")
    checkpoints = sorted(
        path for path in output.glob("global_step_*") if path.is_dir()
    )
    if len(checkpoints) < 2:
        raise ValueError("formal BPO requires R200 and R400 checkpoints")

    return {
        "status": "accepted",
        "optimizer_steps": len(optimizer_events),
        "effective_trees": 100,
        "effective_returns": 400,
        "generated_response_tokens": generated_tokens,
        "backbone_rollouts": backbone,
        "branch_rollouts": branches,
        "environment_transitions": transitions,
        "shopper_api_calls": shopper_calls,
        "checkpoints": [str(path) for path in checkpoints],
    }


def main():
    args = parse_args()
    try:
        result = audit(args.output, args.log)
    except ValueError as exc:
        raise SystemExit(f"BPO FORMAL AUDIT FAILED: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("BPO-N-R400 FORMAL RUN ACCEPTED")


if __name__ == "__main__":
    main()
