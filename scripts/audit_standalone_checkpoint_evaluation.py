#!/usr/bin/env python3
"""Audit one standalone dev evaluation and optionally compare an SFT baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = (
    "gap-ask-enabled",
    "gap-ask-disabled",
    "complete-ask-enabled",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(result: dict, expected_tasks: int) -> dict:
    for condition in CONDITIONS:
        if condition not in result:
            raise ValueError(f"missing condition: {condition}")
    total_tasks = expected_tasks * len(CONDITIONS)
    return {
        "gap_ask": result["gap-ask-enabled"]["strict_success_rate"],
        "gap_noask": result["gap-ask-disabled"]["strict_success_rate"],
        "complete": result["complete-ask-enabled"]["strict_success_rate"],
        "total": sum(
            result[condition]["strict_successes"] for condition in CONDITIONS
        )
        / total_tasks,
        "gap_gain": (
            result["gap-ask-enabled"]["strict_success_rate"]
            - result["gap-ask-disabled"]["strict_success_rate"]
        ),
        "unnecessary_ask": (
            result["complete-ask-enabled"]["complete_unnecessary_ask_tasks"]
            / expected_tasks
        ),
        "mean_reward": sum(
            result[condition]["mean_final_reward"] for condition in CONDITIONS
        )
        / len(CONDITIONS),
        "done": sum(result[condition]["done_tasks"] for condition in CONDITIONS),
        "reward_valid": sum(
            result[condition]["reward_valid_tasks"] for condition in CONDITIONS
        ),
        "guards": sum(
            result[condition]["guard_rejections"] for condition in CONDITIONS
        ),
    }


def print_row(label: str, values: dict, total_tasks: int) -> None:
    print(
        f"{label}\t{values['gap_ask']:.3f}\t{values['gap_noask']:.3f}\t"
        f"{values['complete']:.3f}\t{values['total']:.3f}\t"
        f"{values['gap_gain']:+.3f}\t{values['unnecessary_ask']:.3f}\t"
        f"{values['mean_reward']:.4f}\t{values['done']}/{total_tasks}\t"
        f"{values['reward_valid']}/{total_tasks}\t{values['guards']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected-asset-manifest-sha256", required=True)
    parser.add_argument("--expected-source-step", type=int, required=True)
    parser.add_argument("--expected-model-name", required=True)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--baseline-checkpoint", default="checkpoint-325")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = read_json(args.results.resolve())
    if audit.get("schema_version") != "shopping-standalone-checkpoint-evaluation-v1":
        raise SystemExit("unsupported standalone evaluation schema")
    if audit.get("reward_contract") != "shopsimulator-reward-v4":
        raise SystemExit("evaluation did not use Reward v4")
    if audit.get("final_evaluation_used") is not False:
        raise SystemExit("final200 contamination detected")
    if audit.get("asset_manifest_sha256") != args.expected_asset_manifest_sha256:
        raise SystemExit("dev asset manifest SHA256 mismatch")
    if audit.get("model_name") != args.expected_model_name:
        raise SystemExit("served model name mismatch")
    source = audit.get("source_checkpoint") or {}
    if source.get("step") != args.expected_source_step:
        raise SystemExit("source checkpoint step mismatch")
    expected_tasks = int(audit["expected_tasks_per_condition"])
    if expected_tasks != 500:
        raise SystemExit(f"expected dev500, got {expected_tasks} tasks per condition")

    total_tasks = expected_tasks * len(CONDITIONS)
    candidate = summarize(audit["result"], expected_tasks)
    print(
        "checkpoint\tgap_ask\tgap_noask\tcomplete\ttotal\tgap_gain\t"
        "unnecessary_ask\tmean_reward\tdone\treward_valid\tguards"
    )
    print_row(args.expected_model_name, candidate, total_tasks)

    if args.baseline_results is not None:
        baseline_audit = read_json(args.baseline_results.resolve())
        if (
            baseline_audit.get("schema_version")
            != "shopping-sft-checkpoint-sweep-results-v1"
        ):
            raise SystemExit("unsupported SFT baseline schema")
        if baseline_audit.get("reward_contract") != "shopsimulator-reward-v4":
            raise SystemExit("SFT baseline did not use Reward v4")
        if baseline_audit.get("final_evaluation_used") is not False:
            raise SystemExit("SFT baseline used final200")
        if baseline_audit.get("asset_manifest_sha256") != audit.get(
            "asset_manifest_sha256"
        ):
            raise SystemExit("candidate/baseline dev asset mismatch")
        if baseline_audit.get("expected_tasks_per_condition") != expected_tasks:
            raise SystemExit("candidate/baseline task count mismatch")
        completed = baseline_audit.get("completed") or {}
        if args.baseline_checkpoint not in completed:
            raise SystemExit(
                f"baseline checkpoint is missing: {args.baseline_checkpoint}"
            )
        baseline = summarize(completed[args.baseline_checkpoint], expected_tasks)
        print_row(args.baseline_checkpoint, baseline, total_tasks)
        print("\nBPO MINUS BASELINE")
        for name in (
            "gap_ask",
            "gap_noask",
            "complete",
            "total",
            "gap_gain",
            "unnecessary_ask",
            "mean_reward",
        ):
            print(f"{name}: {candidate[name] - baseline[name]:+.4f}")

    print("\nSTANDALONE DEV500 THREE-PANEL EVALUATION ACCEPTED")


if __name__ == "__main__":
    main()
