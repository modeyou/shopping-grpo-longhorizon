#!/usr/bin/env python3
"""Audit and summarize two or more model runs on one frozen final benchmark."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path


CONDITIONS = (
    "gap-ask-enabled",
    "gap-ask-disabled",
    "complete-ask-enabled",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=ROOT")
    label, root = value.split("=", 1)
    if not label.strip() or not root.strip():
        raise argparse.ArgumentTypeError("--run must be LABEL=ROOT")
    return label.strip(), Path(root)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(gains, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def load_run(label: str, root: Path) -> dict:
    root = root.resolve()
    audit = read_json(root / "evaluation_results.json")
    if audit.get("schema_version") != "shopping-final-model-evaluation-v1":
        raise ValueError(f"{label}: unsupported final evaluation schema")
    if audit.get("evaluation_role") != "final" or audit.get("final_evaluation_used") is not True:
        raise ValueError(f"{label}: run is not marked as final")
    if audit.get("reward_contract") != "shopsimulator-reward-v4":
        raise ValueError(f"{label}: run did not use Reward v4")
    if int(audit.get("expected_tasks_per_condition") or 0) != 200:
        raise ValueError(f"{label}: final evaluation must have 200 tasks per condition")

    summaries = {condition: read_json(root / condition / "summary.json") for condition in CONDITIONS}
    strict_ids = {}
    for condition, summary in summaries.items():
        if summary.get("completed_tasks") != 200 or summary.get("missing_tasks") != []:
            raise ValueError(f"{label}: incomplete condition {condition}")
        protocol = summary.get("protocol") or {}
        if protocol.get("reward_contract") != "shopsimulator-reward-v4":
            raise ValueError(f"{label}: Reward mismatch in {condition}")
        ids = {int(task_id) for task_id in summary.get("strict_success_task_ids") or []}
        if len(ids) != int(summary.get("strict_successes") or 0):
            raise ValueError(f"{label}: strict task IDs disagree with count in {condition}")
        strict_ids[condition] = ids
    return {"root": str(root), "audit": audit, "summaries": summaries, "strict_ids": strict_ids}


def pairwise(source: dict, target: dict) -> dict:
    result = {}
    total_gains = 0
    total_losses = 0
    for condition in CONDITIONS:
        left = source["strict_ids"][condition]
        right = target["strict_ids"][condition]
        gained = sorted(right - left)
        lost = sorted(left - right)
        total_gains += len(gained)
        total_losses += len(lost)
        result[condition] = {
            "gains": len(gained),
            "losses": len(lost),
            "ties": 200 - len(gained) - len(lost),
            "gained_task_ids": gained,
            "lost_task_ids": lost,
            "exact_mcnemar_p": exact_mcnemar_p(len(gained), len(lost)),
        }
    result["all_conditions"] = {
        "gains": total_gains,
        "losses": total_losses,
        "ties": 600 - total_gains - total_losses,
        "descriptive_only": True,
        "note": "Conditions share task IDs; use condition-level McNemar tests.",
    }
    return result


def summarize(runs: dict[str, dict]) -> dict:
    manifest_hashes = {run["audit"]["asset_manifest_sha256"] for run in runs.values()}
    if len(manifest_hashes) != 1:
        raise ValueError("final runs used different asset manifests")
    models = {}
    for label, run in runs.items():
        conditions = {}
        for condition in CONDITIONS:
            summary = run["summaries"][condition]
            successes = int(summary["strict_successes"])
            conditions[condition] = {
                "strict_successes": successes,
                "strict_success_rate": successes / 200,
                "wilson_95_ci": wilson_interval(successes, 200),
                "done_tasks": int(summary["done_tasks"]),
                "reward_valid_tasks": int(summary["reward_valid_tasks"]),
                "mean_final_reward": float(summary["mean_final_reward"]),
            }
        result = run["audit"]["result"]
        models[label] = {
            "model_name": run["audit"]["model_name"],
            "conditions": conditions,
            "three_condition_strict": sum(
                values["strict_successes"] for values in conditions.values()
            ),
            "gap_gain_pp": 100
            * (
                conditions["gap-ask-enabled"]["strict_success_rate"]
                - conditions["gap-ask-disabled"]["strict_success_rate"]
            ),
            "complete_unnecessary_ask_rate": result["derived"][
                "complete_unnecessary_ask_rate"
            ],
        }
    return {
        "schema_version": "shopping-final-comparison-v1",
        "reward_contract": "shopsimulator-reward-v4",
        "asset_manifest_sha256": next(iter(manifest_hashes)),
        "tasks_per_condition": 200,
        "models": models,
        "pairwise": {
            f"{left}_to_{right}": pairwise(runs[left], runs[right])
            for left, right in combinations(runs, 2)
        },
    }


def markdown_report(result: dict) -> str:
    lines = [
        "# Final-200×3 结果",
        "",
        f"资产 manifest SHA-256：`{result['asset_manifest_sha256']}`",
        "",
        "| 模型 | G+ strict | G− strict | C+ strict | 三条件 strict | G+−G− | C+ 多余提问 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, model in result["models"].items():
        conditions = model["conditions"]
        def cell(condition: str) -> str:
            value = conditions[condition]
            return f"{value['strict_successes']}/200 ({value['strict_success_rate']:.1%})"
        lines.append(
            f"| {label} | {cell('gap-ask-enabled')} | {cell('gap-ask-disabled')} | "
            f"{cell('complete-ask-enabled')} | {model['three_condition_strict']}/600 "
            f"({model['three_condition_strict']/600:.1%}) | {model['gap_gain_pp']:+.1f} pp | "
            f"{model['complete_unnecessary_ask_rate']:.1%} |"
        )
    lines.extend(["", "## 逐题配对", ""])
    for name, comparison in result["pairwise"].items():
        overall = comparison["all_conditions"]
        lines.append(
            f"- {name}: gains={overall['gains']}, losses={overall['losses']}, "
            f"ties={overall['ties']}（三条件描述性合计）"
        )
        for condition in CONDITIONS:
            values = comparison[condition]
            lines.append(
                f"  - {condition}: gains={values['gains']}, losses={values['losses']}, "
                f"exact McNemar p={values['exact_mcnemar_p']:.4g}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=run_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.run) < 2:
        raise SystemExit("at least two --run LABEL=ROOT values are required")
    if len({label for label, _ in args.run}) != len(args.run):
        raise SystemExit("final run labels must be unique")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")
    runs = {label: load_run(label, root) for label, root in args.run}
    result = summarize(runs)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = markdown_report(result)
    (output / "final_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"FINAL COMPARISON ACCEPTED: {output}")


if __name__ == "__main__":
    main()
