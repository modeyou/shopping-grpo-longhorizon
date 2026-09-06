#!/usr/bin/env python3
"""Run one frozen Rubric/Judge evaluation grid from a declarative JSON plan.

The plan contains only paths and non-secret Judge settings.  API credentials are
read by the child evaluator from OPENAI_API_KEY, never from the plan or disk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from shopping_grpo.evaluation.comparison import MULTITURN_CONDITIONS


ROOT = Path(__file__).resolve().parents[1]
PANEL_SCRIPT = ROOT / "scripts/evaluate_multiturn_panels.py"
COMPARE_SCRIPT = ROOT / "scripts/compare_multiturn_evaluations.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument(
        "--condition",
        action="append",
        choices=MULTITURN_CONDITIONS,
        help="Condition to run; omit only for the full G+/G-/C+ grid.",
    )
    parser.add_argument("--allow-blind-final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_plan(path: Path) -> dict:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON plan") from exc
    if not isinstance(plan, dict):
        raise ValueError(f"{path}: plan must be an object")
    return plan


def _path(value: object, *, plan_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"plan.{field} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else plan_dir / path


def _judge_args(judge: object) -> list[str]:
    if not isinstance(judge, dict):
        raise ValueError("plan.judge must be an object")
    required = {"model": "--judge-model", "base_url": "--judge-base-url"}
    result = []
    for key, flag in required.items():
        value = judge.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"plan.judge.{key} must be a non-empty string")
        result.extend([flag, value])
    optional = {
        "max_tokens": "--max-tokens",
        "timeout": "--timeout",
        "retries": "--retries",
        "schema_retries": "--schema-retries",
    }
    for key, flag in optional.items():
        if key in judge:
            value = judge[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"plan.judge.{key} must be numeric")
            result.extend([flag, str(value)])
    return result


def build_commands(
    plan: dict,
    *,
    plan_dir: Path,
    output_root: Path,
    allow_blind_final: bool,
    resume: bool,
    comparison_output: Path | None,
    conditions: tuple[str, ...] = MULTITURN_CONDITIONS,
) -> list[list[str]]:
    expected_tasks = _path(
        plan.get("expected_tasks"), plan_dir=plan_dir, field="expected_tasks"
    )
    rubrics = _path(plan.get("rubrics"), plan_dir=plan_dir, field="rubrics")
    judge_args = _judge_args(plan.get("judge"))
    judge_cache_dir = output_root / "semantic-judge-cache"
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("plan.runs must be a non-empty list")

    commands = []
    seen_labels = set()
    compare_runs = []
    for run_index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"plan.runs[{run_index}] must be an object")
        label = run.get("actor_label")
        if not isinstance(label, str) or not label.strip() or label in seen_labels:
            raise ValueError("every plan actor_label must be non-empty and unique")
        seen_labels.add(label)
        trajectories = run.get("trajectories")
        if not isinstance(trajectories, dict):
            raise ValueError(f"plan.runs[{run_index}].trajectories must be an object")
        if not set(conditions).issubset(trajectories):
            raise ValueError(
                f"plan.runs[{run_index}].trajectories must contain "
                f"{list(conditions)}"
            )
        actor_root = output_root / label
        compare_runs.extend(["--run", f"{label}={actor_root}"])
        for condition in conditions:
            trajectory = _path(
                trajectories[condition],
                plan_dir=plan_dir,
                field=f"runs[{run_index}].trajectories.{condition}",
            )
            command = [
                sys.executable,
                str(PANEL_SCRIPT),
                "--expected-tasks",
                str(expected_tasks),
                "--trajectories",
                str(trajectory),
                "--rubrics",
                str(rubrics),
                "--output-dir",
                str(actor_root / condition),
                "--judge-cache-dir",
                str(judge_cache_dir),
                "--actor-label",
                label,
                "--condition",
                condition,
                *judge_args,
            ]
            if allow_blind_final:
                command.append("--allow-blind-final")
            if resume:
                command.append("--resume")
            commands.append(command)

    comparison = comparison_output or output_root / "comparison.json"
    command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--expected-tasks",
        str(expected_tasks),
        *compare_runs,
        "--output",
        str(comparison),
    ]
    for condition in conditions:
        command.extend(["--condition", condition])
    if allow_blind_final:
        command.append("--allow-blind-final")
    if resume:
        command.append("--force")
    commands.append(command)
    return commands


def main():
    args = parse_args()
    plan = _load_plan(args.plan)
    commands = build_commands(
        plan,
        plan_dir=args.plan.parent,
        output_root=args.output_root,
        allow_blind_final=args.allow_blind_final,
        resume=args.resume,
        comparison_output=args.comparison_output,
        conditions=tuple(args.condition or MULTITURN_CONDITIONS),
    )
    for command in commands:
        print(" ".join(json.dumps(part) for part in command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
