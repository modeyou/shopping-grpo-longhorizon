"""Paired metrics for Probe B V2 No-Ask versus Oracle-Ask runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
PROBE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(PROBE_DIR))

from runner import (  # noqa: E402
    ARMS,
    DEFAULT_OUTPUT_ROOT,
    MANIFEST_FILE,
    TRAJECTORIES_FILE,
    load_records,
    write_json,
)
from task_schema import (  # noqa: E402
    DEFAULT_TASKS,
    latent_goal_satisfied,
    load_tasks,
    validate_tasks,
)


EXPECTED_PAIRS = 25
MIN_NET_STRICT_WINS = 3


def strict_success(record: Mapping[str, Any]) -> bool:
    terminal = _terminal(record)
    detail = _reward_detail(record)
    return (
        _record_valid(record)
        and record.get("status") == "done"
        and record.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
        and detail.get("reward_version") == "shopsimulator-reward-v3"
        and detail.get("reward_type") == "gold_purchase"
        and detail.get("reward_valid") is True
        and detail.get("purchase_success") is True
    )


def wrong_purchase(record: Mapping[str, Any]) -> bool:
    return _record_valid(record) and _reward_detail(record).get("reward_type") == "wrong_purchase"


def analyze(
    tasks: list[dict[str, Any]], records: list[dict[str, Any]], mode: str = "real"
) -> dict[str, Any]:
    tasks_by_id = {int(task["task_id"]): task for task in tasks}
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    duplicates = []
    for record in records:
        key = (int(record["task_id"]), str(record.get("arm")))
        if key in by_key:
            duplicates.append(f"{key[0]}:{key[1]}")
        by_key[key] = record
    if duplicates:
        raise ValueError("duplicate task-arm trajectories: " + ", ".join(sorted(duplicates)))

    invalid_keys = sorted(
        f"{task_id}:{arm}"
        for (task_id, arm), record in by_key.items()
        if not _record_valid(record)
    )
    paired_ids = [
        task_id
        for task_id in tasks_by_id
        if all(
            (task_id, arm) in by_key and _record_valid(by_key[(task_id, arm)])
            for arm in ARMS
        )
    ]
    paired_ids.sort()
    arm_stats = {
        arm: _arm_summary(
            [by_key[(task_id, arm)] for task_id in paired_ids],
            tasks_by_id,
        )
        for arm in ARMS
    }

    fail_to_success = 0
    success_to_fail = 0
    both_success = 0
    both_fail = 0
    pair_rows = []
    for task_id in paired_ids:
        no_ask = by_key[(task_id, "no_ask")]
        oracle = by_key[(task_id, "oracle_ask")]
        no_success = strict_success(no_ask)
        oracle_success = strict_success(oracle)
        if not no_success and oracle_success:
            fail_to_success += 1
        elif no_success and not oracle_success:
            success_to_fail += 1
        elif no_success and oracle_success:
            both_success += 1
        else:
            both_fail += 1
        task = tasks_by_id[task_id]
        pair_rows.append(
            {
                "task_id": task_id,
                "field": task["latent_goal"]["field"],
                "no_ask": _row_view(task, no_ask),
                "oracle_ask": _row_view(task, oracle),
            }
        )

    net_strict_wins = fail_to_success - success_to_fail
    paired_count = len(paired_ids)
    strict_delta = net_strict_wins / paired_count if paired_count else None
    full_pairs = paired_count == EXPECTED_PAIRS
    conditions = {
        "net_strict_wins_at_least_3": net_strict_wins >= MIN_NET_STRICT_WINS,
        "latent_satisfaction_higher": (
            arm_stats["oracle_ask"]["latent_satisfaction_rate"]
            > arm_stats["no_ask"]["latent_satisfaction_rate"]
            if paired_count
            else False
        ),
        "wrong_purchase_not_higher": (
            arm_stats["oracle_ask"]["wrong_purchase_count"]
            <= arm_stats["no_ask"]["wrong_purchase_count"]
        ),
    }
    if mode != "real":
        gate_status = "NOT_APPLICABLE_MOCK"
    elif not full_pairs:
        gate_status = "INCOMPLETE"
    else:
        gate_status = "PASS" if all(conditions.values()) else "FAIL"

    attempted_keys = {f"{task_id}:{arm}" for task_id, arm in by_key}
    expected_keys = {f"{task_id}:{arm}" for task_id in tasks_by_id for arm in ARMS}
    return {
        "mode": mode,
        "expected_pair_count": EXPECTED_PAIRS,
        "valid_pair_count": paired_count,
        "trajectory_count": len(records),
        "invalid_keys": invalid_keys,
        "missing_keys": sorted(expected_keys - attempted_keys),
        "arms": arm_stats,
        "paired_transitions": {
            "no_ask_fail_to_oracle_success": fail_to_success,
            "no_ask_success_to_oracle_fail": success_to_fail,
            "both_success": both_success,
            "both_fail": both_fail,
            "net_strict_wins": net_strict_wins,
            "strict_success_delta": strict_delta,
        },
        "continue_gate": {
            "status": gate_status,
            "evaluable": mode == "real" and full_pairs,
            "conditions": conditions,
        },
        "pairs": pair_rows,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    arms = report["arms"]
    transitions = report["paired_transitions"]
    gate = report["continue_gate"]
    lines = [
        "# Probe B V2 配对结果",
        "",
        f"- 模式：`{report['mode']}`",
        f"- 有效任务对：{report['valid_pair_count']} / {report['expected_pair_count']}",
        f"- 判据状态：**{gate['status']}**",
        f"- 无效轨迹：{len(report['invalid_keys'])}",
        f"- 未尝试轨迹：{len(report['missing_keys'])}",
        "",
        "## 两臂汇总",
        "",
        "| 指标 | No-Ask | Oracle-Ask |",
        "|---|---:|---:|",
        (
            "| 严格成功率 | "
            f"{_percent(arms['no_ask']['strict_success_rate'])} | "
            f"{_percent(arms['oracle_ask']['strict_success_rate'])} |"
        ),
        (
            "| 隐藏字段满足率 | "
            f"{_percent(arms['no_ask']['latent_satisfaction_rate'])} | "
            f"{_percent(arms['oracle_ask']['latent_satisfaction_rate'])} |"
        ),
        (
            "| 真实 wrong_purchase | "
            f"{arms['no_ask']['wrong_purchase_count']} | "
            f"{arms['oracle_ask']['wrong_purchase_count']} |"
        ),
        (
            "| 平均环境步数 | "
            f"{arms['no_ask']['average_steps']:.2f} | "
            f"{arms['oracle_ask']['average_steps']:.2f} |"
        ),
        "",
        "## 配对变化",
        "",
        f"- No-Ask 失败 → Oracle 成功：{transitions['no_ask_fail_to_oracle_success']}",
        f"- No-Ask 成功 → Oracle 失败：{transitions['no_ask_success_to_oracle_fail']}",
        f"- 净增加严格成功任务：{transitions['net_strict_wins']}",
        f"- 严格成功率差：{_percent(transitions['strict_success_delta'])}",
        "",
        "## 预登记继续门槛",
        "",
    ]
    labels = {
        "net_strict_wins_at_least_3": "净增加至少 3 个严格成功任务",
        "latent_satisfaction_higher": "隐藏字段满足率提高",
        "wrong_purchase_not_higher": "真实错误购买不增加",
    }
    for key, label in labels.items():
        lines.append(f"- [{'PASS' if gate['conditions'][key] else 'FAIL'}] {label}")
    if not gate["evaluable"]:
        reason = (
            "当前是 Mock 结果，不适用真实实验的 PASS/FAIL 决策。"
            if report["mode"] != "real"
            else "当前结果不足 25 个完整有效真实任务对，不能做 PASS/FAIL 决策。"
        )
        lines.extend(
            [
                "",
                reason,
            ]
        )

    lines.extend(
        [
            "",
            "## 逐任务",
            "",
            "| task | field | No-Ask strict | Oracle strict | No-Ask latent | "
            "Oracle latent | No-Ask type | Oracle type |",
            "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for pair in report["pairs"]:
        no_ask = pair["no_ask"]
        oracle = pair["oracle_ask"]
        lines.append(
            f"| {pair['task_id']} | {pair['field']} | "
            f"{int(no_ask['strict_success'])} | {int(oracle['strict_success'])} | "
            f"{int(no_ask['latent_satisfied'])} | {int(oracle['latent_satisfied'])} | "
            f"{no_ask['reward_type']} | {oracle['reward_type']} |"
        )
    return "\n".join(lines) + "\n"


def _arm_summary(
    records: list[dict[str, Any]], tasks_by_id: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    count = len(records)
    strict_count = sum(strict_success(record) for record in records)
    wrong_count = sum(wrong_purchase(record) for record in records)
    latent_count = sum(
        latent_goal_satisfied(tasks_by_id[int(record["task_id"])], record)
        for record in records
    )
    reward_types = Counter(_reward_type(record) for record in records)
    step_count = sum(len(record.get("steps") or []) for record in records)
    return {
        "paired_record_count": count,
        "strict_success_count": strict_count,
        "strict_success_rate": strict_count / count if count else 0.0,
        "latent_satisfaction_count": latent_count,
        "latent_satisfaction_rate": latent_count / count if count else 0.0,
        "wrong_purchase_count": wrong_count,
        "wrong_purchase_rate": wrong_count / count if count else 0.0,
        "average_steps": step_count / count if count else 0.0,
        "reward_type_counts": dict(sorted(reward_types.items())),
    }


def _row_view(task: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strict_success": strict_success(record),
        "latent_satisfied": latent_goal_satisfied(task, record),
        "wrong_purchase": wrong_purchase(record),
        "reward_type": _reward_type(record),
        "status": record.get("status"),
        "steps": len(record.get("steps") or []),
    }


def _terminal(record: Mapping[str, Any]) -> Mapping[str, Any]:
    terminal = record.get("terminal_result")
    return terminal if isinstance(terminal, Mapping) else {}


def _reward_detail(record: Mapping[str, Any]) -> Mapping[str, Any]:
    detail = _terminal(record).get("reward_detail")
    return detail if isinstance(detail, Mapping) else {}


def _reward_type(record: Mapping[str, Any]) -> str:
    return str(_reward_detail(record).get("reward_type") or record.get("status") or "unknown")


def _record_valid(record: Mapping[str, Any]) -> bool:
    probe = record.get("probe")
    return isinstance(probe, Mapping) and probe.get("valid") is True


def _percent(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Probe B V2 paired metrics")
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if bool(args.run_id) == bool(args.run_dir):
        print("error: provide exactly one of --run-id or --run-dir", file=sys.stderr)
        return 2
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.outdir) / args.run_id
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        print(f"error: missing {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = load_tasks(args.tasks)
    validation = validate_tasks(tasks)
    if manifest.get("task_hash") != validation["task_hash"]:
        print("error: manifest task_hash differs from current V2 task file", file=sys.stderr)
        return 2
    records = load_records(run_dir / TRAJECTORIES_FILE)
    try:
        report = analyze(tasks, records, mode=str(manifest.get("mode") or "unknown"))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report["run_id"] = manifest.get("run_id")
    write_json(run_dir / "metrics_summary.json", report)
    markdown = render_markdown(report)
    (run_dir / "metrics_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"-> {run_dir / 'metrics_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
