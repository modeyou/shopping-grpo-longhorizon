"""Three-condition metrics for the Probe B V3 Autonomous-Ask pilot."""

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

from autonomous_runner import (  # noqa: E402
    AUTONOMOUS_ARM,
    DEFAULT_OUTPUT_ROOT,
    MANIFEST_FILE,
    SELECTED_TASK_IDS,
    TRAJECTORIES_FILE,
    load_records,
    select_pilot_tasks,
    write_json,
)
from metrics import strict_success, wrong_purchase  # noqa: E402
from task_schema import (  # noqa: E402
    DEFAULT_TASKS,
    canonical_hash,
    latent_goal_satisfied,
    load_tasks,
    validate_tasks,
)


REFERENCE_ARMS = ("no_ask", "oracle_ask")
ALL_ARMS = ("no_ask", AUTONOMOUS_ARM, "oracle_ask")
EXPECTED_TASKS = 10
MIN_CORRECT_ASKS = 7
MIN_AUTONOMOUS_STRICT = 8


def load_reference_records(
    run_dir: Path, expected_task_hash: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise ValueError(f"missing V2 reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task_hash") != expected_task_hash:
        raise ValueError("V2 reference task_hash differs from current task file")
    records = load_records(run_dir / TRAJECTORIES_FILE)
    by_key: dict[tuple[int, str], int] = {}
    for index, record in enumerate(records):
        key = (int(record["task_id"]), str(record.get("arm")))
        if key in by_key:
            raise ValueError(f"duplicate V2 reference record: {key[0]}:{key[1]}")
        _require_record_task_hash(record, expected_task_hash)
        by_key[key] = index

    supplemental_path = run_dir / "supplemental_18637_oracle_ask.json"
    audit = {
        "reference_run_id": manifest.get("run_id"),
        "reference_manifest_hash": canonical_hash(manifest),
        "supplemental_path": str(supplemental_path),
        "supplemental_used": False,
        "replaced_key": None,
    }
    if supplemental_path.exists():
        payload = json.loads(supplemental_path.read_text(encoding="utf-8"))
        supplement = payload.get("trajectory") if isinstance(payload, Mapping) else None
        if not isinstance(supplement, Mapping):
            supplement = payload
        if not isinstance(supplement, Mapping) or "task_id" not in supplement:
            raise ValueError("supplemental record must contain one trajectory")
        supplement = dict(supplement)
        key = (int(supplement["task_id"]), str(supplement.get("arm")))
        if key != (18637, "oracle_ask"):
            raise ValueError(
                "supplemental record key must be 18637:oracle_ask, got "
                f"{key[0]}:{key[1]}"
            )
        if key not in by_key:
            raise ValueError("supplemental record has no matching primary record")
        primary = records[by_key[key]]
        if _record_valid(primary):
            raise ValueError("supplemental record may only replace an invalid primary")
        if not _record_valid(supplement):
            raise ValueError("supplemental record must itself be valid")
        _require_record_task_hash(supplement, expected_task_hash)
        records[by_key[key]] = supplement
        audit["supplemental_used"] = True
        audit["replaced_key"] = "18637:oracle_ask"
    return records, audit


def analyze(
    tasks: list[dict[str, Any]],
    autonomous_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    *,
    mode: str = "real",
) -> dict[str, Any]:
    tasks_by_id = {int(task["task_id"]): task for task in tasks}
    expected_ids = set(tasks_by_id)
    if tuple(int(task["task_id"]) for task in tasks) != SELECTED_TASK_IDS:
        raise ValueError("metrics tasks do not match the frozen V3 selection and order")

    reference_by_key = _unique_records(reference_records, "V2 reference")
    autonomous_by_key = _unique_records(autonomous_records, "V3 autonomous")
    unexpected_auto = sorted(
        f"{task_id}:{arm}"
        for task_id, arm in autonomous_by_key
        if task_id not in expected_ids or arm != AUTONOMOUS_ARM
    )
    if unexpected_auto:
        raise ValueError("unexpected V3 records: " + ", ".join(unexpected_auto))

    valid_ids = []
    rows = []
    for task in tasks:
        task_id = int(task["task_id"])
        keys = [
            (task_id, "no_ask"),
            (task_id, AUTONOMOUS_ARM),
            (task_id, "oracle_ask"),
        ]
        records = [
            reference_by_key.get(keys[0]),
            autonomous_by_key.get(keys[1]),
            reference_by_key.get(keys[2]),
        ]
        if all(record is not None and _record_valid(record) for record in records):
            valid_ids.append(task_id)
            rows.append(
                {
                    "task_id": task_id,
                    "field": task["latent_goal"]["field"],
                    "no_ask": _row_view(task, records[0]),
                    AUTONOMOUS_ARM: _row_view(task, records[1]),
                    "oracle_ask": _row_view(task, records[2]),
                }
            )

    valid_set = set(valid_ids)
    arm_records = {
        "no_ask": [reference_by_key[(task_id, "no_ask")] for task_id in valid_ids],
        AUTONOMOUS_ARM: [
            autonomous_by_key[(task_id, AUTONOMOUS_ARM)] for task_id in valid_ids
        ],
        "oracle_ask": [
            reference_by_key[(task_id, "oracle_ask")] for task_id in valid_ids
        ],
    }
    summaries = {
        arm: _arm_summary(records, tasks_by_id) for arm, records in arm_records.items()
    }
    auto_valid = arm_records[AUTONOMOUS_ARM]
    classifications = Counter(_ask_classification(record) for record in auto_valid)
    correct_and_strict = sum(
        _ask_classification(record) == "correct_ask" and strict_success(record)
        for record in auto_valid
    )
    ask_metrics = {
        "ask_count": classifications["correct_ask"] + classifications["incorrect_ask"],
        "ask_rate": (
            (classifications["correct_ask"] + classifications["incorrect_ask"])
            / len(auto_valid)
            if auto_valid
            else 0.0
        ),
        "correct_ask_count": classifications["correct_ask"],
        "correct_ask_rate": (
            classifications["correct_ask"] / len(auto_valid) if auto_valid else 0.0
        ),
        "incorrect_ask_count": classifications["incorrect_ask"],
        "no_ask_count": classifications["no_ask"],
        "correct_ask_and_strict_success_count": correct_and_strict,
        "classification_counts": dict(sorted(classifications.items())),
        "by_field": _ask_by_field(auto_valid, tasks_by_id),
    }

    no_strict = summaries["no_ask"]["strict_success_count"]
    auto_strict = summaries[AUTONOMOUS_ARM]["strict_success_count"]
    oracle_strict = summaries["oracle_ask"]["strict_success_count"]
    full = len(valid_ids) == EXPECTED_TASKS
    if mode == "real" and full and (no_strict, oracle_strict) != (5, 10):
        raise ValueError(
            "selected V2 reference strict counts differ from the frozen 5/10 and 10/10"
        )
    gap = oracle_strict - no_strict
    recovered = auto_strict - no_strict
    recovery_rate = recovered / gap if gap else None
    conditions = {
        "correct_asks_at_least_7": ask_metrics["correct_ask_count"] >= MIN_CORRECT_ASKS,
        "autonomous_strict_at_least_8": auto_strict >= MIN_AUTONOMOUS_STRICT,
        "wrong_purchase_zero": summaries[AUTONOMOUS_ARM]["wrong_purchase_count"] == 0,
    }
    if mode != "real":
        gate_status = "NOT_APPLICABLE_MOCK"
    elif not full:
        gate_status = "INCOMPLETE"
    else:
        gate_status = "PASS" if all(conditions.values()) else "FAIL"

    attempted_auto_ids = {
        task_id
        for task_id, arm in autonomous_by_key
        if arm == AUTONOMOUS_ARM and task_id in expected_ids
    }
    invalid_auto_ids = sorted(
        task_id
        for (task_id, arm), record in autonomous_by_key.items()
        if arm == AUTONOMOUS_ARM
        and task_id in expected_ids
        and not _record_valid(record)
    )
    missing_reference = sorted(
        f"{task_id}:{arm}"
        for task_id in expected_ids
        for arm in REFERENCE_ARMS
        if (task_id, arm) not in reference_by_key
        or not _record_valid(reference_by_key[(task_id, arm)])
    )
    return {
        "mode": mode,
        "expected_task_count": EXPECTED_TASKS,
        "valid_three_condition_task_count": len(valid_ids),
        "valid_task_ids": valid_ids,
        "missing_autonomous_task_ids": sorted(expected_ids - attempted_auto_ids),
        "invalid_autonomous_task_ids": invalid_auto_ids,
        "missing_or_invalid_reference_keys": missing_reference,
        "conditions": summaries,
        "ask_metrics": ask_metrics,
        "oracle_gap": {
            "no_ask_strict_count": no_strict,
            "autonomous_strict_count": auto_strict,
            "oracle_strict_count": oracle_strict,
            "gap_count": gap,
            "recovered_count": recovered,
            "recovery_rate": recovery_rate,
        },
        "continue_gate": {
            "status": gate_status,
            "evaluable": mode == "real" and full,
            "conditions": conditions,
        },
        "tasks": rows,
        "unused_valid_task_ids": sorted(valid_set - expected_ids),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summaries = report["conditions"]
    asks = report["ask_metrics"]
    gap = report["oracle_gap"]
    gate = report["continue_gate"]
    lines = [
        "# Probe B V3 Autonomous-Ask Pilot 结果",
        "",
        f"- 模式：`{report['mode']}`",
        (
            "- 三条件完整有效任务："
            f"{report['valid_three_condition_task_count']} / {report['expected_task_count']}"
        ),
        f"- 判据状态：**{gate['status']}**",
        f"- V2 第 51 次补充记录：{'已使用' if report.get('reference_audit', {}).get('supplemental_used') else '未使用'}",
        "",
        "## 三条件汇总",
        "",
        "| 指标 | No-Ask | Autonomous-Ask | Oracle-Ask |",
        "|---|---:|---:|---:|",
        (
            "| 严格成功 | "
            f"{_count_rate(summaries['no_ask'], 'strict_success')} | "
            f"{_count_rate(summaries[AUTONOMOUS_ARM], 'strict_success')} | "
            f"{_count_rate(summaries['oracle_ask'], 'strict_success')} |"
        ),
        (
            "| 隐藏字段满足 | "
            f"{_count_rate(summaries['no_ask'], 'latent_satisfaction')} | "
            f"{_count_rate(summaries[AUTONOMOUS_ARM], 'latent_satisfaction')} | "
            f"{_count_rate(summaries['oracle_ask'], 'latent_satisfaction')} |"
        ),
        (
            "| 真实 wrong_purchase | "
            f"{summaries['no_ask']['wrong_purchase_count']} | "
            f"{summaries[AUTONOMOUS_ARM]['wrong_purchase_count']} | "
            f"{summaries['oracle_ask']['wrong_purchase_count']} |"
        ),
        (
            "| 平均环境步数 | "
            f"{summaries['no_ask']['average_steps']:.2f} | "
            f"{summaries[AUTONOMOUS_ARM]['average_steps']:.2f} | "
            f"{summaries['oracle_ask']['average_steps']:.2f} |"
        ),
        "",
        "## 自主提问",
        "",
        f"- 发起提问：{asks['ask_count']} / {report['valid_three_condition_task_count']}（{_percent(asks['ask_rate'])}）",
        f"- 正确提问：{asks['correct_ask_count']}（{_percent(asks['correct_ask_rate'])}）",
        f"- 错误提问：{asks['incorrect_ask_count']}",
        f"- 未提问：{asks['no_ask_count']}",
        f"- 正确提问且严格成功：{asks['correct_ask_and_strict_success_count']}",
        "",
        "## Oracle 差距恢复",
        "",
        f"- No-Ask → Oracle 差距：{gap['gap_count']} 题",
        f"- Autonomous 相对 No-Ask 恢复：{gap['recovered_count']} 题",
        f"- 差距恢复率：{_percent(gap['recovery_rate'])}",
        "",
        "## 预登记继续门槛",
        "",
    ]
    labels = {
        "correct_asks_at_least_7": "正确提问至少 7/10",
        "autonomous_strict_at_least_8": "Autonomous 严格成功至少 8/10",
        "wrong_purchase_zero": "Autonomous 真实 wrong_purchase 为 0",
    }
    for key, label in labels.items():
        lines.append(f"- [{'PASS' if gate['conditions'][key] else 'FAIL'}] {label}")
    if not gate["evaluable"]:
        reason = (
            "当前是 Mock 结果，不适用真实实验 PASS/FAIL。"
            if report["mode"] != "real"
            else "当前不足 10 条完整有效三条件任务，不能做 PASS/FAIL 决策。"
        )
        lines.extend(["", reason])
    lines.extend(
        [
            "",
            "## 逐任务",
            "",
            "| task | field | ask | No strict | Auto strict | Oracle strict | No latent | Auto latent | Oracle latent |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["tasks"]:
        no_ask = row["no_ask"]
        autonomous = row[AUTONOMOUS_ARM]
        oracle = row["oracle_ask"]
        lines.append(
            f"| {row['task_id']} | {row['field']} | {autonomous['ask_classification']} | "
            f"{int(no_ask['strict_success'])} | {int(autonomous['strict_success'])} | "
            f"{int(oracle['strict_success'])} | {int(no_ask['latent_satisfied'])} | "
            f"{int(autonomous['latent_satisfied'])} | {int(oracle['latent_satisfied'])} |"
        )
    return "\n".join(lines) + "\n"


def _unique_records(
    records: list[dict[str, Any]], label: str
) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    for record in records:
        key = (int(record["task_id"]), str(record.get("arm")))
        if key in result:
            raise ValueError(f"duplicate {label} record: {key[0]}:{key[1]}")
        result[key] = record
    return result


def _arm_summary(
    records: list[dict[str, Any]], tasks_by_id: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    count = len(records)
    strict_count = sum(strict_success(record) for record in records)
    latent_count = sum(
        latent_goal_satisfied(tasks_by_id[int(record["task_id"])], record)
        for record in records
    )
    wrong_count = sum(wrong_purchase(record) for record in records)
    step_count = sum(len(record.get("steps") or []) for record in records)
    reward_types = Counter(_reward_type(record) for record in records)
    return {
        "record_count": count,
        "strict_success_count": strict_count,
        "strict_success_rate": strict_count / count if count else 0.0,
        "latent_satisfaction_count": latent_count,
        "latent_satisfaction_rate": latent_count / count if count else 0.0,
        "wrong_purchase_count": wrong_count,
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
        "ask_classification": _ask_classification(record),
    }


def _ask_by_field(
    records: list[dict[str, Any]], tasks_by_id: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    result = {}
    for field in ("budget", "color"):
        field_records = [
            record
            for record in records
            if tasks_by_id[int(record["task_id"])]["latent_goal"]["field"] == field
        ]
        counts = Counter(_ask_classification(record) for record in field_records)
        result[field] = {
            "task_count": len(field_records),
            "correct_ask_count": counts["correct_ask"],
            "incorrect_ask_count": counts["incorrect_ask"],
            "no_ask_count": counts["no_ask"],
        }
    return result


def _ask_classification(record: Mapping[str, Any]) -> str:
    probe = record.get("probe")
    ask = probe.get("autonomous_ask") if isinstance(probe, Mapping) else None
    value = ask.get("classification") if isinstance(ask, Mapping) else None
    return str(value or "no_ask")


def _reward_type(record: Mapping[str, Any]) -> str:
    terminal = record.get("terminal_result")
    detail = terminal.get("reward_detail") if isinstance(terminal, Mapping) else None
    if isinstance(detail, Mapping) and detail.get("reward_type"):
        return str(detail["reward_type"])
    return str(record.get("status") or "unknown")


def _record_valid(record: Mapping[str, Any]) -> bool:
    probe = record.get("probe")
    return isinstance(probe, Mapping) and probe.get("valid") is True


def _require_record_task_hash(record: Mapping[str, Any], expected: str) -> None:
    probe = record.get("probe")
    value = probe.get("task_hash") if isinstance(probe, Mapping) else None
    if value is not None and value != expected:
        raise ValueError("trajectory task_hash differs from V2 manifest")


def _count_rate(summary: Mapping[str, Any], prefix: str) -> str:
    return (
        f"{summary[prefix + '_count']}/{summary['record_count']} "
        f"({_percent(summary[prefix + '_rate'])})"
    )


def _percent(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Probe B V3 Autonomous-Ask metrics")
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    parser.add_argument("--reference-run-dir")
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
        print("error: V3 manifest task_hash differs from current task file", file=sys.stderr)
        return 2
    if tuple(manifest.get("selected_task_ids") or ()) != SELECTED_TASK_IDS:
        print("error: V3 manifest selected tasks differ from frozen design", file=sys.stderr)
        return 2
    selected = select_pilot_tasks(tasks, int(manifest.get("seed") or 0))
    reference_arg = args.reference_run_dir
    reference_manifest = manifest.get("reference")
    if not reference_arg and isinstance(reference_manifest, Mapping):
        reference_arg = reference_manifest.get("run_dir")
    if not reference_arg:
        print("error: provide --reference-run-dir for V2 results", file=sys.stderr)
        return 2
    try:
        references, audit = load_reference_records(
            Path(reference_arg), validation["task_hash"]
        )
        if isinstance(reference_manifest, Mapping):
            expected_reference_hash = reference_manifest.get("manifest_hash")
            if (
                expected_reference_hash
                and audit["reference_manifest_hash"] != expected_reference_hash
            ):
                raise ValueError(
                    "V2 reference manifest differs from the one frozen by the V3 run"
                )
        report = analyze(
            selected,
            load_records(run_dir / TRAJECTORIES_FILE),
            references,
            mode=str(manifest.get("mode") or "unknown"),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report["run_id"] = manifest.get("run_id")
    report["reference_audit"] = audit
    write_json(run_dir / "metrics_summary.json", report)
    markdown = render_markdown(report)
    (run_dir / "metrics_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"-> {run_dir / 'metrics_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
