"""Read-only SwanLab history normalization and compact training analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import median
from typing import Any


DECISION_STEPS = (0, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
IMPORTANT_PREFIXES = (
    "validation/", "sampling/", "credit/", "optimization/", "runtime/",
    "summary/", "val-shopping/", "val-core/", "bpo_", "bpo-", "carl_", "carl-", "group/",
    "rollout/", "actor/", "critic/", "training/", "perf/", "timing_",
    "response_length/",
)
IMPORTANT_FRAGMENTS = (
    "loss", "learning_rate", "grad_norm", "entropy", "kl", "clip", "reward",
    "strict", "purchase", "effective_tree", "effective_return", "generated_total",
    "trained_total", "seconds_to_",
)
SYSTEM_FRAGMENTS = (
    "gpu", "cuda", "memory", "vram", "utilization", "cpu", "disk", "network",
)


@dataclass(frozen=True)
class MetricPoint:
    step: int
    value: float
    timestamp: float | None = None


def json_safe(value: Any) -> Any:
    """Convert SwanLab entity/response objects to JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    json_method = getattr(value, "json", None)
    if callable(json_method):
        return json_safe(json_method())
    data = getattr(value, "data", None)
    if data is not None:
        return json_safe(data)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict(orient="records"))
        except TypeError:
            return json_safe(to_dict())
    return str(value)


def metric_key(item: Any) -> str:
    key = getattr(item, "key", None)
    if key is None and isinstance(item, Mapping):
        key = item.get("key") or item.get("name")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"SwanLab series item has no metric key: {item!r}")
    return key.strip()


def is_important_key(key: str, *, system: bool = False) -> bool:
    lowered = key.lower()
    if system:
        return any(fragment in lowered for fragment in SYSTEM_FRAGMENTS)
    return lowered.startswith(IMPORTANT_PREFIXES) or any(
        fragment in lowered for fragment in IMPORTANT_FRAGMENTS
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    numeric = _number(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _record_point(record: Mapping[str, Any]) -> MetricPoint | None:
    step = next((_integer(record.get(name)) for name in ("step", "global_step", "index", "x") if name in record), None)
    value = next((_number(record.get(name)) for name in ("value", "scalar", "y") if name in record), None)
    if step is None or value is None:
        return None
    timestamp = next((_number(record.get(name)) for name in ("timestamp", "created_at", "createdAt") if name in record), None)
    return MetricPoint(step=step, value=value, timestamp=timestamp)


def extract_metric_points(payload: Any, key: str) -> list[MetricPoint]:
    """Extract one key's points from common SwanLab structured response shapes."""
    safe = json_safe(payload)
    points: list[MetricPoint] = []

    def visit(value: Any, active_key: str | None = None) -> None:
        if isinstance(value, Mapping):
            declared = value.get("key") or value.get("name") or value.get("metric")
            declared_key = declared if isinstance(declared, str) else active_key
            point = _record_point(value)
            if point is not None and (declared_key is None or declared_key == key):
                points.append(point)
            for child_key, child in value.items():
                visit(child, key if child_key == key else declared_key)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) in (2, 3):
                step = _integer(value[0])
                numeric = _number(value[1])
                if step is not None and numeric is not None and active_key == key:
                    timestamp = _number(value[2]) if len(value) == 3 else None
                    points.append(MetricPoint(step, numeric, timestamp))
                    return
            for child in value:
                visit(child, active_key)

    visit(safe)
    unique = {(point.step, point.value, point.timestamp): point for point in points}
    return sorted(unique.values(), key=lambda point: (point.step, point.timestamp or 0.0))


def serialize_points(series: Mapping[str, Sequence[MetricPoint]]) -> dict[str, list[dict]]:
    return {key: [asdict(point) for point in points] for key, points in series.items()}


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.6g}"


def _priority(key: str) -> tuple[int, str]:
    ordered = (
        "validation/gold_purchase_success", "validation/completion_success",
        "validation/reward_mean", "validation/terminal_utility_mean",
        "validation/done_rate", "validation/average_steps",
        "validation/sampling_invalid_rate",
        "validation/infrastructure_invalid_rate",
        "validation/reward_unverifiable_rate",
        "validation/shopper_question_rate",
        "summary/strict_success_rate", "summary/purchase_success_rate",
        "summary/mean_reward", "summary/terminal_utility_mean", "summary/done_rate",
        "summary/average_steps", "summary/sampling_invalid_rate",
        "summary/infrastructure_invalid_rate", "summary/reward_unverifiable_rate",
        "summary/shopper_question_rate",
    )
    try:
        return ordered.index(key), key
    except ValueError:
        return len(ordered), key


def _training_alerts(series: Mapping[str, Sequence[MetricPoint]]) -> list[str]:
    alerts: list[str] = []
    for key, points in series.items():
        if not points:
            continue
        values = [point.value for point in points]
        lowered = key.lower()
        if "grad_norm" in lowered:
            baseline = max(abs(median(values)), 1e-12)
            if max(abs(value) for value in values) > max(100.0, baseline * 10):
                alerts.append(f"`{key}` 出现明显梯度尖峰。")
        if "kl" in lowered and max(abs(value) for value in values) > 1.0:
            alerts.append(f"`{key}` 的绝对值超过 1，需要核对 KL 定义与策略漂移。")
        if "loss" in lowered and max(abs(value) for value in values) > 100:
            alerts.append(f"`{key}` 的绝对值超过 100，需要检查异常 batch。")
    return alerts


def build_markdown_report(*, run_path: str, run_metadata: Mapping[str, Any], custom_series: Mapping[str, Sequence[MetricPoint]], system_series: Mapping[str, Sequence[MetricPoint]]) -> str:
    """Build a deterministic Chinese report from complete metric histories."""
    lines = [
        "# SwanLab BPO 训练轨迹分析", "", f"- 运行路径：`{run_path}`",
        f"- 运行名称：`{run_metadata.get('name', '未知')}`",
        f"- 状态：`{run_metadata.get('state', '未知')}`",
        f"- 创建时间：`{run_metadata.get('created_at', '未知')}`",
        f"- 结束时间：`{run_metadata.get('finished_at', '未知')}`",
        f"- 导出时间：`{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- 自定义指标：{len(custom_series)} 个；系统指标：{len(system_series)} 个",
        "", "## 决策节点验证曲线", "",
    ]
    validation = {
        key: points
        for key, points in custom_series.items()
        if (
            key.startswith("validation/")
            and not key.startswith("validation/step0_")
            or key.startswith("val-shopping/summary/")
        )
        and points
    }
    if validation:
        lines.append("| 指标 | " + " | ".join(f"step {step}" for step in DECISION_STEPS) + " |")
        lines.append("|---|" + "---:|" * len(DECISION_STEPS))
        for key in sorted(
            validation, key=lambda name: _priority(name.removeprefix("val-shopping/"))
        ):
            by_step = {point.step: point for point in validation[key]}
            values = [_fmt(by_step.get(step).value if step in by_step else None) for step in DECISION_STEPS]
            lines.append(f"| `{key}` | " + " | ".join(values) + " |")
    else:
        lines.append(
            "未从响应中解析出 `validation/*` 曲线；"
            "原始响应已保存在 JSON 快照中。"
        )

    lines.extend(["", "## 重要训练指标", ""])
    training = {
        key: points
        for key, points in custom_series.items()
        if points
        and not key.startswith(("validation/", "summary/", "val-shopping/", "val-core/"))
    }
    if training:
        lines.extend(["| 指标 | 点数 | 首值 | 末值 | 最小值 | 最大值 |", "|---|---:|---:|---:|---:|---:|"])
        for key in sorted(training):
            values = [point.value for point in training[key]]
            lines.append(f"| `{key}` | {len(values)} | {_fmt(values[0])} | {_fmt(values[-1])} | {_fmt(min(values))} | {_fmt(max(values))} |")
    else:
        lines.append("未解析出重要训练指标。")

    lines.extend(["", "## GPU 与系统资源", ""])
    if system_series:
        lines.extend(["| 指标 | 点数 | 平均值 | 最大值 |", "|---|---:|---:|---:|"])
        for key in sorted(system_series):
            values = [point.value for point in system_series[key]]
            if values:
                lines.append(f"| `{key}` | {len(values)} | {_fmt(sum(values) / len(values))} | {_fmt(max(values))} |")
    else:
        lines.append("未解析出 SwanLab 系统资源曲线。")

    alerts = _training_alerts(training)
    lines.extend(["", "## 自动检查结论", ""])
    lines.extend(f"- {alert}" for alert in alerts)
    if not alerts:
        lines.append("- 未发现极端 KL、梯度尖峰或超大 loss 的规则型信号。")
    lines.append("- 自动规则只能用于定位异常；算法结论仍应以冻结 dev500 三面板与逐任务配对审计为准。")
    lines.append("")
    return "\n".join(lines)
