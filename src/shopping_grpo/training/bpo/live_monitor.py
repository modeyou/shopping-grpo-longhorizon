"""Read-only live diagnostics for an active CARL-BPO run."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "shopping-bpo-live-monitor-v1"


def _distribution(values: Iterable[int | float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
            "max_share": None,
            "coefficient_of_variation": None,
            "hhi": None,
        }
    total = sum(ordered)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    mean = fmean(ordered)
    return {
        "count": len(ordered),
        "total": total,
        "mean": mean,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "max_share": ordered[-1] / total if total > 0 else None,
        "coefficient_of_variation": (
            pstdev(ordered) / mean if len(ordered) > 1 and mean > 0 else 0.0
        ),
        "hhi": (
            sum((value / total) ** 2 for value in ordered) if total > 0 else None
        ),
    }


def _counter_summary(counter: Counter[object]) -> dict[str, Any]:
    uses = list(counter.values())
    total = sum(uses)
    unique = len(counter)
    sum_squares = sum(value * value for value in uses)
    return {
        "groups": total,
        "unique": unique,
        "repeated_groups": max(total - unique, 0),
        "repeat_rate": (1.0 - unique / total) if total else None,
        "maximum_uses": max(uses, default=0),
        "effective_count": (total * total / sum_squares) if sum_squares else 0.0,
    }


def _normalize_uid(value: object) -> str:
    return str(value)


def _task_key(rollouts: Sequence[Mapping[str, object]]) -> tuple[object, str, str]:
    first = rollouts[0]
    task_id = first.get("task_id", "missing")
    condition = str(first.get("interaction_mode", "unknown"))
    group_type = str(first.get("bpo_group_type", "unknown"))
    stage = str(first.get("bpo_local_stage", "root"))
    return task_id, condition, stage if group_type == "local" else "root"


def _span_proxy(rollout: Mapping[str, object]) -> dict[str, Any]:
    starts = [int(value) for value in rollout.get("bpo_action_token_starts") or []]
    ends = [int(value) for value in rollout.get("bpo_action_token_ends") or []]
    if not starts or len(starts) != len(ends):
        return {"available": False, "reason": "action_token_ends_not_recorded"}
    lengths = []
    for start, end in zip(starts, ends, strict=True):
        if start < 0 or end <= start:
            return {"available": False, "reason": "invalid_action_boundaries"}
        lengths.append(end - start)
    group_type = str(rollout.get("bpo_group_type", "unknown"))
    branch_action = int(rollout.get("bpo_branch_action", -1))
    selected = list(range(len(lengths))) if group_type == "root" else [branch_action]
    if any(index < 0 or index >= len(lengths) for index in selected):
        return {"available": False, "reason": "branch_action_outside_boundaries"}
    actions = [
        {
            "action_index": index,
            "span_tokens": length,
            "selected_for_policy": index in selected,
        }
        for index, length in enumerate(lengths)
    ]
    return {
        "available": True,
        "actions": actions,
        "selected_span_tokens": sum(lengths[index] for index in selected),
        "all_span_tokens": sum(lengths),
    }


@dataclass
class BpoLiveMonitor:
    """Incrementally aggregate append-only BPO diagnostics."""

    sibling_count: int = 4
    latest_step: int = 0
    generation_by_step: dict[int, dict[str, list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    candidate_seen: set[tuple[int, str]] = field(default_factory=set)
    selected_seen: set[tuple[int, str]] = field(default_factory=set)
    candidate_tasks: Counter[object] = field(default_factory=Counter)
    candidate_task_condition_stage: Counter[object] = field(default_factory=Counter)
    selected_tasks: Counter[object] = field(default_factory=Counter)
    selected_task_condition_stage: Counter[object] = field(default_factory=Counter)
    accepted_groups: int = 0
    selected_contrasts: Counter[str] = field(default_factory=Counter)
    selected_stages: Counter[str] = field(default_factory=Counter)
    slow_warnings: int = 0
    full_batch_timeouts: int = 0
    skipped_updates: int = 0
    optimizer_rejections: int = 0
    actor_batches: int = 0
    actor_all_finite: bool = True
    active_token_ratios: list[float] = field(default_factory=list)
    proxy_action_lengths: list[int] = field(default_factory=list)
    proxy_sibling_lengths: list[int] = field(default_factory=list)
    proxy_group_lengths: list[int] = field(default_factory=list)
    proxy_missing_rollouts: int = 0
    exact_action_lengths: list[int] = field(default_factory=list)
    exact_sibling_lengths: list[int] = field(default_factory=list)
    exact_group_lengths: list[int] = field(default_factory=list)
    exact_group_count: int = 0
    entropy_probe_types: Counter[str] = field(default_factory=Counter)
    entropy_probe_missing_groups: int = 0
    latest_selected_groups: list[dict[str, Any]] = field(default_factory=list)
    latest_exact_token_groups: list[dict[str, Any]] = field(default_factory=list)

    def consume(self, record: Mapping[str, object]) -> None:
        event = str(record.get("event", ""))
        step = int(record.get("global_step", -1))
        if step >= 0:
            self.latest_step = max(self.latest_step, step)
        if event == "generation_batch":
            self._consume_generation(step, record)
        elif event == "optimizer_selection":
            self._consume_selection(step, record)
        elif event == "bpo_actor_batch":
            self._consume_actor(record)
        elif event == "slow_full_batch_warning":
            self.slow_warnings += 1
        elif event == "full_batch_timeout":
            self.full_batch_timeouts += 1
        elif event == "skipped_update":
            self.skipped_updates += 1
        elif event == "bpo_optimizer_backward":
            if not bool((record.get("audit") or {}).get("accepted")):
                self.optimizer_rejections += 1

    def _consume_generation(self, step: int, record: Mapping[str, object]) -> None:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in record.get("rollouts") or []:
            rollout = dict(raw)
            grouped.setdefault(_normalize_uid(rollout.get("uid")), []).append(rollout)
        step_groups = self.generation_by_step.setdefault(step, {})
        for uid, rollouts in grouped.items():
            step_groups[uid] = rollouts
            identity = (step, uid)
            if identity in self.candidate_seen:
                continue
            self.candidate_seen.add(identity)
            key = _task_key(rollouts)
            self.candidate_tasks[key[0]] += 1
            self.candidate_task_condition_stage[key] += 1

    def _consume_selection(self, step: int, record: Mapping[str, object]) -> None:
        selected = record.get("selected_groups") or {}
        step_groups = self.generation_by_step.get(step, {})
        latest = []
        for group_type in ("root", "local"):
            detail = dict(selected.get(group_type) or {})
            uid = _normalize_uid(detail.get("uid"))
            identity = (step, uid)
            if not uid or identity in self.selected_seen:
                continue
            self.selected_seen.add(identity)
            self.accepted_groups += 1
            contrast = str(detail.get("contrast_type", "unknown"))
            stage = str(detail.get("local_stage", "root"))
            self.selected_contrasts[contrast] += 1
            self.selected_stages[stage] += 1
            rollouts = step_groups.get(uid, [])
            group_snapshot = {
                "step": step,
                "uid": uid,
                "group_type": group_type,
                "contrast_type": contrast,
                "stage": stage,
                "siblings": [],
            }
            if rollouts:
                key = _task_key(rollouts)
                self.selected_tasks[key[0]] += 1
                self.selected_task_condition_stage[key] += 1
                group_snapshot["task_id"] = key[0]
                group_total = 0
                probe_types = {
                    str(item.get("bpo_entropy_probe_argmax_token_type"))
                    for item in rollouts
                    if item.get("bpo_entropy_probe_argmax_token_type")
                }
                if probe_types:
                    for token_type in probe_types:
                        self.entropy_probe_types[token_type] += 1
                    group_snapshot["entropy_probe_token_types"] = sorted(probe_types)
                elif group_type == "local":
                    self.entropy_probe_missing_groups += 1
                for rollout in sorted(
                    rollouts, key=lambda item: int(item.get("bpo_sibling_index", 0))
                ):
                    proxy = _span_proxy(rollout)
                    sibling = {
                        "sibling_index": int(rollout.get("bpo_sibling_index", 0)),
                        "span_proxy": proxy,
                    }
                    group_snapshot["siblings"].append(sibling)
                    if not proxy["available"]:
                        self.proxy_missing_rollouts += 1
                        continue
                    selected_lengths = [
                        int(action["span_tokens"])
                        for action in proxy["actions"]
                        if action["selected_for_policy"]
                    ]
                    self.proxy_action_lengths.extend(selected_lengths)
                    selected_total = int(proxy["selected_span_tokens"])
                    self.proxy_sibling_lengths.append(selected_total)
                    group_total += selected_total
                if group_total > 0:
                    self.proxy_group_lengths.append(group_total)
            latest.append(group_snapshot)
        if latest:
            self.latest_selected_groups = latest
        self.generation_by_step.pop(step, None)

    def _consume_actor(self, record: Mapping[str, object]) -> None:
        self.actor_batches += 1
        diagnostics = record.get("diagnostics") or {}
        self.actor_all_finite = self.actor_all_finite and bool(
            diagnostics.get("all_finite", False)
        )
        metrics = record.get("metrics") or {}
        ratio = metrics.get("bpo_action/active_token_ratio")
        if ratio is not None:
            self.active_token_ratios.append(float(ratio))
        exact = record.get("token_mass") or {}
        groups = exact.get("groups") or []
        if groups:
            self.latest_exact_token_groups = [dict(group) for group in groups]
        for group in groups:
            group_total = 0
            for sibling in group.get("siblings") or []:
                sibling_total = 0
                for action in sibling.get("actions") or []:
                    if not bool(action.get("selected_for_policy")):
                        continue
                    count = int(action.get("actor_tokens", 0))
                    self.exact_action_lengths.append(count)
                    sibling_total += count
                self.exact_sibling_lengths.append(sibling_total)
                group_total += sibling_total
            self.exact_group_lengths.append(group_total)
            self.exact_group_count += 1

    def snapshot(self) -> dict[str, Any]:
        accepted_outcomes = self.accepted_groups * int(self.sibling_count)
        exact_available = self.exact_group_count > 0
        probe_available = bool(self.entropy_probe_types)
        return {
            "schema_version": SCHEMA_VERSION,
            "progress": {
                "latest_generation_step": self.latest_step,
                "actor_batches": self.actor_batches,
            },
            "sampling": {
                "accepted_groups_total": self.accepted_groups,
                "accepted_sibling_terminal_outcomes_total": accepted_outcomes,
                "selected_contrasts": dict(sorted(self.selected_contrasts.items())),
                "selected_stages": dict(sorted(self.selected_stages.items())),
            },
            "tasks": {
                "unit": (
                    "one generated group; K siblings are intentionally excluded "
                    "from repetition"
                ),
                "candidate_task": _counter_summary(self.candidate_tasks),
                "candidate_task_condition_stage": _counter_summary(
                    self.candidate_task_condition_stage
                ),
                "selected_task": _counter_summary(self.selected_tasks),
                "selected_task_condition_stage": _counter_summary(
                    self.selected_task_condition_stage
                ),
            },
            "token_mass": {
                "exact_actor_tokens_available": exact_available,
                "exact_unavailable_reason": (
                    None
                    if exact_available
                    else "running trainer did not record per-action actor-token counts"
                ),
                "exact_selected_action_actor_tokens": _distribution(
                    self.exact_action_lengths
                ),
                "exact_selected_sibling_actor_tokens": _distribution(
                    self.exact_sibling_lengths
                ),
                "exact_selected_group_actor_tokens": _distribution(
                    self.exact_group_lengths
                ),
                "span_proxy_warning": (
                    "end-start includes non-actor tool/environment tokens; it is not gradient mass"
                ),
                "selected_action_span_tokens": _distribution(
                    self.proxy_action_lengths
                ),
                "selected_sibling_span_tokens": _distribution(
                    self.proxy_sibling_lengths
                ),
                "selected_group_span_tokens": _distribution(
                    self.proxy_group_lengths
                ),
                "span_proxy_missing_rollouts": self.proxy_missing_rollouts,
                "batch_active_action_token_ratio": _distribution(
                    self.active_token_ratios
                ),
                "latest_exact_groups": self.latest_exact_token_groups,
            },
            "entropy_probe": {
                "token_type_available": probe_available,
                "unavailable_reason": (
                    None
                    if probe_available
                    else "running trainer only recorded entropy scalar, not probe token identity"
                ),
                "selected_local_group_token_types": dict(
                    sorted(self.entropy_probe_types.items())
                ),
                "selected_local_groups_missing_token_type": (
                    self.entropy_probe_missing_groups
                ),
            },
            "alerts": {
                "slow_full_batch_warning_total": self.slow_warnings,
                "full_batch_timeout_total": self.full_batch_timeouts,
                "skipped_update_total": self.skipped_updates,
                "optimizer_rejection_total": self.optimizer_rejections,
                "actor_finite_available": self.actor_batches > 0,
                "actor_all_finite": (
                    self.actor_all_finite if self.actor_batches > 0 else None
                ),
                "blocking": bool(
                    self.full_batch_timeouts
                    or self.skipped_updates
                    or self.optimizer_rejections
                    or (self.actor_batches > 0 and not self.actor_all_finite)
                ),
            },
            "latest_selected_groups": self.latest_selected_groups,
        }


def aggregate_records(records: Iterable[Mapping[str, object]]) -> dict[str, Any]:
    monitor = BpoLiveMonitor()
    for record in records:
        monitor.consume(record)
    return monitor.snapshot()


def read_complete_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read an append-only JSONL snapshot, ignoring only an incomplete final line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records = []
    import json

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records
