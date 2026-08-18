"""Probe B V2 task schema, offline validation, and latent-slot checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = Path(__file__).resolve().parent / "data" / "tasks_v2.json"
SFT_TRAIN = REPO / "data" / "sft" / "train.jsonl"
EVALUATION_TASKS = REPO / "data" / "evaluation" / "tasks.jsonl"
EXPECTED_DISTRIBUTION = {"budget": 13, "color": 6, "brand": 3, "origin": 3}
ALLOWED_FIELDS = frozenset(EXPECTED_DISTRIBUTION)


class TaskValidationError(ValueError):
    """The frozen Probe B V2 task set violates one or more design invariants."""


def load_tasks(path: str | Path = DEFAULT_TASKS) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TaskValidationError("task file must contain a JSON list")
    return [dict(item) for item in payload]


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_tasks(
    tasks: Iterable[Mapping[str, Any]],
    *,
    sft_path: str | Path = SFT_TRAIN,
    evaluation_path: str | Path = EVALUATION_TASKS,
    expected_count: int | None = 25,
    expected_distribution: Mapping[str, int] | None = EXPECTED_DISTRIBUTION,
) -> dict[str, Any]:
    """Validate the full task set without contacting ShopSimulator or an LLM API."""

    rows = [dict(task) for task in tasks]
    source_queries = _source_queries(Path(sft_path))
    evaluation_ids = _task_ids(Path(evaluation_path))
    errors: list[str] = []
    seen_ids: set[int] = set()
    distribution: Counter[str] = Counter()

    if expected_count is not None and len(rows) != int(expected_count):
        errors.append(f"expected {expected_count} tasks, found {len(rows)}")

    for index, task in enumerate(rows):
        label = f"task[{index}]"
        try:
            task_id = int(task["task_id"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: task_id must be an integer")
            continue
        label = f"task_id={task_id}"
        if task_id in seen_ids:
            errors.append(f"{label}: duplicate task_id")
        seen_ids.add(task_id)
        if task_id in evaluation_ids:
            errors.append(f"{label}: overlaps data/evaluation/tasks.jsonl")
        if task_id not in source_queries:
            errors.append(f"{label}: not found in SFT train source")

        clear_query = _required_text(task, "clear_query", label, errors)
        under_query = _required_text(task, "under_query", label, errors)
        if clear_query and source_queries.get(task_id) != clear_query:
            errors.append(f"{label}: clear_query differs from the SFT source instruction")
        if clear_query and under_query and clear_query == under_query:
            errors.append(f"{label}: under_query does not hide anything")

        if "fake_profile" in task or "hidden_fields" in task:
            errors.append(f"{label}: legacy fake_profile/hidden_fields fields are forbidden")

        latent = task.get("latent_goal")
        if not isinstance(latent, Mapping):
            errors.append(f"{label}: latent_goal must be one object")
            continue
        field = str(latent.get("field") or "")
        if field not in ALLOWED_FIELDS:
            errors.append(f"{label}: unsupported latent field {field!r}")
            continue
        distribution[field] += 1

        raw_value = str(latent.get("raw_value") or "").strip()
        source_span = str(latent.get("source_span") or "").strip()
        if not raw_value:
            errors.append(f"{label}: latent_goal.raw_value is required")
        if not source_span or source_span not in clear_query:
            errors.append(f"{label}: source_span must occur verbatim in clear_query")
        if raw_value and raw_value not in source_span:
            errors.append(f"{label}: raw_value must occur inside source_span")

        leakage_terms = latent.get("leakage_terms")
        if not isinstance(leakage_terms, list) or not leakage_terms:
            errors.append(f"{label}: leakage_terms must be a non-empty list")
            leakage_terms = []
        for term in leakage_terms:
            if str(term).casefold() in under_query.casefold():
                errors.append(f"{label}: under_query leaks latent term {term!r}")

        normalized = latent.get("normalized_value")
        if not isinstance(normalized, Mapping):
            errors.append(f"{label}: normalized_value must be an object")
            normalized = {}
        if field == "budget":
            if normalized.get("operator") not in {"<=", "<"}:
                errors.append(f"{label}: budget operator must be '<=' or '<'")
            if not isinstance(normalized.get("amount"), (int, float)):
                errors.append(f"{label}: budget amount must be numeric")
            if normalized.get("currency") != "CNY":
                errors.append(f"{label}: budget currency must be CNY")
        elif not str(normalized.get("value") or "").strip():
            errors.append(f"{label}: normalized attribute value is required")

        oracle = task.get("oracle_turn")
        if not isinstance(oracle, Mapping):
            errors.append(f"{label}: oracle_turn must be an object")
            continue
        question = str(oracle.get("question") or "").strip()
        answer = str(oracle.get("answer") or "").strip()
        facts = oracle.get("answer_facts")
        if not question or not answer or not isinstance(facts, Mapping):
            errors.append(f"{label}: Oracle question, answer, and answer_facts are required")
        elif field == "budget":
            if facts.get("budget_max") != normalized.get("amount"):
                errors.append(f"{label}: Oracle budget fact disagrees with normalized_value")
        elif facts.get(field) != normalized.get("value"):
            errors.append(f"{label}: Oracle attribute fact disagrees with normalized_value")

        provenance = task.get("provenance")
        if not isinstance(provenance, Mapping):
            errors.append(f"{label}: provenance must be an object")
        else:
            if provenance.get("source_split") != "sft_train":
                errors.append(f"{label}: provenance.source_split must be sft_train")
            if provenance.get("source_task_id") != task_id:
                errors.append(f"{label}: provenance.source_task_id must equal task_id")
            if provenance.get("construction_version") != "probe_b_v2":
                errors.append(f"{label}: construction_version must be probe_b_v2")
            if provenance.get("review_status") != "approved":
                errors.append(f"{label}: review_status must be approved")

    if expected_distribution is not None:
        expected = dict(expected_distribution)
        actual = {field: distribution.get(field, 0) for field in expected}
        if actual != expected:
            errors.append(f"field distribution must be {expected}, found {actual}")

    if errors:
        raise TaskValidationError("Probe B V2 task validation failed:\n- " + "\n- ".join(errors))
    return {
        "task_count": len(rows),
        "distribution": dict(sorted(distribution.items())),
        "task_hash": canonical_hash(rows),
        "task_ids": [int(task["task_id"]) for task in rows],
    }


def latent_goal_satisfied(task: Mapping[str, Any], trajectory: Mapping[str, Any]) -> bool:
    """Deterministically check the hidden slot against the terminal purchase payload."""

    terminal = trajectory.get("terminal_result")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    purchase = terminal.get("purchase")
    if not isinstance(purchase, Mapping) or not purchase:
        return False
    latent = task.get("latent_goal")
    latent = latent if isinstance(latent, Mapping) else {}
    normalized = latent.get("normalized_value")
    normalized = normalized if isinstance(normalized, Mapping) else {}
    field = latent.get("field")
    if field == "budget":
        price = _numeric_price(purchase.get("price"))
        amount = normalized.get("amount")
        if price is None or not isinstance(amount, (int, float)):
            return False
        if normalized.get("operator") == "<":
            return price < float(amount)
        return price <= float(amount)

    match_terms = normalized.get("match_terms") or [normalized.get("value")]
    haystack = json.dumps(purchase, ensure_ascii=False, sort_keys=True).casefold()
    return any(
        str(term).strip().casefold() in haystack
        for term in match_terms
        if str(term).strip()
    )


def _source_queries(path: Path) -> dict[int, str]:
    queries: dict[int, str] = {}
    for row in _jsonl(path):
        task_id = _row_task_id(row)
        if task_id is None:
            continue
        for message in row.get("messages") or []:
            content = str(message.get("content") or "")
            if message.get("role") == "user" and content.startswith("Instruction:"):
                queries[int(task_id)] = content[len("Instruction:") :].strip()
                break
    return queries


def _task_ids(path: Path) -> set[int]:
    return {int(task_id) for row in _jsonl(path) if (task_id := _row_task_id(row)) is not None}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_task_id(row: Mapping[str, Any]) -> int | None:
    if row.get("task_id") is not None:
        return int(row["task_id"])
    extra = row.get("extra_info") or {}
    if extra.get("task_id") is not None:
        return int(extra["task_id"])
    kwargs = extra.get("interaction_kwargs") or {}
    if kwargs.get("task_id") is not None:
        return int(kwargs["task_id"])
    return None


def _required_text(
    task: Mapping[str, Any], key: str, label: str, errors: list[str]
) -> str:
    value = str(task.get(key) or "").strip()
    if not value:
        errors.append(f"{label}: {key} is required")
    return value


def _numeric_price(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(match.group()) if match else None
