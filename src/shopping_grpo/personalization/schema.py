"""Schema and deterministic validation for personalized shopping tasks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy


SCHEMA_VERSION = "personalized-shopping-task-v1"
SOURCE_SCHEMA_VERSION = "shopsim-source-task-v1"
MAX_QUESTIONS = 2

SCENARIOS = {
    "complete_request",
    "profile_resolvable",
    "clarification_required",
    "profile_conflict",
}
ASK_FIELDS = {
    "budget",
    "brand",
    "function",
    "material",
    "color",
    "size",
    "capacity",
    "bundle",
    "specification",
}
CONSTRAINT_FIELDS = ASK_FIELDS | {"model", "quantity", "compatibility"}
CONSTRAINT_SOURCES = {
    "request_explicit",
    "clarification_answer",
    "profile_stable_fact",
    "profile_preference",
}
HARDNESS = {"hard", "soft"}
PROFILE_LIST_FIELDS = {
    "stable_facts",
    "category_preferences",
    "brand_preferences",
    "budget_preferences",
    "attribute_preferences",
    "option_preferences",
}


class TaskValidationError(ValueError):
    """Raised when a generated task violates the frozen data contract."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized_text(value: object) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list:
    return value if isinstance(value, list) else []


def _validate_profile(profile: object, errors: list[str]) -> None:
    if not isinstance(profile, Mapping):
        errors.append("profile must be an object")
        return
    if not _text(profile.get("profile_id")):
        errors.append("profile.profile_id must be non-empty")
    unknown = set(profile) - ({"profile_id"} | PROFILE_LIST_FIELDS)
    if unknown:
        errors.append(f"profile has unsupported fields: {sorted(unknown)}")
    for key in PROFILE_LIST_FIELDS:
        if key in profile and not isinstance(profile[key], list):
            errors.append(f"profile.{key} must be a list")

    for index, fact in enumerate(_list(profile.get("stable_facts"))):
        item = _mapping(fact)
        if item.get("field") not in {"shoe_size", "clothing_size"}:
            errors.append(f"profile.stable_facts[{index}].field is unsupported")
        if item.get("applies_to") != "self":
            errors.append(f"profile.stable_facts[{index}].applies_to must be self")
        if item.get("confidence") != "high":
            errors.append(f"profile.stable_facts[{index}].confidence must be high")
        if not _text(item.get("value")):
            errors.append(f"profile.stable_facts[{index}].value must be non-empty")


def _validate_constraints(private_goal: object, errors: list[str]) -> tuple[dict, set[str]]:
    goal = _mapping(private_goal)
    if not goal:
        errors.append("private_goal must be an object")
        return {}, set()
    if not _text(goal.get("category")):
        errors.append("private_goal.category must be non-empty")
    constraints = goal.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        errors.append("private_goal.constraints must be a non-empty list")
        return {}, set()

    by_id = {}
    clarification_ids = set()
    for index, raw in enumerate(constraints):
        item = _mapping(raw)
        prefix = f"private_goal.constraints[{index}]"
        constraint_id = _text(item.get("constraint_id"))
        if not constraint_id:
            errors.append(f"{prefix}.constraint_id must be non-empty")
        elif constraint_id in by_id:
            errors.append(f"duplicate constraint_id: {constraint_id}")
        else:
            by_id[constraint_id] = item
        if item.get("field") not in CONSTRAINT_FIELDS:
            errors.append(f"{prefix}.field is unsupported")
        if item.get("source") not in CONSTRAINT_SOURCES:
            errors.append(f"{prefix}.source is unsupported")
        elif item.get("source") == "clarification_answer" and constraint_id:
            clarification_ids.add(constraint_id)
        if item.get("hardness") not in HARDNESS:
            errors.append(f"{prefix}.hardness must be hard or soft")
        if item.get("value") in (None, "", [], {}):
            errors.append(f"{prefix}.value must be non-empty")
    return by_id, clarification_ids


def _validate_clarification(
    clarification: object,
    *,
    scenario: object,
    constraints: dict,
    clarification_ids: set[str],
    current_request: str,
    profile: Mapping,
    errors: list[str],
) -> None:
    value = _mapping(clarification)
    if not value:
        errors.append("clarification must be an object")
        return
    should_ask = value.get("should_ask")
    if not isinstance(should_ask, bool):
        errors.append("clarification.should_ask must be boolean")
    if value.get("max_questions") != MAX_QUESTIONS:
        errors.append(f"clarification.max_questions must equal {MAX_QUESTIONS}")
    targets = value.get("targets")
    if not isinstance(targets, list):
        errors.append("clarification.targets must be a list")
        return
    if len(targets) > MAX_QUESTIONS:
        errors.append(f"clarification.targets exceeds {MAX_QUESTIONS}")

    expected_ask = scenario == "clarification_required"
    if should_ask is not expected_ask:
        errors.append("clarification.should_ask disagrees with scenario")
    if expected_ask and not targets:
        errors.append("clarification_required needs at least one target")
    if not expected_ask and targets:
        errors.append("non-clarification scenario cannot contain targets")

    target_ids = set()
    fields = set()
    visible = _normalized_text(current_request + " " + canonical_json(profile))
    for index, raw in enumerate(targets):
        item = _mapping(raw)
        prefix = f"clarification.targets[{index}]"
        constraint_id = _text(item.get("constraint_id"))
        field = item.get("field")
        if constraint_id not in constraints:
            errors.append(f"{prefix}.constraint_id is unknown")
        elif constraints[constraint_id].get("source") != "clarification_answer":
            errors.append(f"{prefix} does not reference a clarification constraint")
        if constraint_id in target_ids:
            errors.append(f"duplicate clarification constraint: {constraint_id}")
        target_ids.add(constraint_id)
        if field not in ASK_FIELDS:
            errors.append(f"{prefix}.field is unsupported")
        elif field in fields:
            errors.append(f"duplicate clarification field: {field}")
        fields.add(field)
        if constraint_id in constraints and constraints[constraint_id].get("field") != field:
            errors.append(f"{prefix}.field disagrees with private_goal")
        answer = _text(item.get("answer"))
        if not answer:
            errors.append(f"{prefix}.answer must be non-empty")
        facts = item.get("answer_facts")
        if not isinstance(facts, Mapping) or field not in facts:
            errors.append(f"{prefix}.answer_facts must contain its field")
        if constraint_id in constraints:
            leaked = _normalized_text(constraints[constraint_id].get("value"))
            if len(leaked) >= 2 and leaked in visible:
                errors.append(f"{prefix} answer leaks into Agent initial context")

    if target_ids != clarification_ids:
        errors.append("clarification targets must exactly match clarification constraints")


def _validate_scenario_semantics(task: Mapping, constraints: dict, errors: list[str]) -> None:
    scenario = task.get("scenario")
    sources = [item.get("source") for item in constraints.values()]
    if scenario == "complete_request" and any(source != "request_explicit" for source in sources):
        errors.append("complete_request constraints must all be request_explicit")
    if scenario == "profile_resolvable" and not any(
        source in {"profile_stable_fact", "profile_preference"} for source in sources
    ):
        errors.append("profile_resolvable needs a profile-sourced constraint")
    if scenario == "profile_conflict" and not _list(task.get("conflicts")):
        errors.append("profile_conflict needs at least one declared conflict")


def validate_task(task: object) -> dict:
    """Validate and return a defensive copy of one private task package."""

    if not isinstance(task, Mapping):
        raise TaskValidationError(["task must be an object"])
    errors = []
    if task.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not _text(task.get("task_id")):
        errors.append("task_id must be non-empty")
    source = _mapping(task.get("source"))
    if not isinstance(source.get("shopsim_task_id"), int) or source.get("shopsim_task_id", -1) < 0:
        errors.append("source.shopsim_task_id must be a non-negative integer")
    if not _text(source.get("target_asin")):
        errors.append("source.target_asin must be non-empty")
    if not _text(source.get("source_environment_version")):
        errors.append("source.source_environment_version must be non-empty")
    scenario = task.get("scenario")
    if scenario not in SCENARIOS:
        errors.append("scenario is unsupported")
    current_request = _text(task.get("current_request"))
    if not current_request:
        errors.append("current_request must be non-empty")

    _validate_profile(task.get("profile"), errors)
    constraints, clarification_ids = _validate_constraints(task.get("private_goal"), errors)
    _validate_clarification(
        task.get("clarification"),
        scenario=scenario,
        constraints=constraints,
        clarification_ids=clarification_ids,
        current_request=current_request,
        profile=_mapping(task.get("profile")),
        errors=errors,
    )
    if not isinstance(task.get("conflicts"), list):
        errors.append("conflicts must be a list")
    _validate_scenario_semantics(task, constraints, errors)

    serialized_profile = canonical_json(task.get("profile"))
    asin = _text(source.get("target_asin"))
    if asin and asin in serialized_profile:
        errors.append("target ASIN leaks into profile")
    if errors:
        raise TaskValidationError(errors)
    return deepcopy(dict(task))


def finalize_task(task: object) -> dict:
    """Validate a task and attach a stable audit hash without self-hashing."""

    result = validate_task(task)
    audit = dict(_mapping(result.get("audit")))
    audit.pop("task_hash", None)
    result["audit"] = audit
    audit["task_hash"] = stable_hash(result)
    return result


def actor_view(task: object) -> dict:
    """Return exactly the task context visible to the shopping policy."""

    validated = validate_task(task)
    return {
        "task_id": validated["task_id"],
        "profile": deepcopy(validated["profile"]),
        "current_request": validated["current_request"],
    }
