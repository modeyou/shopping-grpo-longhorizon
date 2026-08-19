"""Frozen public openings derived from private ShopSimulator goals."""

import hashlib
import json

MULTITURN_TASK_SCHEMA = "shopsimulator-multiturn-task-v1"


def source_goal_hash(context):
    payload = {
        "instruction_full": context["instruction_full"],
        "goal_options": context.get("goal_options") or [],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_task_row(task_id, initial_request, context, model, prompt_hash):
    request = str(initial_request).strip()
    if not request:
        raise ValueError("initial_request must not be empty")
    return {
        "schema_version": MULTITURN_TASK_SCHEMA,
        "task_id": int(task_id),
        "initial_request": request,
        "source_goal_hash": source_goal_hash(context),
        "opening_model": str(model),
        "opening_prompt_hash": str(prompt_hash),
    }


def validate_task_row(task):
    if task.get("schema_version") != MULTITURN_TASK_SCHEMA:
        raise ValueError("unsupported multiturn task schema")
    if not str(task.get("initial_request", "")).strip():
        raise ValueError("multiturn task is missing initial_request")
    return task
