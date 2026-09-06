"""Extract Query-only, versioned records for Rubric drafting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from shopping_grpo.evaluation.rubric import build_task_facts


def task_facts_from_environment(
    *,
    task_ids: Iterable[int],
    goals: list[Mapping],
) -> list[dict]:
    """Build Query-only task facts in requested task order."""

    requested = [int(task_id) for task_id in task_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("task_ids contains duplicates")
    rows = []
    for task_id in requested:
        if task_id < 0 or task_id >= len(goals):
            raise IndexError(
                f"task_id {task_id} is outside goal range [0, {len(goals)})"
            )
        goal = goals[task_id]
        if not isinstance(goal, Mapping):
            raise ValueError(f"goal {task_id} must be an object")
        query = str(goal.get("instruction_text") or "").strip()
        if not query:
            raise ValueError(f"goal {task_id} has no instruction_text")
        rows.append(build_task_facts(task_id=task_id, query=query))
    return rows


def task_facts_from_products(
    *,
    task_ids: Iterable[int],
    products: list[Mapping],
) -> list[dict]:
    """Build the same Query-only facts from frozen product task data."""

    requested = [int(task_id) for task_id in task_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("task_ids contains duplicates")
    rows = []
    for task_id in requested:
        if task_id < 0 or task_id >= len(products):
            raise IndexError(
                f"task_id {task_id} is outside product range [0, {len(products)})"
            )
        product = products[task_id]
        if not isinstance(product, Mapping):
            raise ValueError(f"product {task_id} must be an object")
        instructions = [
            item
            for item in (product.get("instructions") or [])
            if isinstance(item, Mapping) and item.get("attributes")
        ]
        if len(instructions) != 1:
            raise ValueError(
                f"task {task_id} must have exactly one scored instruction"
            )
        instruction = instructions[0]
        query = str(instruction.get("instruction") or "").strip()
        if not query:
            raise ValueError(f"task {task_id} has no instruction text")
        rows.append(build_task_facts(task_id=task_id, query=query))
    return rows
