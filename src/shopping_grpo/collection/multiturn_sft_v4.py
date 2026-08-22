"""Re-audit stored multi-turn Teacher trajectories under Reward v4.

The original collections remain immutable Reward v3 artifacts. This module
reconstructs the Reward v4 goal from frozen ShopSimulator product data and
scores the Teacher's actual purchase and selected options. Only the v3/v4
strict-gold intersection is converted into a new action-only SFT pool.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from shopping_grpo.collection.sft import acceptance_reasons, build_sft_row
from shopping_grpo.multiturn.tasks import source_goal_hash


V3 = "shopsimulator-reward-v3"
V4 = "shopsimulator-reward-v4"
POLICIES = (
    "complete-no-ask-v1",
    "composite-replay-v1",
    "autonomous-gap-v1",
)
POOL_SCHEMA_VERSION = "shopping-multiturn-sft-v4-pool-v1"


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_ids(path: str | Path) -> set[int]:
    rows = read_jsonl(path)
    values = [int(row["task_id"]) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate task IDs")
    return set(values)


def _candidate_product(product: dict) -> dict:
    candidate = deepcopy(product)
    bullet_points = product.get("small_description", "")
    candidate.update(
        {
            "Title": product.get("title", ""),
            "Description": product.get("full_description", ""),
            "BulletPoints": (
                bullet_points if isinstance(bullet_points, list) else [bullet_points]
            ),
            "Attributes": product.get("attribute", []),
            "pricing": product.get("pricing") or [],
        }
    )
    return candidate


def rescore_terminal_purchase_v4(
    trajectory: dict, products: list[dict]
) -> tuple[dict | None, list[str], dict]:
    """Score the actual Teacher terminal purchase under a reconstructed v4 goal."""

    from web_agent_site.engine.reward_registry import (
        REWARD_DEFAULTS,
        compile_reward_features_for_version,
        evaluate_purchase,
    )
    from web_agent_site.engine.variant_price import resolve_variant_price

    reasons = []
    task_id = int(trajectory["task_id"])
    if task_id < 0 or task_id >= len(products):
        return None, ["task_outside_product_data"], {"task_id": task_id}

    target = products[task_id]
    instructions = [
        item
        for item in target.get("instructions") or []
        if isinstance(item, dict) and item.get("attributes")
    ]
    if len(instructions) != 1:
        return None, ["goal_instruction_not_unique"], {
            "task_id": task_id,
            "valid_instruction_count": len(instructions),
        }

    terminal = trajectory.get("terminal_result") or {}
    purchase = terminal.get("purchase") or {}
    actual_asin = str(purchase.get("asin") or "")
    target_asin = str(target.get("asin") or "")
    if not actual_asin:
        reasons.append("terminal_purchase_missing_asin")
    elif actual_asin != target_asin:
        reasons.append("terminal_purchase_not_target_asin")

    selected_options = purchase.get("options")
    if not isinstance(selected_options, dict):
        reasons.append("terminal_purchase_options_not_object")
        selected_options = {}

    candidate = _candidate_product(target)
    goal = {
        "asin": target.get("asin"),
        "category": target.get("category"),
    }
    goal.update(
        compile_reward_features_for_version(
            instructions[0], target, V4
        )
    )
    price_resolution = resolve_variant_price(candidate, selected_options)
    result = evaluate_purchase(
        candidate,
        goal,
        selected_options=selected_options,
        price_resolution=price_resolution,
        rewards={"version": V4, **REWARD_DEFAULTS[V4]},
    )
    detail = result.to_dict()

    stored_price = purchase.get("price")
    resolved_price = price_resolution.get("price")
    try:
        price_matches = abs(float(stored_price) - float(resolved_price)) <= 0.01
    except (TypeError, ValueError):
        price_matches = stored_price is None and resolved_price is None
    if not price_matches:
        reasons.append("terminal_purchase_price_mismatch")

    audit = {
        "task_id": task_id,
        "trajectory_id": trajectory.get("trajectory_id"),
        "teacher_policy": trajectory.get("teacher_policy"),
        "source_reward_version": (
            (terminal.get("reward_detail") or {}).get("reward_version")
        ),
        "actual_asin": actual_asin,
        "target_asin_match": actual_asin == target_asin,
        "selected_options": selected_options,
        "stored_price": stored_price,
        "price_resolution_status": price_resolution.get("status"),
        "price_resolution_method": price_resolution.get("method"),
        "resolved_price": resolved_price,
        "reward_version": detail.get("reward_version"),
        "reward_type": detail.get("reward_type"),
        "reward_valid": detail.get("reward_valid"),
        "purchase_success": detail.get("purchase_success"),
        "termination_reason": detail.get("termination_reason"),
        "constraint_atom_count": len(
            ((detail.get("evidence") or {}).get("constraint_scoring") or {}).get("atoms") or []
        ),
        "evidence_coverage": detail.get("evidence_coverage"),
    }
    return detail, reasons, audit


def audit_source(
    *,
    raw_path: str | Path,
    expected_policy: str,
    products: list[dict],
    allowed_task_ids: set[int],
    excluded_task_ids: set[int],
) -> tuple[list[dict], list[dict], dict]:
    """Return one v4 SFT pool plus accepted/rejected audit rows."""

    if expected_policy not in POLICIES:
        raise ValueError(f"unsupported Teacher policy: {expected_policy}")

    accepted_rows = []
    accepted_audits = []
    rejected = []
    seen_task_ids = set()
    reason_counts = Counter()
    source_rows = read_jsonl(raw_path)

    for trajectory in source_rows:
        task_id = int(trajectory["task_id"])
        reasons = []
        terminal_goal = (trajectory.get("terminal_result") or {}).get("goal") or {}
        if not str(terminal_goal.get("instruction_text") or "").strip():
            goal_hash = None
            reasons.append("source_goal_fields_missing")
        else:
            goal_hash = source_goal_hash(
                {
                    "instruction_full": terminal_goal["instruction_text"],
                    "goal_options": terminal_goal.get("goal_options") or [],
                }
            )
            recorded_hash = trajectory.get("source_goal_hash")
            if recorded_hash is not None and recorded_hash != goal_hash:
                reasons.append("source_goal_hash_mismatch")
        if trajectory.get("teacher_policy") != expected_policy:
            reasons.append("teacher_policy_mismatch")
        if task_id not in allowed_task_ids:
            reasons.append("outside_sft_candidates")
        if task_id in excluded_task_ids:
            reasons.append("held_out_task")

        source_ok, source_reasons = acceptance_reasons(trajectory, V3)
        if not source_ok:
            reasons.extend(f"source_v3.{reason}" for reason in source_reasons)

        detail, rescore_reasons, audit = rescore_terminal_purchase_v4(
            trajectory, products
        )
        reasons.extend(rescore_reasons)
        rescored = deepcopy(trajectory)
        if detail is not None:
            terminal = dict(rescored.get("terminal_result") or {})
            terminal["reward_detail"] = detail
            terminal["reward"] = detail.get("terminal_utility")
            terminal["reward_valid"] = detail.get("reward_valid")
            terminal["termination_reason"] = detail.get("termination_reason")
            rescored["terminal_result"] = terminal
            rescored["final_reward"] = detail.get("terminal_utility")
            v4_ok, v4_reasons = acceptance_reasons(rescored, V4)
            if not v4_ok:
                reasons.extend(f"rescored_v4.{reason}" for reason in v4_reasons)
        else:
            reasons.append("reward_v4_rescore_failed")

        reasons = list(dict.fromkeys(reasons))
        if not reasons and task_id in seen_task_ids:
            reasons = ["duplicate_task"]

        if reasons:
            reason_counts.update(reasons)
            rejected.append(
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory.get("trajectory_id"),
                    "teacher_policy": trajectory.get("teacher_policy"),
                    "reject_reasons": reasons,
                    "v4_audit": audit,
                }
            )
            continue

        seen_task_ids.add(task_id)
        row = build_sft_row(rescored)
        row["teacher_policy"] = expected_policy
        row["source_goal_hash"] = goal_hash
        row["schema_variant"] = (
            "complete-shop-tools-v1"
            if expected_policy == "complete-no-ask-v1"
            else "multiturn-shop-tools-v1"
        )
        accepted_rows.append(row)
        accepted_audits.append(audit)

    summary = {
        "raw_rows": len(source_rows),
        "accepted_v4_rows": len(accepted_rows),
        "rejected_rows": len(rejected),
        "unique_accepted_tasks": len(seen_task_ids),
        "reject_reasons": dict(sorted(reason_counts.items())),
    }
    return accepted_rows, [*accepted_audits, *rejected], summary


def content_hash(row: dict) -> str:
    payload = {
        "task_id": row.get("task_id"),
        "messages": row.get("messages"),
        "tools": row.get("tools"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cross_policy_overlaps(pools: dict[str, Iterable[dict]]) -> dict:
    ids = {
        policy: {int(row["task_id"]) for row in rows}
        for policy, rows in pools.items()
    }
    overlaps = {}
    policies = list(ids)
    for left_index, left in enumerate(policies):
        for right in policies[left_index + 1 :]:
            shared = sorted(ids[left] & ids[right])
            overlaps[f"{left}__{right}"] = {
                "tasks": len(shared),
                "task_ids": shared,
            }
    return overlaps
