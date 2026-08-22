"""Deterministic, token-aware mixing for multi-turn SFT candidate pools."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from copy import deepcopy
from typing import Iterable


POLICY_ORDER = (
    "complete-no-ask-v1",
    "composite-replay-v1",
    "autonomous-gap-v1",
)
MIX_SCHEMA_VERSION = "shopping-multiturn-sft-mix-v1"


def stable_hash(seed: int, *parts: object) -> str:
    payload = ":".join(str(part) for part in parts)
    return hashlib.sha256(f"{seed}:{payload}".encode("utf-8")).hexdigest()


def membership_patterns(pools: dict[str, Iterable[dict]]) -> dict[str, int]:
    memberships: dict[int, set[str]] = {}
    for policy, rows in pools.items():
        for row in rows:
            memberships.setdefault(int(row["task_id"]), set()).add(policy)
    counts = Counter(
        "+".join(
            policy for policy in POLICY_ORDER if policy in policies
        )
        for policies in memberships.values()
    )
    return dict(sorted(counts.items()))


def allocate_row_quotas(
    *,
    total_rows: int,
    token_ratios: dict[str, float],
    average_assistant_tokens: dict[str, float],
) -> dict[str, int]:
    """Convert target token shares into integer row quotas."""

    if total_rows < len(POLICY_ORDER):
        raise ValueError("total_rows must allow at least one row per policy")
    if set(token_ratios) != set(POLICY_ORDER):
        raise ValueError("token ratios must define all Teacher policies")
    if set(average_assistant_tokens) != set(POLICY_ORDER):
        raise ValueError("token averages must define all Teacher policies")
    if any(value <= 0 for value in token_ratios.values()):
        raise ValueError("token ratios must be positive")
    if not math.isclose(sum(token_ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("token ratios must sum to 1")
    if any(value <= 0 for value in average_assistant_tokens.values()):
        raise ValueError("assistant-token averages must be positive")

    weights = {
        policy: token_ratios[policy] / average_assistant_tokens[policy]
        for policy in POLICY_ORDER
    }
    total_weight = sum(weights.values())
    exact = {
        policy: total_rows * weights[policy] / total_weight
        for policy in POLICY_ORDER
    }
    quotas = {policy: max(1, int(math.floor(value))) for policy, value in exact.items()}
    while sum(quotas.values()) < total_rows:
        policy = max(
            POLICY_ORDER,
            key=lambda item: (exact[item] - quotas[item], stable_hash(0, item)),
        )
        quotas[policy] += 1
    while sum(quotas.values()) > total_rows:
        candidates = [policy for policy in POLICY_ORDER if quotas[policy] > 1]
        policy = min(
            candidates,
            key=lambda item: (exact[item] - quotas[item], stable_hash(0, item)),
        )
        quotas[policy] -= 1
    return quotas


def _stratified_select(
    rows: list[dict], *, count: int, seed: int, label: str
) -> list[dict]:
    if count < 0 or count > len(rows):
        raise ValueError(f"{label}: requested {count} rows from {len(rows)}")
    if count == 0:
        return []
    ordered = sorted(
        rows,
        key=lambda item: (
            int(item["assistant_tokens"]),
            stable_hash(seed, label, item["row"]["task_id"]),
        ),
    )
    selected = []
    for index in range(count):
        start = math.floor(index * len(ordered) / count)
        end = math.floor((index + 1) * len(ordered) / count)
        bucket = ordered[start:max(start + 1, end)]
        selected.append(
            min(
                bucket,
                key=lambda item: stable_hash(
                    seed,
                    label,
                    item["row"]["task_id"],
                    item["row"].get("trajectory_id"),
                ),
            )
        )
    return selected


def select_disjoint_rows(
    *,
    pools: dict[str, list[dict]],
    quotas: dict[str, int],
    seed: int,
) -> dict[str, list[dict]]:
    """Select representative rows while excluding task and source-goal reuse."""

    order = sorted(
        POLICY_ORDER,
        key=lambda policy: (len(pools[policy]) / quotas[policy], policy),
    )
    selected: dict[str, list[dict]] = {policy: [] for policy in POLICY_ORDER}
    used_task_ids: set[int] = set()
    used_goal_hashes: set[str] = set()

    for policy in order:
        candidates = [
            item
            for item in pools[policy]
            if int(item["row"]["task_id"]) not in used_task_ids
            and str(item["row"].get("source_goal_hash") or "") not in used_goal_hashes
        ]
        if policy == "complete-no-ask-v1":
            ask_available = [
                item
                for item in candidates
                if item["row"].get("schema_variant") == "complete-ask-available-v1"
            ]
            original = [
                item
                for item in candidates
                if item["row"].get("schema_variant") == "complete-shop-tools-v1"
            ]
            ask_count = quotas[policy] // 2
            chosen = _stratified_select(
                ask_available,
                count=ask_count,
                seed=seed,
                label=f"{policy}:ask-available",
            )
            chosen += _stratified_select(
                original,
                count=quotas[policy] - ask_count,
                seed=seed,
                label=f"{policy}:original",
            )
        else:
            chosen = _stratified_select(
                candidates,
                count=quotas[policy],
                seed=seed,
                label=policy,
            )
        selected[policy] = chosen
        for item in chosen:
            used_task_ids.add(int(item["row"]["task_id"]))
            used_goal_hashes.add(str(item["row"].get("source_goal_hash") or ""))

    return selected


def split_selected(
    selected: dict[str, list[dict]], *, validation_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1)")
    train = []
    validation = []
    for policy in POLICY_ORDER:
        rows = selected[policy]
        validation_count = max(1, round(len(rows) * validation_ratio))
        validation_count = min(validation_count, len(rows) - 1)
        validation_rows = _stratified_select(
            rows,
            count=validation_count,
            seed=seed,
            label=f"{policy}:validation",
        )
        validation_keys = {
            (int(item["row"]["task_id"]), item["row"].get("trajectory_id"))
            for item in validation_rows
        }
        validation.extend(validation_rows)
        train.extend(
            item
            for item in rows
            if (int(item["row"]["task_id"]), item["row"].get("trajectory_id"))
            not in validation_keys
        )
    key = lambda item: stable_hash(
        seed, item["row"]["task_id"], item["row"].get("trajectory_id")
    )
    return sorted(train, key=key), sorted(validation, key=key)


def augment_complete_schemas(rows: Iterable[dict], *, seed: int) -> list[dict]:
    """Create a deterministic 50/50 complete pool with ask available in half."""

    from shopping_grpo.environment.tools import MULTITURN_SHOP_TOOL_SCHEMAS

    augmented = []
    for source in rows:
        row = deepcopy(source)
        if int(stable_hash(seed, row["task_id"]), 16) % 2 == 0:
            row["tools"] = deepcopy(MULTITURN_SHOP_TOOL_SCHEMAS)
            row["schema_variant"] = "complete-ask-available-v1"
        else:
            row["schema_variant"] = "complete-shop-tools-v1"
        augmented.append(row)
    return augmented
