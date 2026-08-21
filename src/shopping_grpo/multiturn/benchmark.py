"""Curate a Reward-reachable, task-disjoint multi-turn evaluation manifest."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from shopping_grpo.multiturn.splits import sha256_file, sha256_text_file


MULTITURN_EVALUATION_SCHEMA = "shopping-multiturn-evaluation-v1"
PRICE_HINT = re.compile(
    r"(?:预算|价格|售价|价钱|价位|总价|费用|成本|花费|多少钱|"
    r"[零一二三四五六七八九十百千万两\d]+(?:\.\d+)?\s*(?:元|块|钱)|"
    r"\d+(?:\.\d+)?\s*(?:左右|上下|出头))"
)


def read_task_ids(path: str | Path) -> list[int]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "task_id" not in row:
                raise ValueError(f"{path}:{line_number} is missing task_id")
            rows.append(int(row["task_id"]))
    if len(rows) != len(set(rows)):
        raise ValueError(f"{path} contains duplicate task IDs")
    return rows


def load_products(path: str | Path) -> list[dict]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    if not isinstance(products, list):
        raise ValueError("ShopSimulator product data must contain a JSON list")
    return products


def audit_gold_task(product: dict, task_id: int) -> dict:
    """Evaluate the deterministic gold purchase without running an Actor."""

    # Imported lazily so ordinary package users do not need ShopSimulator on
    # sys.path. The CLI and focused tests add the embedded environment root.
    from web_agent_site.engine.constraints import explicit_budget_from_instruction
    from web_agent_site.engine.reward import evaluate_purchase
    from web_agent_site.engine.reward_features import compile_reward_features
    from web_agent_site.engine.variant_price import (
        candidate_options_for_evaluation,
        resolve_variant_price,
    )

    instructions = product.get("instructions") or []
    valid_instructions = [
        item
        for item in instructions
        if isinstance(item, dict) and item.get("attributes")
    ]
    if len(valid_instructions) != 1:
        return {
            "task_id": int(task_id),
            "eligible": False,
            "reasons": ["goal_instruction_not_unique"],
            "audit": {"valid_instruction_count": len(valid_instructions)},
        }

    instruction = valid_instructions[0]
    query = str(instruction.get("instruction") or "")
    candidate = deepcopy(product)
    candidate.update({
        "Title": product.get("title", ""),
        "Description": product.get("full_description", ""),
        "BulletPoints": (
            product.get("small_description", "")
            if isinstance(product.get("small_description", ""), list)
            else [product.get("small_description", "")]
        ),
        "Attributes": product.get("attribute", []),
        "pricing": product.get("pricing") or [],
    })
    goal = {
        "asin": product.get("asin"),
        "category": product.get("category"),
        "price_upper": explicit_budget_from_instruction(query),
    }
    goal.update(compile_reward_features(instruction, product))

    selected, option_resolution = candidate_options_for_evaluation(
        candidate, goal.get("required_options_by_key")
    )
    price_resolution = resolve_variant_price(candidate, selected)
    result = evaluate_purchase(
        candidate,
        goal,
        selected_options=selected,
        price_resolution=price_resolution,
    )

    reasons = []
    if goal.get("unresolved_option_requirements"):
        reasons.append("unresolved_option_requirements")
    if option_resolution.get("status") != "pass":
        reasons.append("gold_option_selection_unresolved")
    if price_resolution.get("status") != "pass":
        reasons.append("gold_variant_price_unresolved")
    if PRICE_HINT.search(query) and goal.get("price_upper") is None:
        reasons.append("explicit_price_not_compiled")
    if result.reward_type != "gold_purchase":
        reasons.append("gold_purchase_not_reachable")
    if result.reward_valid is not True:
        reasons.append("gold_reward_invalid")

    return {
        "task_id": int(task_id),
        "eligible": not reasons,
        "reasons": reasons,
        "audit": {
            "asin": str(product.get("asin")),
            "required_option_count": len(
                goal.get("required_options_by_key") or {}
            ),
            "unresolved_option_count": len(
                goal.get("unresolved_option_requirements") or []
            ),
            "option_resolution_status": option_resolution.get("status"),
            "price_resolution_status": price_resolution.get("status"),
            "price_resolution_method": price_resolution.get("method"),
            "compiled_price_upper": goal.get("price_upper"),
            "query_has_price_hint": bool(PRICE_HINT.search(query)),
            "reward_type": result.reward_type,
            "reward_valid": result.reward_valid,
        },
    }


def curate_task_ids(
    *,
    products: list[dict],
    candidates: list[int],
    reserve: list[int],
) -> tuple[list[int], list[dict], dict]:
    """Retain eligible candidates and fill rejected slots from reserve order."""

    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate task IDs must be unique")
    if set(candidates) & set(reserve):
        raise ValueError("candidate and reserve task IDs must be disjoint")
    outside = [
        task_id
        for task_id in [*candidates, *reserve]
        if task_id < 0 or task_id >= len(products)
    ]
    if outside:
        raise ValueError(f"task IDs outside product data: {outside[:10]}")

    candidate_audits = [
        audit_gold_task(products[task_id], task_id) for task_id in candidates
    ]
    accepted = [
        row["task_id"] for row in candidate_audits if row["eligible"]
    ]
    needed = len(candidates) - len(accepted)
    replacement_audits = []
    replacements = []
    for task_id in reserve:
        if len(replacements) >= needed:
            break
        audit = audit_gold_task(products[task_id], task_id)
        replacement_audits.append(audit)
        if audit["eligible"]:
            replacements.append(task_id)
    if len(replacements) != needed:
        raise ValueError(
            f"reserve exhausted: needed {needed}, found {len(replacements)}"
        )

    selected = [
        *[row["task_id"] for row in candidate_audits if row["eligible"]],
        *replacements,
    ]
    if len(selected) != len(candidates) or len(selected) != len(set(selected)):
        raise AssertionError("curated evaluation task IDs are not unique and complete")

    reasons = Counter(
        reason
        for row in candidate_audits
        if not row["eligible"]
        for reason in row["reasons"]
    )
    report = {
        "source_candidates": len(candidates),
        "source_eligible": len(candidates) - needed,
        "source_rejected": needed,
        "replacement_candidates_audited": len(replacement_audits),
        "replacement_tasks": len(replacements),
        "source_reject_reasons": dict(sorted(reasons.items())),
        "rejected_source_task_ids": [
            row["task_id"] for row in candidate_audits if not row["eligible"]
        ],
        "replacement_task_ids": replacements,
    }
    return selected, [*candidate_audits, *replacement_audits], report


def freeze_curated_evaluation(
    *,
    product_data_path: str | Path,
    candidates_path: str | Path,
    reserve_path: str | Path,
    split_metadata_path: str | Path,
    environment_manifest_path: str | Path,
    output_dir: str | Path,
    exclusion_paths=(),
) -> dict:
    """Write an idempotent curated benchmark, audit log, and manifest."""

    product_data_path = Path(product_data_path)
    candidates_path = Path(candidates_path)
    reserve_path = Path(reserve_path)
    split_metadata_path = Path(split_metadata_path)
    environment_manifest_path = Path(environment_manifest_path)
    output_dir = Path(output_dir)

    products = load_products(product_data_path)
    candidates = read_task_ids(candidates_path)
    reserve = read_task_ids(reserve_path)
    excluded = set()
    exclusion_sources = []
    for raw_path in exclusion_paths:
        path = Path(raw_path)
        ids = read_task_ids(path)
        excluded.update(ids)
        exclusion_sources.append({
            "path": str(path.as_posix()),
            "sha256": sha256_text_file(path),
            "tasks": len(ids),
        })
    candidate_overlap = sorted(set(candidates) & excluded)
    if candidate_overlap:
        raise ValueError(
            "evaluation candidates overlap explicit exclusions: "
            f"{candidate_overlap[:10]}"
        )
    reserve = [task_id for task_id in reserve if task_id not in excluded]
    selected, audits, report = curate_task_ids(
        products=products,
        candidates=candidates,
        reserve=reserve,
    )
    tasks_payload = "".join(
        json.dumps({"task_id": task_id}, separators=(",", ":")) + "\n"
        for task_id in selected
    ).encode("utf-8")
    audits_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in audits
    ).encode("utf-8")
    metadata = {
        "schema_version": MULTITURN_EVALUATION_SCHEMA,
        "task_count": len(selected),
        "selection": {
            "policy": "retain-source-order-then-fill-from-frozen-reserve-order",
            **report,
        },
        "inputs": {
            "product_data": str(product_data_path.as_posix()),
            "product_data_sha256": sha256_file(product_data_path),
            "source_candidates": str(candidates_path.as_posix()),
            "source_candidates_sha256": sha256_text_file(candidates_path),
            "reserve": str(reserve_path.as_posix()),
            "reserve_sha256": sha256_text_file(reserve_path),
            "split_metadata": str(split_metadata_path.as_posix()),
            "split_metadata_sha256": sha256_text_file(split_metadata_path),
            "environment_manifest": str(environment_manifest_path.as_posix()),
            "environment_manifest_sha256": sha256_text_file(
                environment_manifest_path
            ),
        },
        "reward_contract": "shopsimulator-reward-v3",
        "task_sha256": hashlib.sha256(tasks_payload).hexdigest(),
        "audit_sha256": hashlib.sha256(audits_payload).hexdigest(),
        "validation": {
            "all_selected_tasks_reward_reachable": True,
            "source_and_replacements_disjoint": not bool(
                set(candidates) & set(report["replacement_task_ids"])
            ),
            "fixed_denominator": len(candidates),
        },
    }
    if exclusion_sources:
        metadata["inputs"].update({
            "exclusion_sources": exclusion_sources,
            "excluded_task_ids": len(excluded),
        })
        metadata["validation"][
            "selected_tasks_disjoint_from_exclusions"
        ] = not bool(set(selected) & excluded)
    metadata_payload = (
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    planned = {
        output_dir / "tasks.jsonl": tasks_payload,
        output_dir / "reward_audit.jsonl": audits_payload,
        output_dir / "metadata.json": metadata_payload,
    }
    mismatched = [
        path for path, payload in planned.items()
        if path.exists() and path.read_bytes() != payload
    ]
    if mismatched:
        raise FileExistsError(
            "curated evaluation artifacts differ; refusing to overwrite: "
            + ", ".join(str(path) for path in mismatched)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in planned.items():
        if not path.exists():
            path.write_bytes(payload)
    return metadata
