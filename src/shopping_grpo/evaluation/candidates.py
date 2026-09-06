"""Build auditable Rubric candidates from Reward v4 constraint atoms."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy

from shopping_grpo.evaluation.rubric import stable_hash


RUBRIC_CANDIDATE_VERSION = "shopping-rubric-candidates-v3"
RUBRIC_EXTRACTOR_VERSION = "reward-v4-constraint-atoms-v2"
REWARD_VERSION = "shopsimulator-reward-v4"


def _field_and_operator(dimension: str, requirement: object) -> tuple[str, str]:
    if dimension == "category":
        return "product.category", "in_category"
    if dimension == "brand":
        return "product.brand", "eq"
    if dimension == "model":
        return "product.model", "eq"
    if dimension == "core_function":
        return "product.attributes", "contains"
    if dimension == "option":
        axis = requirement.get("axis") if isinstance(requirement, Mapping) else None
        return f"purchase.options.{axis or 'unresolved'}", "eq"
    if dimension == "price":
        kind = requirement.get("kind") if isinstance(requirement, Mapping) else None
        return "purchase.price", {
            "hard_max": "lte",
            "hard_min": "gte",
            "hard_range": "between",
            "soft_target": "near",
        }.get(str(kind), "matches")
    raise ValueError(f"unsupported Reward v4 constraint dimension: {dimension!r}")


def _description(dimension: str, requirement: object) -> str:
    if dimension == "category":
        return f"商品品类候选：{str(requirement).split('›')[-1]}"
    if dimension == "option" and isinstance(requirement, Mapping):
        axis = requirement.get("source_axis") or requirement.get("axis")
        return f"规格候选：{axis}={requirement.get('value')}"
    if dimension == "price" and isinstance(requirement, Mapping):
        return f"价格候选：{requirement.get('source_text')}"
    return f"{dimension} 候选：{requirement}"


def _option_components(requirement: object) -> list[str]:
    if not isinstance(requirement, Mapping):
        return []
    value = str(requirement.get("value") or "")
    components = []
    for raw in re.split(r"[+＋\-—_/／,，;；()（）\s]+", value):
        component = raw.strip()
        if len(component) < 2 or component == value or component in components:
            continue
        components.append(component)
    return components


def extract_rubric_candidates(
    *, task_id: int, query: str, instruction: Mapping, product: Mapping
) -> dict:
    """Convert Reward v4 atoms into a candidate superset for LLM filtering."""

    from web_agent_site.engine.reward_registry import compile_reward_features_for_version

    features = compile_reward_features_for_version(
        instruction, product, REWARD_VERSION
    )
    rows = []
    seen = set()
    for atom in features.get("constraint_atoms") or []:
        if not isinstance(atom, Mapping):
            continue
        dimension = str(atom.get("dimension") or "")
        requirement = deepcopy(atom.get("requirement"))
        field_path, operator = _field_and_operator(dimension, requirement)
        identity = stable_hash(
            {
                "dimension": dimension,
                "field_path": field_path,
                "operator": operator,
                "expected_value": requirement,
            }
        )
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(
            {
                "candidate_id": f"c{len(rows) + 1:04d}",
                "constraint_type": dimension,
                "description_hint": _description(dimension, requirement),
                "field_path": field_path,
                "operator": operator,
                "expected_value": requirement,
                "hardness_hint": (
                    "hard" if atom.get("strength") in {"hard", "required"} else "soft"
                ),
                "data_sources": [str(atom.get("source") or "reward_v4")],
                "selection_guidance": (
                    "仅当 Query 原文直接表达该语义时选择；候选来自结构化标注，"
                    "可能过宽、使用同义词或包含组合规格，不能仅因它属于目标商品而选择。"
                ),
            }
        )
        if dimension == "option" and isinstance(requirement, Mapping):
            for component in _option_components(requirement):
                component_value = {
                    "axis": requirement.get("axis"),
                    "source_axis": requirement.get("source_axis"),
                    "selected_option": requirement.get("value"),
                    "component": component,
                }
                component_identity = stable_hash(
                    {
                        "dimension": "option_component",
                        "field_path": field_path,
                        "operator": "contains_component",
                        "expected_value": component_value,
                    }
                )
                if component_identity in seen:
                    continue
                seen.add(component_identity)
                rows.append(
                    {
                        "candidate_id": f"c{len(rows) + 1:04d}",
                        "constraint_type": "option_component",
                        "description_hint": f"组合规格组件候选：{component}",
                        "field_path": field_path,
                        "operator": "contains_component",
                        "expected_value": component_value,
                        "hardness_hint": (
                            "hard"
                            if atom.get("strength") in {"hard", "required"}
                            else "soft"
                        ),
                        "data_sources": [
                            str(atom.get("source") or "reward_v4"),
                            "derived.option_component",
                        ],
                        "selection_guidance": (
                            "仅当 Query 明确表达该组件语义时选择；该组件仍绑定"
                            " selected_option 中的完整实际选项值。"
                        ),
                    }
                )
    bundle = {
        "schema_version": RUBRIC_CANDIDATE_VERSION,
        "extractor_version": RUBRIC_EXTRACTOR_VERSION,
        "reward_version": REWARD_VERSION,
        "task_id": int(task_id),
        "query": str(query),
        "candidates": rows,
    }
    bundle["candidate_hash"] = stable_hash(bundle)
    return bundle
