"""Compile atomic, strength-aware Reward v4 requirements."""

from __future__ import annotations

import re

from web_agent_site.engine.price_constraints import (
    PRICE_CONSTRAINT_VERSION,
    compile_price_constraint,
)
from web_agent_site.engine.reward_features import compile_reward_features


REWARD_FEATURE_VERSION = "shopping-reward-features-v2"
CONSTRAINT_ATOM_VERSION = "shopping-constraint-atoms-v1"
_HARD_MARKERS = ("必须", "一定", "务必", "硬性", "不可缺少", "不能没有", "只要")
_SOFT_MARKERS = ("最好", "优先", "尽量", "希望", "偏好", "倾向", "可以的话")
_STRENGTH_WEIGHTS = {"hard": 1.0, "required": 1.0, "soft": 0.5}


def _clause_for_value(instruction: str, value: object) -> str:
    needle = str(value or "").strip()
    for clause in re.split(r"[，。；;,.!?！？]", instruction):
        if needle and needle in clause:
            return clause
    return instruction


def _strength(instruction: str, value: object) -> str:
    clause = _clause_for_value(instruction, value)
    if any(marker in clause for marker in _HARD_MARKERS):
        return "hard"
    if any(marker in clause for marker in _SOFT_MARKERS):
        return "soft"
    return "required"


def _atom(
    atom_id: str,
    dimension: str,
    strength: str,
    requirement: object,
    source: str,
) -> dict:
    return {
        "atom_id": atom_id,
        "dimension": dimension,
        "strength": strength,
        "weight": _STRENGTH_WEIGHTS[strength],
        "requirement": requirement,
        "source": source,
    }


def compile_reward_features_v2(
    instruction_record: object,
    target_product: object,
) -> dict:
    """Extend v1 features with auditable requirement atoms and price semantics."""

    instruction = (
        instruction_record if isinstance(instruction_record, dict) else {}
    )
    product = target_product if isinstance(target_product, dict) else {}
    text = str(instruction.get("instruction") or "")
    legacy = compile_reward_features(instruction, product)
    atoms = [
        _atom(
            "category:0",
            "category",
            "hard",
            legacy.get("category"),
            "task.target_product.category",
        )
    ]
    for dimension, values in (
        ("brand", legacy.get("expected_brand") or []),
        ("model", legacy.get("expected_model") or []),
        ("core_function", legacy.get("expected_core_functions") or []),
    ):
        for index, value in enumerate(values):
            atoms.append(
                _atom(
                    f"{dimension}:{index}",
                    dimension,
                    _strength(text, value),
                    value,
                    f"instruction.{dimension}",
                )
            )
    for index, (axis, requirement) in enumerate(
        sorted((legacy.get("required_options_by_key") or {}).items())
    ):
        value = requirement.get("value")
        atoms.append(
            _atom(
                f"option:{index}",
                "option",
                _strength(text, value),
                {"axis": axis, **requirement},
                "instruction.instruction_options",
            )
        )
    for index, requirement in enumerate(
        legacy.get("unresolved_option_requirements") or []
    ):
        value = requirement.get("value")
        atoms.append(
            _atom(
                f"unresolved_option:{index}",
                "option",
                _strength(text, value),
                requirement,
                "instruction.instruction_options",
            )
        )
    price = compile_price_constraint(text)
    if price:
        price_strength = "soft" if price["kind"] == "soft_target" else "hard"
        atoms.append(
            _atom(
                "price:0",
                "price",
                price_strength,
                price,
                "instruction",
            )
        )
    return {
        **legacy,
        "reward_feature_version": REWARD_FEATURE_VERSION,
        "constraint_atom_version": CONSTRAINT_ATOM_VERSION,
        "price_constraint_version": PRICE_CONSTRAINT_VERSION,
        "price_constraint": price,
        "constraint_atoms": atoms,
    }
