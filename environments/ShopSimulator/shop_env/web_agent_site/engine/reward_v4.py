"""Atomic, strength-aware terminal Reward v4 for ShopSimulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from web_agent_site.engine.comparators import (
    COMPARATOR_VERSION,
    FAIL,
    PASS,
    UNVERIFIABLE,
    compare_category,
    compare_core_functions,
    compare_model,
    comparison,
)
from web_agent_site.engine.reward import _brand_gate
from web_agent_site.engine.reward_features import normalize_option_text
from web_agent_site.engine.reward_features_v2 import (
    CONSTRAINT_ATOM_VERSION,
    REWARD_FEATURE_VERSION,
)
from web_agent_site.engine.variant_price import (
    VARIANT_PRICE_VERSION,
    candidate_options_for_evaluation,
    compare_required_options,
    resolve_variant_price,
)


REWARD_VERSION = "shopsimulator-reward-v4"
DEFAULT_REWARDS = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.55,
    "partial_purchase_base": -0.30,
    "partial_purchase_scale": 0.55,
    "partial_purchase_cap": 0.25,
    "graceful_stop": -0.15,
    "early_abstain": -0.35,
    "max_steps": -0.50,
    "repeat_loop": -0.65,
    "wrong_purchase": -0.85,
    "reward_unverifiable": 0.0,
}
KNOWN_ACCEPTABLE_MATCH_THRESHOLD = 0.70
KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD = 0.75


@dataclass(frozen=True)
class RewardResultV4:
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    target_asin_match: bool
    hard_gates: dict
    weighted_score: float
    evidence: dict

    def to_dict(self):
        payload = asdict(self)
        scoring = self.evidence.get("constraint_scoring") or {}
        payload.update(
            {
                "reward_version": REWARD_VERSION,
                "terminal_utility": self.reward,
                "purchase_success": self.reward_type
                in {"gold_purchase", "valid_alternative_purchase"},
                "sampling_invalid": not self.reward_valid,
                "evidence_coverage": float(
                    scoring.get("evidence_coverage", 0.0)
                ),
                "constraint_scores": {
                    item["atom_id"]: float(item["score"])
                    for item in scoring.get("atoms") or []
                },
            }
        )
        return payload


def _price_result(requirement: dict, price_resolution: object) -> dict:
    actual = (
        price_resolution.get("price")
        if isinstance(price_resolution, dict)
        else None
    )
    if (
        not isinstance(price_resolution, dict)
        or price_resolution.get("status") != PASS
    ):
        return comparison(
            UNVERIFIABLE,
            comparator="price_constraint_v2",
            required=requirement,
            actual=actual,
            source_field="variant_price",
            evidence=price_resolution,
        )
    try:
        actual = float(actual)
    except (TypeError, ValueError):
        actual = math.nan
    if not math.isfinite(actual) or actual < 0:
        status = UNVERIFIABLE
    else:
        kind = requirement.get("kind")
        lower = requirement.get("lower")
        upper = requirement.get("upper")
        if kind == "hard_max":
            passed = actual <= float(upper)
        elif kind == "hard_min":
            passed = actual >= float(lower)
        elif kind in {"hard_range", "soft_target"}:
            passed = float(lower) <= actual <= float(upper)
        else:
            return comparison(
                UNVERIFIABLE,
                comparator="price_constraint_v2",
                required=requirement,
                actual=actual,
                source_field="variant_price",
                evidence={"reason": "unsupported_price_constraint"},
            )
        status = PASS if passed else FAIL
    return comparison(
        status,
        comparator="price_constraint_v2",
        required=requirement,
        actual=actual,
        source_field="variant_price",
        evidence={
            "price_resolution": price_resolution,
            "distance_from_target": (
                abs(actual - float(requirement["target"]))
                if math.isfinite(actual) and requirement.get("target") is not None
                else None
            ),
        },
    )


def _unresolved_option_result(
    requirement: dict, selected_options: object
) -> dict:
    selected_values = {
        normalize_option_text(value)
        for value in (
            selected_options.values()
            if isinstance(selected_options, dict)
            else ()
        )
    }
    required = requirement.get("value")
    return comparison(
        PASS if normalize_option_text(required) in selected_values else FAIL,
        comparator="exact_option_value_fallback_v1",
        required=required,
        actual=list(
            selected_options.values()
            if isinstance(selected_options, dict)
            else ()
        ),
        source_field="instruction_options",
        evidence=requirement,
    )


def _evaluate_atom(
    atom: dict,
    product: dict,
    selected_options: object,
    price_resolution: object,
) -> dict:
    dimension = atom.get("dimension")
    requirement = atom.get("requirement")
    if dimension == "category":
        result = compare_category(requirement, product)
    elif dimension == "brand":
        result = _brand_gate(requirement, product)
    elif dimension == "model":
        result = compare_model(requirement, product)
    elif dimension == "core_function":
        result = compare_core_functions([requirement], product)
    elif dimension == "option":
        if isinstance(requirement, dict) and requirement.get("axis"):
            axis = requirement["axis"]
            result = compare_required_options(
                product,
                {axis: requirement},
                selected_options,
            )
        else:
            result = _unresolved_option_result(
                requirement if isinstance(requirement, dict) else {},
                selected_options,
            )
    elif dimension == "price":
        result = _price_result(
            requirement if isinstance(requirement, dict) else {},
            price_resolution,
        )
    else:
        result = comparison(
            UNVERIFIABLE,
            comparator="constraint_atom_v1",
            required=requirement,
            actual=None,
            source_field=str(dimension),
            evidence={"reason": "unsupported_dimension"},
        )
    return {
        "atom_id": atom.get("atom_id"),
        "dimension": dimension,
        "strength": atom.get("strength"),
        "weight": float(atom.get("weight", 0.0)),
        "status": result["status"],
        "score": 1.0 if result["status"] == PASS else 0.0,
        "comparison": result,
    }


def _constraint_scoring(
    product: dict,
    goal: dict,
    selected_options: object,
    price_resolution: object,
) -> dict:
    atoms = goal.get("constraint_atoms") or []
    results = [
        _evaluate_atom(
            atom,
            product,
            selected_options,
            price_resolution,
        )
        for atom in atoms
        if isinstance(atom, dict)
    ]
    total_weight = sum(item["weight"] for item in results)
    passed_weight = sum(
        item["weight"] * item["score"] for item in results
    )
    verifiable_weight = sum(
        item["weight"]
        for item in results
        if item["status"] != UNVERIFIABLE
    )
    match_score = passed_weight / total_weight if total_weight else 0.0
    coverage = verifiable_weight / total_weight if total_weight else 0.0
    return {
        "atom_version": goal.get("constraint_atom_version"),
        "active_atom_count": len(results),
        "total_weight": total_weight,
        "match_score": match_score,
        "evidence_coverage": coverage,
        "all_satisfied": bool(results)
        and all(item["status"] == PASS for item in results),
        "hard_failed": any(
            item["strength"] == "hard" and item["status"] == FAIL
            for item in results
        ),
        "hard_unverifiable": any(
            item["strength"] == "hard"
            and item["status"] == UNVERIFIABLE
            for item in results
        ),
        "atoms": results,
    }


def _explicit_price_resolution(price: object) -> dict:
    try:
        value = float(price)
    except (TypeError, ValueError):
        value = math.nan
    return {
        "status": PASS if math.isfinite(value) and value >= 0 else UNVERIFIABLE,
        "price": value if math.isfinite(value) else None,
        "version": VARIANT_PRICE_VERSION,
        "method": "explicit_test_price",
        "evidence": {},
    }


def evaluate_purchase(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: dict | None = None,
    price: object = None,
    rewards: dict[str, float] | None = None,
) -> RewardResultV4:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if price_resolution is None:
        price_resolution = (
            _explicit_price_resolution(price)
            if price is not None
            else resolve_variant_price(product, selected_options)
        )
    scoring = _constraint_scoring(
        product, goal, selected_options, price_resolution
    )
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    if scoring["hard_unverifiable"]:
        reward_type = "reward_unverifiable"
        reward_valid = False
        reward = values[reward_type]
    elif scoring["hard_failed"]:
        reward_type = "wrong_purchase"
        reward_valid = True
        reward = values[reward_type]
    elif scoring["all_satisfied"]:
        reward_type = (
            "gold_purchase" if asin_match else "valid_alternative_purchase"
        )
        reward_valid = True
        reward = values[reward_type]
    else:
        reward_type = "partial_alternative_purchase"
        reward_valid = True
        reward = min(
            values["partial_purchase_cap"],
            values["partial_purchase_base"]
            + values["partial_purchase_scale"] * scoring["match_score"],
        )
    hard_gates = {
        item["atom_id"]: item["comparison"]
        for item in scoring["atoms"]
        if item["strength"] == "hard"
    }
    return RewardResultV4(
        reward=float(reward),
        reward_type=reward_type,
        reward_valid=reward_valid,
        termination_reason=reward_type,
        target_asin_match=asin_match,
        hard_gates=hard_gates,
        weighted_score=float(scoring["match_score"]),
        evidence={
            "reward_feature_version": goal.get("reward_feature_version"),
            "expected_reward_feature_version": REWARD_FEATURE_VERSION,
            "constraint_atom_version": CONSTRAINT_ATOM_VERSION,
            "comparator_version": COMPARATOR_VERSION,
            "variant_price_version": VARIANT_PRICE_VERSION,
            "price_resolution": price_resolution,
            "constraint_scoring": scoring,
        },
    )


def evaluate_candidate_eligibility(product: dict, goal: dict) -> dict:
    selected, option_resolution = candidate_options_for_evaluation(
        product, goal.get("required_options_by_key")
    )
    price_resolution = resolve_variant_price(product, selected)
    scoring = _constraint_scoring(
        product, goal, selected, price_resolution
    )
    known_acceptable = (
        not scoring["hard_failed"]
        and not scoring["hard_unverifiable"]
        and scoring["match_score"] >= KNOWN_ACCEPTABLE_MATCH_THRESHOLD
        and scoring["evidence_coverage"]
        >= KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD
    )
    return {
        "status": PASS if known_acceptable else FAIL,
        "known_acceptable": known_acceptable,
        "known_valid": known_acceptable,
        "selected_options": selected,
        "option_resolution": option_resolution,
        "price_resolution": price_resolution,
        "hard_gates": {
            item["atom_id"]: item["comparison"]
            for item in scoring["atoms"]
            if item["strength"] == "hard"
        },
        "match_score": scoring["match_score"],
        "evidence_coverage": scoring["evidence_coverage"],
    }


def evaluate_abstain(
    *,
    effective_result_sets: int,
    opened_candidates: int,
    known_acceptable_candidates: int | None = None,
    known_valid_candidates: int | None = None,
    rewards: dict[str, float] | None = None,
) -> RewardResultV4:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    known_count = int(
        known_acceptable_candidates
        if known_acceptable_candidates is not None
        else known_valid_candidates or 0
    )
    eligible = (
        int(effective_result_sets) >= 2
        and int(opened_candidates) >= 2
        and known_count == 0
    )
    reward_type = "graceful_stop" if eligible else "early_abstain"
    return RewardResultV4(
        reward=float(values[reward_type]),
        reward_type=reward_type,
        reward_valid=True,
        termination_reason=reward_type,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={
            "eligible": eligible,
            "effective_result_sets": int(effective_result_sets),
            "opened_candidates": int(opened_candidates),
            "known_acceptable_candidates": known_count,
            "constraint_scoring": {},
        },
    )


def fixed_termination(
    reason: str,
    rewards: dict[str, float] | None = None,
) -> RewardResultV4:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if reason not in {"repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    return RewardResultV4(
        reward=float(values[reason]),
        reward_type=reason,
        reward_valid=True,
        termination_reason=reason,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={"constraint_scoring": {}},
    )
