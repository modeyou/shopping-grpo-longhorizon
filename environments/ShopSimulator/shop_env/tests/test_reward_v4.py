from pathlib import Path

from web_agent_site.engine.config import load_config
from web_agent_site.engine.price_constraints import compile_price_constraint
from web_agent_site.engine.reward import (
    DEFAULT_REWARDS as V3_REWARDS,
    REWARD_VERSION as V3_VERSION,
)
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.reward_features_v2 import compile_reward_features_v2
from web_agent_site.engine.reward_registry import evaluate_purchase as dispatch_purchase
from web_agent_site.engine.reward_v4 import (
    DEFAULT_REWARDS as V4_REWARDS,
    REWARD_VERSION as V4_VERSION,
    evaluate_purchase,
)


ROOT = Path(__file__).resolve().parents[1]


def product(asin="111", *, attributes=None):
    return {
        "asin": asin,
        "title": "智能热洗洗地机 白色款",
        "brand": "示例",
        "shop_name": "示例旗舰店",
        "category": "家电›清洁电器›洗地机",
        "attribute": attributes or ["智能洗地", "支持热洗"],
        "pricing": [230],
        "customization_options": {
            "颜色分类": [{"value": "白色", "price": 230}],
        },
    }


def instruction(text="必须支持热洗，最好选择白色，价格在230元左右"):
    return {
        "instruction": text,
        "attributes": ["热洗"],
        "instruction_options": ["白色"],
    }


def goal(text=None):
    target = product()
    record = instruction(text) if text is not None else instruction()
    return {
        "asin": target["asin"],
        "category": target["category"],
        **compile_reward_features_v2(record, target),
    }


def test_price_constraint_distinguishes_max_range_and_soft_target():
    maximum = compile_price_constraint("预算不超过230元")
    interval = compile_price_constraint("价格在200到250元")
    target = compile_price_constraint("预算大约两百三十元")
    scaled_interval = compile_price_constraint("预算10到20万")
    open_lower = compile_price_constraint("预算4k+")

    assert maximum["kind"] == "hard_max"
    assert maximum["upper"] == 230
    assert interval["kind"] == "hard_range"
    assert (interval["lower"], interval["upper"]) == (200, 250)
    assert target["kind"] == "soft_target"
    assert target["target"] == 230
    assert (target["lower"], target["upper"]) == (207, 253)
    assert (scaled_interval["lower"], scaled_interval["upper"]) == (
        100000,
        200000,
    )
    assert open_lower["kind"] == "hard_min"
    assert open_lower["lower"] == 4000


def test_compiler_emits_hard_required_and_soft_atoms():
    features = compile_reward_features_v2(instruction(), product())
    strengths = {
        (atom["dimension"], atom["strength"])
        for atom in features["constraint_atoms"]
    }

    assert ("category", "hard") in strengths
    assert ("core_function", "hard") in strengths
    assert ("option", "soft") in strengths
    assert ("price", "soft") in strengths


def test_v4_gold_purchase_has_atom_evidence():
    result = evaluate_purchase(
        product(),
        goal(),
        selected_options={"颜色分类": "白色"},
        price=230,
    )

    assert result.reward_type == "gold_purchase"
    assert result.reward == 1.0
    detail = result.to_dict()
    assert detail["reward_version"] == V4_VERSION
    assert detail["evidence_coverage"] == 1.0
    assert set(detail["constraint_scores"]) == {
        "category:0",
        "core_function:0",
        "option:0",
        "price:0",
    }


def test_soft_price_miss_reduces_score_but_does_not_block_gold():
    result = evaluate_purchase(
        product(),
        goal(),
        selected_options={"颜色分类": "白色"},
        price=280,
    )

    assert result.reward_type == "gold_purchase"
    assert result.reward_valid is True
    assert result.weighted_score < 1.0
    scoring = result.evidence["constraint_scoring"]
    assert scoring["strict_satisfied"] is True
    assert scoring["all_satisfied"] is False


def test_hard_budget_or_hard_function_failure_is_wrong_purchase():
    budget_goal = goal("支持热洗，预算不超过220元")
    over_budget = evaluate_purchase(
        product(),
        budget_goal,
        selected_options={"颜色分类": "白色"},
        price=230,
    )
    missing_product = product(attributes=["普通清洁"])
    missing_product["title"] = "普通洗地机 白色款"
    missing_function = evaluate_purchase(
        missing_product,
        goal(),
        selected_options={"颜色分类": "白色"},
        price=230,
    )

    assert over_budget.reward_type == "wrong_purchase"
    assert missing_function.reward_type == "wrong_purchase"


def test_registry_keeps_v3_default_and_dispatches_v4_explicitly():
    target = product()
    legacy_goal = {
        "asin": target["asin"],
        "category": target["category"],
        "price_upper": 230,
        **compile_reward_features(instruction(), target),
    }
    v3 = dispatch_purchase(
        target,
        legacy_goal,
        selected_options={"颜色分类": "白色"},
        price=230,
        rewards={"version": V3_VERSION, **V3_REWARDS},
    )
    v4 = dispatch_purchase(
        target,
        goal(),
        selected_options={"颜色分类": "白色"},
        price=230,
        rewards={"version": V4_VERSION, **V4_REWARDS},
    )

    assert v3.to_dict()["reward_version"] == V3_VERSION
    assert v4.to_dict()["reward_version"] == V4_VERSION


def test_both_environment_configs_validate():
    assert load_config(ROOT / "configs/environment.json")["reward"][
        "version"
    ] == V3_VERSION
    assert load_config(ROOT / "configs/environment-v4.json")["reward"][
        "version"
    ] == V4_VERSION
