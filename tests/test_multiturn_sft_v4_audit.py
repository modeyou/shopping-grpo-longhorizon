import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
sys.path.append(str(SHOP_ENV))

from shopping_grpo.collection.multiturn_sft_v4 import (
    cross_policy_overlaps,
    rescore_terminal_purchase_v4,
)


def _product():
    return {
        "asin": "111",
        "title": "智能热洗洗地机 白色款",
        "brand": "示例",
        "shop_name": "示例旗舰店",
        "category": "家电›清洁电器›洗地机",
        "attribute": ["智能洗地", "支持热洗"],
        "pricing": [230],
        "customization_options": {
            "颜色分类": [{"value": "白色", "price": 230}],
        },
        "instructions": [
            {
                "instruction": "必须支持热洗，选择白色，价格在230元左右",
                "attributes": ["热洗"],
                "instruction_options": ["白色"],
            }
        ],
    }


def _trajectory(*, asin="111", option="白色", price=230):
    return {
        "trajectory_id": "teacher-0",
        "task_id": 0,
        "teacher_policy": "complete-no-ask-v1",
        "terminal_result": {
            "purchase": {
                "asin": asin,
                "options": {"颜色分类": option},
                "price": price,
            },
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
            },
        },
    }


def test_rescore_uses_actual_terminal_options_and_price():
    detail, reasons, audit = rescore_terminal_purchase_v4(
        _trajectory(), [_product()]
    )

    assert reasons == []
    assert detail["reward_version"] == "shopsimulator-reward-v4"
    assert detail["reward_type"] == "gold_purchase"
    assert detail["reward_valid"] is True
    assert audit["selected_options"] == {"颜色分类": "白色"}
    assert audit["stored_price"] == 230
    assert audit["resolved_price"] == 230


def test_rescore_rejects_a_purchase_that_is_not_the_target_asin():
    detail, reasons, audit = rescore_terminal_purchase_v4(
        _trajectory(asin="wrong"), [_product()]
    )

    assert detail["reward_type"] == "gold_purchase"
    assert "terminal_purchase_not_target_asin" in reasons
    assert audit["target_asin_match"] is False


def test_cross_policy_overlap_report_preserves_exact_task_ids():
    overlaps = cross_policy_overlaps(
        {
            "complete-no-ask-v1": [{"task_id": 1}, {"task_id": 2}],
            "composite-replay-v1": [{"task_id": 2}, {"task_id": 3}],
            "autonomous-gap-v1": [{"task_id": 3}],
        }
    )

    assert overlaps[
        "complete-no-ask-v1__composite-replay-v1"
    ] == {"tasks": 1, "task_ids": [2]}
    assert overlaps[
        "composite-replay-v1__autonomous-gap-v1"
    ] == {"tasks": 1, "task_ids": [3]}
