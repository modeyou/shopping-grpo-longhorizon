import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "environments/ShopSimulator/shop_env"))

from shopping_grpo.multiturn.benchmark import (
    audit_gold_task_version,
    curate_task_ids,
)


def _product():
    task = {
        "instruction": "必须支持热洗，最好选择白色，价格在230元左右",
        "attributes": ["热洗"],
        "instruction_options": ["白色"],
    }
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
        "instructions": [task],
        "query": "洗地机",
        "full_description": "支持热洗",
        "small_description": ["智能洗地"],
    }


def test_gold_audit_can_dual_score_the_same_task():
    product = _product()
    v3 = audit_gold_task_version(
        product, 7, reward_version="shopsimulator-reward-v3"
    )
    v4 = audit_gold_task_version(
        product, 7, reward_version="shopsimulator-reward-v4"
    )

    assert v3["eligible"] is True
    assert v4["eligible"] is True
    assert v4["audit"]["constraint_atom_count"] == 4


def test_batch_curator_records_selected_reward_version():
    selected, audits, _ = curate_task_ids(
        products=[_product()],
        candidates=[0],
        reserve=[],
        reward_version="shopsimulator-reward-v4",
    )

    assert selected == [0]
    assert audits[0]["reward_version"] == "shopsimulator-reward-v4"
