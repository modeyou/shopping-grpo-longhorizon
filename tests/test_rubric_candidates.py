from shopping_grpo.evaluation.candidates import extract_rubric_candidates
from shopping_grpo.multiturn.benchmark import load_products


PRODUCTS_PATH = (
    "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
)


def _candidates(task_id):
    products = load_products(PRODUCTS_PATH)
    product = products[task_id]
    instruction = next(
        item for item in product["instructions"] if item.get("attributes")
    )
    return extract_rubric_candidates(
        task_id=task_id,
        query=instruction["instruction"],
        instruction=instruction,
        product=product,
    )


def test_reward_v4_candidates_preserve_soft_price_and_combined_option():
    bundle = _candidates(13378)
    assert bundle["reward_version"] == "shopsimulator-reward-v4"
    price = next(row for row in bundle["candidates"] if row["constraint_type"] == "price")
    option = next(row for row in bundle["candidates"] if row["constraint_type"] == "option")
    assert price["hardness_hint"] == "soft"
    assert price["expected_value"]["source_text"] == "价格在60元左右"
    assert option["expected_value"]["value"] == "免组装款-单座+绑带+送布兜-B15"
    components = {
        row["expected_value"]["component"]
        for row in bundle["candidates"]
        if row["constraint_type"] == "option_component"
    }
    assert {"免组装款", "单座", "绑带", "送布兜"}.issubset(components)


def test_reward_v4_candidates_are_a_superset_not_query_truth():
    bundle = _candidates(22262)
    values = [row["expected_value"] for row in bundle["candidates"]]
    assert "满瘤疤" in values
    assert "太行" in values
    assert any(
        isinstance(value, dict) and value.get("value") == "12478"
        for value in values
    )
    assert all("selection_guidance" in row for row in bundle["candidates"])
