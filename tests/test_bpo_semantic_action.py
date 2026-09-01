from types import SimpleNamespace

from shopping_grpo.training.bpo.semantic_action import canonical_semantic_action


def call(name, arguments):
    return SimpleNamespace(name=name, arguments=arguments)


def test_semantic_action_ignores_json_order_and_whitespace():
    left = canonical_semantic_action(
        [call("search_products", '{"query":"  red   running shoes ","page":1}')]
    )
    right = canonical_semantic_action(
        [call("SEARCH_PRODUCTS", '{"page":1,"query":"red running shoes"}')]
    )

    assert left is not None
    assert right is not None
    assert left.canonical_key == right.canonical_key
    assert left.sha256 == right.sha256


def test_semantic_action_distinguishes_tool_arguments():
    first = canonical_semantic_action([call("open_product", '{"asin":"A1"}')])
    second = canonical_semantic_action([call("open_product", '{"asin":"A2"}')])

    assert first is not None
    assert second is not None
    assert first.sha256 != second.sha256


def test_semantic_action_rejects_non_single_or_unparseable_calls():
    assert canonical_semantic_action([]) is None
    assert canonical_semantic_action([call("search_products", "{")]) is None
    assert canonical_semantic_action(
        [call("search_products", "{}"), call("next_page", "{}")]
    ) is None
    assert canonical_semantic_action(
        [call("unknown", "{}")], allowed_tools=["search_products"]
    ) is None


def test_semantic_action_validates_the_active_tool_schema():
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "open_product",
                "parameters": {
                    "type": "object",
                    "properties": {"asin": {"type": "string"}},
                    "required": ["asin"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    assert canonical_semantic_action(
        [call("open_product", '{"asin":"A1"}')], tool_schemas=schemas
    ) is not None
    assert canonical_semantic_action(
        [call("open_product", "{}")], tool_schemas=schemas
    ) is None
    assert canonical_semantic_action(
        [call("open_product", '{"asin":"A1","extra":true}')],
        tool_schemas=schemas,
    ) is None
