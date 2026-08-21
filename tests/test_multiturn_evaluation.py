import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "environments/ShopSimulator/shop_env"))

from shopping_grpo.multiturn.benchmark import (
    PRICE_HINT,
    audit_gold_task,
    read_task_ids,
)
from shopping_grpo.evaluation.rollout import (
    MULTITURN_EVALUATION_SYSTEM_PROMPT,
    collect_for_task,
)
from shopping_grpo.multiturn.tasks import MULTITURN_TASK_SCHEMA, source_goal_hash


def _product(*, instruction="预算不超过20元", pricing=(10.0,)):
    return {
        "asin": "A1",
        "title": "测试商品",
        "full_description": "测试功能",
        "small_description": ["测试功能"],
        "attribute": ["测试功能"],
        "pricing": list(pricing),
        "category": "测试 › 商品",
        "shop_name": "测试店",
        "customization_options": {},
        "instructions": [{
            "instruction": instruction,
            "instruction_simple": instruction,
            "attributes": ["测试功能"],
            "instruction_options": [],
        }],
    }


def test_price_hint_detects_natural_price_phrases():
    assert PRICE_HINT.search("价格在230元左右")
    assert PRICE_HINT.search("控制在15元以内")


def test_gold_audit_accepts_reachable_task():
    result = audit_gold_task(_product(), 0)
    assert result["eligible"] is True
    assert result["reasons"] == []
    assert result["audit"]["reward_type"] == "gold_purchase"


def test_gold_audit_rejects_unresolved_price():
    result = audit_gold_task(_product(pricing=(10.0, 20.0)), 0)
    assert result["eligible"] is False
    assert "gold_variant_price_unresolved" in result["reasons"]


def test_read_task_ids_rejects_duplicates(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps({"task_id": 1}) + "\n" + json.dumps({"task_id": 1}) + "\n",
        encoding="utf-8",
    )
    try:
        read_task_ids(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate task IDs must be rejected")


def _tool(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }


class _Actor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        return self.outputs.pop(0)


class _Shopper:
    call_count = 0

    def answer_audited(self, question, context, omitted_facts, history):
        self.call_count += 1
        return {"answer": "预算20元。", "used_facts": ["预算20元"]}


class _EvaluationEnv:
    def __init__(self, **kwargs):
        assert kwargs["multiturn"] is True
        self.shopper_context = {
            "instruction_full": "需要测试商品，预算20元",
            "goal_options": [],
        }
        self.actions = []

    def reset(self, task_id, initial_request=None):
        return {"env_idx": 0, "instruction": initial_request or ""}

    def step(self, action):
        self.actions.append(action)
        return {
            "instruction": "done",
            "reward": 1.0,
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
        }

    def release(self):
        pass


def _gap_task():
    context = {
        "instruction_full": "需要测试商品，预算20元",
        "goal_options": [],
    }
    return {
        "schema_version": MULTITURN_TASK_SCHEMA,
        "task_id": 0,
        "initial_request": "需要测试商品",
        "source_goal_hash": source_goal_hash(context),
        "opening_audit": {
            "omitted_dimensions": ["预算"],
            "omitted_facts": ["预算20元"],
        },
    }


def test_gap_ask_disabled_uses_opening_without_exposing_ask_tool():
    actor = _Actor([_tool("search_products", {"query": "测试商品"}, "search")])
    trajectory = collect_for_task(
        _gap_task(),
        client=actor,
        env_factory=_EvaluationEnv,
        max_steps=1,
        evaluation_condition="gap-ask-disabled",
    )
    tools = [item["function"]["name"] for item in actor.requests[0]["tools"]]
    assert trajectory["status"] == "done"
    assert trajectory["interaction_mode"] == "gap-ask-disabled"
    assert trajectory["messages"][1]["content"] == "需要测试商品"
    assert trajectory["messages"][0]["content"] == MULTITURN_EVALUATION_SYSTEM_PROMPT
    assert "ask_shopper" not in tools


def test_gap_ask_enabled_has_separate_question_and_shop_budgets():
    actor = _Actor([
        _tool("ask_shopper", {"question": "预算是多少？"}, "ask-1"),
        _tool("ask_shopper", {"question": "确认预算？"}, "ask-2"),
        _tool("search_products", {"query": "测试商品 20元"}, "search"),
    ])
    trajectory = collect_for_task(
        _gap_task(),
        client=actor,
        shopper=_Shopper(),
        env_factory=_EvaluationEnv,
        max_steps=1,
        max_shopper_questions=2,
        evaluation_condition="gap-ask-enabled",
    )
    assert trajectory["status"] == "done"
    assert [step["tool_name"] for step in trajectory["steps"]] == [
        "ask_shopper", "ask_shopper", "search_products",
    ]
    assert trajectory["source_goal_verified"] is True


def test_complete_ask_enabled_uses_private_full_goal_as_public_request():
    actor = _Actor([_tool("search_products", {"query": "测试商品"}, "search")])
    trajectory = collect_for_task(
        {"task_id": 0},
        client=actor,
        shopper=_Shopper(),
        env_factory=_EvaluationEnv,
        max_steps=1,
        evaluation_condition="complete-ask-enabled",
    )
    assert trajectory["status"] == "done"
    assert trajectory["messages"][1]["content"] == "需要测试商品，预算20元"
    tools = [item["function"]["name"] for item in actor.requests[0]["tools"]]
    assert "ask_shopper" in tools
