import json

from shopping_grpo.collection.sft import build_sft_row
from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.evaluation.rollout import collect_for_task
from shopping_grpo.multiturn.tasks import (
    MULTITURN_TASK_SCHEMA, build_task_row, source_goal_hash,
)


def tool_message(name, arguments, call_id):
    return {
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


class Actor:
    def __init__(self):
        self.outputs = [
            tool_message("ask_shopper", {"question": "Which size?"}, "ask-1"),
            tool_message("search_products", {"query": "pillow"}, "search-1"),
        ]

    def complete(self, messages, tools):
        return self.outputs.pop(0)


class Shopper:
    call_count = 0

    def answer(self, question, context, history):
        assert context["instruction_full"] == "full private goal"
        self.call_count += 1
        return "child size"


class Env:
    def __init__(self, **kwargs):
        assert kwargs["multiturn"] is True
        self.shopper_context = {
            "instruction_full": "full private goal", "goal_options": ["child size"],
        }

    def reset(self, task_id, initial_request=None):
        return {"env_idx": 0, "instruction": initial_request}

    def step(self, action):
        assert action == "search[pillow]"
        return {"instruction": "done", "reward": 1.0, "done": True}

    def release(self):
        pass


def test_multiturn_rollout_asks_freely_and_keeps_private_goal_out_of_messages():
    task = {
        "schema_version": MULTITURN_TASK_SCHEMA, "task_id": 7,
        "initial_request": "I need a pillow",
    }
    trajectory = collect_for_task(
        task, client=Actor(), shopper=Shopper(), env_factory=Env, max_steps=3,
    )
    assert trajectory["status"] == "done"
    assert trajectory["shopper_questions"] == [
        {"question": "Which size?", "answer": "child size"}
    ]
    assert trajectory["actor_llm_calls"] == 2
    assert trajectory["shopper_llm_calls"] == 1
    assert "full private goal" not in json.dumps(trajectory["messages"])
    sft = build_sft_row(trajectory)
    assert sft["tools"][0]["function"]["name"] == "ask_shopper"
    assert any(m.get("name") == "ask_shopper" for m in sft["messages"])


def test_multiturn_client_removes_private_context_from_public_reset_result():
    private = {"instruction_full": "gold", "goal_options": ["option"]}

    def transport(url, payload, timeout):
        if payload["action"] == "reset":
            assert payload["if_multiturn"] is True
            return {"result": {"env_idx": 2, "instruction": "", "_shopper_context": private}}
        return {"result": {"message": "released"}}

    env = ShopAgentEnv(transport=transport, multiturn=True)
    result = env.reset(1, initial_request="public opening")
    assert result["instruction"] == "public opening"
    assert "_shopper_context" not in result
    assert env.shopper_context == private
    env.release()


def test_frozen_task_row_contains_hash_not_private_goal():
    context = {"instruction_full": "gold", "goal_options": ["option"]}
    row = build_task_row(3, "public", context, "model", "prompt-hash")
    assert row["source_goal_hash"] == source_goal_hash(context)
    assert "gold" not in json.dumps(row)


def test_question_limit_blocks_without_calling_shopper_again():
    actor = Actor()
    actor.outputs = [
        tool_message("ask_shopper", {"question": f"question-{index}"}, f"ask-{index}")
        for index in range(5)
    ]
    shopper = Shopper()
    task = {
        "schema_version": MULTITURN_TASK_SCHEMA, "task_id": 7,
        "initial_request": "I need a pillow",
    }
    trajectory = collect_for_task(
        task, client=actor, shopper=shopper, env_factory=Env,
        max_steps=6, max_shopper_questions=2,
    )
    assert trajectory["status"] == "invalid_action_limit"
    assert len(trajectory["shopper_questions"]) == 2
    assert trajectory["shopper_llm_calls"] == 2
    assert [item["reason"] for item in trajectory["blocked_tool_calls"]] == [
        "shopper_question_limit", "shopper_question_limit", "shopper_question_limit",
    ]
