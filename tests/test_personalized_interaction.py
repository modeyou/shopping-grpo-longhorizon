import json
import unittest

from shopping_grpo.collection.sft import build_sft_row
from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.tools import PERSONALIZED_SHOP_TOOL_SCHEMAS
from shopping_grpo.evaluation.rollout import (
    PERSONALIZED_SYSTEM_PROMPT,
    _history_reject_reason,
    collect_for_task,
)
from shopping_grpo.personalization import LLMShopper


def assistant_tool(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": [dict(row) for row in messages], "tools": tools})
        return self.responses.pop(0)


class FakePersonaEnv:
    def __init__(self, **kwargs):
        self.persona = kwargs.get("persona")
        self.shopper_context = {
            "instruction_full": "完整秘密要求：预算不超过70元",
            "instruction_simple": "购买卷发片",
            "user_persona": {"budget": "节俭", "__reasoning__": "private"},
        }
        self.actions = []
        self.released = False

    def reset(self, task_id):
        return {
            "env_idx": 0,
            "instruction": "我想买一款卷发片",
            "persona_mode": True,
            "user_persona": {"budget": "节俭"},
        }

    def step(self, action):
        self.actions.append(action)
        return {
            "instruction": '商品详情\n\n可点击的按钮: ["Buy Now"]',
            "reward": 0.0,
            "done": False,
        }

    def release(self):
        self.released = True


class DeterministicShopper:
    def __init__(self):
        self.calls = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return "预算不超过70元。"


class PersonalizedInteractionTest(unittest.TestCase):
    def test_personalized_prompt_requires_profile_checklist_and_early_questions(self):
        self.assertIn("近期搜索词", PERSONALIZED_SYSTEM_PROMPT)
        self.assertIn("尽量在首次搜索前问", PERSONALIZED_SYSTEM_PROMPT)
        self.assertIn("不得询问请求或画像已经给出的信息", PERSONALIZED_SYSTEM_PROMPT)
        self.assertIn("及时购买", PERSONALIZED_SYSTEM_PROMPT)
        self.assertIn("可以重新打开一次", PERSONALIZED_SYSTEM_PROMPT)

    def test_http_client_keeps_private_context_out_of_reset_result(self):
        calls = []

        def transport(url, payload, timeout):
            calls.append(payload)
            return {
                "result": {
                    "env_idx": 4,
                    "instruction": "公开请求",
                    "persona_mode": True,
                    "user_persona": {"style": "简约"},
                    "_shopper_context": {"instruction_full": "私有完整请求"},
                }
            }

        env = ShopAgentEnv(transport=transport, persona=True)
        public = env.reset(9)

        self.assertNotIn("_shopper_context", public)
        self.assertNotIn("私有完整请求", json.dumps(public, ensure_ascii=False))
        self.assertEqual(env.shopper_context["instruction_full"], "私有完整请求")
        self.assertEqual(calls[0], {"action": "reset", "idx": 9, "if_persona": True})

    def test_ask_user_answer_enters_actor_trace_without_private_goal(self):
        actor = SequenceClient(
            [
                assistant_tool("ask_user", {"question": "预算上限是多少？"}, "ask-1"),
                assistant_tool("search_products", {"query": "卷发片"}, "search-1"),
            ]
        )
        shopper = DeterministicShopper()
        env = FakePersonaEnv()

        trajectory = collect_for_task(
            {
                "task_id": 9,
                "prompt": [{"role": "user", "content": "旧数据中的完整秘密要求"}],
            },
            client=actor,
            env_factory=lambda **kwargs: env,
            max_steps=2,
            persona=True,
            shopper=shopper,
        )

        self.assertEqual(trajectory["status"], "max_steps")
        self.assertEqual(
            trajectory["user_questions"],
            [{"question": "预算上限是多少？", "answer": "预算不超过70元。"}],
        )
        self.assertEqual(env.actions, ["search[卷发片]"])
        self.assertTrue(env.released)
        self.assertEqual(shopper.calls[0]["context"]["instruction_full"], "完整秘密要求：预算不超过70元")
        actor_trace = json.dumps(trajectory["messages"], ensure_ascii=False)
        self.assertIn("预算不超过70元。", actor_trace)
        self.assertNotIn("完整秘密要求", actor_trace)
        self.assertNotIn("旧数据中的完整秘密要求", actor_trace)
        self.assertNotIn("__reasoning__", actor_trace)

    def test_question_limit_blocks_third_question_without_touching_environment(self):
        actor = SequenceClient(
            [
                assistant_tool("ask_user", {"question": "预算是多少？"}, "ask-1"),
                assistant_tool("ask_user", {"question": "颜色偏好是什么？"}, "ask-2"),
                assistant_tool("ask_user", {"question": "还有品牌要求吗？"}, "ask-3"),
                assistant_tool("search_products", {"query": "卷发片"}, "search-1"),
            ]
        )
        shopper = DeterministicShopper()
        env = FakePersonaEnv()

        trajectory = collect_for_task(
            {"task_id": 9},
            client=actor,
            env_factory=lambda **kwargs: env,
            max_steps=3,
            persona=True,
            shopper=shopper,
            max_user_questions=2,
        )

        self.assertEqual(len(trajectory["user_questions"]), 2)
        self.assertEqual(len(shopper.calls), 2)
        self.assertEqual(env.actions, ["search[卷发片]"])
        self.assertEqual(
            trajectory["blocked_tool_calls"][0]["reason"],
            "user_question_limit_reached",
        )

    def test_llm_shopper_uses_exactly_one_completion_and_no_tools(self):
        client = SequenceClient([{"role": "assistant", "content": "我希望控制在70元以内。"}])
        shopper = LLMShopper(client)

        answer = shopper.answer(
            question="预算是多少？",
            context={"instruction_full": "预算不超过70元", "user_persona": {}},
        )

        self.assertEqual(answer, "我希望控制在70元以内。")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["tools"], [])

    def test_history_guard_allows_reopening_an_inspected_product(self):
        steps = [
            {
                "tool_name": "open_product",
                "parameters": {"asin": "A1"},
            },
            {"tool_name": "back_to_search", "parameters": {}},
        ]

        self.assertIsNone(
            _history_reject_reason("open_product", {"asin": "A1"}, steps)
        )
        self.assertIsNone(
            _history_reject_reason("open_product", {"asin": "A2"}, steps)
        )

    def test_history_guard_blocks_same_option_on_same_product_only(self):
        steps = [
            {"tool_name": "open_product", "parameters": {"asin": "A1"}},
            {"tool_name": "select_option", "parameters": {"value": "50w"}},
        ]

        self.assertEqual(
            _history_reject_reason("select_option", {"value": "50w"}, steps),
            "option_already_selected",
        )
        self.assertIsNone(
            _history_reject_reason("select_option", {"value": "100w"}, steps)
        )

    def test_personalized_sft_row_keeps_ask_user_schema(self):
        trajectory = {
            "trajectory_id": "t1",
            "task_id": 9,
            "persona_mode": True,
            "messages": [],
            "blocked_tool_calls": [],
        }
        row = build_sft_row(trajectory)
        names = [tool["function"]["name"] for tool in row["tools"]]

        self.assertEqual(names[0], "ask_user")
        self.assertEqual(row["tools"], PERSONALIZED_SHOP_TOOL_SCHEMAS)


if __name__ == "__main__":
    unittest.main()
