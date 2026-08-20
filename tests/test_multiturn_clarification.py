import json

import pytest

from shopping_grpo.collection.sft import acceptance_reasons, build_sft_row
from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.evaluation.rollout import collect_for_task
from shopping_grpo.multiturn.tasks import (
    MULTITURN_TASK_SCHEMA, build_task_row, source_goal_hash,
)
from shopping_grpo.multiturn.shopper import ShopperSimulator
from shopping_grpo.multiturn.teacher import (
    collect_composite_teacher_task,
    generate_gap_question,
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
        "opening_audit": {"omitted_facts": ["private child size"]},
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
    assert "private child size" not in json.dumps(trajectory["messages"])
    sft = build_sft_row(trajectory)
    assert sft["tools"][0]["function"]["name"] == "ask_shopper"
    assert any(m.get("name") == "ask_shopper" for m in sft["messages"])


def test_length_limited_actor_output_is_not_mislabeled_as_final():
    class TruncatedActor:
        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": None,
                "reasoning": "unfinished",
                "_finish_reason": "length",
            }

    task = {
        "schema_version": MULTITURN_TASK_SCHEMA,
        "task_id": 7,
        "initial_request": "I need a pillow",
    }
    trajectory = collect_for_task(
        task,
        client=TruncatedActor(),
        shopper=Shopper(),
        env_factory=Env,
        max_steps=3,
    )
    assert trajectory["status"] == "model_output_truncated"
    assert trajectory["model_output_truncations"] == [{
        "message_index": 2,
        "finish_reason": "length",
    }]


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


def test_opening_generation_returns_audited_gap_in_one_call():
    class OpeningClient:
        def __init__(self):
            self.calls = 0
            self.messages = None

        def complete(self, messages, tools):
            self.calls += 1
            self.messages = messages
            return {
                "role": "assistant",
                "content": json.dumps({
                    "initial_request": "I need a mirror",
                    "omitted_dimensions": ["budget"],
                    "omitted_facts": ["budget 420"],
                }),
            }

    client = OpeningClient()
    result = ShopperSimulator(client).generate_initial_request({
        "instruction_full": "I need a mirror with budget 420",
        "goal_options": [],
    })
    assert client.calls == 1
    assert [message["role"] for message in client.messages] == ["system", "user"]
    assert "PRIVATE FACTS:" in client.messages[0]["content"]
    assert result["initial_request"] == "I need a mirror"
    assert result["omitted_facts"] == ["budget 420"]


def test_opening_generation_repairs_a_leaked_omitted_fact():
    class RepairingClient:
        def __init__(self):
            self.calls = []

        def complete(self, messages, tools):
            self.calls.append(messages)
            request = (
                "I need a mirror with budget 420"
                if len(self.calls) == 1 else "I need a mirror"
            )
            return {
                "role": "assistant",
                "content": json.dumps({
                    "initial_request": request,
                    "omitted_dimensions": ["budget"],
                    "omitted_facts": ["budget 420"],
                }),
            }

    client = RepairingClient()
    result = ShopperSimulator(client).generate_initial_request({
        "instruction_full": "I need a mirror with budget 420",
        "goal_options": [],
    })

    assert result["initial_request"] == "I need a mirror"
    assert len(client.calls) == 2
    assert [message["role"] for message in client.calls[1]] == [
        "system", "user", "assistant", "user",
    ]
    assert "omitted fact leaked" in client.calls[1][-1]["content"]


def test_opening_generation_stops_after_max_attempts():
    class LeakingClient:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            return {
                "role": "assistant",
                "content": json.dumps({
                    "initial_request": "I need a mirror with budget 420",
                    "omitted_dimensions": ["budget"],
                    "omitted_facts": ["budget 420"],
                }),
            }

    client = LeakingClient()
    with pytest.raises(ValueError, match="omitted fact leaked"):
        ShopperSimulator(client).generate_initial_request({
            "instruction_full": "I need a mirror with budget 420",
            "goal_options": [],
        }, max_attempts=2)

    assert client.calls == 2


def test_live_shopper_answer_retains_private_gap_provenance():
    class AnswerClient:
        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": json.dumps({
                    "answer": "预算大约420元。",
                    "used_facts": ["budget 420"],
                }),
            }

    result = ShopperSimulator(AnswerClient()).answer_audited(
        "预算是多少？",
        {"instruction_full": "I need a mirror with budget 420", "goal_options": []},
        ["budget 420"],
    )
    assert result == {
        "answer": "预算大约420元。",
        "used_facts": ["budget 420"],
    }


def test_shopper_combines_rules_and_private_facts_into_one_system_message():
    class CapturingClient:
        def __init__(self):
            self.messages = None

        def complete(self, messages, tools):
            self.messages = messages
            return {"role": "assistant", "content": "没有额外要求。"}

    client = CapturingClient()
    answer = ShopperSimulator(client).answer(
        "还有其他要求吗？",
        {"instruction_full": "full goal", "goal_options": ["option"]},
    )

    assert answer == "没有额外要求。"
    assert [message["role"] for message in client.messages].count("system") == 1
    assert "PRIVATE FACTS:" in client.messages[0]["content"]
    assert "full goal" in client.messages[0]["content"]


def test_shopper_empty_response_reports_truncation_diagnostics():
    class EmptyClient:
        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": None,
                "reasoning": "x" * 17,
                "_finish_reason": "length",
            }

    with pytest.raises(ValueError) as exc_info:
        ShopperSimulator(EmptyClient()).generate_initial_request({
            "instruction_full": "full goal",
            "goal_options": [],
        })

    assert "finish_reason='length'" in str(exc_info.value)
    assert "reasoning_chars=17" in str(exc_info.value)


def test_composite_backbone_failure_retains_diagnostics():
    class FinalOnlyClient:
        def complete(self, messages, tools):
            return {"role": "assistant", "content": "I am done.", "tool_calls": []}

    class BackboneEnv:
        def __init__(self, **kwargs):
            self.shopper_context = {
                "instruction_full": "full private goal",
                "goal_options": ["child size"],
            }

        def reset(self, task_id, initial_request=None):
            return {"env_idx": 0, "instruction": "full private goal"}

        def release(self):
            pass

    context = {
        "instruction_full": "full private goal",
        "goal_options": ["child size"],
    }
    task = build_task_row(
        7, "I need a pillow", context, "model", "prompt",
        omitted_dimensions=["size"], omitted_facts=["child size"],
    )
    trajectory = collect_composite_teacher_task(
        task,
        teacher_client=FinalOnlyClient(),
        shopper=Shopper(),
        env_factory=BackboneEnv,
        max_steps=3,
    )
    assert trajectory["composite_stage"] == "backbone_failed"
    assert trajectory["backbone_status"] == "assistant_final"
    assert trajectory["backbone_done"] is False
    assert trajectory["backbone_steps"] == []
    assert trajectory["backbone_last_assistant"]["content"] == "I am done."


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


def test_teacher_first_ask_forces_only_the_first_tool_choice():
    class ForcedActor(Actor):
        def __init__(self):
            super().__init__()
            self.choices = []

        def complete(self, messages, tools, tool_choice="auto"):
            self.choices.append(tool_choice)
            return self.outputs.pop(0)

    actor = ForcedActor()
    task = {
        "schema_version": MULTITURN_TASK_SCHEMA, "task_id": 7,
        "initial_request": "I need a pillow",
    }
    trajectory = collect_for_task(
        task, client=actor, shopper=Shopper(), env_factory=Env,
        max_steps=3, teacher_first_ask=True,
    )
    assert trajectory["status"] == "done"
    assert actor.choices[0]["function"]["name"] == "ask_shopper"
    assert actor.choices[1] == "auto"
    assert trajectory["teacher_policy"] == "force-first-ask-v1"


def test_gap_question_is_grounded_in_opening_audit_and_does_not_leak_value():
    class QuestionClient:
        def complete(self, messages, tools):
            prompt = " ".join(messages[0]["content"].split())
            assert "shopper-owned goal information" in prompt
            assert "Never ask the shopper to report catalog facts" in prompt
            return {
                "role": "assistant",
                "content": json.dumps({
                    "question": "您需要什么尺寸？",
                    "covered_dimensions": ["size"],
                }),
            }

    task = {
        "schema_version": MULTITURN_TASK_SCHEMA,
        "task_id": 9,
        "initial_request": "I need a pillow",
        "opening_audit": {
            "omitted_dimensions": ["size"],
            "omitted_facts": ["child size"],
        },
    }
    result = generate_gap_question(QuestionClient(), task)
    assert result["covered_dimensions"] == ["size"]
    assert "child size" not in result["question"]


def test_gap_question_rejects_unspecified_product_attribute_question():
    class QuestionClient:
        def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": json.dumps({
                    "question": "这款自动浇水器的材质和价格分别是多少？",
                    "covered_dimensions": ["材质", "价格"],
                }, ensure_ascii=False),
            }

    task = {
        "schema_version": MULTITURN_TASK_SCHEMA,
        "task_id": 53,
        "initial_request": "我想买自动浇水器",
        "opening_audit": {
            "omitted_dimensions": ["材质", "价格"],
            "omitted_facts": ["必须是铜芯电磁阀", "预算230元左右"],
        },
    }
    with pytest.raises(ValueError, match="unspecified product"):
        generate_gap_question(QuestionClient(), task)


def test_gap_answer_uses_natural_paraphrase_but_keeps_verbatim_audit_facts():
    facts = ["浇水器必须是铜芯电磁阀的", "价格在230元左右"]

    class AnswerClient:
        def complete(self, messages, tools):
            prompt = " ".join(messages[0]["content"].split())
            assert "natural first-person statement" in prompt
            assert "Only used_facts, not answer" in prompt
            return {
                "role": "assistant",
                "content": json.dumps({
                    "answer": "需要铜芯电磁阀，预算大约230元。",
                    "used_facts": facts,
                }, ensure_ascii=False),
            }

    result = ShopperSimulator(AnswerClient()).answer_gap(
        "您对电磁阀材质有硬性要求吗？预算大约是多少？",
        {
            "instruction_full": "我需要铜芯电磁阀，价格在230元左右",
            "goal_options": [],
        },
        facts,
    )
    assert result == {
        "answer": "需要铜芯电磁阀，预算大约230元。",
        "used_facts": facts,
    }


def test_composite_teacher_rejects_legacy_opening_before_any_llm_call():
    class NoCallClient:
        calls = 0

        def complete(self, messages, tools, tool_choice="auto"):
            self.calls += 1
            raise AssertionError("legacy opening must fail before an LLM call")

    client = NoCallClient()
    trajectory = collect_composite_teacher_task(
        {
            "schema_version": MULTITURN_TASK_SCHEMA,
            "task_id": 9,
            "initial_request": "legacy opening without audit",
        },
        teacher_client=client,
        shopper=object(),
    )
    assert trajectory["composite_stage"] == "setup_failed"
    assert trajectory["actor_llm_calls"] == 0
    assert client.calls == 0


def test_composite_teacher_replays_question_and_gold_backbone_without_goal_leak():
    context = {
        "instruction_full": "I need a child size pillow",
        "goal_options": [],
    }
    task = build_task_row(
        9,
        "I need a pillow",
        context,
        "teacher",
        "prompt-hash",
        omitted_dimensions=["size"],
        omitted_facts=["child size"],
    )

    class CompositeClient:
        def __init__(self):
            self.outputs = [
                tool_message("search_products", {"query": "pillow"}, "search-1"),
                tool_message("open_product", {"asin": "100000000001"}, "open-1"),
                tool_message("buy_now", {}, "buy-1"),
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "question": "What size do you need?",
                        "covered_dimensions": ["size"],
                    }),
                },
            ]

        def complete(self, messages, tools, tool_choice="auto"):
            return self.outputs.pop(0)

    class GapShopper:
        call_count = 0

        def answer_gap(self, question, private_context, omitted_facts):
            self.call_count += 1
            assert private_context == context
            return {"answer": "child size", "used_facts": ["child size"]}

    class ReplayEnv:
        def __init__(self, **kwargs):
            self.multiturn = kwargs.get("multiturn", False)
            self.shopper_context = None

        def reset(self, task_id, initial_request=None):
            if self.multiturn:
                self.shopper_context = dict(context)
                return {"env_idx": 0, "instruction": initial_request}
            return {"env_idx": 0, "instruction": context["instruction_full"]}

        def step(self, action):
            if action == "search[pillow]":
                return {
                    "instruction": (
                        "1|100000000001|99.0|brand|category|attr|pillow\n"
                        '可点击的按钮: ["100000000001"]'
                    ),
                    "reward": 0.0,
                    "done": False,
                }
            if action == "click[100000000001]":
                return {
                    "instruction": '详情\n可点击的按钮: ["Buy Now"]',
                    "reward": 0.0,
                    "done": False,
                }
            assert action == "click[Buy Now]"
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

    trajectory = collect_composite_teacher_task(
        task,
        teacher_client=CompositeClient(),
        shopper=GapShopper(),
        env_factory=ReplayEnv,
        max_steps=5,
    )
    assert trajectory["status"] == "done"
    assert trajectory["composite_stage"] == "replay_verified"
    assert trajectory["shopper_questions"] == [{
        "question": "What size do you need?",
        "answer": "child size",
    }]
    assert trajectory["actor_llm_calls"] == 4
    assert acceptance_reasons(trajectory) == (True, [])
    payload = json.dumps(build_sft_row(trajectory), ensure_ascii=False)
    assert "I need a child size pillow" not in payload
    assert "ask_shopper" in payload
