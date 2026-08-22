import asyncio
import json
from pathlib import Path

import pytest

from shopping_grpo.environment.tools import MULTITURN_SHOP_TOOL_SCHEMAS
from shopping_grpo.multiturn.tasks import source_goal_hash
from shopping_grpo.training.grpo.adapter.runtime import (
    MULTITURN_HARNESS_VERSION,
    current_runtime_state,
    current_shopper,
    multiturn_spec_from_kwargs,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.adapter.shopper import ControlledShopper
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
from scripts.prepare_multiturn_grpo_dataset import build_record


PRIVATE = {
    "instruction_full": "需要铜芯电磁阀，预算大约230元。",
    "goal_options": [],
}


def spec(mode="gap"):
    facts = ["需要铜芯电磁阀", "预算大约230元"] if mode == "gap" else []
    return {
        "task_id": 53,
        "interaction_mode": mode,
        "initial_request": "想买自动浇水器。" if mode == "gap" else PRIVATE["instruction_full"],
        "source_goal_hash": source_goal_hash(PRIVATE),
        "omitted_facts": facts,
    }


class FakeEnv:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shopper_context = PRIVATE
        self.released = False
        self.__class__.instances.append(self)

    def reset(self, task_id, initial_request=None):
        assert task_id == 53
        assert initial_request == ""
        return {
            "instruction": "",
            "environment_version": "shopsimulator-environment-v2.1",
            "reward_version": "shopsimulator-reward-v4",
        }

    def release(self):
        self.released = True


class FakeClient:
    def __init__(self):
        self.messages = None

    def complete(self, messages, tools):
        self.messages = messages
        return {
            "content": json.dumps(
                {
                    "answer": "需要铜芯电磁阀，预算大约230元。",
                    "used_facts": ["需要铜芯电磁阀", "预算大约230元"],
                },
                ensure_ascii=False,
            )
        }


def make_ask_tool():
    schema = next(
        item for item in MULTITURN_SHOP_TOOL_SCHEMAS
        if item["function"]["name"] == "ask_shopper"
    )
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except ImportError:
        tool_schema = schema
    else:
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
    return ShopSimulatorTool({}, tool_schema)


def test_multiturn_metadata_contract_rejects_inconsistent_modes():
    raw = spec()
    facts = raw.pop("omitted_facts")
    kwargs = {"extra_info": {
        **raw,
        "harness_version": MULTITURN_HARNESS_VERSION,
        "opening_audit": {"omitted_facts": facts},
    }}
    parsed = multiturn_spec_from_kwargs(kwargs, enabled=True)
    assert parsed["interaction_mode"] == "gap"
    assert parsed["omitted_facts"] == ["需要铜芯电磁阀", "预算大约230元"]

    kwargs["extra_info"]["interaction_mode"] = "complete"
    with pytest.raises(ValueError, match="must not expose"):
        multiturn_spec_from_kwargs(kwargs, enabled=True)


def test_gap_shopper_is_private_and_question_does_not_consume_shop_step():
    client = FakeClient()

    def shopper_factory(**kwargs):
        return ControlledShopper(client, **kwargs)

    async def run():
        session = ShopSimulatorSession(
            max_steps=35,
            required_environment_version="shopsimulator-environment-v2.1",
            required_reward_version="shopsimulator-reward-v4",
            multiturn_enable=True,
            shopper_factory=shopper_factory,
            env_factory=FakeEnv,
        )
        state = await session.start(53, multiturn_spec=spec())
        assert current_shopper.get() is not None
        assert "omitted_facts" not in state
        response, _, step = await make_ask_tool().execute(
            "ask-1", {"question": "材质和预算有什么要求？"}
        )
        assert state["steps"] == []
        assert state["shopper_question_count"] == 1
        assert step["question_index"] == 0
        assert "[CLARIFIED_CONSTRAINTS]" in response.text
        assert "instruction_full" not in json.dumps(client.messages, ensure_ascii=False)
        assert "goal_options" not in json.dumps(client.messages, ensure_ascii=False)
        await session.close()
        assert current_shopper.get() is None
        assert current_runtime_state.get() is None

    asyncio.run(run())


def test_complete_opening_uses_deterministic_no_preference_answer():
    class NeverCalled:
        def complete(self, messages, tools):
            raise AssertionError("complete opening must not call the shopper LLM")

    shopper = ControlledShopper(
        NeverCalled(), initial_request="完整要求", allowed_facts=[], max_questions=2
    )
    answer = shopper.answer("还有其他预算要求吗？")
    assert answer["used_facts"] == []
    assert "没有其他补充" in answer["answer"]
    assert shopper.call_count == 0


def test_gap_shopper_empty_provenance_uses_safe_deterministic_fallback():
    class EmptyFactClient:
        def complete(self, messages, tools):
            return {
                "content": json.dumps(
                    {
                        "answer": "我还想要一个未授权的红色版本。",
                        "used_facts": [],
                    },
                    ensure_ascii=False,
                )
            }

    shopper = ControlledShopper(
        EmptyFactClient(),
        initial_request="想买自动浇水器。",
        allowed_facts=["预算大约230元"],
        max_questions=2,
    )

    answer = shopper.answer("颜色有什么要求？")

    assert answer == {
        "question": "颜色有什么要求？",
        "answer": "没有其他补充，请按我已经说明的要求选择。",
        "used_facts": [],
    }
    assert shopper.call_count == 1
    assert shopper.history == [
        {
            "question": "颜色有什么要求？",
            "answer": "没有其他补充，请按我已经说明的要求选择。",
            "used_facts": [],
        }
    ]


def test_repeated_question_is_rejected_without_incrementing_count():
    client = FakeClient()
    shopper = ControlledShopper(
        client,
        initial_request="想买自动浇水器。",
        allowed_facts=["需要铜芯电磁阀", "预算大约230元"],
        max_questions=2,
    )
    shopper.answer("材质和预算有什么要求？")
    with pytest.raises(ValueError, match="repeated"):
        shopper.answer("材质和预算有什么要求？")
    assert len(shopper.history) == 1


def test_generated_tool_config_exactly_matches_canonical_schema():
    config = json.loads(Path("configs/tools.json").read_text(encoding="utf-8"))
    assert [item["tool_schema"] for item in config["tools"]] == MULTITURN_SHOP_TOOL_SCHEMAS
    assert config["tools"][0]["tool_schema"]["function"]["name"] == "ask_shopper"


def test_verl_record_keeps_private_fact_out_of_actor_prompt():
    opening = {
        "schema_version": "shopsimulator-multiturn-task-v1",
        "task_id": 53,
        "initial_request": "想买自动浇水器。",
        "source_goal_hash": source_goal_hash(PRIVATE),
        "opening_audit": {"omitted_facts": ["预算大约230元"]},
    }
    record = build_record(opening, "gap", "train", 0)
    assert "预算大约230元" not in json.dumps(record["prompt"], ensure_ascii=False)
    assert record["extra_info"]["opening_audit"]["omitted_facts"] == ["预算大约230元"]
    assert record["extra_info"]["interaction_mode"] == "gap"
    assert record["extra_info"]["harness_version"] == MULTITURN_HARNESS_VERSION
