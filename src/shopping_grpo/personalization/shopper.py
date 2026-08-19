"""A private Shopper that answers clarification questions from environment facts."""

from __future__ import annotations

import json


SHOPPER_SYSTEM_PROMPT = """你是 ShopSimulator 中的用户模拟器。

你会收到该用户的完整购买要求、简化要求、用户画像，以及购物 Agent 当前提出的一个问题。
只根据这些私有信息，用第一人称简短回答当前问题。不得虚构私有信息中不存在的偏好、预算、品牌、
规格或用途；若信息中没有答案，明确回答“我没有额外要求”或“我不确定”。不要提出新问题，不要
推荐商品，不要解释你的推理，不要输出 JSON，也不要调用工具。
"""


class ShopperProtocolError(RuntimeError):
    """The Shopper produced no usable natural-language answer."""


class LLMShopper:
    """Use one chat completion for each ``ask_user`` call."""

    def __init__(self, client):
        self.client = client

    def answer(self, *, question, context, history=()):
        if not isinstance(context, dict):
            raise ShopperProtocolError("private Shopper context is unavailable")
        if not isinstance(question, str) or not question.strip():
            raise ShopperProtocolError("question must be non-empty")

        payload = {
            "instruction_full": context.get("instruction_full", ""),
            "instruction_simple": context.get("instruction_simple", ""),
            "user_persona": context.get("user_persona") or {},
            "previous_questions_and_answers": list(history),
            "current_question": question.strip(),
        }
        messages = [
            {"role": "system", "content": SHOPPER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = self.client.complete(messages, [])
        if response.get("tool_calls"):
            raise ShopperProtocolError("Shopper must not call tools")
        answer = response.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ShopperProtocolError("Shopper returned an empty answer")
        return answer.strip()
