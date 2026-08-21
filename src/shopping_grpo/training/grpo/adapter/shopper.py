"""Trajectory-local controlled shopper used by the multi-turn GRPO harness."""

from __future__ import annotations

import json

from shopping_grpo.multiturn.shopper import _parse_json_object


SHOPPER_SYSTEM_PROMPT = """You are the shopper in a shopping clarification dialogue.
Answer the agent's latest question in concise natural Chinese. Use only ALLOWED FACTS
and facts already stated in the public dialogue. Never invent, infer, or reveal any
other preference. Return one JSON object only:
{"answer":"...","used_facts":["..."]}
Every used_facts entry must be copied verbatim from ALLOWED FACTS. If no allowed fact
answers the question, use an empty list and say you have no additional preference."""


class ControlledShopper:
    """Expose only opening-owned omitted facts to a separate shopper LLM."""

    def __init__(self, client, *, initial_request, allowed_facts, max_questions=2):
        self.client = client
        self.initial_request = str(initial_request).strip()
        self.allowed_facts = tuple(str(item).strip() for item in allowed_facts if str(item).strip())
        self.max_questions = int(max_questions)
        if self.max_questions < 0:
            raise ValueError("max_questions must be non-negative")
        self.history = []
        self.call_count = 0

    def answer(self, question):
        question = str(question or "").strip()
        if not question:
            raise ValueError("shopper question must be non-empty")
        if len(self.history) >= self.max_questions:
            raise ValueError("maximum shopper questions reached")
        if any(item["question"].casefold() == question.casefold() for item in self.history):
            raise ValueError("repeated shopper question")

        if not self.allowed_facts:
            result = {
                "answer": "没有其他补充，请按我已经说明的要求选择。",
                "used_facts": [],
            }
        else:
            public_history = []
            for item in self.history:
                public_history.extend([
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ])
            messages = [
                {
                    "role": "system",
                    "content": SHOPPER_SYSTEM_PROMPT
                    + "\nALLOWED FACTS: "
                    + json.dumps(self.allowed_facts, ensure_ascii=False),
                },
                {
                    "role": "user",
                    "content": "PUBLIC INITIAL REQUEST: " + self.initial_request,
                },
                *public_history,
                {"role": "user", "content": question},
            ]
            response = self.client.complete(messages, [])
            if response.get("tool_calls"):
                raise ValueError("shopper must return text, not tool calls")
            result = _parse_json_object(response.get("content") or "")
            self.call_count += 1

        answer = str(result.get("answer") or "").strip()
        used = result.get("used_facts") or []
        if not answer or not isinstance(used, list):
            raise ValueError("shopper answer must contain answer and used_facts")
        if not all(isinstance(item, str) and item in self.allowed_facts for item in used):
            raise ValueError("shopper used_facts must come from allowed facts")
        if self.allowed_facts and not used:
            raise ValueError("gap shopper answer must use at least one allowed fact")
        record = {"question": question, "answer": answer, "used_facts": list(used)}
        self.history.append(record)
        return record


def clarified_constraints_block(constraints) -> str:
    values = [str(item).strip() for item in constraints if str(item).strip()]
    if not values:
        return ""
    return "[CLARIFIED_CONSTRAINTS]\n" + "\n".join(f"- {item}" for item in values)
