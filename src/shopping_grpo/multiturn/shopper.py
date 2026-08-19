"""LLM shopper grounded in a private ShopSimulator full goal."""

import hashlib
import json

OPENING_PROMPT = """You simulate a shopper. Rewrite the private shopping goal as a natural,
underspecified first request in Chinese. Keep the product category and several useful
requirements, but omit one or two purchase-critical facts that can be recovered by a
clarifying question. Do not mention that anything was omitted. Return only the request."""

ANSWER_PROMPT = """You simulate a shopper answering a shopping agent in Chinese. Answer only from the
private full goal and option facts. Never invent a new preference. If the question asks
for information absent from those facts, say that you have no additional preference or
are unsure. Be concise and answer the question directly."""

OPENING_PROMPT_HASH = hashlib.sha256(OPENING_PROMPT.encode("utf-8")).hexdigest()


class ShopperSimulator:
    def __init__(self, client):
        self.client = client
        self.call_count = 0

    def generate_initial_request(self, context):
        return self._complete(OPENING_PROMPT, context, [])

    def answer(self, question, context, history=()):
        turns = []
        for item in history:
            turns.extend([
                {"role": "user", "content": str(item["question"])},
                {"role": "assistant", "content": str(item["answer"])},
            ])
        turns.append({"role": "user", "content": str(question)})
        return self._complete(ANSWER_PROMPT, context, turns)

    def _complete(self, system_prompt, context, turns):
        private = json.dumps({
            "full_goal": context["instruction_full"],
            "goal_options": context.get("goal_options") or [],
        }, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": "PRIVATE FACTS: " + private},
            *turns,
        ]
        response = self.client.complete(messages, [])
        if response.get("tool_calls"):
            raise ValueError("shopper must return text, not tool calls")
        content = str(response.get("content") or "").strip()
        if not content:
            raise ValueError("shopper returned an empty response")
        self.call_count += 1
        return content
