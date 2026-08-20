"""LLM shopper grounded in a private ShopSimulator full goal."""

import hashlib
import json

OPENING_PROMPT = """You simulate a shopper. Rewrite the private shopping goal as a natural,
underspecified first request in Chinese. Keep the product category and several useful
requirements, but deliberately omit one or two purchase-critical facts whose values
could change product selection, such as budget, size, model, compatibility, capacity,
or a required material. Do not mention the omission in the request. Return one JSON
object only:
{"initial_request": "...", "omitted_dimensions": ["..."], "omitted_facts": ["..."]}
Copy each omitted_facts value verbatim from the private goal. The omitted facts must not
appear or be paraphrased in initial_request."""

ANSWER_PROMPT = """You simulate a shopper answering a shopping agent in Chinese. Answer only from the
private full goal and option facts. Never invent a new preference. If the question asks
for information absent from those facts, say that you have no additional preference or
are unsure. Be concise and answer the question directly."""

AUDITED_ANSWER_PROMPT = """You simulate a shopper answering a shopping agent in Chinese.
Answer only from the private full goal and option facts. Return one JSON object only:
{"answer": "...", "used_facts": ["..."]}
Answer the question directly and never invent a preference. Copy into used_facts every
supplied omitted fact that materially supports the answer, verbatim. If the question
does not ask about any supplied omitted fact, use an empty used_facts list and say that
you have no additional preference or are unsure. The natural Chinese answer may
paraphrase the facts; only used_facts must be verbatim."""

GAP_ANSWER_PROMPT = """You simulate a shopper answering one controlled clarification
question in Chinese. Use only the supplied omitted facts and the private full goal.
Return one JSON object only:
{"answer": "...", "used_facts": ["..."]}
Copy every used_facts entry verbatim from the supplied omitted facts. Use at least one
omitted fact and never invent a preference. Write answer as a concise, natural
first-person statement of the shopper's requirement, preference, situation, or budget.
Paraphrase naturally instead of mechanically pasting source sentences or repeating the
product category. Only used_facts, not answer, must preserve the verbatim audit text.
For example, turn facts like '浇水器必须是铜芯电磁阀的' and '价格在230元左右' into
an answer like '需要铜芯电磁阀，预算大约230元。'"""

OPENING_PROMPT_HASH = hashlib.sha256(OPENING_PROMPT.encode("utf-8")).hexdigest()


class ShopperSimulator:
    def __init__(self, client):
        self.client = client
        self.call_count = 0

    def generate_initial_request(self, context, max_attempts=3):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        turns = [{
            "role": "user",
            "content": "Generate the public underspecified shopping request now.",
        }]
        last_error = None
        for attempt in range(max_attempts):
            raw = self._complete(OPENING_PROMPT, context, turns)
            try:
                return self._validate_initial_request(raw, context)
            except ValueError as exc:
                last_error = exc
                if attempt + 1 == max_attempts:
                    break
                turns.extend([
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Correct the previous JSON. Validation failed: "
                            f"{exc}. Return one corrected JSON object only. "
                            "Keep omitted_facts verbatim from the private goal, "
                            "but remove those facts and their paraphrases from "
                            "initial_request."
                        ),
                    },
                ])
        raise last_error

    @staticmethod
    def _validate_initial_request(raw, context):
        result = _parse_json_object(raw)
        request = str(result.get("initial_request", "")).strip()
        dimensions = result.get("omitted_dimensions") or []
        facts = result.get("omitted_facts") or []
        if (
            not request
            or not isinstance(dimensions, list) or not dimensions
            or not isinstance(facts, list) or not facts
        ):
            raise ValueError("opening must contain a request and at least one omitted fact")
        if not all(isinstance(item, str) and item.strip() for item in dimensions + facts):
            raise ValueError("opening audit fields must be non-empty strings")
        private_text = context["instruction_full"] + json.dumps(
            context.get("goal_options") or [], ensure_ascii=False
        )
        for fact in facts:
            if fact.strip() not in private_text:
                raise ValueError("omitted fact must be copied from the private goal")
            if fact.strip().casefold() in request.casefold():
                raise ValueError("omitted fact leaked into initial_request")
        return {
            "initial_request": request,
            "omitted_dimensions": [item.strip() for item in dimensions],
            "omitted_facts": [item.strip() for item in facts],
        }

    def answer(self, question, context, history=()):
        turns = []
        for item in history:
            turns.extend([
                {"role": "user", "content": str(item["question"])},
                {"role": "assistant", "content": str(item["answer"])},
            ])
        turns.append({"role": "user", "content": str(question)})
        return self._complete(ANSWER_PROMPT, context, turns)

    def answer_audited(self, question, context, omitted_facts, history=()):
        """Answer a live autonomous question and retain private fact provenance."""

        facts = [str(item).strip() for item in omitted_facts if str(item).strip()]
        turns = []
        for item in history:
            turns.extend([
                {"role": "user", "content": str(item["question"])},
                {"role": "assistant", "content": str(item["answer"])},
            ])
        turns.append({"role": "user", "content": str(question)})
        prompt = AUDITED_ANSWER_PROMPT + "\nOMITTED FACTS: " + json.dumps(
            facts, ensure_ascii=False
        )
        result = _parse_json_object(self._complete(prompt, context, turns))
        answer = str(result.get("answer") or "").strip()
        used = result.get("used_facts") or []
        if not answer or not isinstance(used, list):
            raise ValueError("audited answer must contain answer and used_facts")
        if not all(isinstance(item, str) and item in facts for item in used):
            raise ValueError("audited answer used_facts must come from omitted facts")
        return {"answer": answer, "used_facts": used}

    def answer_gap(self, question, context, omitted_facts):
        facts = [str(item).strip() for item in omitted_facts if str(item).strip()]
        if not facts:
            raise ValueError("controlled clarification requires omitted facts")
        prompt = GAP_ANSWER_PROMPT + "\nOMITTED FACTS: " + json.dumps(
            facts, ensure_ascii=False
        )
        raw = self._complete(
            prompt,
            context,
            [{"role": "user", "content": str(question)}],
        )
        result = _parse_json_object(raw)
        answer = str(result.get("answer") or "").strip()
        used = result.get("used_facts") or []
        if not answer or not isinstance(used, list) or not used:
            raise ValueError("gap answer must contain answer and used_facts")
        if not all(isinstance(item, str) and item in facts for item in used):
            raise ValueError("gap answer used_facts must be copied from omitted facts")
        return {"answer": answer, "used_facts": used}

    def _complete(self, system_prompt, context, turns):
        private = json.dumps({
            "full_goal": context["instruction_full"],
            "goal_options": context.get("goal_options") or [],
        }, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": system_prompt + "\nPRIVATE FACTS: " + private,
            },
            *turns,
        ]
        response = self.client.complete(messages, [])
        if response.get("tool_calls"):
            raise ValueError("shopper must return text, not tool calls")
        content = str(response.get("content") or "").strip()
        if not content:
            reasoning = str(
                response.get("reasoning")
                or response.get("reasoning_content")
                or ""
            )
            raise ValueError(
                "shopper returned an empty response "
                f"(finish_reason={response.get('_finish_reason')!r}, "
                f"reasoning_chars={len(reasoning)})"
            )
        self.call_count += 1
        return content


def _parse_json_object(text):
    content = str(text).strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("shopper opening is not a JSON object")
        value = json.loads(content[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("shopper opening must be a JSON object")
    return value
