"""Replay-verified composite Teacher trajectories for clarification SFT."""

from __future__ import annotations

import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from shopping_grpo.collection.sft import acceptance_reasons
from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.evaluation.rollout import collect_for_task
from shopping_grpo.multiturn.tasks import source_goal_hash, validate_task_row


COMPOSITE_TEACHER_POLICY = "composite-replay-v1"
QUESTION_PROMPT = """You write one concise Chinese clarification question for a
shopping agent. The public opening deliberately omitted the listed dimensions and
facts. Ask only for shopper-owned goal information whose answer can affect product
choice: a desired or required attribute, budget, compatibility, use context, or other
constraint that only the shopper can provide. Frame the question as asking what the
shopper needs, requires, prefers, uses, or can accept. Never ask the shopper to report
catalog facts about an unspecified or candidate product, such as what "this product"
costs or what material it actually has; the shopping agent must learn those facts from
shop tools. Do not ask about disclosed information and do not reveal the omitted values
in the question. Return one JSON object only:
{"question": "...", "covered_dimensions": ["..."]}
Every covered dimension must be copied exactly from OMITTED DIMENSIONS.
Good: '您对电磁阀材质有硬性要求吗？预算大约是多少？'
Bad: '这款自动浇水器的材质和价格分别是多少？'"""

UNSPECIFIED_PRODUCT_PREFIXES = (
    "这款", "该款", "这个商品", "该商品", "这个产品", "该产品",
)


def generate_gap_question(client, task):
    """Generate one auditable question targeted at the frozen opening gap."""

    dimensions, facts = validate_gap_task(task)
    payload = {
        "public_opening": task["initial_request"],
        "omitted_dimensions": dimensions,
        "omitted_facts": facts,
    }
    response = client.complete(
        [
            {"role": "system", "content": QUESTION_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        [],
    )
    result = _parse_json_object(response.get("content"))
    question = str(result.get("question") or "").strip()
    covered = result.get("covered_dimensions") or []
    if not question or not isinstance(covered, list) or not covered:
        raise ValueError("question must contain text and covered_dimensions")
    if not all(isinstance(item, str) and item in dimensions for item in covered):
        raise ValueError("question covered_dimensions must come from opening_audit")
    if any(fact.casefold() in question.casefold() for fact in facts):
        raise ValueError("question leaked an omitted fact")
    if question.lstrip().startswith(UNSPECIFIED_PRODUCT_PREFIXES):
        raise ValueError(
            "question asked the shopper for facts about an unspecified product"
        )
    return {
        "question": question,
        "covered_dimensions": list(covered),
        "omitted_dimensions": dimensions,
        "omitted_facts": facts,
    }


def collect_composite_teacher_task(
    task,
    *,
    teacher_client,
    shopper,
    env_factory=ShopAgentEnv,
    base_url="http://127.0.0.1:5700",
    max_steps=35,
    attempt_index=0,
):
    """Create a clarification prefix, collect a gold backbone, and replay both."""

    backbone = None
    question_audit = None
    answer_audit = None
    question_llm_calls = 0
    source_goal_verified = False
    setup_failure_point = "preflight"
    try:
        # Reject legacy frozen openings before spending any Teacher or Shopper call.
        validate_gap_task(task)
        setup_failure_point = "backbone"
        backbone = collect_for_task(
            task,
            client=teacher_client,
            env_factory=env_factory,
            base_url=base_url,
            max_steps=max_steps,
            attempt_index=attempt_index,
        )
        backbone_ok, backbone_reasons = acceptance_reasons(backbone)
        if not backbone_ok:
            return _rejected(
                task,
                attempt_index,
                "backbone_failed",
                backbone_reasons,
                source=backbone,
            )

        # Only spend clarification calls after the standard upstream-style Teacher
        # has produced a reusable gold shopping backbone.
        setup_failure_point = "private_context"
        context = _load_private_context(task, env_factory, base_url)
        source_goal_verified = True
        setup_failure_point = "question_generation"
        question_llm_calls = 1
        question_audit = generate_gap_question(teacher_client, task)
        setup_failure_point = "shopper_answer"
        answer_audit = shopper.answer_gap(
            question_audit["question"], context, question_audit["omitted_facts"]
        )

        ask_call = _tool_call(
            "ask_shopper",
            {"question": question_audit["question"]},
            f"composite-ask-{uuid4()}",
        )
        scripted = [
            {"role": "assistant", "content": None, "tool_calls": [ask_call]},
            *[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [deepcopy(step["tool_call"])],
                }
                for step in backbone.get("steps") or []
            ],
        ]
        setup_failure_point = "replay"
        replay = collect_for_task(
            task,
            client=_ReplayClient(scripted),
            shopper=_FrozenShopper(
                answer_audit["answer"], expected_hash=task["source_goal_hash"]
            ),
            env_factory=env_factory,
            base_url=base_url,
            max_steps=max_steps + 1,
            attempt_index=attempt_index,
            max_shopper_questions=1,
        )
        replay.update(
            {
                "teacher_policy": COMPOSITE_TEACHER_POLICY,
                "composite_stage": "replay_verified",
                "source_goal_verified": True,
                "clarification_grounded": True,
                "clarification_audit": {
                    **question_audit,
                    "used_facts": answer_audit["used_facts"],
                },
                "backbone_trajectory_id": backbone.get("trajectory_id"),
                "backbone_actor_llm_calls": backbone.get("actor_llm_calls", 0),
                "replay_scripted_calls": replay.get("actor_llm_calls", 0),
                "question_llm_calls": 1,
                "actor_llm_calls": int(backbone.get("actor_llm_calls", 0)) + 1,
                "shopper_llm_calls": 1,
            }
        )
        replay_ok, replay_reasons = acceptance_reasons(replay)
        if replay.get("blocked_tool_calls") or not replay_ok:
            replay["composite_stage"] = "replay_failed"
            replay["composite_rejection_reasons"] = [
                *replay_reasons,
                *(
                    ["replay_blocked_tool_call"]
                    if replay.get("blocked_tool_calls") else []
                ),
            ]
        return replay
    except Exception as exc:
        return _rejected(
            task,
            attempt_index,
            "setup_failed",
            [f"{exc.__class__.__name__}:{exc}"],
            source=backbone,
            question_audit=question_audit,
            answer_audit=answer_audit,
            question_llm_calls=question_llm_calls,
            source_goal_verified=source_goal_verified,
            setup_failure_point=setup_failure_point,
            error=exc,
        )


def _load_private_context(task, env_factory, base_url):
    env = env_factory(base_url=base_url, multiturn=True)
    try:
        env.reset(int(task["task_id"]), initial_request=task["initial_request"])
        context = deepcopy(env.shopper_context)
        if source_goal_hash(context) != task.get("source_goal_hash"):
            raise ValueError("source goal changed after opening generation")
        return context
    finally:
        env.release()


class _ReplayClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def complete(self, messages, tools, tool_choice="auto"):
        if not self.outputs:
            return {"role": "assistant", "content": "", "tool_calls": []}
        return self.outputs.pop(0)


class _FrozenShopper:
    def __init__(self, answer, expected_hash):
        self.answer_text = str(answer)
        self.expected_hash = str(expected_hash)
        self.call_count = 0

    def answer(self, question, context, history=()):
        if source_goal_hash(context) != self.expected_hash:
            raise ValueError("source goal changed before replay")
        self.call_count += 1
        return self.answer_text


def _tool_call(name, arguments, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _rejected(
    task,
    attempt_index,
    stage,
    reasons,
    *,
    source=None,
    question_audit=None,
    answer_audit=None,
    question_llm_calls=0,
    source_goal_verified=None,
    setup_failure_point=None,
    error=None,
):
    source = source or {}
    result = {
        "trajectory_id": str(uuid4()),
        "task_id": int(task["task_id"]),
        "attempt_index": int(attempt_index),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "composite_rejected",
        "done": False,
        "messages": [],
        "steps": [],
        "blocked_tool_calls": [],
        "initial_result": {},
        "terminal_result": deepcopy(source.get("terminal_result") or {}),
        "final_reward": float(source.get("final_reward", 0.0)),
        "error": deepcopy(source.get("error")),
        "release_error": deepcopy(source.get("release_error")),
        "interaction_mode": "multiturn",
        "teacher_policy": COMPOSITE_TEACHER_POLICY,
        "composite_stage": stage,
        "composite_rejection_reasons": list(reasons),
        "source_goal_verified": (
            stage != "setup_failed"
            if source_goal_verified is None
            else bool(source_goal_verified)
        ),
        "clarification_grounded": bool(question_audit and answer_audit),
        "shopper_questions": [],
        "backbone_status": source.get("status"),
        "backbone_done": source.get("done"),
        "backbone_steps": deepcopy(source.get("steps") or []),
        "backbone_blocked_tool_calls": deepcopy(
            source.get("blocked_tool_calls") or []
        ),
        "backbone_last_assistant": _last_assistant_message(
            source.get("messages") or []
        ),
        "backbone_actor_llm_calls": (
            int(source.get("actor_llm_calls", 0)) if source else None
        ),
        "backbone_context_turn_tokens": deepcopy(
            source.get("context_turn_tokens") or []
        ),
        "backbone_context_compactions": deepcopy(
            source.get("context_compactions") or []
        ),
        "backbone_model_output_truncations": deepcopy(
            source.get("model_output_truncations") or []
        ),
        "backbone_tool_call_truncations": deepcopy(
            source.get("tool_call_truncations") or []
        ),
        "question_llm_calls": int(question_llm_calls),
        "actor_llm_calls": (
            int(source.get("actor_llm_calls", 0)) + int(question_llm_calls)
        ),
        "shopper_llm_calls": 1 if answer_audit else 0,
    }
    if error is not None:
        result["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
    if setup_failure_point is not None:
        result["setup_failure_point"] = str(setup_failure_point)
    if question_audit:
        result["clarification_audit"] = {
            **question_audit,
            "used_facts": (answer_audit or {}).get("used_facts", []),
        }
        if answer_audit:
            result["shopper_questions"] = [{
                "question": question_audit["question"],
                "answer": answer_audit["answer"],
            }]
    return result


def _last_assistant_message(messages):
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return deepcopy(message)
    return None


def validate_gap_task(task):
    """Return a normalized frozen gap or fail without contacting an LLM."""

    validate_task_row(task)
    audit = task.get("opening_audit") or {}
    dimensions = [str(item).strip() for item in audit.get("omitted_dimensions") or []]
    facts = [str(item).strip() for item in audit.get("omitted_facts") or []]
    if not dimensions or not facts:
        raise ValueError("composite Teacher requires an opening_audit with a gap")
    if not all(dimensions) or not all(facts):
        raise ValueError("opening_audit gap fields must be non-empty strings")
    return dimensions, facts


def _parse_json_object(content):
    text = str(content or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("question response is not a JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("question response must be a JSON object")
    return value
