"""Probe B V3 runner for a single pre-shopping Autonomous-Ask opportunity."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
PROBE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(PROBE_DIR))

from shopping_grpo.environment.client import ShopAgentEnv  # noqa: E402
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402
from shopping_grpo.evaluation.rollout import (  # noqa: E402
    OpenAIChatClient,
    SYSTEM_PROMPT,
    collect_for_task,
)
from runner import (  # noqa: E402
    CallBudget,
    CallBudgetExceeded,
    CappedTransport,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    DEFAULT_SHOPSIM_BASE_URL,
    FakeShopEnv,
    InstructionOverrideEnv,
    MANIFEST_FILE,
    TRAJECTORIES_FILE,
    _assistant_tool,
    _default_run_id,
    _git_commit,
    _public_api_base,
    _trajectory_validity,
    append_record,
    load_records,
    resolve_service_urls,
    write_json,
)
from task_schema import (  # noqa: E402
    DEFAULT_TASKS,
    canonical_hash,
    load_tasks,
    validate_tasks,
)


AUTONOMOUS_ARM = "autonomous_ask"
DEFAULT_MAX_LLM_CALLS = 200
DEFAULT_OUTPUT_ROOT = PROBE_DIR / "outputs" / "v3"
MAX_TRAJECTORY_ATTEMPTS = 10
SELECTED_TASK_IDS = (
    5496,
    17319,
    13871,
    20518,
    12403,
    21247,
    18637,
    6935,
    13089,
    21175,
)

QUESTION_FIELD_KEYWORDS = {
    "budget": ("预算", "价格范围", "价位", "多少钱", "最高价格", "价格上限"),
    "color": ("颜色", "配色", "色号", "色系"),
    "brand": ("品牌", "牌子", "厂牌"),
    "origin": ("产地", "原产地", "哪里生产", "哪国生产", "国产还是进口"),
}
NEUTRAL_ANSWER = "我暂时没有更多信息，请根据已经说明的需求继续选择。"

ASK_USER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": "首次购物动作前，询问一个会实质影响购买决定的缺失约束。每题最多调用一次。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "只询问一个缺失约束的简洁问题。",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}

_FORBIDDEN_ASK_TEXT = (
    "用户的完整需求只会在开头给出。不得向用户追问、确认、告别，也不要假设存在用户对话工具。"
    "你只能调用提供的标准工具与商店交互。"
)
_AUTONOMOUS_ASK_TEXT = (
    "用户的当前需求只会在开头给出。仅在第一次购物动作之前，若发现一个会实质影响购买决定的重要约束缺失，"
    "可以调用一次 `ask_user` 询问一个简洁问题；否则直接使用购物工具。不要确认或告别。"
    "提问机会用完或开始购物后，只能调用提供的标准购物工具与商店交互。"
)
if SYSTEM_PROMPT.count(_FORBIDDEN_ASK_TEXT) != 1:
    raise RuntimeError("formal SYSTEM_PROMPT no longer contains the frozen ask prohibition")
AUTONOMOUS_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    _FORBIDDEN_ASK_TEXT, _AUTONOMOUS_ASK_TEXT
)


def select_pilot_tasks(
    tasks: list[dict[str, Any]], seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Apply the preregistered 6-color plus first-4-budget selection."""

    ordered = list(tasks)
    random.Random(int(seed)).shuffle(ordered)
    color_ids = {
        int(task["task_id"])
        for task in tasks
        if task["latent_goal"]["field"] == "color"
    }
    budget_ids = [
        int(task["task_id"])
        for task in ordered
        if task["latent_goal"]["field"] == "budget"
    ][:4]
    selected_ids = color_ids | set(budget_ids)
    selected = [task for task in ordered if int(task["task_id"]) in selected_ids]
    actual = tuple(int(task["task_id"]) for task in selected)
    if actual != SELECTED_TASK_IDS:
        raise ValueError(
            "V3 selected task order differs from the frozen design: "
            f"{actual!r} != {SELECTED_TASK_IDS!r}"
        )
    return selected


def answer_question(task: Mapping[str, Any], question: str) -> dict[str, Any]:
    normalized = str(question).casefold()
    recognized = [
        field
        for field, keywords in QUESTION_FIELD_KEYWORDS.items()
        if any(keyword.casefold() in normalized for keyword in keywords)
    ]
    hidden_field = str(task["latent_goal"]["field"])
    correct = recognized == [hidden_field]
    return {
        "called": True,
        "question": str(question),
        "recognized_fields": recognized,
        "hidden_field": hidden_field,
        "classification": "correct_ask" if correct else "incorrect_ask",
        "answer_type": "oracle" if correct else "neutral",
        "answer": str(task["oracle_turn"]["answer"]) if correct else NEUTRAL_ANSWER,
        "dropped_tool_calls": [],
    }


class AutonomousAskClient:
    """Expose ask_user once, answer locally, then permanently remove it."""

    def __init__(self, inner, task: Mapping[str, Any]):
        self.inner = inner
        self.task = task
        self.first_decision_complete = False
        self.ask_event = {
            "called": False,
            "question": None,
            "recognized_fields": [],
            "hidden_field": str(task["latent_goal"]["field"]),
            "classification": "no_ask",
            "answer_type": None,
            "answer": None,
            "dropped_tool_calls": [],
        }
        self.last_context_event = None
        self.last_context_tokens = None

    def complete(self, messages, tools):
        if self.first_decision_complete:
            result = self.inner.complete(messages, tools)
            self._sync_context()
            return result

        assistant = self.inner.complete(messages, [ASK_USER_TOOL_SCHEMA, *tools])
        self._sync_context()
        tool_calls = assistant.get("tool_calls") or []
        first = tool_calls[0] if tool_calls else None
        name = str(((first or {}).get("function") or {}).get("name") or "")
        if name != "ask_user":
            self.first_decision_complete = True
            return assistant

        serial_assistant = deepcopy(assistant)
        serial_assistant["tool_calls"] = [deepcopy(first)]
        dropped = deepcopy(tool_calls[1:])
        question = _question_argument(first)
        self.ask_event = answer_question(self.task, question)
        self.ask_event["dropped_tool_calls"] = dropped
        messages.append(serial_assistant)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(first.get("id") or "ask_user"),
                "name": "ask_user",
                "content": self.ask_event["answer"],
            }
        )
        self.first_decision_complete = True
        result = self.inner.complete(messages, tools)
        self._sync_context()
        return result

    def _sync_context(self) -> None:
        self.last_context_event = getattr(self.inner, "last_context_event", None)
        self.last_context_tokens = getattr(self.inner, "last_context_tokens", None)

    def project_observation(self, tool_name, observation, parameters):
        projector = getattr(self.inner, "project_observation", None)
        if projector is None:
            return observation, None
        return projector(tool_name, observation, parameters)


class MockAutonomousBaseClient:
    """Offline model that asks the frozen correct question, then buys in 3 steps."""

    def __init__(self, task: Mapping[str, Any]):
        self.task = task
        self.shopping_index = 0
        self.asked = False

    def complete(self, messages, tools):
        names = {tool["function"]["name"] for tool in tools}
        if "ask_user" in names and not self.asked:
            self.asked = True
            return _assistant_tool(
                "ask_user",
                {"question": str(self.task["oracle_turn"]["question"])},
                f"mock_ask_{self.task['task_id']}",
            )
        responses = (
            _assistant_tool(
                "search_products", {"query": "探针测试商品"}, "mock_search"
            ),
            _assistant_tool(
                "open_product", {"asin": "100000000001"}, "mock_open"
            ),
            _assistant_tool("buy_now", {}, "mock_buy"),
        )
        if self.shopping_index >= len(responses):
            return {"role": "assistant", "content": "mock exhausted"}
        response = responses[self.shopping_index]
        self.shopping_index += 1
        return response


def run_autonomous_trajectory(
    task: Mapping[str, Any],
    *,
    base_client,
    env_factory,
    shopsim_base_url: str,
    max_steps: int,
    run_id: str,
    task_hash: str,
    llm_calls_before: int,
) -> dict[str, Any]:
    visible_instruction = "Instruction: " + str(task["under_query"])
    prompt = [{"role": "system", "content": AUTONOMOUS_SYSTEM_PROMPT}]
    client = (
        base_client
        if isinstance(base_client, AutonomousAskClient)
        else AutonomousAskClient(base_client, task)
    )

    def factory(**kwargs):
        inner = env_factory(task, kwargs.get("base_url", shopsim_base_url))
        return InstructionOverrideEnv(
            inner,
            visible_instruction=visible_instruction,
            expected_clear_query=str(task["clear_query"]),
        )

    trajectory = collect_for_task(
        {"task_id": int(task["task_id"]), "prompt": prompt},
        client=client,
        env_factory=factory,
        base_url=shopsim_base_url,
        max_steps=int(max_steps),
        tools=SHOP_TOOL_SCHEMAS,
        attempt_index=0,
    )
    trajectory["run_id"] = run_id
    trajectory["arm"] = AUTONOMOUS_ARM
    valid, validity_reason = _trajectory_validity(trajectory)
    trajectory["probe"] = {
        "query": task["under_query"],
        "clear_query": task["clear_query"],
        "latent_goal": deepcopy(task["latent_goal"]),
        "oracle_turn": None,
        "oracle_injected": False,
        "autonomous_ask": deepcopy(client.ask_event),
        "task_hash": task_hash,
        "llm_calls_before": int(llm_calls_before),
        "valid": valid,
        "validity_reason": validity_reason,
    }
    return trajectory


def create_or_load_manifest(
    run_dir: Path,
    *,
    run_id: str,
    mode: str,
    tasks_path: Path,
    validation: Mapping[str, Any],
    seed: int,
    max_steps: int,
    max_llm_calls: int,
    temperature: float,
    model: str,
    base_url: str,
    shopsim_base_url: str,
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = run_dir / MANIFEST_FILE
    immutable = {
        "run_id": run_id,
        "mode": mode,
        "arm": AUTONOMOUS_ARM,
        "task_file": str(tasks_path.resolve()),
        "task_hash": validation["task_hash"],
        "selected_task_ids": list(SELECTED_TASK_IDS),
        "selected_task_hash": canonical_hash(list(SELECTED_TASK_IDS)),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "max_llm_calls": int(max_llm_calls),
        "temperature": float(temperature),
        "model": model,
        "api_base": _public_api_base(base_url),
        "shopsim_base": _public_api_base(shopsim_base_url),
        "system_prompt_hash": canonical_hash(AUTONOMOUS_SYSTEM_PROMPT),
        "ask_tool_schema_hash": canonical_hash(ASK_USER_TOOL_SCHEMA),
        "shop_tool_schema_hash": canonical_hash(SHOP_TOOL_SCHEMAS),
        "reference": deepcopy(reference),
        "git_commit": _git_commit(),
        "environment_version": "shopsimulator-environment-v2.1",
        "reward_version": "shopsimulator-reward-v3",
        "observation_version": "shopping-observation-v2",
        "tool_schema_version": "shop-tool-schema-v2",
    }
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for key, value in immutable.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"run_id {run_id!r} already exists with a different {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        return manifest
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        **immutable,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llm_call_count": 0,
        "trajectory_attempt_count": 0,
        "completed_task_ids": [],
        "invalid_task_ids": [],
    }
    write_json(path, manifest)
    return manifest


def inspect_reference_run(
    run_dir: Path, task_hash: str, expected_model: str
) -> dict[str, Any]:
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise ValueError(f"missing V2 reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "real":
        raise ValueError("V2 reference run must be real mode")
    if manifest.get("task_hash") != task_hash:
        raise ValueError("V2 reference task_hash differs from the current task file")
    if manifest.get("model") != expected_model:
        raise ValueError(
            "V3 model must match V2 reference model: "
            f"{expected_model!r} != {manifest.get('model')!r}"
        )
    return {
        "run_id": manifest.get("run_id"),
        "run_dir": str(run_dir.resolve()),
        "manifest_hash": canonical_hash(manifest),
        "model": manifest.get("model"),
        "supplemental_file": "supplemental_18637_oracle_ask.json",
    }


def run(args) -> int:
    tasks_path = Path(args.tasks)
    tasks = load_tasks(tasks_path)
    validation = validate_tasks(tasks)
    selected = select_pilot_tasks(tasks, args.seed)
    print(
        f"validated {validation['task_count']} tasks; selected "
        f"{[int(task['task_id']) for task in selected]} "
        f"hash={validation['task_hash'][:12]}"
    )
    if args.mode == "validate":
        return 0
    if args.mode == "real" and not args.allow_real_api:
        raise ValueError("real mode requires --allow-real-api after explicit user approval")
    if args.mode == "real" and not args.run_id:
        raise ValueError("real mode requires an explicit --run-id for safe resume")
    if args.mode == "real" and not args.reference_run_dir:
        raise ValueError("real mode requires --reference-run-dir for frozen V2 results")
    if args.mode == "real" and float(args.temperature) != 0.0:
        raise ValueError("Probe B V3 real mode requires temperature=0")
    if args.limit_tasks is not None and not 1 <= int(args.limit_tasks) <= len(selected):
        raise ValueError(f"limit_tasks must be between 1 and {len(selected)}")

    run_id = args.run_id or _default_run_id(args.mode)
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one plain directory name")
    run_dir = Path(args.outdir) / run_id
    model = args.model or os.environ.get("OPENAI_MODEL", "deepseek-chat")
    llm_base_url, shopsim_base_url = resolve_service_urls(args)
    reference = None
    if args.mode == "real":
        reference = inspect_reference_run(
            Path(args.reference_run_dir), validation["task_hash"], model
        )
    manifest = create_or_load_manifest(
        run_dir,
        run_id=run_id,
        mode=args.mode,
        tasks_path=tasks_path,
        validation=validation,
        seed=args.seed,
        max_steps=args.max_steps,
        max_llm_calls=args.max_llm_calls,
        temperature=args.temperature,
        model=model if args.mode == "real" else "mock",
        base_url=llm_base_url if args.mode == "real" else "mock://offline",
        shopsim_base_url=shopsim_base_url if args.mode == "real" else "mock://offline",
        reference=reference,
    )
    manifest_path = run_dir / MANIFEST_FILE

    def persist_call_count(used):
        manifest["llm_call_count"] = int(used)
        write_json(manifest_path, manifest)

    budget = CallBudget(
        args.max_llm_calls,
        used=int(manifest.get("llm_call_count", 0)),
        on_change=persist_call_count,
    )
    if args.mode == "real":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for real mode")

        def client_factory(task):
            return OpenAIChatClient(
                model=model,
                base_url=llm_base_url,
                api_key=api_key,
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                transport=CappedTransport(budget),
            )

        env_factory = lambda task, url: ShopAgentEnv(  # noqa: E731
            base_url=url, timeout=args.timeout
        )
    else:
        client_factory = MockAutonomousBaseClient
        env_factory = lambda task, url: FakeShopEnv(task)  # noqa: E731

    trajectories_path = run_dir / TRAJECTORIES_FILE
    existing = load_records(trajectories_path)
    attempted = {int(row["task_id"]) for row in existing}
    schedule = selected[: int(args.limit_tasks)] if args.limit_tasks else selected
    wrote = 0
    for task in schedule:
        task_id = int(task["task_id"])
        if task_id in attempted:
            continue
        if int(manifest.get("trajectory_attempt_count", 0)) >= MAX_TRAJECTORY_ATTEMPTS:
            print("trajectory attempt budget exhausted (10/10); stopping")
            break
        if args.mode == "real" and budget.used >= budget.maximum:
            print(
                f"LLM call budget exhausted ({budget.used}/{budget.maximum}); "
                "stopped before starting a new trajectory"
            )
            break
        calls_before = budget.used
        trajectory = run_autonomous_trajectory(
            task,
            base_client=client_factory(task),
            env_factory=env_factory,
            shopsim_base_url=shopsim_base_url,
            max_steps=args.max_steps,
            run_id=run_id,
            task_hash=validation["task_hash"],
            llm_calls_before=calls_before,
        )
        trajectory["probe"]["llm_calls_after"] = budget.used
        append_record(trajectories_path, trajectory)
        attempted.add(task_id)
        wrote += 1
        manifest["trajectory_attempt_count"] = int(
            manifest.get("trajectory_attempt_count", 0)
        ) + 1
        manifest.setdefault("completed_task_ids", []).append(task_id)
        if not trajectory["probe"]["valid"]:
            manifest.setdefault("invalid_task_ids", []).append(task_id)
        manifest["llm_call_count"] = budget.used
        write_json(manifest_path, manifest)
        ask_class = trajectory["probe"]["autonomous_ask"]["classification"]
        print(
            f"[{AUTONOMOUS_ARM}] task={task_id} ask={ask_class} "
            f"status={trajectory['status']} steps={len(trajectory.get('steps') or [])} "
            f"calls={budget.used}"
        )
        if not trajectory["probe"]["valid"]:
            print("invalid infrastructure/local trajectory; stopped before the next attempt")
            break

    print(f"wrote {wrote} trajectories -> {run_dir}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Probe B V3 Autonomous-Ask pilot")
    parser.add_argument("--mode", choices=["validate", "mock", "real"], default="validate")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--run-id")
    parser.add_argument("--reference-run-dir")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-llm-calls", type=int, default=DEFAULT_MAX_LLM_CALLS)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--model")
    parser.add_argument("--base-url", help="LLM OpenAI-compatible base URL")
    parser.add_argument("--shopsim-base-url")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--allow-real-api", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        return run(parse_args(argv))
    except (CallBudgetExceeded, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _question_argument(tool_call: Mapping[str, Any]) -> str:
    raw = ((tool_call.get("function") or {}).get("arguments"))
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(arguments, Mapping):
        return ""
    question = arguments.get("question")
    return str(question) if isinstance(question, str) else ""


if __name__ == "__main__":
    raise SystemExit(main())
