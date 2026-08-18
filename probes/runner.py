"""Probe B V2 controlled runner: No-Ask versus fixed Oracle-Ask.

Examples:
  python probes/runner.py --mode validate
  python probes/runner.py --mode mock --run-id mock-v2
  python probes/runner.py --mode real --run-id oracle-v2-smoke --limit-pairs 1 --allow-real-api

Real API execution is deliberately gated by ``--allow-real-api``. The runner
never exposes an ask tool: the Oracle arm receives one frozen dialogue before
the first model-generated shopping action.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen


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
from task_schema import (  # noqa: E402
    DEFAULT_TASKS,
    canonical_hash,
    load_tasks,
    validate_tasks,
)


ARMS = ("no_ask", "oracle_ask")
DEFAULT_SEED = 20260818
DEFAULT_MAX_STEPS = 35
DEFAULT_MAX_LLM_CALLS = 700
DEFAULT_SHOPSIM_BASE_URL = "http://127.0.0.1:5700"
DEFAULT_OUTPUT_ROOT = PROBE_DIR / "outputs" / "v2"
TRAJECTORIES_FILE = "trajectories.jsonl"
MANIFEST_FILE = "manifest.json"


class CallBudgetExceeded(RuntimeError):
    """The configured global LLM request cap has been reached."""


class CallBudget:
    def __init__(self, maximum: int, used: int = 0, on_change=None):
        self.maximum = int(maximum)
        self.used = int(used)
        self.on_change = on_change
        if self.maximum < 1:
            raise ValueError("max_llm_calls must be positive")
        if not 0 <= self.used <= self.maximum:
            raise ValueError("persisted llm_call_count is outside the configured budget")

    def consume(self) -> None:
        if self.used >= self.maximum:
            raise CallBudgetExceeded(
                f"LLM call budget exhausted ({self.used}/{self.maximum}); no request was sent"
            )
        self.used += 1
        if self.on_change is not None:
            self.on_change(self.used)


class CappedTransport:
    """Count every HTTP attempt, including retries inside OpenAIChatClient."""

    def __init__(self, budget: CallBudget):
        self.budget = budget

    def __call__(self, url, payload, headers, timeout):
        self.budget.consume()
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class InstructionOverrideEnv:
    """Keep the original hidden TaskFacts while replacing only visible user text."""

    def __init__(self, inner, *, visible_instruction: str, expected_clear_query: str):
        self.inner = inner
        self.visible_instruction = str(visible_instruction)
        self.expected_clear_query = str(expected_clear_query)

    def reset(self, task_id):
        result = dict(self.inner.reset(task_id))
        original = str(result.get("instruction") or "")
        if original and _strip_instruction_prefix(original) != self.expected_clear_query.strip():
            raise ValueError(
                "ShopSimulator instruction differs from frozen clear_query for "
                f"task_id={task_id}"
            )
        result["environment_instruction"] = original
        result["instruction"] = self.visible_instruction
        return result

    def step(self, action):
        return self.inner.step(action)

    def release(self):
        return self.inner.release()


class FakeShopEnv:
    """Offline ShopSimulator substitute used by the full V2 mock run."""

    def __init__(self, task: Mapping[str, Any]):
        self.task = task
        self.released = False

    def reset(self, task_id):
        return {"env_idx": 0, "instruction": f"Instruction: {self.task['clear_query']}"}

    def step(self, action):
        if action == "search[探针测试商品]":
            return {
                "instruction": (
                    "[SHOPPING_OBSERVATION_V2]\npage_type: search_results\n"
                    "1|100000000001|1.0|测试品牌|测试品类|测试属性|探针测试商品\n"
                    '可点击的按钮: ["100000000001"]'
                ),
                "reward": 0.0,
                "done": False,
            }
        if action == "click[100000000001]":
            return {
                "instruction": (
                    "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
                    "asin: 100000000001\nprice: 1.0\n"
                    '可点击的按钮: ["Buy Now"]'
                ),
                "reward": 0.0,
                "done": False,
            }
        if action == "click[Buy Now]":
            purchase = _mock_purchase(self.task)
            detail = {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
                "terminal_utility": 1.0,
                "weighted_score": 1.0,
            }
            return {
                "instruction": "done",
                "reward": 1.0,
                "done": True,
                "over": True,
                "purchase": purchase,
                "reward_detail": detail,
                "termination_reason": "gold_purchase",
            }
        raise AssertionError(f"unexpected mock action: {action}")

    def release(self):
        self.released = True


class MockClient:
    def __init__(self):
        self.responses = [
            _assistant_tool("search_products", {"query": "探针测试商品"}, "mock_search"),
            _assistant_tool("open_product", {"asin": "100000000001"}, "mock_open"),
            _assistant_tool("buy_now", {}, "mock_buy"),
        ]

    def complete(self, messages, tools):
        if not self.responses:
            return {"role": "assistant", "content": "mock exhausted"}
        return self.responses.pop(0)


def build_arm_inputs(task: Mapping[str, Any], arm: str) -> tuple[list[dict], str]:
    """Return prompt prefix and the final user message supplied by reset."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    system = {"role": "system", "content": SYSTEM_PROMPT}
    under = "Instruction: " + str(task["under_query"])
    if arm == "no_ask":
        return [system], under
    oracle = task["oracle_turn"]
    return (
        [
            system,
            {"role": "user", "content": under},
            {"role": "assistant", "content": str(oracle["question"])},
        ],
        str(oracle["answer"]),
    )


def schedule_pairs(
    tasks: list[dict], seed: int, limit_pairs: int | None = None
) -> list[tuple[dict, str]]:
    rng = random.Random(int(seed))
    ordered = list(tasks)
    rng.shuffle(ordered)
    if limit_pairs is not None:
        if not 1 <= int(limit_pairs) <= len(ordered):
            raise ValueError(f"limit_pairs must be between 1 and {len(ordered)}")
        ordered = ordered[: int(limit_pairs)]
    schedule: list[tuple[dict, str]] = []
    for task in ordered:
        arms = list(ARMS)
        rng.shuffle(arms)
        schedule.extend((task, arm) for arm in arms)
    return schedule


def run_trajectory(
    task: Mapping[str, Any],
    arm: str,
    *,
    client,
    env_factory,
    shopsim_base_url: str,
    max_steps: int,
    run_id: str,
    task_hash: str,
    llm_calls_before: int,
) -> dict[str, Any]:
    prompt, visible_instruction = build_arm_inputs(task, arm)

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
    trajectory["arm"] = arm
    valid, validity_reason = _trajectory_validity(trajectory)
    trajectory["probe"] = {
        "query": task["under_query"],
        "clear_query": task["clear_query"],
        "latent_goal": deepcopy(task["latent_goal"]),
        "oracle_turn": deepcopy(task["oracle_turn"]) if arm == "oracle_ask" else None,
        "oracle_injected": arm == "oracle_ask",
        "task_hash": task_hash,
        "llm_calls_before": int(llm_calls_before),
        "valid": valid,
        "validity_reason": validity_reason,
    }
    return trajectory


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
) -> dict[str, Any]:
    path = run_dir / MANIFEST_FILE
    immutable = {
        "run_id": run_id,
        "mode": mode,
        "task_file": str(tasks_path.resolve()),
        "task_hash": validation["task_hash"],
        "task_count": validation["task_count"],
        "field_distribution": validation["distribution"],
        "seed": int(seed),
        "max_steps": int(max_steps),
        "max_llm_calls": int(max_llm_calls),
        "temperature": float(temperature),
        "model": model,
        "api_base": _public_api_base(base_url),
        "shopsim_base": _public_api_base(shopsim_base_url),
        "system_prompt_hash": canonical_hash(SYSTEM_PROMPT),
        "tool_schema_hash": canonical_hash(SHOP_TOOL_SCHEMAS),
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
        "completed_keys": [],
        "invalid_keys": [],
    }
    write_json(path, manifest)
    return manifest


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args) -> int:
    tasks_path = Path(args.tasks)
    tasks = load_tasks(tasks_path)
    validation = validate_tasks(tasks)
    print(
        "validated "
        f"{validation['task_count']} tasks {validation['distribution']} "
        f"hash={validation['task_hash'][:12]}"
    )
    if args.mode == "validate":
        return 0
    if args.mode == "real" and not args.allow_real_api:
        raise ValueError("real mode requires --allow-real-api after explicit user approval")
    if args.mode == "real" and not args.run_id:
        raise ValueError("real mode requires an explicit --run-id for safe resume")
    if args.mode == "real" and float(args.temperature) != 0.0:
        raise ValueError("Probe B V2 real mode requires temperature=0")

    run_id = args.run_id or _default_run_id(args.mode)
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one plain directory name")
    run_dir = Path(args.outdir) / run_id
    model = args.model or os.environ.get("OPENAI_MODEL", "deepseek-chat")
    llm_base_url, shopsim_base_url = resolve_service_urls(args)
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
        shopsim_base_url=(
            shopsim_base_url if args.mode == "real" else "mock://offline"
        ),
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
        client_factory = lambda: OpenAIChatClient(  # noqa: E731
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
        client_factory = MockClient
        env_factory = lambda task, url: FakeShopEnv(task)  # noqa: E731

    trajectories_path = run_dir / TRAJECTORIES_FILE
    existing = load_records(trajectories_path)
    attempted = {(int(row["task_id"]), str(row["arm"])) for row in existing}
    schedule = schedule_pairs(tasks, args.seed, args.limit_pairs)
    task_hash = validation["task_hash"]
    wrote = 0
    for task, arm in schedule:
        key = (int(task["task_id"]), arm)
        if key in attempted:
            continue
        if int(manifest.get("trajectory_attempt_count", 0)) >= 50:
            print("trajectory attempt budget exhausted (50/50); stopping")
            break
        if args.mode == "real" and budget.used >= budget.maximum:
            print(
                f"LLM call budget exhausted ({budget.used}/{budget.maximum}); "
                "stopped before starting a new trajectory"
            )
            break
        calls_before = budget.used
        trajectory = run_trajectory(
            task,
            arm,
            client=client_factory(),
            env_factory=env_factory,
            shopsim_base_url=shopsim_base_url,
            max_steps=args.max_steps,
            run_id=run_id,
            task_hash=task_hash,
            llm_calls_before=calls_before,
        )
        trajectory["probe"]["llm_calls_after"] = budget.used
        append_record(trajectories_path, trajectory)
        attempted.add(key)
        wrote += 1
        manifest["trajectory_attempt_count"] = int(manifest.get("trajectory_attempt_count", 0)) + 1
        key_text = f"{key[0]}:{key[1]}"
        manifest.setdefault("completed_keys", []).append(key_text)
        if not trajectory["probe"]["valid"]:
            manifest.setdefault("invalid_keys", []).append(key_text)
        manifest["llm_call_count"] = budget.used
        write_json(manifest_path, manifest)
        print(
            f"[{arm}] task={key[0]} status={trajectory['status']} "
            f"steps={len(trajectory.get('steps') or [])} calls={budget.used}"
        )
        if not trajectory["probe"]["valid"]:
            print("invalid infrastructure/local trajectory; stopped before the next attempt")
            break

    print(f"wrote {wrote} trajectories -> {run_dir}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Probe B V2 controlled paired runner")
    parser.add_argument("--mode", choices=["validate", "mock", "real"], default="validate")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument(
        "--run-id", help="required for stable resume; auto-generated only when omitted"
    )
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-llm-calls", type=int, default=DEFAULT_MAX_LLM_CALLS)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--model")
    parser.add_argument("--base-url", help="LLM OpenAI-compatible base URL")
    parser.add_argument(
        "--shopsim-base-url",
        help=(
            "ShopSimulator service URL "
            "(default: SHOPSIM_BASE_URL or 127.0.0.1:5700)"
        ),
    )
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--allow-real-api", action="store_true")
    return parser.parse_args(argv)


def resolve_service_urls(args) -> tuple[str, str]:
    """Resolve the LLM and ShopSimulator endpoints without conflating them."""

    llm_base_url = args.base_url or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    shopsim_base_url = args.shopsim_base_url or os.environ.get(
        "SHOPSIM_BASE_URL", DEFAULT_SHOPSIM_BASE_URL
    )
    return llm_base_url.rstrip("/"), shopsim_base_url.rstrip("/")


def main(argv=None) -> int:
    try:
        return run(parse_args(argv))
    except (CallBudgetExceeded, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _assistant_tool(name: str, arguments: Mapping[str, Any], call_id: str) -> dict[str, Any]:
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


def _mock_purchase(task: Mapping[str, Any]) -> dict[str, Any]:
    latent = task["latent_goal"]
    normalized = latent["normalized_value"]
    purchase = {
        "asin": "100000000001",
        "title": "探针测试商品",
        "price": 1.0,
        "brand": "测试品牌",
        "options": {},
        "attributes": [],
    }
    field = latent["field"]
    if field == "budget":
        purchase["price"] = max(0.01, float(normalized["amount"]) - 1.0)
    elif field == "brand":
        purchase["brand"] = normalized["value"]
    elif field == "color":
        purchase["options"] = {"颜色": normalized["value"]}
    elif field == "origin":
        purchase["attributes"] = [normalized["value"]]
    return purchase


def _trajectory_validity(trajectory: Mapping[str, Any]) -> tuple[bool, str]:
    if trajectory.get("release_error"):
        return False, "environment_release_error"
    error = trajectory.get("error")
    if isinstance(error, Mapping):
        steps = trajectory.get("steps") or []
        last_step = steps[-1] if steps else {}
        messages = trajectory.get("messages") or []
        last_message = messages[-1] if messages else {}
        has_agent_tool_call = bool(
            last_step.get("tool_call") or last_message.get("tool_calls")
        )
        error_type = str(error.get("type") or "")
        if has_agent_tool_call and error_type in {
            "JSONDecodeError",
            "KeyError",
            "TypeError",
            "ValueError",
        }:
            return True, "agent_malformed_tool_call"
        return False, "infrastructure_or_local_error"
    if trajectory.get("status") == "environment_release_failed":
        return False, "environment_release_error"
    return True, "behavioral_result"


def _strip_instruction_prefix(value: str) -> str:
    text = str(value).strip()
    if text.startswith("Instruction:"):
        return text[len("Instruction:") :].strip()
    return text


def _default_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}-{stamp}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _public_api_base(base_url: str) -> str:
    return str(base_url).split("?", 1)[0].split("#", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
