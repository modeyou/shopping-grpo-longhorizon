#!/usr/bin/env python3
"""Generate and freeze one vague Shopper opening per ShopSimulator task."""

import argparse
import json
import os
import sys
from pathlib import Path

from shopping_grpo.collection.sft import task_ids_from_jsonl
from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.evaluation.rollout import OpenAIChatClient, load_tasks
from shopping_grpo.multiturn.shopper import OPENING_PROMPT_HASH, ShopperSimulator
from shopping_grpo.multiturn.tasks import build_task_row, source_goal_hash


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-tasks", type=Path, default=Path("data/evaluation/tasks.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-url", default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--opening-attempts", type=int, default=3,
        help="Maximum attempts to generate and repair each opening.",
    )
    parser.add_argument(
        "--disable-model-thinking", action="store_true",
        help="Disable chat-template thinking for structured opening generation.",
    )
    return parser.parse_args()


def read_existing(path):
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["task_id"])] = row
    return rows


def main():
    args = parse_args()
    if not args.llm_base_url or not args.api_key:
        raise SystemExit("OPENAI_BASE_URL and OPENAI_API_KEY are required")
    if args.opening_attempts < 1:
        raise SystemExit("--opening-attempts must be at least 1")
    held_out = task_ids_from_jsonl(args.held_out_tasks)
    tasks = [row for row in load_tasks(args.tasks) if int(row["task_id"]) not in held_out]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    existing = read_existing(args.output)
    client = OpenAIChatClient(
        model=args.model, base_url=args.llm_base_url, api_key=args.api_key,
        temperature=0.0, timeout=args.timeout, max_tokens=args.max_tokens,
        chat_template_kwargs=(
            {"enable_thinking": False} if args.disable_model_thinking else None
        ),
    )
    shopper = ShopperSimulator(client)
    generated = 0
    failed = 0
    skipped_existing = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for task in tasks:
            task_id = int(task["task_id"])
            with ShopAgentEnv(base_url=args.base_url, multiturn=True) as env:
                env.reset(task_id)
                context = env.shopper_context
                prior = existing.get(task_id)
                if prior is not None:
                    if prior.get("source_goal_hash") != source_goal_hash(context):
                        raise RuntimeError(f"source goal changed for task {task_id}")
                    skipped_existing += 1
                    continue
                try:
                    opening = shopper.generate_initial_request(
                        context, max_attempts=args.opening_attempts,
                    )
                except ValueError as exc:
                    failed += 1
                    print(
                        f"task={task_id} opening_error="
                        f"{type(exc).__name__}:{exc}",
                        file=sys.stderr,
                    )
                    continue
                row = build_task_row(
                    task_id, opening["initial_request"], context,
                    args.model, OPENING_PROMPT_HASH,
                    omitted_dimensions=opening["omitted_dimensions"],
                    omitted_facts=opening["omitted_facts"],
                )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                existing[task_id] = row
                generated += 1
                print(
                    f"task={task_id} opening={opening['initial_request']} "
                    f"omitted={opening['omitted_dimensions']}"
                )
    print(json.dumps({
        "selected": len(tasks),
        "generated": generated,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "shopper_calls": shopper.call_count,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
