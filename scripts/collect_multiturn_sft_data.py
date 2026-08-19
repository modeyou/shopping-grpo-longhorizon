#!/usr/bin/env python3
"""Collect native multi-turn Teacher trajectories and build SFT artifacts."""

import argparse
import json
import os
from pathlib import Path

from shopping_grpo.collection.sft import build_collection_artifacts, task_ids_from_jsonl
from shopping_grpo.evaluation.rollout import (
    OpenAIChatClient, append_jsonl, collect_for_task, completed_task_attempts, load_tasks,
)
from shopping_grpo.multiturn.shopper import ShopperSimulator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-tasks", type=Path, default=Path("data/evaluation/tasks.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attempts-per-task", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--shopper-model")
    parser.add_argument("--shopper-base-url")
    parser.add_argument("--shopper-api-key")
    parser.add_argument("--max-shopper-questions", type=int, default=2)
    parser.add_argument(
        "--teacher-first-ask", action="store_true",
        help="Force only the first Teacher tool choice to ask_shopper; never use for evaluation.",
    )
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def make_client(model, base_url, api_key, args):
    return OpenAIChatClient(
        model=model, base_url=base_url, api_key=api_key, temperature=0.0,
        timeout=args.timeout, max_tokens=args.max_tokens,
    )


def main():
    args = parse_args()
    if not args.llm_base_url or not args.api_key:
        raise SystemExit("OPENAI_BASE_URL and OPENAI_API_KEY are required")
    if args.max_shopper_questions < 0:
        raise SystemExit("--max-shopper-questions must be non-negative")
    if args.teacher_first_ask and args.max_shopper_questions < 1:
        raise SystemExit("--teacher-first-ask requires --max-shopper-questions >= 1")
    shopper_model = args.shopper_model or args.model
    shopper_base = args.shopper_base_url or args.llm_base_url
    shopper_key = args.shopper_api_key or args.api_key
    held_out = task_ids_from_jsonl(args.held_out_tasks)
    tasks = [row for row in load_tasks(args.tasks) if int(row["task_id"]) not in held_out]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / "raw.jsonl"
    completed = completed_task_attempts(raw)
    actor = make_client(args.model, args.llm_base_url, args.api_key, args)
    shopper = ShopperSimulator(make_client(shopper_model, shopper_base, shopper_key, args))
    written = 0
    for task in tasks:
        for attempt in range(args.attempts_per_task):
            key = (int(task["task_id"]), attempt)
            if key in completed:
                continue
            trajectory = collect_for_task(
                task, client=actor, base_url=args.base_url, max_steps=args.max_steps,
                attempt_index=attempt, shopper=shopper,
                max_shopper_questions=args.max_shopper_questions,
                teacher_first_ask=args.teacher_first_ask,
            )
            append_jsonl(raw, [trajectory])
            written += 1
            print(f"task={task['task_id']} status={trajectory['status']} asks={len(trajectory['shopper_questions'])} actor_calls={trajectory['actor_llm_calls']} shopper_calls={trajectory['shopper_llm_calls']}")
    config = {
        "tasks": str(args.tasks), "model": args.model,
        "shopper_model": shopper_model, "max_shopper_questions": args.max_shopper_questions,
        "max_steps": args.max_steps, "attempts_per_task": args.attempts_per_task,
        "opening_policy": "frozen-once",
        "teacher_first_ask": args.teacher_first_ask,
    }
    summary = build_collection_artifacts(
        raw_path=raw, output_dir=args.output_dir, held_out_task_ids=held_out,
        validation_ratio=args.validation_ratio, seed=args.seed, collection_config=config,
    )
    print(f"collected_raw={written}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
