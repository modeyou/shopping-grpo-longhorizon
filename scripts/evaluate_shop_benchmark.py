#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.summary import summarize_trajectories
from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks
from shopping_grpo.multiturn.shopper import ShopperSimulator


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--expected-tasks",
        type=Path,
        help="Optional fixed-denominator task-ID manifest for frozen openings.",
    )
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=24576,
        help="上下文窗口；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument(
        "--observation-token-budget",
        type=int,
        default=1536,
        help="Observation token 预算；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument(
        "--condition",
        choices=(
            "standard",
            "gap-ask-enabled",
            "gap-ask-disabled",
            "complete-ask-enabled",
        ),
        default="standard",
        help="Evaluation interaction condition; gap modes require frozen opening rows.",
    )
    parser.add_argument("--shopper-model")
    parser.add_argument("--shopper-base-url")
    parser.add_argument("--shopper-api-key")
    parser.add_argument("--max-shopper-questions", type=int, default=2)
    parser.add_argument(
        "--disable-model-thinking",
        action="store_true",
        help="Pass enable_thinking=false to the Actor chat template.",
    )
    parser.add_argument(
        "--disable-shopper-thinking",
        action="store_true",
        help="Pass enable_thinking=false to the Shopper chat template.",
    )
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window < 0:
        raise SystemExit("--context-window 不能为负数")
    if args.context_window and args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    if args.observation_token_budget < 0:
        raise SystemExit("--observation-token-budget 不能为负数")
    if args.max_shopper_questions < 0:
        raise SystemExit("--max-shopper-questions 不能为负数")
    ask_enabled = args.condition in {
        "gap-ask-enabled", "complete-ask-enabled",
    }
    if ask_enabled and not (
        args.shopper_model and args.shopper_base_url and args.shopper_api_key
    ):
        raise SystemExit(
            f"{args.condition} requires --shopper-model, --shopper-base-url "
            "and --shopper-api-key"
        )
    tasks = load_tasks(args.benchmark)
    expected_tasks = load_tasks(args.expected_tasks) if args.expected_tasks else tasks
    expected_ids = [int(task["task_id"]) for task in expected_tasks]
    actual_ids = [int(task["task_id"]) for task in tasks]
    if len(expected_ids) != len(set(expected_ids)):
        raise SystemExit("--expected-tasks contains duplicate task IDs")
    if len(actual_ids) != len(set(actual_ids)):
        raise SystemExit("--benchmark contains duplicate task IDs")
    if set(actual_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise SystemExit(
            "benchmark task IDs differ from fixed denominator: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
        chat_template_kwargs=(
            {"enable_thinking": False}
            if args.disable_model_thinking else None
        ),
    )
    shopper = None
    if ask_enabled:
        shopper = ShopperSimulator(OpenAIChatClient(
            model=args.shopper_model,
            base_url=args.shopper_base_url,
            api_key=args.shopper_api_key,
            temperature=0.0,
            top_p=1.0,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            chat_template_kwargs=(
                {"enable_thinking": False}
                if args.disable_shopper_thinking else None
            ),
        ))
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        shopper=shopper,
        max_shopper_questions=args.max_shopper_questions,
        evaluation_condition=(
            None if args.condition == "standard" else args.condition
        ),
    )
    summary = summarize_trajectories(
        expected_ids, _read_jsonl(args.output)
    )
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "expected_tasks": str(args.expected_tasks or args.benchmark),
        "model": args.model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
        "condition": args.condition,
        "shopper_model": args.shopper_model if ask_enabled else None,
        "max_shopper_questions": args.max_shopper_questions,
        "disable_model_thinking": args.disable_model_thinking,
        "disable_shopper_thinking": args.disable_shopper_thinking,
        "shop_step_budget": args.max_steps,
        "ask_budget_is_separate_from_shop_steps": args.condition != "standard",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
