#!/usr/bin/env python3
"""Apply the five-panel evaluator to one saved multi-turn rollout condition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    index_jsonl,
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.comparison import MULTITURN_CONDITIONS
from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    rubric_ids,
    validate_judge_result,
)
from shopping_grpo.evaluation.manifest import build_run_manifest, sha256_file
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.model_client import OpenAIJSONClient
from shopping_grpo.evaluation.prompts import (
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_trajectory_judge_messages,
)
from shopping_grpo.evaluation.results import (
    assemble_task_evaluation,
    build_not_judged_result,
    summarize_evaluations,
)
from shopping_grpo.evaluation.trajectory import normalize_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tasks", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actor-label", required=True)
    parser.add_argument(
        "--condition", choices=MULTITURN_CONDITIONS, required=True
    )
    parser.add_argument(
        "--judge-model", default="deepseek-v4-flash-0731"
    )
    parser.add_argument(
        "--judge-base-url", default=os.environ.get("OPENAI_BASE_URL")
    )
    parser.add_argument(
        "--judge-api-key", default=os.environ.get("OPENAI_API_KEY")
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _task_ids(path: Path) -> list[int]:
    task_ids = [int(row["task_id"]) for row in iter_jsonl(path)]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path} contains duplicate task IDs")
    return task_ids


def _judge(client, normalized, metrics, rubric, schema_retries):
    messages = build_trajectory_judge_messages(
        normalized=normalized,
        rubric_bundle=rubric,
        deterministic_metrics=metrics,
    )
    allowed_events = [
        event["event_id"]
        for event in normalized.get("events") or []
        if event.get("event_id")
    ]
    last_error = None
    for attempt in range(schema_retries + 1):
        response = client.complete_json(messages)
        try:
            validated = validate_judge_result(
                response["result"],
                rubric_ids=rubric_ids(rubric),
                expected_task_id=normalized["task_id"],
                expected_trajectory_id=normalized["trajectory_id"],
                allowed_event_ids=allowed_events,
            )
            return response, validated
        except ContractValidationError as exc:
            last_error = exc
            if attempt >= schema_retries:
                break
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            response["result"], ensure_ascii=False
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "上一个 JSON 未通过冻结 schema："
                            f"{exc}。只修复 JSON；不得新增 rubric_id 或 event_id。"
                        ),
                    },
                ]
            )
    raise ContractValidationError(
        f"task {normalized['task_id']} judge schema retries exhausted: {last_error}"
    )


def main():
    args = parse_args()
    if not args.judge_base_url or not args.judge_api_key:
        raise SystemExit(
            "--judge-base-url/--judge-api-key or OPENAI_BASE_URL/OPENAI_API_KEY "
            "are required"
        )
    if args.max_tokens < 1 or args.retries < 0 or args.schema_retries < 0:
        raise SystemExit("token and retry limits are invalid")
    expected_ids = _task_ids(args.expected_tasks)
    expected_set = set(expected_ids)
    trajectories = index_jsonl(
        args.trajectories, key="task_id", allowed_keys=expected_set
    )
    rubrics = index_jsonl(
        args.rubrics, key="task_id", allowed_keys=expected_set
    )
    missing_rubrics = sorted(expected_set - set(rubrics))
    if missing_rubrics:
        raise SystemExit(f"missing frozen rubrics: {missing_rubrics[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judges_path = args.output_dir / "judges.jsonl"
    final_paths = {
        "preprocessed": args.output_dir / "preprocessed.jsonl",
        "evaluations": args.output_dir / "evaluations.jsonl",
        "summary": args.output_dir / "evaluation_summary.json",
        "manifest": args.output_dir / "run_manifest.json",
    }
    if not args.resume and (
        judges_path.exists() or any(path.exists() for path in final_paths.values())
    ):
        raise SystemExit(
            f"output already exists under {args.output_dir}; pass --resume"
        )
    cached_judges = (
        index_jsonl(
            judges_path, key="task_id", allowed_keys=expected_set
        )
        if judges_path.exists()
        else {}
    )
    client = OpenAIJSONClient(
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        response_format_json=True,
        thinking=False,
    )
    preprocessed = []
    evaluations = []
    for index, task_id in enumerate(expected_ids, start=1):
        raw = trajectories.get(task_id)
        if raw is None:
            continue
        normalized = normalize_trajectory(raw)
        metrics = compute_deterministic_metrics(normalized)
        actual_condition = metrics["clarification"]["interaction_mode"]
        if actual_condition != args.condition:
            raise SystemExit(
                f"task {task_id} interaction_mode={actual_condition!r}, "
                f"expected {args.condition!r}"
            )
        preprocessed.append(
            {
                "task_id": task_id,
                "trajectory_id": normalized["trajectory_id"],
                "normalized_trajectory": normalized,
                "deterministic_metrics": metrics,
            }
        )
        if task_id in cached_judges:
            cached = cached_judges[task_id]
            if cached.get("trajectory_id") != normalized["trajectory_id"]:
                raise SystemExit(f"cached trajectory mismatch for task {task_id}")
            judge_result = cached["judge_result"]
        elif metrics["validity"]["infrastructure_invalid"]:
            judge_result = build_not_judged_result(
                task_id=task_id,
                trajectory_id=normalized["trajectory_id"],
                reason="infrastructure_invalid",
            )
            append_jsonl_fsync(
                judges_path,
                {
                    "task_id": task_id,
                    "trajectory_id": normalized["trajectory_id"],
                    "judge_result": judge_result,
                    "request_metadata": None,
                },
            )
        else:
            response, judge_result = _judge(
                client,
                normalized,
                metrics,
                rubrics[task_id],
                args.schema_retries,
            )
            append_jsonl_fsync(
                judges_path,
                {
                    "task_id": task_id,
                    "trajectory_id": normalized["trajectory_id"],
                    "judge_result": judge_result,
                    "request_metadata": response["metadata"],
                },
            )
        evaluations.append(
            assemble_task_evaluation(
                actor={
                    "label": args.actor_label,
                    "condition": args.condition,
                },
                normalized_trajectory=normalized,
                deterministic_metrics=metrics,
                rubric_bundle=rubrics[task_id],
                judge_result=judge_result,
            )
        )
        print(f"evaluate {index}/{len(expected_ids)} task={task_id}")

    write_jsonl_atomic(
        final_paths["preprocessed"], preprocessed, force=args.resume
    )
    write_jsonl_atomic(
        final_paths["evaluations"], evaluations, force=args.resume
    )
    summary = summarize_evaluations(
        expected_task_ids=expected_ids,
        evaluations=evaluations,
    )
    write_json_atomic(final_paths["summary"], summary, force=args.resume)
    reward_versions = sorted(
        {
            str(
                record["reward_and_terminal"]["metrics"].get(
                    "reward_version"
                )
            )
            for record in evaluations
            if record["reward_and_terminal"]["metrics"].get(
                "reward_version"
            )
        }
    )
    manifest = build_run_manifest(
        run_id=f"{args.actor_label}-{args.condition}",
        actor={"label": args.actor_label},
        task_manifest={
            "path": str(args.expected_tasks),
            "sha256": sha256_file(args.expected_tasks),
            "task_count": len(expected_ids),
        },
        environment={"reward_versions": reward_versions},
        protocol={"condition": args.condition, "composite_score": None},
        code={"repository": "shopping-grpo-longhorizon"},
        judge={
            "model": args.judge_model,
            "prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
            "thinking": False,
        },
        outputs={
            "trajectories_sha256": sha256_file(args.trajectories),
            "rubrics_sha256": sha256_file(args.rubrics),
            "judges_sha256": sha256_file(judges_path),
            "evaluations_sha256": sha256_file(final_paths["evaluations"]),
            "summary_sha256": sha256_file(final_paths["summary"]),
        },
    )
    write_json_atomic(final_paths["manifest"], manifest, force=args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
