#!/usr/bin/env python3
"""Apply the five-panel evaluator to one saved multi-turn rollout condition."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    index_jsonl,
    iter_jsonl,
    load_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.blind_guard import guard_declared_final_tasks
from shopping_grpo.evaluation.comparison import MULTITURN_CONDITIONS
from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    RUBRIC_APPROVAL_SCHEMA_VERSION,
    rubric_ids,
    validate_rubric_bundle,
    validate_judge_result,
)
from shopping_grpo.evaluation.manifest import (
    build_run_manifest,
    canonical_json_sha256,
    sha256_file,
)
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


PANEL_RUN_VERSION = "shopping-multiturn-panel-run-v3"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tasks", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--rubric-approval", type=Path, required=True)
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
    parser.add_argument("--allow-blind-final", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _task_ids(path: Path) -> list[int]:
    task_ids = [int(row["task_id"]) for row in iter_jsonl(path)]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path} contains duplicate task IDs")
    return task_ids


def _judge(
    client,
    normalized,
    metrics,
    rubric,
    schema_retries,
    *,
    run_plan_sha256: str,
    on_request,
):
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
    request_ids = []
    last_error = None
    for attempt in range(schema_retries + 1):
        request = {
            "schema_version": "shopping-judge-request-v1",
            "run_plan_sha256": run_plan_sha256,
            "task_id": int(normalized["task_id"]),
            "trajectory_id": str(normalized["trajectory_id"]),
            "schema_attempt": attempt,
            "messages": deepcopy(messages),
        }
        request["request_sha256"] = canonical_json_sha256(request)
        request["request_id"] = request["request_sha256"]
        on_request(request)
        request_ids.append(request["request_id"])
        response = client.complete_json(messages)
        try:
            validated = validate_judge_result(
                response["result"],
                rubric_ids=rubric_ids(rubric),
                expected_task_id=normalized["task_id"],
                expected_trajectory_id=normalized["trajectory_id"],
                allowed_event_ids=allowed_events,
            )
            return response, validated, request_ids
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


def _run_plan(args) -> dict:
    """Freeze all non-secret inputs that determine a panel evaluation run."""

    return {
        "schema_version": PANEL_RUN_VERSION,
        "expected_tasks_sha256": sha256_file(args.expected_tasks),
        "trajectories_sha256": sha256_file(args.trajectories),
        "rubrics_sha256": sha256_file(args.rubrics),
        "rubric_approval_sha256": sha256_file(args.rubric_approval),
        "actor": {"label": args.actor_label},
        "condition": args.condition,
        "judge": {
            "model": args.judge_model,
            "base_url": args.judge_base_url,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "retries": args.retries,
            "schema_retries": args.schema_retries,
            "prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
            "thinking": False,
            "temperature": 0.0,
        },
    }


def _validate_cached_request(cached: dict, request: dict) -> None:
    fields = (
        "run_plan_sha256",
        "task_id",
        "trajectory_id",
        "schema_attempt",
        "request_sha256",
    )
    if any(cached.get(field) != request[field] for field in fields):
        raise SystemExit(
            f"cached Judge request does not match request {request['request_id']}"
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
    guard_declared_final_tasks(
        args.expected_tasks,
        allowed=args.allow_blind_final,
    )
    run_plan = _run_plan(args)
    run_plan_sha256 = canonical_json_sha256(run_plan)
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
    missing_trajectories = sorted(expected_set - set(trajectories))
    if missing_trajectories:
        raise SystemExit(
            f"missing trajectories: {missing_trajectories[:10]} "
            f"(total={len(missing_trajectories)})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judges_path = args.output_dir / "judges.jsonl"
    requests_path = args.output_dir / "judge_requests.jsonl"
    final_paths = {
        "preprocessed": args.output_dir / "preprocessed.jsonl",
        "evaluations": args.output_dir / "evaluations.jsonl",
        "summary": args.output_dir / "evaluation_summary.json",
        "manifest": args.output_dir / "run_manifest.json",
    }
    if not args.resume and (
        judges_path.exists()
        or requests_path.exists()
        or any(path.exists() for path in final_paths.values())
    ):
        raise SystemExit(
            f"output already exists under {args.output_dir}; pass --resume"
        )
    if args.resume and final_paths["manifest"].exists():
        previous_manifest = load_json(final_paths["manifest"])
        if previous_manifest.get("run_plan_sha256") != run_plan_sha256:
            raise SystemExit(
                "Judge resume plan mismatch; use a new output directory for a "
                "different input, actor, condition, model, or prompt"
            )
    cached_judges = (
        index_jsonl(
            judges_path, key="task_id", allowed_keys=expected_set
        )
        if judges_path.exists()
        else {}
    )
    approval = load_json(args.rubric_approval)
    if approval.get("schema_version") != RUBRIC_APPROVAL_SCHEMA_VERSION:
        raise SystemExit("unsupported rubric approval manifest")
    rubrics_sha256 = sha256_file(args.rubrics)
    if approval.get("approved_rubrics_sha256") != rubrics_sha256:
        raise SystemExit("rubric approval does not match --rubrics")
    if approval.get("task_count") != len(expected_ids):
        raise SystemExit("rubric approval task_count does not match expected tasks")
    for task_id, bundle in rubrics.items():
        try:
            validate_rubric_bundle(
                bundle, expected_task_id=task_id, require_approved=True
            )
        except ContractValidationError as exc:
            raise SystemExit(
                f"unapproved or invalid rubric task {task_id}: {exc}"
            ) from exc
    cached_requests = (
        index_jsonl(requests_path, key="request_id")
        if requests_path.exists()
        else {}
    )

    def record_request(request: dict) -> None:
        cached = cached_requests.get(request["request_id"])
        if cached is not None:
            _validate_cached_request(cached, request)
            return
        append_jsonl_fsync(requests_path, request)
        cached_requests[request["request_id"]] = request

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
        if raw is None:  # guarded before the loop; preserve the invariant here.
            raise RuntimeError(f"missing trajectory after coverage check: {task_id}")
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
            if cached.get("run_plan_sha256") != run_plan_sha256:
                raise SystemExit(f"cached Judge run plan mismatch for task {task_id}")
            if cached.get("trajectory_id") != normalized["trajectory_id"]:
                raise SystemExit(f"cached trajectory mismatch for task {task_id}")
            judge_result = cached["judge_result"]
            request_ids = cached.get("request_ids")
            if not isinstance(request_ids, list):
                raise SystemExit(
                    f"cached Judge result lacks request IDs for task {task_id}"
                )
            if judge_result.get("judge_status") != "not_judged" and not request_ids:
                raise SystemExit(
                    f"cached Judge result lacks auditable requests for task {task_id}"
                )
            if any(
                not isinstance(request_id, str)
                or request_id not in cached_requests
                for request_id in request_ids
            ):
                raise SystemExit(
                    f"cached Judge result has missing request artifacts for task {task_id}"
                )
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
                    "run_plan_sha256": run_plan_sha256,
                    "trajectory_id": normalized["trajectory_id"],
                    "judge_result": judge_result,
                    "request_metadata": None,
                    "request_ids": [],
                },
            )
        else:
            response, judge_result, request_ids = _judge(
                client,
                normalized,
                metrics,
                rubrics[task_id],
                args.schema_retries,
                run_plan_sha256=run_plan_sha256,
                on_request=record_request,
            )
            append_jsonl_fsync(
                judges_path,
                {
                    "task_id": task_id,
                    "run_plan_sha256": run_plan_sha256,
                    "trajectory_id": normalized["trajectory_id"],
                    "judge_result": judge_result,
                    "request_metadata": response["metadata"],
                    "request_ids": request_ids,
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

    if len(evaluations) != len(expected_ids):
        raise RuntimeError("evaluation coverage invariant failed")

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
            "judge_requests_sha256": sha256_file(requests_path),
            "judges_sha256": sha256_file(judges_path),
            "evaluations_sha256": sha256_file(final_paths["evaluations"]),
            "summary_sha256": sha256_file(final_paths["summary"]),
        },
    )
    manifest["run_plan"] = run_plan
    manifest["run_plan_sha256"] = run_plan_sha256
    write_json_atomic(final_paths["manifest"], manifest, force=args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
