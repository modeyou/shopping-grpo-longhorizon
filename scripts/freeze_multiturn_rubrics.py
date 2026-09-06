#!/usr/bin/env python3
"""Freeze one shared Query-only Rubric bundle per evaluation task."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
sys.path.append(str(SHOP_ENV))

from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    index_jsonl,
    iter_jsonl,
    load_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.blind_guard import guard_declared_final_tasks
from shopping_grpo.evaluation.contracts import ContractValidationError
from shopping_grpo.evaluation.manifest import canonical_json_sha256, sha256_file
from shopping_grpo.evaluation.model_client import OpenAIJSONClient
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (
    materialize_rubric_bundle,
    RUBRIC_CURATOR_VERSION,
    QUERY_EVIDENCE_ANCHOR_VERSION,
)
from shopping_grpo.evaluation.task_facts import task_facts_from_products
from shopping_grpo.multiturn.benchmark import load_products


RUBRIC_FREEZE_VERSION = "shopping-multiturn-rubric-freeze-v6"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--products",
        type=Path,
        default=SHOP_ENV / "data/fine_items_eval_train_all.json.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="local-qwen")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
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


def _curate(
    client,
    facts,
    schema_retries,
    *,
    run_plan_sha256: str,
    on_request,
    on_attempt,
):
    messages = build_rubric_curator_messages(
        task_id=facts["task_id"],
        query=facts["query"],
    )
    request_ids = []
    last_error = None
    for attempt in range(schema_retries + 1):
        request = {
            "schema_version": "shopping-rubric-curator-request-v1",
            "run_plan_sha256": run_plan_sha256,
            "task_id": int(facts["task_id"]),
            "schema_attempt": attempt,
            "messages": deepcopy(messages),
        }
        request["request_sha256"] = canonical_json_sha256(request)
        request["request_id"] = request["request_sha256"]
        on_request(request)
        request_ids.append(request["request_id"])
        response = client.complete_json(messages)
        try:
            bundle = materialize_rubric_bundle(
                task_facts=facts,
                curator_response=response["result"],
                curator_model=client.model,
                curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                rubric_version=RUBRIC_FREEZE_VERSION,
            )
            on_attempt(
                {
                    "task_id": int(facts["task_id"]),
                    "request_id": request["request_id"],
                    "schema_attempt": attempt,
                    "validation_status": "valid",
                    "validation_error": None,
                    "curator_response": response["result"],
                    "request_metadata": response["metadata"],
                }
            )
            return response, bundle, request_ids
        except ContractValidationError as exc:
            last_error = exc
            on_attempt(
                {
                    "task_id": int(facts["task_id"]),
                    "request_id": request["request_id"],
                    "schema_attempt": attempt,
                    "validation_status": "invalid",
                    "validation_error": str(exc),
                    "curator_response": response["result"],
                    "request_metadata": response["metadata"],
                }
            )
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
                            f"{exc}。只修复 JSON，且只基于输入 Query。"
                        ),
                    },
                ]
            )
    raise ContractValidationError(
        f"task {facts['task_id']} curator schema retries exhausted: {last_error}"
    )


def _run_plan(args) -> dict:
    """Describe every non-secret input that may change a frozen Rubric."""

    return {
        "schema_version": RUBRIC_FREEZE_VERSION,
        "task_manifest_sha256": sha256_file(args.tasks),
        "product_data_sha256": sha256_file(args.products),
        "curator_version": RUBRIC_CURATOR_VERSION,
        "query_evidence_anchor_version": QUERY_EVIDENCE_ANCHOR_VERSION,
        "curator": {
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "retries": args.retries,
            "schema_retries": args.schema_retries,
            "prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
            "thinking": False,
            "temperature": 0.0,
        },
    }


def _require_exact_task_coverage(name: str, rows: list[dict], task_ids: list[int]) -> None:
    actual = [int(row["task_id"]) for row in rows]
    if actual != task_ids:
        raise SystemExit(
            f"{name} task coverage mismatch: expected={len(task_ids)} "
            f"actual={len(actual)}"
        )


def _validate_cached_request(cached: dict, request: dict) -> None:
    fields = (
        "run_plan_sha256",
        "task_id",
        "schema_attempt",
        "request_sha256",
    )
    if any(cached.get(field) != request[field] for field in fields):
        raise SystemExit(
            f"cached curator request does not match request {request['request_id']}"
        )


def main():
    args = parse_args()
    if args.max_tokens < 1 or args.retries < 0 or args.schema_retries < 0:
        raise SystemExit("token and retry limits are invalid")
    guard_declared_final_tasks(
        args.tasks,
        allowed=args.allow_blind_final,
    )
    run_plan = _run_plan(args)
    run_plan_sha256 = canonical_json_sha256(run_plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calls_path = args.output_dir / "curator_calls.jsonl"
    requests_path = args.output_dir / "curator_requests.jsonl"
    attempts_path = args.output_dir / "curator_attempts.jsonl"
    final_paths = [
        args.output_dir / "task_facts.jsonl",
        args.output_dir / "rubrics.jsonl",
        args.output_dir / "manifest.json",
    ]
    obsolete_paths = [args.output_dir / "rubric_candidates.jsonl"]
    if any(path.exists() for path in obsolete_paths):
        raise SystemExit(
            "output contains obsolete candidate-based Rubric artifacts; use a "
            "new output directory"
        )
    if not args.resume and (
        calls_path.exists()
        or requests_path.exists()
        or attempts_path.exists()
        or any(path.exists() for path in final_paths)
    ):
        raise SystemExit(
            f"output already exists under {args.output_dir}; pass --resume"
        )
    if args.resume and final_paths[2].exists():
        previous_manifest = load_json(final_paths[2])
        if previous_manifest.get("run_plan_sha256") != run_plan_sha256:
            raise SystemExit(
                "Rubric resume plan mismatch; use a new output directory for "
                "a different model, prompt, input, or retry configuration"
            )

    task_ids = _task_ids(args.tasks)
    facts_rows = task_facts_from_products(
        task_ids=task_ids,
        products=load_products(args.products),
    )
    facts_by_id = {row["task_id"]: row for row in facts_rows}
    cached = (
        index_jsonl(calls_path, key="task_id", allowed_keys=set(task_ids))
        if calls_path.exists()
        else {}
    )
    cached_requests = (
        index_jsonl(requests_path, key="request_id")
        if requests_path.exists()
        else {}
    )

    def record_request(request: dict) -> None:
        cached_request = cached_requests.get(request["request_id"])
        if cached_request is not None:
            _validate_cached_request(cached_request, request)
            return
        append_jsonl_fsync(requests_path, request)
        cached_requests[request["request_id"]] = request

    def record_attempt(attempt: dict) -> None:
        append_jsonl_fsync(attempts_path, attempt)

    client = OpenAIJSONClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        response_format_json=True,
        thinking=False,
    )
    bundles = []
    for index, task_id in enumerate(task_ids, start=1):
        facts = facts_by_id[task_id]
        if task_id in cached:
            cached_row = cached[task_id]
            if cached_row.get("run_plan_sha256") != run_plan_sha256:
                raise SystemExit(f"cached Rubric run plan mismatch for {task_id}")
            if cached_row.get("task_data_hash") != facts["task_data_hash"]:
                raise SystemExit(f"cached task hash mismatch for {task_id}")
            request_ids = cached_row.get("request_ids")
            if not isinstance(request_ids, list) or not request_ids:
                raise SystemExit(
                    f"cached curator response lacks auditable request IDs for {task_id}"
                )
            if any(
                not isinstance(request_id, str)
                or request_id not in cached_requests
                for request_id in request_ids
            ):
                raise SystemExit(
                    f"cached curator response has missing request artifacts for {task_id}"
                )
            bundle = materialize_rubric_bundle(
                task_facts=facts,
                curator_response=cached_row["curator_response"],
                curator_model=args.model,
                curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                rubric_version=RUBRIC_FREEZE_VERSION,
            )
        else:
            response, bundle, request_ids = _curate(
                client,
                facts,
                args.schema_retries,
                run_plan_sha256=run_plan_sha256,
                on_request=record_request,
                on_attempt=record_attempt,
            )
            append_jsonl_fsync(
                calls_path,
                {
                    "task_id": task_id,
                    "run_plan_sha256": run_plan_sha256,
                    "task_data_hash": facts["task_data_hash"],
                    "query_hash": facts["query_hash"],
                    "curator_response": response["result"],
                    "request_metadata": response["metadata"],
                    "request_ids": request_ids,
                },
            )
        bundles.append(bundle)
        print(f"rubric {index}/{len(task_ids)} task={task_id}")

    _require_exact_task_coverage("task facts", facts_rows, task_ids)
    _require_exact_task_coverage("Rubric bundles", bundles, task_ids)
    write_jsonl_atomic(final_paths[0], facts_rows, force=args.resume)
    write_jsonl_atomic(final_paths[1], bundles, force=args.resume)
    manifest = {
        "schema_version": RUBRIC_FREEZE_VERSION,
        "task_count": len(task_ids),
        "task_manifest": str(args.tasks),
        "task_manifest_sha256": sha256_file(args.tasks),
        "product_data_sha256": sha256_file(args.products),
        "curator_version": RUBRIC_CURATOR_VERSION,
        "query_evidence_anchor_version": QUERY_EVIDENCE_ANCHOR_VERSION,
        "curator_model": args.model,
        "curator_prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
        "thinking": False,
        "temperature": 0.0,
        "run_plan": run_plan,
        "run_plan_sha256": run_plan_sha256,
        "artifacts": {
            path.name: sha256_file(path)
            for path in [
                calls_path,
                requests_path,
                attempts_path,
                *final_paths[:2],
            ]
        },
    }
    write_json_atomic(final_paths[2], manifest, force=args.resume)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
