#!/usr/bin/env python3
"""Freeze one shared Qwen-curated Rubric bundle per evaluation task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments/ShopSimulator/shop_env"
sys.path.append(str(SHOP_ENV))

from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    index_jsonl,
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.contracts import ContractValidationError
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.evaluation.model_client import OpenAIJSONClient
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (
    RUBRIC_EXTRACTOR_VERSION,
    extract_rubric_candidates,
    materialize_rubric_bundle,
)
from shopping_grpo.evaluation.task_facts import task_facts_from_products
from shopping_grpo.multiturn.benchmark import load_products


RUBRIC_FREEZE_VERSION = "shopping-multiturn-rubric-freeze-v1"


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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _task_ids(path: Path) -> list[int]:
    task_ids = [int(row["task_id"]) for row in iter_jsonl(path)]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path} contains duplicate task IDs")
    return task_ids


def _curate(client, facts, candidates, schema_retries):
    messages = build_rubric_curator_messages(
        task_id=facts["task_id"],
        query=facts["query"],
        candidates=candidates["candidates"],
    )
    last_error = None
    for attempt in range(schema_retries + 1):
        response = client.complete_json(messages)
        try:
            bundle = materialize_rubric_bundle(
                task_facts=facts,
                candidates=candidates,
                curator_response=response["result"],
                curator_model=client.model,
                curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                rubric_version=RUBRIC_FREEZE_VERSION,
            )
            return response, bundle
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
                            f"{exc}。只修复 JSON，仍只能引用输入 candidate_id。"
                        ),
                    },
                ]
            )
    raise ContractValidationError(
        f"task {facts['task_id']} curator schema retries exhausted: {last_error}"
    )


def main():
    args = parse_args()
    if args.max_tokens < 1 or args.retries < 0 or args.schema_retries < 0:
        raise SystemExit("token and retry limits are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calls_path = args.output_dir / "curator_calls.jsonl"
    final_paths = [
        args.output_dir / "task_facts.jsonl",
        args.output_dir / "rubric_candidates.jsonl",
        args.output_dir / "rubrics.jsonl",
        args.output_dir / "manifest.json",
    ]
    if not args.resume and (
        calls_path.exists() or any(path.exists() for path in final_paths)
    ):
        raise SystemExit(
            f"output already exists under {args.output_dir}; pass --resume"
        )

    task_ids = _task_ids(args.tasks)
    facts_rows = task_facts_from_products(
        task_ids=task_ids,
        products=load_products(args.products),
    )
    candidate_rows = [extract_rubric_candidates(row) for row in facts_rows]
    facts_by_id = {row["task_id"]: row for row in facts_rows}
    candidates_by_id = {row["task_id"]: row for row in candidate_rows}
    cached = (
        index_jsonl(calls_path, key="task_id", allowed_keys=set(task_ids))
        if calls_path.exists()
        else {}
    )
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
        candidates = candidates_by_id[task_id]
        if task_id in cached:
            cached_row = cached[task_id]
            if cached_row.get("task_data_hash") != facts["task_data_hash"]:
                raise SystemExit(f"cached task hash mismatch for {task_id}")
            bundle = materialize_rubric_bundle(
                task_facts=facts,
                candidates=candidates,
                curator_response=cached_row["curator_response"],
                curator_model=args.model,
                curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                rubric_version=RUBRIC_FREEZE_VERSION,
            )
        else:
            response, bundle = _curate(
                client, facts, candidates, args.schema_retries
            )
            append_jsonl_fsync(
                calls_path,
                {
                    "task_id": task_id,
                    "task_data_hash": facts["task_data_hash"],
                    "query_hash": facts["query_hash"],
                    "curator_response": response["result"],
                    "request_metadata": response["metadata"],
                },
            )
        bundles.append(bundle)
        print(f"rubric {index}/{len(task_ids)} task={task_id}")

    write_jsonl_atomic(final_paths[0], facts_rows, force=args.resume)
    write_jsonl_atomic(final_paths[1], candidate_rows, force=args.resume)
    write_jsonl_atomic(final_paths[2], bundles, force=args.resume)
    manifest = {
        "schema_version": RUBRIC_FREEZE_VERSION,
        "task_count": len(task_ids),
        "task_manifest": str(args.tasks),
        "task_manifest_sha256": sha256_file(args.tasks),
        "product_data_sha256": sha256_file(args.products),
        "extractor_version": RUBRIC_EXTRACTOR_VERSION,
        "curator_model": args.model,
        "curator_prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
        "thinking": False,
        "temperature": 0.0,
        "artifacts": {
            path.name: sha256_file(path)
            for path in [calls_path, *final_paths[:3]]
        },
    }
    write_json_atomic(final_paths[3], manifest, force=args.resume)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
