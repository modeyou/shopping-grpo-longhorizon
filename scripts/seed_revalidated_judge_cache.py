#!/usr/bin/env python3
"""Seed a new prompt cache only with prior Judge outputs valid under new rules."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from shopping_grpo.evaluation.artifacts import index_jsonl, iter_jsonl
from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    rubric_ids,
    validate_judge_deterministic_consistency,
    validate_judge_result,
)
from shopping_grpo.evaluation.judge_cache import (
    SemanticJudgeCache,
    semantic_request_spec,
)
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.prompts import (
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_trajectory_judge_messages,
    deterministic_judge_facts,
    semantic_judge_trajectory_id,
)
from shopping_grpo.evaluation.trajectory import normalize_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        action="append",
        required=True,
        help="TRAJECTORIES=RUN_DIR; RUN_DIR must contain judges.jsonl",
    )
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def _source_spec(value: str) -> tuple[Path, Path]:
    trajectory_path, separator, run_dir = str(value).partition("=")
    if not separator or not trajectory_path.strip() or not run_dir.strip():
        raise ValueError("--source-run must use TRAJECTORIES=RUN_DIR")
    return Path(trajectory_path), Path(run_dir)


def main():
    args = parse_args()
    rubrics = index_jsonl(args.rubrics, key="task_id")
    cache = SemanticJudgeCache(args.output_cache)
    seen_requests = set()
    seeded = 0
    reused = 0
    rejected = []
    for trajectory_path, run_dir in map(_source_spec, args.source_run):
        judges = index_jsonl(run_dir / "judges.jsonl", key="task_id")
        for raw in iter_jsonl(trajectory_path):
            task_id = int(raw["task_id"])
            rubric = rubrics[task_id]
            prior = judges[task_id]
            normalized = normalize_trajectory(raw)
            metrics = compute_deterministic_metrics(normalized)
            messages = build_trajectory_judge_messages(
                normalized=normalized,
                rubric_bundle=rubric,
                deterministic_metrics=metrics,
            )
            semantic_id = semantic_judge_trajectory_id(messages)
            result = deepcopy(prior["judge_result"])
            if result.get("judge_status") != "valid":
                rejected.append(
                    {"task_id": task_id, "reason": "prior Judge result is not valid"}
                )
                continue
            result["trajectory_id"] = semantic_id
            allowed_events = [
                event["event_id"]
                for event in normalized.get("events") or []
                if event.get("event_id")
            ]
            facts = deterministic_judge_facts(
                normalized=normalized,
                rubric_bundle=rubric,
            )
            try:
                result = validate_judge_result(
                    result,
                    rubric_ids=rubric_ids(rubric),
                    expected_task_id=task_id,
                    expected_trajectory_id=semantic_id,
                    allowed_event_ids=allowed_events,
                )
                validate_judge_deterministic_consistency(result, facts)
            except ContractValidationError as exc:
                rejected.append({"task_id": task_id, "reason": str(exc)})
                continue
            request_spec = semantic_request_spec(
                model=args.model,
                base_url=args.base_url,
                prompt_version=TRAJECTORY_JUDGE_PROMPT_VERSION,
                max_tokens=args.max_tokens,
                thinking=False,
                messages=messages,
            )
            metadata = deepcopy(prior.get("request_metadata") or {})
            metadata["revalidated_cache_seed"] = True
            response = {"judge_result": result, "request_metadata": metadata}
            _, cache_hit = cache.get_or_compute(
                request_spec, lambda response=response: response
            )
            request_key = semantic_id
            if request_key in seen_requests or cache_hit:
                reused += 1
            else:
                seeded += 1
                seen_requests.add(request_key)
    print(
        f"revalidated_cache seeded={seeded} reused={reused} "
        f"rejected={len(rejected)} prompt={TRAJECTORY_JUDGE_PROMPT_VERSION}"
    )
    for item in rejected:
        print(f"rejected task={item['task_id']}: {item['reason']}")


if __name__ == "__main__":
    main()
