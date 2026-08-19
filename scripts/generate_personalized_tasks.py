#!/usr/bin/env python3
"""Generate audited personalized tasks with an OpenAI-compatible LLM API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shopping_grpo.personalization.generation import (
    ARCHITECT_PROMPT_VERSION,
    ARCHITECT_SYSTEM_PROMPT,
    CRITIC_PROMPT_VERSION,
    CRITIC_SYSTEM_PROMPT,
    GenerationAPIError,
    GenerationTransportError,
    OpenAICompatibleJSONClient,
    architect_user_prompt,
    build_architect_task,
    critic_user_prompt,
    question_count_for_index,
    scenario_for_run,
    validate_critic_response,
)
from shopping_grpo.personalization.schema import TaskValidationError, finalize_task, stable_hash
from shopping_grpo.personalization.source import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-accepted", type=int, default=20)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-critic", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_resume_state(output_dir: Path) -> tuple[set[int], list[dict]]:
    attempts_path = output_dir / "attempts.jsonl"
    accepted_path = output_dir / "accepted_tasks.jsonl"
    attempts = _read_jsonl(attempts_path) if attempts_path.exists() else []
    accepted = _read_jsonl(accepted_path) if accepted_path.exists() else []
    processed = {int(row["shopsim_task_id"]) for row in attempts}
    return processed, accepted


def _initial_manifest(args: argparse.Namespace, sources: list[dict]) -> dict:
    return {
        "schema_version": "personalized-generation-run-v1",
        "source_tasks": _portable_path(args.source_tasks),
        "source_rows_hash": stable_hash(sources),
        "target_accepted": args.target_accepted,
        "model": args.model,
        "api_base": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "critic_enabled": not args.skip_critic,
        "architect_prompt_version": ARCHITECT_PROMPT_VERSION,
        "architect_prompt_hash": stable_hash(ARCHITECT_SYSTEM_PROMPT),
        "critic_prompt_version": CRITIC_PROMPT_VERSION,
        "critic_prompt_hash": stable_hash(CRITIC_SYSTEM_PROMPT),
    }


def _prepare_run(args: argparse.Namespace, sources: list[dict]) -> tuple[set[int], list[dict]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_config.json"
    expected = _initial_manifest(args, sources)
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("--resume requires an existing run_config.json")
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != expected:
            raise SystemExit("resume configuration differs from run_config.json")
        return _load_resume_state(args.output_dir)
    occupied = [path for path in args.output_dir.iterdir() if path.is_file()]
    if occupied:
        raise SystemExit("output directory already contains files; use --resume or a new directory")
    manifest_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return set(), []


def main() -> int:
    args = parse_args()
    if args.target_accepted < 1:
        raise SystemExit("--target-accepted must be positive")
    if not args.model or not args.base_url or not args.api_key:
        raise SystemExit("set OPENAI_MODEL, OPENAI_BASE_URL and OPENAI_API_KEY")
    sources = _read_jsonl(args.source_tasks)
    processed, accepted = _prepare_run(args, sources)
    client = OpenAICompatibleJSONClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    for source in sources:
        source_task_id = int(source["shopsim_task_id"])
        if source_task_id in processed or len(accepted) >= args.target_accepted:
            continue
        # Rotate by attempted sources, not accepted rows. A schema/model problem
        # in one scenario must not spend the entire source pool on that scenario.
        attempt_index = len(processed)
        scenario = scenario_for_run(
            attempt_index,
            [row["scenario"] for row in accepted],
            args.target_accepted,
        )
        question_count = question_count_for_index(attempt_index, scenario)
        attempt = {
            "shopsim_task_id": source_task_id,
            "scenario": scenario,
            "status": "running",
            "reasons": [],
        }
        try:
            generated, raw_architect = client.complete_json(
                system=ARCHITECT_SYSTEM_PROMPT,
                user=architect_user_prompt(source, scenario, question_count),
            )
            _append_jsonl(
                args.output_dir / "raw_architect.jsonl",
                {
                    "shopsim_task_id": source_task_id,
                    "scenario": scenario,
                    "response": raw_architect,
                },
            )
            task = build_architect_task(
                generated,
                source=source,
                scenario=scenario,
                sequence=len(accepted) + 1,
                model=args.model,
            )
            if scenario == "clarification_required" and len(
                task["clarification"]["targets"]
            ) != question_count:
                raise ValueError(
                    f"Architect returned {len(task['clarification']['targets'])} questions; "
                    f"expected {question_count}"
                )
            if args.skip_critic:
                critic = {"verdict": "accept", "issues": []}
            else:
                generated_critic, raw_critic = client.complete_json(
                    system=CRITIC_SYSTEM_PROMPT,
                    user=critic_user_prompt(source, task),
                )
                _append_jsonl(
                    args.output_dir / "raw_critic.jsonl",
                    {
                        "shopsim_task_id": source_task_id,
                        "scenario": scenario,
                        "response": raw_critic,
                    },
                )
                critic = validate_critic_response(generated_critic)
            if critic["verdict"] != "accept":
                attempt["status"] = "critic_rejected"
                attempt["reasons"] = critic["issues"]
            else:
                task["generation"]["critic_model"] = None if args.skip_critic else args.model
                task["generation"]["critic_prompt_version"] = CRITIC_PROMPT_VERSION
                task["audit"]["critic_verdict"] = "accept"
                task = finalize_task(task)
                accepted.append(task)
                _append_jsonl(args.output_dir / "accepted_tasks.jsonl", task)
                attempt["status"] = "accepted"
                attempt["task_id"] = task["task_id"]
        except GenerationTransportError:
            raise
        except (GenerationAPIError, TaskValidationError, ValueError) as exc:
            attempt["status"] = "rejected"
            attempt["reasons"] = getattr(exc, "errors", [str(exc)])
        _append_jsonl(args.output_dir / "attempts.jsonl", attempt)
        processed.add(source_task_id)
        print(
            f"[{attempt['status']}] source={source_task_id} scenario={scenario} "
            f"accepted={len(accepted)}/{args.target_accepted} calls={client.call_count}"
        )

    summary = {
        "accepted": len(accepted),
        "target_accepted": args.target_accepted,
        "processed": len(_load_resume_state(args.output_dir)[0]),
        "api_calls_this_process": client.call_count,
        "complete": len(accepted) >= args.target_accepted,
        "accepted_task_ids": [row["task_id"] for row in accepted],
        "accepted_hash": stable_hash(accepted),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not summary["complete"]:
        print("source pool exhausted before target acceptance; export more source tasks and start a new run")
        return 2
    print(f"generated {len(accepted)} accepted tasks -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
