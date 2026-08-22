#!/usr/bin/env python3
"""Freeze a token-aware, task-disjoint multi-turn SFT smoke or formal mix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from shopping_grpo.collection.multiturn_sft_mix import (
    MIX_SCHEMA_VERSION,
    POLICY_ORDER,
    allocate_row_quotas,
    augment_complete_schemas,
    membership_patterns,
    select_disjoint_rows,
    split_selected,
)
from shopping_grpo.collection.multiturn_sft_v4 import read_jsonl, sha256_file
from shopping_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic
from shopping_grpo.training.sft.dataset import IGNORE_INDEX, build_supervised_example


POLICY_ARTIFACTS = {
    "complete-no-ask-v1": "complete_pool",
    "composite-replay-v1": "composite_pool",
    "autonomous-gap-v1": "autonomous_pool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=64)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--complete-token-ratio", type=float, default=0.5)
    parser.add_argument("--composite-token-ratio", type=float, default=0.3)
    parser.add_argument("--autonomous-token-ratio", type=float, default=0.2)
    parser.add_argument("--token-share-tolerance", type=float, default=0.05)
    return parser.parse_args()


def _load_preprocessing(model: str, revision: str | None):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    kwargs = {"trust_remote_code": True, "local_files_only": True}
    if revision:
        kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model, **kwargs)
    if str(getattr(config, "model_type", "")).startswith("qwen3_5"):
        processor = AutoProcessor.from_pretrained(model, **kwargs)
        return processor.tokenizer, processor
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
    return tokenizer, tokenizer


def _validate_public_row(row: dict, source: str) -> None:
    required = {
        "task_id",
        "trajectory_id",
        "messages",
        "tools",
        "teacher_policy",
        "source_goal_hash",
        "schema_variant",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"{source}: missing fields {sorted(missing)}")
    forbidden_keys = {
        "reasoning_content",
        "opening_audit",
        "clarification_audit",
        "reward_detail",
        "hidden_answer",
    }
    for index, message in enumerate(row["messages"]):
        leaked = forbidden_keys.intersection(message)
        if leaked:
            raise ValueError(
                f"{source}: message {index} contains private keys {sorted(leaked)}"
            )
        if (
            message.get("role") == "tool"
            and "reward:" in str(message.get("content") or "").casefold()
        ):
            raise ValueError(f"{source}: terminal Reward text was not sanitized")


def _tokenize_pool(
    *, rows: list[dict], tokenizer, chat_template, max_length: int, policy: str
) -> tuple[list[dict], dict]:
    annotated = []
    dropped = Counter()
    for index, row in enumerate(rows, start=1):
        _validate_public_row(row, f"{policy}:{row.get('task_id')}")
        example = build_supervised_example(
            messages=row["messages"],
            tools=row["tools"],
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=max_length,
        )
        if example is None:
            dropped["chat_template_or_length"] += 1
        else:
            assistant_tokens = sum(
                token != IGNORE_INDEX for token in example["labels"]
            )
            if assistant_tokens <= 0:
                dropped["no_assistant_loss_tokens"] += 1
            else:
                annotated.append(
                    {
                        "row": row,
                        "input_tokens": len(example["input_ids"]),
                        "assistant_tokens": assistant_tokens,
                    }
                )
        if index % 100 == 0 or index == len(rows):
            print(
                f"TOKENIZE policy={policy} progress={index}/{len(rows)} "
                f"kept={len(annotated)}"
            )
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(annotated),
        "dropped_rows": sum(dropped.values()),
        "drop_reasons": dict(sorted(dropped.items())),
        "input_tokens": {
            "min": min(item["input_tokens"] for item in annotated),
            "max": max(item["input_tokens"] for item in annotated),
            "mean": sum(item["input_tokens"] for item in annotated) / len(annotated),
        },
        "assistant_tokens": {
            "min": min(item["assistant_tokens"] for item in annotated),
            "max": max(item["assistant_tokens"] for item in annotated),
            "mean": sum(item["assistant_tokens"] for item in annotated) / len(annotated),
        },
    }
    return annotated, summary


def _selected_summary(selected: dict[str, list[dict]]) -> dict:
    policy_rows = {}
    total_assistant_tokens = sum(
        item["assistant_tokens"]
        for rows in selected.values()
        for item in rows
    )
    for policy, rows in selected.items():
        assistant_tokens = sum(item["assistant_tokens"] for item in rows)
        policy_rows[policy] = {
            "rows": len(rows),
            "input_tokens": sum(item["input_tokens"] for item in rows),
            "assistant_tokens": assistant_tokens,
            "assistant_token_share": assistant_tokens / total_assistant_tokens,
            "schema_variants": dict(
                sorted(Counter(item["row"]["schema_variant"] for item in rows).items())
            ),
        }
    return {
        "rows": sum(len(rows) for rows in selected.values()),
        "assistant_tokens": total_assistant_tokens,
        "policies": policy_rows,
    }


def main() -> int:
    args = parse_args()
    if args.total_rows < 6:
        raise SystemExit("--total-rows must be at least 6")
    if args.max_length < 1:
        raise SystemExit("--max-length must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")

    audit_manifest = json.loads(
        args.audit_manifest.read_text(encoding="utf-8")
    )
    if audit_manifest.get("audit_reward") != "shopsimulator-reward-v4":
        raise SystemExit("audit manifest is not a Reward v4 pool")

    pools = {}
    pool_inputs = {}
    for policy, artifact_name in POLICY_ARTIFACTS.items():
        artifact = audit_manifest["artifacts"][artifact_name]
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"pool hash mismatch: {path}")
        rows = read_jsonl(path)
        if policy == "complete-no-ask-v1":
            rows = augment_complete_schemas(rows, seed=args.seed)
        pools[policy] = rows
        pool_inputs[policy] = {
            "path": str(path.resolve()),
            "sha256": artifact["sha256"],
            "rows": len(rows),
        }

    tokenizer, chat_template = _load_preprocessing(args.model, args.revision)
    tokenized = {}
    tokenization = {}
    for policy in POLICY_ORDER:
        tokenized[policy], tokenization[policy] = _tokenize_pool(
            rows=pools[policy],
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=args.max_length,
            policy=policy,
        )
        if not tokenized[policy]:
            raise SystemExit(f"no tokenizable rows for {policy}")

    token_ratios = {
        "complete-no-ask-v1": args.complete_token_ratio,
        "composite-replay-v1": args.composite_token_ratio,
        "autonomous-gap-v1": args.autonomous_token_ratio,
    }
    averages = {
        policy: tokenization[policy]["assistant_tokens"]["mean"]
        for policy in POLICY_ORDER
    }
    quotas = allocate_row_quotas(
        total_rows=args.total_rows,
        token_ratios=token_ratios,
        average_assistant_tokens=averages,
    )
    selected = select_disjoint_rows(
        pools=tokenized,
        quotas=quotas,
        seed=args.seed,
    )
    train, validation = split_selected(
        selected,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    selected_rows = [item["row"] for rows in selected.values() for item in rows]
    train_rows = [item["row"] for item in train]
    validation_rows = [item["row"] for item in validation]

    task_values = [int(row["task_id"]) for row in selected_rows]
    goal_hashes = [str(row["source_goal_hash"]) for row in selected_rows]
    if len(task_values) != len(set(task_values)):
        raise AssertionError("selected mix contains duplicate task IDs")
    if len(goal_hashes) != len(set(goal_hashes)):
        raise AssertionError("selected mix contains duplicate source-goal hashes")

    output.mkdir(parents=True, exist_ok=True)
    all_path = output / "all.jsonl"
    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    audit_path = output / "selection-audit.jsonl"
    write_jsonl_atomic(all_path, selected_rows)
    write_jsonl_atomic(train_path, train_rows)
    write_jsonl_atomic(validation_path, validation_rows)
    write_jsonl_atomic(
        audit_path,
        [
            {
                "task_id": item["row"]["task_id"],
                "trajectory_id": item["row"].get("trajectory_id"),
                "teacher_policy": item["row"]["teacher_policy"],
                "schema_variant": item["row"]["schema_variant"],
                "source_goal_hash": item["row"]["source_goal_hash"],
                "input_tokens": item["input_tokens"],
                "assistant_tokens": item["assistant_tokens"],
                "split": "validation" if item in validation else "train",
            }
            for rows in selected.values()
            for item in rows
        ],
    )

    summary = _selected_summary(selected)
    if not 0 <= args.token_share_tolerance < 1:
        raise SystemExit("--token-share-tolerance must be in [0, 1)")
    for policy in POLICY_ORDER:
        actual_share = summary["policies"][policy]["assistant_token_share"]
        if abs(actual_share - token_ratios[policy]) > args.token_share_tolerance:
            raise SystemExit(
                f"selected {policy} assistant-token share {actual_share:.4f} "
                f"exceeds tolerance from target {token_ratios[policy]:.4f}"
            )
    manifest = {
        "schema_version": MIX_SCHEMA_VERSION,
        "source_pool_schema": audit_manifest.get("schema_version"),
        "reward": "shopsimulator-reward-v4",
        "model": args.model,
        "model_revision": args.revision,
        "max_length": args.max_length,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "target_assistant_token_ratios": token_ratios,
        "token_share_tolerance": args.token_share_tolerance,
        "row_quotas_from_token_averages": quotas,
        "pool_inputs": pool_inputs,
        "pool_membership_patterns": membership_patterns(pools),
        "tokenization": tokenization,
        "selected": summary,
        "split": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "task_disjoint": not (
                {int(row["task_id"]) for row in train_rows}
                & {int(row["task_id"]) for row in validation_rows}
            ),
        },
        "artifacts": {
            path.name: {
                "rows": len(read_jsonl(path)),
                "sha256": sha256_file(path),
            }
            for path in (all_path, train_path, validation_path, audit_path)
        },
    }
    manifest_path = output / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "row_quotas": quotas,
                "selected": summary,
                "split": manifest["split"],
                "membership_patterns": manifest["pool_membership_patterns"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
