#!/usr/bin/env python3
"""Freeze a deterministic, result-blind subset of the multi-turn final set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "tasks.jsonl",
    "gap_openings.jsonl",
    "complete_openings.jsonl",
    "conditions.jsonl",
    "reward_audit.jsonl",
)
CONDITIONS = {
    "gap-ask-enabled",
    "gap-ask-disabled",
    "complete-ask-enabled",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_text_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256_bytes(payload)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_jsonl(rows: list[dict]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def select_task_ids(task_ids: list[int], *, size: int, seed: str) -> set[int]:
    if size < 1 or size > len(task_ids):
        raise ValueError(f"subset size must be in [1, {len(task_ids)}]")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("source tasks contain duplicate task IDs")
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{seed}:{task_id}".encode("utf-8")
        ).digest(),
    )
    return set(ranked[:size])


def build_subset(source: Path, *, size: int, seed: str) -> tuple[dict[str, bytes], dict]:
    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"source final assets are incomplete: {missing}")

    source_rows = {name: read_jsonl(source / name) for name in SOURCE_FILES}
    task_ids = [int(row["task_id"]) for row in source_rows["tasks.jsonl"]]
    selected = select_task_ids(task_ids, size=size, seed=seed)
    output_rows = {
        name: [row for row in rows if int(row["task_id"]) in selected]
        for name, rows in source_rows.items()
    }

    selected_order = [int(row["task_id"]) for row in output_rows["tasks.jsonl"]]
    selected_set = set(selected_order)
    for name in ("gap_openings.jsonl", "complete_openings.jsonl", "reward_audit.jsonl"):
        ids = [int(row["task_id"]) for row in output_rows[name]]
        if len(ids) != size or set(ids) != selected_set or len(ids) != len(set(ids)):
            raise ValueError(f"source asset task mismatch: {name}")

    condition_rows = output_rows["conditions.jsonl"]
    if len(condition_rows) != size * len(CONDITIONS):
        raise ValueError("source conditions do not contain three rows per selected task")
    by_task: dict[int, set[str]] = {task_id: set() for task_id in selected_set}
    for row in condition_rows:
        by_task[int(row["task_id"])].add(str(row["condition"]))
    if any(values != CONDITIONS for values in by_task.values()):
        raise ValueError("selected tasks do not have the frozen G+/G-/C+ conditions")

    files = {name: canonical_jsonl(rows) for name, rows in output_rows.items()}
    manifest = {
        "schema_version": "shopping-multiturn-final-subset-v1",
        "evaluation_role": "final",
        "final_evaluation_used": True,
        "reward_contract": "shopsimulator-reward-v4",
        "task_count": size,
        "condition_count": size * len(CONDITIONS),
        "conditions": sorted(CONDITIONS),
        "selection": {
            "policy": "sha256-rank-without-replacement-v1",
            "seed": seed,
            "source_task_count": len(task_ids),
            "result_blind": True,
            "preserve_source_order": True,
        },
        "source": {
            "directory": display_path(source),
            "hash_normalization": "CRLF/CR-to-LF-v1",
            "files_sha256": {
                name: normalized_text_sha256(source / name) for name in SOURCE_FILES
            },
        },
        "subset_sha256": {
            name.removesuffix(".jsonl"): sha256_bytes(payload)
            for name, payload in files.items()
        },
        "selected_task_ids_sha256": sha256_bytes(
            canonical_jsonl([{"task_id": task_id} for task_id in selected_order])
        ),
    }
    return files, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/multiturn/evaluation-v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/multiturn/final-200-v1",
    )
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", default="shopping-final-200-v1")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rewrite an existing subset only when it has the expected final schema.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    files, manifest = build_subset(source, size=args.size, seed=args.seed)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    expected = {**files, "manifest.json": manifest_bytes}

    if args.check:
        missing = [name for name in expected if not (output / name).is_file()]
        mismatched = [
            name
            for name, payload in expected.items()
            if (output / name).is_file()
            and (output / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            != payload
        ]
        if missing or mismatched:
            raise SystemExit(
                f"FINAL SUBSET CHECK FAILED missing={missing} mismatched={mismatched}"
            )
        print(f"FINAL SUBSET CHECK PASSED: {output} ({args.size} tasks)")
        return

    if output.exists() and any(output.iterdir()):
        existing_manifest = output / "manifest.json"
        existing = (
            json.loads(existing_manifest.read_text(encoding="utf-8"))
            if existing_manifest.is_file()
            else {}
        )
        if not args.refresh or existing.get("schema_version") != manifest["schema_version"]:
            raise SystemExit(f"output must be new or an existing final subset passed with --refresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        (output / name).write_bytes(payload)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"FROZEN FINAL SUBSET CREATED: {output}")


if __name__ == "__main__":
    main()
