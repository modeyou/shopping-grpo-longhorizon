#!/usr/bin/env python3
"""Seed and merge resumable deterministic evaluation shards."""

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.rollout import load_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("seed", "merge"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--tasks", type=Path, required=True)
        subparser.add_argument("--shard-root", type=Path, required=True)
        subparser.add_argument("--shard-count", type=int, required=True)
        subparser.add_argument("--limit", type=int)
        if command == "seed":
            subparser.add_argument("--input", type=Path, required=True)
        else:
            subparser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _key(row):
    if "task_id" not in row:
        raise ValueError("trajectory is missing task_id")
    return int(row["task_id"]), int(row.get("attempt_index", 0))


def _task_order(tasks_path, limit):
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    tasks = load_tasks(tasks_path)
    if limit is not None:
        tasks = tasks[:limit]
    task_ids = [int(task["task_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("tasks contains duplicate task IDs")
    return {task_id: index for index, task_id in enumerate(task_ids)}


def _validate_shard_count(shard_count):
    if shard_count < 1:
        raise ValueError("shard-count must be a positive integer")


def _add_rows(by_key, rows, order, shard_count, shard_index=None):
    for row in rows:
        key = _key(row)
        task_id = key[0]
        if task_id not in order:
            raise ValueError(f"trajectory task {task_id} is outside frozen tasks")
        expected_shard = order[task_id] % shard_count
        if shard_index is not None and expected_shard != shard_index:
            raise ValueError(
                f"trajectory task {task_id} is in shard {shard_index}, "
                f"expected shard {expected_shard}"
            )
        previous = by_key.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"conflicting trajectory for task/attempt {key}")
        by_key[key] = row


def _sorted_rows(by_key, order):
    return [
        row
        for _, row in sorted(
            by_key.items(),
            key=lambda item: (order[item[0][0]], item[0][1]),
        )
    ]


def seed(args):
    _validate_shard_count(args.shard_count)
    order = _task_order(args.tasks, args.limit)
    combined = _read_jsonl(args.input)
    totals = {}
    for shard_index in range(args.shard_count):
        shard_path = args.shard_root / str(shard_index) / "trajectories.jsonl"
        by_key = {}
        _add_rows(
            by_key,
            _read_jsonl(shard_path),
            order,
            args.shard_count,
            shard_index,
        )
        shard_rows = [
            row
            for row in combined
            if order.get(_key(row)[0], -1) % args.shard_count == shard_index
        ]
        _add_rows(by_key, shard_rows, order, args.shard_count, shard_index)
        rows = _sorted_rows(by_key, order)
        _write_jsonl(shard_path, rows)
        totals[str(shard_index)] = len(rows)
    print(json.dumps({"seeded": totals}, ensure_ascii=False))


def merge(args):
    _validate_shard_count(args.shard_count)
    order = _task_order(args.tasks, args.limit)
    by_key = {}
    for shard_index in range(args.shard_count):
        shard_path = args.shard_root / str(shard_index) / "trajectories.jsonl"
        _add_rows(
            by_key,
            _read_jsonl(shard_path),
            order,
            args.shard_count,
            shard_index,
        )
    rows = _sorted_rows(by_key, order)
    _write_jsonl(args.output, rows)
    print(json.dumps({"merged": len(rows), "output": str(args.output)}))


def main():
    args = parse_args()
    if args.command == "seed":
        seed(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
