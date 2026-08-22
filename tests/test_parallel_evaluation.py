import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.evaluate_shop_benchmark import main as benchmark_main
from scripts.manage_evaluation_shards import merge, seed


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_seed_preserves_serial_progress_and_merge_restores_source_order(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    combined = tmp_path / "trajectories.jsonl"
    shard_root = tmp_path / "shards"
    merged = tmp_path / "merged.jsonl"
    _write_jsonl(tasks, [{"task_id": task_id} for task_id in range(8)])
    _write_jsonl(
        combined,
        [{"task_id": task_id, "attempt_index": 0} for task_id in (0, 3, 4)],
    )

    seed(SimpleNamespace(
        tasks=tasks,
        input=combined,
        shard_root=shard_root,
        shard_count=4,
        limit=None,
    ))

    assert [row["task_id"] for row in _read_jsonl(
        shard_root / "0/trajectories.jsonl"
    )] == [0, 4]
    assert [row["task_id"] for row in _read_jsonl(
        shard_root / "3/trajectories.jsonl"
    )] == [3]

    for shard_index in range(4):
        shard_path = shard_root / str(shard_index) / "trajectories.jsonl"
        rows = _read_jsonl(shard_path)
        present = {row["task_id"] for row in rows}
        rows.extend(
            {"task_id": task_id, "attempt_index": 0}
            for task_id in range(shard_index, 8, 4)
            if task_id not in present
        )
        _write_jsonl(shard_path, rows)

    merge(SimpleNamespace(
        tasks=tasks,
        shard_root=shard_root,
        shard_count=4,
        limit=None,
        output=merged,
    ))

    assert [row["task_id"] for row in _read_jsonl(merged)] == list(range(8))


def test_seed_rejects_conflicting_existing_trajectory(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    combined = tmp_path / "trajectories.jsonl"
    shard_root = tmp_path / "shards"
    _write_jsonl(tasks, [{"task_id": 10}])
    _write_jsonl(combined, [{"task_id": 10, "status": "done"}])
    _write_jsonl(
        shard_root / "0/trajectories.jsonl",
        [{"task_id": 10, "status": "error"}],
    )

    with pytest.raises(ValueError, match="conflicting trajectory"):
        seed(SimpleNamespace(
            tasks=tasks,
            input=combined,
            shard_root=shard_root,
            shard_count=1,
            limit=None,
        ))


def test_summary_only_uses_selected_shard_denominator(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    output = tmp_path / "trajectories.jsonl"
    summary = tmp_path / "summary.json"
    _write_jsonl(tasks, [{"task_id": task_id} for task_id in range(4)])
    _write_jsonl(output, [])

    with patch.object(
        sys,
        "argv",
        [
            "evaluate_shop_benchmark.py",
            "--benchmark", str(tasks),
            "--expected-tasks", str(tasks),
            "--output", str(output),
            "--summary", str(summary),
            "--model", "qwen3.5-2b",
            "--llm-base-url", "http://127.0.0.1:18002/v1",
            "--api-key", "EMPTY",
            "--shard-count", "2",
            "--shard-index", "1",
            "--summary-only",
        ],
    ):
        benchmark_main()

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["expected_tasks"] == 2
    assert payload["missing_tasks"] == [1, 3]
    assert payload["protocol"]["shard_count"] == 2
    assert payload["protocol"]["shard_index"] == 1
    assert payload["protocol"]["summary_only"] is True
    assert payload["protocol"]["execution_shards"] == 2


def test_parallel_launcher_runs_one_wave_per_condition():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluate_multiturn_parallel.sh"
    ).read_text(encoding="utf-8")
    assert "LLM_BASE_URLS" in launcher
    assert "manage_evaluation_shards.py\" seed" in launcher
    assert "manage_evaluation_shards.py\" merge" in launcher
    assert "--shard-count \"$SHARD_COUNT\"" in launcher
    assert "--shard-index \"$shard_index\"" in launcher
    assert "--summary-only" in launcher
