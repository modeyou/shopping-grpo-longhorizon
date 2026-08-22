import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.evaluate_shop_benchmark import main as benchmark_main


def test_limit_must_be_positive():
    with patch.object(
        sys,
        "argv",
        [
            "evaluate_shop_benchmark.py",
            "--benchmark",
            "data/evaluation/tasks.jsonl",
            "--output",
            "outputs/eval/base/raw.jsonl",
            "--summary",
            "outputs/eval/base/summary.json",
            "--model",
            "Qwen/Qwen3.5-2B",
            "--llm-base-url",
            "http://127.0.0.1:8000/v1",
            "--api-key",
            "EMPTY",
            "--limit",
            "0",
        ],
    ):
        with pytest.raises(
            SystemExit, match="--limit must be a positive integer"
        ):
            benchmark_main()


def test_multiturn_launcher_uses_dev_v2_frozen_openings():
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts/evaluate_multiturn.sh"
    ).read_text(encoding="utf-8")

    assert "data/multiturn/evaluation-dev-v2" in launcher
    assert "MULTITURN_GAP_OPENINGS" in launcher
    assert "MULTITURN_COMPLETE_OPENINGS" in launcher
    assert 'run_condition gap-ask-enabled "$GAP_OPENINGS"' in launcher
    assert 'run_condition gap-ask-disabled "$GAP_OPENINGS"' in launcher
    assert (
        'run_condition complete-ask-enabled "$COMPLETE_OPENINGS"'
        in launcher
    )
    assert "MULTITURN_LIMIT" in launcher
    assert 'limit_args=(--limit "$EVALUATION_LIMIT")' in launcher
    assert "evaluation-v1" not in launcher
