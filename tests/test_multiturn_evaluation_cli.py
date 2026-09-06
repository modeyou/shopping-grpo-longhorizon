import sys
from pathlib import Path
from unittest.mock import patch

from scripts.compare_multiturn_evaluations import (
    _run_spec,
    parse_args as parse_comparison_args,
)
from scripts.evaluate_multiturn_panels import parse_args as parse_panel_args
from scripts.freeze_multiturn_rubrics import parse_args as parse_rubric_args


def test_frozen_model_defaults_match_selected_evaluation_stack():
    with patch.object(
        sys,
        "argv",
        [
            "freeze_multiturn_rubrics.py",
            "--tasks",
            "tasks.jsonl",
            "--output-dir",
            "shared",
        ],
    ):
        rubric = parse_rubric_args()
    assert rubric.model == "qwen3.8-27b"
    assert rubric.base_url == "http://127.0.0.1:8001/v1"

    with patch.object(
        sys,
        "argv",
        [
            "evaluate_multiturn_panels.py",
            "--expected-tasks",
            "tasks.jsonl",
            "--trajectories",
            "raw.jsonl",
            "--rubrics",
            "rubrics.jsonl",
            "--output-dir",
            "out",
            "--judge-cache-dir",
            "shared-judge-cache",
            "--actor-label",
            "base",
            "--condition",
            "gap-ask-enabled",
        ],
    ):
        panel = parse_panel_args()
    assert panel.judge_model == "deepseek-v4-flash-0731"
    assert panel.condition == "gap-ask-enabled"
    assert panel.judge_cache_dir == Path("shared-judge-cache")


def test_comparison_run_spec_is_label_and_root():
    assert _run_spec("base=outputs/base") == ("base", Path("outputs/base"))


def test_comparison_cli_accepts_a_single_condition():
    with patch.object(
        sys,
        "argv",
        [
            "compare_multiturn_evaluations.py",
            "--expected-tasks",
            "tasks.jsonl",
            "--run",
            "base=outputs/base",
            "--condition",
            "gap-ask-enabled",
            "--output",
            "comparison.json",
        ],
    ):
        args = parse_comparison_args()
    assert args.condition == ["gap-ask-enabled"]
