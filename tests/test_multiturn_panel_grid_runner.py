from pathlib import Path

from scripts.run_multiturn_panel_grid import build_commands


def test_grid_runner_builds_one_panel_run_per_actor_and_condition(tmp_path):
    plan = {
        "expected_tasks": "tasks.jsonl",
        "rubrics": "rubrics.jsonl",
        "judge": {"model": "judge", "base_url": "https://judge.example/v1"},
        "runs": [
            {
                "actor_label": "base",
                "trajectories": {
                    "gap-ask-enabled": "base-g+.jsonl",
                    "gap-ask-disabled": "base-g-.jsonl",
                    "complete-ask-enabled": "base-c+.jsonl",
                },
            },
            {
                "actor_label": "rl",
                "trajectories": {
                    "gap-ask-enabled": "rl-g+.jsonl",
                    "gap-ask-disabled": "rl-g-.jsonl",
                    "complete-ask-enabled": "rl-c+.jsonl",
                },
            },
        ],
    }

    commands = build_commands(
        plan,
        plan_dir=tmp_path,
        output_root=tmp_path / "out",
        allow_blind_final=True,
        resume=False,
        comparison_output=None,
    )

    assert len(commands) == 7
    assert "--rubric-approval" not in commands[0]
    assert commands[0][commands[0].index("--output-dir") + 1] == str(
        tmp_path / "out" / "base" / "gap-ask-enabled"
    )
    assert commands[0][commands[0].index("--judge-cache-dir") + 1] == str(
        tmp_path / "out" / "semantic-judge-cache"
    )
    assert commands[3][commands[3].index("--judge-cache-dir") + 1] == str(
        tmp_path / "out" / "semantic-judge-cache"
    )
    assert commands[-1][-1] == "--allow-blind-final"
    assert "base=" + str(tmp_path / "out" / "base") in commands[-1]
    assert "rl=" + str(tmp_path / "out" / "rl") in commands[-1]


def test_grid_runner_supports_a_single_gplus_condition(tmp_path):
    plan = {
        "expected_tasks": "tasks.jsonl",
        "rubrics": "rubrics.jsonl",
        "judge": {"model": "judge", "base_url": "https://judge.example/v1"},
        "runs": [
            {
                "actor_label": "base",
                "trajectories": {"gap-ask-enabled": "base-gplus.jsonl"},
            }
        ],
    }

    commands = build_commands(
        plan,
        plan_dir=tmp_path,
        output_root=tmp_path / "out",
        allow_blind_final=False,
        resume=False,
        comparison_output=None,
        conditions=("gap-ask-enabled",),
    )

    assert len(commands) == 2
    assert commands[-1][-2:] == ["--condition", "gap-ask-enabled"]
