import pytest

from scripts.summarize_final_evaluations import exact_mcnemar_p, summarize, wilson_interval


def _run(successes):
    summaries = {}
    strict_ids = {}
    result_conditions = {}
    for condition, ids in successes.items():
        strict_ids[condition] = set(ids)
        summaries[condition] = {
            "strict_successes": len(ids),
            "done_tasks": 200,
            "reward_valid_tasks": 200,
            "mean_final_reward": 0.5,
        }
        result_conditions[condition] = {
            "complete_unnecessary_ask_tasks": 100 if condition == "complete-ask-enabled" else 0
        }
    return {
        "audit": {
            "asset_manifest_sha256": "a" * 64,
            "model_name": "model",
            "result": {
                **result_conditions,
                "derived": {"complete_unnecessary_ask_rate": 0.5},
            },
        },
        "summaries": summaries,
        "strict_ids": strict_ids,
    }


def test_exact_mcnemar_matches_known_symmetric_case():
    assert exact_mcnemar_p(18, 20) == pytest.approx(0.871414679365)
    assert exact_mcnemar_p(0, 0) == 1.0


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(100, 200)
    assert low < 0.5 < high


def test_summary_builds_condition_and_model_pairing():
    conditions = (
        "gap-ask-enabled",
        "gap-ask-disabled",
        "complete-ask-enabled",
    )
    base = _run({condition: [1] for condition in conditions})
    sft = _run({condition: [1, 2] for condition in conditions})
    rl = _run({condition: [1, 2, 3] for condition in conditions})
    result = summarize({"base": base, "sft": sft, "rl": rl})
    assert result["models"]["rl"]["three_condition_strict"] == 9
    paired = result["pairwise"]["sft_to_rl"]["all_conditions"]
    assert paired["gains"] == 3
    assert paired["losses"] == 0
    assert paired["ties"] == 597


def test_summary_supports_arbitrary_model_count():
    conditions = (
        "gap-ask-enabled",
        "gap-ask-disabled",
        "complete-ask-enabled",
    )
    runs = {
        label: _run(
            {
                condition: list(range(1, success_count + 1))
                for condition in conditions
            }
        )
        for label, success_count in (
            ("base", 1),
            ("sft", 2),
            ("bpo-v1", 3),
            ("carl-bpo-v2.1", 4),
        )
    }

    result = summarize(runs)

    assert list(result["models"]) == list(runs)
    assert len(result["pairwise"]) == 6
    assert "bpo-v1_to_carl-bpo-v2.1" in result["pairwise"]
