from __future__ import annotations

import json

import torch

from shopping_grpo.training.bpo.advantage import summarize_bpo_token_mass
from shopping_grpo.training.bpo.live_monitor import (
    aggregate_records,
    read_complete_jsonl,
)


def _rollouts(uid, task_id, group_type, *, stage="root", token_type=None):
    rows = []
    for sibling in range(4):
        row = {
            "uid": uid,
            "task_id": task_id,
            "interaction_mode": "gap",
            "bpo_group_type": group_type,
            "bpo_local_stage": stage,
            "bpo_sibling_index": sibling,
            "bpo_branch_action": -1 if group_type == "root" else 1,
            "bpo_action_token_starts": [0, 4],
            "bpo_action_token_ends": [4, 10],
        }
        if token_type:
            row["bpo_entropy_probe_argmax_token_type"] = token_type
        rows.append(row)
    return rows


def _step(step, task_ids):
    root_uid = f"root-{step}"
    local_uid = f"local-{step}"
    generation = {
        "event": "generation_batch",
        "global_step": step,
        "rollouts": [
            *_rollouts(root_uid, task_ids[0], "root"),
            *_rollouts(
                local_uid,
                task_ids[1],
                "local",
                stage="option",
                token_type="protocol" if step == 1 else "semantic",
            ),
        ],
    }
    selection = {
        "event": "optimizer_selection",
        "global_step": step,
        "selected_groups": {
            "root": {
                "uid": root_uid,
                "contrast_type": "completion_contrast",
                "local_stage": "root",
            },
            "local": {
                "uid": local_uid,
                "contrast_type": "completion_contrast",
                "local_stage": "option",
            },
        },
    }
    return generation, selection


def test_live_monitor_names_outcomes_and_tracks_task_repetition():
    records = [*_step(1, (7, 7)), *_step(2, (7, 8))]
    records.extend(
        [
            {
                "event": "bpo_actor_batch",
                "global_step": 1,
                "diagnostics": {"all_finite": True},
                "metrics": {"bpo_action/active_token_ratio": 0.5},
                "token_mass": {
                    "groups": [
                        {
                            "siblings": [
                                {
                                    "actions": [
                                        {
                                            "actor_tokens": 2,
                                            "selected_for_policy": True,
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
            },
            {"event": "slow_full_batch_warning", "global_step": 2},
        ]
    )

    snapshot = aggregate_records(records)

    assert snapshot["sampling"]["accepted_groups_total"] == 4
    assert snapshot["sampling"]["accepted_sibling_terminal_outcomes_total"] == 16
    assert "effective_returns" not in json.dumps(snapshot)
    assert snapshot["tasks"]["selected_task"] == {
        "groups": 4,
        "unique": 2,
        "repeated_groups": 2,
        "repeat_rate": 0.5,
        "maximum_uses": 3,
        "effective_count": 1.6,
    }
    assert snapshot["token_mass"]["exact_actor_tokens_available"] is True
    assert snapshot["entropy_probe"]["selected_local_group_token_types"] == {
        "protocol": 1,
        "semantic": 1,
    }
    assert snapshot["alerts"]["slow_full_batch_warning_total"] == 1
    assert snapshot["alerts"]["blocking"] is False


def test_live_monitor_labels_old_action_boundaries_as_unavailable():
    generation, selection = _step(1, (1, 2))
    for rollout in generation["rollouts"]:
        rollout.pop("bpo_action_token_ends")

    snapshot = aggregate_records([generation, selection])

    assert snapshot["token_mass"]["exact_actor_tokens_available"] is False
    assert snapshot["token_mass"]["span_proxy_missing_rollouts"] == 8
    assert snapshot["entropy_probe"]["token_type_available"] is True


def test_complete_jsonl_ignores_only_partial_last_line(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    path.write_text(
        '{"event":"slow_full_batch_warning","global_step":1}\n{"event":',
        encoding="utf-8",
    )

    assert read_complete_jsonl(path) == [
        {"event": "slow_full_batch_warning", "global_step": 1}
    ]


def test_exact_token_mass_preserves_group_sibling_and_action_levels():
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0],
        ]
    )
    metadata = {
        "bpo_group_id": ["root", "local"],
        "bpo_group_type": ["root", "local"],
        "bpo_sibling_index": [0, 0],
        "bpo_branch_action": [-1, 1],
        "bpo_action_token_starts": [[0, 4], [0, 4]],
        "bpo_action_token_ends": [[4, 6], [4, 6]],
        "bpo_action_metadata_valid": [True, True],
    }

    summary = summarize_bpo_token_mass(
        response_mask,
        metadata=metadata,
        sibling_count=1,
    )

    root, local = summary["groups"]
    assert root["selected_actor_tokens"] == 4
    assert local["selected_actor_tokens"] == 1
    assert [
        action["selected_for_policy"]
        for action in local["siblings"][0]["actions"]
    ] == [False, True]
