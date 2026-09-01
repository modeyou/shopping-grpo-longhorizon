"""Unit tests for the project-side reward-group filter."""

import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    append_training_diagnostic,
    build_carl_stage_assignments,
    build_rollout_diagnostics,
    carl_candidate_pools_ready,
    carl_candidate_priority,
    carl_local_stage_counts,
    effective_group_update_target,
    extract_aligned_bpo_fields,
    extract_shopping_group_signals,
    select_carl_local_stage_target,
    select_reward_varying_groups,
    summarize_bpo_group_diagnostics,
    update_carl_candidate_pool,
)


class RewardGroupSelectionTest(unittest.TestCase):
    def test_carl_local_stage_ledger_closes_exact_weighted_coverage(self):
        self.assertEqual(
            carl_local_stage_counts(20),
            {"product": 8, "option": 7, "search_strategy": 5},
        )
        self.assertEqual(
            carl_local_stage_counts(500),
            {"product": 200, "option": 175, "search_strategy": 125},
        )
        stage, counts = select_carl_local_stage_target(0)
        self.assertEqual(stage, "product")
        self.assertEqual(counts, {"product": 0, "option": 0, "search_strategy": 0})
        self.assertEqual(
            build_carl_stage_assignments(("root", "local"), "option"),
            ("root", "option"),
        )

    def test_carl_reservoir_priority_is_goal_first_then_signal_strength(self):
        completion = {
            "uid": "completion",
            "contrast_type": "completion_contrast",
            "train_returns": (0.0, 1.0, 0.0, 0.0),
        }
        gold = {
            "uid": "gold",
            "contrast_type": "gold_contrast",
            "train_returns": (0.0, 1.25, 0.0, 0.0),
        }
        failure = {
            "uid": "failure",
            "contrast_type": "failure_utility_contrast",
            "train_returns": (-0.8, -0.2, -0.5, -0.5),
        }
        self.assertLess(carl_candidate_priority(gold, 3), carl_candidate_priority(completion, 1))
        self.assertLess(carl_candidate_priority(completion, 3), carl_candidate_priority(failure, 1))

        pool, first = update_carl_candidate_pool(
            {},
            group_type="root",
            candidate={"uid": "failure", "group": failure, "payload": "old"},
            generation_batch=1,
        )
        pool, second = update_carl_candidate_pool(
            pool,
            group_type="root",
            candidate={"uid": "completion", "group": completion, "payload": "new"},
            generation_batch=2,
        )
        pool, third = update_carl_candidate_pool(
            pool,
            group_type="root",
            candidate={"uid": "gold", "group": gold, "payload": "best"},
            generation_batch=3,
        )
        self.assertTrue(first["pool_selected"])
        self.assertEqual(second["replaced_uid"], "failure")
        self.assertEqual(third["replaced_uid"], "completion")
        self.assertEqual(pool["root"]["uid"], "gold")

        local_pool, _ = update_carl_candidate_pool(
            pool,
            group_type="local",
            candidate={"uid": "local", "group": completion},
            generation_batch=3,
        )
        self.assertEqual(
            carl_candidate_pools_ready(
                local_pool,
                generation_batch=3,
                quality_search_gen_batches=10,
            ),
            (True, True),
        )

    def test_carl_failure_pair_waits_for_quality_window(self):
        failure = {
            "uid": "failure",
            "contrast_type": "failure_utility_contrast",
            "train_returns": (-0.8, -0.2, -0.5, -0.5),
        }
        pool = {
            "root": {"group": failure},
            "local": {"group": failure},
        }
        self.assertEqual(
            carl_candidate_pools_ready(
                pool,
                generation_batch=9,
                quality_search_gen_batches=10,
            ),
            (False, False),
        )
        self.assertEqual(
            carl_candidate_pools_ready(
                pool,
                generation_batch=10,
                quality_search_gen_batches=10,
            ),
            (True, False),
        )

    def test_local_stage_fallback_is_not_trainable(self):
        indices, stats = select_reward_varying_groups(
            ["local"] * 4,
            [0.0, 1.0, 0.0, 0.0],
            purchase_success=[False, True, False, False],
            group_types=["local"] * 4,
            local_stage_fallback=[True] * 4,
        )
        self.assertEqual(indices, [])
        self.assertEqual(stats["local_stage_mismatch_group_count"], 1)
        self.assertEqual(stats["groups"][0]["drop_reason"], "local_stage_mismatch")

    def test_effective_return_budget_caps_the_last_update_without_overshoot(self):
        self.assertEqual(
            effective_group_update_target(
                effective_return_budget=1600,
                rollout_n=4,
                trained_groups=398,
                update_target=2,
                update_minimum=2,
                require_full_batch=True,
            ),
            (2, 2),
        )
        with self.assertRaisesRegex(ValueError, "already exhausted"):
            effective_group_update_target(
                effective_return_budget=1600,
                rollout_n=4,
                trained_groups=400,
                update_target=2,
                update_minimum=2,
                require_full_batch=True,
            )
        with self.assertRaisesRegex(ValueError, "divisible by update_target"):
            effective_group_update_target(
                effective_return_budget=1596,
                rollout_n=4,
                trained_groups=0,
                update_target=2,
                update_minimum=2,
                require_full_batch=True,
            )

    def test_bpo_diagnostics_preserve_branch_location_and_diversity(self):
        non_tensors = {
            "bpo_sibling_index": [0, 1],
            "bpo_branch_action": [1, 1],
            "bpo_branch_entropy": [2.4, 2.4],
            "bpo_branch_prefix_sha256": ["same", "same"],
            "bpo_branch_action_sha256": ["action-a", "action-b"],
            "bpo_backbone_action_count": [4, 4],
            "bpo_branch_relative_position": [1 / 3, 1 / 3],
            "bpo_branch_prefix_steps": [1, 1],
            "bpo_branch_prefix_shopper_calls": [1, 1],
            "bpo_branch_prefix_environment_transitions": [1, 1],
        }
        fields = extract_aligned_bpo_fields(non_tensors, expected_length=2)
        rollouts = build_rollout_diagnostics(
            ["tree", "tree"],
            [
                {
                    "actions": [{"tool": "search"}, {"tool": "click"}],
                    "termination_reason": "done",
                    "steps": 4,
                    "shopper_llm_calls": 2,
                },
                {
                    "actions": [{"tool": "search"}, {"tool": "ask_shopper"}],
                    "termination_reason": "max_steps",
                    "steps": 3,
                    "shopper_llm_calls": 2,
                },
            ],
            aligned_fields=fields,
        )
        summary = summarize_bpo_group_diagnostics(rollouts)["tree"]
        self.assertEqual(summary["bpo_branch_action"], 1)
        self.assertEqual(summary["bpo_backbone_action_count"], 4)
        self.assertEqual(summary["bpo_unique_branch_action_count"], 2)
        self.assertEqual(summary["bpo_unique_tool_sequence_count"], 2)
        self.assertEqual(summary["bpo_termination_reasons"], ("done", "max_steps"))
        self.assertEqual(summary["bpo_cost_backbone_rollouts"], 1)
        self.assertEqual(summary["bpo_cost_branch_rollouts"], 1)
        self.assertEqual(summary["bpo_cost_environment_transitions"], 2)
        self.assertEqual(summary["bpo_cost_shopper_api_calls"], 3)

    def test_root_diagnostics_allow_independent_action_counts(self):
        records = []
        for sibling, action_count in enumerate((1, 2, 2, 3)):
            records.append(
                {
                    "uid": "root-tree",
                    "bpo_group_type": "root",
                    "bpo_local_stage": "root",
                    "bpo_local_stage_target": "root",
                    "bpo_local_stage_fallback": False,
                    "bpo_local_stage_unavailable": False,
                    "bpo_branch_action": -1,
                    "bpo_branch_entropy": 0.0,
                    "bpo_branch_prefix_sha256": "same-prompt",
                    "bpo_branch_action_sha256": f"response-{sibling}",
                    "bpo_backbone_action_count": action_count,
                    "bpo_branch_relative_position": -1.0,
                    "bpo_branch_prefix_steps": 0,
                    "bpo_branch_prefix_shopper_calls": 0,
                    "bpo_branch_prefix_environment_transitions": 0,
                    "actions": [{}] * action_count,
                }
            )

        summary = summarize_bpo_group_diagnostics(records)["root-tree"]

        self.assertEqual(summary["bpo_backbone_action_count"], 2.0)
        self.assertEqual(summary["bpo_backbone_action_count_min"], 1)
        self.assertEqual(summary["bpo_backbone_action_count_max"], 3)

    def test_training_diagnostics_append_public_rollouts_as_jsonl(self):
        rollouts = build_rollout_diagnostics(
            ["task-a", "task-a"],
            [
                {
                    "task_id": 7,
                    "actions": [
                        {"tool": "search", "parameters": {"query": "shoe"}}
                    ],
                },
                {"task_id": 7, "termination_reason": "max_steps"},
            ],
        )
        self.assertEqual([item["rollout_index"] for item in rollouts], [0, 1])
        self.assertEqual(rollouts[0]["actions"][0]["tool"], "search")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics" / "training.jsonl"
            append_training_diagnostic(
                path,
                "generation_batch",
                3,
                rollouts=rollouts,
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["event"], "generation_batch")
        self.assertEqual(record["global_step"], 3)
        self.assertEqual(record["rollouts"][1]["termination_reason"], "max_steps")

    def test_all_zero_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(["a"] * 4, [0, 0, 0, 0])
        self.assertEqual(indices, [])
        self.assertEqual(stats["dropped_uids"], ("a",))
        self.assertEqual(stats["all_zero_utility_group_count"], 1)
        self.assertEqual(stats["all_purchase_success_group_count"], 0)

    def test_all_one_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [1, 1, 1, 1],
            terminal_utilities=[1.0, 1.0, 1.0, 1.0],
            purchase_success=[True] * 4,
        )
        self.assertEqual(indices, [])
        self.assertEqual(stats["kept_group_count"], 0)
        self.assertEqual(stats["all_purchase_success_group_count"], 1)

    def test_fractional_reward_variance_is_kept(self):
        rewards = [2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(["a"] * 4, rewards)
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(stats["kept_uids"], ("a",))

    def test_mixed_uids_preserve_trajectory_indices(self):
        uids = ["a", "b", "a", "b", "a", "b", "a", "b"]
        rewards = [0, 2 / 7, 0, 4 / 7, 0, 2 / 7, 0, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [1, 3, 5, 7])
        self.assertEqual(stats["kept_uids"], ("b",))
        self.assertEqual(stats["dropped_uids"], ("a",))

    def test_zero_and_varying_groups_keep_only_varying_group(self):
        uids = ["zero"] * 4 + ["signal"] * 4
        rewards = [0, 0, 0, 0, 2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [4, 5, 6, 7])
        self.assertEqual(stats["kept_group_count"], 1)
        self.assertEqual(stats["dropped_group_count"], 1)

    def test_group_type_is_preserved_for_root_local_batch_selection(self):
        uids = ["root"] * 4 + ["local"] * 4
        rewards = [1.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        indices, stats = select_reward_varying_groups(
            uids,
            rewards,
            group_types=["root"] * 4 + ["local"] * 4,
            purchase_success=[True, False, False, False] + [True, False, False, False],
        )
        self.assertEqual(indices, list(range(8)))
        self.assertEqual(
            {group["uid"]: group["group_type"] for group in stats["groups"]},
            {"root": "root", "local": "local"},
        )
        with self.assertRaisesRegex(ValueError, "inconsistent CARL group types"):
            select_reward_varying_groups(
                ["root"] * 4,
                [1.0, 0.0, 0.0, 0.0],
                group_types=["root", "local", "root", "root"],
            )

    def test_tolerance_treats_tiny_roundoff_as_constant(self):
        indices, _ = select_reward_varying_groups(
            ["a"] * 4,
            [0.5, 0.5 + 1.0e-9, 0.5, 0.5],
            tolerance=1.0e-8,
        )
        self.assertEqual(indices, [])

    def test_varying_terminal_utility_is_kept_without_purchase_success(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.85, -0.65, -0.50, -0.35],
            terminal_utilities=[-0.85, -0.65, -0.50, -0.35],
            purchase_success=[False] * 4,
            sampling_invalid=[False] * 4,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])
        self.assertEqual(stats["no_purchase_success_group_count"], 1)

    def test_varying_group_with_purchase_success_is_kept(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.5, 0.55, -0.5, -0.5],
            terminal_utilities=[-0.5, 0.55, -0.5, -0.5],
            purchase_success=[False, True, False, False],
            sampling_invalid=[False] * 4,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])

    def test_sampling_invalid_member_drops_the_whole_group_with_reason(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [0.0, 0.2, 0.0, 0.0],
            terminal_utilities=[0.0, 0.2, 0.0, 0.0],
            purchase_success=[False, True, False, False],
            sampling_invalid=[False, True, False, False],
            sampling_invalid_reasons=[(), ("infrastructure_invalid",), (), ()],
        )

        self.assertEqual(indices, [])
        self.assertEqual(stats["groups"][0]["drop_reason"], "sampling_invalid")
        self.assertEqual(stats["sampling_invalid_group_count"], 1)
        self.assertEqual(
            stats["sampling_invalid_reason_counts"]["infrastructure_invalid"],
            1,
        )

    def test_shopping_extra_fields_are_reduced_to_filter_signals(self):
        utility, success, invalid, reasons = extract_shopping_group_signals(
            [
                {
                    "infrastructure_invalid": False,
                    "reward": {
                        "terminal_utility": 0.55,
                        "purchase_success": True,
                        "sampling_invalid": False,
                    },
                },
                {
                    "infrastructure_invalid": True,
                    "reward": {
                        "terminal_utility": 0.0,
                        "purchase_success": False,
                        "sampling_invalid": True,
                    },
                },
            ]
        )

        self.assertEqual(utility, [0.55, 0.0])
        self.assertEqual(success, [True, False])
        self.assertEqual(invalid, [False, True])
        self.assertEqual(reasons, [(), ("infrastructure_invalid",)])

    def test_unverifiable_reward_is_sampling_invalid_but_not_infrastructure(self):
        utility, success, invalid, reasons = extract_shopping_group_signals(
            [
                {
                    "infrastructure_invalid": False,
                    "reward_unverifiable": True,
                    "reward": {
                        "terminal_utility": 0.0,
                        "purchase_success": False,
                        "sampling_invalid": True,
                    },
                }
            ]
        )
        self.assertEqual(utility, [0.0])
        self.assertEqual(success, [False])
        self.assertEqual(invalid, [True])
        self.assertEqual(reasons, [("reward_unverifiable",)])

    def test_missing_shopping_filter_signal_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "shopping"):
            extract_shopping_group_signals([None])

    def test_shopping_metrics_are_aggregated_for_a0_and_a1(self):
        infos = [
            {
                "steps": 10,
                "done": True,
                "termination_reason": "environment_done",
                "infrastructure_invalid": False,
                "reward_valid": True,
                "shopper_questions": 1,
                "shopper_rejections": 0,
                "reward": {
                    "full": 1.0,
                    "strict": 1.0,
                    "native": 1.0,
                    "semantic": 1.7,
                    "total": 1.73,
                    "efficiency": 0.03,
                    "penalty_overlong": 0.0,
                    "penalty_unfinished": 0.0,
                    "penalty_repeat": 0.0,
                    "repeat_action_rate": 0.0,
                    "terminal_utility": 1.73,
                    "purchase_success": 1.0,
                    "sampling_invalid": False,
                    "r_type": 1.0,
                    "r_att": 1.0,
                    "r_option": 1.0,
                    "r_price": 1.0,
                },
            },
            {
                "steps": 35,
                "done": False,
                "termination_reason": "max_steps",
                "infrastructure_invalid": False,
                "reward_valid": False,
                "shopper_questions": 0,
                "shopper_rejections": 1,
                "reward": {
                    "full": 0.0,
                    "strict": 0.0,
                    "native": 0.0,
                    "semantic": 0.0,
                    "total": -0.05,
                    "efficiency": 0.0,
                    "penalty_overlong": 0.05,
                    "penalty_unfinished": 0.0,
                    "penalty_repeat": 0.0,
                    "repeat_action_rate": 0.0,
                    "terminal_utility": -0.05,
                    "purchase_success": 0.0,
                    "sampling_invalid": False,
                    "r_type": 0.0,
                    "r_att": 0.0,
                    "r_option": 0.0,
                    "r_price": 0.0,
                },
            },
        ]

        metrics = aggregate_shopping_metrics(infos)

        self.assertEqual(metrics["reward/full_mean"], 0.5)
        self.assertEqual(metrics["reward/shaped_min"], -0.05)
        self.assertEqual(metrics["reward/shaped_max"], 1.73)
        self.assertEqual(metrics["reward/purchase_success_rate"], 0.5)
        self.assertEqual(metrics["reward/valid_rate"], 0.5)
        self.assertEqual(metrics["component/r_type_mean"], 0.5)
        self.assertEqual(metrics["trajectory/average_steps"], 22.5)
        self.assertEqual(metrics["trajectory/done_rate"], 0.5)
        self.assertEqual(metrics["trajectory/max_steps_rate"], 0.5)
        self.assertEqual(metrics["trajectory/shopper_question_rate"], 0.5)
        self.assertEqual(metrics["trajectory/shopper_questions_mean"], 0.5)
        self.assertEqual(metrics["trajectory/shopper_rejections_mean"], 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
