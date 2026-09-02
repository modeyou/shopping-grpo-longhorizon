import unittest
from pathlib import Path

from scripts.check_grpo_runtime import (
    ppo_gradient_accumulation_steps,
    validate_grpo_seeds,
    validate_reward_shaping_profile,
    validate_visible_gpu_headroom,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    apply_bounded_reward_shaping,
    apply_reward_length_shaping,
    reward_breakdown,
    reward_shaping_config,
)
from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    aggregate_validation_shopping_metrics,
    augment_training_monitor_metrics,
    extract_shopping_group_signals,
)


class GrpoAblationTest(unittest.TestCase):
    def test_formal_gpu_preflight_audits_sparse_physical_mask(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        free_bytes = 21 * 1024 ** 3
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 4,
                mem_get_info=lambda index: (free_bytes, 24 * 1024 ** 3),
            )
        )
        environment = {
            "CUDA_VISIBLE_DEVICES": "0,2,3,4",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        with patch.dict(os.environ, environment, clear=True):
            audit = validate_visible_gpu_headroom(torch, expected_devices=4)
        self.assertEqual(audit["physical_devices"], ["0", "2", "3", "4"])
        self.assertEqual(
            audit["physical_to_logical"],
            {"0": 0, "2": 1, "3": 2, "4": 3},
        )

    def test_formal_gpu_preflight_rejects_implicit_or_ray_rewritten_mask(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "explicit CUDA_VISIBLE_DEVICES"):
                validate_visible_gpu_headroom(torch, expected_devices=4)
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0,2,3,4"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                SystemExit,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1",
            ):
                validate_visible_gpu_headroom(torch, expected_devices=4)

    def test_grpo_yaml_decodes_environment_seed_as_integer(self):
        config = (
            Path(__file__).resolve().parents[1] / "configs" / "grpo.yaml"
        ).read_text(encoding="utf-8")
        decoded_seed = "${oc.decode:${oc.env:GRPO_SEED,20260823}}"
        self.assertEqual(config.count(decoded_seed), 2)

    def test_grpo_yaml_freezes_formal_native_500_contract(self):
        config = (
            Path(__file__).resolve().parents[1] / "configs" / "grpo.yaml"
        ).read_text(encoding="utf-8")
        for line in (
            "use_remove_padding: true",
            "use_fused_kernels: true",
            "use_liger: true",
            "impl_backend: torch",
            "lr_warmup_steps: -1",
            "lr_warmup_steps_ratio: 0.03",
            "lr_scheduler_type: constant",
            "total_training_steps: 500",
            "save_freq: 25",
            "test_freq: 50",
            "logger: [console, swanlab]",
            "n_gpus_per_node: 4",
        ):
            self.assertIn(line, config)
        self.assertNotIn("shopping_scheduler:", config)

    def test_ppo_mini_and_micro_batches_define_gradient_accumulation(self):
        self.assertEqual(ppo_gradient_accumulation_steps(4, 2), 2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            ppo_gradient_accumulation_steps(3, 2)

    def test_grpo_data_and_actor_seeds_must_match(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        config = SimpleNamespace(
            data=SimpleNamespace(seed=20260823),
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(data_loader_seed=20260823)
            ),
        )
        with patch.dict(os.environ, {"GRPO_SEED": "20260823"}, clear=False):
            validate_grpo_seeds(config)
        config.actor_rollout_ref.actor.data_loader_seed = 7
        with patch.dict(os.environ, {"GRPO_SEED": "20260823"}, clear=False):
            with self.assertRaisesRegex(SystemExit, "must match"):
                validate_grpo_seeds(config)

    def test_grpo_seeds_must_resolve_to_integers(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        config = SimpleNamespace(
            data=SimpleNamespace(seed="20260823"),
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(data_loader_seed="20260823")
            ),
        )
        with patch.dict(os.environ, {"GRPO_SEED": "20260823"}, clear=False):
            with self.assertRaisesRegex(SystemExit, "must resolve to integers"):
                validate_grpo_seeds(config)

    def test_reward_profiles_are_frozen_and_reject_unknown_values(self):
        self.assertEqual(reward_shaping_config("none"), {"profile": "none"})
        bounded = reward_shaping_config("bounded-v1")
        self.assertEqual(bounded["unfinished_penalty"], 0.75)
        self.assertEqual(bounded["behavior_penalty_cap"], 0.10)
        with self.assertRaisesRegex(ValueError, "unknown reward shaping profile"):
            reward_shaping_config("experimental")

    def test_native_profile_keeps_model_failure_as_zero_reward_signal(self):
        state = {
            "done": False,
            "error": "assistant_finished_without_environment_done",
            "infrastructure_invalid": False,
            "reward_unverifiable": False,
            "steps": [],
        }
        native = reward_breakdown(state)
        shaped = apply_bounded_reward_shaping(native, state, profile="none")

        self.assertTrue(shaped["model_failure"])
        self.assertFalse(shaped["sampling_invalid"])
        self.assertEqual(shaped["terminal_utility"], 0.0)
        self.assertEqual(shaped["native_terminal_utility"], 0.0)

    def test_bounded_profile_turns_model_failure_into_finite_negative_signal(self):
        state = {
            "done": False,
            "error": "too_many_guard_rejections",
            "infrastructure_invalid": False,
            "reward_unverifiable": False,
            "steps": [],
        }
        shaped = apply_bounded_reward_shaping(
            reward_breakdown(state),
            state,
            profile="bounded-v1",
        )

        self.assertFalse(shaped["sampling_invalid"])
        self.assertEqual(shaped["penalty_unfinished"], 0.75)
        self.assertEqual(shaped["terminal_utility"], -0.75)
        self.assertEqual(shaped["native_terminal_utility"], 0.0)

    def test_bounded_behavior_cost_is_capped_and_first_question_is_free(self):
        reward = {
            "total": 1.0,
            "terminal_utility": 1.0,
            "native_terminal_utility": 1.0,
            "sampling_invalid": False,
            "infrastructure_invalid": False,
            "reward_unverifiable": False,
            "model_failure": False,
            "penalty_repeat": 0.0,
        }
        state = {
            "shopper_question_count": 2,
            "shopper_rejection_count": 2,
            "guard_rejection_count": 8,
            "repeat_action_count": 4,
        }
        shaped = apply_bounded_reward_shaping(
            reward,
            state,
            profile="bounded-v1",
        )

        self.assertEqual(shaped["penalty_extra_questions"], 0.02)
        self.assertEqual(shaped["penalty_behavior"], 0.10)
        self.assertAlmostEqual(shaped["terminal_utility"], 0.90)
        self.assertEqual(shaped["native_terminal_utility"], 1.0)

    def test_bounded_profile_never_converts_infrastructure_failure(self):
        state = {
            "done": False,
            "error": "shopper_error:HTTPError",
            "infrastructure_invalid": True,
            "reward_unverifiable": False,
            "steps": [],
        }
        shaped = apply_bounded_reward_shaping(
            reward_breakdown(state),
            state,
            profile="bounded-v1",
        )

        self.assertTrue(shaped["sampling_invalid"])
        self.assertTrue(shaped["infrastructure_invalid"])
        self.assertEqual(shaped["terminal_utility"], 0.0)

    def test_reward_profile_preflight_rejects_compound_shaping(self):
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "SHOPPING_REWARD_SHAPING_PROFILE": "bounded-v1",
                "SHOPPING_LENGTH_SHAPING_ENABLE": "true",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(SystemExit, "only native Reward v4"):
                validate_reward_shaping_profile()

    def test_disabled_length_shaping_is_a_no_op(self):
        reward = {"total": 0.8, "terminal_utility": 0.8, "penalty_overlong": 0.0, "sampling_invalid": False}
        state = {"steps": [{}] * 25, "max_steps": 35, "termination_reason": "gold_purchase"}

        shaped = apply_reward_length_shaping(reward, state, enabled=False)

        self.assertEqual(shaped, reward)
        self.assertIsNot(shaped, reward)

    def test_penalty_starts_after_soft_threshold_and_is_capped(self):
        reward = {"total": 0.8, "terminal_utility": 0.8, "penalty_overlong": 0.0, "sampling_invalid": False}
        state = {"steps": [{}] * 30, "max_steps": 35, "termination_reason": "gold_purchase"}

        shaped = apply_reward_length_shaping(
            reward,
            state,
            enabled=True,
            soft_threshold=20,
            penalty_per_step=0.02,
            max_penalty=0.1,
        )

        self.assertAlmostEqual(shaped["penalty_overlong"], 0.1)
        self.assertAlmostEqual(shaped["total"], 0.7)
        self.assertAlmostEqual(shaped["terminal_utility"], 0.7)
        self.assertFalse(shaped["sampling_invalid"])

    def test_max_step_trajectory_is_invalid_for_dynamic_resampling(self):
        reward = {"total": 0.0, "terminal_utility": 0.0, "penalty_overlong": 0.0, "sampling_invalid": False}
        state = {"steps": [{}] * 35, "max_steps": 35, "termination_reason": "max_steps"}
        shaped = apply_reward_length_shaping(
            reward,
            state,
            enabled=True,
            soft_threshold=20,
            penalty_per_step=0.01,
            max_penalty=0.2,
        )
        info = {
            "overlong": shaped["overlong"],
            "infrastructure_invalid": False,
            "reward_unverifiable": False,
            "reward": {**shaped, "purchase_success": False},
        }

        _, _, invalid, reasons = extract_shopping_group_signals([info])

        self.assertEqual(invalid, [True])
        self.assertEqual(reasons, [("overlong",)])

    def test_metrics_keep_overlong_repeat_loop_and_max_step_rates(self):
        base_reward = {
            "full": 0.0, "strict": 0.0, "native": 0.0, "semantic": 0.0,
            "total": 0.0, "terminal_utility": 0.0, "efficiency": 0.0,
            "penalty_overlong": 0.0, "penalty_unfinished": 0.0, "penalty_repeat": 0.0,
            "repeat_action_rate": 1.0, "purchase_success": False, "sampling_invalid": True,
            "r_type": 0.0, "r_att": 0.0, "r_option": 0.0, "r_price": 0.0,
        }
        metrics = aggregate_shopping_metrics([
            {
                "steps": 35,
                "done": True,
                "overlong": True,
                "termination_reason": "max_steps",
                "reward_type": "repeat_loop",
                "infrastructure_invalid": False,
                "reward": base_reward,
            }
        ])

        self.assertEqual(metrics["trajectory/overlong_rate"], 1.0)
        self.assertEqual(metrics["trajectory/repeat_loop_rate"], 1.0)
        self.assertEqual(metrics["trajectory/max_steps_rate"], 1.0)

    def test_validation_metrics_are_balanced_and_split_by_mode(self):
        def info(mode, strict):
            reward = {
                "full": strict, "strict": strict, "native": strict,
                "semantic": strict, "total": strict,
                "terminal_utility": strict, "efficiency": 1.0,
                "penalty_overlong": 0.0, "penalty_unfinished": 0.0,
                "penalty_repeat": 0.0, "repeat_action_rate": 0.0,
                "purchase_success": bool(strict), "sampling_invalid": False,
                "r_type": strict, "r_att": strict,
                "r_option": strict, "r_price": strict,
            }
            return {
                "interaction_mode": mode, "steps": 2, "done": True,
                "termination_reason": "gold_purchase", "reward_type": "gold_purchase",
                "infrastructure_invalid": False, "reward": reward,
                "shopper_questions": 1 if mode == "gap" else 0,
            }

        metrics = aggregate_validation_shopping_metrics([
            info("gap", 1.0), info("gap", 0.0), info("complete", 1.0),
        ])

        self.assertEqual(metrics["validation/gap/reward/strict_mean"], 0.5)
        self.assertEqual(metrics["validation/complete/reward/strict_mean"], 1.0)
        self.assertEqual(
            metrics["validation/selection/balanced_strict_success_rate"], 0.75
        )
        self.assertEqual(metrics["validation/overall/trajectory/count"], 3.0)
        self.assertEqual(metrics["validation/gap/trajectory/shopper_ask_rate"], 1.0)
        self.assertEqual(
            metrics["validation/complete/trajectory/shopper_ask_rate"], 0.0
        )

    def test_training_monitor_exposes_native_verl_metrics_without_entropy_cost(self):
        source = {
            "actor/pg_loss": 0.2, "actor/grad_norm": 0.5, "actor/lr": 1e-6,
            "actor/pg_clipfrac": 0.1, "actor/pg_clipfrac_lower": 0.0,
            "actor/ppo_kl": 0.01, "critic/advantages/mean": 0.0,
            "critic/advantages/min": -1.0, "critic/advantages/max": 1.0,
        }
        metrics = augment_training_monitor_metrics(source)
        self.assertEqual(metrics["optimization/pg_loss"], 0.2)
        self.assertEqual(metrics["optimization/grad_norm"], 0.5)
        self.assertEqual(metrics["monitor/critical_metric_present_ratio"], 1.0)
        self.assertEqual(metrics["monitor/observed_metrics_all_finite"], 1.0)
        self.assertEqual(metrics["monitor/entropy_available"], 0.0)


if __name__ == "__main__":
    unittest.main()
