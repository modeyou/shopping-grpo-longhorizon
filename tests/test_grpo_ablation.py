import unittest

from scripts.check_grpo_runtime import (
    ppo_gradient_accumulation_steps,
    validate_grpo_seeds,
    validate_reward_shaping_profile,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    apply_bounded_reward_shaping,
    apply_reward_length_shaping,
    reward_breakdown,
    reward_shaping_config,
)
from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    extract_shopping_group_signals,
)


class GrpoAblationTest(unittest.TestCase):
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
            with self.assertRaisesRegex(SystemExit, "cannot be combined"):
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


if __name__ == "__main__":
    unittest.main()
