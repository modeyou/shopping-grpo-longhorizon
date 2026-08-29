import pytest

from shopping_grpo.training.bpo.reward import completion_aligned_train_return


@pytest.mark.parametrize(
    ("reward_type", "utility", "expected"),
    [
        ("gold_purchase", 1.0, 1.25),
        ("valid_alternative_purchase", 0.55, 1.0),
        ("partial_alternative_purchase", 0.2145, 0.02145),
        ("wrong_purchase", -0.85, -0.085),
        ("repeat_loop", -0.65, -0.065),
    ],
)
def test_completion_aligned_return_orders_completion_before_native_utility(
    reward_type, utility, expected
):
    assert completion_aligned_train_return(
        {
            "reward_type": reward_type,
            "terminal_utility": utility,
            "native_terminal_utility": utility,
        }
    ) == pytest.approx(expected)


def test_model_failure_has_a_finite_negative_training_signal():
    assert completion_aligned_train_return(
        {
            "reward_type": "assistant_finished_without_environment_done",
            "terminal_utility": 0.0,
            "model_failure": True,
        }
    ) == pytest.approx(-0.075)


def test_invalid_outcome_is_neutral_and_left_for_sampling_filter():
    assert completion_aligned_train_return(
        {
            "reward_type": "reward_unverifiable",
            "terminal_utility": 0.0,
            "reward_unverifiable": True,
        }
    ) == 0.0
