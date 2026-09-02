"""Tests for the frozen GRPO optimizer-eligibility policy."""

import unittest

from shopping_grpo.training.grpo.optimizer_eligibility import (
    SHOPPER_REJECTION_EXCLUSION_REASON,
    SHOPPER_REJECTION_EXCLUSION_THRESHOLD,
    classify_policy_pathology,
)


class OptimizerEligibilityTest(unittest.TestCase):
    def test_three_rejections_is_the_first_policy_pathology(self):
        self.assertEqual(SHOPPER_REJECTION_EXCLUSION_THRESHOLD, 3)
        for count in (0, 1, 2):
            self.assertEqual(
                classify_policy_pathology(count),
                {
                    "policy_pathology": False,
                    "policy_pathology_reason": None,
                },
            )
        self.assertEqual(
            classify_policy_pathology(3),
            {
                "policy_pathology": True,
                "policy_pathology_reason": SHOPPER_REJECTION_EXCLUSION_REASON,
            },
        )
        self.assertTrue(classify_policy_pathology(36)["policy_pathology"])

    def test_rejection_count_must_be_a_non_negative_integer(self):
        for value in (-1, True, 3.0, "3", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "non-negative integer"
            ):
                classify_policy_pathology(value)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
