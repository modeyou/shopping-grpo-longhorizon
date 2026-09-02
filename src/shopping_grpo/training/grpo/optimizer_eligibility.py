"""Neutral policy-pathology classification for completed shopping rollouts."""

from __future__ import annotations

from numbers import Integral


SHOPPER_REJECTION_EXCLUSION_THRESHOLD = 3
SHOPPER_REJECTION_EXCLUSION_REASON = "shopper_rejections_gte_3"


def classify_policy_pathology(shopper_rejections: object) -> dict[str, object]:
    """Classify rejection loops without changing reward or environment validity."""
    if isinstance(shopper_rejections, bool) or not isinstance(
        shopper_rejections, Integral
    ):
        raise ValueError("shopper_rejections must be a non-negative integer")
    rejection_count = int(shopper_rejections)
    if rejection_count < 0:
        raise ValueError("shopper_rejections must be a non-negative integer")

    pathological = rejection_count >= SHOPPER_REJECTION_EXCLUSION_THRESHOLD
    return {
        "policy_pathology": pathological,
        "policy_pathology_reason": (
            SHOPPER_REJECTION_EXCLUSION_REASON if pathological else None
        ),
    }
