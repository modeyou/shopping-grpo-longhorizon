"""CARL-BPO training-return alignment for the immutable Reward v4 outcome."""

from __future__ import annotations

import math
from collections.abc import Mapping


GOLD_PURCHASE = "gold_purchase"
VALID_ALTERNATIVE_PURCHASE = "valid_alternative_purchase"


def completion_aligned_train_return(
    reward: Mapping[str, object],
    *,
    reward_type: str | None = None,
) -> float:
    """Map a validated Reward-v4 terminal outcome to the CARL train score.

    Reward v4 remains the evaluation authority.  This score is only consumed by
    the on-policy BPO return/LOO path, so the success ordering is explicit while
    normal failures retain a small amount of native utility ordering.
    """
    if not isinstance(reward, Mapping):
        raise ValueError("BPO training return requires a reward mapping")
    kind = str(reward_type or reward.get("reward_type") or "")
    if bool(reward.get("infrastructure_invalid")) or bool(
        reward.get("reward_unverifiable")
    ) or bool(reward.get("sampling_invalid")):
        # Invalid rows are excluded by dynamic sampling.  A finite neutral score
        # keeps serialization and diagnostics well-defined before that filter.
        return 0.0
    if kind == GOLD_PURCHASE:
        return 1.25
    if kind == VALID_ALTERNATIVE_PURCHASE:
        return 1.0
    if bool(reward.get("model_failure")):
        return -0.075
    try:
        utility = float(
            reward.get(
                "native_terminal_utility",
                reward.get("terminal_utility", reward.get("total", 0.0)),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Reward v4 terminal utility is not numeric") from exc
    if not math.isfinite(utility):
        raise ValueError("Reward v4 terminal utility is not finite")
    return 0.10 * max(-1.0, min(1.0, utility))
