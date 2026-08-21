"""Explicit Reward v3/v4 dispatch without changing the v3 default."""

from __future__ import annotations

from web_agent_site.engine import reward as reward_v3
from web_agent_site.engine import reward_v4
from web_agent_site.engine.reward_features import (
    REWARD_FEATURE_VERSION as REWARD_FEATURE_VERSION_V1,
    compile_reward_features,
)
from web_agent_site.engine.reward_features_v2 import (
    REWARD_FEATURE_VERSION as REWARD_FEATURE_VERSION_V2,
    compile_reward_features_v2,
)


DEFAULT_REWARD_VERSION = reward_v3.REWARD_VERSION
SUPPORTED_REWARD_VERSIONS = {
    reward_v3.REWARD_VERSION,
    reward_v4.REWARD_VERSION,
}
REWARD_FEATURE_VERSIONS = {
    reward_v3.REWARD_VERSION: REWARD_FEATURE_VERSION_V1,
    reward_v4.REWARD_VERSION: REWARD_FEATURE_VERSION_V2,
}
REWARD_DEFAULTS = {
    reward_v3.REWARD_VERSION: reward_v3.DEFAULT_REWARDS,
    reward_v4.REWARD_VERSION: reward_v4.DEFAULT_REWARDS,
}


def _version(rewards: object) -> str:
    if isinstance(rewards, dict):
        version = str(rewards.get("version") or DEFAULT_REWARD_VERSION)
    else:
        version = DEFAULT_REWARD_VERSION
    if version not in SUPPORTED_REWARD_VERSIONS:
        raise ValueError(f"unsupported reward version: {version}")
    return version


def compile_reward_features_for_version(
    instruction_record: object,
    target_product: object,
    reward_version: str,
) -> dict:
    if reward_version == reward_v3.REWARD_VERSION:
        return compile_reward_features(instruction_record, target_product)
    if reward_version == reward_v4.REWARD_VERSION:
        return compile_reward_features_v2(
            instruction_record, target_product
        )
    raise ValueError(f"unsupported reward version: {reward_version}")


def evaluate_purchase(*args, rewards=None, **kwargs):
    module = (
        reward_v4
        if _version(rewards) == reward_v4.REWARD_VERSION
        else reward_v3
    )
    return module.evaluate_purchase(*args, rewards=rewards, **kwargs)


def evaluate_candidate_eligibility(*args, rewards=None, **kwargs):
    module = (
        reward_v4
        if _version(rewards) == reward_v4.REWARD_VERSION
        else reward_v3
    )
    return module.evaluate_candidate_eligibility(*args, **kwargs)


def evaluate_abstain(*args, rewards=None, **kwargs):
    module = (
        reward_v4
        if _version(rewards) == reward_v4.REWARD_VERSION
        else reward_v3
    )
    return module.evaluate_abstain(*args, rewards=rewards, **kwargs)


def fixed_termination(reason, rewards=None):
    module = (
        reward_v4
        if _version(rewards) == reward_v4.REWARD_VERSION
        else reward_v3
    )
    return module.fixed_termination(reason, rewards=rewards)
