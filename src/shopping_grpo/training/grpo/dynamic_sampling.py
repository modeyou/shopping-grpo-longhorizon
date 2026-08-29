"""Pure reward-group selection used by the CARL-BPO sampling patch."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shopping_grpo.training.bpo.reward import completion_aligned_train_return


def effective_group_update_target(
    *,
    effective_return_budget: int,
    rollout_n: int,
    trained_groups: int,
    update_target: int,
    update_minimum: int,
    require_full_batch: bool = False,
) -> tuple[int, int]:
    """Bound one update so a return-budget run cannot overshoot its target."""
    values = (
        effective_return_budget,
        rollout_n,
        trained_groups,
        update_target,
        update_minimum,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("effective-return budget values must be non-negative")
    if rollout_n <= 0 or update_target <= 0 or update_minimum <= 0:
        raise ValueError("rollout and update targets must be positive")
    if update_minimum > update_target:
        raise ValueError("update minimum cannot exceed update target")
    if not effective_return_budget:
        return int(update_target), int(update_minimum)
    if effective_return_budget % rollout_n:
        raise ValueError("effective return budget must be divisible by rollout_n")
    tree_budget = effective_return_budget // rollout_n
    if require_full_batch and tree_budget % update_target:
        raise ValueError("strict tree budget must be divisible by update_target")
    remaining_groups = tree_budget - trained_groups
    if remaining_groups <= 0:
        raise ValueError("effective return budget is already exhausted")
    current_target = min(update_target, remaining_groups)
    if require_full_batch and current_target != update_target:
        raise ValueError("strict full-batch budget cannot produce a partial update")
    return current_target, min(update_minimum, current_target)


def build_carl_group_assignments(group_count: int, group_schedule) -> tuple[str, ...]:
    """Freeze Root/Local roles on the driver before rollout-worker sharding."""
    schedule = tuple(str(value) for value in group_schedule)
    if schedule != ("root", "local"):
        raise ValueError("CARL-BPO group schedule must be exactly ('root', 'local')")
    if int(group_count) != len(schedule):
        raise ValueError(
            "CARL-BPO driver batch must contain exactly one Root and one Local prompt"
        )
    return schedule


def build_rollout_diagnostics(
    uids: Sequence[Hashable],
    shopping_infos: Sequence[object],
    *,
    aligned_fields: Mapping[str, Sequence[object]] | None = None,
) -> list[dict[str, Any]]:
    """Attach stable group/rollout identities to public AgentLoop diagnostics."""
    if len(uids) != len(shopping_infos):
        raise ValueError("uids and shopping_infos must have equal length")
    rollout_counts: dict[Hashable, int] = {}
    fields = dict(aligned_fields or {})
    for name, values in fields.items():
        if not str(name).startswith("bpo_"):
            raise ValueError(f"diagnostic field is not BPO namespaced: {name}")
        if len(values) != len(uids):
            raise ValueError(f"diagnostic field {name} is not trajectory-aligned")
    records = []
    for index, (uid, info) in enumerate(zip(uids, shopping_infos, strict=True)):
        if not isinstance(info, Mapping):
            raise ValueError(f"shopping extra field at index {index} is not an object")
        rollout_index = rollout_counts.get(uid, 0)
        rollout_counts[uid] = rollout_index + 1
        bpo = {name: values[index] for name, values in fields.items()}
        records.append(
            {"uid": uid, "rollout_index": rollout_index, **dict(info), **bpo}
        )
    return records


BPO_DIAGNOSTIC_FIELDS = (
    "bpo_group_id",
    "bpo_group_type",
    "bpo_sibling_index",
    "bpo_branch_action",
    "bpo_branch_entropy",
    "bpo_action_token_starts",
    "bpo_return_budget",
    "bpo_env_idx",
    "bpo_branch_prefix_sha256",
    "bpo_branch_action_sha256",
    "bpo_backbone_action_count",
    "bpo_branch_relative_position",
    "bpo_branch_prefix_steps",
    "bpo_branch_prefix_shopper_calls",
    "bpo_branch_prefix_environment_transitions",
    "bpo_local_stage",
    "bpo_local_stage_target",
    "bpo_local_stage_fallback",
    "bpo_local_stage_unavailable",
)


def extract_aligned_bpo_fields(
    non_tensor_batch: Mapping[str, object], *, expected_length: int
) -> dict[str, Sequence[object]]:
    """Extract trajectory-aligned BPO metadata from a veRL non-tensor batch."""
    fields: dict[str, Sequence[object]] = {}
    for name in BPO_DIAGNOSTIC_FIELDS:
        if name not in non_tensor_batch:
            continue
        raw_values = non_tensor_batch[name]
        tolist = getattr(raw_values, "tolist", None)
        values = tolist() if callable(tolist) else list(raw_values)
        if len(values) != int(expected_length):
            raise ValueError(f"BPO field {name} is not trajectory-aligned")
        fields[name] = values
    return fields


def summarize_bpo_group_diagnostics(
    rollout_records: Sequence[Mapping[str, object]],
) -> dict[Hashable, dict[str, Any]]:
    """Expose branch location and sibling diversity for each BPO group."""
    grouped: dict[Hashable, list[Mapping[str, object]]] = {}
    for record in rollout_records:
        grouped.setdefault(record["uid"], []).append(record)

    summaries: dict[Hashable, dict[str, Any]] = {}
    for uid, records in grouped.items():
        if not any("bpo_branch_action" in record for record in records):
            continue
        required = (
            "bpo_branch_action",
            "bpo_branch_entropy",
            "bpo_branch_prefix_sha256",
            "bpo_branch_action_sha256",
            "bpo_backbone_action_count",
            "bpo_branch_relative_position",
            "bpo_branch_prefix_steps",
            "bpo_branch_prefix_shopper_calls",
            "bpo_branch_prefix_environment_transitions",
        )
        if any(
            any(record.get(name) is None for name in required)
            for record in records
        ):
            summaries[uid] = {"bpo_diagnostic_incomplete": True}
            continue
        branch_actions = {int(record["bpo_branch_action"]) for record in records}
        group_types = {str(record.get("bpo_group_type", "local")) for record in records}
        if len(group_types) != 1 or next(iter(group_types)) not in {"root", "local"}:
            raise ValueError("BPO sibling diagnostics disagree on group type")
        group_type = next(iter(group_types))
        stages = {str(record.get("bpo_local_stage", "unknown")) for record in records}
        if len(stages) != 1:
            raise ValueError("BPO sibling diagnostics disagree on local stage")
        local_stage = next(iter(stages))
        stage_targets = {
            str(record.get("bpo_local_stage_target", "auto")) for record in records
        }
        if len(stage_targets) != 1:
            raise ValueError("BPO sibling diagnostics disagree on stage target")
        stage_fallbacks = {
            bool(record.get("bpo_local_stage_fallback", False)) for record in records
        }
        if len(stage_fallbacks) != 1:
            raise ValueError("BPO sibling diagnostics disagree on stage fallback")
        stage_unavailable = {
            bool(record.get("bpo_local_stage_unavailable", False))
            for record in records
        }
        if len(stage_unavailable) != 1:
            raise ValueError("BPO sibling diagnostics disagree on stage availability")
        backbone_counts = {
            int(record["bpo_backbone_action_count"]) for record in records
        }
        prefix_hashes = {
            str(record["bpo_branch_prefix_sha256"]) for record in records
        }
        if len(branch_actions) != 1 or len(backbone_counts) != 1:
            raise ValueError("BPO sibling diagnostics disagree on branch location")
        if len(prefix_hashes) != 1:
            raise ValueError("BPO sibling diagnostics disagree on branch prefix")
        branch_action = next(iter(branch_actions))
        backbone_action_count = next(iter(backbone_counts))
        prefix_steps = {int(record["bpo_branch_prefix_steps"]) for record in records}
        prefix_calls = {
            int(record["bpo_branch_prefix_shopper_calls"]) for record in records
        }
        if len(prefix_steps) != 1 or len(prefix_calls) != 1:
            raise ValueError("BPO sibling diagnostics disagree on branch-prefix cost")
        prefix_step_count = next(iter(prefix_steps))
        prefix_call_count = next(iter(prefix_calls))
        prefix_environment_transitions = {
            int(record["bpo_branch_prefix_environment_transitions"])
            for record in records
        }
        if len(prefix_environment_transitions) != 1:
            raise ValueError(
                "BPO sibling diagnostics disagree on environment-transition prefix"
            )
        prefix_environment_transition_count = next(
            iter(prefix_environment_transitions)
        )

        def environment_transitions(record):
            return sum(
                str(action.get("tool") or action.get("name") or "")
                != "ask_shopper"
                for action in (record.get("actions") or [])
                if isinstance(action, Mapping)
            )
        sibling_action_hashes = []
        tool_sequences = []
        termination_reasons = []
        error_types = []
        errors = []
        for record in records:
            actions = list(record.get("actions") or [])
            sibling_action_hashes.append(str(record["bpo_branch_action_sha256"]))
            tool_sequences.append(
                tuple(
                    str(action.get("name") or action.get("tool") or "")
                    if isinstance(action, Mapping)
                    else str(action)
                    for action in actions
                )
            )
            termination_reasons.append(str(record.get("termination_reason") or ""))
            error = str(record.get("error_type") or record.get("error") or "")
            errors.append(error)
            error_types.append(error.split(":", 1)[0] if error else "")

        summaries[uid] = {
            "bpo_group_type": group_type,
            "bpo_local_stage": local_stage,
            "bpo_local_stage_target": next(iter(stage_targets)),
            "bpo_local_stage_fallback": next(iter(stage_fallbacks)),
            "bpo_local_stage_unavailable": next(iter(stage_unavailable)),
            "bpo_branch_action": branch_action,
            "bpo_backbone_action_count": backbone_action_count,
            "bpo_branch_relative_position": float(
                records[0]["bpo_branch_relative_position"]
            ),
            "bpo_branch_entropy": float(records[0]["bpo_branch_entropy"]),
            "bpo_branch_prefix_steps": prefix_step_count,
            "bpo_branch_prefix_shopper_calls": prefix_call_count,
            "bpo_branch_prefix_environment_transitions": (
                prefix_environment_transition_count
            ),
            "bpo_unique_branch_action_count": len(
                set(sibling_action_hashes)
            ),
            "bpo_unique_tool_sequence_count": len(set(tool_sequences)),
            "bpo_termination_reasons": tuple(termination_reasons),
            "bpo_error_types": tuple(error_types),
            "bpo_errors": tuple(errors),
            "bpo_cost_backbone_rollouts": 1,
            "bpo_cost_branch_rollouts": len(records) - 1,
            "bpo_cost_environment_transitions": (
                sum(environment_transitions(record) for record in records)
                if group_type == "root"
                else environment_transitions(records[0])
                + sum(
                    max(
                        0,
                        environment_transitions(record)
                        - prefix_environment_transition_count,
                    )
                    for record in records[1:]
                )
            ),
            "bpo_cost_shopper_api_calls": (
                sum(int(record.get("shopper_llm_calls", 0)) for record in records)
                if group_type == "root"
                else int(records[0].get("shopper_llm_calls", 0))
                + sum(
                    max(
                        0,
                        int(record.get("shopper_llm_calls", 0)) - prefix_call_count,
                    )
                    for record in records[1:]
                )
            ),
        }
    return summaries


def append_training_diagnostic(
    path: str | Path | None,
    event: str,
    global_step: int,
    **payload: object,
) -> None:
    """Append one driver-side training event; an unset path disables persistence."""
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def scalar(value):
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    record = {
        "schema_version": 1,
        "event": str(event),
        "global_step": int(global_step),
        **payload,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=scalar))
        handle.write("\n")


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    if not shopping_infos:
        return {}

    reward_keys = (
        "full",
        "strict",
        "native",
        "semantic",
        "total",
        "efficiency",
        "penalty_overlong",
        "penalty_unfinished",
        "penalty_repeat",
        "repeat_action_rate",
        "r_type",
        "r_att",
        "r_option",
        "r_price",
    )
    rewards = {key: [] for key in reward_keys}
    steps = []
    done = []
    max_steps = []
    overlong = []
    repeat_loop = []
    infrastructure_invalid = []
    reward_unverifiable = []
    terminal_utilities = []
    train_returns = []
    native_terminal_utilities = []
    behavior_penalties = []
    model_failures = []
    purchase_success = []
    reward_valid = []
    sampling_invalid = []
    shopper_questions = []
    shopper_rejections = []
    match_scores = []
    evidence_coverage = []
    partial_purchase = []
    bpo_metrics = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        for key in reward_keys:
            try:
                value = float(reward[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"shopping reward at index {index} is missing numeric {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"shopping reward {key} at index {index} is not finite")
            rewards[key].append(value)
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        overlong.append(float(bool(info.get("overlong"))))
        repeat_loop.append(float(info.get("reward_type") == "repeat_loop"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))
        terminal_utilities.append(
            float(reward.get("terminal_utility", reward["total"]))
        )
        train_return = reward.get("train_return", info.get("train_return"))
        if train_return is None:
            train_return = completion_aligned_train_return(
                reward, reward_type=info.get("reward_type")
            )
        train_returns.append(float(train_return))
        native_terminal_utilities.append(
            float(
                reward.get(
                    "native_terminal_utility",
                    reward.get("terminal_utility", reward["total"]),
                )
            )
        )
        behavior_penalties.append(float(reward.get("penalty_behavior", 0.0)))
        model_failures.append(float(bool(reward.get("model_failure", False))))
        purchase_success.append(
            float(bool(reward.get("purchase_success", reward["full"])))
        )
        reward_valid.append(
            float(bool(info.get("reward_valid", not info.get("reward_unverifiable"))))
        )
        sampling_invalid.append(
            float(
                bool(
                    reward.get(
                        "sampling_invalid",
                        info.get("infrastructure_invalid")
                        or info.get("reward_unverifiable"),
                    )
                )
            )
        )
        match_scores.append(float(reward.get("match_score", reward["r_att"])))
        evidence_coverage.append(
            float(reward.get("evidence_coverage", 0.0))
        )
        partial_purchase.append(
            float(info.get("reward_type") == "partial_alternative_purchase")
        )
        shopper_questions.append(float(info.get("shopper_questions", 0)))
        shopper_rejections.append(float(info.get("shopper_rejections", 0)))
        raw_bpo_metrics = info.get("bpo_metrics")
        if raw_bpo_metrics is not None:
            if not isinstance(raw_bpo_metrics, Mapping):
                raise ValueError("shopping bpo_metrics must be an object")
            normalized_bpo_metrics = {}
            for name, raw_value in raw_bpo_metrics.items():
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(f"shopping BPO metric {name} is not finite")
                normalized_bpo_metrics[str(name)] = value
            bpo_metrics.append(normalized_bpo_metrics)

    def mean(values):
        return sum(values) / len(values)

    metrics = {
        "reward/full_mean": mean(rewards["full"]),
        "reward/strict_mean": mean(rewards["strict"]),
        "reward/native_mean": mean(rewards["native"]),
        "reward/semantic_mean": mean(rewards["semantic"]),
        "reward/shaped_min": min(rewards["total"]),
        "reward/shaped_mean": mean(rewards["total"]),
        "reward/shaped_max": max(rewards["total"]),
        "reward/terminal_utility_min": min(terminal_utilities),
        "reward/terminal_utility_mean": mean(terminal_utilities),
        "reward/terminal_utility_max": max(terminal_utilities),
        "reward/native_terminal_utility_mean": mean(native_terminal_utilities),
        "reward/train_return_min": min(train_returns),
        "reward/train_return_mean": mean(train_returns),
        "reward/train_return_max": max(train_returns),
        "reward/behavior_penalty_mean": mean(behavior_penalties),
        "reward/model_failure_rate": mean(model_failures),
        "reward/purchase_success_rate": mean(purchase_success),
        "reward/valid_rate": mean(reward_valid),
        "reward/partial_purchase_rate": mean(partial_purchase),
        "reward/match_score_mean": mean(match_scores),
        "reward/evidence_coverage_mean": mean(evidence_coverage),
        "reward/efficiency_mean": mean(rewards["efficiency"]),
        "penalty/overlong_mean": mean(rewards["penalty_overlong"]),
        "penalty/unfinished_mean": mean(rewards["penalty_unfinished"]),
        "penalty/repeat_mean": mean(rewards["penalty_repeat"]),
        "component/r_type_mean": mean(rewards["r_type"]),
        "component/r_att_mean": mean(rewards["r_att"]),
        "component/r_option_mean": mean(rewards["r_option"]),
        "component/r_price_mean": mean(rewards["r_price"]),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/overlong_rate": mean(overlong),
        "trajectory/repeat_loop_rate": mean(repeat_loop),
        "trajectory/repeat_action_rate": mean(rewards["repeat_action_rate"]),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
        "trajectory/sampling_invalid_rate": mean(sampling_invalid),
        "trajectory/shopper_question_rate": mean(
            [float(value > 0) for value in shopper_questions]
        ),
        "trajectory/shopper_questions_mean": mean(shopper_questions),
        "trajectory/shopper_rejections_mean": mean(shopper_rejections),
    }
    metrics.update(swanlab_key_metrics(metrics))
    if bpo_metrics:
        if len(bpo_metrics) != len(shopping_infos):
            raise ValueError("shopping BPO metrics are missing from some trajectories")
        names = set(bpo_metrics[0])
        if any(set(values) != names for values in bpo_metrics[1:]):
            raise ValueError("shopping BPO metric keys are not trajectory-aligned")
        metrics.update(
            {
                name: mean([values[name] for values in bpo_metrics])
                for name in sorted(names)
            }
        )
    return metrics



def swanlab_key_metrics(
    shopping_metrics: Mapping[str, object],
) -> dict[str, float]:
    """Expose a compact, stable dashboard over the full validation diagnostics."""
    aliases = {
        "strict_success_rate": "reward/strict_mean",
        "purchase_success_rate": "reward/purchase_success_rate",
        "combined_completion_rate": "reward/purchase_success_rate",
        "mean_reward": "reward/shaped_mean",
        "terminal_utility_mean": "reward/terminal_utility_mean",
        "done_rate": "trajectory/done_rate",
        "average_steps": "trajectory/average_steps",
        "sampling_invalid_rate": "trajectory/sampling_invalid_rate",
        "infrastructure_invalid_rate": "trajectory/infrastructure_invalid_rate",
        "reward_unverifiable_rate": "trajectory/reward_unverifiable_rate",
        "shopper_question_rate": "trajectory/shopper_question_rate",
    }
    summary = {}
    for alias, source in aliases.items():
        try:
            value = float(shopping_metrics[source])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping validation metrics are missing numeric {source}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"shopping validation metric {source} is not finite")
        summary[f"summary/{alias}"] = value
    return summary


SWANLAB_DASHBOARD_SECTIONS = frozenset(
    {"validation", "sampling", "credit", "optimization", "runtime"}
)

_SWANLAB_DASHBOARD_ALIASES = {
    # Frozen validation: primary acceptance and validity.
    "val-shopping/summary/strict_success_rate": "validation/gold_purchase_success",
    "val-shopping/summary/purchase_success_rate": "validation/completion_success",
    "val-shopping/summary/mean_reward": "validation/reward_mean",
    "val-shopping/summary/terminal_utility_mean": "validation/terminal_utility_mean",
    "val-shopping/summary/done_rate": "validation/done_rate",
    "val-shopping/summary/average_steps": "validation/average_steps",
    "val-shopping/summary/sampling_invalid_rate": "validation/sampling_invalid_rate",
    "val-shopping/summary/infrastructure_invalid_rate": "validation/infrastructure_invalid_rate",
    "val-shopping/summary/reward_unverifiable_rate": "validation/reward_unverifiable_rate",
    "val-shopping/summary/shopper_question_rate": "validation/shopper_question_rate",
    "val-step0/cache_hit": "validation/step0_cache_hit",
    "val-step0/contract_verified": "validation/step0_contract_verified",
    # Sampling contract and accepted-budget progress.
    "bpo_batch/trees": "sampling/accepted_groups",
    "bpo_batch/sibling_returns": "sampling/accepted_returns",
    "bpo_batch/root_groups": "sampling/accepted_root_groups",
    "bpo_batch/local_groups": "sampling/accepted_local_groups",
    "bpo_batch/full_batch": "sampling/full_root_local_batch",
    "bpo_sampling/candidate_batches": "sampling/candidate_batches",
    "bpo_sampling/candidate_trees": "sampling/candidate_groups",
    "bpo_sampling/accepted_trees_pending": "sampling/accepted_groups_pending",
    "bpo_sampling/seconds_to_first_tree": "sampling/seconds_to_first_accept",
    "bpo_sampling/seconds_to_full_batch": "sampling/seconds_to_full_batch",
    "bpo_sampling/slow_full_batch_warning": "sampling/slow_batch_warning",
    "bpo_sampling/full_batch_timeout": "sampling/full_batch_timeout",
    "group/generated": "sampling/generated_groups",
    "group/trained": "sampling/trained_groups",
    "group/effective_ratio": "sampling/effective_group_ratio",
    "group/all_equal_ratio": "sampling/constant_return_ratio",
    "group/all_zero_utility_ratio": "sampling/all_zero_utility_ratio",
    "group/no_purchase_success_ratio": "sampling/no_completion_ratio",
    "group/all_purchase_success_ratio": "sampling/all_completion_ratio",
    "group/sampling_invalid": "sampling/invalid_groups",
    "group/completion_contrast": "sampling/completion_contrast_groups",
    "group/gold_contrast": "sampling/gold_contrast_groups",
    "group/failure_utility_contrast": "sampling/failure_utility_contrast_groups",
    "bpo_stage/product_count": "sampling/local_product_groups",
    "bpo_stage/option_count": "sampling/local_option_groups",
    "bpo_stage/search_recovery_count": "sampling/local_search_recovery_groups",
    "bpo_stage/fallback_count": "sampling/local_fallback_groups",
    "bpo_stage/unavailable_count": "sampling/local_stage_unavailable_groups",
    "bpo_diversity/unique_branch_actions_mean": "sampling/unique_branch_actions_mean",
    "bpo_diversity/unique_tool_sequences_mean": "sampling/unique_tool_sequences_mean",
    "carl_budget/accepted_groups_total": "sampling/accepted_groups_total",
    "carl_budget/accepted_returns_total": "sampling/accepted_returns_total",
    "carl_budget/target_returns": "sampling/target_returns",
    # Return signal and branching-credit coverage.
    "reward/train_return_mean": "credit/train_return_mean",
    "reward/train_return_min": "credit/train_return_min",
    "reward/train_return_max": "credit/train_return_max",
    "reward/native_terminal_utility_mean": "credit/native_terminal_utility_mean",
    "reward/strict_mean": "credit/train_rollout_gold_rate",
    "reward/purchase_success_rate": "credit/train_rollout_completion_rate",
    "reward/partial_purchase_rate": "credit/train_rollout_partial_rate",
    "reward/model_failure_rate": "credit/train_rollout_model_failure_rate",
    "reward/valid_rate": "credit/train_rollout_reward_valid_rate",
    "trajectory/repeat_loop_rate": "credit/train_rollout_repeat_loop_rate",
    "trajectory/max_steps_rate": "credit/train_rollout_max_steps_rate",
    "trajectory/overlong_rate": "credit/train_rollout_overlong_rate",
    "bpo_return/sibling_std_mean": "credit/sibling_return_std_mean",
    "bpo_return/sibling_range_mean": "credit/sibling_return_range_mean",
    "bpo_return/sibling_unique_count_mean": "credit/sibling_unique_returns_mean",
    "bpo_branch/entropy_mean": "credit/branch_entropy_mean",
    "bpo_branch/relative_position_mean": "credit/branch_relative_position_mean",
    "bpo_branch/backbone_actions_mean": "credit/backbone_actions_mean",
    "bpo_branch/prefix_steps_mean": "credit/prefix_steps_mean",
    "bpo_branch/prefix_shopper_calls_mean": "credit/prefix_shopper_calls_mean",
    "bpo_branch/prefix_environment_transitions_mean": "credit/prefix_environment_transitions_mean",
    # Optimizer state.
    "training/global_step": "optimization/global_step",
    "training/optimizer_updated": "optimization/optimizer_updated",
    "shopping_dynamic_sampling/skipped_update": "optimization/skipped_update",
    "shopping_dynamic_sampling/skipped_updates_total": "optimization/skipped_updates_total",
    "shopping_dynamic_sampling/consecutive_skips": "optimization/consecutive_skips",
}


def swanlab_dashboard_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """Project raw trainer metrics into five readable SwanLab sections.

    The raw dictionary remains authoritative for console output and local JSONL
    diagnostics. This compact projection is exclusively for the SwanLab backend.
    """
    dashboard = {}
    for raw_name, value in metrics.items():
        name = str(raw_name)
        alias = _SWANLAB_DASHBOARD_ALIASES.get(name)
        if alias is None and name.startswith("val-core/"):
            alias = "validation/condition." + name.removeprefix("val-core/").replace(
                "/", "."
            )
        if alias is None and name.startswith("actor/"):
            metric = name.removeprefix("actor/")
            if any(
                fragment in metric.lower()
                for fragment in (
                    "loss",
                    "grad_norm",
                    "learning_rate",
                    "lr",
                    "kl",
                    "clip",
                    "entropy",
                )
            ):
                alias = "optimization/" + metric.replace("/", ".")
        if alias is None and name.startswith(("timing_s/", "perf/", "response_length/")):
            alias = "runtime/" + name.replace("/", ".")
        if alias is None and name.startswith("bpo_cost/"):
            alias = "runtime/" + name.removeprefix("bpo_cost/")
        if alias is None and name in {
            "rollout/generated_response_tokens",
            "rollout/generated_response_tokens_total",
            "rollout/generated_total",
            "rollout/generated_total_cumulative",
        }:
            alias = "runtime/" + name.removeprefix("rollout/")
        if alias is not None:
            dashboard[alias] = value
    unknown_sections = {
        name.split("/", 1)[0] for name in dashboard if "/" in name
    }.difference(SWANLAB_DASHBOARD_SECTIONS)
    if unknown_sections:
        raise ValueError(
            "SwanLab dashboard projection produced unknown sections: "
            + ", ".join(sorted(unknown_sections))
        )
    return dashboard


def aggregate_bpo_tree_metrics(
    group_summaries: Mapping[Hashable, Mapping[str, object]],
    reward_groups: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Aggregate the branch position, diversity, and sibling-return signal."""
    if not group_summaries or not reward_groups:
        return {}
    expected_uids = {group["uid"] for group in reward_groups}
    if set(group_summaries) != expected_uids:
        raise ValueError("BPO tree diagnostics are not aligned with reward groups")
    summaries = [group_summaries[uid] for uid in expected_uids]
    if any(bool(summary.get("bpo_diagnostic_incomplete")) for summary in summaries):
        raise ValueError("BPO tree diagnostics are incomplete")

    def values(name):
        result = [float(summary[name]) for summary in summaries]
        if any(not math.isfinite(value) for value in result):
            raise ValueError(f"BPO tree diagnostic {name} is not finite")
        return result

    def mean(items):
        return sum(items) / len(items)

    sibling_stds = []
    sibling_ranges = []
    sibling_unique_counts = []
    for group in reward_groups:
        returns = [float(value) for value in group["rewards"]]
        if len(returns) < 2 or any(not math.isfinite(value) for value in returns):
            raise ValueError("BPO sibling returns must contain finite groups")
        center = mean(returns)
        sibling_stds.append(
            math.sqrt(mean([(value - center) ** 2 for value in returns]))
        )
        sibling_ranges.append(max(returns) - min(returns))
        sibling_unique_counts.append(float(len(set(returns))))

    return {
        "bpo_group/root_count": float(
            sum(summary.get("bpo_group_type") == "root" for summary in summaries)
        ),
        "bpo_group/local_count": float(
            sum(summary.get("bpo_group_type") == "local" for summary in summaries)
        ),
        "bpo_stage/product_count": float(
            sum(summary.get("bpo_local_stage") == "product" for summary in summaries)
        ),
        "bpo_stage/option_count": float(
            sum(summary.get("bpo_local_stage") == "option" for summary in summaries)
        ),
        "bpo_stage/search_recovery_count": float(
            sum(
                summary.get("bpo_local_stage") == "search_recovery"
                for summary in summaries
            )
        ),
        "bpo_stage/fallback_count": float(
            sum(bool(summary.get("bpo_local_stage_fallback")) for summary in summaries)
        ),
        "bpo_stage/unavailable_count": float(
            sum(
                bool(summary.get("bpo_local_stage_unavailable"))
                for summary in summaries
            )
        ),
        "bpo_branch/relative_position_mean": mean(
            values("bpo_branch_relative_position")
        ),
        "bpo_branch/entropy_mean": mean(values("bpo_branch_entropy")),
        "bpo_branch/backbone_actions_mean": mean(
            values("bpo_backbone_action_count")
        ),
        "bpo_branch/prefix_steps_mean": mean(values("bpo_branch_prefix_steps")),
        "bpo_branch/prefix_shopper_calls_mean": mean(
            values("bpo_branch_prefix_shopper_calls")
        ),
        "bpo_branch/prefix_environment_transitions_mean": mean(
            values("bpo_branch_prefix_environment_transitions")
        ),
        "bpo_diversity/unique_branch_actions_mean": mean(
            values("bpo_unique_branch_action_count")
        ),
        "bpo_diversity/unique_tool_sequences_mean": mean(
            values("bpo_unique_tool_sequence_count")
        ),
        "bpo_return/sibling_std_mean": mean(sibling_stds),
        "bpo_return/sibling_range_mean": mean(sibling_ranges),
        "bpo_return/sibling_unique_count_mean": mean(sibling_unique_counts),
    }
def extract_shopping_group_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[float], list[bool], list[bool], list[tuple[str, ...]]]:
    """Return native utility, completion metrics, and explicit invalid reasons."""
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    invalid_reasons = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            terminal_utility = float(info["reward"]["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping extra field at index {index} is missing terminal_utility"
            ) from exc
        if not math.isfinite(terminal_utility):
            raise ValueError(
                f"shopping terminal_utility at index {index} is not finite"
            )
        raw_purchase_success = info["reward"].get("purchase_success")
        if not isinstance(raw_purchase_success, (bool, int, float)):
            raise ValueError(
                f"shopping extra field at index {index} is missing purchase_success"
            )
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        reasons = []
        if bool(info["infrastructure_invalid"]):
            reasons.append("infrastructure_invalid")
        if bool(info.get("reward_unverifiable")):
            reasons.append("reward_unverifiable")
        if bool(info.get("overlong")):
            reasons.append("overlong")
        reward_sampling_invalid = bool(
            info["reward"].get("sampling_invalid", False)
        )
        if reward_sampling_invalid and not reasons:
            reasons.append("reward_sampling_invalid")
        terminal_utilities.append(terminal_utility)
        purchase_success.append(bool(raw_purchase_success))
        sampling_invalid.append(bool(reasons))
        invalid_reasons.append(tuple(reasons))
    return (
        terminal_utilities,
        purchase_success,
        sampling_invalid,
        invalid_reasons,
    )


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    terminal_utilities: Sequence[float] | None = None,
    purchase_success: Sequence[bool] | None = None,
    group_types: Sequence[str] | None = None,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with non-constant reward.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    optional_sequences = {
        "terminal_utilities": terminal_utilities,
        "purchase_success": purchase_success,
        "group_types": group_types,
        "sampling_invalid": sampling_invalid,
        "sampling_invalid_reasons": sampling_invalid_reasons,
    }
    for name, values in optional_sequences.items():
        if values is not None and len(values) != len(uids):
            raise ValueError(f"{name} must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    # `seq_rewards` is the actual score sent to PPO.  CARL-BPO deliberately
    # filters on completion-aligned train returns, not the native diagnostic
    # utility supplied separately by the environment.
    native_values = (
        terminal_utilities if terminal_utilities is not None else seq_rewards
    )
    success_values = (
        purchase_success if purchase_success is not None else [False] * len(uids)
    )
    invalid_values = (
        sampling_invalid if sampling_invalid is not None else [False] * len(uids)
    )
    reason_values = (
        sampling_invalid_reasons
        if sampling_invalid_reasons is not None
        else [()] * len(uids)
    )
    group_type_values = (
        [str(value) for value in group_types]
        if group_types is not None
        else ["unknown"] * len(uids)
    )
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (
        uid,
        raw_reward,
        raw_utility,
        raw_success,
        raw_group_type,
        raw_invalid,
        raw_reasons,
    ) in enumerate(
        zip(
            uids,
            seq_rewards,
            native_values,
            success_values,
            group_type_values,
            invalid_values,
            reason_values,
            strict=True,
        )
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        utility = float(raw_utility)
        if not math.isfinite(utility):
            raise ValueError(
                f"terminal_utility at index {index} is not finite: {raw_utility!r}"
            )

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "train_returns": [],
                "terminal_utilities": [],
                "purchase_success": [],
                "group_types": [],
                "sampling_invalid": [],
                "sampling_invalid_reasons": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["train_returns"].append(reward)
        group["terminal_utilities"].append(utility)
        group["purchase_success"].append(bool(raw_success))
        group["group_types"].append(str(raw_group_type))
        group["sampling_invalid"].append(bool(raw_invalid))
        group["sampling_invalid_reasons"].extend(str(reason) for reason in raw_reasons)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        train_returns = group["train_returns"]
        native_utilities = group["terminal_utilities"]
        group_type_set = set(group["group_types"])
        if len(group_type_set) != 1:
            raise ValueError(f"group {uid!r} has inconsistent CARL group types")
        group_type = next(iter(group_type_set))
        utility_min = min(train_returns)
        utility_max = max(train_returns)
        utility_varying = utility_max - utility_min > tolerance
        has_sampling_invalid = any(group["sampling_invalid"])
        reasons = tuple(sorted(set(group["sampling_invalid_reasons"])))
        all_success = all(group["purchase_success"])
        any_success = any(group["purchase_success"])
        if any_success and not all_success:
            contrast_type = "completion_contrast"
        elif all_success and utility_max - utility_min > tolerance:
            contrast_type = "gold_contrast"
        elif utility_varying:
            contrast_type = "failure_utility_contrast"
        else:
            contrast_type = "constant"
        if has_sampling_invalid:
            drop_reason = "sampling_invalid"
        elif not utility_varying:
            drop_reason = "constant_reward"
        else:
            drop_reason = None
        keep = drop_reason is None
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(group["rewards"]),
                "terminal_utilities": tuple(native_utilities),
                "train_returns": tuple(train_returns),
                "purchase_success": tuple(group["purchase_success"]),
                "group_type": group_type,
                "utility_min": utility_min,
                "utility_max": utility_max,
                "reward_varying": utility_varying,
                "sampling_invalid": has_sampling_invalid,
                "sampling_invalid_reasons": reasons,
                "drop_reason": drop_reason,
                "contrast_type": contrast_type,
                "kept": keep,
            }
        )

    priority = {
        "completion_contrast": 0,
        "gold_contrast": 1,
        "failure_utility_contrast": 2,
        "constant": 3,
    }
    kept_uids.sort(
        key=lambda uid: (
            priority[next(group["contrast_type"] for group in groups if group["uid"] == uid)],
            list(grouped).index(uid),
        )
    )
    kept_uid_set = set(kept_uids)
    trajectory_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "all_zero_utility_group_count": sum(
            max(abs(value) for value in group["terminal_utilities"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            all(group["purchase_success"])
            for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(group["purchase_success"]) for group in groups
        ),
        "completion_contrast_group_count": sum(
            group["contrast_type"] == "completion_contrast" for group in groups
        ),
        "gold_contrast_group_count": sum(
            group["contrast_type"] == "gold_contrast" for group in groups
        ),
        "failure_utility_contrast_group_count": sum(
            group["contrast_type"] == "failure_utility_contrast" for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "sampling_invalid_reason_counts": {
            reason: sum(
                reason in group["sampling_invalid_reasons"] for group in groups
            )
            for reason in sorted(
                {
                    reason
                    for group in groups
                    for reason in group["sampling_invalid_reasons"]
                }
            )
        },
        # Compatibility aliases for existing monitoring code.
        "infrastructure_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats
