"""Versioned contracts for this repository's offline evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy

CONTRACT_VERSION = "shopping-trajectory-evaluation-v2"
RUBRIC_SCHEMA_VERSION = "shopping-requirement-rubric-v2"
JUDGE_SCHEMA_VERSION = "shopping-trajectory-judge-v2"

JUDGE_DIMENSIONS = (
    "search_strategy",
    "candidate_utilization",
    "evidence_verification",
    "decision_quality",
    "termination_efficiency",
)
RUBRIC_STATUSES = frozenset(
    {"satisfied", "violated", "unknown", "not_applicable"}
)
RUBRIC_HARDNESS = frozenset({"hard", "soft", "needs_review"})
JUDGE_STATUSES = frozenset({"valid", "invalid", "not_judged"})
CLARIFICATION_STATUSES = frozenset(
    {"effective", "ineffective", "unnecessary", "not_applicable", "unknown"}
)
ERROR_TAXONOMY = frozenset(
    {
        "search_core_requirement_missed",
        "search_ineffective_reformulation",
        "search_mechanical_repeat",
        "promising_candidate_ignored",
        "candidate_comparison_insufficient",
        "candidate_overexploration",
        "critical_evidence_missing",
        "unreliable_evidence_used",
        "final_price_unverified",
        "wrong_category",
        "budget_violation",
        "requirement_violation",
        "wrong_option",
        "premature_purchase",
        "premature_abstain",
        "unjustified_abstain",
        "overexploration_after_convergence",
        "repeat_loop",
        "max_steps_exhaustion",
        "illegal_action",
        "context_loss",
        "clarification_needed_but_not_asked",
        "clarification_unnecessary",
        "clarification_question_ungrounded",
        "clarification_answer_not_used",
        "reward_rubric_disagreement",
        "infrastructure_invalid",
        "other",
    }
)


class ContractValidationError(ValueError):
    """Raised when a cached evaluator artifact violates its frozen schema."""


def _mapping(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list:
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} must be a list")
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise ContractValidationError(f"{path} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{path} must be an integer") from exc
    if value != integer:
        raise ContractValidationError(f"{path} must be an integer")
    return integer


def _unique_nonempty_strings(values: object, path: str) -> list[str]:
    result = []
    for index, value in enumerate(_list(values, path)):
        text = _nonempty_text(value, f"{path}[{index}]")
        if text in result:
            raise ContractValidationError(f"{path} contains duplicate {text!r}")
        result.append(text)
    return result


def validate_rubric_bundle(
    bundle: object,
    *,
    expected_task_id: int | None = None,
) -> dict:
    """Validate and defensively copy one frozen task-level Rubric bundle."""

    payload = _mapping(bundle, "rubric_bundle")
    if payload.get("schema_version") != RUBRIC_SCHEMA_VERSION:
        raise ContractValidationError(
            "rubric_bundle.schema_version must be "
            f"{RUBRIC_SCHEMA_VERSION!r}"
        )
    task_id = _integer(payload.get("task_id"), "rubric_bundle.task_id")
    if expected_task_id is not None and task_id != int(expected_task_id):
        raise ContractValidationError(
            f"rubric task_id {task_id} does not match {int(expected_task_id)}"
        )
    query = _nonempty_text(payload.get("query"), "rubric_bundle.query")
    _nonempty_text(
        payload.get("rubric_version"), "rubric_bundle.rubric_version"
    )
    generation = _mapping(
        payload.get("generation"), "rubric_bundle.generation"
    )
    for field in (
        "curator_version",
        "curator_model",
        "curator_prompt_version",
        "task_data_hash",
        "query_hash",
    ):
        _nonempty_text(generation.get(field), f"rubric_bundle.generation.{field}")

    rubric_ids = set()
    rubrics = _list(payload.get("rubrics"), "rubric_bundle.rubrics")
    if not rubrics:
        raise ContractValidationError(
            "rubric_bundle.rubrics must contain at least one Query-backed requirement"
        )
    for index, item_value in enumerate(rubrics):
        path = f"rubric_bundle.rubrics[{index}]"
        item = _mapping(item_value, path)
        rubric_id = _nonempty_text(item.get("rubric_id"), f"{path}.rubric_id")
        if rubric_id in rubric_ids:
            raise ContractValidationError(f"duplicate rubric_id {rubric_id!r}")
        rubric_ids.add(rubric_id)
        if item.get("rubric_source") != "query_only_llm":
            raise ContractValidationError(
                f"{path}.rubric_source must be 'query_only_llm'"
            )
        _nonempty_text(
            item.get("description"), f"{path}.description"
        )
        _nonempty_text(
            item.get("acceptance_criteria"), f"{path}.acceptance_criteria"
        )
        hardness = _nonempty_text(item.get("hardness"), f"{path}.hardness")
        if hardness not in RUBRIC_HARDNESS:
            raise ContractValidationError(
                f"{path}.hardness must be one of {sorted(RUBRIC_HARDNESS)}"
            )
        _nonempty_text(
            item.get("hardness_source"), f"{path}.hardness_source"
        )
        if item.get("data_sources") != ["query"]:
            raise ContractValidationError(
                f"{path}.data_sources must be ['query'] for Query-only Rubrics"
            )
        _nonempty_text(
            item.get("selection_reason"), f"{path}.selection_reason"
        )
        anchor_ids = _unique_nonempty_strings(
            item.get("query_anchor_ids"), f"{path}.query_anchor_ids"
        )
        if not anchor_ids:
            raise ContractValidationError(
                f"{path}.query_anchor_ids must contain at least one anchor"
            )
        spans = _list(item.get("query_spans"), f"{path}.query_spans")
        if not spans:
            raise ContractValidationError(
                f"{path}.query_spans must contain direct Query evidence"
            )
        for span_index, span_value in enumerate(spans):
            span_path = f"{path}.query_spans[{span_index}]"
            span = _mapping(span_value, span_path)
            _nonempty_text(span.get("text"), f"{span_path}.text")
            start = _integer(span.get("start"), f"{span_path}.start")
            end = _integer(span.get("end"), f"{span_path}.end")
            if start < 0 or end <= start:
                raise ContractValidationError(
                    f"{span_path} must satisfy 0 <= start < end"
                )
            if end > len(query) or query[start:end] != span["text"]:
                raise ContractValidationError(
                    f"{span_path} does not match rubric_bundle.query"
                )

    return deepcopy(dict(payload))


def validate_curator_response(
    response: object,
    *,
    anchor_ids: Iterable[str],
) -> dict:
    """Validate an LLM-generated requirement list against Query anchor IDs."""

    payload = _mapping(response, "curator_response")
    allowed_anchor_ids = {str(anchor_id) for anchor_id in anchor_ids}
    requirements = _list(
        payload.get("requirements"), "curator_response.requirements"
    )
    if not requirements:
        raise ContractValidationError(
            "curator_response.requirements must contain at least one requirement"
        )
    seen = set()
    for index, item_value in enumerate(requirements):
        path = f"curator_response.requirements[{index}]"
        item = _mapping(item_value, path)
        description = _nonempty_text(item.get("description"), f"{path}.description")
        selected_anchor_ids = _unique_nonempty_strings(
            item.get("query_anchor_ids"), f"{path}.query_anchor_ids"
        )
        if not selected_anchor_ids:
            raise ContractValidationError(
                f"{path}.query_anchor_ids must contain at least one anchor"
            )
        unknown_anchor_ids = sorted(set(selected_anchor_ids) - allowed_anchor_ids)
        if unknown_anchor_ids:
            raise ContractValidationError(
                f"{path}.query_anchor_ids references unknown anchors: "
                f"{unknown_anchor_ids}"
            )
        identity = (description, tuple(selected_anchor_ids))
        if identity in seen:
            raise ContractValidationError(
                f"{path} repeats a requirement with the same description and quote"
            )
        seen.add(identity)
        _nonempty_text(
            item.get("acceptance_criteria"), f"{path}.acceptance_criteria"
        )
        hardness = _nonempty_text(item.get("hardness"), f"{path}.hardness")
        if hardness not in RUBRIC_HARDNESS:
            raise ContractValidationError(
                f"{path}.hardness must be one of {sorted(RUBRIC_HARDNESS)}"
            )
        _nonempty_text(
            item.get("selection_reason"), f"{path}.selection_reason"
        )
    return deepcopy(dict(payload))


def validate_judge_result(
    result: object,
    *,
    rubric_ids: Iterable[str],
    expected_task_id: int | None = None,
    expected_trajectory_id: str | None = None,
    allowed_event_ids: Iterable[str] | None = None,
) -> dict:
    """Validate one Pro Judge result without manufacturing missing scores."""

    payload = _mapping(result, "judge_result")
    if payload.get("schema_version") != JUDGE_SCHEMA_VERSION:
        raise ContractValidationError(
            f"judge_result.schema_version must be {JUDGE_SCHEMA_VERSION!r}"
        )
    for forbidden in ("total_score", "overall_score", "weighted_total"):
        if forbidden in payload:
            raise ContractValidationError(
                f"judge_result must not contain {forbidden}"
            )
    task_id = _integer(payload.get("task_id"), "judge_result.task_id")
    trajectory_id = _nonempty_text(
        payload.get("trajectory_id"), "judge_result.trajectory_id"
    )
    if expected_task_id is not None and task_id != int(expected_task_id):
        raise ContractValidationError(
            f"judge task_id {task_id} does not match {int(expected_task_id)}"
        )
    if (
        expected_trajectory_id is not None
        and trajectory_id != str(expected_trajectory_id)
    ):
        raise ContractValidationError(
            "judge trajectory_id does not match the evaluated trajectory"
        )
    judge_status = _nonempty_text(
        payload.get("judge_status"), "judge_result.judge_status"
    )
    if judge_status not in JUDGE_STATUSES:
        raise ContractValidationError(
            f"judge_status must be one of {sorted(JUDGE_STATUSES)}"
        )
    allowed_events = (
        {str(event_id) for event_id in allowed_event_ids}
        if allowed_event_ids is not None
        else None
    )

    def validate_evidence(value: object, path: str) -> list[str]:
        event_ids = _unique_nonempty_strings(value, path)
        if allowed_events is not None:
            unknown = sorted(set(event_ids) - allowed_events)
            if unknown:
                raise ContractValidationError(
                    f"{path} references unknown event IDs: {unknown}"
                )
        return event_ids

    expected_rubrics = {str(rubric_id) for rubric_id in rubric_ids}
    assessments = _list(
        payload.get("rubric_assessments"),
        "judge_result.rubric_assessments",
    )
    dimensions = _mapping(
        payload.get("dimension_scores"), "judge_result.dimension_scores"
    )
    if judge_status == "valid":
        seen_rubrics = set()
        for index, assessment_value in enumerate(assessments):
            path = f"judge_result.rubric_assessments[{index}]"
            assessment = _mapping(assessment_value, path)
            rubric_id = _nonempty_text(
                assessment.get("rubric_id"), f"{path}.rubric_id"
            )
            if rubric_id not in expected_rubrics:
                raise ContractValidationError(
                    f"{path} references unknown rubric_id {rubric_id!r}"
                )
            if rubric_id in seen_rubrics:
                raise ContractValidationError(
                    f"{path} repeats rubric_id {rubric_id!r}"
                )
            seen_rubrics.add(rubric_id)
            status = _nonempty_text(
                assessment.get("status"), f"{path}.status"
            )
            if status not in RUBRIC_STATUSES:
                raise ContractValidationError(
                    f"{path}.status must be one of {sorted(RUBRIC_STATUSES)}"
                )
            _nonempty_text(assessment.get("reason"), f"{path}.reason")
            validate_evidence(
                assessment.get("evidence_event_ids"),
                f"{path}.evidence_event_ids",
            )
        if seen_rubrics != expected_rubrics:
            missing = sorted(expected_rubrics - seen_rubrics)
            raise ContractValidationError(
                f"judge_result is missing rubric assessments: {missing}"
            )
        if set(dimensions) != set(JUDGE_DIMENSIONS):
            raise ContractValidationError(
                "dimension_scores must contain exactly "
                f"{list(JUDGE_DIMENSIONS)}"
            )
        for name in JUDGE_DIMENSIONS:
            path = f"judge_result.dimension_scores.{name}"
            score_payload = _mapping(dimensions[name], path)
            score = _integer(score_payload.get("score"), f"{path}.score")
            if score not in {0, 1, 2}:
                raise ContractValidationError(f"{path}.score must be 0, 1, or 2")
            _nonempty_text(score_payload.get("reason"), f"{path}.reason")
            validate_evidence(
                score_payload.get("evidence_event_ids"),
                f"{path}.evidence_event_ids",
            )
        clarification = _mapping(
            payload.get("clarification_assessment"),
            "judge_result.clarification_assessment",
        )
        clarification_status = _nonempty_text(
            clarification.get("status"),
            "judge_result.clarification_assessment.status",
        )
        if clarification_status not in CLARIFICATION_STATUSES:
            raise ContractValidationError(
                "judge_result.clarification_assessment.status must be one of "
                f"{sorted(CLARIFICATION_STATUSES)}"
            )
        _nonempty_text(
            clarification.get("reason"),
            "judge_result.clarification_assessment.reason",
        )
        validate_evidence(
            clarification.get("evidence_event_ids"),
            "judge_result.clarification_assessment.evidence_event_ids",
        )
    else:
        if assessments or dimensions:
            raise ContractValidationError(
                "invalid/not_judged results must not contain inferred assessments "
                "or dimension scores"
            )
        clarification = payload.get("clarification_assessment")
        if clarification not in ({}, None):
            raise ContractValidationError(
                "invalid/not_judged results must not contain a clarification "
                "assessment"
            )

    errors = _mapping(payload.get("errors"), "judge_result.errors")
    primary = errors.get("primary")
    if primary is not None and not isinstance(primary, str):
        raise ContractValidationError(
            "judge_result.errors.primary must be a string or null"
        )
    if primary is not None and primary not in ERROR_TAXONOMY:
        raise ContractValidationError(
            "judge_result.errors.primary is not in the frozen taxonomy"
        )
    secondary = _unique_nonempty_strings(
        errors.get("secondary"), "judge_result.errors.secondary"
    )
    if len(secondary) > 2:
        raise ContractValidationError(
            "judge_result.errors.secondary must contain at most two values"
        )
    unknown_secondary = sorted(set(secondary) - ERROR_TAXONOMY)
    if unknown_secondary:
        raise ContractValidationError(
            "judge_result.errors.secondary contains unknown taxonomy values: "
            f"{unknown_secondary}"
        )
    if primary is not None and primary in secondary:
        raise ContractValidationError(
            "judge_result.errors.primary must not be repeated in secondary"
        )
    if primary is None and secondary:
        raise ContractValidationError(
            "judge_result.errors.secondary must be empty when primary is null"
        )
    validate_evidence(
        errors.get("evidence_event_ids"),
        "judge_result.errors.evidence_event_ids",
    )
    diagnosis = payload.get("overall_diagnosis")
    if not isinstance(diagnosis, str):
        raise ContractValidationError(
            "judge_result.overall_diagnosis must be a string"
        )
    return deepcopy(dict(payload))


def rubric_ids(bundle: Mapping) -> list[str]:
    """Return rubric IDs after validating the bundle."""

    validated = validate_rubric_bundle(bundle)
    return [item["rubric_id"] for item in validated["rubrics"]]
