import pytest

from shopping_grpo.evaluation.contracts import ContractValidationError
from shopping_grpo.evaluation.rubric import (
    RUBRIC_CANDIDATE_VERSION,
    RUBRIC_EXTRACTOR_VERSION,
    TASK_FACTS_VERSION,
    materialize_rubric_bundle,
    stable_hash,
)


def _facts_and_candidates():
    query = "预算100元以内"
    facts = {
        "schema_version": TASK_FACTS_VERSION,
        "task_id": 7,
        "query": query,
    }
    facts["task_data_hash"] = stable_hash(facts)
    facts["query_hash"] = stable_hash(query)
    candidates = {
        "schema_version": RUBRIC_CANDIDATE_VERSION,
        "extractor_version": RUBRIC_EXTRACTOR_VERSION,
        "task_id": 7,
        "query": query,
        "task_data_hash": facts["task_data_hash"],
        "query_hash": facts["query_hash"],
        "candidates": [
            {
                "candidate_id": "c0001",
                "constraint_type": "budget_upper",
                "field_path": "purchase.price",
                "operator": "lte",
                "expected_value": {"value": 100, "currency": "CNY"},
                "hardness_source": "explicit_upper_budget_rule",
                "query_spans": [
                    {"text": "100元以内", "start": 2, "end": 8}
                ],
                "data_sources": ["query"],
            }
        ],
    }
    return facts, candidates


def _response(quote):
    return {
        "selected_constraints": [
            {
                "candidate_id": "c0001",
                "description": "价格不超过100元",
                "hardness": "hard",
                "query_quote": quote,
                "selection_reason": "用户明确给出了预算上限",
            }
        ],
        "unmapped_query_requirements": [],
    }


def test_materialized_rubric_requires_exact_query_evidence():
    facts, candidates = _facts_and_candidates()
    bundle = materialize_rubric_bundle(
        task_facts=facts,
        candidates=candidates,
        curator_response=_response("100元以内"),
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )

    assert bundle["rubrics"][0]["query_spans"] == [
        {"text": "100元以内", "start": 2, "end": 8}
    ]


@pytest.mark.parametrize("quote", ["", "预算200元以内", "预算"])
def test_materialized_rubric_rejects_missing_or_unrelated_query_evidence(quote):
    facts, candidates = _facts_and_candidates()

    with pytest.raises(ContractValidationError):
        materialize_rubric_bundle(
            task_facts=facts,
            candidates=candidates,
            curator_response=_response(quote),
            curator_model="curator",
            curator_prompt_version="test",
            rubric_version="test",
        )


def test_materialized_rubric_fails_closed_on_candidate_coverage_gap():
    facts, candidates = _facts_and_candidates()
    response = _response("100元以内")
    response["unmapped_query_requirements"] = [
        {"description": "颜色要求没有可选候选", "query_quote": "预算"}
    ]

    with pytest.raises(ContractValidationError, match="coverage gaps"):
        materialize_rubric_bundle(
            task_facts=facts,
            candidates=candidates,
            curator_response=response,
            curator_model="curator",
            curator_prompt_version="test",
            rubric_version="test",
        )


def test_materialized_rubric_rejects_an_empty_selection():
    facts, candidates = _facts_and_candidates()

    with pytest.raises(ContractValidationError, match="at least one"):
        materialize_rubric_bundle(
            task_facts=facts,
            candidates=candidates,
            curator_response={
                "selected_constraints": [],
                "unmapped_query_requirements": [],
            },
            curator_model="curator",
            curator_prompt_version="test",
            rubric_version="test",
        )


def test_semantic_candidate_without_lexical_anchor_uses_curator_quote():
    facts, candidates = _facts_and_candidates()
    candidates["candidates"][0]["query_spans"] = []
    bundle = materialize_rubric_bundle(
        task_facts=facts,
        candidates=candidates,
        curator_response=_response("预算"),
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )

    assert bundle["rubrics"][0]["query_spans"] == [
        {"text": "预算", "start": 0, "end": 2}
    ]
