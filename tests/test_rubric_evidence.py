import json

import pytest

from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_SYSTEM_PROMPT,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (
    GENERIC_ACCEPTANCE_CRITERIA,
    build_task_facts,
    materialize_rubric_bundle,
    stable_hash,
)
from scripts.freeze_multiturn_rubrics import _curate


def _facts():
    return build_task_facts(task_id=7, query="预算100元以内，优先木制手串")


def _candidates():
    bundle = {
        "schema_version": "shopping-rubric-candidates-v3",
        "extractor_version": "test",
        "reward_version": "shopsimulator-reward-v4",
        "task_id": 7,
        "query": _facts()["query"],
        "candidates": [
            {
                "candidate_id": "c0001", "constraint_type": "price",
                "field_path": "purchase.price", "operator": "lte",
                "expected_value": {"upper": 100}, "data_sources": ["instruction"],
            },
            {
                "candidate_id": "c0002", "constraint_type": "option",
                "field_path": "purchase.options.color", "operator": "eq",
                "expected_value": "木制", "data_sources": ["instruction.option"],
            },
        ],
    }
    bundle["candidate_hash"] = stable_hash(bundle)
    return bundle


def _response(*, anchor_ids=None, candidate_ids=None):
    return {
        "requirements": [{
            "candidate_ids": candidate_ids or ["c0001"],
            "description": "价格不超过100元",
            "hardness": "hard",
            "query_anchor_ids": anchor_ids or ["q0001"],
            "selection_reason": "用户明确给出了预算上限",
        }],
    }


def _materialize(response=None):
    return materialize_rubric_bundle(
        task_facts=_facts(), candidates=_candidates(),
        curator_response=response or _response(), curator_model="curator",
        curator_prompt_version="test", rubric_version="test",
    )


def test_candidate_constrained_rubric_preserves_query_evidence_and_provenance():
    bundle = _materialize()
    rubric = bundle["rubrics"][0]
    assert rubric["rubric_source"] == "reward_v4_candidates_llm"
    assert rubric["candidate_ids"] == ["c0001"]
    assert rubric["candidate_semantics"][0]["operator"] == "lte"
    assert rubric["candidate_semantics"][0]["expected_value"] == {"upper": 100}
    assert rubric["query_spans"] == [{"text": "预算100元以内", "start": 0, "end": 8}]
    assert rubric["acceptance_criteria"] == GENERIC_ACCEPTANCE_CRITERIA
    assert "query" in rubric["data_sources"]
    assert bundle["review"] == {
        "status": "auto_drafted",
    }


def test_curator_request_contains_candidates_and_query_anchors():
    messages = build_rubric_curator_messages(
        task_id=7, query="预算100元以内", candidates=_candidates()["candidates"]
    )
    payload = json.loads(messages[1]["content"])
    assert payload["query_evidence_anchors"][0]["text"] == "预算100元以内"
    assert [row["candidate_id"] for row in payload["candidates"]] == ["c0001", "c0002"]


def test_prompt_forbids_free_generation_and_uses_frozen_candidates():
    assert "只能选择 candidate_id 已存在的候选" in RUBRIC_CURATOR_SYSTEM_PROMPT
    assert "没有候选支持的要求不输出" in RUBRIC_CURATOR_SYSTEM_PROMPT
    assert "同一个组合 option 候选可以支持拆开的多条原子要求" in RUBRIC_CURATOR_SYSTEM_PROMPT
    assert "冷泡也行" in RUBRIC_CURATOR_SYSTEM_PROMPT


@pytest.mark.parametrize("response, message", [
    (_response(candidate_ids=["invented"]), "unknown candidates"),
    (_response(anchor_ids=["q9999"]), "unknown anchors"),
    ({"requirements": []}, "at least one"),
])
def test_curator_rejects_invalid_candidate_or_anchor_references(response, message):
    with pytest.raises(ContractValidationError, match=message):
        _materialize(response)


def test_atomic_requirements_may_share_candidate_and_anchor():
    response = _response(candidate_ids=["c0002"])
    response["requirements"].append({
        "candidate_ids": ["c0002"], "description": "购买前核验规格约束",
        "hardness": "hard", "query_anchor_ids": ["q0001"],
        "selection_reason": "同一组合候选支持独立要求",
    })
    bundle = _materialize(response)
    assert [row["candidate_ids"] for row in bundle["rubrics"]] == [["c0002"], ["c0002"]]


def test_reference_flow_allows_candidate_reuse_across_requirements():
    response = _response()
    response["requirements"].append({
        "candidate_ids": ["c0001"], "description": "另一个价格解释",
        "hardness": "hard", "query_anchor_ids": ["q0002"],
        "selection_reason": "错误地重复候选",
    })
    bundle = _materialize(response)
    assert [row["candidate_ids"] for row in bundle["rubrics"]] == [
        ["c0001"], ["c0001"]
    ]


def test_auto_draft_is_valid_without_a_separate_approval_manifest():
    bundle = _materialize()
    assert bundle["review"]["status"] == "auto_drafted"
    validate_rubric_bundle(bundle)


@pytest.mark.parametrize("field", ["query_hash", "task_data_hash"])
def test_rubric_rejects_tampered_query_fact_hash(field):
    bundle = _materialize()
    bundle["generation"][field] = "0" * 64
    with pytest.raises(ContractValidationError, match="does not match"):
        validate_rubric_bundle(bundle)


def test_curator_records_invalid_response_before_schema_retry():
    class FakeClient:
        model = "fake-curator"
        def __init__(self):
            self.responses = [_response(candidate_ids=["invented"]), _response()]
        def complete_json(self, messages):
            return {"result": self.responses.pop(0), "metadata": {"provider_model": self.model}}

    requests, attempts = [], []
    _, bundle, request_ids = _curate(
        FakeClient(), _facts(), _candidates(), 1, run_plan_sha256="plan",
        on_request=requests.append, on_attempt=attempts.append,
    )
    assert len(request_ids) == 2
    assert [row["validation_status"] for row in attempts] == ["invalid", "valid"]
    assert bundle["review"]["status"] == "auto_drafted"


def test_candidate_hash_prevents_post_generation_mutation():
    candidates = _candidates()
    candidates["candidates"][0]["data_sources"] = ["tampered"]
    with pytest.raises(ContractValidationError, match="candidate_hash"):
        materialize_rubric_bundle(
            task_facts=_facts(), candidates=candidates,
            curator_response=_response(), curator_model="curator",
            curator_prompt_version="test", rubric_version="test",
        )
