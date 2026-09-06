import json

import pytest

from shopping_grpo.evaluation.contracts import ContractValidationError
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_SYSTEM_PROMPT,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (
    GENERIC_ACCEPTANCE_CRITERIA,
    build_task_facts,
    materialize_rubric_bundle,
)
from scripts.approve_multiturn_rubrics import approve_bundles
from scripts.freeze_multiturn_rubrics import _curate


def _facts():
    return build_task_facts(task_id=7, query="预算100元以内，优先木制手串")


def _response(*, anchor_ids=None):
    if anchor_ids is None:
        anchor_ids = ["q0001"]
    return {
        "requirements": [
            {
                "description": "价格不超过100元",
                "hardness": "hard",
                "query_anchor_ids": anchor_ids,
                "selection_reason": "用户明确给出了预算上限",
            }
        ]
    }


def test_query_only_rubric_has_exact_query_evidence_and_no_private_fields():
    facts = _facts()
    bundle = materialize_rubric_bundle(
        task_facts=facts,
        curator_response=_response(),
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )

    rubric = bundle["rubrics"][0]
    assert rubric["rubric_source"] == "query_only_llm"
    assert rubric["data_sources"] == ["query"]
    assert rubric["query_spans"] == [
        {"text": "预算100元以内", "start": 0, "end": 8}
    ]
    assert rubric["query_anchor_ids"] == ["q0001"]
    assert rubric["acceptance_criteria"] == GENERIC_ACCEPTANCE_CRITERIA
    assert bundle["review"]["status"] == "auto_drafted"
    assert "candidate_id" not in rubric
    assert "expected_value" not in rubric
    assert "target_product" not in facts
    assert "reward_features" not in facts


def test_curator_request_contains_only_task_id_and_query():
    messages = build_rubric_curator_messages(
        task_id=7,
        query="预算100元以内",
    )

    payload = json.loads(messages[1]["content"])
    assert payload["task_id"] == 7
    assert payload["query"] == "预算100元以内"
    assert payload["query_evidence_anchors"] == [
        {"anchor_id": "q0001", "text": "预算100元以内", "start": 0, "end": 8}
    ]
    assert "candidates" not in payload


def test_curator_prompt_separates_requirement_strength_from_verifiability():
    assert "不能按单个关键词机械分类" in RUBRIC_CURATOR_SYSTEM_PROMPT
    assert "产品可以/能够完成 A" in RUBRIC_CURATOR_SYSTEM_PROMPT
    assert "颜色都行、款式不限" in RUBRIC_CURATOR_SYSTEM_PROMPT
    assert "即使要求没有量化阈值，也仍然是 hard" in (
        RUBRIC_CURATOR_SYSTEM_PROMPT
    )
    assert "acceptance_criteria" not in RUBRIC_CURATOR_SYSTEM_PROMPT


@pytest.mark.parametrize("anchor_ids", [[], ["q9999"], ["q0001", "q0001"]])
def test_query_only_rubric_rejects_missing_or_unknown_query_evidence(anchor_ids):
    with pytest.raises(ContractValidationError):
        materialize_rubric_bundle(
            task_facts=_facts(),
            curator_response=_response(anchor_ids=anchor_ids),
            curator_model="curator",
            curator_prompt_version="test",
            rubric_version="test",
        )


def test_query_only_rubric_rejects_an_empty_requirement_list():
    with pytest.raises(ContractValidationError, match="at least one"):
        materialize_rubric_bundle(
            task_facts=_facts(),
            curator_response={"requirements": []},
            curator_model="curator",
            curator_prompt_version="test",
            rubric_version="test",
        )


def test_query_only_rubric_preserves_multiple_independent_requirements():
    response = _response()
    response["requirements"].append(
        {
            "description": "优先选择木制手串",
            "hardness": "soft",
            "query_anchor_ids": ["q0002"],
            "selection_reason": "用户以优先措辞提出材质和品类偏好",
        }
    )

    bundle = materialize_rubric_bundle(
        task_facts=_facts(),
        curator_response=response,
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )

    assert [item["hardness"] for item in bundle["rubrics"]] == [
        "hard",
        "soft",
    ]


def test_curator_rejects_llm_authored_acceptance_criteria():
    response = _response()
    response["requirements"][0]["acceptance_criteria"] = "自行发明的标准"
    with pytest.raises(ContractValidationError, match="exactly"):
        materialize_rubric_bundle(
            task_facts=_facts(),
            curator_response=response,
            curator_model="curator",
            curator_prompt_version="test",
            rubric_version="test",
        )


@pytest.mark.parametrize("field", ["query_hash", "task_data_hash"])
def test_rubric_rejects_tampered_generation_hash(field):
    bundle = materialize_rubric_bundle(
        task_facts=_facts(),
        curator_response=_response(),
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )
    bundle["generation"][field] = "0" * 64
    from shopping_grpo.evaluation.contracts import validate_rubric_bundle

    with pytest.raises(ContractValidationError, match="does not match"):
        validate_rubric_bundle(bundle)


def test_rubric_rejects_anchor_span_mismatch():
    bundle = materialize_rubric_bundle(
        task_facts=_facts(),
        curator_response=_response(),
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )
    bundle["rubrics"][0]["query_anchor_ids"] = ["q0002"]
    from shopping_grpo.evaluation.contracts import validate_rubric_bundle

    with pytest.raises(ContractValidationError, match="query_spans"):
        validate_rubric_bundle(bundle)


def test_judge_requires_explicit_review_approval():
    bundle = materialize_rubric_bundle(
        task_facts=_facts(),
        curator_response=_response(),
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )
    from shopping_grpo.evaluation.contracts import validate_rubric_bundle

    with pytest.raises(ContractValidationError, match="explicitly approved"):
        validate_rubric_bundle(bundle, require_approved=True)
    approved = approve_bundles(
        [bundle], reviewer="human", reviewed_at="2026-09-06T00:00:00+00:00"
    )
    assert approved[0]["review"]["status"] == "approved"


def test_needs_review_cannot_be_batch_approved():
    response = _response()
    response["requirements"][0]["hardness"] = "needs_review"
    bundle = materialize_rubric_bundle(
        task_facts=_facts(),
        curator_response=response,
        curator_model="curator",
        curator_prompt_version="test",
        rubric_version="test",
    )
    assert bundle["review"]["status"] == "needs_review"
    with pytest.raises(ValueError, match="unresolved"):
        approve_bundles(
            [bundle], reviewer="human", reviewed_at="2026-09-06T00:00:00+00:00"
        )


def test_curator_records_invalid_response_before_schema_retry():
    invalid = _response()
    invalid["requirements"][0]["acceptance_criteria"] = "模型自定义标准"

    class FakeClient:
        model = "fake-curator"

        def __init__(self):
            self.responses = [invalid, _response()]

        def complete_json(self, messages):
            return {
                "result": self.responses.pop(0),
                "metadata": {"provider_model": self.model},
            }

    requests = []
    attempts = []
    _, bundle, request_ids = _curate(
        FakeClient(),
        _facts(),
        1,
        run_plan_sha256="plan",
        on_request=requests.append,
        on_attempt=attempts.append,
    )

    assert len(request_ids) == 2
    assert [row["validation_status"] for row in attempts] == [
        "invalid",
        "valid",
    ]
    invalid_requirement = attempts[0]["curator_response"]["requirements"][0]
    assert "acceptance_criteria" in invalid_requirement
    assert bundle["review"]["status"] == "auto_drafted"
