import json

import pytest

from shopping_grpo.evaluation.contracts import ContractValidationError
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_SYSTEM_PROMPT,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (
    build_task_facts,
    materialize_rubric_bundle,
)


def _facts():
    return build_task_facts(task_id=7, query="预算100元以内，优先木制手串")


def _response(*, anchor_ids=None, criteria="最终购买价格不超过100元"):
    if anchor_ids is None:
        anchor_ids = ["q0001"]
    return {
        "requirements": [
            {
                "description": "价格不超过100元",
                "acceptance_criteria": criteria,
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


def test_curator_prompt_marks_optional_and_unverifiable_requirements():
    assert "也行、可以、即可、都行" in RUBRIC_CURATOR_SYSTEM_PROMPT
    assert "没有用户给出可见且可验证阈值的模糊要求必须为" in (
        RUBRIC_CURATOR_SYSTEM_PROMPT
    )


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
            "acceptance_criteria": "轨迹中的候选或最终购买商品有可见木制证据",
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
