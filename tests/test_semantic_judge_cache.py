import json

import pytest

from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    validate_judge_deterministic_consistency,
)
from shopping_grpo.evaluation.judge_cache import (
    SemanticJudgeCache,
    semantic_request_spec,
)
from shopping_grpo.evaluation.prompts import (
    build_trajectory_judge_messages,
    deterministic_judge_facts,
    semantic_judge_trajectory_id,
)
from shopping_grpo.evaluation.rubric import (
    build_task_facts,
    materialize_rubric_bundle,
    stable_hash,
)
from scripts.evaluate_multiturn_panels import _judge


def _rubric():
    facts = build_task_facts(task_id=7, query="预算100元以内")
    candidates = {
        "schema_version": "shopping-rubric-candidates-v3",
        "extractor_version": "test",
        "reward_version": "shopsimulator-reward-v4",
        "task_id": 7,
        "query": facts["query"],
        "candidates": [{
            "candidate_id": "c0001",
            "constraint_type": "price",
            "field_path": "purchase.price",
            "operator": "lte",
            "expected_value": {
                "version": "shopping-price-constraint-v2",
                "kind": "hard_max",
                "lower": None,
                "upper": 100.0,
                "target": None,
                "tolerance": None,
                "source_text": "100元以内",
            },
            "data_sources": ["instruction"],
        }],
    }
    candidates["candidate_hash"] = stable_hash(candidates)
    return materialize_rubric_bundle(
        task_facts=facts,
        candidates=candidates,
        curator_response={"requirements": [{
            "candidate_ids": ["c0001"],
            "description": "价格不超过100元",
            "hardness": "hard",
            "query_anchor_ids": ["q0001"],
            "selection_reason": "明确预算",
        }]},
        curator_model="test",
        curator_prompt_version="test",
        rubric_version="test",
    )


def _normalized(*, trajectory_id, tool_call_id, query="测试商品"):
    return {
        "schema_version": "shopping-normalized-trajectory-v2",
        "task_id": 7,
        "trajectory_id": trajectory_id,
        "actor_query": query,
        "status": "done",
        "done": True,
        "events": [{
            "event_id": "e0001",
            "event_type": "tool_step",
            "tool_call_id": tool_call_id,
            "tool_name": "buy_now",
            "parameters": {},
            "actor_visible_observation": "购买完成",
            "done": True,
        }],
        "terminal": {
            "done": True,
            "over": True,
            "purchase": {
                "asin": "A1",
                "name": "测试商品",
                "category": "测试",
                "price": 98.0,
                "options": {"尺寸": "M"},
                "attributes": ["测试"],
            },
        },
    }


def _metrics():
    return {
        "actions_and_efficiency": {"executed_tool_steps": 1},
        "repetition": {},
        "legality": {},
        "context": {},
    }


def _judge_result(trajectory_id, *, price_status="satisfied"):
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "task_id": 7,
        "trajectory_id": trajectory_id,
        "judge_status": "valid",
        "rubric_assessments": [{
            "rubric_id": "r0001",
            "status": price_status,
            "reason": "代码价格比较",
            "evidence_event_ids": ["e0001"],
        }],
        "dimension_scores": {
            name: {"score": 1, "reason": "ok", "evidence_event_ids": ["e0001"]}
            for name in JUDGE_DIMENSIONS
        },
        "clarification_assessment": {
            "status": "not_applicable",
            "reason": "none",
            "evidence_event_ids": [],
        },
        "errors": {"primary": None, "secondary": [], "evidence_event_ids": []},
        "overall_diagnosis": "ok",
    }


def test_random_trajectory_and_tool_call_ids_do_not_change_judge_request():
    rubric = _rubric()
    left = build_trajectory_judge_messages(
        normalized=_normalized(trajectory_id="left", tool_call_id="call-left"),
        rubric_bundle=rubric,
        deterministic_metrics=_metrics(),
    )
    right = build_trajectory_judge_messages(
        normalized=_normalized(trajectory_id="right", tool_call_id="call-right"),
        rubric_bundle=rubric,
        deterministic_metrics=_metrics(),
    )
    assert left == right
    assert semantic_judge_trajectory_id(left).startswith("semantic-")
    payload = json.loads(left[1]["content"])
    assert "tool_call_id" not in json.dumps(payload)


def test_semantic_cache_computes_once_for_identical_requests(tmp_path):
    messages = build_trajectory_judge_messages(
        normalized=_normalized(trajectory_id="one", tool_call_id="call-one"),
        rubric_bundle=_rubric(),
        deterministic_metrics=_metrics(),
    )
    spec = semantic_request_spec(
        model="judge",
        base_url="https://judge.example/v1",
        prompt_version="prompt-v1",
        max_tokens=4096,
        thinking=False,
        messages=messages,
    )
    calls = []

    def compute():
        calls.append(True)
        return {"judge_result": _judge_result(semantic_judge_trajectory_id(messages))}

    cache = SemanticJudgeCache(tmp_path)
    first, first_hit = cache.get_or_compute(spec, compute)
    second, second_hit = cache.get_or_compute(spec, compute)
    assert first == second
    assert (first_hit, second_hit) == (False, True)
    assert len(calls) == 1


def test_price_arithmetic_is_authoritative_for_judge_assessment():
    rubric = _rubric()
    facts = deterministic_judge_facts(
        normalized=_normalized(trajectory_id="one", tool_call_id="call-one"),
        rubric_bundle=rubric,
    )
    assert facts["price_checks"][0]["status"] == "pass"
    valid = _judge_result("one", price_status="satisfied")
    validate_judge_deterministic_consistency(valid, facts)
    invalid = _judge_result("one", price_status="violated")
    with pytest.raises(ContractValidationError, match="price_checks"):
        validate_judge_deterministic_consistency(invalid, facts)


def test_final_option_is_checked_from_terminal_state_not_transient_selection():
    rubric = {
        "rubrics": [{
            "rubric_id": "r-option",
            "candidate_semantics": [{
                "constraint_type": "option",
                "operator": "eq",
                "expected_value": {
                    "source_axis": "尺寸",
                    "axis": "size",
                    "value": "M",
                },
            }],
        }],
    }
    facts = deterministic_judge_facts(
        normalized=_normalized(trajectory_id="one", tool_call_id="call-one"),
        rubric_bundle=rubric,
    )
    assert facts["option_checks"] == [{
        "rubric_id": "r-option",
        "status": "pass",
        "comparisons": [{
            "source_axis": "尺寸",
            "operator": "eq",
            "expected": "M",
            "actual": "M",
            "passed": True,
        }],
    }]
    result = _judge_result("one")
    result["rubric_assessments"] = [{
        "rubric_id": "r-option",
        "status": "violated",
        "reason": "read an earlier option",
        "evidence_event_ids": ["e0001"],
    }]
    with pytest.raises(ContractValidationError, match="option_checks"):
        validate_judge_deterministic_consistency(result, facts)


def test_missing_final_option_requires_unknown_judge_status():
    facts = {
        "price_checks": [],
        "option_checks": [{
            "rubric_id": "r0001",
            "status": "unavailable",
            "comparisons": [{"actual": None, "passed": None}],
        }],
    }
    invalid = _judge_result("one", price_status="satisfied")
    with pytest.raises(ContractValidationError, match="final option is unavailable"):
        validate_judge_deterministic_consistency(invalid, facts)
    valid = _judge_result("one", price_status="unknown")
    validate_judge_deterministic_consistency(valid, facts)


@pytest.mark.parametrize(
    "fact_name,error_label",
    [("price_checks", "budget_violation"), ("option_checks", "wrong_option")],
)
def test_error_label_cannot_contradict_all_passing_facts(fact_name, error_label):
    facts = {
        "price_checks": [],
        "option_checks": [],
    }
    facts[fact_name] = [{"rubric_id": "r0001", "status": "pass"}]
    result = _judge_result("one")
    result["errors"] = {
        "primary": error_label,
        "secondary": [],
        "evidence_event_ids": ["e0001"],
    }
    with pytest.raises(ContractValidationError, match=error_label):
        validate_judge_deterministic_consistency(result, facts)


def test_panel_judge_rebinds_one_cached_result_to_two_trajectories(tmp_path):
    class Client:
        model = "judge"
        base_url = "https://judge.example/v1"
        max_tokens = 4096
        thinking = False

        def __init__(self):
            self.calls = 0

        def complete_json(self, messages):
            self.calls += 1
            return {
                "result": _judge_result(semantic_judge_trajectory_id(messages)),
                "metadata": {"provider_request_id": "request-1"},
            }

    client = Client()
    cache = SemanticJudgeCache(tmp_path)
    requests = []
    common = {
        "client": client,
        "cache": cache,
        "metrics": _metrics(),
        "rubric": _rubric(),
        "schema_retries": 0,
        "run_plan_sha256": "plan",
        "on_request": requests.append,
    }
    first_metadata, first_result, _ = _judge(
        normalized=_normalized(trajectory_id="first", tool_call_id="random-1"),
        **common,
    )
    second_metadata, second_result, _ = _judge(
        normalized=_normalized(trajectory_id="second", tool_call_id="random-2"),
        **common,
    )
    assert client.calls == 1
    assert first_result["trajectory_id"] == "first"
    assert second_result["trajectory_id"] == "second"
    assert first_metadata["semantic_cache_hit"] is False
    assert second_metadata["semantic_cache_hit"] is True
    assert first_metadata["semantic_request_sha256"] == second_metadata[
        "semantic_request_sha256"
    ]
    assert [row["request_kind"] for row in requests] == [
        "semantic_cache_lookup",
        "external_api_call",
        "semantic_cache_lookup",
    ]
