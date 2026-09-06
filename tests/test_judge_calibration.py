from pathlib import Path

from scripts.audit_judge_calibration import CASE_VERSION, _load_cases
from scripts.apply_judge_calibration_rubric_revisions import (
    OUTPUT_RUBRIC_VERSION,
    _load_revisions,
    materialize_revision,
)
from scripts.prepare_judge_calibration_inputs import _filter


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ROOT / "data/evaluation/judge-calibration-review-v1/cases.jsonl"
)
REVISIONS = (
    ROOT
    / "data/evaluation/judge-calibration-review-v1/rubric-revisions.jsonl"
)


def test_judge_calibration_review_set_has_30_unique_classified_tasks():
    cases = _load_cases(CASES)
    assert len(cases) == 30
    assert len({int(case["task_id"]) for case in cases}) == 30
    assert {case["schema_version"] for case in cases} == {CASE_VERSION}
    assert {
        case["disposition"] for case in cases
    } == {"model_issue", "evaluator_error", "mixed_review"}
    assert sum(case["manual_review_required"] for case in cases) == 0
    assert sum("human_review" in case for case in cases) == 9
    assert sum("expected_rubric_version" in case for case in cases) == 6


def test_every_nonmanual_calibration_case_has_an_automated_assertion():
    for case in _load_cases(CASES):
        if case["manual_review_required"]:
            continue
        assert (
            case["actors"]
            or case.get("pair")
            or case.get("expected_rubric_version")
        ), case["task_id"]


def test_human_rubric_revisions_exactly_cover_confirmed_revision_tasks():
    cases = _load_cases(CASES)
    expected = {
        int(case["task_id"])
        for case in cases
        if case.get("expected_rubric_version") == OUTPUT_RUBRIC_VERSION
    }
    revisions = _load_revisions(REVISIONS)
    assert set(revisions) == expected == {
        973,
        2057,
        2240,
        4479,
        14960,
        18004,
    }


def test_human_revision_materializes_as_valid_approved_rubric():
    revision = _load_revisions(REVISIONS)[14960]
    base = {
        "task_id": 14960,
        "query": (
            "修改后的指令：想找一件适合小型犬穿着的蓬蓬裙，颜色为蓝色，"
            "尺码最小，预算18元，是否有推荐的产品？"
        ),
    }
    result = materialize_revision(base, revision)
    assert result["rubric_version"] == OUTPUT_RUBRIC_VERSION
    assert result["review"] == {"status": "human_approved"}
    assert {item["rubric_source"] for item in result["rubrics"]} == {
        "human_review_revision"
    }
    assert result["rubrics"][2]["candidate_semantics"][0][
        "expected_value"
    ]["value"] == "s"


def test_calibration_input_filter_preserves_case_order_and_requires_coverage(
    tmp_path,
):
    source = tmp_path / "rows.jsonl"
    source.write_text(
        '{"task_id":2,"value":"two"}\n'
        '{"task_id":1,"value":"one"}\n',
        encoding="utf-8",
    )
    assert [row["task_id"] for row in _filter(source, [1, 2])] == [1, 2]

    try:
        _filter(source, [1, 3])
    except ValueError as exc:
        assert "lacks calibration tasks" in str(exc)
    else:
        raise AssertionError("missing calibration tasks must fail")
