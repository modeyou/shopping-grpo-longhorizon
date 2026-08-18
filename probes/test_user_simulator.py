# -*- coding: utf-8 -*-
"""user_simulator 单元测试（无需外部依赖）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from user_simulator import answer_question, UNKNOWN_ANSWER


def _profile(**kw):
    return kw


def test_budget_question_only_answers_budget():
    q = "您的预算大概是多少呢？"
    ans, info = answer_question(q, _profile(budget="预算40以内"))
    assert "40" in ans and "预算" in ans
    assert info["field"] == "budget" and not info["unknown"]
    # 只答所问：不应把 brand/color 说出来
    assert "白色" not in ans


def test_brand_question():
    ans, info = answer_question("您有偏好的品牌吗？", _profile(brand="海外"))
    assert "海外" in ans
    assert info["field"] == "brand"


def test_color_question():
    ans, info = answer_question("颜色想要什么样的？", _profile(color="白色"))
    assert "白色" in ans
    assert info["field"] == "color"


def test_unknown_field_returns_neutral():
    ans, info = answer_question("这个有什么保修吗？", _profile(budget="40以内"))
    assert ans == UNKNOWN_ANSWER
    assert info["unknown"] is True


def test_deterministic():
    p = _profile(budget="40以内", color="白色")
    a1, _ = answer_question("预算多少？", p)
    a2, _ = answer_question("预算多少？", p)
    assert a1 == a2


if __name__ == "__main__":
    failures = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        fn = globals()[name]
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print("----")
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)
