# -*- coding: utf-8 -*-
"""阶段 2：受控版用户模拟器（规则 + 模板）。

设计约束：
  - 只回答被问到的字段，不主动补充其他信息；
  - 0 随机性（同输入 → 同输出），可复现、可调试；
  - 问题意图按关键词/字段表匹配，未命中走中性分支。
"""
from __future__ import annotations

from typing import Any

BUDGET_KW = ["预算", "价格", "多少钱", "价位", "能接受", "花销", "贵", "便宜", "预算内", "价格上限"]
BRAND_KW = ["品牌", "牌子", "哪个牌", "什么牌"]
COLOR_KW = ["颜色", "什么色", "什么颜色"]
SPEC_KW = ["尺寸", "容量", "规格", "大小", "尺码", "型号", "数量", "多少支", "多少克", "几盒", "几件", "几包"]

# 字段 → (回答模板, 意图关键词)
FIELD_TEMPLATES = {
    "budget": ("预算大概{value}就行，别超太多。", BUDGET_KW),
    "brand": ("品牌方面想要{value}。", BRAND_KW),
    "color": ("颜色想要{value}。", COLOR_KW),
}

UNKNOWN_ANSWER = "这个我不太确定，你按合适的来就行。"


def _match_intent(question: str) -> str | None:
    """返回命中的字段名（budget/brand/color），未命中返回 None。"""
    q = question.lower()
    for field, (_tpl, kws) in FIELD_TEMPLATES.items():
        for kw in kws:
            if kw.lower() in q:
                return field
    # 规格/数量等其它字段：若 fake_profile 中有则按 key 回答
    return "spec"


def answer_question(question: str, fake_profile: dict[str, Any]) -> tuple[str, dict]:
    """根据问题与 fake_profile 生成用户回答。

    返回 (回答文本, info)，info 包含命中字段与是否未知。
    """
    field = _match_intent(question)
    if field == "spec":
        # 尝试匹配 fake_profile 里的具体键
        for key, value in fake_profile.items():
            if key in ("budget", "brand", "color"):
                continue
            if str(key) in question or str(value)[:2] in question:
                tpl = f"{value}就行。"
                return tpl, {"field": key, "unknown": False}
        return UNKNOWN_ANSWER, {"field": "spec", "unknown": True}

    value = fake_profile.get(field)
    if value is None:
        return UNKNOWN_ANSWER, {"field": field, "unknown": True}

    tpl = FIELD_TEMPLATES[field][0]
    answer = tpl.format(value=value)
    return answer, {"field": field, "unknown": False}
