# -*- coding: utf-8 -*-
"""阶段 1：从 data/sft 构造"欠约束"任务（clear/under query + fake_profile）。"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SFT_TRAIN = REPO / "data" / "sft" / "train.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_JSON = OUT_DIR / "tasks.json"
OUT_REVIEW = OUT_DIR / "tasks_review.md"

MAX_CANDIDATES = 25
SEED = 42
MIN_HIDEABLE = 2

CN_NUM = r"[0-9一二三四五六七八九十百千]+"

BUDGET_PATTERNS = [
    re.compile(
        r"(?:价格|费用|预算|售价|花费|钱)\s*"
        r"(?:在|为|是|大概|大约|就|只要|别超)?\s*"
        + CN_NUM + r"(?:\.\d+)?\s*(?:元|块钱|块)?\s*"
        r"(?:左右|以内|以内吧|上下|这样|吧|就行|就好|之内|以下)?"
    ),
    re.compile(
        r"(?:不能超过|不超过|控制在|控制到|预算到|别超过|别超)\s*"
        + CN_NUM + r"(?:\.\d+)?\s*(?:元|块钱|块)?\s*(?:左右|以内|吧|就行|之内)?"
    ),
    re.compile(
        CN_NUM + r"(?:\.\d+)?\s*(?:块钱|元)\s*(?:左右|以内|上下|吧|就行|之内|以内吧)?"
    ),
]

BRAND_PATTERNS = [
    re.compile(r"([\u4e00-\u9fa5]{1,6})(?:品牌|牌子|牌的|牌)"),
    re.compile(r"([\u4e00-\u9fa5]{2,6})(?:生产的|进口的|制造的|代工的|旗下)"),
]

COLOR_WORDS = [
    "白色", "黑色", "红色", "酒红色", "灰色", "蓝色", "绿色", "黄色", "粉色",
    "紫色", "金色", "银色", "棕色", "米色", "藏青", "枣红", "咖啡色",
    "浅灰", "深灰", "天蓝", "浅蓝", "香槟", "黑底", "白底", "酒红",
]


def extract_query(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user" and str(m.get("content", "")).startswith("Instruction:"):
            return m["content"][len("Instruction:"):].strip()
    return ""
def hide_budget(query: str) -> tuple[str, str | None]:
    for p in BUDGET_PATTERNS:
        m = p.search(query)
        if not m:
            continue
        s, e = m.span()
        return query[:s] + "（预算合适即可）" + query[e:], query[s:e]
    return query, None


def hide_brand(query: str) -> tuple[str, str | None]:
    for p in BRAND_PATTERNS:
        m = p.search(query)
        if not m:
            continue
        s, e = m.span()
        hidden = m.group(1) if m.lastindex else m.group(0)
        return query[:s] + "某个品牌" + query[e:], hidden
    return query, None


def hide_color(query: str) -> tuple[str, str | None]:
    for c in COLOR_WORDS:
        if c in query:
            return query.replace(c, "", 1), c
    return query, None


def build_under_query(query: str) -> tuple[str, dict, list[str]]:
    """返回 (under_query, fake_profile, hidden_fields)。最多隐藏 2 个维度。"""
    under = query
    profile: dict = {}
    hidden: list[str] = []

    under, budget = hide_budget(under)
    if budget is not None:
        profile["budget"] = budget
        hidden.append("budget")

    if len(hidden) < 2:
        under, brand = hide_brand(under)
        if brand is not None:
            profile["brand"] = brand
            hidden.append("brand")

    if len(hidden) < 2:
        under, color = hide_color(under)
        if color is not None:
            profile["color"] = color
            hidden.append("color")

    return under, profile, hidden


def hideable_count(query: str) -> int:
    n = 0
    for p in BUDGET_PATTERNS:
        if p.search(query):
            n += 1
            break
    if any(p.search(query) for p in BRAND_PATTERNS):
        n += 1
    if any(c in query for c in COLOR_WORDS):
        n += 1
    return n


def main() -> int:
    if not SFT_TRAIN.exists():
        print(f"[build_tasks] 找不到 {SFT_TRAIN}", file=sys.stderr)
        return 1

    random.seed(SEED)
    records = []
    with open(SFT_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    candidates = []
    for rec in records:
        q = extract_query(rec.get("messages", []))
        if not q:
            continue
        if hideable_count(q) >= MIN_HIDEABLE:
            candidates.append(rec.get("task_id"))

    random.shuffle(candidates)
    selected = candidates[:MAX_CANDIDATES]

    tasks = []
    for task_id in selected:
        # 重新从原始记录里取该 task 的 query（避免重复解析）
        for rec in records:
            if rec.get("task_id") == task_id:
                q = extract_query(rec.get("messages", []))
                break
        under, profile, hidden = build_under_query(q)
        tasks.append({
            "task_id": int(task_id),
            "clear_query": q,
            "under_query": under,
            "fake_profile": profile,
            "hidden_fields": hidden,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    lines = ["# 欠约束任务抽查（人工）\n",
             "请逐条确认：under_query 确实隐藏了约束、仍有歧义/多解；fake_profile 与 TaskFacts 一致。\n"]
    for t in tasks:
        lines.append(f"\n## task_id={t['task_id']} hidden={t['hidden_fields']}")
        lines.append(f"- **clear**: {t['clear_query']}")
        lines.append(f"- **under**: {t['under_query']}")
        lines.append(f"- **profile**: {json.dumps(t['fake_profile'], ensure_ascii=False)}")
        lines.append("- [ ] 确认")
    with open(OUT_REVIEW, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[build_tasks] 候选={len(candidates)} 选定={len(tasks)}")
    print(f"[build_tasks] 输出 -> {OUT_JSON}")
    print(f"[build_tasks] 抽查 -> {OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

