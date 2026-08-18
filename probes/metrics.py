# -*- coding: utf-8 -*-
"""阶段 4：指标计算与判据判断。

从 probes/outputs/trajectories_{arm}.jsonl 读取，按 Reward V3 终局分档：
  - reward >= 0.999            -> gold_purchase（严格成功）
  - reward == 0.55             -> valid_alternative（替代购买，非严格成功）
  - 0 < reward < 0.55         -> partial（部分满足）
  - reward < 0                 -> failure（错误购买/过早放弃/循环/超步）
  - reward is None             -> 无法判定（unknown）

输出对比表与 PASS/FAIL 判据结论。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "outputs"
ARMS = ["baseline", "clarify"]

# 判据（设计文档 §5）
SS_IMPROVE_PP = 10.0         # 模糊子集严格成功率 ≥ +10pp
WR_NOT_HIGHER = True         # 错误购买率不得高于基线
CLARIFY_MAX = 2              # 澄清 ≤ 2 轮
STEPS_RATIO_MAX = 1.3        # 平均步数 ≤ ~1.3x 基线


def classify(reward):
    if reward is None:
        return "unknown"
    if reward >= 0.999:
        return "gold_purchase"
    if reward == 0.55 or abs(reward - 0.55) < 1e-6:
        return "valid_alternative"
    if reward > 0:
        return "partial"
    return "failure"


def load(arm):
    path = OUT_DIR / f"trajectories_{arm}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize(records):
    n = len(records)
    if n == 0:
        return {"n": 0, "gold": 0, "failure": 0, "unknown": 0, "success_rate": None,
                "wrong_rate": None, "avg_steps": None, "avg_clarify": None}
    cats = defaultdict(int)
    steps = 0
    clarify = 0
    for r in records:
        cats[classify(r.get("terminal", {}).get("reward"))] += 1
        steps += len(r.get("steps", []))
        clarify += r.get("clarify_turns", 0)
    gold = cats.get("gold_purchase", 0)
    failure = cats.get("failure", 0)
    unknown = cats.get("unknown", 0) + cats.get("valid_alternative", 0) + cats.get("partial", 0)
    return {
        "n": n,
        "gold": gold,
        "failure": failure,
        "unknown": unknown,
        "categories": dict(cats),
        "success_rate": gold / n if n else None,
        "wrong_rate": failure / n if n else None,
        "avg_steps": steps / n if n else None,
        "avg_clarify": clarify / n if n else None,
    }


def decision(base, clarify):
    """按判据输出 PASS/FAIL 逐项结果。"""
    rows = []
    if base["n"] == 0 or clarify["n"] == 0:
        return [("数据不足", "NEED_DATA", False)]
    imp_pp = (clarify["success_rate"] - base["success_rate"]) * 100 if base["success_rate"] is not None and clarify["success_rate"] is not None else 0
    rows.append((f"严格成功率提升(pp): {imp_pp:+.1f}",
                 f"要求 >= +{SS_IMPROVE_PP:.0f}",
                 imp_pp >= SS_IMPROVE_PP))
    wr_ratio = (clarify["wrong_rate"] / base["wrong_rate"]) if base["wrong_rate"] else None
    rows.append((f"错误购买率: 澄清={clarify['wrong_rate']:.3f} 基线={base['wrong_rate']:.3f}",
                 f"要求 <= 基线",
                 (clarify["wrong_rate"] or 0) <= (base["wrong_rate"] or 0)))
    rows.append((f"平均澄清轮数: {clarify['avg_clarify']:.2f}",
                 f"要求 <= {CLARIFY_MAX}",
                 (clarify["avg_clarify"] or 0) <= CLARIFY_MAX))
    st_ratio = (clarify["avg_steps"] / base["avg_steps"]) if base["avg_steps"] else None
    rows.append((f"平均步数比: {st_ratio:.2f}x" if st_ratio else "平均步数: 数据不足",
                 f"要求 <= {STEPS_RATIO_MAX}x",
                 st_ratio is not None and st_ratio <= STEPS_RATIO_MAX))
    return rows


def main():
    records = {a: load(a) for a in ARMS}
    sums = {a: summarize(records[a]) for a in ARMS}
    base, clarify = sums["baseline"], sums["clarify"]

    lines = ["# B 探针对比结果\n"]
    lines.append("| 指标 | 基线臂 | 澄清臂 |")
    lines.append("|---|---|---|")
    for key, label in [("success_rate", "严格成功率"), ("wrong_rate", "错误购买率"),
                       ("avg_steps", "平均步数"), ("avg_clarify", "平均澄清轮数"), ("n", "任务数")]:
        lines.append(f"| {label} | {base.get(key)} | {clarify.get(key)} |")
    lines.append("\n## 判据判断")
    for msg, req, ok in decision(base, clarify):
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {msg} — {req}")

    out = OUT_DIR / "metrics_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
