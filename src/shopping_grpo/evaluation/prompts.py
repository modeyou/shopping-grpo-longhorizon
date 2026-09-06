"""Frozen prompts and Judge-safe payload renderers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy

from shopping_grpo.evaluation.contracts import (
    ERROR_TAXONOMY,
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION

RUBRIC_CURATOR_PROMPT_VERSION = "rubric-curator-v2-query-only-r1"
TRAJECTORY_JUDGE_PROMPT_VERSION = "trajectory-judge-v2-draft-r1"
_JUDGE_VISIBLE_ERROR_TAXONOMY = ERROR_TAXONOMY - {
    "reward_rubric_disagreement",
    "infrastructure_invalid",
}

RUBRIC_CURATOR_SYSTEM_PROMPT = """\
你是 Shopping Agent 项目的需求 Rubric 起草器。输入只包含用户 Query；你看不到、也不得猜测
目标商品、Gold 商品、Reward、商品库或隐藏环境状态。

将 Query 中每一项彼此独立、会影响商品选择或购买决策的明确要求写成一条 Rubric。必须完整覆盖
明确要求；但同义重复、礼貌语和不影响选择的背景描述不单列。不要根据常识补写用户没有说过的
品牌、材质、规格、功能、价格、数量或商品属性。

每条 requirement 必须：
- 用 query_quote 给出 Query 中连续、逐字存在且非空的原文；
- description 只重述该项要求，不扩写；
- acceptance_criteria 说明 Judge 应从 Actor 可见的搜索结果、详情、规格、价格或最终操作中看到什么
  才能判为 satisfied；证据不可见时 Judge 应判 unknown；
- 明确的品类、预算上限、否定要求、指定规格或数量为 hard；“优先、最好、倾向、左右、也行、可以、即可、都行”
  等偏好或可选条件为 soft；
- “大、小、高档、好看、舒适、质量好”等没有用户给出可见且可验证阈值的模糊要求必须为
  needs_review，不能伪装成可严格判定的 hard；
- 无法可靠判断时为 needs_review；
- selection_reason 简要说明原文为何支持该需求。

只输出一个 JSON 对象，不输出 Markdown 或额外字段：
{
  "requirements": [
    {
      "description": "非空、简短、用户可读的要求重述",
      "acceptance_criteria": "可由可见轨迹核验的满足条件",
      "hardness": "hard | soft | needs_review",
      "query_quote": "Query 中逐字存在的连续原文",
      "selection_reason": "该原文如何直接支持此要求"
    }
  ]
}
"""

TRAJECTORY_JUDGE_SYSTEM_PROMPT = f"""\
你是当前 Shopping Agent / ShopSimulator 项目的离线轨迹 Judge。

你必须只依据输入中的 actor_visible_trajectory 评价 Actor 的行为。不得假设你能看到
audit raw_observation、Gold 商品私有字段或未展示给 Actor 的候选。输入不会包含
Environment Reward、Reward 分项或代码判定的任务成功结论；这些结果由独立面板
负责，不能由你推断、覆盖或改写。

逐条需求状态只能是 satisfied、violated、unknown、not_applicable。没有可见证据时
使用 unknown。每项判断尽量引用真实 event_id；不得伪造不存在的 event_id。

另行判断澄清行为，不把“问过问题”本身视为加分：
- effective：缺失信息确实影响选择，问题具体，回答后有可见的利用证据；
- ineffective：提问或回答存在，但没有推动后续搜索、核验或决策；
- unnecessary：公开请求已经足够，仍提出不必要问题；
- not_applicable：没有发生澄清且当前条件不要求评价澄清；
- unknown：可见证据不足。不得使用隐藏 omitted_facts 或 used_facts 作判断。

五个维度分别打 0、1、2 分，不加权、不计算总分：
- search_strategy：搜索是否覆盖品类和关键条件，改写是否有效，是否机械重复；
- candidate_utilization：是否利用可见的高匹配候选，比较是否必要且不过度；
- evidence_verification：购买前是否核验关键属性、规格和最终价格；
- decision_quality：最终选择、规格和购买/放弃决策是否合理；
- termination_efficiency：是否过早购买/放弃、无效探索或耗尽步骤。

错误类型必须从输入提供的 frozen_error_taxonomy 中选择。primary 只选一个
最能解释轨迹失败或低分的根因；secondary 最多两个次要错误，不得与
primary 重复。没有明显错误时 primary 使用 null，secondary 必须为空列表。
errors.evidence_event_ids 必须引用能支持归因的真实轨迹事件。
只输出 JSON，不输出 Markdown。
schema_version 必须是 {JUDGE_SCHEMA_VERSION}。禁止输出 total_score、overall_score
或任何综合分。"""


def build_rubric_curator_messages(
    *,
    task_id: int,
    query: str,
) -> list[dict]:
    """Build a Query-only OpenAI-compatible curator request."""

    payload = {
        "task_id": int(task_id),
        "query": str(query),
    }
    return [
        {"role": "system", "content": RUBRIC_CURATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def actor_visible_trajectory(normalized: Mapping) -> dict:
    """Strip all audit-only content before rendering a Judge request."""

    if normalized.get("schema_version") != NORMALIZED_TRAJECTORY_VERSION:
        raise ValueError("unsupported normalized trajectory schema")
    events = []
    allowed_event_fields = (
        "event_id",
        "event_type",
        "action_attempt_id",
        "executed_step_id",
        "assistant_text",
        "tool_call_id",
        "tool_name",
        "parameters",
        "tool_call_parse_error",
        "guard_reason",
        "guard_consecutive_count",
        "latest_observation_truncated",
        "env_action",
        "actor_visible_observation",
        "done",
        "step_error",
    )
    for event_value in normalized.get("events") or []:
        if not isinstance(event_value, Mapping):
            continue
        event = {
            key: deepcopy(event_value[key])
            for key in allowed_event_fields
            if key in event_value
        }
        events.append(event)
    return {
        "trajectory_id": normalized.get("trajectory_id"),
        "task_id": normalized.get("task_id"),
        "status": normalized.get("status"),
        "done": normalized.get("done"),
        "events": events,
    }


def sanitize_terminal_for_judge(terminal: Mapping) -> dict:
    """Expose only neutral lifecycle flags, never Reward-derived conclusions."""

    if not isinstance(terminal, Mapping):
        terminal = {}
    return {
        "done": bool(terminal.get("done")),
        "over": bool(terminal.get("over")),
    }


def judge_visible_metrics(deterministic_metrics: Mapping) -> dict:
    """Remove Reward/outcome and validity conclusions before LLM judging."""

    if not isinstance(deterministic_metrics, Mapping):
        deterministic_metrics = {}
    allowed_sections = (
        "actions_and_efficiency",
        "repetition",
        "legality",
        "context",
    )
    return {
        section: deepcopy(deterministic_metrics.get(section) or {})
        for section in allowed_sections
    }


def build_trajectory_judge_messages(
    *,
    normalized: Mapping,
    rubric_bundle: Mapping,
    deterministic_metrics: Mapping,
) -> list[dict]:
    """Build one Pro request with exactly the evidence the Actor could use."""

    rubric = validate_rubric_bundle(
        rubric_bundle,
        expected_task_id=int(normalized["task_id"]),
    )
    dimensions = {
        name: {"allowed_scores": [0, 1, 2]}
        for name in JUDGE_DIMENSIONS
    }
    payload = {
        "task_id": normalized["task_id"],
        "trajectory_id": normalized["trajectory_id"],
        "query": normalized.get("actor_query") or rubric["query"],
        "rubric": rubric["rubrics"],
        "dimension_spec": dimensions,
        "frozen_error_taxonomy": sorted(
            _JUDGE_VISIBLE_ERROR_TAXONOMY
        ),
        "actor_visible_trajectory": actor_visible_trajectory(normalized),
        "terminal_state": sanitize_terminal_for_judge(
            normalized.get("terminal") or {}
        ),
        "judge_visible_metrics": judge_visible_metrics(
            deterministic_metrics
        ),
        "required_output": {
            "schema_version": JUDGE_SCHEMA_VERSION,
            "task_id": normalized["task_id"],
            "trajectory_id": normalized["trajectory_id"],
            "judge_status": "valid | invalid | not_judged",
            "rubric_assessments": [
                {
                    "rubric_id": "每条输入 rubric_id 恰好一次",
                    "status": "satisfied | violated | unknown | not_applicable",
                    "reason": "简短理由",
                    "evidence_event_ids": ["e0001"],
                }
            ],
            "dimension_scores": {
                name: {
                    "score": "0 | 1 | 2",
                    "reason": "简短理由",
                    "evidence_event_ids": ["e0001"],
                }
                for name in JUDGE_DIMENSIONS
            },
            "clarification_assessment": {
                "status": (
                    "effective | ineffective | unnecessary | "
                    "not_applicable | unknown"
                ),
                "reason": "简短理由",
                "evidence_event_ids": ["支持判断的真实 event_id"],
            },
            "errors": {
                "primary": "一个 frozen_error_taxonomy 值或 null",
                "secondary": ["最多两个不同的次要错误"],
                "evidence_event_ids": ["支持主次错误归因的 event_id"],
            },
            "overall_diagnosis": "简短整体诊断",
        },
    }
    return [
        {"role": "system", "content": TRAJECTORY_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]
