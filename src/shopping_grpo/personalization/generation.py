"""LLM-assisted generation for personalized clarification task packages."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shopping_grpo.personalization.schema import (
    MAX_QUESTIONS,
    PROFILE_LIST_FIELDS,
    SCHEMA_VERSION,
    finalize_task,
    stable_hash,
    validate_task_against_source,
)


ARCHITECT_PROMPT_VERSION = "personalized-task-architect-v3"
CRITIC_PROMPT_VERSION = "personalized-task-critic-v1"

ARCHITECT_OUTPUT_CONTRACT = r"""
Follow this exact output shape. Every listed key is required. Do not invent wrapper
keys such as `preferences` or `target_item`.

{
  "profile": {
    "stable_facts": [],
    "category_preferences": [],
    "brand_preferences": [],
    "budget_preferences": [],
    "attribute_preferences": [],
    "option_preferences": []
  },
  "current_request": "natural Chinese request",
  "private_goal": {
    "category": "category copied or grounded from the source facts",
    "constraints": [
      {
        "constraint_id": "c1",
        "field": "function",
        "value": "grounded value",
        "hardness": "hard",
        "source": "request_explicit",
        "evidence": {
          "source_path": "attributes",
          "source_value": "an exact value or exact substring present at source_path"
        }
      }
    ]
  },
  "clarification": {
    "should_ask": false,
    "max_questions": 2,
    "targets": []
  },
  "conflicts": []
}

Exact constraint fields: budget, brand, function, material, color, size,
capacity, bundle, specification, model, quantity, compatibility.
Exact evidence source_path values: category, title, shop_name, pricing,
attributes, required_options, available_options, original_instruction.
Use only `hard` or `soft` for hardness. Copy evidence text exactly; do not
paraphrase it. Keep category directly at private_goal.category.
"""

SCENARIO_ORDER = (
    "complete_request",
    "profile_resolvable",
    "clarification_required",
    "profile_conflict",
)

ARCHITECT_SYSTEM_PROMPT = """你是个性化购物训练数据的 Task Architect。你会收到一条私有的
ShopSimulator 商品与任务事实，以及指定场景。请生成一条自然、可验证、无答案泄漏的中文任务。

只输出 JSON object，不要输出 Markdown。JSON 只能包含：profile、current_request、private_goal、
clarification、conflicts。不要输出 schema_version、task_id、source、generation 或 audit，它们由代码添加。

硬规则：
1. private_goal.constraints 中每条约束必须有唯一 constraint_id、field、value、hardness、source 和
evidence。evidence 必须含 source_path 与 source_value，逐字引用输入事实；source_path 只用 category、
title、shop_name、pricing、attributes、required_options、available_options、original_instruction。field
只用 budget/brand/function/material/color/size/capacity/bundle/specification/model/quantity/compatibility。
2. source 只用 request_explicit、clarification_answer、profile_stable_fact、profile_preference。
3. clarification.max_questions 固定为 2；只有 clarification_required 场景 should_ask=true 且 targets
为 1~2 条，其他场景应为 false 和空列表。每个 target 只问一个不同字段，并引用对应 constraint_id。
4. 隐藏答案不得出现在 current_request 或 profile。问题答案必须来自输入商品事实。
5. complete_request 的全部约束来自 request_explicit；profile_resolvable 至少一条来自画像；
profile_conflict 必须声明当前请求覆盖画像的冲突；clarification_required 的缺失事实画像也不能确定。
6. profile 只使用约定的紧凑字段。stable_facts 仅允许高置信、applies_to=self 的 shoe_size 或
clothing_size；品牌、预算、颜色、材质和功能必须放在偏好列表，不能伪装成稳定事实。
7. 不得复制完整商品标题、ASIN，不能为目标商品拼接唯一检索短语。当前请求要像真实用户说话。
8. profile_resolvable 若使用尺码稳定事实，current_request 必须明确是为本人购买；给他人购买或对象
不明时，尺码必须澄清。
"""

CRITIC_SYSTEM_PROMPT = """你是独立的数据质量 Critic。请检查候选个性化购物任务是否自然、事实
一致、符合指定场景，并且没有把 clarification_answer 泄漏到画像或当前请求。特别检查长期偏好
是否被错误写成硬事实、当前请求是否覆盖画像冲突、问题是否真的有必要、答案是否由源商品事实
支持。只输出 JSON：{"verdict":"accept|reject","issues":["..."]}。不要改写任务。"""


class GenerationAPIError(RuntimeError):
    """Remote generation did not produce a usable response."""


class GenerationTransportError(GenerationAPIError):
    """The provider call failed; the source task must remain resumable."""


def extract_json_object(value: object) -> dict:
    """Parse a JSON object from plain text or a fenced model response."""

    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    text = str(value or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise GenerationAPIError("model response does not contain a JSON object")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise GenerationAPIError(f"invalid JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GenerationAPIError("model response JSON must be an object")
    return parsed


class OpenAICompatibleJSONClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        retries: int = 2,
        transport: Callable | None = None,
    ):
        if not model or not base_url or not api_key:
            raise ValueError("model, base_url and api_key are required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.transport = transport
        self.call_count = 0

    def complete_json(self, *, system: str, user: str) -> tuple[dict, object]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model.casefold().startswith("deepseek-v4"):
            payload["thinking"] = {"type": "disabled"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-grpo-personalized-data/0.1",
        }
        endpoint = f"{self.base_url}/chat/completions"
        for attempt in range(self.retries + 1):
            self.call_count += 1
            try:
                if self.transport:
                    response = self.transport(endpoint, payload, headers, self.timeout)
                else:
                    request = Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout) as raw:
                        response = json.loads(raw.read().decode("utf-8"))
                content = response["choices"][0]["message"]["content"]
                return extract_json_object(content), response
            except (HTTPError, URLError, OSError, TimeoutError, KeyError, IndexError) as exc:
                if attempt >= self.retries:
                    raise GenerationTransportError(f"generation request failed: {exc}") from exc
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")


def _profile_with_defaults(value: object, profile_id: str) -> dict:
    profile = dict(value) if isinstance(value, Mapping) else {}
    generic_preferences = profile.pop("preferences", [])
    if isinstance(generic_preferences, list):
        buckets = {
            "category": "category_preferences",
            "brand": "brand_preferences",
            "budget": "budget_preferences",
            "size": "option_preferences",
            "capacity": "option_preferences",
            "bundle": "option_preferences",
            "specification": "option_preferences",
        }
        for preference in generic_preferences:
            if not isinstance(preference, Mapping):
                continue
            bucket = buckets.get(preference.get("field"), "attribute_preferences")
            profile.setdefault(bucket, []).append(dict(preference))
    profile["profile_id"] = profile_id
    for key in PROFILE_LIST_FIELDS:
        profile.setdefault(key, [])
    return profile


def build_architect_task(
    generated: object,
    *,
    source: Mapping,
    scenario: str,
    sequence: int,
    model: str,
) -> dict:
    """Attach code-owned identity/provenance and validate an Architect response."""

    candidate = extract_json_object(generated)
    task_id = f"pca-{sequence:06d}"
    allowed = {"profile", "current_request", "private_goal", "clarification", "conflicts"}
    extra = set(candidate) - allowed
    if extra:
        raise GenerationAPIError(f"Architect returned code-owned fields: {sorted(extra)}")
    private_goal = deepcopy(candidate.get("private_goal"))
    if not isinstance(private_goal, Mapping):
        private_goal = {}
    else:
        private_goal = dict(private_goal)
    # Some providers put catalog metadata in a redundant target_item wrapper.
    # Category is source-owned, so fill it deterministically and discard the
    # wrapper rather than persisting a copied title/ASIN in generated data.
    private_goal.pop("target_item", None)
    private_goal.setdefault("category", source.get("category"))

    task = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "source": {
            "shopsim_task_id": int(source["shopsim_task_id"]),
            "target_asin": str(source["target_asin"]),
            "source_environment_version": "shopsimulator-environment-v2.1",
            "source_hash": source.get("source_hash"),
        },
        "scenario": scenario,
        "profile": _profile_with_defaults(candidate.get("profile"), f"profile-{sequence:06d}"),
        "current_request": candidate.get("current_request"),
        "private_goal": private_goal,
        "clarification": candidate.get("clarification"),
        "conflicts": candidate.get("conflicts", []),
        "generation": {
            "architect_model": model,
            "architect_prompt_version": ARCHITECT_PROMPT_VERSION,
            "architect_prompt_hash": stable_hash(ARCHITECT_SYSTEM_PROMPT),
            "source_hash": source.get("source_hash"),
        },
        "audit": {},
    }
    task = finalize_task(task)
    validate_task_against_source(task, source)
    return task


def architect_user_prompt(source: Mapping, scenario: str, question_count: int) -> str:
    scenario_contracts = {
        "complete_request": (
            "All constraints must use source=request_explicit, and every constraint value must "
            "appear in current_request. Use an empty profile unless a harmless profile is needed. "
            "clarification must be {should_ask:false,max_questions:2,targets:[]}."
        ),
        "profile_resolvable": (
            "At least one constraint must use profile_stable_fact or profile_preference, and its "
            "value must appear in the appropriate profile list. The request must make the profile "
            "applicable. clarification must be {should_ask:false,max_questions:2,targets:[]}."
        ),
        "clarification_required": (
            f"Create exactly {question_count} constraint(s) with source=clarification_answer and "
            "exactly the same number of targets. Each target must be "
            '{"constraint_id":"cX","field":"budget","question":"自然的单属性问题",'
            '"answer":"由源事实支持且未在请求或画像出现的回答",'
            '"answer_facts":{"budget":"与 constraint value 相同的值"}}. '
            "Set should_ask=true. Each target uses a distinct allowed field; replace budget with "
            "the actual field in both field and answer_facts."
        ),
        "profile_conflict": (
            "Create at least one profile preference that conflicts with an explicit current request "
            "constraint. The explicit request wins. Declare the conflict in conflicts. Constraint "
            "values must appear in their declared request/profile view. clarification must be "
            "{should_ask:false,max_questions:2,targets:[]}."
        ),
    }
    return (
        f"指定场景：{scenario}\n"
        f"clarification_required 时目标提问数：{question_count}；其他场景忽略该数字。\n"
        + ARCHITECT_OUTPUT_CONTRACT
        + "\nScenario-specific contract:\n"
        + scenario_contracts[scenario]
        + "\n私有 ShopSimulator 事实如下：\n"
        + json.dumps(source, ensure_ascii=False, indent=2)
    )


def critic_user_prompt(source: Mapping, task: Mapping) -> str:
    return (
        "源事实：\n"
        + json.dumps(source, ensure_ascii=False, indent=2)
        + "\n候选任务：\n"
        + json.dumps(task, ensure_ascii=False, indent=2)
    )


def validate_critic_response(value: object) -> dict:
    result = extract_json_object(value)
    if result.get("verdict") not in {"accept", "reject"}:
        raise GenerationAPIError("Critic verdict must be accept or reject")
    issues = result.get("issues")
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise GenerationAPIError("Critic issues must be a list of strings")
    if result["verdict"] == "accept" and issues:
        raise GenerationAPIError("Critic accept response must not contain issues")
    return {"verdict": result["verdict"], "issues": issues}


def scenario_for_index(index: int) -> str:
    return SCENARIO_ORDER[index % len(SCENARIO_ORDER)]


def scenario_for_run(index: int, accepted_scenarios: list[str], target: int) -> str:
    """Rotate attempts while enforcing deterministic final scenario quotas."""

    base, remainder = divmod(target, len(SCENARIO_ORDER))
    quotas = {
        scenario: base + (position < remainder)
        for position, scenario in enumerate(SCENARIO_ORDER)
    }
    counts = Counter(accepted_scenarios)
    for offset in range(len(SCENARIO_ORDER)):
        scenario = scenario_for_index(index + offset)
        if counts[scenario] < quotas[scenario]:
            return scenario
    raise ValueError("all scenario quotas are already satisfied")


def question_count_for_index(index: int, scenario: str) -> int:
    if scenario != "clarification_required":
        return 0
    return 1 + ((index // 4) % MAX_QUESTIONS)
