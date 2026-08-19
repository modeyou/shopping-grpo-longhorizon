"""Deterministic Actor-only redaction for clarification-positive tasks."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy


MASK_SCHEMA_VERSION = "shopsimulator-persona-mask-v1"
MASKED_TEACHER_GUIDANCE_VERSION = "masked-gap-teacher-v1"
MASKED_TEACHER_HINT = (
    "数据采集专用提示：Actor 当前可见信息中有且只有一项会影响正确商品或规格选择的具体事实被遮蔽。"
    "请结合当前请求、剩余画像和商品类型，判断最可能缺少的适用对象、兼容型号或关键规格，并在首次搜索前"
    "调用 `ask_user` 提出一个简洁问题。不得猜测答案，一次只问一个主题；用户回答后立即按回答继续购物。"
    "该提示不包含被遮蔽属性或答案。"
)


class PersonaMaskError(ValueError):
    """A mask spec does not match the frozen ShopSimulator persona."""


def apply_persona_mask(persona, spec):
    """Apply exact string redactions and return a masked copy plus private audit."""
    if not isinstance(persona, dict):
        raise PersonaMaskError("persona must be an object")
    if not isinstance(spec, dict):
        raise PersonaMaskError("persona_mask must be an object")
    if spec.get("schema_version") != MASK_SCHEMA_VERSION:
        raise PersonaMaskError("unsupported persona mask schema")
    mask_id = spec.get("mask_id")
    if not isinstance(mask_id, str) or not mask_id:
        raise PersonaMaskError("mask_id must be non-empty")
    redactions = spec.get("redactions")
    if not isinstance(redactions, list) or not redactions:
        raise PersonaMaskError("redactions must be a non-empty list")
    expected_terms = spec.get("expected_answer_terms")
    if not (
        isinstance(expected_terms, list)
        and expected_terms
        and all(isinstance(term, str) and term for term in expected_terms)
    ):
        raise PersonaMaskError("expected_answer_terms must contain non-empty strings")

    masked = deepcopy(persona)
    for redaction in redactions:
        _apply_redaction(masked, redaction)
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    audit = {
        "schema_version": MASK_SCHEMA_VERSION,
        "mask_id": mask_id,
        "spec_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "redaction_count": len(redactions),
        # Private audit metadata; it is never copied into Actor messages or SFT rows.
        "expected_answer_terms": list(expected_terms),
    }
    return masked, audit


def _apply_redaction(root, redaction):
    if not isinstance(redaction, dict):
        raise PersonaMaskError("each redaction must be an object")
    path = redaction.get("path")
    old = redaction.get("old")
    replacement = redaction.get("replacement", "")
    if not isinstance(path, list) or not path:
        raise PersonaMaskError("redaction path must be a non-empty list")
    if not isinstance(old, str) or not old:
        raise PersonaMaskError("redaction old text must be non-empty")
    if not isinstance(replacement, str):
        raise PersonaMaskError("redaction replacement must be a string")

    parent = root
    for key in path[:-1]:
        try:
            parent = parent[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise PersonaMaskError(f"redaction path does not exist: {path}") from exc
    final_key = path[-1]
    try:
        value = parent[final_key]
    except (KeyError, IndexError, TypeError) as exc:
        raise PersonaMaskError(f"redaction path does not exist: {path}") from exc
    if not isinstance(value, str):
        raise PersonaMaskError(f"redaction target must be a string: {path}")
    if value.count(old) != 1:
        raise PersonaMaskError(f"redaction text must occur exactly once: {path}")
    parent[final_key] = value.replace(old, replacement, 1)
