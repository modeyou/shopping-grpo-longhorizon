"""Canonical semantic identities for CARL-BPO branch actions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for


@dataclass(frozen=True)
class SemanticAction:
    """One parsed tool action independent of reasoning and protocol formatting."""

    tool: str
    canonical_key: str
    sha256: str


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic action arguments contain NaN/Inf")
        return value
    raise ValueError(
        "semantic action arguments contain an unsupported value: "
        f"{type(value).__name__}"
    )


def canonical_semantic_action(
    tool_calls: Sequence[object] | None,
    *,
    allowed_tools: Sequence[str] | None = None,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> SemanticAction | None:
    """Return one canonical tool action, or ``None`` for an invalid action.

    The key deliberately excludes assistant reasoning and serialized XML.  It
    contains only the normalized tool name and parsed JSON arguments.
    """

    calls = list(tool_calls or ())
    if len(calls) != 1:
        return None
    tool = str(getattr(calls[0], "name", "")).strip().lower()
    if not tool:
        return None
    if allowed_tools is not None and tool not in {
        str(name).strip().lower() for name in allowed_tools
    }:
        return None
    raw_arguments = getattr(calls[0], "arguments", "{}")
    if raw_arguments in (None, ""):
        raw_arguments = "{}"
    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )
        if not isinstance(arguments, Mapping):
            return None
        normalized_arguments = _normalize(arguments)
        if tool_schemas is not None:
            parameter_schema = None
            for raw_schema in tool_schemas:
                function = raw_schema.get("function", raw_schema)
                if str(function.get("name", "")).strip().lower() == tool:
                    parameter_schema = function.get("parameters")
                    break
            if not isinstance(parameter_schema, Mapping):
                return None
            validator_class = validator_for(parameter_schema)
            validator_class.check_schema(parameter_schema)
            validator_class(parameter_schema).validate(normalized_arguments)
        payload = {
            "tool": tool,
            "arguments": normalized_arguments,
        }
        key = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
        SchemaError,
        ValidationError,
    ):
        return None
    return SemanticAction(
        tool=tool,
        canonical_key=key,
        sha256=hashlib.sha256(key.encode("utf-8")).hexdigest(),
    )
