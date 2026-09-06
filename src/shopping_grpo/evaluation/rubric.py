"""Query-only Rubric drafting and immutable bundle materialization."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

from shopping_grpo.evaluation.contracts import (
    RUBRIC_SCHEMA_VERSION,
    ContractValidationError,
    validate_curator_response,
    validate_rubric_bundle,
)


TASK_FACTS_VERSION = "shopping-query-facts-v1"
RUBRIC_CURATOR_VERSION = "shopping-query-rubric-curator-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def build_task_facts(*, task_id: int, query: str) -> dict:
    """Create the only task record permitted to reach the Rubric curator.

    Target products, Reward fields, Gold items, and environment-private
    annotations deliberately do not appear in this record.
    """

    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError(f"task {task_id} has no Query")
    payload = {
        "schema_version": TASK_FACTS_VERSION,
        "task_id": int(task_id),
        "query": normalized_query,
    }
    payload["task_data_hash"] = stable_hash(payload)
    payload["query_hash"] = stable_hash(normalized_query)
    return payload


def _spans_from_quote(query: str, quote: str) -> list[dict]:
    quote = str(quote or "").strip()
    if not quote:
        return []
    return [
        {
            "text": query[match.start() : match.end()],
            "start": match.start(),
            "end": match.end(),
        }
        for match in re.finditer(re.escape(quote), query, flags=re.IGNORECASE)
    ]


def materialize_rubric_bundle(
    *,
    task_facts: Mapping,
    curator_response: Mapping,
    curator_model: str,
    curator_prompt_version: str,
    rubric_version: str,
) -> dict:
    """Freeze one Query-only curator response into a validated Rubric bundle."""

    if task_facts.get("schema_version") != TASK_FACTS_VERSION:
        raise ContractValidationError("unsupported query facts schema")
    query = str(task_facts.get("query") or "")
    response = validate_curator_response(curator_response, query=query)

    rubrics = []
    for requirement in response["requirements"]:
        quote_spans = _spans_from_quote(query, requirement["query_quote"])
        if not quote_spans:
            raise ContractValidationError(
                "requirement query_quote must produce at least one Query span"
            )
        rubrics.append(
            {
                "rubric_id": f"r{len(rubrics) + 1:04d}",
                "rubric_source": "query_only_llm",
                "description": requirement["description"].strip(),
                "acceptance_criteria": requirement[
                    "acceptance_criteria"
                ].strip(),
                "hardness": requirement["hardness"],
                "hardness_source": "curator_query_interpretation",
                "query_spans": quote_spans,
                "data_sources": ["query"],
                "selection_reason": requirement["selection_reason"].strip(),
            }
        )

    bundle = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_version": str(rubric_version),
        "task_id": int(task_facts["task_id"]),
        "query": query,
        "generation": {
            "curator_version": RUBRIC_CURATOR_VERSION,
            "curator_model": str(curator_model),
            "curator_prompt_version": str(curator_prompt_version),
            "task_data_hash": str(task_facts["task_data_hash"]),
            "query_hash": str(task_facts["query_hash"]),
        },
        "rubrics": rubrics,
    }
    return validate_rubric_bundle(bundle)
