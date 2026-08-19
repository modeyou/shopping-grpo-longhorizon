"""Export task facts from the frozen embedded ShopSimulator data."""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from shopping_grpo.personalization.schema import SOURCE_SCHEMA_VERSION, stable_hash


DEFAULT_PRODUCT_DATA = Path(
    "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
)


def read_task_ids(path: str | Path) -> list[int]:
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id", row.get("id"))
            if not isinstance(task_id, int) or task_id < 0:
                raise ValueError(f"invalid task_id at {path}:{line_number}")
            result.append(task_id)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate task_id in {path}")
    return result


def _option_snapshot(raw: object) -> dict[str, list[dict]]:
    result = {}
    for axis, entries in (raw if isinstance(raw, Mapping) else {}).items():
        values = []
        for entry in entries or []:
            if not isinstance(entry, Mapping) or not str(entry.get("value") or "").strip():
                continue
            values.append(
                {
                    "value": str(entry["value"]).strip(),
                    "price": entry.get("price"),
                    "available": bool(entry.get("is_available", True)),
                }
            )
        if values:
            result[str(axis)] = values
    return result


def iter_source_tasks(product_data: str | Path = DEFAULT_PRODUCT_DATA) -> Iterator[dict]:
    """Yield the same task order used by ShopSimulator's non-persona goal loader."""

    with gzip.open(product_data, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    if not isinstance(products, list):
        raise ValueError("ShopSimulator product data must be a list")

    seen_asins = set()
    task_id = 0
    for product in products:
        if not isinstance(product, Mapping):
            continue
        asin = str(product.get("asin") or "")
        if asin == "nan" or not asin or len(asin) > 20 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        for instruction in product.get("instructions") or []:
            if not isinstance(instruction, Mapping) or not instruction.get("attributes"):
                continue
            row = {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "shopsim_task_id": task_id,
                "target_asin": asin,
                "category": str(product.get("category") or "").strip(),
                "title": str(product.get("title") or "").strip(),
                "shop_name": str(product.get("shop_name") or "").strip(),
                "pricing": product.get("pricing") or [],
                "attributes": list(instruction.get("attributes") or []),
                "required_options": instruction.get("instruction_options") or [],
                "available_options": _option_snapshot(product.get("customization_options")),
                "original_instruction": str(instruction.get("instruction") or "").strip(),
                "reference_simple_instruction": str(
                    instruction.get("instruction_simple") or ""
                ).strip(),
                "has_reference_persona": bool(product.get("user_persona")),
            }
            row["source_hash"] = stable_hash(row)
            yield row
            task_id += 1


def load_source_tasks(
    task_ids: Iterable[int],
    *,
    product_data: str | Path = DEFAULT_PRODUCT_DATA,
) -> list[dict]:
    requested = [int(task_id) for task_id in task_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("requested task_ids contains duplicates")
    wanted = set(requested)
    found = {}
    for row in iter_source_tasks(product_data):
        task_id = row["shopsim_task_id"]
        if task_id in wanted:
            found[task_id] = row
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - set(found))
    if missing:
        raise IndexError(f"ShopSimulator task IDs not found: {missing[:10]}")
    return [found[task_id] for task_id in requested]


def select_source_tasks(
    task_ids: Iterable[int],
    *,
    count: int,
    seed: int,
) -> list[int]:
    values = list(task_ids)
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(values):
        raise ValueError("count exceeds available task IDs")
    random.Random(seed).shuffle(values)
    return values[:count]


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
