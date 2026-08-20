"""Freeze task-disjoint multi-turn datasets from the ShopSimulator goal space."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from shopping_grpo.environment.manifest import validate_manifest


SPLIT_SCHEMA_VERSION = "shopping-multiturn-task-splits-v1"
GOAL_ORDER_SEED = 223
SPLIT_ORDER = (
    "evaluation",
    "sft_candidates",
    "grpo_validation",
    "grpo_train",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decompressed_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(Path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_shop_goals(path: str | Path) -> int:
    """Mirror the runtime's ASIN and non-empty-attribute goal filters."""

    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    if not isinstance(products, list):
        raise ValueError("ShopSimulator product data must contain a JSON list")

    seen_asins = set()
    count = 0
    for product in products:
        if not isinstance(product, dict):
            raise ValueError("ShopSimulator product rows must be JSON objects")
        asin = str(product.get("asin", ""))
        if asin == "nan" or len(asin) > 20 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        instructions = product.get("instructions") or []
        if not isinstance(instructions, list):
            raise ValueError("product instructions must be a list")
        count += sum(
            1
            for instruction in instructions
            if isinstance(instruction, dict) and instruction.get("attributes")
        )
    if count < 1:
        raise ValueError("ShopSimulator product data produced no goals")
    return count


def task_ids_from_file(path: str | Path) -> set[int]:
    path = Path(path)
    if path.suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and isinstance(document.get("task_ids"), list):
            return {int(task_id) for task_id in document["task_ids"]}
        if isinstance(document, dict) and "task_id" in document:
            return {int(document["task_id"])}
        raise ValueError(f"{path} must contain task_id or task_ids")

    task_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" not in row:
                raise ValueError(f"{path}:{line_number} is missing task_id")
            task_ids.add(int(row["task_id"]))
    return task_ids


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def stable_task_order(task_ids: Iterable[int], seed: int) -> list[int]:
    """Return a language-independent deterministic permutation."""

    def rank(task_id: int) -> bytes:
        return hashlib.sha256(f"{int(seed)}:{int(task_id)}".encode("utf-8")).digest()

    return sorted({int(task_id) for task_id in task_ids}, key=rank)


def build_task_splits(
    *,
    goal_count: int,
    excluded_task_ids: Iterable[int],
    split_sizes: Mapping[str, int],
    seed: int,
) -> dict[str, list[int]]:
    if int(goal_count) < 1:
        raise ValueError("goal_count must be positive")
    if set(split_sizes) != set(SPLIT_ORDER):
        raise ValueError("split_sizes must contain exactly: " + ", ".join(SPLIT_ORDER))
    sizes = {name: int(split_sizes[name]) for name in SPLIT_ORDER}
    if any(size < 1 for size in sizes.values()):
        raise ValueError("every frozen split must contain at least one task")

    excluded = {int(task_id) for task_id in excluded_task_ids}
    outside = sorted(
        task_id for task_id in excluded if task_id < 0 or task_id >= int(goal_count)
    )
    if outside:
        raise ValueError(f"excluded task IDs are outside the goal space: {outside[:10]}")

    candidates = set(range(int(goal_count))) - excluded
    required = sum(sizes.values())
    if required > len(candidates):
        raise ValueError(
            f"requested {required} tasks but only {len(candidates)} candidates remain"
        )

    ordered = stable_task_order(candidates, seed)
    splits = {}
    offset = 0
    for name in SPLIT_ORDER:
        next_offset = offset + sizes[name]
        splits[name] = ordered[offset:next_offset]
        offset = next_offset
    splits["reserve"] = ordered[offset:]
    validate_disjoint_splits(splits, excluded)
    return splits


def validate_disjoint_splits(
    splits: Mapping[str, Iterable[int]], excluded_task_ids: Iterable[int] = ()
) -> None:
    excluded = {int(task_id) for task_id in excluded_task_ids}
    seen = set()
    for name, values in splits.items():
        ids = [int(task_id) for task_id in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"split {name} contains duplicate task IDs")
        overlap = sorted(set(ids) & excluded)
        if overlap:
            raise ValueError(f"split {name} overlaps exclusions: {overlap[:10]}")
        cross_overlap = sorted(set(ids) & seen)
        if cross_overlap:
            raise ValueError(f"split {name} overlaps another split: {cross_overlap[:10]}")
        seen.update(ids)


def freeze_task_splits(
    *,
    product_data_path: str | Path,
    environment_manifest_path: str | Path,
    exclusion_paths: Iterable[str | Path],
    output_dir: str | Path,
    split_sizes: Mapping[str, int],
    seed: int,
) -> dict:
    product_data_path = Path(product_data_path)
    environment_manifest_path = Path(environment_manifest_path)
    output_dir = Path(output_dir)
    repository_root = environment_manifest_path.resolve().parent.parent

    environment_manifest = validate_manifest(
        json.loads(environment_manifest_path.read_text(encoding="utf-8"))
    )
    product_hash = decompressed_sha256(product_data_path)
    if product_hash != environment_manifest["product_data_sha256"]:
        raise ValueError(
            "decompressed product data hash does not match data/environment.json"
        )
    goal_count = count_shop_goals(product_data_path)

    excluded = set()
    exclusion_sources = []
    for source in exclusion_paths:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"missing exclusion source: {path}")
        ids = task_ids_from_file(path)
        excluded.update(ids)
        exclusion_sources.append({
            "path": _portable_path(path, repository_root),
            "sha256": sha256_file(path),
            "unique_task_ids": len(ids),
        })

    splits = build_task_splits(
        goal_count=goal_count,
        excluded_task_ids=excluded,
        split_sizes=split_sizes,
        seed=seed,
    )
    split_payloads = {
        name: "".join(
            json.dumps({"task_id": task_id}, separators=(",", ":")) + "\n"
            for task_id in ids
        ).encode("utf-8")
        for name, ids in splits.items()
    }
    split_metadata = {
        name: {
            "path": _portable_path(
                output_dir / f"{name}.jsonl", repository_root
            ),
            "tasks": len(ids),
            "sha256": hashlib.sha256(split_payloads[name]).hexdigest(),
        }
        for name, ids in splits.items()
    }
    metadata = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "environment": {
            "manifest_path": _portable_path(
                environment_manifest_path, repository_root
            ),
            "manifest_sha256": sha256_file(environment_manifest_path),
            "environment_version": environment_manifest.get(
                "environment_version", "shopsimulator-environment-v2.1"
            ),
            "shopsimulator_commit": environment_manifest["shopsimulator_commit"],
            "product_data_path": _portable_path(product_data_path, repository_root),
            "product_data_compressed_sha256": sha256_file(product_data_path),
            "product_data_decompressed_sha256": product_hash,
            "goal_order_seed": GOAL_ORDER_SEED,
            "goal_count": goal_count,
        },
        "selection": {
            "seed": int(seed),
            "method": "sha256(seed:task_id) ascending",
            "split_order": list(SPLIT_ORDER),
            "assignment_boundary": "candidate-pool-before-any-llm-call",
        },
        "exclusions": {
            "policy": "exclude all reference SFT/GRPO/evaluation task IDs",
            "sources": exclusion_sources,
            "unique_task_ids": len(excluded),
        },
        "splits": split_metadata,
        "validation": {
            "all_splits_pairwise_disjoint": True,
            "all_splits_disjoint_from_exclusions": True,
            "task_id_min": 0,
            "task_id_max": goal_count - 1,
        },
    }
    metadata_payload = (
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    planned = {
        output_dir / f"{name}.jsonl": payload
        for name, payload in split_payloads.items()
    }
    planned[output_dir / "metadata.json"] = metadata_payload

    mismatched = [
        path for path, payload in planned.items()
        if path.exists() and path.read_bytes() != payload
    ]
    if mismatched:
        raise FileExistsError(
            "frozen split artifacts differ; refusing to overwrite: "
            + ", ".join(str(path) for path in mismatched)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in planned.items():
        if not path.exists():
            path.write_bytes(payload)
    return metadata
