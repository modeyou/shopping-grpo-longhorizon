"""Validate and promote the accepted multi-turn SFT data into data/sft."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from shopping_grpo.evaluation.artifacts import write_json_atomic


SFT_DATA_SCHEMA = "shopping-multiturn-sft-mix-v2"
SFT_PROMOTION_SCHEMA = "shopping-formal-sft-promotion-v1"
REWARD_VERSION = "shopsimulator-reward-v4"
FORMAL_SFT_MANIFEST_SHA256 = (
    "11be05b2d4e2cfb49529542a23030988e21ea59266cad97b48598302e56e4eeb"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def validate_formal_sft_source(
    source: str | Path,
    *,
    expected_manifest_sha256: str = FORMAL_SFT_MANIFEST_SHA256,
) -> dict:
    source = Path(source).resolve()
    manifest_path = source / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "formal SFT manifest hash mismatch: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SFT_DATA_SCHEMA:
        raise ValueError(f"formal SFT manifest must use {SFT_DATA_SCHEMA}")
    if manifest.get("reward") != REWARD_VERSION:
        raise ValueError(f"formal SFT manifest must use {REWARD_VERSION}")
    if (manifest.get("split") or {}).get("task_disjoint") is not True:
        raise ValueError("formal SFT manifest must certify task-disjoint splits")
    if (
        (manifest.get("evaluation_exclusion") or {}).get(
            "selected_overlap_count"
        )
        != 0
    ):
        raise ValueError("formal SFT manifest contains evaluation overlap")

    artifacts = manifest.get("artifacts") or {}
    validated = {}
    for name in ("train.jsonl", "validation.jsonl"):
        path = source / name
        detail = artifacts.get(name) or {}
        if not path.is_file():
            raise ValueError(f"formal SFT artifact is missing: {path}")
        if detail.get("sha256") != sha256_file(path):
            raise ValueError(f"formal SFT artifact hash mismatch: {name}")
        if int(detail.get("rows", -1)) != _row_count(path):
            raise ValueError(f"formal SFT artifact row count mismatch: {name}")
        validated[name] = {
            "path": path,
            "sha256": detail["sha256"],
            "rows": detail["rows"],
        }
    return {
        "source": source,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "artifacts": validated,
    }


def _copy_atomic(source: Path, destination: Path) -> None:
    temporary = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def promote_formal_sft_data(
    source: str | Path,
    destination: str | Path,
    *,
    repo_root: str | Path,
    expected_manifest_sha256: str = FORMAL_SFT_MANIFEST_SHA256,
) -> dict:
    repo_root = Path(repo_root).resolve()
    source_detail = validate_formal_sft_source(
        source,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    destination = Path(destination).resolve()
    try:
        source_detail["source"].relative_to(repo_root)
        destination.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("formal SFT source and destination must be in repository") from exc
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"formal SFT destination must be new or empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        _copy_atomic(source_detail["source"] / name, destination / name)

    promotion = {
        "schema_version": SFT_PROMOTION_SCHEMA,
        "status": "accepted",
        "source": {
            "path": source_detail["source"].relative_to(repo_root).as_posix(),
            "manifest_sha256": source_detail["manifest_sha256"],
        },
        "destination": destination.relative_to(repo_root).as_posix(),
        "artifacts": {
            name: {
                "path": f"{destination.relative_to(repo_root).as_posix()}/{name}",
                "sha256": sha256_file(destination / name),
                **(
                    {"rows": source_detail["artifacts"][name]["rows"]}
                    if name in source_detail["artifacts"]
                    else {}
                ),
            }
            for name in ("train.jsonl", "validation.jsonl", "manifest.json")
        },
        "validation": {
            "source_preserved": True,
            "byte_identical_copy": True,
            "task_disjoint": True,
            "evaluation_overlap_count": 0,
        },
    }
    write_json_atomic(destination / "promotion.json", promotion)
    return promotion
