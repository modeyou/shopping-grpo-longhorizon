"""Machine-verifiable contracts for the formal multi-turn GRPO dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import load_unique_task_ids


SELECTION_SCHEMA = "shopping-multiturn-grpo-selection-v2"
DATASET_SCHEMA = "shopping-multiturn-grpo-dataset-v2"
REWARD_VERSION = "shopsimulator-reward-v4"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: str | Path, root: str | Path) -> str:
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact must stay inside repository root: {resolved}") from exc


def resolve_repo_path(recorded: str, root: str | Path) -> Path:
    candidate = Path(recorded)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"manifest path must be repository-relative: {recorded}")
    root = Path(root).resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes repository root: {recorded}") from exc
    return resolved


def deterministic_active_split(
    reservoir_ids: set[int],
    excluded_ids: set[int],
    *,
    seed: int,
    train_count: int,
    validation_count: int,
) -> dict[str, list[int]]:
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation counts must be positive")
    eligible = reservoir_ids - excluded_ids
    required = train_count + validation_count
    if len(eligible) < required:
        raise ValueError(
            f"only {len(eligible)} eligible reservoir tasks for {required} requested"
        )

    def rank(task_id: int) -> bytes:
        return hashlib.sha256(f"{int(seed)}:{task_id}".encode("utf-8")).digest()

    ordered = sorted(eligible, key=lambda task_id: (rank(task_id), task_id))
    validation = ordered[:validation_count]
    train = ordered[validation_count : validation_count + train_count]
    unused = ordered[validation_count + train_count :]
    return {"validation": validation, "train": train, "unused": unused}


def _validate_recorded_artifact(
    detail: dict,
    *,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(detail, dict) or not detail.get("path"):
        raise ValueError(f"GRPO manifest {label} is missing")
    recorded = resolve_repo_path(str(detail["path"]), root)
    if not recorded.is_file():
        raise ValueError(f"GRPO manifest {label} is missing: {recorded}")
    if detail.get("sha256") != sha256_file(recorded):
        raise ValueError(f"GRPO manifest {label} hash mismatch")
    return recorded


def _validate_artifact(
    detail: dict,
    expected_path: Path,
    *,
    root: Path,
    label: str,
) -> None:
    recorded = _validate_recorded_artifact(detail, root=root, label=label)
    if recorded != expected_path.resolve():
        raise ValueError(
            f"GRPO manifest {label} path mismatch: {recorded} != {expected_path.resolve()}"
        )


def validate_grpo_data_manifest(
    manifest_path: str | Path,
    *,
    train_data: str | Path,
    validation_data: str | Path,
    environment_manifest: str | Path,
    root: str | Path,
) -> dict:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid GRPO data manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise ValueError(f"GRPO data manifest must use {DATASET_SCHEMA}")
    if manifest.get("status") != "accepted":
        raise ValueError("GRPO data manifest status must be accepted")
    if manifest.get("reward_version") != REWARD_VERSION:
        raise ValueError(f"GRPO data manifest must use {REWARD_VERSION}")

    environment = manifest.get("environment") or {}
    _validate_artifact(
        environment,
        Path(environment_manifest),
        root=root,
        label="environment",
    )
    _validate_recorded_artifact(
        manifest.get("selection_source") or {},
        root=root,
        label="selection source",
    )
    _validate_recorded_artifact(
        manifest.get("reward_reachability_audit") or {},
        root=root,
        label="Reward v4 reachability audit",
    )
    _validate_recorded_artifact(
        manifest.get("source_reservoir") or {},
        root=root,
        label="source reservoir",
    )
    for label, detail in (manifest.get("openings") or {}).items():
        _validate_recorded_artifact(
            detail,
            root=root,
            label=f"opening {label}",
        )
    if not manifest.get("openings"):
        raise ValueError("GRPO manifest openings are missing")
    for detail in manifest.get("exclusions") or []:
        _validate_recorded_artifact(
            detail,
            root=root,
            label=f"exclusion {detail.get('label', 'unknown')}",
        )
    if not manifest.get("exclusions"):
        raise ValueError("GRPO manifest exclusions are missing")
    artifacts = manifest.get("artifacts") or {}
    _validate_artifact(
        artifacts.get("train") or {},
        Path(train_data),
        root=root,
        label="train parquet",
    )
    _validate_artifact(
        artifacts.get("validation") or {},
        Path(validation_data),
        root=root,
        label="validation parquet",
    )

    selection = manifest.get("selection") or {}
    train_tasks = selection.get("train") or {}
    validation_tasks = selection.get("validation") or {}
    train_task_path = resolve_repo_path(str(train_tasks.get("path", "")), root)
    validation_task_path = resolve_repo_path(
        str(validation_tasks.get("path", "")), root
    )
    _validate_artifact(
        train_tasks, train_task_path, root=root, label="train task IDs"
    )
    _validate_artifact(
        validation_tasks,
        validation_task_path,
        root=root,
        label="validation task IDs",
    )
    train_ids = load_unique_task_ids(train_task_path)
    validation_ids = load_unique_task_ids(validation_task_path)
    if train_ids & validation_ids:
        raise ValueError("GRPO train and validation task IDs overlap")
    if int(train_tasks.get("tasks", -1)) != len(train_ids):
        raise ValueError("GRPO train task count mismatch")
    if int(validation_tasks.get("tasks", -1)) != len(validation_ids):
        raise ValueError("GRPO validation task count mismatch")
    if int((artifacts.get("train") or {}).get("tasks", -1)) != len(train_ids):
        raise ValueError("GRPO train parquet task count mismatch")
    if int((artifacts.get("validation") or {}).get("tasks", -1)) != len(
        validation_ids
    ):
        raise ValueError("GRPO validation parquet task count mismatch")
    if int((artifacts.get("train") or {}).get("rows", -1)) != 2 * len(train_ids):
        raise ValueError("GRPO train parquet must contain gap and complete rows")
    if int((artifacts.get("validation") or {}).get("rows", -1)) != 2 * len(
        validation_ids
    ):
        raise ValueError(
            "GRPO validation parquet must contain gap and complete rows"
        )
    audit = manifest.get("audit") or {}
    if audit.get("train_validation_overlap_count") != 0:
        raise ValueError("GRPO manifest does not certify train/validation disjointness")
    if audit.get("selected_exclusion_overlap_count") != 0:
        raise ValueError("GRPO manifest does not certify exclusion disjointness")
    if audit.get("all_selected_tasks_reward_reachable") is not True:
        raise ValueError("GRPO manifest does not certify Reward v4 reachability")
    return manifest
