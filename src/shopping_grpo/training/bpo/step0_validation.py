"""Freeze and safely reuse deterministic step-0 BPO validation metrics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from inspect import signature
import json
import math
import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4


CONTRACT_SCHEMA_VERSION = "shopping-bpo-step0-validation-contract-v1"
CACHE_SCHEMA_VERSION = "shopping-bpo-step0-validation-cache-v1"
CACHE_PATH_ENV = "SHOPPING_BPO_STEP0_CACHE_PATH"
CONTRACT_SHA256_ENV = "SHOPPING_BPO_STEP0_CONTRACT_SHA256"
REFRESH_ENV = "SHOPPING_BPO_STEP0_REFRESH"
_VALIDATION_HOOK_MARKER = "SHOPPING_BPO_STEP0_VALIDATION_CACHE_V1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256(contract: Mapping[str, object]) -> str:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("invalid BPO step-0 validation contract schema")
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


def build_validation_contract(
    *,
    root: str | Path,
    git_commit: str,
    inputs: Mapping[str, str | Path],
    settings: Mapping[str, object],
) -> dict[str, object]:
    """Build a deterministic contract from exact validation inputs and settings."""
    project_root = Path(root).resolve()
    frozen_inputs = {}
    for name, raw_path in sorted(inputs.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"step-0 validation input is missing: {path}")
        try:
            display_path = path.relative_to(project_root).as_posix()
        except ValueError:
            display_path = str(path)
        frozen_inputs[str(name)] = {
            "path": display_path,
            "sha256": sha256_file(path),
        }
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "git_commit": str(git_commit),
        "inputs": frozen_inputs,
        "settings": dict(settings),
    }
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def validate_contract(contract: Mapping[str, object]) -> str:
    expected = str(contract.get("contract_sha256") or "")
    if len(expected) != 64:
        raise ValueError("BPO step-0 validation contract is missing its SHA256")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    actual = contract_sha256(unsigned)
    if actual != expected:
        raise ValueError(
            "BPO step-0 validation contract hash mismatch: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _normalize_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    normalized = {}
    for raw_name, raw_value in metrics.items():
        name = str(raw_name)
        value = raw_value
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        if not isinstance(value, (bool, int, float)):
            raise ValueError(
                f"step-0 validation metric {name!r} is not a scalar: "
                f"{type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"step-0 validation metric {name!r} is not finite")
        normalized[name] = number
    if not normalized:
        raise ValueError("step-0 validation produced no scalar metrics")
    return normalized


def load_validation_cache(
    path: str | Path, *, expected_contract_sha256: str
) -> dict[str, float] | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    try:
        record = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid BPO step-0 validation cache: {cache_path}") from exc
    if record.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("invalid BPO step-0 validation cache schema")
    actual_contract = str(record.get("contract_sha256") or "")
    if actual_contract != str(expected_contract_sha256):
        raise ValueError(
            "BPO step-0 validation cache contract mismatch: "
            f"expected {expected_contract_sha256}, got {actual_contract}"
        )
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("BPO step-0 validation cache is missing metrics")
    return _normalize_metrics(metrics)


def freeze_validation_cache(
    path: str | Path,
    *,
    contract_sha256_value: str,
    metrics: Mapping[str, object],
) -> dict[str, float]:
    """Atomically publish one validated cache record."""
    normalized = _normalize_metrics(metrics)
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": str(contract_sha256_value),
        "metrics": normalized,
    }
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def install_step0_validation_cache() -> None:
    """Wrap veRL validation so a matching step-0 result is returned to its logger."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    current = RayPPOTrainer._validate
    if getattr(current, "_shopping_bpo_marker", None) == _VALIDATION_HOOK_MARKER:
        return
    parameters = tuple(signature(current).parameters)
    if parameters != ("self", "merged"):
        raise RuntimeError(
            "unsupported RayPPOTrainer._validate signature for BPO step-0 cache: "
            f"{parameters}"
        )


    def validate(trainer, *args, **kwargs):
        cache_path = os.environ.get(CACHE_PATH_ENV)
        expected_contract = os.environ.get(CONTRACT_SHA256_ENV)
        if not cache_path and not expected_contract:
            return current(trainer, *args, **kwargs)
        if not cache_path or not expected_contract:
            raise RuntimeError(
                "BPO step-0 validation cache requires both path and contract SHA256"
            )
        global_step = int(getattr(trainer, "global_steps", -1))
        if global_step != 0:
            return current(trainer, *args, **kwargs)

        refresh = os.environ.get(REFRESH_ENV) == "1"
        cached = None
        if not refresh:
            cached = load_validation_cache(
                cache_path,
                expected_contract_sha256=expected_contract,
            )
        if cached is None:
            metrics = current(trainer, *args, **kwargs)
            cached = freeze_validation_cache(
                cache_path,
                contract_sha256_value=expected_contract,
                metrics=metrics,
            )
            cache_hit = 0.0
            event = "frozen"
        else:
            cache_hit = 1.0
            event = "reused"

        result = dict(cached)
        result.update(
            {
                "val-step0/cache_hit": cache_hit,
                "val-step0/contract_verified": 1.0,
                "val-step0/metric_count": float(len(cached)),
            }
        )
        print(
            "BPO step-0 validation cache "
            + event
            + ": "
            + _canonical_json(
                {
                    "cache": str(Path(cache_path).resolve()),
                    "contract_sha256": expected_contract,
                    "metrics": len(cached),
                }
            )
        )
        return result

    validate._shopping_bpo_marker = _VALIDATION_HOOK_MARKER
    RayPPOTrainer._validate = validate
