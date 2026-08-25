from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shopping_grpo.training.bpo.step0_validation import (
    CACHE_PATH_ENV,
    CONTRACT_SHA256_ENV,
    REFRESH_ENV,
    build_validation_contract,
    freeze_validation_cache,
    install_step0_validation_cache,
    load_validation_cache,
    validate_contract,
)


def test_step0_contract_is_content_addressed_and_input_sensitive(tmp_path):
    source = tmp_path / "validation.parquet"
    source.write_bytes(b"frozen-validation")
    contract = build_validation_contract(
        root=tmp_path,
        git_commit="a" * 40,
        inputs={"validation": source},
        settings={"reward": "shopsimulator-reward-v4"},
    )

    digest = validate_contract(contract)

    assert digest == contract["contract_sha256"]
    assert contract["inputs"]["validation"]["path"] == "validation.parquet"
    source.write_bytes(b"changed")
    changed = build_validation_contract(
        root=tmp_path,
        git_commit="a" * 40,
        inputs={"validation": source},
        settings={"reward": "shopsimulator-reward-v4"},
    )
    assert changed["contract_sha256"] != digest


def test_step0_cache_round_trip_and_contract_mismatch(tmp_path):
    path = tmp_path / "cache.json"
    metrics = freeze_validation_cache(
        path,
        contract_sha256_value="b" * 64,
        metrics={"val-shopping/reward/strict_mean": 0.625},
    )

    assert metrics == {"val-shopping/reward/strict_mean": 0.625}
    assert load_validation_cache(
        path, expected_contract_sha256="b" * 64
    ) == metrics
    with pytest.raises(ValueError, match="contract mismatch"):
        load_validation_cache(path, expected_contract_sha256="c" * 64)


def test_step0_cache_rejects_nonfinite_or_corrupt_metrics(tmp_path):
    with pytest.raises(ValueError, match="not finite"):
        freeze_validation_cache(
            tmp_path / "nan.json",
            contract_sha256_value="d" * 64,
            metrics={"bad": float("nan")},
        )
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cache schema"):
        load_validation_cache(corrupt, expected_contract_sha256="d" * 64)


def test_real_validate_wrapper_freezes_then_replays_to_logger(
    tmp_path, monkeypatch, capsys
):
    calls = []

    class _RayPPOTrainer:
        def _validate(self, merged=False):
            calls.append(merged)
            return {"val-shopping/reward/strict_mean": 0.75}

    module = ModuleType("verl.trainer.ppo.ray_trainer")
    module.RayPPOTrainer = _RayPPOTrainer
    for name in ("verl", "verl.trainer", "verl.trainer.ppo"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)

    cache = tmp_path / f"{'e' * 64}.json"
    monkeypatch.setenv(CACHE_PATH_ENV, str(cache))
    monkeypatch.setenv(CONTRACT_SHA256_ENV, "e" * 64)
    monkeypatch.setenv(REFRESH_ENV, "0")
    install_step0_validation_cache()
    trainer = SimpleNamespace(global_steps=0)

    first = _RayPPOTrainer._validate(trainer)
    second = _RayPPOTrainer._validate(trainer)

    assert calls == [False]
    assert first["val-step0/cache_hit"] == 0.0
    assert second["val-step0/cache_hit"] == 1.0
    assert second["val-shopping/reward/strict_mean"] == 0.75
    assert json.loads(cache.read_text(encoding="utf-8"))["metrics"] == {
        "val-shopping/reward/strict_mean": 0.75
    }
    output = capsys.readouterr().out
    assert "BPO step-0 validation cache frozen" in output
    assert "BPO step-0 validation cache reused" in output


def test_validate_wrapper_never_reuses_after_step_zero(tmp_path, monkeypatch):
    calls = []

    class _RayPPOTrainer:
        def _validate(self, merged=False):
            calls.append(merged)
            return {"later": 1.0}

    module = ModuleType("verl.trainer.ppo.ray_trainer")
    module.RayPPOTrainer = _RayPPOTrainer
    for name in ("verl", "verl.trainer", "verl.trainer.ppo"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(CACHE_PATH_ENV, str(tmp_path / "cache.json"))
    monkeypatch.setenv(CONTRACT_SHA256_ENV, "f" * 64)
    monkeypatch.setenv(REFRESH_ENV, "0")
    install_step0_validation_cache()

    assert _RayPPOTrainer._validate(SimpleNamespace(global_steps=10)) == {
        "later": 1.0
    }
    assert calls == [False]
