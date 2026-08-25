import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.check_bpo_runtime import validate_shopper_api, validate_swanlab


class _ConfigNode(dict):
    __getattr__ = dict.__getitem__


def _config(*loggers):
    return SimpleNamespace(
        trainer=_ConfigNode({
            "logger": list(loggers),
            "project_name": "shopping-multiturn-agentic",
        })
    )


def test_swanlab_preflight_performs_real_login(monkeypatch, capsys):
    calls = []
    swanlab = ModuleType("swanlab")
    swanlab.login = lambda key: calls.append(key) or True
    monkeypatch.setitem(sys.modules, "swanlab", swanlab)
    monkeypatch.setenv("SWANLAB_MODE", "online")
    monkeypatch.setenv("SWANLAB_API_KEY", "secret")

    validate_swanlab(_config("console", "swanlab"))

    assert calls == ["secret"]
    output = capsys.readouterr().out
    assert "BPO SwanLab authentication preflight passed" in output
    assert "secret" not in output


def test_swanlab_preflight_rejects_remote_authentication_failure(monkeypatch):
    swanlab = ModuleType("swanlab")

    def reject(_key):
        raise RuntimeError("401 Unauthorized")

    swanlab.login = reject
    monkeypatch.setitem(sys.modules, "swanlab", swanlab)
    monkeypatch.setenv("SWANLAB_MODE", "online")
    monkeypatch.setenv("SWANLAB_API_KEY", "invalid")

    with pytest.raises(SystemExit, match="SwanLab authentication failed"):
        validate_swanlab(_config("console", "swanlab"))


def test_console_only_preflight_does_not_contact_swanlab(monkeypatch):
    monkeypatch.delitem(sys.modules, "swanlab", raising=False)
    monkeypatch.delenv("SWANLAB_MODE", raising=False)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)

    validate_swanlab(_config("console"))


def test_shopper_preflight_uses_formal_credentials(monkeypatch):
    calls = []
    monkeypatch.setenv("SHOPPER_BASE_URL", "https://shopper.example/v1")
    monkeypatch.setenv("SHOPPER_API_KEY", "shopper-secret")
    monkeypatch.setenv("SHOPPER_MODEL", "deepseek-v4-flash-0731")
    monkeypatch.setattr(
        "scripts.run_sft_checkpoint_sweep.validate_shopper_api",
        lambda **kwargs: calls.append(kwargs),
    )

    validate_shopper_api()

    assert calls == [
        {
            "base_url": "https://shopper.example/v1",
            "api_key": "shopper-secret",
            "model": "deepseek-v4-flash-0731",
            "timeout": 30,
        }
    ]


def test_shopper_preflight_rejects_remote_authentication_failure(monkeypatch):
    monkeypatch.setenv("SHOPPER_BASE_URL", "https://shopper.example/v1")
    monkeypatch.setenv("SHOPPER_API_KEY", "invalid")
    monkeypatch.setenv("SHOPPER_MODEL", "deepseek-v4-flash-0731")

    def reject(**_kwargs):
        raise RuntimeError("Shopper API preflight failed with HTTP 403")

    monkeypatch.setattr(
        "scripts.run_sft_checkpoint_sweep.validate_shopper_api",
        reject,
    )

    with pytest.raises(SystemExit, match="Shopper API authentication failed"):
        validate_shopper_api()
