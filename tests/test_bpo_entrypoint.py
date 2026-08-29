from argparse import Namespace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import train_bpo


def test_bpo_launcher_uses_independent_entrypoint_and_native_v4(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    manifest = tmp_path / "manifest.json"
    for path in (train, validation, manifest):
        path.write_bytes(b"fixture")
    monkeypatch.setenv("SHOPPER_API_KEY", "secret")
    args = Namespace(
        model=model,
        train_data=train,
        val_data=validation,
        data_manifest=manifest,
        env_url="http://127.0.0.1:5700",
        shopper_model="deepseek-v4-flash-0731",
        shopper_base_url="https://shopper.invalid/v1",
        output=tmp_path / "output",
        experiment_name="bpo-test",
        logger="console",
        seed=20260823,
        diagnostic_steps=None,
        dry_run=False,
        preflight_only=True,
        hydra_overrides=[],
    )
    with patch.object(train_bpo, "validate_grpo_data_manifest"):
        command, environment, audit = train_bpo.build(args)
    assert command[1:3] == ["-m", "shopping_grpo.training.bpo.entrypoint"]
    assert environment["SHOPPING_REWARD_SHAPING_PROFILE"] == "none"
    assert environment["SHOP_REWARD_VERSION"] == "shopsimulator-reward-v4"
    assert environment["GRPO_CONFIG_NAME"] == "bpo"
    assert environment["SHOPPING_BPO_REQUIRE_PARAMETER_UPDATE"] == "1"
    assert environment["SHOPPING_BPO_SCHEDULER_HORIZON"] == "500"
    assert environment["SHOPPING_BPO_WARMUP_STEPS"] == "10"
    assert environment["SHOPPING_BPO_MIN_LR_RATIO"] == "0.1"
    assert Path(environment["SHOPPING_ENV_MANIFEST"]).resolve() == (
        train_bpo.BPO_RUNTIME_MANIFEST.resolve()
    )
    assert audit["algorithm"] == "carl-bpo-v1"
    assert audit["reward_profile"] == "none"
    assert audit["execution_mode"] == "formal"

    step0_contract = json.loads(environment["SHOPPING_BPO_STEP0_CONTRACT_JSON"])
    step0_digest = train_bpo.validate_contract(step0_contract)
    assert environment["SHOPPING_BPO_STEP0_CONTRACT_SHA256"] == step0_digest
    assert Path(environment["SHOPPING_BPO_STEP0_CACHE_PATH"]).name == (
        f"{step0_digest}.json"
    )
    assert audit["step0_validation"]["contract_sha256"] == step0_digest

    args.output.mkdir()
    destination = train_bpo.write_contract(environment, audit)
    run_contract = json.loads(destination.read_text(encoding="utf-8"))
    frozen_step0 = json.loads(
        (args.output / "step0_validation_contract.json").read_text(encoding="utf-8")
    )
    assert run_contract["step0_validation"]["contract_sha256"] == step0_digest
    assert run_contract["step0_validation"]["reuse_policy"] == (
        "exact-contract-sha256-v1"
    )
    assert run_contract["frozen_method"]["fused_ppo_input_gradient_backport"] == (
        "ctx-needs-input-grad-v1"
    )
    assert "bpo_fused_ppo_gradient_patch" in run_contract["inputs"]

def test_bpo_launcher_rejects_an_external_ray_address(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:26379")
    with pytest.raises(SystemExit, match="launcher-owned local Ray runtime"):
        train_bpo.validate_launcher_owned_ray()


def test_bpo_diagnostic_mode_owns_safe_one_step_overrides():
    args = Namespace(
        logger="console",
        experiment_name="bpo-diagnostic",
        diagnostic_steps=1,
        hydra_overrides=[],
    )

    assert train_bpo._overrides(args) == [
        "trainer.logger=[console]",
        "trainer.experiment_name=bpo-diagnostic",
        "trainer.total_training_steps=1",
        "trainer.val_before_train=false",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
    ]


def test_bpo_diagnostic_mode_rejects_conflicting_step_override():
    args = Namespace(
        logger="console",
        experiment_name="bpo-diagnostic",
        diagnostic_steps=1,
        hydra_overrides=["--", "trainer.total_training_steps=2"],
    )

    with pytest.raises(SystemExit, match="owns trainer step/save/test overrides"):
        train_bpo._overrides(args)
