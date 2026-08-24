from argparse import Namespace
from unittest.mock import patch

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
        seed=20260824,
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
    assert environment["SHOPPING_ENV_MANIFEST"].endswith(
        "data/environment-bpo-v1.json"
    )
    assert audit["algorithm"] == "full-bpo-v1"
    assert audit["reward_profile"] == "none"
