import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.apply_verl_bpo_patch as patcher
from scripts.apply_verl_bpo_patch import verify
from shopping_grpo.training.bpo.entropy_patch import (
    LEGACY_PATCH_MARKERS,
    PATCH_MARKER,
    patch_source,
)


SOURCE = (
    "import logging\n\n"
    "class Server:\n"
    "    async def generate(self, prompt_ids, sampling_params):\n"
    "        extra_fields = {}\n"
    "        prompt_ids = normalize_token_ids(prompt_ids)\n"
    '        sampling_params["logprobs"] = 0 if '
    'sampling_params.pop("logprobs", False) else None\n'
    "        final_res = await engine.generate(prompt_ids, sampling_params)\n"
    "        token_ids = final_res.outputs[0].token_ids\n"
    "        return token_ids, extra_fields\n"
)


def test_exact_entropy_patch_is_idempotent_and_scalarizes_distribution():
    patched = patch_source(SOURCE)
    assert patched.count(PATCH_MARKER) == 1
    assert 'sampling_params["logprobs"] = -1 if bpo_entropy_probe' in patched
    assert 'extra_fields["bpo_full_vocab_entropy"]' in patched
    assert "probability_mass" in patched
    assert "value == -math.inf" in patched
    assert "if probability > 0.0" in patched
    assert patch_source(patched) == patched


def test_exact_entropy_patch_rejects_unknown_source():
    try:
        patch_source("import logging\n")
    except ValueError as exc:
        assert "anchors mismatch" in str(exc)
    else:
        raise AssertionError("unknown veRL source must be rejected")


def test_entropy_patch_verifier_rejects_marker_only_source(tmp_path):
    target = tmp_path / "vllm_async_server.py"
    target.write_text(f"# {PATCH_MARKER}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="backup is missing"):
        verify(target)


def test_combined_installer_loads_sibling_xml_patcher_by_file_path():
    module = patcher.load_xml_patcher()
    assert Path(module.__file__).resolve() == (
        Path(patcher.__file__).resolve().with_name(
            "apply_verl_bpo_tool_parser_patch.py"
        )
    )
    assert callable(module.apply)


def test_zero_probability_logprob_does_not_create_nan():
    values = [math.log(0.5), math.log(0.5), -math.inf]
    probabilities = [
        0.0 if value == -math.inf else math.exp(value) for value in values
    ]
    entropy = -sum(
        probability * logprob
        for probability, logprob in zip(probabilities, values, strict=True)
        if probability > 0.0
    )
    assert math.isfinite(entropy)
    assert entropy == pytest.approx(math.log(2))


def test_generated_patch_handles_realistic_negative_infinity_logprob():
    class Engine:
        async def generate(self, prompt_ids, sampling_params):
            assert sampling_params["logprobs"] == -1
            distribution = {
                0: SimpleNamespace(logprob=math.log(0.5)),
                1: SimpleNamespace(logprob=math.log(0.5)),
                2: SimpleNamespace(logprob=-math.inf),
            }
            output = SimpleNamespace(token_ids=[0], logprobs=[distribution])
            return SimpleNamespace(outputs=[output])

    namespace = {
        "engine": Engine(),
        "normalize_token_ids": lambda value: value,
    }
    exec(patch_source(SOURCE), namespace)
    token_ids, extra_fields = asyncio.run(
        namespace["Server"]().generate(
            [1, 2, 3],
            {"logprobs": True, "bpo_entropy_probe": True},
        )
    )
    assert token_ids == [0]
    assert math.isfinite(extra_fields["bpo_full_vocab_entropy"])
    assert extra_fields["bpo_full_vocab_entropy"] == pytest.approx(math.log(2))


def test_apply_upgrades_v1_from_verified_original_backup(tmp_path, monkeypatch):
    target = tmp_path / "vllm_async_server.py"
    backup = tmp_path / f"vllm_async_server.py{patcher.BACKUP_SUFFIX}"
    backup.write_text(SOURCE, encoding="utf-8", newline="\n")
    monkeypatch.setattr(
        patcher,
        "EXPECTED_ORIGINAL_SHA256",
        patcher.sha256(backup),
    )
    legacy = patch_source(SOURCE).replace(PATCH_MARKER, LEGACY_PATCH_MARKERS[0])
    target.write_text(legacy, encoding="utf-8", newline="\n")

    patcher.apply(target)

    upgraded = target.read_text(encoding="utf-8")
    assert upgraded.count(PATCH_MARKER) == 1
    assert LEGACY_PATCH_MARKERS[0] not in upgraded
    patcher.verify(target)
