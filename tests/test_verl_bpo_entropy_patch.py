import pytest

from scripts.apply_verl_bpo_patch import EXPECTED_PATCHED_SHA256, verify
from shopping_grpo.training.bpo.entropy_patch import PATCH_MARKER, patch_source


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
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify(target)
    assert EXPECTED_PATCHED_SHA256 == (
        "f99cd883946cdae4ade97871ef8b44c063529f21232f446d22e0e2b9ad701570"
    )
