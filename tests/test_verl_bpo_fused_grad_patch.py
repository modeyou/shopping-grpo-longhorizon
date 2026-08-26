from pathlib import Path

import pytest

from scripts import apply_verl_bpo_fused_grad_patch as installer
from shopping_grpo.training.bpo.fused_ppo_grad_patch import (
    PATCH_MARKER,
    patch_source,
)


ALLOCATION = (
    "        # Allocate memory for outputs\n"
    "        dhidden_states = None\n"
    "        if hidden_states.requires_grad:\n"
    "            dhidden_states = torch.zeros_like(hidden_states)\n"
    "        dvocab_weights = None\n"
    "        if vocab_weights.requires_grad:\n"
    "            dvocab_weights = torch.zeros_like(vocab_weights)\n"
)
SOURCE = (
    "class FusedLinearForPPOFunction:\n"
    + ALLOCATION
    + "            if hidden_states.requires_grad:\n"
    + "                dhidden_states[chunk_start:chunk_end] += h\n"
    + "            if vocab_weights.requires_grad:\n"
    + "                dvocab_weights += v\n"
    + "        if orig_ndim == 3 and hidden_states.requires_grad:\n"
    + "            pass\n"
)


def test_fused_ppo_patch_uses_autograd_input_contract_and_is_idempotent():
    patched = patch_source(SOURCE)
    assert patched.count(PATCH_MARKER) == 1
    assert "needs_hidden_grad = ctx.needs_input_grad[0]" in patched
    assert "needs_vocab_grad = ctx.needs_input_grad[1]" in patched
    assert "if hidden_states.requires_grad:" not in patched
    assert "if vocab_weights.requires_grad:" not in patched
    assert patch_source(patched) == patched


def test_fused_ppo_patch_rejects_unknown_source():
    with pytest.raises(ValueError, match="allocation anchor mismatch"):
        patch_source("class Unknown: pass\n")


def test_fused_ppo_installer_round_trip(tmp_path):
    target = tmp_path / "torch_functional.py"
    target.write_text(SOURCE, encoding="utf-8", newline="\n")

    installer.apply(target)
    installer.verify(target)
    assert PATCH_MARKER in target.read_text(encoding="utf-8")

    installer.restore(target)
    assert target.read_text(encoding="utf-8") == SOURCE
    assert Path(str(target) + installer.BACKUP_SUFFIX).is_file()
