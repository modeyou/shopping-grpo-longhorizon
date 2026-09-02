'''Unit tests for the GRPO fused-PPO input-gradient backport.'''

import pytest

from shopping_grpo.training.grpo.fused_ppo_grad_patch import PATCH_MARKER, patch_source


SOURCE = '''        # Allocate memory for outputs
        dhidden_states = None
        if hidden_states.requires_grad:
            dhidden_states = torch.zeros_like(hidden_states)
        dvocab_weights = None
        if vocab_weights.requires_grad:
            dvocab_weights = torch.zeros_like(vocab_weights)
            if hidden_states.requires_grad:
                dhidden_states[chunk_start:chunk_end] += h
            if vocab_weights.requires_grad:
                dvocab_weights += v
        if orig_ndim == 3 and hidden_states.requires_grad:
'''


def test_patch_uses_autograd_input_contract_and_is_idempotent():
    patched = patch_source(SOURCE)
    assert patched.count(PATCH_MARKER) == 1
    assert 'needs_hidden_grad = ctx.needs_input_grad[0]' in patched
    assert 'needs_vocab_grad = ctx.needs_input_grad[1]' in patched
    assert 'if orig_ndim == 3 and needs_hidden_grad:' in patched
    assert patch_source(patched) == patched


def test_patch_rejects_unknown_source():
    with pytest.raises(ValueError, match='allocation anchor mismatch'):
        patch_source('unknown upstream source')
