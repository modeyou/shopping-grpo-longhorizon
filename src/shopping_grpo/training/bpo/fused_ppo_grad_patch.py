"""Backport veRL's fused-PPO autograd input-gradient fix.

veRL 0.8.0 inspects ``requires_grad`` on tensors saved from the custom
``forward``.  That forward runs without gradient recording, so flattening a
non-contiguous hidden-state tensor can save a copy whose flag is false even
though the original input requires a gradient.  ``ctx.needs_input_grad`` is
the autograd contract for deciding which input gradients ``backward`` must
return.
"""

PATCH_MARKER = "SHOPPING_BPO_FUSED_PPO_NEEDS_INPUT_GRAD_PATCH_V1"


def patch_source(source: str) -> str:
    """Apply the narrowly anchored upstream fix to veRL 0.8.0 source."""
    if PATCH_MARKER in source:
        return source

    allocation_anchor = (
        "        # Allocate memory for outputs\n"
        "        dhidden_states = None\n"
        "        if hidden_states.requires_grad:\n"
        "            dhidden_states = torch.zeros_like(hidden_states)\n"
        "        dvocab_weights = None\n"
        "        if vocab_weights.requires_grad:\n"
        "            dvocab_weights = torch.zeros_like(vocab_weights)\n"
    )
    allocation_replacement = (
        "        # Allocate memory for outputs\n"
        f"        # {PATCH_MARKER}\n"
        "        needs_hidden_grad = ctx.needs_input_grad[0]\n"
        "        needs_vocab_grad = ctx.needs_input_grad[1]\n"
        "        dhidden_states = None\n"
        "        if needs_hidden_grad:\n"
        "            dhidden_states = torch.zeros_like(hidden_states)\n"
        "        dvocab_weights = None\n"
        "        if needs_vocab_grad:\n"
        "            dvocab_weights = torch.zeros_like(vocab_weights)\n"
    )
    if source.count(allocation_anchor) != 1:
        raise ValueError("pinned veRL fused-PPO allocation anchor mismatch")
    patched = source.replace(allocation_anchor, allocation_replacement, 1)

    hidden_accumulation = (
        "            if hidden_states.requires_grad:\n"
        "                dhidden_states[chunk_start:chunk_end] += h\n"
    )
    hidden_replacement = (
        "            if needs_hidden_grad:\n"
        "                dhidden_states[chunk_start:chunk_end] += h\n"
    )
    vocab_accumulation = (
        "            if vocab_weights.requires_grad:\n"
        "                dvocab_weights += v\n"
    )
    vocab_replacement = (
        "            if needs_vocab_grad:\n"
        "                dvocab_weights += v\n"
    )
    reshape_anchor = "        if orig_ndim == 3 and hidden_states.requires_grad:\n"
    reshape_replacement = "        if orig_ndim == 3 and needs_hidden_grad:\n"
    for anchor, replacement, label in (
        (hidden_accumulation, hidden_replacement, "hidden accumulation"),
        (vocab_accumulation, vocab_replacement, "vocab accumulation"),
        (reshape_anchor, reshape_replacement, "hidden reshape"),
    ):
        if patched.count(anchor) != 1:
            raise ValueError(f"pinned veRL fused-PPO {label} anchor mismatch")
        patched = patched.replace(anchor, replacement, 1)
    return patched
