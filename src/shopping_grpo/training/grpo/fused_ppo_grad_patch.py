'''Backport veRL 0.8 fused-PPO input-gradient handling.'''

PATCH_MARKER = 'SHOPPING_GRPO_FUSED_PPO_NEEDS_INPUT_GRAD_PATCH_V1'


def patch_source(source: str) -> str:
    '''Apply the narrowly anchored upstream fix to veRL 0.8.0 source.'''
    if PATCH_MARKER in source:
        return source
    anchor = (
        '        # Allocate memory for outputs\n'
        '        dhidden_states = None\n'
        '        if hidden_states.requires_grad:\n'
        '            dhidden_states = torch.zeros_like(hidden_states)\n'
        '        dvocab_weights = None\n'
        '        if vocab_weights.requires_grad:\n'
        '            dvocab_weights = torch.zeros_like(vocab_weights)\n'
    )
    replacement = (
        '        # Allocate memory for outputs\n'
        f'        # {PATCH_MARKER}\n'
        '        needs_hidden_grad = ctx.needs_input_grad[0]\n'
        '        needs_vocab_grad = ctx.needs_input_grad[1]\n'
        '        dhidden_states = None\n'
        '        if needs_hidden_grad:\n'
        '            dhidden_states = torch.zeros_like(hidden_states)\n'
        '        dvocab_weights = None\n'
        '        if needs_vocab_grad:\n'
        '            dvocab_weights = torch.zeros_like(vocab_weights)\n'
    )
    if source.count(anchor) != 1:
        raise ValueError('pinned veRL fused-PPO allocation anchor mismatch')
    patched = source.replace(anchor, replacement, 1)
    edits = (
        (
            '            if hidden_states.requires_grad:\n'
            '                dhidden_states[chunk_start:chunk_end] += h\n',
            '            if needs_hidden_grad:\n'
            '                dhidden_states[chunk_start:chunk_end] += h\n',
            'hidden accumulation',
        ),
        (
            '            if vocab_weights.requires_grad:\n'
            '                dvocab_weights += v\n',
            '            if needs_vocab_grad:\n'
            '                dvocab_weights += v\n',
            'vocab accumulation',
        ),
        (
            '        if orig_ndim == 3 and hidden_states.requires_grad:\n',
            '        if orig_ndim == 3 and needs_hidden_grad:\n',
            'hidden reshape',
        ),
    )
    for old, new, label in edits:
        if patched.count(old) != 1:
            raise ValueError(f'pinned veRL fused-PPO {label} anchor mismatch')
        patched = patched.replace(old, new, 1)
    return patched
