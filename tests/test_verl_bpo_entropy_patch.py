from shopping_grpo.training.bpo.entropy_patch import PATCH_MARKER, patch_source


def fixture_source():
    return '''import logging

async def generate(self, prompt_ids, sampling_params):
        prompt_ids = normalize_token_ids(prompt_ids)
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        final_res = result
        extra_fields = {"global_steps": self.global_steps}
        token_ids = final_res.outputs[0].token_ids
        log_probs = None
'''


def test_entropy_patch_is_idempotent_and_drops_full_vector():
    once = patch_source(fixture_source())
    twice = patch_source(once)
    assert once == twice
    assert once.count(PATCH_MARKER) == 1
    assert 'sampling_params["logprobs"] = -1 if bpo_entropy_probe' in once
    assert 'extra_fields["bpo_full_vocab_entropy"]' in once
    assert "vocabulary_logprobs" in once
