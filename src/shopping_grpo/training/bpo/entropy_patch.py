"""Deterministic source transform for the pinned veRL vLLM entropy probe."""

PATCH_MARKER = "SHOPPING_BPO_EXACT_ENTROPY_PATCH_V1"


def patch_source(source: str) -> str:
    if PATCH_MARKER in source:
        return source
    anchors = {
        "import": "import logging\n",
        "normalize": "        prompt_ids = normalize_token_ids(prompt_ids)\n",
        "logprobs": (
            '        sampling_params["logprobs"] = 0 if '
            'sampling_params.pop("logprobs", False) else None\n'
        ),
        "tokens": "        token_ids = final_res.outputs[0].token_ids\n",
    }
    missing = [name for name, anchor in anchors.items() if source.count(anchor) != 1]
    if missing:
        raise ValueError(f"pinned veRL vLLM source anchors mismatch: {missing}")
    source = source.replace(anchors["import"], "import logging\nimport math\n", 1)
    source = source.replace(
        anchors["normalize"],
        anchors["normalize"]
        + f"        # {PATCH_MARKER}\n"
        + '        bpo_entropy_probe = bool(sampling_params.pop("bpo_entropy_probe", False))\n',
        1,
    )
    source = source.replace(
        anchors["logprobs"],
        '        requested_logprobs = bool(sampling_params.pop("logprobs", False))\n'
        '        sampling_params["logprobs"] = -1 if bpo_entropy_probe else '
        "(0 if requested_logprobs else None)\n",
        1,
    )
    source = source.replace(
        anchors["tokens"],
        anchors["tokens"]
        + "        if bpo_entropy_probe:\n"
        + "            distributions = final_res.outputs[0].logprobs\n"
        + "            if len(token_ids) != 1 or not distributions or len(distributions) != 1:\n"
        + '                raise RuntimeError("BPO entropy probe requires exactly one token")\n'
        + "            vocabulary_logprobs = [\n"
        + "                float(item.logprob) for item in distributions[0].values()\n"
        + "            ]\n"
        + "            probabilities = [math.exp(value) for value in vocabulary_logprobs]\n"
        + "            probability_mass = sum(probabilities)\n"
        + "            if not 0.999 <= probability_mass <= 1.001:\n"
        + "                raise RuntimeError(\n"
        + '                    "BPO entropy probe did not receive complete vocabulary logprobs"\n'
        + "                )\n"
        + '            extra_fields["bpo_full_vocab_entropy"] = -sum(\n'
        + "                probability * logprob\n"
        + "                for probability, logprob in zip(\n"
        + "                    probabilities, vocabulary_logprobs, strict=True\n"
        + "                )\n"
        + "            )\n",
        1,
    )
    return source
