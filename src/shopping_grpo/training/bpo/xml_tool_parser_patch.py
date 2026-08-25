"""Anchored source transform for malformed Qwen XML tool parameters."""

PATCH_MARKER = "SHOPPING_BPO_TOLERANT_XML_PARAMETER_PATCH_V1"


def patch_source(source: str) -> str:
    """Skip a malformed parameter without discarding the whole tool call."""
    if PATCH_MARKER in source:
        return source
    anchor = (
        '            idx = match_text.index(">")\n'
        "            param_name = match_text[:idx]\n"
        "            param_value = str(match_text[idx + 1 :])\n"
    )
    if source.count(anchor) != 1:
        raise ValueError("pinned veRL Qwen XML parser anchor mismatch")
    replacement = (
        f"            # {PATCH_MARKER}\n"
        '            idx = match_text.find(">")\n'
        "            if idx <= 0:\n"
        "                logger.warning(\n"
        '                    "Skipping malformed XML tool parameter: %r", match_text\n'
        "                )\n"
        "                continue\n"
        "            param_name = match_text[:idx]\n"
        "            param_value = str(match_text[idx + 1 :])\n"
    )
    return source.replace(anchor, replacement, 1)
