"""Pure helpers for the frozen ShopSimulator persona task contract."""


def has_persona_request(item, instruction):
    """Return whether a source row can be used in persona mode."""
    persona = item.get("user_persona")
    return bool(
        isinstance(persona, dict)
        and persona
        and instruction.get("instruction_simple")
    )


def actor_instruction(item, instruction, if_persona=False):
    """Keep standard IDs stable while selecting the Actor-visible request."""
    if if_persona and has_persona_request(item, instruction):
        return instruction["instruction_simple"]
    return instruction["instruction"]
