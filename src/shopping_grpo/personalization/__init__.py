"""Personalized ShopSimulator interaction primitives."""

from shopping_grpo.personalization.shopper import LLMShopper, ShopperProtocolError
from shopping_grpo.personalization.masking import (
    MASKED_TEACHER_GUIDANCE_VERSION,
    MASKED_TEACHER_HINT,
    MASK_SCHEMA_VERSION,
    PersonaMaskError,
    apply_persona_mask,
)

__all__ = [
    "LLMShopper",
    "MASKED_TEACHER_GUIDANCE_VERSION",
    "MASKED_TEACHER_HINT",
    "MASK_SCHEMA_VERSION",
    "PersonaMaskError",
    "ShopperProtocolError",
    "apply_persona_mask",
]
