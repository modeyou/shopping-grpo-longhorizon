"""Personalized ShopSimulator interaction primitives."""

from shopping_grpo.personalization.shopper import LLMShopper, ShopperProtocolError
from shopping_grpo.personalization.masking import (
    MASK_SCHEMA_VERSION,
    PersonaMaskError,
    apply_persona_mask,
)

__all__ = [
    "LLMShopper",
    "MASK_SCHEMA_VERSION",
    "PersonaMaskError",
    "ShopperProtocolError",
    "apply_persona_mask",
]
