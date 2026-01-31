"""LLM Pricing Reference.

Single source of truth for LLM pricing across all providers.
All prices are per 1 MILLION tokens.

Last updated: 2026-01-10
Sources:
  - docs/openai_pricing.txt
  - docs/claude_pricing.md
  - docs/google_pricing.md
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelPricing:
    """Pricing for a single model."""

    input: float  # $ per 1M input tokens
    output: float  # $ per 1M output tokens
    cached_input: float | None = None  # $ per 1M cached input tokens (if supported)
    batch_coefficient: float = 1.0  # Multiplier for batch API (e.g., 0.5 = 50% discount)
    notes: str = ""


# =============================================================================
# OpenAI Pricing (Standard tier)
# Source: docs/openai_pricing.txt
# Batch API = 50% discount (coefficient 0.5)
# =============================================================================

OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-5.2 family (latest)
    "gpt-5.2": ModelPricing(
        input=1.75,
        output=14.00,
        cached_input=0.175,
        batch_coefficient=0.5,
        notes="Latest, supports temperature=0",
    ),
    "gpt-5.2-pro": ModelPricing(
        input=21.00,
        output=168.00,
        cached_input=None,
        batch_coefficient=0.5,
    ),
    # GPT-5.1 family
    "gpt-5.1": ModelPricing(
        input=1.25,
        output=10.00,
        cached_input=0.125,
        batch_coefficient=0.5,
        notes="Supports temperature=0",
    ),
    # GPT-5 family
    "gpt-5": ModelPricing(
        input=1.25,
        output=10.00,
        cached_input=0.125,
        batch_coefficient=0.5,
        notes="Does NOT support temperature=0",
    ),
    "gpt-5-mini": ModelPricing(
        input=0.25,
        output=2.00,
        cached_input=0.025,
        batch_coefficient=0.5,
        notes="Does NOT support temperature=0, needs max_completion_tokens>=512",
    ),
    "gpt-5-nano": ModelPricing(
        input=0.05,
        output=0.40,
        cached_input=0.005,
        batch_coefficient=0.5,
    ),
    "gpt-5-pro": ModelPricing(
        input=15.00,
        output=120.00,
        cached_input=None,
        batch_coefficient=0.5,
    ),
    # GPT-4.1 family
    "gpt-4.1": ModelPricing(
        input=2.00,
        output=8.00,
        cached_input=0.50,
        batch_coefficient=0.5,
    ),
    "gpt-4.1-mini": ModelPricing(
        input=0.40,
        output=1.60,
        cached_input=0.10,
        batch_coefficient=0.5,
    ),
    "gpt-4.1-nano": ModelPricing(
        input=0.10,
        output=0.40,
        cached_input=0.025,
        batch_coefficient=0.5,
    ),
    # GPT-4o family (legacy but still available)
    "gpt-4o": ModelPricing(
        input=2.50,
        output=10.00,
        cached_input=1.25,
        batch_coefficient=0.5,
    ),
    "gpt-4o-mini": ModelPricing(
        input=0.15,
        output=0.60,
        cached_input=0.075,
        batch_coefficient=0.5,
        notes="Very cheap, good quality",
    ),
    # o-series reasoning models
    "o1": ModelPricing(
        input=15.00,
        output=60.00,
        cached_input=7.50,
        batch_coefficient=0.5,
    ),
    "o3": ModelPricing(
        input=2.00,
        output=8.00,
        cached_input=0.50,
        batch_coefficient=0.5,
    ),
    "o4-mini": ModelPricing(
        input=1.10,
        output=4.40,
        cached_input=0.275,
        batch_coefficient=0.5,
    ),
}

# =============================================================================
# Anthropic Pricing
# Source: docs/claude_pricing.md
# Batch API = 50% discount (coefficient 0.5)
# =============================================================================

ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude 4.5 family (latest)
    "claude-opus-4-5-20251101": ModelPricing(
        input=5.00,
        output=25.00,
        cached_input=0.50,
        batch_coefficient=0.5,
        notes="Best quality",
    ),
    "claude-sonnet-4-5-20250929": ModelPricing(
        input=3.00,
        output=15.00,
        cached_input=0.30,
        batch_coefficient=0.5,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input=1.00,
        output=5.00,
        cached_input=0.10,
        batch_coefficient=0.5,
        notes="Fast and cheap",
    ),
    # Claude 4.1 family
    "claude-opus-4-1": ModelPricing(
        input=15.00,
        output=75.00,
        cached_input=1.50,
        batch_coefficient=0.5,
    ),
    # Claude 4 family
    "claude-sonnet-4-20250514": ModelPricing(
        input=3.00,
        output=15.00,
        cached_input=0.30,
        batch_coefficient=0.5,
    ),
    "claude-opus-4-20250514": ModelPricing(
        input=15.00,
        output=75.00,
        cached_input=1.50,
        batch_coefficient=0.5,
    ),
    # Claude 3.5 family
    "claude-haiku-3-5": ModelPricing(
        input=0.80,
        output=4.00,
        cached_input=0.08,
        batch_coefficient=0.5,
    ),
    # Claude 3 family (legacy)
    "claude-3-haiku-20240307": ModelPricing(
        input=0.25,
        output=1.25,
        cached_input=0.03,
        batch_coefficient=0.5,
    ),
}

# =============================================================================
# Google Gemini Pricing (Paid tier, Standard)
# Source: docs/google_pricing.md
# Batch API = 50% discount (coefficient 0.5)
# =============================================================================

GEMINI_PRICING: dict[str, ModelPricing] = {
    # Gemini 3 family (preview)
    "gemini-3-pro-preview": ModelPricing(
        input=2.00,
        output=12.00,  # includes thinking tokens!
        cached_input=0.20,
        batch_coefficient=0.5,
        notes="Output includes thinking tokens, expensive!",
    ),
    "gemini-3-flash-preview": ModelPricing(
        input=0.50,
        output=3.00,  # includes thinking tokens
        cached_input=0.05,
        batch_coefficient=0.5,
        notes="Best value for Gemini 3",
    ),
    # Gemini 2.5 family (stable)
    "gemini-2.5-pro": ModelPricing(
        input=1.25,
        output=10.00,
        cached_input=0.125,
        batch_coefficient=0.5,
    ),
    "gemini-2.5-flash": ModelPricing(
        input=0.30,
        output=2.50,
        cached_input=0.03,
        batch_coefficient=0.5,
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        input=0.10,
        output=0.40,
        cached_input=0.01,
        batch_coefficient=0.5,
        notes="Cheapest Gemini 2.5",
    ),
    # Gemini 2.0 family (legacy)
    "gemini-2.0-flash": ModelPricing(
        input=0.10,
        output=0.40,
        cached_input=0.025,
        batch_coefficient=0.5,
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        input=0.075,
        output=0.30,
        cached_input=None,
        batch_coefficient=0.5,
        notes="Cheapest option, no caching",
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_pricing(provider: Literal["openai", "anthropic", "gemini"], model: str) -> ModelPricing:
    """Get pricing for a model.

    Args:
        provider: Provider name
        model: Model name

    Returns:
        ModelPricing dataclass

    Raises:
        KeyError: If model not found
    """
    pricing_map = {
        "openai": OPENAI_PRICING,
        "anthropic": ANTHROPIC_PRICING,
        "gemini": GEMINI_PRICING,
    }
    return pricing_map[provider][model]


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    use_batch: bool = False,
    use_cache: bool = False,
) -> float:
    """Estimate cost for API call.

    Args:
        provider: Provider name
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        use_batch: Whether using batch API
        use_cache: Whether using cached input

    Returns:
        Estimated cost in USD
    """
    pricing = get_pricing(provider, model)

    # Input cost
    if use_cache and pricing.cached_input:
        input_cost = input_tokens * pricing.cached_input / 1_000_000
    else:
        input_cost = input_tokens * pricing.input / 1_000_000

    # Output cost
    output_cost = output_tokens * pricing.output / 1_000_000

    # Total with batch discount
    total = input_cost + output_cost
    if use_batch:
        total *= pricing.batch_coefficient

    return total


def print_pricing_table() -> None:
    """Print pricing table for all models."""
    print("=" * 100)
    print("LLM PRICING REFERENCE (per 1M tokens)")
    print("=" * 100)
    print(
        f"{'Provider':<12} {'Model':<30} {'Input':<8} {'Output':<8} {'Cached':<8} {'Batch':<6} {'Notes'}"
    )
    print("-" * 100)

    for provider, pricing_dict in [
        ("OpenAI", OPENAI_PRICING),
        ("Anthropic", ANTHROPIC_PRICING),
        ("Gemini", GEMINI_PRICING),
    ]:
        for model, pricing in pricing_dict.items():
            batch = f"{pricing.batch_coefficient:.0%}" if pricing.batch_coefficient != 1.0 else "-"
            cached = f"${pricing.cached_input:.3f}" if pricing.cached_input else "-"
            print(
                f"{provider:<12} {model:<30} ${pricing.input:<7.2f} ${pricing.output:<7.2f} {cached:<8} {batch:<6} {pricing.notes[:20]}"
            )
        print()


# =============================================================================
# Recommended Models for Card Generation
# =============================================================================

RECOMMENDED_MODELS = {
    # Best value options for card generation
    "cheapest": ("gemini", "gemini-2.0-flash-lite"),  # $0.075/$0.30
    "cheap_fast": ("gemini", "gemini-3-flash-preview"),  # $0.50/$3.00
    "cheap_reliable": ("openai", "gpt-4o-mini"),  # $0.15/$0.60, batch 50%
    "balanced": ("anthropic", "claude-haiku-4-5-20251001"),  # $1.00/$5.00, batch 50%
    "quality": ("anthropic", "claude-sonnet-4-5-20250929"),  # $3.00/$15.00
    "best": ("anthropic", "claude-opus-4-5-20251101"),  # $5.00/$25.00
}


if __name__ == "__main__":
    print_pricing_table()
