"""Anthropic provider for WSD Gold labeling."""

import os

from anthropic import Anthropic

from eng_words.wsd_gold.models import GoldExample, LLMUsage, ModelOutput
from eng_words.wsd_gold.providers.base import GoldLabelProvider
from eng_words.wsd_gold.providers.prompts import (
    build_gold_labeling_prompt,
    parse_gold_label_response,
)

# Default model for gold labeling
# claude-haiku-4-5 is fast & cheap, claude-sonnet-4-5 is balanced
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Pricing per 1M tokens (as of Jan 2026)
PRICING = {
    # Claude 4.5 family (latest, recommended)
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    # Claude 4 family
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 5.00, "output": 25.00},
    # Claude 3 family (legacy)
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

# Estimated tokens per example
EST_INPUT_TOKENS = 800
EST_OUTPUT_TOKENS = 50


class AnthropicGoldProvider(GoldLabelProvider):
    """Anthropic provider for WSD Gold labeling.

    Uses Claude models with JSON output.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        """Initialize Anthropic Gold provider.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Model to use
            temperature: Temperature for sampling (0 for determinism)
            max_retries: Max retries for invalid JSON
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it as environment variable or pass api_key."
            )

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = Anthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    def _get_pricing(self) -> dict[str, float]:
        """Get pricing for current model."""
        return PRICING.get(self.model, PRICING[DEFAULT_MODEL])

    def label_one(self, example: GoldExample) -> ModelOutput:
        """Label a single example.

        Args:
            example: GoldExample to label

        Returns:
            ModelOutput with the labeling result
        """
        prompt = build_gold_labeling_prompt(example)
        candidates = example.get_candidate_ids()

        for attempt in range(self.max_retries + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            pricing = self._get_pricing()
            cost = (
                input_tokens * pricing["input"] / 1_000_000
                + output_tokens * pricing["output"] / 1_000_000
            )

            usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=0,
                cost_usd=cost,
            )

            output = parse_gold_label_response(raw_text, candidates, usage)
            if output is not None:
                return output

            # Retry with slightly modified prompt
            if attempt < self.max_retries:
                prompt = prompt + "\n(Please respond with valid JSON only)"

        # Return None if all retries exhausted
        return None  # type: ignore

    def label_batch(self, examples: list[GoldExample]) -> list[ModelOutput]:
        """Label a batch of examples.

        Currently uses sequential label_one calls.
        TODO: Implement Message Batches API for cost savings.

        Args:
            examples: List of GoldExample to label

        Returns:
            List of ModelOutput, one per example
        """
        return [self.label_one(ex) for ex in examples]

    def estimate_cost(self, examples: list[GoldExample]) -> float:
        """Estimate cost for labeling examples.

        Args:
            examples: List of examples to estimate cost for

        Returns:
            Estimated cost in USD
        """
        pricing = self._get_pricing()
        n = len(examples)

        input_cost = n * EST_INPUT_TOKENS * pricing["input"] / 1_000_000
        output_cost = n * EST_OUTPUT_TOKENS * pricing["output"] / 1_000_000

        return input_cost + output_cost
