"""OpenAI provider for WSD Gold labeling."""

import os

from openai import OpenAI

from eng_words.wsd_gold.models import GoldExample, LLMUsage, ModelOutput
from eng_words.wsd_gold.providers.base import GoldLabelProvider
from eng_words.wsd_gold.providers.prompts import (
    build_gold_labeling_prompt,
    parse_gold_label_response,
)

# Default model for gold labeling (gpt-5-nano is cheapest, gpt-5-mini is balanced)
DEFAULT_MODEL = "gpt-5-mini"

# Models that require max_completion_tokens instead of max_tokens
GPT5_MODELS = {"gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro"}

# Only base GPT-5 models (not 5.1/5.2) don't support temperature=0
GPT5_NO_TEMPERATURE = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro"}

# Pricing per 1M tokens (as of Jan 2026)
PRICING = {
    # GPT-5.2 family (latest)
    "gpt-5.2": {"input": 2.00, "cached_input": 0.20, "output": 16.0},
    "gpt-5.2-pro": {"input": 5.00, "cached_input": 0.50, "output": 40.0},
    # GPT-5 family
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "gpt-5-mini": {"input": 0.30, "cached_input": 0.03, "output": 2.40},
    "gpt-5-nano": {"input": 0.06, "cached_input": 0.006, "output": 0.48},
    "gpt-5-pro": {"input": 3.00, "cached_input": 0.30, "output": 24.0},
    # GPT-4.1 family (legacy)
    "gpt-4.1": {"input": 2.00, "cached_input": 0.20, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.04, "output": 1.60},
}

# Estimated tokens per example
EST_INPUT_TOKENS = 800
EST_OUTPUT_TOKENS = 50


class OpenAIGoldProvider(GoldLabelProvider):
    """OpenAI provider for WSD Gold labeling.

    Uses GPT models with JSON mode for structured output.
    Supports prompt caching and retry logic.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        """Initialize OpenAI Gold provider.

        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            model: Model to use
            temperature: Temperature for sampling (0 for determinism)
            max_retries: Max retries for invalid JSON
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it as environment variable or pass api_key."
            )

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

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

        import time

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # GPT-5+ models use max_completion_tokens instead of max_tokens
                is_gpt5 = any(self.model.startswith(m) for m in GPT5_MODELS)
                # GPT-5 and GPT-5.1 don't support temperature=0, but GPT-5.2 does
                no_temp = any(self.model.startswith(m) for m in GPT5_NO_TEMPERATURE)

                if is_gpt5:
                    kwargs = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": 512,
                        "seed": 42,
                        "response_format": {"type": "json_object"},
                    }
                    if not no_temp:
                        kwargs["temperature"] = self.temperature
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=512,
                        seed=42,
                        response_format={"type": "json_object"},
                    )

                raw_text = response.choices[0].message.content or ""
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                # Calculate cost
                pricing = self._get_pricing()
                cost = (
                    input_tokens * pricing["input"] / 1_000_000
                    + output_tokens * pricing["output"] / 1_000_000
                )

                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=0,  # Not available in standard API
                    cost_usd=cost,
                )

                output = parse_gold_label_response(raw_text, candidates, usage)
                if output is not None:
                    return output

                # Retry with slightly modified prompt for parse errors
                if attempt < self.max_retries:
                    prompt = prompt + "\n(Please respond with valid JSON only)"

            except Exception as e:
                last_error = str(e)
                # Exponential backoff for rate limits
                if "rate" in last_error.lower() or "429" in last_error:
                    time.sleep(2**attempt)
                elif attempt < self.max_retries:
                    time.sleep(1)

        # Return None if all retries exhausted (caller should handle)
        return None  # type: ignore

    def label_batch(self, examples: list[GoldExample]) -> list[ModelOutput]:
        """Label a batch of examples.

        Currently uses sequential label_one calls.
        TODO: Implement true batch API for cost savings.

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
