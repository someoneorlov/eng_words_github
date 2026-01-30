"""Anthropic provider implementation."""

import json
import os

from anthropic import Anthropic

from eng_words.constants.llm_config import DEFAULT_MODEL_ANTHROPIC, DEFAULT_TEMPERATURE
from eng_words.constants.llm_pricing import estimate_cost
from eng_words.llm.base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Anthropic API provider.

    Supports Claude 3 Haiku (default), Sonnet, and Opus.

    Args:
        api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        model: Model to use. Defaults to claude-3-haiku-20240307.
        temperature: Temperature for sampling. Defaults to 0.0 for determinism.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL_ANTHROPIC,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it as environment variable or pass api_key."
            )

        self.model = model
        self.temperature = temperature
        self.client = Anthropic(api_key=self.api_key)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Execute a completion request.

        Args:
            prompt: The prompt text to send.
            **kwargs: Additional parameters (model, temperature, max_tokens).

        Returns:
            LLMResponse with the completion result.
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 4096)

        response = self.client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Calculate cost using centralized pricing
        cost_usd = self.estimate_cost(input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        try:
            return estimate_cost(
                provider="anthropic",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                use_batch=False,
            )
        except KeyError:
            # Unknown model: return 0
            return 0.0

    def complete_json(self, prompt: str, schema: dict | None = None, **kwargs) -> dict:
        """Execute a completion request expecting JSON response.

        Args:
            prompt: The prompt text to send.
            schema: Optional JSON schema (not enforced, just for documentation).
            **kwargs: Additional parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If response is not valid JSON.
        """
        response = self.complete(prompt, **kwargs)

        try:
            # Try to extract JSON from response
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}\nContent: {response.content}")
