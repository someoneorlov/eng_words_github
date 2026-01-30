"""OpenAI provider implementation."""

import json
import os

from openai import OpenAI

from eng_words.constants.llm_config import DEFAULT_MODEL_OPENAI, DEFAULT_TEMPERATURE
from eng_words.constants.llm_pricing import estimate_cost
from eng_words.llm.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI API provider.

    Supports GPT-4o-mini (default), GPT-4o, and GPT-4-turbo.

    Args:
        api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        model: Model to use. Defaults to gpt-4o-mini.
        temperature: Temperature for sampling. Defaults to 0.0 for determinism.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL_OPENAI,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it as environment variable or pass api_key."
            )

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Execute a completion request.

        Args:
            prompt: The prompt text to send.
            **kwargs: Additional parameters (model, temperature, max_tokens, seed).

        Returns:
            LLMResponse with the completion result.
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 16384)  # Increased for batch processing
        seed = kwargs.get("seed", 42)  # Deterministic by default

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

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
                provider="openai",
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
