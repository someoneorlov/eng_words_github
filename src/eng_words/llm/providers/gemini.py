"""Gemini provider implementation."""

import json
import os

from google import genai
from google.genai import types

from eng_words.constants.llm_pricing import estimate_cost
from eng_words.llm.base import LLMProvider, LLMResponse

# Default model for card generation
DEFAULT_MODEL_GEMINI = "gemini-3-flash-preview"
DEFAULT_TEMPERATURE = 0.0


class GeminiProvider(LLMProvider):
    """Google Gemini API provider.

    Supports Gemini 3 Flash (default), Gemini 2.5 Flash, and other models.

    Args:
        api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
        model: Model to use. Defaults to gemini-3-flash-preview.
        temperature: Temperature for sampling. Defaults to 0.0 for determinism.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL_GEMINI,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it as environment variable or pass api_key."
            )

        self.model = model
        self.temperature = temperature
        self._client = genai.Client(api_key=self.api_key)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Execute a completion request.

        Args:
            prompt: The prompt text to send.
            **kwargs: Additional parameters (temperature, max_output_tokens).

        Returns:
            LLMResponse with the completion result.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_output_tokens = kwargs.get("max_output_tokens", 4096)

        # Configure generation
        config_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        # Add thinking_budget=0 for flash models to avoid expensive internal thinking
        if "flash" in self.model:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        content = response.text or ""

        # Get token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        # Calculate cost using centralized pricing
        cost_usd = self.estimate_cost(input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

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
        # For JSON responses, add response_mime_type to config
        kwargs.setdefault("response_mime_type", "application/json")

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

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        return estimate_cost(
            provider="gemini",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            use_batch=False,
        )
