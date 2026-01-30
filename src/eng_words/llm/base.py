"""Base types and abstract classes for LLM integration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The text content of the response.
        model: The model used for generation.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cost_usd: Estimated cost in USD.
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations should handle:
    - API authentication
    - Rate limiting and retries
    - Cost calculation
    - Temperature and other parameters
    """

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Execute a completion request.

        Args:
            prompt: The prompt text to send to the LLM.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse with the completion result.
        """
        pass

    @abstractmethod
    def complete_json(self, prompt: str, schema: dict | None = None, **kwargs) -> dict:
        """Execute a completion request expecting JSON response.

        Args:
            prompt: The prompt text to send to the LLM.
            schema: Optional JSON schema for validation.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If response is not valid JSON.
        """
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pass


# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3-flash-preview",
}


def get_provider(provider_name: str, model: str | None = None, **kwargs) -> LLMProvider:
    """Factory function to get an LLM provider.

    Args:
        provider_name: Name of provider ("openai", "anthropic", "gemini")
        model: Optional model name. Uses default if not specified.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider_name is unknown.

    Examples:
        >>> provider = get_provider("gemini")  # default model
        >>> provider = get_provider("openai", "gpt-5-mini")
        >>> provider = get_provider("anthropic", "claude-haiku-4.5")
    """
    # Import here to avoid circular imports
    from eng_words.llm.providers.anthropic import AnthropicProvider
    from eng_words.llm.providers.gemini import GeminiProvider
    from eng_words.llm.providers.openai import OpenAIProvider

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {list(providers.keys())}"
        )

    provider_class = providers[provider_name]
    model = model or DEFAULT_MODELS.get(provider_name)

    return provider_class(model=model, **kwargs)
