"""Base class for WSD Gold labeling providers."""

from abc import ABC, abstractmethod

from eng_words.wsd_gold.models import GoldExample, ModelOutput


class GoldLabelProvider(ABC):
    """Abstract base class for Gold labeling LLM providers.

    Each provider implements labeling using a specific LLM API
    (OpenAI, Anthropic, Google) with consistent interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic', 'gemini')."""
        pass

    @abstractmethod
    def label_one(self, example: GoldExample) -> ModelOutput:
        """Label a single example.

        Args:
            example: GoldExample to label

        Returns:
            ModelOutput with the labeling result
        """
        pass

    @abstractmethod
    def label_batch(self, examples: list[GoldExample]) -> list[ModelOutput]:
        """Label a batch of examples.

        Args:
            examples: List of GoldExample to label

        Returns:
            List of ModelOutput, one per example
        """
        pass

    @abstractmethod
    def estimate_cost(self, examples: list[GoldExample]) -> float:
        """Estimate cost for labeling examples.

        Args:
            examples: List of examples to estimate cost for

        Returns:
            Estimated cost in USD
        """
        pass
