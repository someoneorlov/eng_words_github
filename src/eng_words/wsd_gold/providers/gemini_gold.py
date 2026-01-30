"""Gemini provider for WSD Gold labeling."""

import os

from eng_words.wsd_gold.models import GoldExample, LLMUsage, ModelOutput
from eng_words.wsd_gold.providers.base import GoldLabelProvider
from eng_words.wsd_gold.providers.prompts import (
    build_gold_labeling_prompt,
    parse_gold_label_response,
)

# Default model for gold labeling
# gemini-3-flash-preview is newest with best quality/price ratio
DEFAULT_MODEL = "gemini-3-flash-preview"

# Pricing per 1M tokens (as of Jan 2026)
PRICING = {
    # Gemini 3 family (latest, preview)
    "gemini-3-pro-preview": {"input": 1.50, "output": 6.00},
    "gemini-3-flash-preview": {"input": 0.15, "output": 0.60},
    # Gemini 2.5 family (stable)
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.08, "output": 0.32},
    "gemini-2.5-flash-lite": {"input": 0.02, "output": 0.08},
    # Gemini 2.0 family (legacy)
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.02, "output": 0.10},
}

# Estimated tokens per example
EST_INPUT_TOKENS = 800
EST_OUTPUT_TOKENS = 50

# Models with "thinking mode" that CANNOT be disabled
THINKING_MODELS_ALWAYS_ON = {"gemini-3-pro-preview", "gemini-2.5-pro"}
THINKING_MAX_TOKENS = 8192  # Includes ~5000 thinking + actual output

# Models that support thinking config to minimize/disable thinking
# gemini-3-flash: thinkingLevel="minimal"
# gemini-2.5-flash: thinkingBudget=0
THINKING_CONFIGURABLE_MODELS = {
    "gemini-3-flash-preview": {"thinking_level": "minimal"},
    "gemini-2.5-flash": {"thinking_budget": 0},
    "gemini-2.5-flash-lite": {"thinking_budget": 0},
}


class GeminiGoldProvider(GoldLabelProvider):
    """Gemini provider for WSD Gold labeling.

    Uses Google's Gemini models via the new google-genai SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        """Initialize Gemini Gold provider.

        Args:
            api_key: Google API key (or from GOOGLE_API_KEY env var)
            model: Model to use
            temperature: Temperature for sampling (0 for determinism)
            max_retries: Max retries for invalid JSON
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it as environment variable or pass api_key."
            )

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                # Fallback to deprecated package
                try:
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        import google.generativeai as genai_old

                    genai_old.configure(api_key=self.api_key)
                    self._client = ("old", genai_old.GenerativeModel(self.model))
                except ImportError:
                    raise ImportError(
                        "google-genai package not installed. "
                        "Install with: pip install google-genai"
                    )
        return self._client

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini"

    def _get_pricing(self) -> dict[str, float]:
        """Get pricing for current model."""
        return PRICING.get(self.model, PRICING[DEFAULT_MODEL])

    def label_one(self, example: GoldExample) -> ModelOutput:
        """Label a single example."""
        prompt = build_gold_labeling_prompt(example)
        candidates = example.get_candidate_ids()

        raw_text = ""
        usage = LLMUsage(input_tokens=0, output_tokens=0)

        for attempt in range(self.max_retries + 1):
            try:
                client = self.client

                # Determine max_output_tokens based on model
                # Thinking models that can't be disabled need higher limit
                max_tokens = (
                    THINKING_MAX_TOKENS
                    if self.model in THINKING_MODELS_ALWAYS_ON
                    else 512
                )

                # Get thinking config for models that support it
                thinking_config = THINKING_CONFIGURABLE_MODELS.get(self.model)

                # Check if using old or new API
                if isinstance(client, tuple) and client[0] == "old":
                    # Old deprecated API
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        response = client[1].generate_content(
                            prompt,
                            generation_config={
                                "temperature": self.temperature,
                                "max_output_tokens": max_tokens,
                            },
                        )
                    raw_text = response.text or ""
                    # Try to get usage from response
                    try:
                        input_tokens = response.usage_metadata.prompt_token_count
                        output_tokens = response.usage_metadata.candidates_token_count
                    except (AttributeError, TypeError):
                        input_tokens = len(prompt) // 4
                        output_tokens = len(raw_text) // 4
                else:
                    # New google-genai API
                    config = {
                        "temperature": self.temperature,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json",
                    }
                    # Add thinking config if model supports it
                    if thinking_config:
                        config["thinking_config"] = thinking_config

                    response = client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                    raw_text = response.text or ""
                    # Get usage from response
                    try:
                        input_tokens = response.usage_metadata.prompt_token_count
                        # Include thinking tokens in output (billed as output)
                        output_tokens = response.usage_metadata.candidates_token_count or 0
                        thinking_tokens = response.usage_metadata.thoughts_token_count or 0
                        output_tokens = output_tokens + thinking_tokens
                    except (AttributeError, TypeError):
                        input_tokens = len(prompt) // 4
                        output_tokens = len(raw_text) // 4

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

            except Exception as e:
                error_msg = str(e)
                # Check for quota errors - don't retry these
                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                    raw_text = f"Quota exceeded: {error_msg[:200]}"
                    break
                # Log error but continue to retry for other errors
                if attempt == self.max_retries:
                    raw_text = f"Error: {error_msg[:200]}"

            # Retry with slightly modified prompt
            if attempt < self.max_retries:
                prompt = prompt + "\n(Please respond with valid JSON only)"

        # Return None if all retries exhausted
        return None  # type: ignore

    def label_batch(self, examples: list[GoldExample]) -> list[ModelOutput]:
        """Label a batch of examples."""
        return [self.label_one(ex) for ex in examples]

    def estimate_cost(self, examples: list[GoldExample]) -> float:
        """Estimate cost for labeling examples."""
        pricing = self._get_pricing()
        n = len(examples)

        input_cost = n * EST_INPUT_TOKENS * pricing["input"] / 1_000_000

        # Thinking models that can't be disabled use ~5600 thinking tokens
        if self.model in THINKING_MODELS_ALWAYS_ON:
            thinking_tokens = 5600
            output_tokens = EST_OUTPUT_TOKENS + thinking_tokens
        else:
            # Models with configurable thinking or no thinking
            output_tokens = EST_OUTPUT_TOKENS

        output_cost = n * output_tokens * pricing["output"] / 1_000_000

        return input_cost + output_cost
