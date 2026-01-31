"""LLM configuration constants."""

from eng_words.constants.paths import LLM_CACHE_DIR

# =============================================================================
# Provider settings
# =============================================================================

DEFAULT_TEMPERATURE = 0.0  # Determinism for reproducibility

# OpenAI models (pricing/source: OpenAI pricing page snapshot provided by user)
# Recommended defaults:
# - For WSD evaluation: gpt-4.1-mini (cheap + strong)
# - For heavier tasks: gpt-4.1 or gpt-5-mini (optional)
DEFAULT_MODEL_OPENAI = "gpt-4.1-mini"
MODEL_OPENAI_GPT41 = "gpt-4.1"
MODEL_OPENAI_GPT41_MINI = "gpt-4.1-mini"
MODEL_OPENAI_GPT41_NANO = "gpt-4.1-nano"
MODEL_OPENAI_GPT4O = "gpt-4o"
MODEL_OPENAI_GPT4O_MINI = "gpt-4o-mini"
MODEL_OPENAI_GPT5 = "gpt-5"
MODEL_OPENAI_GPT5_MINI = "gpt-5-mini"
MODEL_OPENAI_GPT5_NANO = "gpt-5-nano"

# Anthropic models
# Note: keep as explicit pinned model ids for determinism; can be swapped by config later.
DEFAULT_MODEL_ANTHROPIC = "claude-3-5-haiku-20241022"
MODEL_ANTHROPIC_SONNET = "claude-3-5-sonnet-20241022"

# =============================================================================
# Evaluation settings
# =============================================================================

MIN_JURY_CONFIDENCE = 0.8  # Minimum confidence for jury agreement
MAX_CANDIDATES_PER_ITEM = 5  # Max sense candidates in evaluation prompt
EVALUATION_BATCH_SIZE = 20  # Items per batch in evaluation

# =============================================================================
# Card generation settings
# =============================================================================

MAX_EXAMPLE_LENGTH = 300  # Max chars per example sentence
MAX_EXAMPLES_PER_SENSE = 15  # Max examples per sense in prompt
CARD_BATCH_SIZE = 10  # Senses per batch in card generation (keep small for reliability)

# =============================================================================
# Spoiler policy
# =============================================================================

SPOILER_RISK_THRESHOLD = "none"  # Only "none" passes to final cards
# Valid values: "none", "low", "medium", "high"

# =============================================================================
# Cache settings
# =============================================================================

DEFAULT_CACHE_DIR = LLM_CACHE_DIR
CACHE_BACKEND = "json"  # "json" | "sqlite" (for future)

# =============================================================================
# Prompt versioning
# =============================================================================

PROMPT_VERSION_EVALUATION = "v2.1"  # Removed fake confidence, added examples
PROMPT_VERSION_CARD_GENERATION = "v1.0"
