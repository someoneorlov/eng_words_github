"""Synset validation for checking example sentences against synset groups."""

import json
import logging
from typing import Any

from nltk.corpus import wordnet as wn

from eng_words.llm.base import LLMProvider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.retry import call_llm_with_retry

logger = logging.getLogger(__name__)


VALIDATION_PROMPT = """You are helping validate example sentences for Anki flashcards.

## Word Information
- Word: "{lemma}" ({pos})
- Synset Group: {synset_group_info}
- Primary Definition: {primary_definition}

## Synset Group Definitions
{synset_definitions}

## Example Sentences
{examples_numbered}

## Your Task
For each sentence, determine if it matches ANY of the synset meanings in the group.

**Criteria:**
1. The sentence must contain the word "{lemma}" (or its grammatical forms)
2. The meaning of the word in the sentence must match at least ONE synset from the group
3. If the word has a different meaning (not in the synset group) → mark as invalid

## Response Format (JSON only, no markdown)
{{
  "valid_indices": [1, 2, 3],
  "invalid_indices": [4, 5],
  "validation_details": {{
    "1": {{"valid": true, "reason": "Matches synset_group meaning"}},
    "4": {{"valid": false, "reason": "Different meaning, not in synset_group"}}
  }}
}}

Return ONLY valid JSON, no explanations."""


def _get_synset_definitions(synset_ids: list[str]) -> dict[str, str]:
    """Get definitions for synsets from WordNet.

    Args:
        synset_ids: List of synset IDs

    Returns:
        Dictionary mapping synset_id to definition
    """
    definitions = {}
    for synset_id in synset_ids:
        try:
            synset = wn.synset(synset_id)
            definitions[synset_id] = synset.definition()
        except Exception as e:
            logger.warning(f"Failed to get definition for {synset_id}: {e}")
            definitions[synset_id] = f"Unknown synset: {synset_id}"
    return definitions


def _format_validation_prompt(
    lemma: str,
    pos: str,
    synset_group: list[str],
    primary_synset: str,
    synset_definitions: dict[str, str],
    examples: list[tuple[int, str]],
) -> str:
    """Format the validation prompt.

    Args:
        lemma: Word lemma
        pos: Part of speech
        synset_group: List of synset IDs
        primary_synset: Primary synset ID
        synset_definitions: Dictionary of synset_id -> definition
        examples: List of (sentence_id, sentence) tuples

    Returns:
        Formatted prompt string
    """
    synset_group_info = ", ".join(synset_group)
    primary_definition = synset_definitions.get(primary_synset, "Unknown")

    synset_defs_text = "\n".join(f"- {sid}: {defn}" for sid, defn in synset_definitions.items())

    examples_numbered = "\n".join(f'{i+1}. "{ex}"' for i, (_, ex) in enumerate(examples))

    return VALIDATION_PROMPT.format(
        lemma=lemma,
        pos=pos.lower(),
        synset_group_info=synset_group_info,
        primary_definition=primary_definition,
        synset_definitions=synset_defs_text,
        examples_numbered=examples_numbered,
    )


def _parse_validation_response(
    response_content: str,
    examples: list[tuple[int, str]],
    max_retries: int = 2,
) -> dict[str, Any]:
    """Parse LLM response for validation.

    Args:
        response_content: Raw response from LLM
        examples: List of (sentence_id, sentence) tuples
        max_retries: Maximum retries for parsing errors

    Returns:
        Parsed validation result
    """
    # Try to parse JSON
    content = response_content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        # Try to extract JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(content[start:end])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON response: {content[:200]}")
        else:
            raise ValueError(f"Invalid JSON response: {content[:200]}")

    # Extract valid and invalid indices
    valid_indices = result.get("valid_indices", [])
    invalid_indices = result.get("invalid_indices", [])

    # Convert 1-based indices to sentence_ids
    valid_sentence_ids = [examples[i - 1][0] for i in valid_indices if 1 <= i <= len(examples)]

    invalid_sentence_ids = [examples[i - 1][0] for i in invalid_indices if 1 <= i <= len(examples)]

    return {
        "valid_sentence_ids": valid_sentence_ids,
        "invalid_sentence_ids": invalid_sentence_ids,
        "has_valid": len(valid_sentence_ids) > 0,
        "validation_details": result.get("validation_details", {}),
    }


# Use cache.generate_key() directly (no _generate_cache_key)


# Maximum examples per batch to avoid JSON parsing errors
MAX_EXAMPLES_PER_BATCH = 50


def _validate_examples_batch(
    lemma: str,
    synset_group: list[str],
    primary_synset: str,
    examples: list[tuple[int, str]],  # (sentence_id, sentence)
    provider: LLMProvider,
    cache: ResponseCache,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Validate a single batch of examples (internal function)."""
    # Handle empty examples
    if not examples:
        return {
            "valid_sentence_ids": [],
            "invalid_sentence_ids": [],
            "has_valid": False,
            "validation_details": {},
        }

    # Get synset definitions
    synset_definitions = _get_synset_definitions(synset_group)

    # Determine POS from primary synset
    try:
        primary_synset_obj = wn.synset(primary_synset)
        pos = primary_synset_obj.pos()
    except Exception:
        # Fallback: try to extract from synset_id
        pos = primary_synset.split(".")[1] if "." in primary_synset else "n"
        logger.warning(f"Could not get POS for {primary_synset}, using {pos}")

    # Format prompt
    prompt = _format_validation_prompt(
        lemma=lemma,
        pos=pos,
        synset_group=synset_group,
        primary_synset=primary_synset,
        synset_definitions=synset_definitions,
        examples=examples,
    )

    # Check cache first (before calling retry utility)
    # This allows tests to verify caching behavior
    model_name = getattr(provider, "model", "unknown-model")
    temperature = getattr(provider, "temperature", 0.0)
    cache_key = cache.generate_key(model_name, prompt, temperature)
    cached_response = cache.get(cache_key)

    if cached_response:
        logger.debug(f"Cache hit for {lemma} validation")
        try:
            return _parse_validation_response(cached_response.content, examples)
        except Exception as e:
            logger.warning(f"Failed to parse cached response: {e}, retrying")
            # Continue to LLM call

    # Use universal retry utility
    try:
        response = call_llm_with_retry(
            provider=provider,
            prompt=prompt,
            cache=None,  # Don't use cache here, we handle it above
            max_retries=max_retries,
            retry_delay=1.0,
            validate_json=True,
            on_retry=lambda attempt, error: logger.debug(
                f"Retry {attempt} for {lemma} validation: {error}"
            ),
        )

        # Cache the response manually
        try:
            cache.set(cache_key, response)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

        # Parse validation response
        return _parse_validation_response(response.content, examples)

    except (ValueError, json.JSONDecodeError, Exception) as e:
        # All retries failed - use conservative approach
        logger.error(f"All retries failed for {lemma} validation: {e}, marking all as invalid")
        return {
            "valid_sentence_ids": [],
            "invalid_sentence_ids": [sid for sid, _ in examples],
            "has_valid": False,
            "validation_details": {},
        }


def validate_examples_for_synset_group(
    lemma: str,
    synset_group: list[str],
    primary_synset: str,
    examples: list[tuple[int, str]],  # (sentence_id, sentence)
    provider: LLMProvider,
    cache: ResponseCache,
    max_retries: int = 2,
    max_examples_per_batch: int = MAX_EXAMPLES_PER_BATCH,
) -> dict[str, Any]:
    """Validate examples against synset group.

    Checks if example sentences match any of the synsets in the group.
    Uses LLM to determine if the meaning of the word in each sentence
    matches the synset group meanings.

    If there are more examples than max_examples_per_batch, processes them in batches
    to avoid JSON parsing errors with very long responses.

    Args:
        lemma: Word lemma (e.g., "long")
        synset_group: List of synset IDs in the group (e.g., ["long.r.01", "long.s.01"])
        primary_synset: Primary synset for definition (e.g., "long.r.01")
        examples: List of (sentence_id, sentence) tuples
        provider: LLM provider for validation
        cache: Response cache for LLM responses
        max_retries: Maximum retries for parsing errors
        max_examples_per_batch: Maximum examples per batch (default: 50)

    Returns:
        Dictionary with:
        - valid_sentence_ids: List of sentence IDs that match synset_group
        - invalid_sentence_ids: List of sentence IDs that don't match
        - has_valid: Boolean indicating if any valid examples exist
        - validation_details: Optional details from LLM response
    """
    # Handle empty examples
    if not examples:
        logger.debug(f"No examples provided for {lemma}")
        return {
            "valid_sentence_ids": [],
            "invalid_sentence_ids": [],
            "has_valid": False,
            "validation_details": {},
        }

    # If examples fit in one batch, process directly
    if len(examples) <= max_examples_per_batch:
        return _validate_examples_batch(
            lemma=lemma,
            synset_group=synset_group,
            primary_synset=primary_synset,
            examples=examples,
            provider=provider,
            cache=cache,
            max_retries=max_retries,
        )

    # Process in batches
    logger.info(
        f"Processing {len(examples)} examples for {lemma} in batches of {max_examples_per_batch}"
    )

    all_valid_ids = []
    all_invalid_ids = []
    all_validation_details = {}

    num_batches = (len(examples) + max_examples_per_batch - 1) // max_examples_per_batch

    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_examples_per_batch
        end_idx = min(start_idx + max_examples_per_batch, len(examples))
        batch_examples = examples[start_idx:end_idx]

        logger.debug(
            f"Processing batch {batch_idx + 1}/{num_batches} for {lemma} ({len(batch_examples)} examples)"
        )

        batch_result = _validate_examples_batch(
            lemma=lemma,
            synset_group=synset_group,
            primary_synset=primary_synset,
            examples=batch_examples,
            provider=provider,
            cache=cache,
            max_retries=max_retries,
        )

        # Aggregate results
        all_valid_ids.extend(batch_result["valid_sentence_ids"])
        all_invalid_ids.extend(batch_result["invalid_sentence_ids"])

        # Merge validation_details (adjust indices if needed)
        if batch_result.get("validation_details"):
            for key, value in batch_result["validation_details"].items():
                # Key is the example index in the batch (1-based)
                # Convert to global index if needed for tracking
                all_validation_details[f"batch_{batch_idx + 1}_ex_{key}"] = value

    return {
        "valid_sentence_ids": all_valid_ids,
        "invalid_sentence_ids": all_invalid_ids,
        "has_valid": len(all_valid_ids) > 0,
        "validation_details": all_validation_details,
    }
