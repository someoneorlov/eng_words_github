"""LLM-based Word Sense Disambiguation.

This module provides functions for using LLMs to determine
the correct WordNet synset for a word in context, and
generate translations for new cards.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from eng_words.llm.base import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from eng_words.llm.response_cache import ResponseCache
    from eng_words.llm.smart_card_generator import SmartCard

logger = logging.getLogger(__name__)


# =============================================================================
# Translation Generation
# =============================================================================

TRANSLATION_PROMPT_TEMPLATE = """Translate this English word to Russian for the given meaning.

Word: "{lemma}"
Part of speech: {pos}
Definition: {definition}
Example: "{example}"

Return ONLY the Russian translation (one word or short phrase).
Example response: run"""


WSD_PROMPT_TEMPLATE = """Word: "{lemma}"
Sentence: "{sentence}"

Which meaning best fits this context?

{candidates_list}

Return ONLY the synset_id (e.g., "{example_synset}") or "NONE" if unclear.
Do not explain, just return the synset_id."""


def _build_wsd_prompt(
    lemma: str,
    sentence: str,
    candidate_synsets: list[dict[str, Any]],
) -> str:
    """Build the WSD prompt for LLM.

    Args:
        lemma: Word lemma (e.g., "accompany")
        sentence: Sentence containing the word
        candidate_synsets: List of dicts with synset_id and definition

    Returns:
        Formatted prompt string
    """
    candidates_list = []
    for i, candidate in enumerate(candidate_synsets, 1):
        synset_id = candidate["synset_id"]
        definition = candidate["definition"]
        candidates_list.append(f"{i}. {synset_id}: {definition}")

    example_synset = candidate_synsets[0]["synset_id"] if candidate_synsets else "word.v.01"

    return WSD_PROMPT_TEMPLATE.format(
        lemma=lemma,
        sentence=sentence,
        candidates_list="\n".join(candidates_list),
        example_synset=example_synset,
    )


def _parse_wsd_response(
    response: str,
    valid_synsets: set[str],
) -> str | None:
    """Parse LLM response to extract synset_id.

    Args:
        response: Raw LLM response text
        valid_synsets: Set of valid synset_ids from candidates

    Returns:
        synset_id if valid, None otherwise
    """
    # Clean response
    cleaned = response.strip().strip('"\'')

    # Check for NONE
    if cleaned.upper() == "NONE":
        return None

    # Validate against candidates
    if cleaned in valid_synsets:
        return cleaned

    # Not a valid synset
    logger.warning(f"LLM returned invalid synset: {cleaned}")
    return None


def llm_wsd_sentence(
    lemma: str,
    sentence: str,
    candidate_synsets: list[dict[str, Any]],
    provider: LLMProvider,
) -> str | None:
    """Use LLM to determine correct synset for a word in context.

    Args:
        lemma: Word lemma (e.g., "accompany")
        sentence: Sentence containing the word
        candidate_synsets: List of possible synsets with definitions.
            Each dict must have 'synset_id' and 'definition' keys.
        provider: LLM provider instance

    Returns:
        synset_id of the best match, or None if unclear.

    Example:
        >>> llm_wsd_sentence(
        ...     "accompany",
        ...     "She accompanied him to the store",
        ...     [
        ...         {"synset_id": "accompany.v.02", "definition": "go with someone"},
        ...         {"synset_id": "play_along.v.02", "definition": "perform music"},
        ...     ],
        ...     provider
        ... )
        "accompany.v.02"
    """
    # Handle edge cases
    if not candidate_synsets:
        logger.debug(f"No candidates for '{lemma}', returning None")
        return None

    if len(candidate_synsets) == 1:
        # Single candidate - no need to call LLM
        synset_id = candidate_synsets[0]["synset_id"]
        logger.debug(f"Single candidate for '{lemma}': {synset_id}")
        return synset_id

    # Build prompt
    prompt = _build_wsd_prompt(lemma, sentence, candidate_synsets)

    # Call LLM
    response: LLMResponse = provider.complete(prompt, max_output_tokens=50)

    # Parse response
    valid_synsets = {c["synset_id"] for c in candidate_synsets}
    result = _parse_wsd_response(response.content, valid_synsets)

    logger.debug(
        f"LLM WSD for '{lemma}' in '{sentence[:50]}...': {result} "
        f"(cost: ${response.cost_usd:.6f})"
    )

    return result


def _get_candidate_synsets_for_lemma(lemma: str, pos: str) -> list[dict[str, Any]]:
    """Get all possible synsets for a lemma from WordNet.

    Args:
        lemma: Word lemma
        pos: Part of speech (n, v, a, r)

    Returns:
        List of dicts with synset_id and definition
    """
    from eng_words.wsd.wordnet_utils import get_synsets_with_definitions

    pos_map = {"noun": "n", "verb": "v", "adj": "a", "adv": "r", "n": "n", "v": "v", "a": "a", "r": "r"}
    wn_pos = pos_map.get(pos, pos)

    # get_synsets_with_definitions returns list of tuples: (synset_id, definition)
    synsets = get_synsets_with_definitions(lemma, wn_pos)
    return [{"synset_id": synset_id, "definition": definition} for synset_id, definition in synsets]


def redistribute_empty_cards(
    cards: list[SmartCard],
    provider: LLMProvider,
    cache: ResponseCache,
) -> list[SmartCard]:
    """Redistribute sentences from empty cards using LLM WSD.

    For cards with no selected_examples:
    1. For each excluded_example, determine correct synset via LLM
    2. Add sentence to existing card with matching synset, or create new card
    3. Remove empty cards

    Args:
        cards: List of SmartCard objects
        provider: LLM provider for WSD calls
        cache: Response cache for LLM calls

    Returns:
        Updated list of cards with 0% empty.
    """
    from eng_words.llm.response_cache import CachedProvider
    from eng_words.llm.smart_card_generator import SmartCard

    # Wrap provider with cache
    cached_provider = CachedProvider(provider, cache)

    # Separate empty and non-empty cards
    empty_cards = [c for c in cards if not c.selected_examples]
    non_empty_cards = [deepcopy(c) for c in cards if c.selected_examples]

    if not empty_cards:
        logger.info("No empty cards to redistribute")
        return cards

    logger.info(f"Redistributing {len(empty_cards)} empty cards...")

    # Build index of existing cards by (lemma, synset)
    card_index: dict[tuple[str, str], SmartCard] = {}
    for card in non_empty_cards:
        key = (card.lemma, card.primary_synset)
        card_index[key] = card

    # Track new cards to create
    new_cards: dict[tuple[str, str], SmartCard] = {}

    # Process each empty card
    for empty_card in empty_cards:
        lemma = empty_card.lemma
        pos = empty_card.pos

        # Get all candidate synsets for this lemma
        candidates = _get_candidate_synsets_for_lemma(lemma, pos)

        if not candidates:
            logger.warning(f"No candidate synsets for '{lemma}' ({pos})")
            continue

        # Process each excluded example
        for sentence in empty_card.excluded_examples:
            # Call LLM WSD
            correct_synset = llm_wsd_sentence(
                lemma=lemma,
                sentence=sentence,
                candidate_synsets=candidates,
                provider=cached_provider,
            )

            if correct_synset is None:
                logger.debug(f"LLM returned NONE for '{lemma}' in '{sentence[:50]}...'")
                continue

            key = (lemma, correct_synset)

            # Check if card already exists
            if key in card_index:
                # Add to existing card
                card_index[key].selected_examples.append(sentence)
                logger.debug(f"Added '{sentence[:30]}...' to existing card {correct_synset}")
            elif key in new_cards:
                # Add to newly created card
                new_cards[key].selected_examples.append(sentence)
                logger.debug(f"Added '{sentence[:30]}...' to new card {correct_synset}")
            else:
                # Create new card
                # Get definition for this synset
                synset_def = next(
                    (c["definition"] for c in candidates if c["synset_id"] == correct_synset),
                    "",
                )
                new_card = SmartCard(
                    lemma=lemma,
                    pos=pos,
                    supersense=empty_card.supersense,  # Keep original supersense
                    selected_examples=[sentence],
                    excluded_examples=[],
                    simple_definition=synset_def,  # Use WordNet definition
                    translation_ru="",  # Will need regeneration
                    generated_example="",
                    wn_definition=synset_def,
                    book_name=empty_card.book_name,
                    primary_synset=correct_synset,
                    synset_group=[correct_synset],
                )
                new_cards[key] = new_card
                logger.info(f"Created new card for '{lemma}' with synset {correct_synset}")

    # Generate translations for new cards
    if new_cards:
        logger.info(f"Generating translations for {len(new_cards)} new cards...")
        for key, card in new_cards.items():
            translation = _generate_translation(
                lemma=card.lemma,
                pos=card.pos,
                definition=card.simple_definition or card.wn_definition,
                example=card.selected_examples[0] if card.selected_examples else "",
                provider=cached_provider,
            )
            card.translation_ru = translation

    # Combine results
    result = list(non_empty_cards) + list(new_cards.values())

    # Log stats
    original_empty = len(empty_cards)
    final_empty = len([c for c in result if not c.selected_examples])

    logger.info(
        f"Redistribution complete: {original_empty} empty → {final_empty} empty, "
        f"{len(new_cards)} new cards created with translations"
    )

    return result


def _generate_translation(
    lemma: str,
    pos: str,
    definition: str,
    example: str,
    provider: LLMProvider,
) -> str:
    """Generate Russian translation for a word using LLM.

    Args:
        lemma: Word lemma
        pos: Part of speech
        definition: Word definition
        example: Example sentence
        provider: LLM provider instance

    Returns:
        Russian translation string
    """
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        lemma=lemma,
        pos=pos,
        definition=definition,
        example=example,
    )

    try:
        response: LLMResponse = provider.complete(prompt, max_output_tokens=50)
        translation = response.content.strip().strip('"\'')
        logger.debug(f"Translation for '{lemma}': {translation}")
        return translation
    except Exception as e:
        logger.warning(f"Failed to generate translation for '{lemma}': {e}")
        return ""

