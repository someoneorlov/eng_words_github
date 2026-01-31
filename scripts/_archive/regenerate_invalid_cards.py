#!/usr/bin/env python3
"""Regenerate examples for invalid cards using improved prompt.

This script:
1. Loads cards from the final output
2. Finds cards with invalid examples
3. Regenerates examples using _generate_example_fallback
4. Validates and saves results

Usage:
    uv run python scripts/regenerate_invalid_cards.py
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import SmartCard, SmartCardGenerator
from eng_words.validation.example_validator import validate_card_examples

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
CARDS_PATH = Path("data/synset_cards/synset_smart_cards_final.json")
OUTPUT_PATH = Path("data/synset_cards/synset_smart_cards_regenerated.json")
CACHE_DIR = Path("data/synset_cards/llm_cache")


def main():
    """Regenerate examples for invalid cards."""
    logger.info("=" * 70)
    logger.info("REGENERATION OF INVALID CARDS")
    logger.info("=" * 70)

    # Step 1: Load cards
    logger.info("\n## Step 1: Load cards")
    if not CARDS_PATH.exists():
        logger.error(f"Error: Card file not found at {CARDS_PATH}")
        return

    with open(CARDS_PATH, "r", encoding="utf-8") as f:
        cards_data = json.load(f)

    cards = [SmartCard(**c) for c in cards_data]
    logger.info(f"  Loaded {len(cards):,} cards")

    # Step 2: Find invalid cards
    logger.info("\n## Step 2: Find invalid cards")
    invalid_cards = []
    for card in cards:
        if card.selected_examples:
            result = validate_card_examples(card)
            if not result.is_valid:
                invalid_cards.append((card, result))

    logger.info(f"  Found {len(invalid_cards)} invalid cards")
    if not invalid_cards:
        logger.info("  ✅ All cards are valid!")
        return

    # Step 3: Initialize generator
    logger.info("\n## Step 3: Initialize SmartCardGenerator")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)
    generator = SmartCardGenerator(
        provider=provider, cache=cache, book_name="american_tragedy", max_retries=2
    )

    # Step 4: Regenerate examples
    logger.info("\n## Step 4: Regenerate examples")
    regenerated_count = 0
    fixed_count = 0

    for i, (card, validation_result) in enumerate(invalid_cards, 1):
        lemma = card.lemma
        pos = card.pos
        definition = card.simple_definition or card.wn_definition

        logger.info(f"\n  [{i}/{len(invalid_cards)}] Regenerating '{lemma}' ({pos})")
        logger.info(f"    Definition: {definition}")
        logger.info(f"    Invalid examples: {len(validation_result.invalid_examples)}")

        # Regenerate example
        new_example = generator._generate_example_fallback(
            lemma=lemma,
            pos=pos,
            definition=definition,
        )

        if not new_example:
            logger.warning(f"    ❌ Failed to generate example for '{lemma}'")
            continue

        logger.info(f"    ✅ Generated: {new_example}")

        # Update card
        card.generated_example = new_example
        # Clear invalid examples and use generated_example
        card.selected_examples = [new_example]
        card.excluded_examples = card.excluded_examples + validation_result.invalid_examples

        # Validate new card
        new_result = validate_card_examples(card)
        if new_result.is_valid:
            logger.info("    ✅ New example is VALID")
            fixed_count += 1
        else:
            logger.warning(f"    ⚠️  New example is still INVALID: {new_result.invalid_examples}")

        regenerated_count += 1

    # Step 5: Save results
    logger.info("\n## Step 5: Save results")
    logger.info(f"  Regenerated: {regenerated_count}/{len(invalid_cards)}")
    logger.info(f"  Fixed (valid): {fixed_count}/{len(invalid_cards)}")

    # Convert cards back to dict
    cards_dict = [card.to_dict() for card in cards]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cards_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"  Saved to {OUTPUT_PATH}")

    # Step 6: Final validation
    logger.info("\n## Step 6: Final validation")
    final_valid = sum(1 for c in cards if validate_card_examples(c).is_valid)
    final_total = len(cards)

    logger.info(f"  Total cards: {final_total:,}")
    logger.info(f"  Valid cards: {final_valid:,} ({final_valid/final_total*100:.1f}%)")
    logger.info(
        f"  Invalid cards: {final_total - final_valid} ({(final_total - final_valid)/final_total*100:.1f}%)"
    )

    # LLM stats
    stats = generator.stats()
    logger.info(f"\n  LLM Cost: ${stats['total_cost']:.4f}")
    logger.info(f"  Total cards processed: {stats.get('total_cards', 0)}")

    logger.info("\n" + "=" * 70)
    logger.info("REGENERATION COMPLETE")
    logger.info("=" * 70)

    if fixed_count == len(invalid_cards):
        logger.info("  ✅ All invalid cards fixed!")
    else:
        logger.warning(f"  ⚠️  {len(invalid_cards) - fixed_count} cards still invalid")


if __name__ == "__main__":
    main()
