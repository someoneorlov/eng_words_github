#!/usr/bin/env python3
"""Test full card generation pipeline with all fixes.

Includes:
1. Smart Fallback
2. LLM WSD redistribution
3. Example validation
4. Dialect filter
5. Translation generation
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from eng_words.aggregation.synset_aggregator import aggregate_by_synset
from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import SmartCardGenerator
from eng_words.text_processing import (
    create_sentences_dataframe,
    reconstruct_sentences_from_tokens,
)
from eng_words.validation.example_validator import fix_invalid_cards
from eng_words.wsd.llm_wsd import redistribute_empty_cards

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# Paths
SENSE_TOKENS_PATH = Path("data/processed/american_tragedy_wsd_sense_tokens.parquet")
TOKENS_PATH = Path("data/processed/american_tragedy_tokens.parquet")
OUTPUT_DIR = Path("data/full_pipeline_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(limit: int | None = None):
    """Load and prepare all data."""
    logger.info("Loading data...")

    # Load tokens and reconstruct sentences
    tokens_df = pd.read_parquet(TOKENS_PATH)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    logger.info(f"  Sentences: {len(sentences_df)}")

    # Load sense tokens and aggregate
    sense_tokens = pd.read_parquet(SENSE_TOKENS_PATH)
    logger.info(f"  Sense tokens: {len(sense_tokens)}")

    # Aggregate by synset (with dialect filter)
    synset_stats = aggregate_by_synset(sense_tokens, min_freq=2, filter_dialect=True)
    logger.info(f"  Synset aggregations: {len(synset_stats)}")

    if limit:
        synset_stats = synset_stats.head(limit)
        logger.info(f"  Limited to: {len(synset_stats)}")

    return synset_stats, sentences_df


def prepare_items(synset_stats: pd.DataFrame, sentences_df: pd.DataFrame) -> list[dict]:
    """Prepare items for card generation."""
    items = []
    for _, row in synset_stats.iterrows():
        # Get examples from sentence_ids
        sentence_ids = row["sentence_ids"]
        if hasattr(sentence_ids, "tolist"):
            sentence_ids = sentence_ids.tolist()

        examples = sentences_df[sentences_df["sentence_id"].isin(sentence_ids)][
            "sentence"
        ].tolist()[:10]

        items.append(
            {
                "lemma": row["lemma"],
                "pos": row["pos"],
                "supersense": row["supersense"],
                "wn_definition": row["definition"],
                "examples": examples,
                "synset_group": [row["synset_id"]],
                "primary_synset": row["synset_id"],
            }
        )

    return items


def run_full_pipeline(items: list[dict], provider, cache: ResponseCache):
    """Run full pipeline with all fixes."""

    # STEP 1: Generate cards with Smart Fallback
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Card Generation with Smart Fallback")
    logger.info("=" * 60)

    generator = SmartCardGenerator(
        provider=provider,
        cache=cache,
        book_name="american_tragedy",
    )

    start_time = time.time()
    cards = generator.generate_batch(items, progress=True, enable_fallback=True)
    gen_time = time.time() - start_time

    cards_with_examples = [c for c in cards if c.selected_examples]
    cards_without_examples = [c for c in cards if not c.selected_examples]
    fallback_used = [c for c in cards if c.fallback_used]

    logger.info(f"  Generated: {len(cards)} cards")
    logger.info(f"  With examples: {len(cards_with_examples)}")
    logger.info(f"  Without examples: {len(cards_without_examples)}")
    logger.info(f"  Fallback used: {len(fallback_used)}")
    logger.info(f"  Time: {gen_time:.1f}s")

    step1_stats = {
        "total": len(cards),
        "with_examples": len(cards_with_examples),
        "without_examples": len(cards_without_examples),
        "fallback_used": len(fallback_used),
        "time": gen_time,
    }

    # STEP 2: LLM WSD Redistribution
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: LLM WSD Redistribution")
    logger.info("=" * 60)

    start_time = time.time()
    cards_after_wsd = redistribute_empty_cards(cards, provider, cache)
    wsd_time = time.time() - start_time

    cards_with_examples_2 = [c for c in cards_after_wsd if c.selected_examples]
    cards_without_examples_2 = [c for c in cards_after_wsd if not c.selected_examples]

    logger.info(f"  Cards after WSD: {len(cards_after_wsd)}")
    logger.info(f"  With examples: {len(cards_with_examples_2)}")
    logger.info(f"  Without examples: {len(cards_without_examples_2)}")
    logger.info(f"  Time: {wsd_time:.1f}s")

    step2_stats = {
        "total": len(cards_after_wsd),
        "with_examples": len(cards_with_examples_2),
        "without_examples": len(cards_without_examples_2),
        "time": wsd_time,
    }

    # STEP 3: Example Validation
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Example Validation")
    logger.info("=" * 60)

    start_time = time.time()
    fixed_cards, removed_cards = fix_invalid_cards(
        cards_after_wsd,
        use_generated_example=True,
        remove_unfixable=True,
    )
    val_time = time.time() - start_time

    logger.info(f"  Valid cards: {len(fixed_cards)}")
    logger.info(f"  Removed: {len(removed_cards)}")
    logger.info(f"  Time: {val_time:.1f}s")

    step3_stats = {
        "valid": len(fixed_cards),
        "removed": len(removed_cards),
        "time": val_time,
    }

    # Final stats
    with_translation = len([c for c in fixed_cards if c.translation_ru])
    with_definition = len([c for c in fixed_cards if c.simple_definition])

    return fixed_cards, {
        "step1": step1_stats,
        "step2": step2_stats,
        "step3": step3_stats,
        "final": {
            "total_cards": len(fixed_cards),
            "with_translation": with_translation,
            "with_definition": with_definition,
            "translation_rate": with_translation / len(fixed_cards) * 100 if fixed_cards else 0,
        },
        "cache_stats": dict(cache.stats()),
    }


def main():
    """Main test function."""
    # Load data
    synset_stats, sentences_df = load_data(limit=500)
    items = prepare_items(synset_stats, sentences_df)
    logger.info(f"Prepared {len(items)} items for testing")

    # Initialize LLM provider and cache
    provider = get_provider("gemini")
    cache = ResponseCache(Path("data/llm_cache"))

    # Run full pipeline
    cards, stats = run_full_pipeline(items, provider, cache)

    # Save results
    cards_data = [asdict(c) for c in cards]
    cards_path = OUTPUT_DIR / "cards_validated.json"
    with open(cards_path, "w", encoding="utf-8") as f:
        json.dump(cards_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(cards)} cards to {cards_path}")

    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Input items:         {len(items)}")
    logger.info(f"  Final cards:         {stats['final']['total_cards']}")
    logger.info(
        f"  With translation:    {stats['final']['with_translation']} ({stats['final']['translation_rate']:.1f}%)"
    )
    logger.info(f"  Cards removed:       {stats['step3']['removed']}")
    logger.info(f"  Cache hits:          {stats['cache_stats'].get('hits', 0)}")
    logger.info(f"  Cache misses:        {stats['cache_stats'].get('misses', 0)}")

    logger.info("\n✅ Full pipeline test complete!")


if __name__ == "__main__":
    main()
