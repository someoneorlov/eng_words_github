#!/usr/bin/env python3
"""
Run Pipeline C (Word Family with WordNet hints) on experiment sample.

This is Pipeline B + WordNet definitions as hints for LLM.

Usage:
    uv run python scripts/experiment/run_pipeline_c.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eng_words.experiment.word_family_clusterer import (
    WordFamilyClusterer,
    group_examples_by_lemma,
)
from eng_words.llm.providers.gemini import GeminiProvider
from eng_words.llm.response_cache import ResponseCache
from eng_words.wsd.wordnet_utils import get_synsets_with_definitions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/experiment")
CACHE_DIR = Path("data/cache/llm_responses")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pipeline C (Word Family + WordNet hints)")
    parser.add_argument("--limit", type=int, default=None, help="Limit lemmas")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    return parser.parse_args()


def get_wordnet_hints(lemma: str, pos_variants: list) -> str:
    """Get WordNet definitions as hints for the LLM."""
    hints = []

    # Map POS variants to WordNet POS
    wn_pos_map = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r",
    }

    seen_definitions = set()

    for pos in pos_variants:
        wn_pos = wn_pos_map.get(pos)
        if not wn_pos:
            continue

        synsets = get_synsets_with_definitions(lemma, wn_pos)

        for synset_id, definition in synsets[:5]:  # Limit to 5 per POS
            if definition not in seen_definitions:
                seen_definitions.add(definition)
                hints.append(f"- {synset_id}: {definition}")

    if hints:
        return "\n".join(hints[:10])  # Max 10 hints total

    return ""


def main():
    args = parse_args()

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PIPELINE C (WORD FAMILY + WORDNET HINTS)")
    logger.info("=" * 70)

    # Load sample
    logger.info("\n## Loading sample data...")
    tokens = pd.read_parquet(DATA_DIR / "tokens_sample.parquet")
    sentences = pd.read_parquet(DATA_DIR / "sentences_sample.parquet")
    logger.info(f"  Loaded {len(tokens):,} tokens, {len(sentences):,} sentences")

    # Group by lemma
    logger.info("\n## Grouping examples by lemma...")
    lemma_groups = group_examples_by_lemma(tokens, sentences)
    logger.info(f"  Found {len(lemma_groups)} content lemmas")

    if args.limit:
        lemma_groups = lemma_groups.head(args.limit)
        logger.info(f"  Limited to {len(lemma_groups)} lemmas")

    # Initialize clusterer with WordNet hints
    provider = GeminiProvider()
    cache = ResponseCache(CACHE_DIR, enabled=not args.no_cache)

    clusterer = WordFamilyClusterer(
        provider=provider,
        cache=cache,
        use_wordnet_hints=True,  # KEY DIFFERENCE from Pipeline B
    )

    # Process lemmas
    all_cards = []
    errors = []

    total = len(lemma_groups)
    pbar = tqdm(lemma_groups.iterrows(), total=total, desc="Processing lemmas")

    for idx, row in pbar:
        lemma = row["lemma"]
        examples = row["examples"]
        sentence_ids = row["sentence_ids"]
        pos_variants = row["pos_variants"]

        pbar.set_postfix({"lemma": lemma[:15], "cards": len(all_cards)})

        try:
            # Get WordNet hints for this lemma
            wordnet_hints = get_wordnet_hints(lemma, pos_variants)

            result = clusterer.cluster_lemma(
                lemma=lemma,
                examples=examples,
                sentence_ids=sentence_ids,
                wordnet_hints=wordnet_hints if wordnet_hints else None,
            )

            if result.cards:
                for card in result.cards:
                    card["source"] = "pipeline_c"
                    card["total_lemma_examples"] = len(examples)
                    card["had_wordnet_hints"] = bool(wordnet_hints)
                all_cards.extend(result.cards)
            else:
                errors.append(
                    {
                        "lemma": lemma,
                        "error": "no_cards",
                        "examples_count": len(examples),
                    }
                )

        except Exception as e:
            logger.error(f"Error for '{lemma}': {e}")
            errors.append(
                {
                    "lemma": lemma,
                    "error": str(e),
                    "examples_count": len(examples),
                }
            )

    # Save results
    elapsed = time.time() - start_time
    stats = clusterer.stats()

    output_path = Path(args.output) if args.output else DATA_DIR / "cards_C.json"

    results = {
        "pipeline": "C",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "use_wordnet_hints": True,
            "max_examples_per_batch": 100,
            "limit": args.limit,
        },
        "stats": {
            "lemmas_processed": total,
            "cards_generated": len(all_cards),
            "errors": len(errors),
            **stats,
        },
        "cards": all_cards,
        "errors": errors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE C COMPLETE")
    print("=" * 70)
    print(f"Lemmas processed: {total}")
    print(f"Cards generated: {len(all_cards)}")
    print(f"Errors: {len(errors)}")
    print("\nAPI stats:")
    print(f"  - API calls: {stats['total_api_calls']}")
    print(f"  - Cache hits: {stats['cache_hits']}")
    print(f"  - Cost: ${stats['total_cost_usd']:.4f}")
    print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
