#!/usr/bin/env python3
"""
Run Pipeline B (Word Family) on experiment sample.

This script:
1. Loads the sample data
2. Groups examples by lemma
3. Clusters each lemma using LLM
4. Saves results to data/experiment/cards_B.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eng_words.experiment.word_family_clusterer import (
    WordFamilyClusterer,
    group_examples_by_lemma,
)
from eng_words.llm.providers.gemini import GeminiProvider
from eng_words.llm.response_cache import ResponseCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/experiment")
CACHE_DIR = Path("data/cache/llm_responses")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Pipeline B (Word Family)")

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of lemmas to process (for testing)"
    )
    parser.add_argument(
        "--use-hints", action="store_true", help="Use WordNet hints (Pipeline C variant)"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable response caching")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/experiment/cards_B.json)",
    )

    return parser.parse_args()


def load_sample_data():
    """Load sample tokens and sentences."""
    tokens_path = DATA_DIR / "tokens_sample.parquet"
    sentences_path = DATA_DIR / "sentences_sample.parquet"

    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Sample not found: {tokens_path}\n"
            "Run: uv run python scripts/experiment/prepare_sample.py"
        )

    logger.info("Loading sample data...")
    tokens = pd.read_parquet(tokens_path)
    sentences = pd.read_parquet(sentences_path)

    logger.info(f"Loaded {len(tokens):,} tokens, {len(sentences):,} sentences")
    return tokens, sentences


def main():
    args = parse_args()

    start_time = datetime.now()
    logger.info(f"Starting Pipeline B at {start_time}")

    # 1. Load data
    tokens, sentences = load_sample_data()

    # 2. Group examples by lemma
    logger.info("Grouping examples by lemma...")
    lemma_groups = group_examples_by_lemma(tokens, sentences)
    logger.info(f"Found {len(lemma_groups)} content lemmas")

    # Apply limit if specified
    if args.limit:
        lemma_groups = lemma_groups.head(args.limit)
        logger.info(f"Limited to {len(lemma_groups)} lemmas")

    # 3. Initialize clusterer
    provider = GeminiProvider()
    cache = ResponseCache(CACHE_DIR, enabled=not args.no_cache)

    clusterer = WordFamilyClusterer(
        provider=provider,
        cache=cache,
        use_wordnet_hints=args.use_hints,
    )

    # 4. Process each lemma
    all_cards = []
    errors = []

    total = len(lemma_groups)

    from tqdm import tqdm

    pbar = tqdm(lemma_groups.iterrows(), total=total, desc="Processing lemmas")

    for idx, row in pbar:
        lemma = row["lemma"]
        examples = row["examples"]
        sentence_ids = row["sentence_ids"]

        pbar.set_postfix({"lemma": lemma[:15], "cards": len(all_cards)})

        try:
            result = clusterer.cluster_lemma(
                lemma=lemma,
                examples=examples,
                sentence_ids=sentence_ids,
            )

            if result.cards:
                for card in result.cards:
                    card["source"] = "pipeline_b"
                    card["total_lemma_examples"] = len(examples)
                all_cards.extend(result.cards)
            else:
                logger.debug(f"No cards created for '{lemma}'")
                errors.append(
                    {
                        "lemma": lemma,
                        "error": "no_cards",
                        "examples_count": len(examples),
                    }
                )

        except Exception as e:
            logger.error(f"  -> Error: {e}")
            errors.append(
                {
                    "lemma": lemma,
                    "error": str(e),
                    "examples_count": len(examples),
                }
            )

    # 5. Save results
    output_path = Path(args.output) if args.output else DATA_DIR / "cards_B.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "pipeline": "B" if not args.use_hints else "C",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "use_wordnet_hints": args.use_hints,
            "max_examples_per_batch": 100,
            "limit": args.limit,
        },
        "stats": {
            "lemmas_processed": total,
            "cards_generated": len(all_cards),
            "errors": len(errors),
            **clusterer.stats(),
        },
        "cards": all_cards,
        "errors": errors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(all_cards)} cards to {output_path}")

    # 6. Print summary
    elapsed = datetime.now() - start_time
    stats = clusterer.stats()

    print("\n" + "=" * 60)
    print("PIPELINE B COMPLETE")
    print("=" * 60)
    print(f"Lemmas processed: {total}")
    print(f"Cards generated: {len(all_cards)}")
    print(f"Errors: {len(errors)}")
    print("\nAPI stats:")
    print(f"  - API calls: {stats['total_api_calls']}")
    print(f"  - Cache hits: {stats['cache_hits']}")
    print(f"  - Input tokens: {stats['total_input_tokens']:,}")
    print(f"  - Output tokens: {stats['total_output_tokens']:,}")
    print(f"  - Cost: ${stats['total_cost_usd']:.4f}")
    print(f"\nElapsed time: {elapsed}")
    print(f"Output: {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
