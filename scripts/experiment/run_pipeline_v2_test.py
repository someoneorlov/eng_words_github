#!/usr/bin/env python3
"""
Test Pipeline v2 (two-stage) on a small sample.

Usage:
    uv run python scripts/experiment/run_pipeline_v2_test.py --limit 10
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

from eng_words.experiment.word_family_clusterer import group_examples_by_lemma
from eng_words.llm.providers.gemini import GeminiProvider
from eng_words.llm.response_cache import ResponseCache
from eng_words.pipeline_v2 import WordFamilyPipelineV2, save_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path('data/experiment')
CACHE_DIR = Path('data/cache/llm_responses')


def parse_args():
    parser = argparse.ArgumentParser(description='Test Pipeline v2')
    parser.add_argument('--limit', type=int, default=10, help='Number of lemmas')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='Model to use')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    return parser.parse_args()


def main():
    args = parse_args()
    
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PIPELINE V2 TEST (Two-Stage)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Limit: {args.limit} lemmas")
    
    # Load sample
    logger.info("\n## Loading sample data...")
    tokens = pd.read_parquet(DATA_DIR / 'tokens_sample.parquet')
    sentences = pd.read_parquet(DATA_DIR / 'sentences_sample.parquet')
    logger.info(f"  Loaded {len(tokens):,} tokens, {len(sentences):,} sentences")
    
    # Group by lemma
    logger.info("\n## Grouping examples by lemma...")
    lemma_groups = group_examples_by_lemma(tokens, sentences)
    logger.info(f"  Found {len(lemma_groups)} content lemmas")
    
    # Apply limit
    if args.limit:
        lemma_groups = lemma_groups.head(args.limit)
        logger.info(f"  Limited to {len(lemma_groups)} lemmas")
    
    # Initialize pipeline
    logger.info("\n## Initializing Pipeline v2...")
    provider = GeminiProvider(model=args.model, temperature=0.0)
    cache = ResponseCache(CACHE_DIR, enabled=not args.no_cache)
    pipeline = WordFamilyPipelineV2(provider=provider, cache=cache)
    
    # Process
    logger.info("\n## Processing lemmas (two stages)...")
    results = pipeline.process_batch(lemma_groups, progress=True)
    
    # Save results
    elapsed = time.time() - start_time
    stats = pipeline.stats()
    
    output_path = Path(args.output) if args.output else DATA_DIR / 'cards_v2_test.json'
    save_results(results, output_path, include_extraction=True)
    
    # Summary
    total_cards = sum(r.total_cards for r in results)
    total_meanings = sum(len(r.extraction.meanings) for r in results)
    phrasal_cards = sum(
        1 for r in results 
        for c in r.generation.cards 
        if c.is_phrasal
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE V2 TEST COMPLETE")
    print("=" * 70)
    print(f"Lemmas processed: {len(results)}")
    print(f"Meanings extracted: {total_meanings}")
    print(f"Cards generated: {total_cards}")
    print(f"  - Regular: {total_cards - phrasal_cards}")
    print(f"  - Phrasal verbs: {phrasal_cards}")
    print(f"\nAPI stats:")
    print(f"  - Stage 1 calls: {stats['extraction']['total_api_calls']}")
    print(f"  - Stage 2 calls: {stats['generation']['total_api_calls']}")
    print(f"  - Cache hits: {stats['total_cache_hits']}")
    print(f"  - Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    # Show sample cards
    print("\n## Sample cards:")
    for r in results[:3]:
        print(f"\n### {r.extraction.lemma}")
        print(f"  Meanings: {len(r.extraction.meanings)}")
        for card in r.generation.cards[:2]:
            print(f"  - [{card.part_of_speech}] {card.lemma_display}: {card.definition_en[:50]}...")
            if card.is_phrasal:
                print(f"    (PHRASAL: {card.phrasal_form})")


if __name__ == '__main__':
    main()
