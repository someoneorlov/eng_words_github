#!/usr/bin/env python3
"""
Test prompt refinements on selected lemmas.

Test set includes:
- OVERSPLIT: lemmas where B created too many similar cards
- CORRECT: lemmas where B correctly split into different meanings

Usage:
    uv run python scripts/experiment/prompt_refinement_test.py
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test set definitions - EXPANDED (50+ each)
# Auto-selected based on POS distribution heuristic

def build_test_sets():
    """Build test sets from cards_B.json using POS heuristic."""
    import json
    
    with open('data/experiment/cards_B.json') as f:
        b = json.load(f)
    
    # Group cards by lemma
    cards_by_lemma = {}
    for c in b['cards']:
        lemma = c['lemma']
        if lemma not in cards_by_lemma:
            cards_by_lemma[lemma] = []
        cards_by_lemma[lemma].append(c)
    
    oversplit = []  # Same POS, likely over-split
    correct = []    # Different POS, likely correct
    
    for lemma, cards in cards_by_lemma.items():
        if len(cards) < 2:
            continue
        
        pos_set = set(c.get('part_of_speech', 'unknown') for c in cards)
        n_cards = len(cards)
        
        # Same POS = likely oversplit
        if len(pos_set) == 1:
            # Target: reduce by ~50%
            target = max(1, n_cards // 2)
            oversplit.append((lemma, n_cards, target))
        # Different POS = likely correct
        else:
            # Target: keep same
            target = n_cards
            correct.append((lemma, n_cards, target))
    
    # Sort by card count and take top 50
    oversplit.sort(key=lambda x: -x[1])
    correct.sort(key=lambda x: -x[1])
    
    return oversplit[:50], correct[:50]

# Build test sets dynamically
_oversplit, _correct = build_test_sets()

OVERSPLIT_LEMMAS = [x[0] for x in _oversplit]
CORRECT_LEMMAS = [x[0] for x in _correct]

# Expected card counts
EXPECTED_AFTER = {}
for lemma, n_cards, target in _oversplit:
    EXPECTED_AFTER[lemma] = target
for lemma, n_cards, target in _correct:
    EXPECTED_AFTER[lemma] = target


def load_test_data():
    """Load examples for test lemmas from tokens and sentences."""
    import pandas as pd
    
    tokens = pd.read_parquet('data/experiment/tokens_sample.parquet')
    sentences = pd.read_parquet('data/experiment/sentences_sample.parquet')
    
    # Filter content words
    content = tokens[
        (tokens['is_alpha'] == True) & 
        (tokens['is_stop'] == False) &
        (tokens['pos'].isin(['NOUN', 'VERB', 'ADJ', 'ADV']))
    ]
    
    sent_lookup = dict(zip(sentences['sentence_id'], sentences['text']))
    
    test_data = {}
    all_lemmas = OVERSPLIT_LEMMAS + CORRECT_LEMMAS
    
    for lemma in all_lemmas:
        lemma_tokens = content[content['lemma'] == lemma]
        sentence_ids = lemma_tokens['sentence_id'].unique().tolist()
        examples = [sent_lookup[sid] for sid in sentence_ids if sid in sent_lookup]
        
        test_data[lemma] = {
            'lemma': lemma,
            'examples': examples,
            'sentence_ids': sentence_ids,
            'category': 'oversplit' if lemma in OVERSPLIT_LEMMAS else 'correct',
            'expected_cards': EXPECTED_AFTER.get(lemma, 2),
        }
    
    return test_data


def run_baseline(test_data: dict, model: str = "gemini-2.5-flash"):
    """Run baseline prompt on test data."""
    from eng_words.experiment.word_family_clusterer import WordFamilyClusterer
    from eng_words.llm.providers.gemini import GeminiProvider
    
    provider = GeminiProvider(model=model, temperature=0.0)
    clusterer = WordFamilyClusterer(provider=provider, cache=None, use_wordnet_hints=False)
    
    results = {}
    
    for lemma, data in test_data.items():
        logger.info(f"Processing {lemma} ({data['category']})...")
        
        try:
            result = clusterer.cluster_lemma(
                lemma=lemma,
                examples=data['examples'],
                sentence_ids=data['sentence_ids'],
            )
            
            results[lemma] = {
                'lemma': lemma,
                'category': data['category'],
                'expected': data['expected_cards'],
                'actual': len(result.cards),
                'cards': result.cards,
                'cost': result.cost_usd,
            }
            
            status = "✅" if len(result.cards) == data['expected_cards'] else "⚠️"
            logger.info(f"  {status} {lemma}: {len(result.cards)} cards (expected {data['expected_cards']})")
            
        except Exception as e:
            logger.error(f"  ❌ Error for {lemma}: {e}")
            results[lemma] = {'error': str(e)}
    
    return results


def evaluate_results(results: dict):
    """Evaluate results against expectations."""
    oversplit_before = 0
    oversplit_after = 0
    correct_before = 0
    correct_after = 0
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\n## OVERSPLIT (should decrease)")
    for lemma in OVERSPLIT_LEMMAS:
        if lemma in results and 'actual' in results[lemma]:
            r = results[lemma]
            expected = r['expected']
            actual = r['actual']
            
            # Original count from cards_B.json
            with open('data/experiment/cards_B.json') as f:
                b = json.load(f)
            original = len([c for c in b['cards'] if c['lemma'] == lemma])
            
            status = "✅" if actual <= expected else "❌"
            print(f"  {status} {lemma}: {original} → {actual} (target: {expected})")
            
            oversplit_before += original
            oversplit_after += actual
    
    print("\n## CORRECT (should stay same)")
    for lemma in CORRECT_LEMMAS:
        if lemma in results and 'actual' in results[lemma]:
            r = results[lemma]
            expected = r['expected']
            actual = r['actual']
            
            with open('data/experiment/cards_B.json') as f:
                b = json.load(f)
            original = len([c for c in b['cards'] if c['lemma'] == lemma])
            
            status = "✅" if actual >= expected - 1 else "❌"
            print(f"  {status} {lemma}: {original} → {actual} (target: {expected})")
            
            correct_before += original
            correct_after += actual
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    oversplit_reduction = (oversplit_before - oversplit_after) / oversplit_before * 100 if oversplit_before > 0 else 0
    correct_preservation = correct_after / correct_before * 100 if correct_before > 0 else 0
    
    print(f"\nOVERSPLIT reduction: {oversplit_before} → {oversplit_after} ({oversplit_reduction:.1f}% reduction)")
    print(f"CORRECT preservation: {correct_before} → {correct_after} ({correct_preservation:.1f}% preserved)")
    
    success = oversplit_reduction >= 30 and correct_preservation >= 90
    print(f"\n{'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}")
    print(f"  - Oversplit reduction ≥ 30%: {oversplit_reduction:.1f}% {'✅' if oversplit_reduction >= 30 else '❌'}")
    print(f"  - Correct preservation ≥ 90%: {correct_preservation:.1f}% {'✅' if correct_preservation >= 90 else '❌'}")


def main():
    logger.info("Loading test data...")
    test_data = load_test_data()
    
    logger.info(f"Test set: {len(OVERSPLIT_LEMMAS)} OVERSPLIT + {len(CORRECT_LEMMAS)} CORRECT")
    
    logger.info("\nRunning baseline (gemini-2.5-flash)...")
    results = run_baseline(test_data, model="gemini-2.5-flash")
    
    # Save results
    output_path = Path('data/experiment/prompt_test_baseline.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved to {output_path}")
    
    # Evaluate
    evaluate_results(results)


if __name__ == '__main__':
    main()
