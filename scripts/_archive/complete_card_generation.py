#!/usr/bin/env python3
"""
Завершить генерацию карточек: выполнить все недостающие шаги.

Этот скрипт:
1. Загружает частично сгенерированные карточки (partial файл)
2. Выполняет redistribute_empty_cards (Step 5) для карточек без примеров
3. Выполняет fix_invalid_cards (Step 6) для карточек с невалидными примерами
4. Гарантирует, что все карточки валидны и имеют примеры с леммой
5. Сохраняет финальный результат

Usage:
    uv run python scripts/complete_card_generation.py
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import SmartCard, SmartCardGenerator
from eng_words.validation.example_validator import fix_invalid_cards, validate_card_examples
from eng_words.wsd.llm_wsd import redistribute_empty_cards

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
OUTPUT_DIR = Path("data/synset_cards")
CACHE_DIR = OUTPUT_DIR / "llm_cache"
PARTIAL_PATH = OUTPUT_DIR / "synset_smart_cards_partial.json"
FINAL_PATH = OUTPUT_DIR / "synset_smart_cards_final.json"


def load_partial_cards() -> list[SmartCard]:
    """Загрузить частично сгенерированные карточки."""
    logger.info(f"Loading partial cards from {PARTIAL_PATH}")
    
    if not PARTIAL_PATH.exists():
        logger.error(f"Partial file not found: {PARTIAL_PATH}")
        logger.info("  Run run_synset_card_generation.py first to generate cards")
        return []
    
    with open(PARTIAL_PATH, "r", encoding="utf-8") as f:
        cards_data = json.load(f)
    
    cards = [SmartCard(**c) for c in cards_data]
    logger.info(f"  Loaded {len(cards):,} cards")
    return cards


def analyze_cards(cards: list[SmartCard]) -> dict[str, Any]:
    """Проанализировать текущее состояние карточек."""
    logger.info("\n## Analyzing current state")
    
    total = len(cards)
    empty = [c for c in cards if not c.selected_examples]
    invalid = []
    valid = []
    
    for card in cards:
        if card.selected_examples:
            validation = validate_card_examples(card)
            if validation.is_valid:
                valid.append(card)
            else:
                invalid.append((card, validation))
        else:
            invalid.append((card, None))
    
    stats = {
        "total": total,
        "empty": len(empty),
        "empty_pct": len(empty) / total * 100 if total > 0 else 0,
        "invalid": len(invalid),
        "invalid_pct": len(invalid) / total * 100 if total > 0 else 0,
        "valid": len(valid),
        "valid_pct": len(valid) / total * 100 if total > 0 else 0,
    }
    
    logger.info(f"  Total cards:        {stats['total']:,}")
    logger.info(f"  Empty (no examples): {stats['empty']} ({stats['empty_pct']:.1f}%)")
    logger.info(f"  Invalid examples:    {stats['invalid'] - stats['empty']} ({(stats['invalid'] - stats['empty'])/total*100:.1f}%)")
    logger.info(f"  Valid:              {stats['valid']} ({stats['valid_pct']:.1f}%)")
    
    # Показываем примеры проблемных карточек
    if empty:
        logger.info(f"\n  Examples of empty cards:")
        for i, card in enumerate(empty[:5], 1):
            logger.info(f"    {i}. {card.lemma} ({card.pos}) - {card.primary_synset}")
    
    invalid_examples_only = [c for c, v in invalid if v is not None]
    if invalid_examples_only:
        logger.info(f"\n  Examples of invalid cards (lemma not in examples):")
        for i, (card, validation) in enumerate(invalid[:5], 1):
            if validation is not None:
                logger.info(f"    {i}. {card.lemma} ({card.pos}) - {len(validation.invalid_examples)} invalid examples")
                logger.info(f"       Invalid: {validation.invalid_examples[0][:80]}..." if validation.invalid_examples else "")
    
    return stats


def main():
    """Главная функция - завершить генерацию карточек."""
    logger.info("=" * 70)
    logger.info("COMPLETE CARD GENERATION PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Load partial cards
    cards = load_partial_cards()
    
    if not cards:
        logger.error("No cards to process. Exiting.")
        return
    
    # Step 2: Analyze current state
    initial_stats = analyze_cards(cards)
    
    # Initialize provider and cache
    logger.info("\n## Initializing LLM provider and cache")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)
    card_generator = SmartCardGenerator(
        provider=provider, cache=cache, book_name="american_tragedy", max_retries=2
    )
    
    # Step 3: Redistribute empty cards (Step 5)
    logger.info("\n## Step 5: LLM WSD Redistribution")
    logger.info("  Redistributing empty cards...")
    start_wsd = time.time()
    
    cards_after_wsd = redistribute_empty_cards(
        cards,
        provider,
        cache,
    )
    
    wsd_time = time.time() - start_wsd
    
    wsd_stats = analyze_cards(cards_after_wsd)
    logger.info(f"  Time: {wsd_time:.1f}s")
    logger.info(f"  Improvement: {initial_stats['empty'] - wsd_stats['empty']} cards fixed")
    
    # Step 4: Fix invalid cards (Step 6)
    logger.info("\n## Step 6: Example Validation")
    logger.info("  Fixing invalid cards (ensuring lemma is in examples)...")
    start_val = time.time()
    
    final_cards, removed_cards = fix_invalid_cards(
        cards_after_wsd,
        use_generated_example=True,
        remove_unfixable=True,
    )
    
    val_time = time.time() - start_val
    
    logger.info(f"  Valid cards: {len(final_cards):,}")
    logger.info(f"  Removed (unfixable): {len(removed_cards):,}")
    logger.info(f"  Time: {val_time:.1f}s")
    
    # Final analysis
    logger.info("\n## Final Analysis")
    final_stats = analyze_cards(final_cards)
    
    # Verify all cards are valid
    logger.info("\n## Verification")
    all_valid = all(
        validate_card_examples(c).is_valid 
        for c in final_cards 
        if c.selected_examples
    )
    all_have_examples = all(c.selected_examples for c in final_cards)
    
    logger.info(f"  All cards have examples: {'✅' if all_have_examples else '❌'}")
    logger.info(f"  All examples are valid:  {'✅' if all_valid else '❌'}")
    
    if not all_have_examples:
        empty_final = [c for c in final_cards if not c.selected_examples]
        logger.warning(f"  ⚠️  {len(empty_final)} cards still have no examples:")
        for card in empty_final[:10]:
            logger.warning(f"      - {card.lemma} ({card.pos})")
    
    if not all_valid:
        invalid_final = [
            c for c in final_cards 
            if c.selected_examples and not validate_card_examples(c).is_valid
        ]
        logger.warning(f"  ⚠️  {len(invalid_final)} cards still have invalid examples:")
        for card in invalid_final[:10]:
            validation = validate_card_examples(card)
            logger.warning(f"      - {card.lemma} ({card.pos}): {len(validation.invalid_examples)} invalid")
    
    # Step 5: Save final results
    logger.info("\n## Step 7: Save Results")
    
    # Convert to serializable format
    final_data = [asdict(card) for card in final_cards]
    
    # Handle numpy types
    for card_dict in final_data:
        for key, value in card_dict.items():
            if hasattr(value, 'tolist'):  # numpy array
                card_dict[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                card_dict[key] = [
                    v.tolist() if hasattr(v, 'tolist') else v for v in value
                ]
    
    # Save final cards
    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved {len(final_cards):,} cards to {FINAL_PATH}")
    
    # Remove partial file if exists
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        logger.info(f"  Removed partial file: {PARTIAL_PATH}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Initial cards:     {initial_stats['total']:,}")
    logger.info(f"  Final cards:       {len(final_cards):,}")
    logger.info(f"  Removed:           {len(removed_cards):,}")
    logger.info("")
    logger.info(f"  Empty → Fixed:     {initial_stats['empty']} → {final_stats['empty']} ({initial_stats['empty'] - final_stats['empty']} fixed)")
    logger.info(f"  Invalid → Fixed:   {initial_stats['invalid']} → {final_stats['invalid']} ({initial_stats['invalid'] - final_stats['invalid']} fixed)")
    logger.info(f"  Valid cards:       {final_stats['valid']} ({final_stats['valid_pct']:.1f}%)")
    logger.info()
    logger.info(f"  Total time:        {wsd_time + val_time:.1f}s")
    logger.info(f"  WSD time:          {wsd_time:.1f}s")
    logger.info(f"  Validation time:   {val_time:.1f}s")
    logger.info()
    logger.info(f"  Cache stats:       {cache.stats()}")
    logger.info()
    
    if all_have_examples and all_valid:
        logger.info("✅ SUCCESS: All cards are valid and have examples with lemma!")
    else:
        logger.warning("⚠️  WARNING: Some cards still need attention (see above)")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

