#!/usr/bin/env python3
"""
Compare supersense-based vs synset-based card generation quality.

Этап 2.3: Сравнение качества подходов

Usage:
    uv run python scripts/compare_approaches.py
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eng_words.llm.smart_card_generator import SmartCard
from eng_words.validation.example_validator import validate_card_examples

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
OLD_CARDS_PATH = Path("data/output/american_tragedy_smart_cards/smart_cards_2000.json")
NEW_CARDS_PATH = Path("data/synset_cards/synset_smart_cards_final.json")
TOKENS_PATH = Path("data/processed/american_tragedy_tokens.parquet")
OUTPUT_DIR = Path("data/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_old_cards() -> list[dict]:
    """Load old supersense-based cards."""
    logger.info(f"Loading old cards from {OLD_CARDS_PATH}")
    with open(OLD_CARDS_PATH, "r", encoding="utf-8") as f:
        cards = json.load(f)
    logger.info(f"  Loaded {len(cards)} cards")
    return cards


def identify_problematic_cards(cards: list[dict], n: int = 50) -> list[dict]:
    """Identify problematic cards from old approach.

    Criteria:
    1. Cards without examples
    2. Cards with very few examples (1-2)
    3. Cards where lemma might not be in examples
    """
    logger.info(f"Identifying {n} problematic cards...")

    problematic = []

    # Category 1: No examples
    no_examples = [c for c in cards if not c.get("selected_examples")]
    logger.info(f"  Cards without examples: {len(no_examples)}")
    problematic.extend(no_examples[:20])

    # Category 2: Very few examples
    few_examples = [
        c for c in cards if c.get("selected_examples") and len(c["selected_examples"]) <= 2
    ]
    logger.info(f"  Cards with 1-2 examples: {len(few_examples)}")
    problematic.extend(few_examples[:20])

    # Category 3: Lemma might not be in examples (heuristic check)
    lemma_issues = []
    for c in cards:
        if c.get("selected_examples"):
            lemma = c.get("lemma", "").lower()
            has_lemma = any(lemma in ex.lower() for ex in c["selected_examples"])
            if not has_lemma:
                lemma_issues.append(c)

    logger.info(f"  Cards where lemma not in examples: {len(lemma_issues)}")
    problematic.extend(lemma_issues[:10])

    # Remove duplicates (by lemma + primary_synset if available)
    seen = set()
    unique_problematic = []
    for c in problematic:
        key = (c.get("lemma"), c.get("primary_synset", ""))
        if key not in seen:
            seen.add(key)
            unique_problematic.append(c)

    # Fill to n if needed
    if len(unique_problematic) < n:
        remaining = [c for c in cards if c not in unique_problematic]
        random.seed(42)
        unique_problematic.extend(random.sample(remaining, n - len(unique_problematic)))

    logger.info(f"  Selected {len(unique_problematic)} problematic cards")
    return unique_problematic[:n]


def find_matching_new_cards(problematic_old: list[dict], new_cards: list[dict]) -> dict[str, dict]:
    """Find matching new cards for problematic old cards.

    Matches by lemma and tries to match by synset if available.
    """
    logger.info("Finding matching new cards...")

    # Index new cards by lemma
    new_by_lemma: dict[str, list[dict]] = {}
    for c in new_cards:
        lemma = c.get("lemma", "").lower()
        if lemma not in new_by_lemma:
            new_by_lemma[lemma] = []
        new_by_lemma[lemma].append(c)

    matches = {}
    for old_card in problematic_old:
        lemma = old_card.get("lemma", "").lower()
        old_synset = old_card.get("primary_synset", "")

        if lemma not in new_by_lemma:
            continue

        # Try exact synset match first
        exact_match = None
        for new_card in new_by_lemma[lemma]:
            if new_card.get("primary_synset") == old_synset:
                exact_match = new_card
                break

        # If no exact match, take first available
        if exact_match:
            matches[lemma] = {"old": old_card, "new": exact_match, "match_type": "exact"}
        elif new_by_lemma[lemma]:
            matches[lemma] = {
                "old": old_card,
                "new": new_by_lemma[lemma][0],
                "match_type": "lemma_only",
            }

    logger.info(f"  Found {len(matches)} matches")
    return matches


def compare_card_quality(old_card: dict, new_card: dict) -> dict[str, Any]:
    """Compare quality of old vs new card."""
    comparison = {
        "lemma": old_card.get("lemma"),
        "old_synset": old_card.get("primary_synset", ""),
        "new_synset": new_card.get("primary_synset", ""),
        "match_type": (
            "exact"
            if old_card.get("primary_synset") == new_card.get("primary_synset")
            else "different"
        ),
    }

    # Examples comparison
    old_examples = old_card.get("selected_examples", [])
    new_examples = new_card.get("selected_examples", [])

    comparison["old_examples_count"] = len(old_examples)
    comparison["new_examples_count"] = len(new_examples)
    comparison["examples_improvement"] = len(new_examples) - len(old_examples)

    # Translation comparison
    old_translation = old_card.get("translation_ru", "")
    new_translation = new_card.get("translation_ru", "")

    comparison["old_has_translation"] = bool(old_translation)
    comparison["new_has_translation"] = bool(new_translation)

    # Definition comparison
    old_def = old_card.get("simple_definition", "")
    new_def = new_card.get("simple_definition", "")

    comparison["old_definition_length"] = len(old_def)
    comparison["new_definition_length"] = len(new_def)

    # Validate new card examples
    try:
        new_smart_card = SmartCard(
            lemma=new_card.get("lemma", ""),
            pos=new_card.get("pos", ""),
            supersense=new_card.get("supersense", ""),
            selected_examples=new_examples,
            excluded_examples=new_card.get("excluded_examples", []),
            simple_definition=new_def,
            translation_ru=new_translation,
            generated_example=new_card.get("generated_example", ""),
            wn_definition=new_card.get("wn_definition", ""),
            book_name=new_card.get("book_name", "american_tragedy"),
            primary_synset=new_card.get("primary_synset", ""),
            synset_group=new_card.get("synset_group", [new_card.get("primary_synset", "")]),
        )
        validation = validate_card_examples(new_smart_card)
        comparison["new_card_valid"] = validation.is_valid
        comparison["new_card_found_forms"] = validation.found_forms
    except Exception as e:
        logger.warning(f"Failed to validate card {new_card.get('lemma')}: {e}")
        comparison["new_card_valid"] = False

    # Validate old card examples (if possible)
    try:
        old_smart_card = SmartCard(
            lemma=old_card.get("lemma", ""),
            pos=old_card.get("pos", ""),
            supersense=old_card.get("supersense", ""),
            selected_examples=old_examples,
            excluded_examples=old_card.get("excluded_examples", []),
            simple_definition=old_card.get("simple_definition", ""),
            translation_ru=old_translation,
            generated_example=old_card.get("generated_example", ""),
            wn_definition=old_card.get("wn_definition", ""),
            book_name=old_card.get("book_name", "american_tragedy"),
            primary_synset=old_card.get("primary_synset", ""),
            synset_group=[old_card.get("primary_synset", "")],
        )
        old_validation = validate_card_examples(old_smart_card)
        comparison["old_card_valid"] = old_validation.is_valid
    except Exception:
        comparison["old_card_valid"] = None

    return comparison


def generate_comparison_report(matches: dict[str, dict]) -> dict[str, Any]:
    """Generate comparison report."""
    logger.info("Generating comparison report...")

    comparisons = []
    for lemma, match_data in matches.items():
        comp = compare_card_quality(match_data["old"], match_data["new"])
        comp["match_type"] = match_data["match_type"]
        comparisons.append(comp)

    # Calculate statistics
    total = len(comparisons)
    old_no_examples = sum(1 for c in comparisons if c["old_examples_count"] == 0)
    new_no_examples = sum(1 for c in comparisons if c["new_examples_count"] == 0)

    examples_improved = sum(1 for c in comparisons if c["examples_improvement"] > 0)
    examples_same = sum(1 for c in comparisons if c["examples_improvement"] == 0)
    examples_worse = sum(1 for c in comparisons if c["examples_improvement"] < 0)

    old_valid = sum(1 for c in comparisons if c.get("old_card_valid") is True)
    new_valid = sum(1 for c in comparisons if c.get("new_card_valid") is True)

    report = {
        "summary": {
            "total_comparisons": total,
            "old_approach": {
                "cards_without_examples": old_no_examples,
                "cards_without_examples_pct": old_no_examples / total * 100 if total > 0 else 0,
                "valid_cards": old_valid,
                "valid_cards_pct": old_valid / total * 100 if total > 0 else 0,
            },
            "new_approach": {
                "cards_without_examples": new_no_examples,
                "cards_without_examples_pct": new_no_examples / total * 100 if total > 0 else 0,
                "valid_cards": new_valid,
                "valid_cards_pct": new_valid / total * 100 if total > 0 else 0,
            },
            "improvements": {
                "examples_improved": examples_improved,
                "examples_same": examples_same,
                "examples_worse": examples_worse,
                "examples_improved_pct": examples_improved / total * 100 if total > 0 else 0,
            },
        },
        "comparisons": comparisons,
    }

    return report


def main():
    """Main comparison function."""
    logger.info("=" * 70)
    logger.info("COMPARISON: Supersense vs Synset Approaches")
    logger.info("=" * 70)

    # Step 1: Load old cards
    old_cards = load_old_cards()

    # Step 2: Identify problematic cards
    problematic = identify_problematic_cards(old_cards, n=50)

    # Step 3: Load new cards
    logger.info(f"\nLoading new cards from {NEW_CARDS_PATH}")
    if not NEW_CARDS_PATH.exists():
        logger.warning(f"  New cards file not found: {NEW_CARDS_PATH}")
        logger.info("  Waiting for full generation to complete...")
        return

    with open(NEW_CARDS_PATH, "r", encoding="utf-8") as f:
        new_cards = json.load(f)
    logger.info(f"  Loaded {len(new_cards)} new cards")

    # Step 4: Find matches
    matches = find_matching_new_cards(problematic, new_cards)

    if not matches:
        logger.warning("No matches found. Cannot compare.")
        return

    # Step 5: Compare quality
    report = generate_comparison_report(matches)

    # Step 6: Save report
    report_path = OUTPUT_DIR / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved report to {report_path}")

    # Step 7: Print summary
    summary = report["summary"]
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n📊 Сравнение {summary['total_comparisons']} карточек:")
    print()
    print("Старый подход (supersense):")
    print(
        f"  Без примеров:     {summary['old_approach']['cards_without_examples']} ({summary['old_approach']['cards_without_examples_pct']:.1f}%)"
    )
    print(
        f"  Валидные:         {summary['old_approach']['valid_cards']} ({summary['old_approach']['valid_cards_pct']:.1f}%)"
    )
    print()
    print("Новый подход (synset):")
    print(
        f"  Без примеров:     {summary['new_approach']['cards_without_examples']} ({summary['new_approach']['cards_without_examples_pct']:.1f}%)"
    )
    print(
        f"  Валидные:         {summary['new_approach']['valid_cards']} ({summary['new_approach']['valid_cards_pct']:.1f}%)"
    )
    print()
    print("Улучшения:")
    print(
        f"  Примеров больше:  {summary['improvements']['examples_improved']} ({summary['improvements']['examples_improved_pct']:.1f}%)"
    )
    print(f"  Примеров столько же: {summary['improvements']['examples_same']}")
    print(f"  Примеров меньше:  {summary['improvements']['examples_worse']}")
    print()

    # Show top improvements
    top_improvements = sorted(
        report["comparisons"],
        key=lambda x: x.get("examples_improvement", 0),
        reverse=True,
    )[:10]

    print("Топ-10 улучшений по количеству примеров:")
    for i, comp in enumerate(top_improvements, 1):
        print(
            f"  {i}. {comp['lemma']}: {comp['old_examples_count']} → {comp['new_examples_count']} примеров"
        )

    print("\n" + "=" * 70)
    print("✅ Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
