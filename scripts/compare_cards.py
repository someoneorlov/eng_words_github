#!/usr/bin/env python3
"""
Compare cards between reference (expected) and current results.

Usage:
    python scripts/compare_cards.py --expected backups/2026-01-19/benchmark_100/ --actual data/synset_cards/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_cards(json_path: Path) -> list[dict[str, Any]]:
    """Load cards from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_card(card: dict[str, Any]) -> dict[str, Any]:
    """Normalize card for comparison (sort lists, strip None)."""
    normalized = {
        "lemma": card.get("lemma", ""),
        "pos": card.get("pos", ""),
        "simple_definition": card.get("simple_definition", ""),
        "translation_ru": card.get("translation_ru", ""),
        "selected_examples": sorted(card.get("selected_examples", [])),
        "generated_examples": sorted(card.get("generated_examples", [])),
    }
    return normalized


def compare_cards(expected: list[dict], actual: list[dict]) -> tuple[bool, list[str]]:
    """Compare two lists of cards.

    Returns:
        (is_match, list_of_diffs)
    """
    diffs = []

    # Build dicts by lemma+pos for fast lookup
    expected_dict = {}
    for card in expected:
        key = (card.get("lemma", ""), card.get("pos", ""))
        expected_dict[key] = normalize_card(card)

    actual_dict = {}
    for card in actual:
        key = (card.get("lemma", ""), card.get("pos", ""))
        actual_dict[key] = normalize_card(card)

    # Check all expected cards
    for key in expected_dict:
        if key not in actual_dict:
            diffs.append(f"❌ Card {key[0]} ({key[1]}) missing in actual")
            continue

        exp = expected_dict[key]
        act = actual_dict[key]

        for field in ["lemma", "pos", "simple_definition", "translation_ru"]:
            if exp[field] != act[field]:
                diffs.append(
                    f"❌ {key[0]} ({key[1]}): field '{field}' differs\n"
                    f"   Expected: {exp[field]}\n"
                    f"   Actual:   {act[field]}"
                )

        if exp["selected_examples"] != act["selected_examples"]:
            diffs.append(
                f"❌ {key[0]} ({key[1]}): selected_examples differ\n"
                f"   Expected: {exp['selected_examples']}\n"
                f"   Actual:   {act['selected_examples']}"
            )

        if exp["generated_examples"] != act["generated_examples"]:
            diffs.append(
                f"❌ {key[0]} ({key[1]}): generated_examples differ\n"
                f"   Expected: {exp['generated_examples']}\n"
                f"   Actual:   {act['generated_examples']}"
            )

    # Check for extra cards in actual
    for key in actual_dict:
        if key not in expected_dict:
            diffs.append(f"⚠️  Card {key[0]} ({key[1]}) present in actual but not in expected")

    return len(diffs) == 0, diffs


def main():
    parser = argparse.ArgumentParser(
        description="Compare cards between reference and current results"
    )
    parser.add_argument(
        "--expected",
        type=Path,
        required=True,
        help="Path to directory with reference results (must contain synset_smart_cards_final.json)",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        required=True,
        help="Path to directory with current results (must contain synset_smart_cards_final.json)",
    )

    args = parser.parse_args()

    expected_path = args.expected / "synset_smart_cards_final.json"
    actual_path = args.actual / "synset_smart_cards_final.json"

    if not expected_path.exists():
        print(f"❌ Error: reference file not found: {expected_path}")
        sys.exit(1)

    if not actual_path.exists():
        print(f"❌ Error: actual file not found: {actual_path}")
        sys.exit(1)

    print(f"📊 Comparing cards:")
    print(f"   Expected: {expected_path}")
    print(f"   Actual:   {actual_path}")
    print()

    expected_cards = load_cards(expected_path)
    actual_cards = load_cards(actual_path)

    print(f"   Expected: {len(expected_cards)} cards")
    print(f"   Actual:   {len(actual_cards)} cards")
    print()

    is_match, diffs = compare_cards(expected_cards, actual_cards)

    if is_match:
        print("✅ All cards match!")
        sys.exit(0)
    else:
        print(f"❌ Found {len(diffs)} difference(s):\n")
        for diff in diffs:
            print(diff)
        sys.exit(1)


if __name__ == "__main__":
    main()
