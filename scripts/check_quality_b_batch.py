#!/usr/bin/env python3
"""Quality report for Pipeline B batch result (cards_B_batch.json).

Computes stats and writes a markdown report with a random sample of cards
for manual quality review.

Usage (from project root):
    uv run python scripts/check_quality_b_batch.py
    uv run python scripts/check_quality_b_batch.py --sample 50 --output data/experiment/quality_report_B_batch.md
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "experiment" / "cards_B_batch.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "experiment" / "quality_report_B_batch.md"


def load_cards(path: Path) -> tuple[list, dict, list]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return (
        data.get("cards", []),
        data.get("stats", {}),
        data.get("lemmas_with_zero_cards", []),
    )


def format_example(ex: str, max_len: int = 250) -> str:
    """One example line: take first line to avoid chapter headers, truncate."""
    first_line = ex.split("\n")[0].strip()
    if len(first_line) > max_len:
        first_line = first_line[: max_len - 3] + "..."
    return first_line


def lemma_stats(cards: list) -> dict:
    """Cards per lemma: lemma -> count."""
    by_lemma: dict[str, int] = {}
    for c in cards:
        lemma = c.get("lemma", "")
        by_lemma[lemma] = by_lemma.get(lemma, 0) + 1
    return by_lemma


def distribution(by_lemma: dict) -> dict[int, int]:
    """Cards per lemma -> number of lemmas with that count."""
    dist: dict[int, int] = {}
    for count in by_lemma.values():
        dist[count] = dist.get(count, 0) + 1
    return dict(sorted(dist.items()))


def main():
    parser = argparse.ArgumentParser(description="Quality report for cards_B_batch.json")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Input JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output markdown path",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=30,
        help="Number of cards to sample for manual review",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    cards, file_stats, lemmas_with_zero_cards = load_cards(args.input)
    if not cards:
        raise SystemExit("No cards in file.")

    by_lemma = lemma_stats(cards)
    lemmas_count = len(by_lemma)
    cards_count = len(cards)
    avg = cards_count / lemmas_count if lemmas_count else 0
    dist = distribution(by_lemma)
    zero_count = len(lemmas_with_zero_cards)

    # Random sample for manual review
    rng = random.Random(args.seed)
    sample_size = min(args.sample, len(cards))
    sample_cards = rng.sample(cards, sample_size)

    lines = [
        "# Quality Report: Pipeline B (Batch API)",
        "",
        "**Source:** `cards_B_batch.json`",
        "",
        "## Stats",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total cards | {cards_count:,} |",
        f"| Unique lemmas | {lemmas_count:,} |",
        f"| **Cards/lemma (avg)** | **{avg:.2f}** |",
        f"| Errors (parse/API) | {file_stats.get('errors', 0)} |",
        f"| Lemmas with 0 cards | {zero_count} |",
        "",
        "**Reference (doc, B Standard API full run):** 4,515 cards, 3,229 lemmas → 1.40 cards/lemma.",
        "",
    ]
    if lemmas_with_zero_cards:
        lines.extend([
            "**Lemmas with 0 cards** (LLM returned empty `cards`; see `download_log.json` for full list):",
            ", ".join(lemmas_with_zero_cards[:20]) + (f" ... +{len(lemmas_with_zero_cards) - 20}" if len(lemmas_with_zero_cards) > 20 else ""),
            "",
        ])
    lines.extend([
        "## Distribution (cards per lemma)",
        "",
        "| Cards/lemma | # lemmas |",
        "|------------|----------|",
    ])
    for k, v in dist.items():
        lines.append(f"| {k} | {v:,} |")
    lines.extend([
        "",
        "---",
        "",
        f"## Random sample for manual review ({len(sample_cards)} cards, seed={args.seed})",
        "",
        "Check: definition accuracy, translation, oversplit (too many cards for one meaning), missing senses.",
        "",
    ])

    for i, c in enumerate(sample_cards, 1):
        lemma = c.get("lemma", "?")
        def_en = c.get("definition_en", "")
        def_ru = c.get("definition_ru", "")
        pos = c.get("part_of_speech", "")
        examples = c.get("examples") or []
        ex_lines = [format_example(ex) for ex in examples if ex]
        if not ex_lines:
            ex_lines = ["(no examples)"]
        lines.extend([
            f"### {i}. **{lemma}** ({pos})",
            "",
            f"- **EN:** {def_en}",
            f"- **RU:** {def_ru}",
            "- *Examples:*",
        ])
        for j, ex in enumerate(ex_lines, 1):
            lines.append(f"  {j}. {ex}")
        lines.append("")

    # Append problems file if present (manual review findings)
    problems_path = args.output.parent / "quality_problems_B_batch.md"
    if problems_path.exists():
        lines.extend([
            "---",
            "",
            "## Problems found (manual review)",
            "",
            f"See **{problems_path.name}** for the list of issues found in the sample above.",
            "",
        ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {args.output}")
    print(f"  Cards: {cards_count:,}, Lemmas: {lemmas_count:,}, Avg cards/lemma: {avg:.2f}")
    print(f"  Sample: {len(sample_cards)} cards for manual review.")


if __name__ == "__main__":
    main()
