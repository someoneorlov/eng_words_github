#!/usr/bin/env python3
"""Check 1-based vs 0-based hypothesis using stored batch results.

Uses data/experiment/batch_b/results.jsonl and lemma_examples.json to see:
- How many cards would have 0 examples with strict 1-based parsing
- How many of those would have >0 examples if we treat indices as 0-based

If the second number is significant, that supports the hypothesis that some
"no examples" were due to the LLM returning 0-based indices.

Usage (from project root):
    uv run python scripts/check_index_basis_hypothesis.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = PROJECT_ROOT / "data" / "experiment" / "batch_b"
RESULTS_PATH = BATCH_DIR / "results.jsonl"
LEMMA_EXAMPLES_PATH = BATCH_DIR / "lemma_examples.json"


def _extract_text(response: dict) -> str | None:
    """Extract JSON text from Gemini response (same logic as run_pipeline_b_batch)."""
    if "candidates" not in response or not response["candidates"]:
        return None
    parts = response["candidates"][0].get("content", {}).get("parts", [])
    if not parts:
        return None
    text = parts[0].get("text", "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _count_with_1based(indices: list[int], n_examples: int) -> int:
    """Number of valid examples if we interpret indices as 1-based."""
    return sum(1 for i in indices if isinstance(i, int) and 0 < i <= n_examples)


def _count_with_0based(indices: list[int], n_examples: int) -> int:
    """Number of valid examples if we interpret indices as 0-based (when 0 present)."""
    raw = [i for i in indices if isinstance(i, int)]
    if not raw or 0 not in raw or max(raw) >= n_examples:
        return 0
    return sum(1 for i in raw if 0 <= i < n_examples)


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Missing {RESULTS_PATH}. Run batch download first.")
        sys.exit(1)
    if not LEMMA_EXAMPLES_PATH.exists():
        print(f"Missing {LEMMA_EXAMPLES_PATH}. Re-run 'create' for the batch.")
        sys.exit(1)

    with open(LEMMA_EXAMPLES_PATH, encoding="utf-8") as f:
        lemma_examples = json.load(f)

    # Cards that would be empty with 1-based but non-empty with 0-based
    fixed_by_0based: list[dict] = []
    # Cards that are empty with both (truly empty / invalid indices)
    empty_both: list[dict] = []
    total_cards = 0
    empty_n1: list[dict] = []
    empty_n_gt1: list[dict] = []

    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key", "")
            resp = row.get("response", {})
            if not key.startswith("lemma:"):
                continue
            lemma = key[6:]
            text = _extract_text(resp)
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            examples = lemma_examples.get(lemma, [])
            n_ex = len(examples)
            for card in data.get("cards", []):
                if not isinstance(card, dict):
                    continue
                total_cards += 1
                raw = card.get("selected_example_indices", [])
                indices = [x for x in raw if isinstance(x, int)]
                n1 = _count_with_1based(indices, n_ex)
                n0 = _count_with_0based(indices, n_ex)
                if n1 == 0 and n0 > 0:
                    fixed_by_0based.append({
                        "lemma": lemma,
                        "selected_example_indices": indices,
                        "n_examples_available": n_ex,
                        "definition_en": (card.get("definition_en") or "")[:60],
                    })
                elif n1 == 0 and n0 == 0 and (indices or raw):
                    empty_both.append({
                        "lemma": lemma,
                        "selected_example_indices": indices or raw,
                        "n_examples_available": n_ex,
                    })

    empty_n1 = [e for e in empty_both if e["n_examples_available"] == 1]
    empty_n_gt1 = [e for e in empty_both if e["n_examples_available"] > 1]

    print("=== Index basis hypothesis (1-based vs 0-based) ===\n")
    print(f"Total cards in results: {total_cards}")
    print(f"Cards that would have 0 examples with 1-based but >0 with 0-based: {len(fixed_by_0based)}")
    print(f"Cards empty with both (invalid/out-of-range indices): {len(empty_both)}")
    if total_cards:
        pct = 100.0 * len(fixed_by_0based) / total_cards
        print(f"\nShare of 'fixed by 0-based' among all cards: {pct:.1f}%")
    if fixed_by_0based:
        print("\n--- Lemmas/cards that would be fixed by 0-based (first 30) ---")
        for e in fixed_by_0based[:30]:
            print(f"  {e['lemma']}: indices={e['selected_example_indices']}, n_available={e['n_examples_available']}")
        if len(fixed_by_0based) > 30:
            print(f"  ... and {len(fixed_by_0based) - 30} more")
    if empty_both:
        print("\n--- Sample of cards empty with both (first 15) ---")
        for e in empty_both[:15]:
            print(f"  {e['lemma']}: indices={e['selected_example_indices']}, n_available={e['n_examples_available']}")

    print("\n=== If we remove 'first example' fallback entirely ===")
    print(f"Cards with 0 examples (invalid indices): {len(empty_both)}")
    print(f"  — with 1 example only: {len(empty_n1)} (would get first-example fallback only)")
    print(f"  — with >1 examples: {len(empty_n_gt1)} (would need retry e.g. Thinking model)")
    if empty_n_gt1:
        print("\nCards that would need Thinking retry (n>1, 0 examples):")
        for e in empty_n_gt1:
            print(f"  {e['lemma']}: n_available={e['n_examples_available']}, indices={e['selected_example_indices']}")

    out = BATCH_DIR / "index_basis_report.json"
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_cards": total_cards,
                "fixed_by_0based_count": len(fixed_by_0based),
                "empty_both_count": len(empty_both),
                "fixed_by_0based_sample": fixed_by_0based[:50],
                "empty_both_sample": empty_both[:30],
                "empty_both_n1_count": len(empty_n1),
                "empty_both_n_gt1_count": len(empty_n_gt1),
                "empty_both_n_gt1": empty_n_gt1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
