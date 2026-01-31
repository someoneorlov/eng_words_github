#!/usr/bin/env python3
"""
Test LLM-based synset aggregation on the sample of complex lemmas.

Usage:
    uv run python scripts/test_synset_aggregation.py
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
SAMPLE_PATH = Path("data/synset_aggregation_test/sample_20.jsonl")
OUTPUT_DIR = Path("data/synset_aggregation_test")
CACHE_DIR = OUTPUT_DIR / "llm_cache"


@dataclass
class SynsetGroup:
    """A group of synsets to be merged into one card."""

    synsets: list[int]  # 1-based indices
    primary_synset: int  # 1-based index
    reason: str


@dataclass
class AggregationResult:
    """Result of LLM aggregation for one lemma."""

    lemma: str
    original_count: int
    aggregated_count: int
    groups: list[SynsetGroup]
    llm_cost: float
    llm_tokens: int


def build_aggregation_prompt(lemma: str, synsets: list[dict]) -> str:
    """Build the prompt for LLM aggregation."""
    synsets_numbered = "\n".join(
        [
            f"{i+1}. {s['synset_id']} (встречается {s['freq']}x)\n"
            f"   Определение: {s['definition']}"
            for i, s in enumerate(synsets)
        ]
    )

    json_schema = """{
  "groups": [
    {
      "synsets": [1, 3],
      "primary_synset": 1,
      "reason": "почему объединяем"
    },
    {
      "synsets": [2],
      "primary_synset": 2,
      "reason": "почему отдельно"
    }
  ],
  "total_cards": 3
}"""

    return f"""Ты помогаешь создавать Anki-карточки для изучения английского (уровень B1-B2).

## Задача
Для слова "{lemma}" есть несколько значений (synsets) из WordNet.
Реши, какие значения стоит ОБЪЕДИНИТЬ в одну карточку, а какие оставить отдельными.

## Критерии объединения:
- Значения очень похожи и различаются только нюансами
- Для изучающего язык разница несущественна
- Лучше учить как одно понятие

## Критерии разделения:
- Значения принципиально разные (разные концепции)
- Важно понимать разницу для правильного использования
- Разные контексты использования (formal/informal, technical/everyday)
- Разные части речи (глагол vs существительное)

## Значения слова "{lemma}":
{synsets_numbered}

## Формат ответа (JSON):
{json_schema}

Ответь ТОЛЬКО JSON без пояснений.
"""


def parse_aggregation_response(response_text: str) -> dict | None:
    """Parse the LLM response to extract aggregation groups.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Truncated JSON (attempts repair)
    - Extra text around JSON
    """
    try:
        # Clean up response - remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Find the closing ```
            end_idx = len(lines) - 1
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            text = "\n".join(lines[1:end_idx])

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Try to repair truncated JSON
        repaired = repair_truncated_json(text)
        if repaired:
            return json.loads(repaired)

        return None

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Response: {response_text[:500]}...")
        return None


def repair_truncated_json(text: str) -> str | None:
    """Attempt to repair truncated JSON by adding missing brackets."""
    # Count brackets
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")

    # If truncated, try to close it
    if open_braces > close_braces or open_brackets > close_brackets:
        # Remove incomplete last field
        lines = text.rstrip().split("\n")

        # Find and remove incomplete lines
        while lines:
            last_line = lines[-1].strip()
            # Check if line is incomplete (no comma or closing bracket)
            if (
                last_line
                and not last_line.endswith(",")
                and not last_line.endswith("}")
                and not last_line.endswith("]")
                and not last_line.endswith("{")
                and not last_line.endswith("[")
            ):
                lines.pop()
            elif last_line.endswith(","):
                # Remove trailing comma and close
                lines[-1] = lines[-1].rstrip().rstrip(",")
                break
            else:
                break

        text = "\n".join(lines)

        # Add missing brackets/braces
        missing_brackets = "]" * (open_brackets - text.count("]"))
        missing_braces = "}" * (open_braces - text.count("}"))

        # Add total_cards if missing
        if '"total_cards"' not in text:
            text = text.rstrip().rstrip(",") + '\n  ],\n  "total_cards": 0\n}'
        else:
            text = text + missing_brackets + missing_braces

        try:
            json.loads(text)
            logger.info("Successfully repaired truncated JSON")
            return text
        except json.JSONDecodeError:
            logger.warning("JSON repair failed")
            return None

    return None


def test_aggregation_on_lemma(
    lemma: str,
    synsets: list[dict],
    provider,
    cache: ResponseCache,
) -> AggregationResult | None:
    """Test LLM aggregation on a single lemma."""
    prompt = build_aggregation_prompt(lemma, synsets)

    # Generate cache key
    cache_key = cache.generate_key(provider.model, prompt, provider.temperature)

    # Calculate appropriate max_output_tokens based on synset count
    # ~70 tokens per synset group, plus overhead
    estimated_tokens = len(synsets) * 80 + 200
    max_output = max(2048, estimated_tokens)  # Minimum 2048, scale with synsets

    # Try cache first
    cached_response = cache.get(cache_key)
    if cached_response:
        response = cached_response
    else:
        response = provider.complete(prompt, max_output_tokens=max_output)
        cache.set(cache_key, response)

    parsed = parse_aggregation_response(response.content)
    if not parsed:
        return None

    groups = []
    for g in parsed.get("groups", []):
        groups.append(
            SynsetGroup(
                synsets=g.get("synsets", []),
                primary_synset=g.get("primary_synset", g.get("synsets", [1])[0]),
                reason=g.get("reason", ""),
            )
        )

    return AggregationResult(
        lemma=lemma,
        original_count=len(synsets),
        aggregated_count=parsed.get("total_cards", len(groups)),
        groups=groups,
        llm_cost=response.cost_usd,
        llm_tokens=response.input_tokens + response.output_tokens,
    )


def run_test(sample_path: Path = SAMPLE_PATH, limit: int | None = None):
    """Run aggregation test on the sample."""
    logger.info(f"Loading sample from {sample_path}")

    # Load sample
    sample_data = []
    with open(sample_path, encoding="utf-8") as f:
        for line in f:
            sample_data.append(json.loads(line))

    if limit:
        sample_data = sample_data[:limit]

    logger.info(f"Testing on {len(sample_data)} lemmas")

    # Initialize provider and cache
    provider = get_provider("gemini", "gemini-3-flash-preview")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)

    # Run tests
    results = []
    total_cost = 0.0
    total_original = 0
    total_aggregated = 0

    for item in sample_data:
        lemma = item["lemma"]
        synsets = item["synsets"]

        logger.info(f"Testing '{lemma}' ({len(synsets)} synsets)...")

        result = test_aggregation_on_lemma(lemma, synsets, provider, cache)

        if result:
            results.append(result)
            total_cost += result.llm_cost
            total_original += result.original_count
            total_aggregated += result.aggregated_count

            print(f"\n{'='*70}")
            print(f"LEMMA: {lemma}")
            print(f"{'='*70}")
            print(f"Synsets: {result.original_count} → {result.aggregated_count} cards")
            print(f"Reduction: {(1 - result.aggregated_count/result.original_count)*100:.0f}%")
            print(f"Cost: ${result.llm_cost:.4f}")

            for i, group in enumerate(result.groups[:5]):
                synset_ids = [
                    synsets[idx - 1]["synset_id"] for idx in group.synsets if idx <= len(synsets)
                ]
                print(f"\n  Group {i+1}: {synset_ids}")
                print(f"  Reason: {group.reason[:80]}...")

            if len(result.groups) > 5:
                print(f"\n  ... and {len(result.groups) - 5} more groups")
        else:
            logger.error(f"Failed to aggregate '{lemma}'")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Lemmas tested:      {len(results)}")
    print(f"Total synsets:      {total_original}")
    print(f"Total cards:        {total_aggregated}")
    print(f"Reduction:          {(1 - total_aggregated/total_original)*100:.1f}%")
    print(f"Total cost:         ${total_cost:.4f}")
    print(f"Avg cost per lemma: ${total_cost/len(results):.4f}")
    print(f"Cache stats:        {cache.stats()}")

    # Save results
    output_path = OUTPUT_DIR / "aggregation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "lemmas_tested": len(results),
                    "total_synsets": total_original,
                    "total_cards": total_aggregated,
                    "reduction_pct": round((1 - total_aggregated / total_original) * 100, 1),
                    "total_cost": round(total_cost, 4),
                },
                "results": [asdict(r) for r in results],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_test()
