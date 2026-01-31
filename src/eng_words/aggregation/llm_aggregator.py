"""
LLM-based synset aggregation.

Uses LLM to intelligently group similar synsets for language learning.
"""

import json
import logging
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.response_cache import ResponseCache

logger = logging.getLogger(__name__)


@dataclass
class SynsetGroup:
    """A group of synsets that should be merged into one card."""

    synset_ids: list[str]
    primary_synset: str  # The synset whose definition will be used
    reason: str


@dataclass
class AggregationResult:
    """Result of LLM aggregation for one lemma."""

    lemma: str
    groups: list[SynsetGroup]
    original_count: int
    aggregated_count: int
    llm_cost: float


def build_aggregation_prompt(lemma: str, synsets: list[dict]) -> str:
    """
    Build the prompt for LLM aggregation.

    Args:
        lemma: The word lemma
        synsets: List of dicts with synset_id, definition, freq

    Returns:
        Prompt string for LLM
    """
    synsets_numbered = "\n".join(
        [
            f"{i+1}. {s['synset_id']} (occurs {s['freq']}x)\n"
            f"   Definition: {s['definition']}"
            for i, s in enumerate(synsets)
        ]
    )

    json_schema = """{
  "groups": [
    {
      "synsets": [1, 3],
      "primary_synset": 1,
      "reason": "why we merge these"
    },
    {
      "synsets": [2],
      "primary_synset": 2,
      "reason": "why kept separate"
    }
  ],
  "total_cards": 2
}"""

    return f"""You are helping create Anki flashcards for English learning (B1-B2 level).

## Task
For the word "{lemma}" there are several senses (synsets) from WordNet.
Decide which senses should be MERGED into one card and which should stay separate.

## Merge criteria:
- Senses are very similar and differ only in nuance
- For a language learner the difference is negligible
- Better to learn as one concept

## Split criteria:
- Senses are fundamentally different (different concepts)
- Important to understand the difference for correct usage
- Different contexts (formal/informal, technical/everyday)
- Different parts of speech (verb vs noun)

## Senses for "{lemma}":
{synsets_numbered}

## Response format (JSON):
{json_schema}

Reply with ONLY JSON, no explanations.
"""


def parse_aggregation_response(response_text: str) -> dict | None:
    """
    Parse the LLM response to extract aggregation groups.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Truncated JSON (attempts repair)
    - Extra text around JSON
    """
    try:
        text = response_text.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
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
        repaired = _repair_truncated_json(text)
        if repaired:
            return json.loads(repaired)

        return None

    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        return None


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair truncated JSON by adding missing brackets."""
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")

    if open_braces > close_braces or open_brackets > close_brackets:
        # Remove incomplete last field
        lines = text.rstrip().split("\n")

        while lines:
            last_line = lines[-1].strip()
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
                lines[-1] = lines[-1].rstrip().rstrip(",")
                break
            else:
                break

        text = "\n".join(lines)

        # Add missing brackets/braces
        missing_brackets = "]" * (open_brackets - text.count("]"))
        missing_braces = "}" * (open_braces - text.count("}"))

        if '"total_cards"' not in text:
            text = text.rstrip().rstrip(",") + f'\n  ],\n  "total_cards": 0\n}}'
        else:
            text = text + missing_brackets + missing_braces

        try:
            json.loads(text)
            logger.info("Successfully repaired truncated JSON")
            return text
        except json.JSONDecodeError:
            return None

    return None


class LLMAggregator:
    """
    LLM-based synset aggregator.

    Uses LLM to intelligently group similar synsets for language learning,
    reducing the number of cards while preserving important distinctions.
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: ResponseCache,
        max_retries: int = 2,
    ):
        """
        Initialize the aggregator.

        Args:
            provider: LLM provider to use
            cache: Response cache for storing results
            max_retries: Maximum retry attempts for failed LLM calls
        """
        self.provider = provider
        self.cache = cache
        self.max_retries = max_retries
        self._stats = {
            "total_lemmas": 0,
            "total_synsets": 0,
            "total_groups": 0,
            "total_cost": 0.0,
            "cache_hits": 0,
        }

    def aggregate_lemma(
        self,
        lemma: str,
        synsets: list[dict],
    ) -> AggregationResult:
        """
        Aggregate synsets for a single lemma.

        For lemmas with only 1 synset, returns immediately without LLM call.
        For lemmas with 2+ synsets, uses LLM to determine groupings.

        Args:
            lemma: The word lemma
            synsets: List of dicts with synset_id, definition, freq

        Returns:
            AggregationResult with groups
        """
        self._stats["total_lemmas"] += 1
        self._stats["total_synsets"] += len(synsets)

        # Single synset - no aggregation needed
        if len(synsets) <= 1:
            groups = [
                SynsetGroup(
                    synset_ids=[synsets[0]["synset_id"]] if synsets else [],
                    primary_synset=synsets[0]["synset_id"] if synsets else "",
                    reason="Single synset",
                )
            ] if synsets else []

            self._stats["total_groups"] += len(groups)

            return AggregationResult(
                lemma=lemma,
                groups=groups,
                original_count=len(synsets),
                aggregated_count=len(groups),
                llm_cost=0.0,
            )

        # Multiple synsets - use LLM
        prompt = build_aggregation_prompt(lemma, synsets)

        # Calculate max tokens based on synset count
        estimated_tokens = len(synsets) * 80 + 200
        max_output = max(2048, estimated_tokens)

        # Check cache
        cache_key = self.cache.generate_key(
            self.provider.model, prompt, self.provider.temperature
        )
        cached = self.cache.get(cache_key)

        if cached:
            self._stats["cache_hits"] += 1
            response = cached
        else:
            response = self._call_llm_with_retry(prompt, max_output)
            if response:
                self.cache.set(cache_key, response)

        if not response:
            # Fallback: each synset as separate group
            groups = [
                SynsetGroup(
                    synset_ids=[s["synset_id"]],
                    primary_synset=s["synset_id"],
                    reason="LLM failed - fallback",
                )
                for s in synsets
            ]
            self._stats["total_groups"] += len(groups)

            return AggregationResult(
                lemma=lemma,
                groups=groups,
                original_count=len(synsets),
                aggregated_count=len(synsets),
                llm_cost=0.0,
            )

        # Parse response
        parsed = parse_aggregation_response(response.content)

        if not parsed:
            # Fallback on parse failure
            groups = [
                SynsetGroup(
                    synset_ids=[s["synset_id"]],
                    primary_synset=s["synset_id"],
                    reason="Parse failed - fallback",
                )
                for s in synsets
            ]
        else:
            groups = []
            for g in parsed.get("groups", []):
                # Convert indices to synset_ids
                # Handle both int and str indices from LLM
                raw_indices = g.get("synsets", [])
                indices = []
                for idx in raw_indices:
                    try:
                        indices.append(int(idx))
                    except (ValueError, TypeError):
                        continue

                synset_ids = [
                    synsets[idx - 1]["synset_id"]
                    for idx in indices
                    if 0 < idx <= len(synsets)
                ]

                # Handle primary_synset
                raw_primary = g.get("primary_synset", indices[0] if indices else 1)
                try:
                    primary_idx = int(raw_primary)
                except (ValueError, TypeError):
                    primary_idx = indices[0] if indices else 1

                primary_synset = (
                    synsets[primary_idx - 1]["synset_id"]
                    if 0 < primary_idx <= len(synsets)
                    else synset_ids[0] if synset_ids else ""
                )

                if synset_ids:
                    groups.append(
                        SynsetGroup(
                            synset_ids=synset_ids,
                            primary_synset=primary_synset,
                            reason=g.get("reason", ""),
                        )
                    )

        self._stats["total_groups"] += len(groups)
        self._stats["total_cost"] += response.cost_usd

        return AggregationResult(
            lemma=lemma,
            groups=groups,
            original_count=len(synsets),
            aggregated_count=len(groups),
            llm_cost=response.cost_usd,
        )

    def _call_llm_with_retry(
        self, prompt: str, max_output_tokens: int
    ) -> LLMResponse | None:
        """Call LLM with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.provider.complete(
                    prompt, max_output_tokens=max_output_tokens
                )
                return response
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

        return None

    def aggregate_batch(
        self,
        synset_stats_df: pd.DataFrame,
        progress: bool = True,
    ) -> list[AggregationResult]:
        """
        Aggregate synsets for all lemmas in the DataFrame.

        Args:
            synset_stats_df: DataFrame with lemma, synset_id, definition, freq
            progress: Show progress bar

        Returns:
            List of AggregationResult for each lemma
        """
        # Group by lemma
        grouped = synset_stats_df.groupby("lemma")

        results = []
        iterator = tqdm(grouped, desc="Aggregating synsets") if progress else grouped

        for lemma, group in iterator:
            synsets = [
                {
                    "synset_id": row["synset_id"],
                    "definition": row["definition"],
                    "freq": row["freq"],
                }
                for _, row in group.iterrows()
            ]

            result = self.aggregate_lemma(lemma, synsets)
            results.append(result)

        return results

    def apply_aggregation(
        self,
        synset_stats_df: pd.DataFrame,
        aggregation_results: list[AggregationResult],
    ) -> pd.DataFrame:
        """
        Apply aggregation results to create final card items.

        Each group becomes one card item, with examples from all synsets
        in the group and the definition from the primary synset.

        Args:
            synset_stats_df: Original DataFrame with synset stats
            aggregation_results: Results from aggregate_batch

        Returns:
            DataFrame with one row per group (card)
        """
        # Create lookup for results
        results_by_lemma = {r.lemma: r for r in aggregation_results}

        # Create lookup for synset data
        synset_lookup = {}
        for _, row in synset_stats_df.iterrows():
            synset_lookup[row["synset_id"]] = row.to_dict()

        cards = []
        for lemma, result in results_by_lemma.items():
            for group in result.groups:
                # Get primary synset data
                primary_data = synset_lookup.get(group.primary_synset, {})

                # Collect all sentence_ids from all synsets in the group
                all_sentence_ids = []
                total_freq = 0
                for synset_id in group.synset_ids:
                    synset_data = synset_lookup.get(synset_id, {})
                    sentence_ids = synset_data.get("sentence_ids", [])
                    if isinstance(sentence_ids, list):
                        all_sentence_ids.extend(sentence_ids)
                    total_freq += synset_data.get("freq", 0)

                cards.append({
                    "lemma": lemma,
                    "synset_group": group.synset_ids,
                    "primary_synset": group.primary_synset,
                    "pos": primary_data.get("pos", ""),
                    "definition": primary_data.get("definition", ""),
                    "supersense": primary_data.get("supersense", ""),
                    "freq": total_freq,
                    "sentence_ids": all_sentence_ids,
                    "group_reason": group.reason,
                })

        return pd.DataFrame(cards)

    def get_stats(self) -> dict:
        """Get aggregation statistics."""
        return {
            "total_lemmas": self._stats["total_lemmas"],
            "total_synsets": self._stats["total_synsets"],
            "total_groups": self._stats["total_groups"],
            "total_cost": self._stats["total_cost"],
            "cache_hits": self._stats["cache_hits"],
            "reduction": (
                f"{(1 - self._stats['total_groups'] / max(self._stats['total_synsets'], 1)) * 100:.1f}%"
            ),
        }

