"""Stage 1: Extract meanings from examples.

Identifies all distinct meanings of a lemma, including:
- Different parts of speech
- Phrasal verbs (as separate meanings)
- Spoiler detection (but not filtering)
"""

import json
import logging
from typing import Any

from eng_words.llm.base import LLMProvider
from eng_words.llm.response_cache import ResponseCache

from .data_models import ExtractedMeaning, ExtractionResult, SourceExample

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are analyzing word usage to identify distinct meanings.

LEMMA: {lemma}

EXAMPLES FROM BOOK (numbered):
{numbered_examples}

TASK:
Identify ALL distinct meanings of "{lemma}" found in these examples.

CRITICAL RULES:

1. PHRASAL VERBS — only TRUE idiomatic ones:
   - TRUE phrasal verbs have meanings you CANNOT guess from parts:
     ✅ "give up" (surrender) — not predictable from "give" + "up"
     ✅ "look after" (care for) — idiomatic meaning
     ✅ "go on" (continue) — idiomatic meaning
   - NOT phrasal verbs (just verb + preposition/direction):
     ❌ "look at" — just looking in a direction
     ❌ "go to" — just movement to place
     ❌ "look up" (raise eyes) — literal meaning
   - Set is_phrasal=true ONLY for true idioms

2. SPOILERS - identify but DON'T exclude:
   - Mark has_spoiler=true if example reveals plot (deaths, crimes, twists)
   - Still use the example to identify the meaning

3. DIFFERENT POS = SEPARATE meanings:
   - Noun "run" ≠ Verb "run"

4. PREFER FEWER MEANINGS:
   - Most words have 2-4 core meanings, not 10+
   - Group similar usages under ONE meaning
   - "reach a place" and "reach a goal" = ONE meaning (arrive at/achieve)

5. DO NOT over-split:
   - Same meaning with different objects = ONE meaning
   - Slight nuances = ONE meaning
   - Ask: "Would a learner need different translations?" If not, merge.

OUTPUT (strict JSON, no markdown):
{{
  "meanings": [
    {{
      "meaning_id": 1,
      "definition_en": "clear, concise definition",
      "part_of_speech": "verb",
      "is_phrasal": false,
      "phrasal_form": null,
      "source_examples": [
        {{"index": 1, "has_spoiler": false}},
        {{"index": 5, "has_spoiler": true, "spoiler_type": "character death"}}
      ]
    }},
    {{
      "meaning_id": 2,
      "definition_en": "to leave a place",
      "part_of_speech": "phrasal verb",
      "is_phrasal": true,
      "phrasal_form": "go away",
      "source_examples": [
        {{"index": 3, "has_spoiler": false}}
      ]
    }}
  ]
}}
"""


class MeaningExtractor:
    """Stage 1: Extract meanings from examples.

    Args:
        provider: LLM provider for API calls
        cache: Response cache (optional)
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: ResponseCache | None = None,
    ):
        self.provider = provider
        self.cache = cache

        # Stats
        self.total_api_calls = 0
        self.cache_hits = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def extract(
        self,
        lemma: str,
        examples: list[str],
        sentence_ids: list[int],
    ) -> ExtractionResult:
        """Extract meanings from examples.

        Args:
            lemma: The word to analyze
            examples: List of example sentences
            sentence_ids: Corresponding sentence IDs

        Returns:
            ExtractionResult with all identified meanings
        """
        if not examples:
            return ExtractionResult(
                lemma=lemma,
                all_sentence_ids=[],
                meanings=[],
            )

        # Build numbered examples
        numbered = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples))

        prompt = EXTRACTION_PROMPT.format(
            lemma=lemma,
            numbered_examples=numbered,
        )

        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                self.provider.model, prompt, self.provider.temperature
            )
            cached = self.cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                return self._parse_response(lemma, cached.content, examples, sentence_ids)

        # Call LLM
        response = self.provider.complete(prompt)
        self.total_api_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost += response.cost_usd

        # Cache response
        if self.cache:
            self.cache.set(cache_key, response)

        return self._parse_response(
            lemma,
            response.content,
            examples,
            sentence_ids,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

    def _parse_response(
        self,
        lemma: str,
        content: str,
        examples: list[str],
        sentence_ids: list[int],
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> ExtractionResult:
        """Parse LLM response into ExtractionResult."""
        try:
            # Clean up response
            text = content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            data = json.loads(text.strip())

            meanings = []
            for m in data.get("meanings", []):
                source_examples = []
                for se in m.get("source_examples", []):
                    idx = se.get("index", 0)
                    # Convert 1-based index to sentence_id
                    sid = sentence_ids[idx - 1] if 0 < idx <= len(sentence_ids) else -1
                    source_examples.append(
                        SourceExample(
                            index=idx,
                            sentence_id=sid,
                            has_spoiler=se.get("has_spoiler", False),
                            spoiler_type=se.get("spoiler_type"),
                        )
                    )

                meanings.append(
                    ExtractedMeaning(
                        meaning_id=m.get("meaning_id", len(meanings) + 1),
                        definition_en=m.get("definition_en", ""),
                        part_of_speech=m.get("part_of_speech", "unknown"),
                        is_phrasal=m.get("is_phrasal", False),
                        phrasal_form=m.get("phrasal_form"),
                        source_examples=source_examples,
                    )
                )

            return ExtractionResult(
                lemma=lemma,
                all_sentence_ids=list(sentence_ids),
                meanings=meanings,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse extraction response for '{lemma}': {e}")
            logger.debug(f"Response content: {content[:500]}")

            # Return empty result on parse error
            return ExtractionResult(
                lemma=lemma,
                all_sentence_ids=list(sentence_ids),
                meanings=[],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

    def stats(self) -> dict[str, Any]:
        """Return extraction statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "cache_hits": self.cache_hits,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
        }
