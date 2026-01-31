"""Word Family clustering using LLM.

This module implements Pipeline B for the experiment:
- Groups all examples of a lemma
- Uses LLM to cluster them into distinct semantic groups (word families)
- Generates Anki cards for each group
"""

import json
import logging
from dataclasses import dataclass

import pandas as pd

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.response_cache import ResponseCache

logger = logging.getLogger(__name__)

# Constants
MAX_EXAMPLES_PER_BATCH = 100
SIMILARITY_THRESHOLD = 0.85


# Prompt template for clustering (without WordNet hints - Pipeline B)
# V1: Original prompt (tends to oversplit)
CLUSTER_PROMPT_TEMPLATE_V1 = """You are creating Anki flashcards for English learners (B1-B2 level).

LEMMA: {lemma}

EXAMPLES FROM BOOK "American Tragedy":
{numbered_examples}

TASK:
1. Analyze all examples and identify DISTINCT MEANINGS of "{lemma}"
2. For each meaning, create a flashcard

RULES:
- If all examples have the SAME meaning → create 1 card
- If there are 2-4 DISTINCT meanings → create separate cards for each
- IGNORE examples where:
  * Sentence is incomplete or unclear
  * Contains major plot spoilers (death, crime revelation, etc.)
  * Word is used in an unusual or idiomatic way that's hard to explain
- Each example can belong to ONLY ONE card
- Select 2-3 BEST examples per card (10-30 words each, clear context)
- Generate 1 additional simple example per card (not from the book)

OUTPUT (strict JSON):
{{
  "cards": [
    {{
      "meaning_id": 1,
      "definition_en": "clear, concise definition in English",
      "definition_ru": "translation of definition",
      "part_of_speech": "noun/verb/adj/adv",
      "selected_example_indices": [1, 5, 12],
      "generated_example": "Simple example sentence using the word."
    }}
  ],
  "ignored_indices": [3, 7],
  "ignore_reasons": {{"3": "plot spoiler", "7": "unclear context"}}
}}

IMPORTANT:
- Return ONLY valid JSON, no markdown
- Use example indices as shown (1-based)
- Definition should be learner-friendly (B1-B2 level)
"""

# V2: Too aggressive - oversplits but also collapses correct splits
CLUSTER_PROMPT_TEMPLATE_V2 = """You are creating Anki flashcards for English learners (B1-B2 level).

LEMMA: {lemma}

EXAMPLES FROM BOOK "American Tragedy":
{numbered_examples}

TASK:
Analyze examples and group them by CORE MEANING. Create ONE card per truly distinct meaning.

CRITICAL RULES FOR GROUPING:
1. PREFER FEWER CARDS: Most words have 1-2 core meanings. Only split if meanings are FUNDAMENTALLY different.
2. SAME PART OF SPEECH = likely same meaning (group together unless very different)
3. DO NOT create separate cards for:
   - Slight nuances of the same meaning (e.g., "arrive at place" vs "arrive at destination")
   - Phrasal verbs with similar core meaning (e.g., "go away" and "go out" = both "leave")
   - Figurative vs literal uses of same concept
   - Different grammatical constructions with same meaning
4. DO create separate cards for:
   - Different parts of speech with different meanings (noun vs verb)
   - Completely unrelated meanings (e.g., "bank" = financial institution vs riverbank)

WHEN IN DOUBT: Combine into fewer cards. A learner benefits more from one solid card than multiple confusing ones.

IGNORE examples where:
- Sentence is incomplete or unclear
- Contains major plot spoilers
- Word is in a fixed phrase that doesn't teach the core meaning

OUTPUT (strict JSON):
{{
  "cards": [
    {{
      "meaning_id": 1,
      "definition_en": "clear definition covering the core meaning",
      "definition_ru": "translation",
      "part_of_speech": "noun/verb/adj/adv",
      "selected_example_indices": [1, 5, 12],
      "generated_example": "Simple example sentence."
    }}
  ],
  "ignored_indices": [3, 7],
  "ignore_reasons": {{"3": "plot spoiler", "7": "unclear"}}
}}

Target: 1-3 cards for most words. Only common polysemous words (get, take, run) may need 4-5.
"""

# V3: Balanced - but still oversplits
CLUSTER_PROMPT_TEMPLATE_V3 = """You are creating Anki flashcards for English learners (B1-B2 level).

LEMMA: {lemma}

EXAMPLES FROM BOOK "American Tragedy":
{numbered_examples}

TASK:
Group examples by meaning and create flashcards. Most words need 1-3 cards.

WHEN TO CREATE SEPARATE CARDS (DO split):
- Different parts of speech with different meanings (noun "care" vs verb "care")
- Truly unrelated meanings (e.g., "story" = tale vs floor of building)
- Meanings that would confuse a learner if combined
- Transitive vs intransitive with different meanings

WHEN TO COMBINE INTO ONE CARD (DON'T split):
- Same core meaning with slight variations (e.g., "reach a place" and "reach a goal" = same core "arrive at")
- Same meaning used literally vs figuratively
- Phrasal verbs that share the same base meaning
- Variations that a B1-B2 learner would understand as "the same word"

EXAMPLES OF GOOD vs BAD SPLITTING:
- GOOD: "mean" → verb (signify), verb (intend), adjective (unkind), noun (method) = 4 cards ✓
- BAD: "reach" → "arrive at place" + "arrive at destination" + "achieve level" = should be 2 cards max ✗
- GOOD: "care" → verb (feel concern) + noun (attention/protection) = 2 cards ✓
- BAD: "opposition" → "resistance" + "disagreement" = same meaning, 1 card ✗

IGNORE examples where:
- Sentence is incomplete or unclear
- Contains major plot spoilers

OUTPUT (strict JSON):
{{
  "cards": [
    {{
      "meaning_id": 1,
      "definition_en": "clear, concise definition",
      "definition_ru": "translation",
      "part_of_speech": "noun/verb/adj/adv",
      "selected_example_indices": [1, 5, 12],
      "generated_example": "Simple example sentence."
    }}
  ],
  "ignored_indices": [3, 7],
  "ignore_reasons": {{"3": "spoiler", "7": "unclear"}}
}}
"""

# V4: Final - strict on oversplit, but always split different POS
CLUSTER_PROMPT_TEMPLATE = """You are creating Anki flashcards for English learners (B1-B2 level).

LEMMA: {lemma}

EXAMPLES FROM BOOK "American Tragedy":
{numbered_examples}

TASK:
Create 1-3 flashcards grouping examples by CORE MEANING.

MANDATORY SPLITS (always create separate cards):
1. Different parts of speech (noun vs verb vs adjective)
2. Completely unrelated meanings (e.g., "bank" = money vs river)
3. Different core actions/concepts that would confuse if mixed

MANDATORY MERGES (always combine into one card):
1. Same meaning with different objects ("reach a place" = "reach a goal" = "reach a conclusion")
2. Literal vs figurative uses of same concept
3. Active vs passive forms
4. Similar nuances that B1-B2 learner sees as "same word"

TEST: Before creating 2 cards for same POS, ask: "Would a learner be confused if these were in ONE card?"
- If NO → merge them
- If YES → split them

IGNORE: incomplete sentences, major plot spoilers

OUTPUT (JSON):
{{
  "cards": [
    {{
      "meaning_id": 1,
      "definition_en": "clear definition",
      "definition_ru": "translation",
      "part_of_speech": "noun/verb/adj/adv",
      "selected_example_indices": [1, 5],
      "generated_example": "Simple example."
    }}
  ],
  "ignored_indices": [],
  "ignore_reasons": {{}}
}}

REMEMBER: 1-3 cards is normal. 4+ only for very common polysemous words (get, take, make).
"""


# Prompt template WITH WordNet hints (Pipeline C)
CLUSTER_PROMPT_WITH_HINTS_TEMPLATE = """You are creating Anki flashcards for English learners (B1-B2 level).

LEMMA: {lemma}

WORDNET REFERENCE (for guidance only):
{wordnet_hints}

About WordNet Reference:
- WordNet is a lexical database with standard dictionary definitions
- These definitions are provided as HINTS to help you identify distinct meanings
- DO NOT create cards for WordNet meanings that are NOT present in the examples
- DO create cards for meanings you find in examples even if they're NOT in WordNet
- If a meaning matches WordNet definition, you MAY reference its synset ID
- Your primary source of truth is the EXAMPLES, not WordNet

EXAMPLES FROM BOOK "American Tragedy":
{numbered_examples}

TASK:
1. Read all examples carefully and identify DISTINCT MEANINGS of "{lemma}" as used in this book
2. Use WordNet definitions only as reference to help distinguish between meanings
3. Create a flashcard for each distinct meaning you ACTUALLY FIND in the examples

RULES:
- If all examples have the SAME meaning → create 1 card
- If there are 2-4 DISTINCT meanings → create separate cards for each
- SKIP WordNet meanings that have NO examples in the book
- INCLUDE meanings found in examples even if WordNet doesn't list them (idioms, figurative uses)
- IGNORE examples where:
  * Sentence is incomplete or unclear
  * Contains major plot spoilers (death, crime revelation, etc.)
  * Word is used in an unusual or idiomatic way that's hard to explain
- Each example can belong to ONLY ONE card
- Select 2-3 BEST examples per card (10-30 words each, clear context)
- Generate 1 additional simple example per card (not from the book)

OUTPUT (strict JSON):
{{
  "cards": [
    {{
      "meaning_id": 1,
      "definition_en": "clear, concise definition in English",
      "definition_ru": "translation of definition",
      "part_of_speech": "noun/verb/adj/adv",
      "wordnet_synset": "matching synset ID if applicable, or null",
      "selected_example_indices": [1, 5, 12],
      "generated_example": "Simple example sentence using the word."
    }}
  ],
  "ignored_indices": [3, 7],
  "ignore_reasons": {{"3": "plot spoiler", "7": "unclear context"}}
}}

IMPORTANT:
- Return ONLY valid JSON, no markdown
- Use example indices as shown (1-based)
- Definition should be learner-friendly (B1-B2 level)
- Base your decisions on EXAMPLES, not on WordNet coverage
"""


@dataclass
class ClusterResult:
    """Result of clustering a single lemma."""

    lemma: str
    cards: list[dict]
    ignored_indices: list[int]
    ignore_reasons: dict[str, str]
    total_examples: int
    batches_processed: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class WordFamilyClusterer:
    """Clusters lemma examples into word families using LLM.

    Args:
        provider: LLM provider (e.g., GeminiProvider)
        cache: Response cache for avoiding duplicate API calls
        use_wordnet_hints: Whether to include WordNet definitions as hints (Pipeline C)
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: ResponseCache | None = None,
        use_wordnet_hints: bool = False,
    ):
        self.provider = provider
        self.cache = cache
        self.use_wordnet_hints = use_wordnet_hints

        # Statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.total_api_calls = 0
        self.cache_hits = 0

    def cluster_lemma(
        self,
        lemma: str,
        examples: list[str],
        sentence_ids: list[int] | None = None,
        wordnet_hints: str | None = None,
    ) -> ClusterResult:
        """Cluster examples for a single lemma into semantic groups.

        Args:
            lemma: The lemma to cluster
            examples: List of example sentences containing the lemma
            sentence_ids: Optional list of sentence IDs for tracking
            wordnet_hints: Optional WordNet definitions for Pipeline C

        Returns:
            ClusterResult with cards and metadata
        """
        if len(examples) <= MAX_EXAMPLES_PER_BATCH:
            # Single batch - process directly
            return self._cluster_single_batch(lemma, examples, sentence_ids, wordnet_hints)

        # Multi-pass for frequent lemmas
        return self._cluster_multi_pass(lemma, examples, sentence_ids, wordnet_hints)

    def _cluster_single_batch(
        self,
        lemma: str,
        examples: list[str],
        sentence_ids: list[int] | None,
        wordnet_hints: str | None,
    ) -> ClusterResult:
        """Process a single batch of examples."""
        # Format examples
        numbered_examples = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples))

        # Choose prompt template
        if self.use_wordnet_hints and wordnet_hints:
            prompt = CLUSTER_PROMPT_WITH_HINTS_TEMPLATE.format(
                lemma=lemma,
                numbered_examples=numbered_examples,
                wordnet_hints=wordnet_hints,
            )
        else:
            prompt = CLUSTER_PROMPT_TEMPLATE.format(
                lemma=lemma,
                numbered_examples=numbered_examples,
            )

        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                self.provider.model, prompt, self.provider.temperature
            )
            cached = self.cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                return self._parse_response(lemma, cached, examples, sentence_ids, batches=1)

        # Call LLM
        response = self.provider.complete(prompt)
        self.total_api_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost += response.cost_usd

        # Cache response
        if self.cache:
            self.cache.set(cache_key, response)

        return self._parse_response(lemma, response, examples, sentence_ids, batches=1)

    def _cluster_multi_pass(
        self,
        lemma: str,
        examples: list[str],
        sentence_ids: list[int] | None,
        wordnet_hints: str | None,
    ) -> ClusterResult:
        """Process lemma in multiple batches and merge results."""
        all_cards = []
        all_ignored = []
        all_ignore_reasons = {}
        total_input = 0
        total_output = 0
        total_cost = 0.0

        num_batches = (len(examples) + MAX_EXAMPLES_PER_BATCH - 1) // MAX_EXAMPLES_PER_BATCH

        for batch_idx in range(num_batches):
            start = batch_idx * MAX_EXAMPLES_PER_BATCH
            end = min(start + MAX_EXAMPLES_PER_BATCH, len(examples))

            batch_examples = examples[start:end]
            batch_ids = sentence_ids[start:end] if sentence_ids else None

            result = self._cluster_single_batch(lemma, batch_examples, batch_ids, wordnet_hints)

            # Adjust indices for batch offset
            for card in result.cards:
                adjusted_indices = [idx + start for idx in card.get("selected_example_indices", [])]
                card["selected_example_indices"] = adjusted_indices

            all_cards.extend(result.cards)
            all_ignored.extend([idx + start for idx in result.ignored_indices])
            all_ignore_reasons.update(
                {str(int(k) + start): v for k, v in result.ignore_reasons.items()}
            )

            total_input += result.input_tokens
            total_output += result.output_tokens
            total_cost += result.cost_usd

        # Merge similar cards
        merged_cards = self._merge_similar_cards(all_cards)

        return ClusterResult(
            lemma=lemma,
            cards=merged_cards,
            ignored_indices=all_ignored,
            ignore_reasons=all_ignore_reasons,
            total_examples=len(examples),
            batches_processed=num_batches,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=total_cost,
        )

    def _parse_response(
        self,
        lemma: str,
        response: LLMResponse,
        examples: list[str],
        sentence_ids: list[int] | None,
        batches: int,
    ) -> ClusterResult:
        """Parse LLM response into ClusterResult."""
        try:
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())

            cards = data.get("cards", [])
            ignored = data.get("ignored_indices", [])
            reasons = data.get("ignore_reasons", {})

            # Add lemma to each card
            for card in cards:
                card["lemma"] = lemma

                # Add actual example texts
                indices = card.get("selected_example_indices", [])
                card["examples"] = [examples[i - 1] for i in indices if 0 < i <= len(examples)]

                # Add sentence IDs if available
                if sentence_ids:
                    card["sentence_ids"] = [
                        sentence_ids[i - 1] for i in indices if 0 < i <= len(sentence_ids)
                    ]

            return ClusterResult(
                lemma=lemma,
                cards=cards,
                ignored_indices=ignored,
                ignore_reasons=reasons,
                total_examples=len(examples),
                batches_processed=batches,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse response for lemma '{lemma}': {e}")
            logger.debug(f"Response content: {response.content[:500]}")

            return ClusterResult(
                lemma=lemma,
                cards=[],
                ignored_indices=[],
                ignore_reasons={"error": str(e)},
                total_examples=len(examples),
                batches_processed=batches,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )

    def _merge_similar_cards(self, cards: list[dict]) -> list[dict]:
        """Merge cards with similar definitions using embedding similarity."""
        if len(cards) <= 1:
            return cards

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            from eng_words.wsd.embeddings import get_batch_embeddings

            # Get embeddings for definitions
            definitions = [c.get("definition_en", "") for c in cards]
            embeddings = get_batch_embeddings(definitions)

            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)

            merged = []
            used = set()

            for i, card in enumerate(cards):
                if i in used:
                    continue

                # Find similar cards
                similar_indices = np.where(similarity_matrix[i] >= SIMILARITY_THRESHOLD)[0]

                # Merge examples from similar cards
                merged_examples = list(card.get("selected_example_indices", []))
                merged_texts = list(card.get("examples", []))
                merged_sids = list(card.get("sentence_ids", []))

                for j in similar_indices:
                    if j != i and j not in used:
                        used.add(j)
                        merged_examples.extend(cards[j].get("selected_example_indices", []))
                        merged_texts.extend(cards[j].get("examples", []))
                        merged_sids.extend(cards[j].get("sentence_ids", []))

                # Create merged card (limit to 3 best examples)
                merged_card = card.copy()
                merged_card["selected_example_indices"] = list(set(merged_examples))[:3]
                merged_card["examples"] = merged_texts[:3]
                if merged_sids:
                    merged_card["sentence_ids"] = merged_sids[:3]

                merged.append(merged_card)
                used.add(i)

            logger.info(
                f"Merged {len(cards)} cards into {len(merged)} "
                f"(threshold={SIMILARITY_THRESHOLD})"
            )
            return merged

        except ImportError:
            logger.warning("sklearn not available, skipping merge")
            return cards
        except Exception as e:
            logger.warning(f"Merge failed: {e}, returning original cards")
            return cards

    def stats(self) -> dict:
        """Get clustering statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "cache_hits": self.cache_hits,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost,
        }


def group_examples_by_lemma(
    tokens_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
) -> pd.DataFrame:
    """Group examples by lemma for clustering.

    Args:
        tokens_df: DataFrame with tokens (must have 'lemma', 'sentence_id', 'pos')
        sentences_df: DataFrame with sentences (must have 'sentence_id', 'text')

    Returns:
        DataFrame with columns: lemma, examples, sentence_ids, pos_variants
    """
    # Filter to content words
    content = tokens_df[
        (tokens_df["is_alpha"] == True)
        & (tokens_df["is_stop"] == False)
        & (tokens_df["pos"].isin(["NOUN", "VERB", "ADJ", "ADV"]))
    ].copy()

    # Create sentence lookup
    sentence_lookup = sentences_df.set_index("sentence_id")["text"].to_dict()

    lemma_data = []

    for lemma in content["lemma"].unique():
        lemma_tokens = content[content["lemma"] == lemma]
        sentence_ids = lemma_tokens["sentence_id"].unique().tolist()

        examples = [sentence_lookup.get(sid, "") for sid in sentence_ids if sid in sentence_lookup]

        # Filter out empty examples
        valid_pairs = [(sid, ex) for sid, ex in zip(sentence_ids, examples) if ex]

        if valid_pairs:
            sentence_ids, examples = zip(*valid_pairs)

            lemma_data.append(
                {
                    "lemma": lemma,
                    "examples": list(examples),
                    "sentence_ids": list(sentence_ids),
                    "pos_variants": lemma_tokens["pos"].unique().tolist(),
                    "example_count": len(examples),
                }
            )

    result = pd.DataFrame(lemma_data)

    if len(result) > 0:
        result = result.sort_values("example_count", ascending=False)
        total_examples = result["example_count"].sum()
    else:
        total_examples = 0

    logger.info(f"Grouped {len(result)} lemmas, " f"total examples: {total_examples}")

    return result
