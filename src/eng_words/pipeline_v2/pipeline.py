"""Word Family Pipeline v2: Two-stage processing.

Stage 1: MeaningExtractor - identify all meanings
Stage 2: CardGenerator - create flashcards
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from eng_words.llm.base import LLMProvider
from eng_words.llm.response_cache import ResponseCache

from .card_generator import CardGenerator
from .data_models import PipelineResult
from .meaning_extractor import MeaningExtractor

logger = logging.getLogger(__name__)


class WordFamilyPipelineV2:
    """Two-stage pipeline for creating Anki flashcards.

    Stage 1: Extract meanings from all examples (including spoilers)
    Stage 2: Generate clean flashcards for each meaning

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

        self.extractor = MeaningExtractor(provider, cache)
        self.generator = CardGenerator(provider, cache)

    def process_lemma(
        self,
        lemma: str,
        examples: list[str],
        sentence_ids: list[int],
    ) -> PipelineResult:
        """Process a single lemma through both stages.

        Args:
            lemma: The word to process
            examples: All example sentences containing the lemma
            sentence_ids: Corresponding sentence IDs

        Returns:
            PipelineResult with extraction and generation results
        """
        # Stage 1: Extract meanings
        extraction = self.extractor.extract(lemma, examples, sentence_ids)

        # Stage 2: Generate cards
        generation = self.generator.generate(extraction, examples, sentence_ids)

        return PipelineResult(
            lemma=lemma,
            extraction=extraction,
            generation=generation,
        )

    def process_batch(
        self,
        lemma_groups: pd.DataFrame,
        progress: bool = True,
    ) -> list[PipelineResult]:
        """Process multiple lemmas.

        Args:
            lemma_groups: DataFrame with columns: lemma, examples, sentence_ids
            progress: Show progress bar

        Returns:
            List of PipelineResult for each lemma
        """
        results = []

        iterator = lemma_groups.iterrows()
        if progress:
            iterator = tqdm(
                iterator,
                total=len(lemma_groups),
                desc="Processing lemmas",
            )

        for idx, row in iterator:
            lemma = row["lemma"]
            examples = row["examples"]
            sentence_ids = row["sentence_ids"]

            if progress:
                iterator.set_postfix(
                    {
                        "lemma": lemma[:15],
                        "cards": sum(r.total_cards for r in results),
                    }
                )

            try:
                result = self.process_lemma(lemma, examples, sentence_ids)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing '{lemma}': {e}")

        return results

    def stats(self) -> dict[str, Any]:
        """Return combined statistics."""
        ext_stats = self.extractor.stats()
        gen_stats = self.generator.stats()

        return {
            "extraction": ext_stats,
            "generation": gen_stats,
            "total_api_calls": ext_stats["total_api_calls"] + gen_stats["total_api_calls"],
            "total_cache_hits": ext_stats["cache_hits"] + gen_stats["cache_hits"],
            "total_cost_usd": round(ext_stats["total_cost_usd"] + gen_stats["total_cost_usd"], 4),
        }


def save_results(
    results: list[PipelineResult],
    output_path: Path | str,
    include_extraction: bool = True,
) -> None:
    """Save pipeline results to JSON.

    Args:
        results: List of pipeline results
        output_path: Path to save JSON
        include_extraction: Include Stage 1 extraction data
    """
    output_path = Path(output_path)

    data = {
        "pipeline_version": "v2",
        "total_lemmas": len(results),
        "total_cards": sum(r.total_cards for r in results),
        "total_cost_usd": round(sum(r.total_cost_usd for r in results), 4),
        "cards": [],
    }

    if include_extraction:
        data["extractions"] = []

    for result in results:
        # Add cards
        for card in result.generation.cards:
            card_dict = asdict(card)
            data["cards"].append(card_dict)

        # Add extraction if requested
        if include_extraction:
            ext_dict = {
                "lemma": result.extraction.lemma,
                "all_sentence_ids": result.extraction.all_sentence_ids,
                "meanings": [asdict(m) for m in result.extraction.meanings],
                "cost_usd": result.extraction.cost_usd,
            }
            data["extractions"].append(ext_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(data['cards'])} cards to {output_path}")


def load_results(input_path: Path | str) -> dict:
    """Load pipeline results from JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
