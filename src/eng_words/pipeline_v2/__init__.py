"""Word Family Pipeline v2: Two-stage LLM processing.

Stage 1: MeaningExtractor - identify all meanings from examples
Stage 2: CardGenerator - create flashcards with clean examples

Usage:
    from eng_words.pipeline_v2 import WordFamilyPipelineV2
    
    pipeline = WordFamilyPipelineV2(provider, cache)
    result = pipeline.process_lemma(lemma, examples, sentence_ids)
"""

from .card_generator import CardGenerator
from .data_models import (
    CleanExample,
    ExtractedMeaning,
    ExtractionResult,
    FinalCard,
    GenerationResult,
    PipelineResult,
    SourceExample,
)
from .meaning_extractor import MeaningExtractor
from .pipeline import WordFamilyPipelineV2, load_results, save_results

__all__ = [
    # Main pipeline
    "WordFamilyPipelineV2",
    # Stage 1
    "MeaningExtractor",
    "ExtractionResult",
    "ExtractedMeaning",
    "SourceExample",
    # Stage 2
    "CardGenerator",
    "GenerationResult",
    "FinalCard",
    "CleanExample",
    # Combined
    "PipelineResult",
    # Utils
    "save_results",
    "load_results",
]
