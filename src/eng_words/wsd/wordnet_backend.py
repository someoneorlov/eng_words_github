"""WordNet-based Word Sense Disambiguation backend.

This module implements WSD using Sentence-Transformers embeddings and
WordNet definitions. For each word in context, it:
1. Gets all possible synsets from WordNet
2. Computes embeddings for the sentence and synset definitions
3. Selects the synset with highest cosine similarity
4. Maps the synset to a supersense category
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from eng_words.wsd.aggregator import aggregate_sense_statistics
from eng_words.wsd.base import (
    SenseAnnotation,
    SenseBackend,
    validate_sentences_df,
    validate_tokens_df,
)
from eng_words.wsd.candidate_selector import compute_combined_score, get_context_boost
from eng_words.wsd.embeddings import (
    compute_cosine_similarity,
    get_batch_embeddings,
    get_definition_cache,
    get_sentence_embedding,
)
from eng_words.wsd.phrasal_verbs import detect_all_constructions
from eng_words.wsd.wordnet_utils import (
    get_definition,
    get_synsets,
    map_spacy_pos_to_wordnet,
    synset_to_supersense,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Default confidence threshold for WSD predictions.
# This is an empirical value chosen as a reasonable default:
# - Cosine similarity ranges from -1.0 to 1.0
# - 0.3 is a relatively low threshold, allowing more results
# - Higher values (0.4-0.5) are more strict but may filter out valid senses
# - Lower values (0.2) are more permissive but may include incorrect matches
# Users can adjust this based on their needs and evaluation results.
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

# POS tags that we process for WSD (content words)
# Note: PROPN (proper nouns) are excluded as names don't have WordNet senses
CONTENT_POS_TAGS = {"NOUN", "VERB", "ADJ", "ADV"}


# =============================================================================
# WORDNET SENSE BACKEND
# =============================================================================


class WordNetSenseBackend(SenseBackend):
    """WSD backend using Sentence-Transformers and WordNet.

    This backend disambiguates word senses by:
    1. Getting the sentence embedding using Sentence-Transformers
    2. Getting all possible synsets for the target word from WordNet
    3. Computing embeddings for each synset's definition
    4. Selecting the synset with highest cosine similarity to the sentence
    5. Mapping the selected synset to a supersense category

    Attributes:
        confidence_threshold: Minimum confidence score to consider reliable
        _definition_cache: Cache for WordNet definition embeddings

    Example:
        >>> backend = WordNetSenseBackend()
        >>> result = backend.disambiguate_word(
        ...     sentence="I deposited money at the bank",
        ...     lemma="bank",
        ...     pos="n"
        ... )
        >>> print(result.sense_label)
        'noun.group'
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        """Initialize the WordNet sense backend.

        Args:
            confidence_threshold: Minimum confidence score for reliable predictions.
                                  Results below this threshold may be less accurate.
        """
        self.confidence_threshold = confidence_threshold
        self._definition_cache = get_definition_cache()

    @property
    def name(self) -> str:
        """Return backend name."""
        return "WordNet + Sentence-Transformers"

    def is_confident(self, annotation: SenseAnnotation) -> bool:
        """Check if annotation meets confidence threshold.

        Args:
            annotation: The sense annotation to check

        Returns:
            True if confidence >= threshold
        """
        return annotation.confidence >= self.confidence_threshold

    # =========================================================================
    # SINGLE WORD DISAMBIGUATION
    # =========================================================================

    def disambiguate_word(
        self,
        sentence: str,
        lemma: str,
        pos: Optional[str] = None,
    ) -> SenseAnnotation:
        """Disambiguate a single word in context.

        Args:
            sentence: The sentence containing the target word
            lemma: The lemmatized form of the target word
            pos: Optional WordNet POS filter ('n', 'v', 'a', 'r')
                 or spaCy POS ('NOUN', 'VERB', 'ADJ', 'ADV')

        Returns:
            SenseAnnotation with the disambiguated sense
        """
        # Normalize POS from spaCy format if needed
        wn_pos = self._normalize_pos(pos)
        spacy_pos = pos if pos in CONTENT_POS_TAGS else None

        # =====================================================================
        # CONSTRUCTION DETECTION
        # Check for grammatical constructions before WSD
        # Includes both regex patterns and spaCy-based phrasal verb detection
        # =====================================================================
        constructions = detect_all_constructions(sentence, lemma, spacy_pos or "")
        construction_tag = None

        if constructions:
            match = constructions[0]  # Take highest priority match
            construction_tag = match.construction_id

            if match.policy == "SKIP":
                # Don't assign any synset for grammatical constructions
                # (e.g., "going to" future tense, "point of view" multiword)
                return SenseAnnotation(
                    lemma=lemma,
                    sense_id=None,
                    sense_label="construction",
                    confidence=0.0,
                    definition=match.reason,
                    construction_tag=construction_tag,
                )
            # Note: CONSTRAIN and OVERRIDE policies are recorded but not applied
            # to avoid over-fitting to patterns that may conflict with gold labels

        # =====================================================================
        # GET SYNSETS
        # =====================================================================
        synsets = get_synsets(lemma, pos=wn_pos)

        # Fallback: if no synsets with POS filter, try without POS
        # This handles cases where spaCy POS differs from WordNet POS
        # (e.g., "stricken" is VERB in spaCy but satellite adj in WordNet)
        if not synsets and wn_pos is not None:
            synsets = get_synsets(lemma, pos=None)

        if not synsets:
            return SenseAnnotation(
                lemma=lemma,
                sense_id=None,
                sense_label="unknown",
                confidence=0.0,
                definition=None,
                construction_tag=construction_tag,
            )

        # =====================================================================
        # SMART SCORING WITH CONTEXT BOOST
        # Combines embedding similarity with context keyword matching
        # =====================================================================
        # Get sentence embedding
        sentence_emb = get_sentence_embedding(sentence)

        # Get synset definitions and their embeddings
        synset_defs = [(s.name(), get_definition(s)) for s in synsets]
        def_embeddings = self._definition_cache.get_batch(synset_defs)

        # Compute embedding similarities for all synsets
        embedding_scores: dict[str, float] = {}
        for synset in synsets:
            synset_id = synset.name()
            if synset_id in def_embeddings:
                def_emb = def_embeddings[synset_id]
                score = compute_cosine_similarity(sentence_emb, def_emb)
                embedding_scores[synset_id] = score

        # Compute combined scores (embedding + context boost)
        best_synset = None
        best_score = -1.0
        best_definition = None

        for synset in synsets:
            synset_id = synset.name()
            if synset_id not in embedding_scores:
                continue

            emb_score = embedding_scores[synset_id]
            context_boost = get_context_boost(synset, sentence)
            combined_score = compute_combined_score(synset, emb_score, sentence, context_boost)

            if combined_score > best_score:
                best_score = combined_score
                best_synset = synset
                best_definition = get_definition(synset)

        # If no synset found OR best score is non-positive (bad match), mark as unknown
        if best_synset is None or best_score <= 0:
            return SenseAnnotation(
                lemma=lemma,
                sense_id=None,
                sense_label="unknown",
                confidence=0.0,
                definition=None,
                construction_tag=construction_tag,
            )

        return SenseAnnotation(
            lemma=lemma,
            sense_id=best_synset.name(),
            sense_label=synset_to_supersense(best_synset),
            confidence=float(best_score),
            definition=best_definition,
            construction_tag=construction_tag,
        )

    def disambiguate_batch(
        self,
        items: List[Tuple[str, str, Optional[str]]],
    ) -> List[SenseAnnotation]:
        """Disambiguate multiple words efficiently.

        This method is more efficient than calling disambiguate_word repeatedly
        because it batches embedding computations.

        Args:
            items: List of (sentence, lemma, pos) tuples

        Returns:
            List of SenseAnnotation results
        """
        if not items:
            return []

        # Collect all unique sentences for batch embedding
        unique_sentences = list(set(item[0] for item in items))
        sentence_embeddings = get_batch_embeddings(unique_sentences)
        sentence_to_emb = dict(zip(unique_sentences, sentence_embeddings))

        # Collect all synset definitions needed
        all_synset_defs: List[Tuple[str, str]] = []
        item_synsets = []

        for sentence, lemma, pos in items:
            wn_pos = self._normalize_pos(pos)
            synsets = get_synsets(lemma, pos=wn_pos)

            # Fallback: if no synsets with POS filter, try without POS
            # This handles cases where spaCy POS differs from WordNet POS
            if not synsets and wn_pos is not None:
                synsets = get_synsets(lemma, pos=None)

            item_synsets.append(synsets)

            for synset in synsets:
                synset_id = synset.name()
                definition = get_definition(synset)
                all_synset_defs.append((synset_id, definition))

        # Batch compute definition embeddings
        if all_synset_defs:
            self._definition_cache.get_batch(all_synset_defs)

        # Now disambiguate each item
        results = []
        for (sentence, lemma, pos), synsets in zip(items, item_synsets):
            if not synsets:
                results.append(
                    SenseAnnotation(
                        lemma=lemma,
                        sense_id=None,
                        sense_label="unknown",
                        confidence=0.0,
                        definition=None,
                    )
                )
                continue

            sentence_emb = sentence_to_emb[sentence]

            best_synset = None
            best_score = -1.0
            best_definition = None

            for synset in synsets:
                synset_id = synset.name()
                def_emb = self._definition_cache.get(synset_id)

                if def_emb is not None:
                    score = compute_cosine_similarity(sentence_emb, def_emb)

                    if score > best_score:
                        best_score = score
                        best_synset = synset
                        best_definition = get_definition(synset)

            # If no synset found OR best score is non-positive, mark as unknown
            if best_synset is None or best_score <= 0:
                results.append(
                    SenseAnnotation(
                        lemma=lemma,
                        sense_id=None,
                        sense_label="unknown",
                        confidence=0.0,
                        definition=None,
                    )
                )
            else:
                results.append(
                    SenseAnnotation(
                        lemma=lemma,
                        sense_id=best_synset.name(),
                        sense_label=synset_to_supersense(best_synset),
                        confidence=float(best_score),
                        definition=best_definition,
                    )
                )

        return results

    # =========================================================================
    # BATCH ANNOTATION
    # =========================================================================

    def annotate(
        self,
        tokens_df: pd.DataFrame,
        sentences_df: pd.DataFrame,
        checkpoint_path: Optional[Path | str] = None,
        checkpoint_interval: int = 1000,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Add sense annotations to tokens DataFrame.

        Args:
            tokens_df: DataFrame with token information (lemma, sentence_id, pos)
            sentences_df: DataFrame with sentences (sentence_id, sentence)
            checkpoint_path: Optional path to save intermediate results (parquet format).
                            If provided, saves every checkpoint_interval tokens.
            checkpoint_interval: Number of tokens to process before saving checkpoint.
            show_progress: Whether to show progress bar.

        Returns:
            DataFrame with added columns: synset_id, supersense, sense_confidence
        """
        validate_tokens_df(tokens_df)
        validate_sentences_df(sentences_df)

        if tokens_df.empty:
            result = tokens_df.copy()
            result["synset_id"] = pd.Series(dtype="object")
            result["supersense"] = pd.Series(dtype="object")
            result["sense_confidence"] = pd.Series(dtype="float64")
            return result

        # Create sentence lookup
        sentence_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))

        # Prepare items for batch processing
        items = []
        indices = []

        for idx, row in tokens_df.iterrows():
            pos = row["pos"]
            if pos.upper() not in CONTENT_POS_TAGS:
                continue

            sentence_id = row["sentence_id"]
            if sentence_id not in sentence_lookup:
                continue

            sentence = sentence_lookup[sentence_id]
            lemma = row["lemma"]

            items.append((sentence, lemma, pos))
            indices.append(idx)

        # Build result DataFrame
        result = tokens_df.copy()
        result["synset_id"] = None
        result["supersense"] = "unknown"
        result["sense_confidence"] = 0.0

        # Process in batches with progress bar and checkpointing
        total_items = len(items)
        if total_items == 0:
            return result

        # Convert checkpoint_path to Path if string
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)

        # Process with progress bar
        iterator = range(0, total_items, checkpoint_interval)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Annotating tokens",
                unit="batch",
                total=(total_items + checkpoint_interval - 1) // checkpoint_interval,
            )

        for batch_start in iterator:
            batch_end = min(batch_start + checkpoint_interval, total_items)
            batch_items = items[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]

            # Batch disambiguate
            batch_annotations = self.disambiguate_batch(batch_items)

            # Update result DataFrame
            for idx, annotation in zip(batch_indices, batch_annotations):
                result.at[idx, "synset_id"] = annotation.sense_id
                result.at[idx, "supersense"] = annotation.sense_label
                result.at[idx, "sense_confidence"] = annotation.confidence

            # Save checkpoint if path provided
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_parquet(checkpoint_path)

        return result

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def aggregate(
        self,
        annotated_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate statistics by (lemma, supersense).

        Args:
            annotated_df: DataFrame with sense annotations

        Returns:
            DataFrame with aggregated sense statistics including:
            - lemma, supersense, sense_freq, book_freq, sense_ratio
            - doc_count, sense_count, dominant_supersense
        """
        return aggregate_sense_statistics(annotated_df)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _normalize_pos(self, pos: Optional[str]) -> Optional[str]:
        """Normalize POS tag to WordNet format.

        Args:
            pos: POS tag (spaCy or WordNet format)

        Returns:
            WordNet POS tag ('n', 'v', 'a', 'r') or None
        """
        if pos is None:
            return None

        pos_upper = pos.upper()

        # Already WordNet format
        if pos_upper in ("N", "V", "A", "R"):
            return pos_upper.lower()

        # spaCy format
        return map_spacy_pos_to_wordnet(pos_upper)
