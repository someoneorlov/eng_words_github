"""
Smart Candidate Selection for WSD.

This module implements improved candidate selection and scoring
for Word Sense Disambiguation, combining:
1. Embedding-based similarity scores
2. Context keyword matching
3. Supersense diversity

Based on analysis of Gold Dataset errors:
- Gold synset is almost always in candidates (99.6%)
- But gold synset ranks #2-3 in 39% of errors
- Smart selection aims to improve this ranking
"""

import re
from typing import Any

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

# Stop words to exclude from context matching
STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "although",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "any",
    "about",
}

# Weight for context boost in combined score
# Note: Reduced from 0.15 after benchmark showed regressions
CONTEXT_BOOST_WEIGHT = 0.08


def get_context_boost(synset: Synset, sentence: str) -> float:
    """
    Calculate context boost based on keyword overlap.

    Args:
        synset: WordNet synset
        sentence: Context sentence

    Returns:
        Boost value (0.0 to 1.0)
    """
    # Extract content words from sentence
    sentence_words = set(
        word.lower()
        for word in re.findall(r"\b[a-z]+\b", sentence.lower())
        if word not in STOP_WORDS and len(word) > 2
    )

    if not sentence_words:
        return 0.0

    # Get words from synset definition and examples
    definition = synset.definition().lower()
    examples = " ".join(synset.examples()).lower()
    gloss_text = definition + " " + examples

    gloss_words = set(
        word.lower()
        for word in re.findall(r"\b[a-z]+\b", gloss_text)
        if word not in STOP_WORDS and len(word) > 2
    )

    if not gloss_words:
        return 0.0

    # Calculate overlap
    overlap = sentence_words & gloss_words
    if not overlap:
        return 0.0

    # Normalize by smaller set size
    overlap_ratio = len(overlap) / min(len(sentence_words), len(gloss_words))

    # Return boost (capped at 0.5)
    return min(overlap_ratio, 0.5)


def compute_combined_score(
    synset: Synset,
    embedding_score: float,
    sentence: str,
    context_boost: float = 0.0,
) -> float:
    """
    Compute combined score for a synset.

    Args:
        synset: WordNet synset
        embedding_score: Similarity score from embeddings (0-1)
        sentence: Context sentence
        context_boost: Pre-computed context boost (or 0 to compute)

    Returns:
        Combined score (0-1)
    """
    # Base score from embeddings
    base_score = embedding_score

    # Add context boost
    final_score = base_score + (context_boost * CONTEXT_BOOST_WEIGHT)

    # Normalize to [0, 1]
    return min(max(final_score, 0.0), 1.0)


def select_smart_candidates(
    lemma: str,
    pos: str,
    sentence: str,
    all_synsets: list[Synset],
    embedding_scores: dict[str, float],
    max_candidates: int = 15,
) -> list[tuple[str, float, str]]:
    """
    Select diverse and relevant candidates for WSD.

    Strategy:
    1. Score all candidates with combined score (embedding + context)
    2. Take top-K by combined score (K=60% of max)
    3. Add diverse synsets by supersense (remaining slots)

    Args:
        lemma: Target word lemma
        pos: Part of speech
        sentence: Context sentence
        all_synsets: All WordNet synsets for the lemma
        embedding_scores: Pre-computed embedding similarity scores
        max_candidates: Maximum number of candidates to return

    Returns:
        List of (synset_id, combined_score, selection_reason)
    """
    if not all_synsets:
        return []

    # Step 1: Compute combined scores for all synsets
    scored_synsets: list[tuple[Synset, float, float]] = []  # (synset, combined, boost)

    for synset in all_synsets:
        synset_id = synset.name()
        emb_score = embedding_scores.get(synset_id, 0.0)
        context_boost = get_context_boost(synset, sentence)
        combined = compute_combined_score(synset, emb_score, sentence, context_boost)
        scored_synsets.append((synset, combined, context_boost))

    # Step 2: Sort by combined score
    scored_synsets.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Select candidates
    selected: list[tuple[str, float, str]] = []
    seen_supersenses: set[str] = set()

    # Take top by score (60% of slots)
    top_k = max(1, int(max_candidates * 0.6))

    for synset, combined, boost in scored_synsets[:top_k]:
        reason = "top_score"
        if boost > 0.1:
            reason = "top_score+context"
        selected.append((synset.name(), combined, reason))
        seen_supersenses.add(synset.lexname())

    # Fill remaining slots with diverse supersenses
    remaining_slots = max_candidates - len(selected)

    for synset, combined, boost in scored_synsets[top_k:]:
        if remaining_slots <= 0:
            break

        supersense = synset.lexname()
        if supersense not in seen_supersenses:
            selected.append((synset.name(), combined, "diverse_supersense"))
            seen_supersenses.add(supersense)
            remaining_slots -= 1

    # If still have slots, add by context boost
    if remaining_slots > 0:
        already_selected = {s[0] for s in selected}
        context_sorted = sorted(scored_synsets, key=lambda x: x[2], reverse=True)

        for synset, combined, boost in context_sorted:
            if remaining_slots <= 0:
                break
            if synset.name() in already_selected:
                continue
            if boost > 0.05:
                selected.append((synset.name(), combined, "context_match"))
                remaining_slots -= 1

    return selected[:max_candidates]


def get_selection_stats(
    all_synsets: list[Synset],
    selected: list[tuple[str, float, str]],
) -> dict[str, Any]:
    """
    Get statistics about candidate selection.

    Args:
        all_synsets: All available synsets
        selected: Selected candidates

    Returns:
        Dictionary with selection statistics
    """
    reasons = {}
    for _, _, reason in selected:
        reasons[reason] = reasons.get(reason, 0) + 1

    supersenses = set()
    for synset_id, _, _ in selected:
        try:
            s = wn.synset(synset_id)
            supersenses.add(s.lexname())
        except Exception:
            pass

    return {
        "total_available": len(all_synsets),
        "selected_count": len(selected),
        "unique_supersenses": len(supersenses),
        "selection_reasons": reasons,
    }
