"""
Smart Fallback for WSD errors.

When LLM rejects all examples for a synset, this module provides
fallback logic to try the next most frequent synset.
"""

import logging

from nltk.corpus import wordnet as wn

logger = logging.getLogger(__name__)


def get_synsets_by_frequency(lemma: str, pos: str) -> list[str]:
    """
    Get all synsets for a lemma, sorted by frequency (most common first).

    WordNet returns synsets in frequency order by default.
    Note: wn.synsets() returns all synsets containing the lemma,
    but we only want synsets named after this specific lemma.

    Args:
        lemma: Word lemma (e.g., "run")
        pos: Part of speech (e.g., "v" for verb, "n" for noun)

    Returns:
        List of synset IDs in frequency order (e.g., ["run.v.01", "run.v.02", ...])
    """
    try:
        # Get synsets filtered by POS
        synsets = wn.synsets(lemma, pos=pos)

        # Filter to only synsets named after this lemma
        # wn.synsets("run") returns scat.v.01, flee.v.01 etc. - we only want run.v.XX
        lemma_prefix = f"{lemma}.{pos}."
        filtered = [s.name() for s in synsets if s.name().startswith(lemma_prefix)]

        return filtered
    except Exception as e:
        logger.warning(f"Error getting synsets for {lemma}.{pos}: {e}")
        return []


def get_fallback_synset(
    lemma: str,
    pos: str,
    failed_synset: str,
    existing_synsets: set[str],
) -> str | None:
    """
    Find the next most frequent synset for fallback.

    When a synset fails (LLM rejects all examples), this function
    finds the next available synset to try.

    Args:
        lemma: Word lemma (e.g., "about")
        pos: Part of speech (e.g., "r" for adverb)
        failed_synset: Synset that failed (e.g., "about.r.05")
        existing_synsets: Synsets already used in the deck

    Returns:
        Next synset_id to try, or None if all exhausted.

    Example:
        >>> get_fallback_synset("about", "r", "about.r.05", {"about.r.01", "about.r.03"})
        "about.r.02"  # .01 exists, .02 is next available
    """
    # Get all synsets sorted by frequency
    all_synsets = get_synsets_by_frequency(lemma, pos)

    if not all_synsets:
        logger.debug(f"No synsets found for {lemma}.{pos}")
        return None

    # Find first synset that is:
    # 1. Not the failed synset
    # 2. Not already in existing_synsets
    for synset_id in all_synsets:
        if synset_id != failed_synset and synset_id not in existing_synsets:
            logger.debug(f"Fallback for {lemma}: {failed_synset} -> {synset_id}")
            return synset_id

    logger.debug(f"No fallback available for {lemma}.{pos} - all synsets exhausted")
    return None
