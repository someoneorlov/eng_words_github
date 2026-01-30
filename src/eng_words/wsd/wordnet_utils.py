"""WordNet utilities for Word Sense Disambiguation.

This module provides functions for:
- Getting synsets for lemmas
- Mapping synsets to supersenses
- Converting spaCy POS tags to WordNet POS tags
- Getting definitions for synsets
"""

from typing import List, Optional, Tuple

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from eng_words.constants.supersenses import get_supersense

# =============================================================================
# POS MAPPING
# =============================================================================

# spaCy POS tags to WordNet POS tags
SPACY_TO_WORDNET_POS: dict[str, str] = {
    "NOUN": "n",
    "PROPN": "n",  # Proper nouns treated as nouns
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}


def map_spacy_pos_to_wordnet(spacy_pos: str) -> Optional[str]:
    """Map spaCy POS tag to WordNet POS tag.

    Args:
        spacy_pos: spaCy POS tag (e.g., "NOUN", "VERB", "ADJ", "ADV")

    Returns:
        WordNet POS tag ('n', 'v', 'a', 'r') or None if not mappable

    Examples:
        >>> map_spacy_pos_to_wordnet("NOUN")
        'n'
        >>> map_spacy_pos_to_wordnet("VERB")
        'v'
        >>> map_spacy_pos_to_wordnet("DET")
        None
    """
    return SPACY_TO_WORDNET_POS.get(spacy_pos.upper())


# =============================================================================
# SYNSET FUNCTIONS
# =============================================================================


def get_synsets(lemma: str, pos: Optional[str] = None) -> List[Synset]:
    """Get WordNet synsets for a lemma.

    Args:
        lemma: The word to look up (case-insensitive).
               Spaces are automatically converted to underscores.
        pos: WordNet POS tag ('n', 'v', 'a', 'r') or None for all POS

    Returns:
        List of Synset objects, empty list if word not found

    Examples:
        >>> synsets = get_synsets("bank", pos="n")
        >>> len(synsets) > 0
        True
        >>> synsets = get_synsets("xyznonexistent", pos="n")
        >>> synsets
        []
    """
    if not lemma or not lemma.strip():
        return []

    # Normalize: lowercase and replace spaces with underscores
    normalized = lemma.lower().strip().replace(" ", "_")

    try:
        synsets = wn.synsets(normalized, pos=pos)
        return list(synsets)
    except Exception:
        # Handle any WordNet lookup errors gracefully
        return []


# Special mapping for noun.Tops synsets to their appropriate supersenses
# These are top-level abstract synsets that need specific handling
_NOUN_TOPS_MAPPING: dict[str, str] = {
    "person.n.01": "noun.person",
    "group.n.01": "noun.group",
    "cognition.n.01": "noun.cognition",
    "motivation.n.01": "noun.motive",
    "time.n.05": "noun.time",
    "animal.n.01": "noun.animal",
    "feeling.n.01": "noun.feeling",
    "possession.n.02": "noun.possession",
    "space.n.01": "noun.location",
    "act.n.02": "noun.act",
    "state.n.02": "noun.state",
    "thing.n.12": "noun.object",
    "organism.n.01": "noun.animal",
    "entity.n.01": "noun.object",
    "abstraction.n.06": "noun.cognition",
    "physical_entity.n.01": "noun.object",
    "object.n.01": "noun.object",
    "whole.n.02": "noun.group",
    "living_thing.n.01": "noun.animal",
    "causal_agent.n.01": "noun.person",
    "plant.n.02": "noun.plant",
    "life.n.10": "noun.animal",
    "cell.n.02": "noun.body",
    "location.n.01": "noun.location",
    "attribute.n.02": "noun.attribute",
    "communication.n.02": "noun.communication",
    "relation.n.01": "noun.relation",
    "measure.n.02": "noun.quantity",
    "event.n.01": "noun.event",
    "process.n.06": "noun.process",
    "phenomenon.n.01": "noun.phenomenon",
    "psychological_feature.n.01": "noun.cognition",
}


def synset_to_supersense(synset: Synset) -> str:
    """Get supersense category for a synset.

    Uses the synset's lexname (lexicographer filename) to determine
    the supersense category. Special handling for 'noun.Tops' synsets
    which are top-level abstract categories.

    Args:
        synset: WordNet Synset object

    Returns:
        Supersense string (e.g., "noun.person", "verb.motion")
        Returns "unknown" if supersense cannot be determined

    Examples:
        >>> from nltk.corpus import wordnet as wn
        >>> synset = wn.synset("dog.n.01")
        >>> synset_to_supersense(synset)
        'noun.animal'
        >>> synset = wn.synset("person.n.01")  # noun.Tops
        >>> synset_to_supersense(synset)
        'noun.person'
    """
    try:
        lexname = synset.lexname()

        # Special handling for noun.Tops
        if lexname == "noun.Tops":
            synset_name = synset.name()
            if synset_name in _NOUN_TOPS_MAPPING:
                return _NOUN_TOPS_MAPPING[synset_name]
            # Fallback: try to infer from synset name
            base_name = synset_name.split(".")[0]
            potential_supersense = f"noun.{base_name}"
            if get_supersense(potential_supersense):
                return potential_supersense
            # Final fallback for noun.Tops
            return "noun.object"

        supersense = get_supersense(lexname)
        return supersense if supersense else "unknown"
    except Exception:
        return "unknown"


def get_definition(synset: Synset) -> str:
    """Get definition for a synset.

    Falls back to lemma names if definition is empty.

    Args:
        synset: WordNet Synset object

    Returns:
        Definition string, never empty

    Examples:
        >>> from nltk.corpus import wordnet as wn
        >>> synset = wn.synset("dog.n.01")
        >>> definition = get_definition(synset)
        >>> len(definition) > 0
        True
    """
    try:
        definition = synset.definition()
        if definition and definition.strip():
            return definition.strip()

        # Fallback to lemma names
        lemma_names = synset.lemma_names()
        if lemma_names:
            return ", ".join(lemma_names)

        # Ultimate fallback: synset name
        return synset.name()
    except Exception:
        return synset.name() if synset else "unknown"


def get_synsets_with_definitions(lemma: str, pos: Optional[str] = None) -> List[Tuple[str, str]]:
    """Get synsets with their definitions for a lemma.

    Convenient function that returns synset IDs paired with definitions,
    ready for embedding cache lookup.

    Args:
        lemma: The word to look up
        pos: WordNet POS tag or None for all POS

    Returns:
        List of (synset_id, definition) tuples

    Examples:
        >>> result = get_synsets_with_definitions("bank", pos="n")
        >>> len(result) > 0
        True
        >>> synset_id, definition = result[0]
        >>> "bank" in synset_id
        True
    """
    synsets = get_synsets(lemma, pos=pos)
    return [(synset.name(), get_definition(synset)) for synset in synsets]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_all_supersenses_for_lemma(
    lemma: str, pos: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """Get all possible supersenses for a lemma.

    Args:
        lemma: The word to look up
        pos: WordNet POS tag or None for all POS

    Returns:
        List of (synset_id, supersense, definition) tuples

    Examples:
        >>> result = get_all_supersenses_for_lemma("bank", pos="n")
        >>> len(result) > 0
        True
        >>> synset_id, supersense, definition = result[0]
        >>> supersense.startswith("noun.")
        True
    """
    synsets = get_synsets(lemma, pos=pos)
    return [
        (synset.name(), synset_to_supersense(synset), get_definition(synset)) for synset in synsets
    ]


def has_multiple_senses(lemma: str, pos: Optional[str] = None) -> bool:
    """Check if a lemma has multiple senses in WordNet.

    Args:
        lemma: The word to check
        pos: WordNet POS tag or None for all POS

    Returns:
        True if lemma has more than one sense

    Examples:
        >>> has_multiple_senses("bank", pos="n")
        True
        >>> has_multiple_senses("xyznonexistent", pos="n")
        False
    """
    synsets = get_synsets(lemma, pos=pos)
    return len(synsets) > 1


def get_most_common_synset(lemma: str, pos: Optional[str] = None) -> Optional[Synset]:
    """Get the most common (first) synset for a lemma.

    WordNet orders synsets by frequency of use, so the first synset
    is typically the most common meaning.

    Args:
        lemma: The word to look up
        pos: WordNet POS tag or None for all POS

    Returns:
        Most common Synset or None if not found

    Examples:
        >>> synset = get_most_common_synset("bank", pos="n")
        >>> synset is not None
        True
    """
    synsets = get_synsets(lemma, pos=pos)
    return synsets[0] if synsets else None
