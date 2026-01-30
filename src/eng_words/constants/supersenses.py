"""WordNet Supersense constants.

Supersenses are coarse-grained semantic categories from WordNet.
There are 43 supersenses total: 25 for nouns, 15 for verbs, 2 for adjectives, 1 for adverbs.

Reference: https://wordnet.princeton.edu/documentation/lexnames5wn
"""

# =============================================================================
# Noun Supersenses (25 categories)
# =============================================================================

NOUN_ACT = "noun.act"  # nouns denoting acts or actions
NOUN_ANIMAL = "noun.animal"  # nouns denoting animals
NOUN_ARTIFACT = "noun.artifact"  # nouns denoting man-made objects
NOUN_ATTRIBUTE = "noun.attribute"  # nouns denoting attributes of people and objects
NOUN_BODY = "noun.body"  # nouns denoting body parts
NOUN_COGNITION = "noun.cognition"  # nouns denoting cognitive processes and contents
NOUN_COMMUNICATION = "noun.communication"  # nouns denoting communicative processes/contents
NOUN_EVENT = "noun.event"  # nouns denoting natural events
NOUN_FEELING = "noun.feeling"  # nouns denoting feelings and emotions
NOUN_FOOD = "noun.food"  # nouns denoting foods and drinks
NOUN_GROUP = "noun.group"  # nouns denoting groupings of people or objects
NOUN_LOCATION = "noun.location"  # nouns denoting spatial position
NOUN_MOTIVE = "noun.motive"  # nouns denoting goals
NOUN_OBJECT = "noun.object"  # nouns denoting natural objects (not man-made)
NOUN_PERSON = "noun.person"  # nouns denoting people
NOUN_PHENOMENON = "noun.phenomenon"  # nouns denoting natural phenomena
NOUN_PLANT = "noun.plant"  # nouns denoting plants
NOUN_POSSESSION = "noun.possession"  # nouns denoting possession and transfer of possession
NOUN_PROCESS = "noun.process"  # nouns denoting natural processes
NOUN_QUANTITY = "noun.quantity"  # nouns denoting quantities and units of measure
NOUN_RELATION = "noun.relation"  # nouns denoting relations between people/things
NOUN_SHAPE = "noun.shape"  # nouns denoting two and three dimensional shapes
NOUN_STATE = "noun.state"  # nouns denoting stable states of affairs
NOUN_SUBSTANCE = "noun.substance"  # nouns denoting substances
NOUN_TIME = "noun.time"  # nouns denoting time and temporal relations

# =============================================================================
# Verb Supersenses (15 categories)
# =============================================================================

VERB_BODY = "verb.body"  # verbs of grooming, dressing and bodily care
VERB_CHANGE = "verb.change"  # verbs of size, temperature change, intensifying, etc.
VERB_COGNITION = "verb.cognition"  # verbs of thinking, judging, analyzing, doubting
VERB_COMMUNICATION = "verb.communication"  # verbs of telling, asking, ordering, singing
VERB_COMPETITION = "verb.competition"  # verbs of fighting, athletic activities
VERB_CONSUMPTION = "verb.consumption"  # verbs of eating and drinking
VERB_CONTACT = "verb.contact"  # verbs of touching, hitting, tying, digging
VERB_CREATION = "verb.creation"  # verbs of sewing, baking, painting, performing
VERB_EMOTION = "verb.emotion"  # verbs of feeling
VERB_MOTION = "verb.motion"  # verbs of walking, flying, swimming
VERB_PERCEPTION = "verb.perception"  # verbs of seeing, hearing, feeling
VERB_POSSESSION = "verb.possession"  # verbs of buying, selling, owning
VERB_SOCIAL = "verb.social"  # verbs of political and social activities and events
VERB_STATIVE = "verb.stative"  # verbs of being, having, spatial relations
VERB_WEATHER = "verb.weather"  # verbs of raining, snowing, thawing, thundering

# =============================================================================
# Adjective and Adverb Supersenses
# =============================================================================

ADJ_ALL = "adj.all"  # all adjective clusters
ADJ_PPL = "adj.ppl"  # participial adjectives
ADV_ALL = "adv.all"  # all adverbs

# =============================================================================
# Special Categories
# =============================================================================

SUPERSENSE_UNKNOWN = "unknown"  # for words not in WordNet

# =============================================================================
# Collections
# =============================================================================

NOUN_SUPERSENSES = frozenset(
    [
        NOUN_ACT,
        NOUN_ANIMAL,
        NOUN_ARTIFACT,
        NOUN_ATTRIBUTE,
        NOUN_BODY,
        NOUN_COGNITION,
        NOUN_COMMUNICATION,
        NOUN_EVENT,
        NOUN_FEELING,
        NOUN_FOOD,
        NOUN_GROUP,
        NOUN_LOCATION,
        NOUN_MOTIVE,
        NOUN_OBJECT,
        NOUN_PERSON,
        NOUN_PHENOMENON,
        NOUN_PLANT,
        NOUN_POSSESSION,
        NOUN_PROCESS,
        NOUN_QUANTITY,
        NOUN_RELATION,
        NOUN_SHAPE,
        NOUN_STATE,
        NOUN_SUBSTANCE,
        NOUN_TIME,
    ]
)

VERB_SUPERSENSES = frozenset(
    [
        VERB_BODY,
        VERB_CHANGE,
        VERB_COGNITION,
        VERB_COMMUNICATION,
        VERB_COMPETITION,
        VERB_CONSUMPTION,
        VERB_CONTACT,
        VERB_CREATION,
        VERB_EMOTION,
        VERB_MOTION,
        VERB_PERCEPTION,
        VERB_POSSESSION,
        VERB_SOCIAL,
        VERB_STATIVE,
        VERB_WEATHER,
    ]
)

ADJ_SUPERSENSES = frozenset([ADJ_ALL, ADJ_PPL])

ADV_SUPERSENSES = frozenset([ADV_ALL])

ALL_SUPERSENSES = NOUN_SUPERSENSES | VERB_SUPERSENSES | ADJ_SUPERSENSES | ADV_SUPERSENSES

# =============================================================================
# Mappings
# =============================================================================

# WordNet lexname (lexicographer file name) to supersense
# The lexname is what synset.lexname() returns
# This is essentially an identity mapping since lexnames ARE supersenses
LEXNAME_TO_SUPERSENSE: dict[str, str] = {ss: ss for ss in ALL_SUPERSENSES}

# Add unknown as fallback
LEXNAME_TO_SUPERSENSE[SUPERSENSE_UNKNOWN] = SUPERSENSE_UNKNOWN

# Special WordNet lexnames that need mapping:
# - noun.Tops: top-level abstract nouns (entity, abstraction, etc.)
#   We map these to the most specific supersense based on synset name
# - adj.pert: pertaining adjectives (relational adjectives like "American", "legal")
#   We map these to adj.all since they are still adjectives
LEXNAME_TO_SUPERSENSE["noun.Tops"] = NOUN_OBJECT  # Generic fallback for top-level nouns
LEXNAME_TO_SUPERSENSE["adj.pert"] = ADJ_ALL  # Pertaining adjectives → all adjectives

# POS tag to valid supersenses (for validation)
POS_TO_SUPERSENSES: dict[str, frozenset[str]] = {
    "n": NOUN_SUPERSENSES,
    "v": VERB_SUPERSENSES,
    "a": ADJ_SUPERSENSES,
    "s": ADJ_SUPERSENSES,  # satellite adjectives
    "r": ADV_SUPERSENSES,
}

# Human-readable descriptions for supersenses
SUPERSENSE_DESCRIPTIONS: dict[str, str] = {
    # Nouns
    NOUN_ACT: "acts or actions",
    NOUN_ANIMAL: "animals",
    NOUN_ARTIFACT: "man-made objects",
    NOUN_ATTRIBUTE: "attributes of people and objects",
    NOUN_BODY: "body parts",
    NOUN_COGNITION: "cognitive processes and contents",
    NOUN_COMMUNICATION: "communicative processes and contents",
    NOUN_EVENT: "natural events",
    NOUN_FEELING: "feelings and emotions",
    NOUN_FOOD: "foods and drinks",
    NOUN_GROUP: "groupings of people or objects",
    NOUN_LOCATION: "spatial positions",
    NOUN_MOTIVE: "goals",
    NOUN_OBJECT: "natural objects (not man-made)",
    NOUN_PERSON: "people",
    NOUN_PHENOMENON: "natural phenomena",
    NOUN_PLANT: "plants",
    NOUN_POSSESSION: "possession and transfer",
    NOUN_PROCESS: "natural processes",
    NOUN_QUANTITY: "quantities and units",
    NOUN_RELATION: "relations between people/things",
    NOUN_SHAPE: "two and three dimensional shapes",
    NOUN_STATE: "stable states of affairs",
    NOUN_SUBSTANCE: "substances",
    NOUN_TIME: "time and temporal relations",
    # Verbs
    VERB_BODY: "grooming, dressing, bodily care",
    VERB_CHANGE: "size, temperature change, intensifying",
    VERB_COGNITION: "thinking, judging, analyzing",
    VERB_COMMUNICATION: "telling, asking, ordering, singing",
    VERB_COMPETITION: "fighting, athletic activities",
    VERB_CONSUMPTION: "eating and drinking",
    VERB_CONTACT: "touching, hitting, tying, digging",
    VERB_CREATION: "sewing, baking, painting, performing",
    VERB_EMOTION: "feeling",
    VERB_MOTION: "walking, flying, swimming",
    VERB_PERCEPTION: "seeing, hearing, feeling",
    VERB_POSSESSION: "buying, selling, owning",
    VERB_SOCIAL: "political and social activities",
    VERB_STATIVE: "being, having, spatial relations",
    VERB_WEATHER: "raining, snowing, thawing",
    # Adjectives and adverbs
    ADJ_ALL: "adjectives",
    ADJ_PPL: "participial adjectives",
    ADV_ALL: "adverbs",
    # Special
    SUPERSENSE_UNKNOWN: "unknown (not in WordNet)",
}


def get_supersense(lexname: str) -> str:
    """
    Get supersense from WordNet lexname.

    Args:
        lexname: The lexicographer file name from synset.lexname()

    Returns:
        The supersense string, or SUPERSENSE_UNKNOWN if not found
    """
    return LEXNAME_TO_SUPERSENSE.get(lexname, SUPERSENSE_UNKNOWN)


def is_valid_supersense(supersense: str) -> bool:
    """Check if a string is a valid supersense."""
    return supersense in ALL_SUPERSENSES or supersense == SUPERSENSE_UNKNOWN
