"""DataFrame column name constants."""

# Token columns
BOOK = "book"
SENTENCE_ID = "sentence_id"
POSITION = "position"
SURFACE = "surface"
LEMMA = "lemma"
POS = "pos"
IS_STOP = "is_stop"
IS_ALPHA = "is_alpha"
WHITESPACE = "whitespace"

# Statistics columns
BOOK_FREQ = "book_freq"
DOC_COUNT = "doc_count"
GLOBAL_ZIPF = "global_zipf"
SCORE = "score"
VERB_COUNT = "verb_count"
OTHER_POS_COUNT = "other_pos_count"
STOPWORD_COUNT = "stopword_count"

# Phrasal verb columns
PHRASAL = "phrasal"
VERB = "verb"
PARTICLE = "particle"
SENTENCE_TEXT = "sentence_text"

# Known words columns
STATUS = "status"
ITEM_TYPE = "item_type"
TAGS = "tags"

# WSD (Word Sense Disambiguation) columns
SYNSET_ID = "synset_id"
SUPERSENSE = "supersense"
SENSE_CONFIDENCE = "sense_confidence"
DEFINITION = "definition"
SENSE_FREQ = "sense_freq"
SENSE_RATIO = "sense_ratio"
SENSE_COUNT = "sense_count"
DOMINANT_SUPERSENSE = "dominant_supersense"

# Anki export columns
FRONT = "front"
BACK = "back"
EXAMPLE = "example"

# Sentence columns
SENTENCE = "sentence"

# Column groups (for validation and DataFrame creation)
REQUIRED_TOKEN_COLUMNS = [
    LEMMA,
    SENTENCE_ID,
    IS_STOP,
    POS,
    IS_ALPHA,
]

TOKEN_COLUMNS = [
    BOOK,
    SENTENCE_ID,
    POSITION,
    SURFACE,
    LEMMA,
    POS,
    IS_STOP,
    IS_ALPHA,
    WHITESPACE,
]

SENTENCE_COLUMNS = {SENTENCE_ID, SENTENCE}

REQUIRED_KNOWN_WORDS_COLUMNS = [LEMMA, STATUS, ITEM_TYPE, TAGS]
# Optional synset_id: when set, filtering is by (lemma, synset_id); when empty, by lemma only (all senses).
KNOWN_WORDS_COLUMNS = [LEMMA, STATUS, ITEM_TYPE, TAGS, SYNSET_ID]

ANKI_COLUMNS = [FRONT, BACK, TAGS]
