"""Default values for functions and processing."""

# Language code
LANGUAGE_EN = "en"

# File encoding
ENCODING_UTF8 = "utf-8"

# Text processing defaults
MAX_CHARS_DEFAULT = 250_000
SPACY_MAX_LENGTH = 2_500_000

# Example sentence selection defaults
EXAMPLE_MIN_LENGTH = 50
EXAMPLE_MAX_LENGTH = 150
EXAMPLE_FALLBACK_MIN = 20
EXAMPLE_FALLBACK_MAX = 300

# Number of examples to attach per item by default
EXAMPLES_PER_ITEM_DEFAULT = 3

# Frequency filtering defaults
MIN_BOOK_FREQ_DEFAULT = 3
MIN_ZIPF_DEFAULT = 2.0
MAX_ZIPF_DEFAULT = 5.5

# Ranking defaults
TARGET_ZIPF_DEFAULT = 4.0

# Phrasal verb filtering defaults
PHRASAL_MIN_FREQ_DEFAULT = 2

# Top N defaults
TOP_N_DEFAULT = 100

# Messages
MSG_NO_EXAMPLE = "No example yet."

# Dialect/tokenization error lemmas to filter out
# These are common tokenization errors from non-standard English (dialects, slang)
# e.g., "ain't" → "ai" + "n't", "de" (dialect "the"), etc.
DIALECT_LEMMAS_FILTER = frozenset({
    "ai",    # ain't → ai (tokenization error)
    "de",    # dialect "the"
    "dis",   # dialect "this"
    "dat",   # dialect "that"
    "dey",   # dialect "they"
    "dem",   # dialect "them"
    "wid",   # dialect "with"
    "wuz",   # dialect "was"
    "kin",   # dialect "can" (also valid word, but rare in this context)
    "fer",   # dialect "for"
    "yer",   # dialect "your"
    "ta",    # dialect "to"
    "ter",   # dialect "to"
    "git",   # dialect "get"
    "goin",  # dialect "going"
    "doin",  # dialect "doing"
    "nothin",  # dialect "nothing"
    "somethin",  # dialect "something"
})
