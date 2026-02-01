"""File extension and path template constants."""

# File extensions
EXT_PARQUET = ".parquet"
EXT_CSV = ".csv"

# File name templates (used with f-strings: f"{book_name}{TEMPLATE_TOKENS}")
TEMPLATE_TOKENS = "_tokens.parquet"
TEMPLATE_LEMMA_STATS = "_lemma_stats.parquet"
TEMPLATE_LEMMA_STATS_FULL = "_lemma_stats_full.parquet"
TEMPLATE_PHRASAL_VERBS = "_phrasal_verbs.parquet"
TEMPLATE_PHRASAL_VERB_STATS = "_phrasal_verb_stats.parquet"
TEMPLATE_ANKI = "_anki.csv"

# Directory names
DIR_ANKI_EXPORTS = "anki_exports"
