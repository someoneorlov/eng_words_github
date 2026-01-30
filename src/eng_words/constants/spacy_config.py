"""spaCy model and component constants."""

# Default model name
DEFAULT_MODEL_NAME = "en_core_web_sm"

# spaCy component names
COMPONENT_PARSER = "parser"
COMPONENT_NER = "ner"
COMPONENT_TEXTCAT = "textcat"
COMPONENT_SENTER = "senter"

# Disabled components for tokenization (no parser needed)
TOKENIZATION_DISABLED = [COMPONENT_PARSER, COMPONENT_NER, COMPONENT_TEXTCAT]

# Disabled components for phrasal verb detection (parser needed)
PHRASAL_DISABLED = [COMPONENT_NER, COMPONENT_TEXTCAT]

# POS tags
POS_VERB = "VERB"
POS_PROPN = "PROPN"

# Dependency labels
DEP_PRT = "prt"
