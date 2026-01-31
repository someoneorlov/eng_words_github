# Architecture Overview

This document provides a high-level overview of the English Words pipeline architecture, designed to help developers understand the codebase structure and data flow.

## High-Level Flow

```
┌─────────────┐
│  EPUB Book  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Stage 1: Tokenization & Statistics │
│  - Extract text from EPUB           │
│  - Tokenize & lemmatize (spaCy)     │
│  - Calculate frequency stats        │
│  - Filter by Zipf/known words       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  WSD: Word Sense Disambiguation     │
│  - Annotate tokens with WordNet     │
│    synsets (optional)               │
│  - Aggregate by supersense          │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Aggregation: Synset Grouping       │
│  - Group WordNet synsets by meaning │
│  - LLM-based semantic clustering    │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Card Generation                    │
│  - Validate examples                │
│  - Generate definitions/translations │
│  - Select & filter examples         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Anki CSV   │
└─────────────┘
```

## Module Responsibilities

### Core Pipeline (`src/eng_words/pipeline.py`)
- **Stage 1 orchestration**: Coordinates tokenization, statistics calculation, and filtering
- **Full pipeline**: Runs complete flow from EPUB to Anki CSV (optional)
- **Entry point**: `python -m eng_words.pipeline` CLI

### LLM Layer (`src/eng_words/llm/`)
- **Providers** (`llm/providers/`): Abstraction for Gemini, OpenAI, Anthropic APIs
- **Smart Card Generator** (`smart_card_generator.py`): Generates flashcards with definitions, translations, examples
- **Response Cache** (`response_cache.py`): Caches LLM responses to reduce API costs
- **Retry Logic** (`retry.py`): Handles transient failures and rate limits

### Word Sense Disambiguation (`src/eng_words/wsd/`)
- **WordNet Backend** (`wordnet_backend.py`): Annotates tokens with WordNet synsets
- **LLM WSD** (`llm_wsd.py`): Uses LLM for disambiguation when WordNet is ambiguous
- **Candidate Selector** (`candidate_selector.py`): Selects best synset candidates using embeddings
- **Aggregator** (`aggregator.py`): Aggregates sense statistics by supersense

### Aggregation (`src/eng_words/aggregation/`)
- **Synset Aggregator** (`synset_aggregator.py`): Groups WordNet synsets by frequency/statistics
- **LLM Aggregator** (`llm_aggregator.py`): Uses LLM to semantically group synsets into meaning families

### Validation (`src/eng_words/validation/`)
- **Example Validator** (`example_validator.py`): Validates that book examples match synset meanings
- **Synset Validator** (`synset_validator.py`): Validates examples for synset groups using LLM

### Storage (`src/eng_words/storage/`)
- **Backends** (`backends.py`): CSV and Google Sheets backends for known words
- **Loader** (`loader.py`): Unified interface for loading/saving known words

### Constants (`src/eng_words/constants/`)
- **Configuration**: Model names, file templates, column names, defaults
- **LLM Pricing** (`llm_pricing.py`): Cost estimation for different providers/models
- **Supersenses** (`supersenses.py`): WordNet supersense taxonomy

### Supporting Modules
- **Text Processing** (`text_processing.py`): Tokenization, sentence reconstruction, spaCy integration
- **Statistics** (`statistics.py`): Frequency calculation, Zipf scoring
- **Filtering** (`filtering.py`): Filter by frequency, known words, supersense
- **Examples** (`examples.py`): Extract and select example sentences from book
- **Anki Export** (`anki_export.py`): Convert cards to Anki CSV format
- **EPUB Reader** (`epub_reader.py`): Extract text from EPUB files

## Data Flow

### Input
- **Raw books**: `data/raw/*.epub` - EPUB files containing source material
- **Known words**: `data/known_words.csv` or Google Sheets URL - Words already learned

### Intermediate Files (`data/processed/`)
- `{book_name}_tokens.parquet` - Tokenized text with lemmas, POS tags, sentence IDs
- `{book_name}_lemma_stats.parquet` - Frequency statistics per lemma
- `{book_name}_sense_tokens.parquet` - Tokens annotated with WordNet synsets (if WSD enabled)
- `{book_name}_supersense_stats.parquet` - Aggregated statistics by supersense

### Output
- **Synset cards**: `data/synset_cards/synset_smart_cards_final.json` - Generated flashcards in JSON format
- **Anki CSV**: `data/synset_cards/synset_anki.csv` or `anki_exports/` - Import-ready Anki deck
- **LLM cache**: `data/synset_cards/llm_cache/` - Cached LLM responses

## Entry Points

### Main Scripts

**Full Card Generation** (`scripts/run_synset_card_generation.py`):
```bash
uv run python scripts/run_synset_card_generation.py [limit]
```
- Loads aggregated synset cards
- Generates smart cards with LLM
- Validates examples
- Exports to Anki CSV

**Stage 1 Pipeline** (`python -m eng_words.pipeline`):
```bash
uv run python -m eng_words.pipeline \
  --book-path data/raw/book.epub \
  --book-name book_name \
  --output-dir data/processed \
  --known-words data/known_words.csv \
  --enable-wsd
```
- Tokenization and statistics
- Optional WSD annotation
- Example extraction
- Anki export (if using full pipeline)

### Testing
```bash
make check          # Format + lint + test
make test           # Run all tests
make test-wsd       # Run WSD tests only
```

### Other Scripts
- `scripts/eval_wsd_on_gold.py` - Evaluate WSD accuracy on gold dataset
- `scripts/verify_gold_checksum.py` - Verify gold dataset integrity
- `scripts/benchmark_wsd.py` - Benchmark WSD performance

## Key Design Decisions

1. **Modular Architecture**: Each stage (tokenization, WSD, aggregation, generation) is independent and can be run separately
2. **LLM Abstraction**: Provider-agnostic interface allows switching between Gemini, OpenAI, Anthropic
3. **Caching Strategy**: LLM responses are cached to reduce API costs and enable resumable generation
4. **Parquet Format**: Intermediate data stored as Parquet for efficient I/O and resumability
5. **Validation Pipeline**: Multi-stage validation ensures card quality (synset matching, example length, spoiler detection)

## Dependencies

- **spaCy**: Tokenization, POS tagging, lemmatization
- **WordNet**: Word sense disambiguation
- **sentence-transformers**: Embedding-based candidate selection (optional)
- **LLM APIs**: Gemini, OpenAI, Anthropic (via provider abstraction)
- **pandas**: Data manipulation and Parquet I/O
