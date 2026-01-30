# English Words Learning Tool

A personal tool for extracting words from books, filtering out known ones, and creating Anki flashcards for learning English.

## Project description

This tool helps you learn English through reading books:

1. **Word extraction** from book text files
2. **Frequency analysis** — finds the most frequent words
3. **Filtering**:
   - Excludes words you already know
   - Removes overly frequent (basic) words
   - Removes overly rare or archaic words
4. **Ranking** of candidates for learning
5. **Anki card generation** (with future LLM integration)

## Architecture

The project is built as a modular text-processing pipeline:

```
Book text → Tokenization → Lemmatization → Statistics → Filtering → Ranking → Export
```

### Key components

- **Tokenization and lemmatization**: Uses spaCy for text processing
- **Frequency analysis**: Combination of local frequency (in the book) and global frequency (in the language)
- **Filtering**: Excluding known words via CSV or Google Sheets
- **Phrasal verbs**: Separate handling of phrasal verbs as a distinct entity type
- **Export**: CSV generation for Anki import

## High-level plan

### Phase 1: Basic text processing
- Tokenization and lemmatization with spaCy
- Word statistics extraction
- Saving intermediate results (parquet)

### Phase 2: Filtering and ranking
- Integration with global frequency (wordfreq)
- Filtering by known words (CSV)
- Candidate ranking

### Phase 3: Phrasal verbs
- Phrasal verb detection via dependency parsing
- Processing and filtering of phrasal verbs

### Phase 4: Export and examples
- Sentence example extraction for each word
- Export to Anki format (CSV)

### Phase 5: LLM integration (future)
- Definition and translation generation via LLM API
- Improved Anki cards with context

## Development principles

### 1. Modularity and testability
- **Everything split into functions** — no monolithic scripts
- **Each function has a single responsibility**
- **Functions are easy to test in isolation**

### 2. Test coverage
- **Every function is covered by tests**
- **Integration between functions is also tested**
- **Tests are written in parallel with the code**

### 3. Incremental development
- **Development proceeds in small logical blocks**
- **After each block: testing and debugging**
- **Only after a block works stably do we move to the next**
- **After each block**: `git status` → `git add` → meaningful commit (Git-First Workflow)

### 4. Saving intermediate results
- **Tokens are saved in parquet** for fast access
- **Lemma statistics are saved separately**
- **Allows reusing results without reprocessing**

### 5. Simplicity and clarity
- **Code should be understandable a year from now**
- **Processing pipeline is linear and transparent**
- **Minimal magic, maximum explicitness**
- **Long texts are processed in chunks** (~250k characters) and normalized (quotes, apostrophes, invisible characters)
- **Known words are filtered** from candidates via CSV
- **Frequency filtering and ranking** are built into Stage 1
- **Phrasal verbs** are written to a separate parquet when needed
- **Candidates are filtered by frequency** and ranked by score

## Project structure

```
eng_words/
├── src/eng_words/          # Application code
│   ├── __init__.py
│   ├── text_processing.py  # Tokenization, lemmatization
│   ├── statistics.py       # Frequency counts, statistics
│   ├── filtering.py        # Filtering by known words
│   ├── phrasal_verbs.py    # Phrasal verb handling
│   └── ...
├── tests/                  # Tests
│   ├── __init__.py
│   ├── test_text_processing.py
│   ├── test_statistics.py
│   └── ...
├── data/                   # Data
│   ├── raw/                # Raw book texts
│   └── processed/          # Intermediate results (parquet)
├── anki_exports/           # Exported cards
├── pyproject.toml          # Project configuration
└── README.md
```

## Input data

- Current primary format for books is **EPUB**
- Files live in `data/raw/` (e.g. `data/raw/theodore-dreiser_an-american-tragedy.epub`)
- Support for other formats (PDF, etc.) will be added as needed

## Installation and usage

### Requirements
- Python 3.10+
- uv (for dependency management)

### Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Usage

#### Full pipeline (text → Anki CSV)

```bash
python -m eng_words.pipeline \
  --book-path data/raw/theodore-dreiser_an-american-tragedy.epub \
  --book-name american_tragedy \
  --output-dir data/processed \
  --known-words data/known_words.csv \
  --min-book-freq 3 \
  --min-zipf 2.0 \
  --max-zipf 5.3 \
  --top-n 150 \
  --phrasal-model en_core_web_sm
```

Useful flags:
- `--top-n` — how many lemmas/phrases go into the final Anki CSV
- `--no-phrasals` — disable phrasal verb processing
- `--phrasal-model` — separate spaCy model for phrasals (default: `--model-name`)

Output files:
- `data/processed/american_tragedy_tokens.parquet`
- `data/processed/american_tragedy_lemma_stats_full.parquet` (all tokens with raw `book_freq` and `global_zipf`)
- `data/processed/american_tragedy_lemma_stats.parquet` (filtered candidates with thresholds and `score`)
- `data/processed/american_tragedy_phrasal_verbs.parquet` (if not using `--no-phrasals`)
- `data/processed/american_tragedy_phrasal_verb_stats.parquet` (aggregated statistics)
- `data/processed/anki_exports/american_tragedy_anki.csv` — ready CSV for Anki

### Chunk processing and normalization
- Text is automatically split into chunks (~250k characters) to avoid spaCy limits and save memory
- Before tokenization, normalization is applied: invisible characters (`\ufeff`, `\u200b`, `\u00ad`) are removed; long dashes and “smart” quotes/apostrophes are replaced
- The `max_chars` parameter for chunks can be changed in code (`tokenize_text_in_chunks`) if you need different granularity

### Export for manual review

To update the CSV for review (e.g. before uploading to Google Sheets):

```bash
python scripts/export_for_review.py \
  --lemma-stats data/processed/american_tragedy_lemma_stats_full.parquet \
  --score-source data/processed/american_tragedy_lemma_stats.parquet \
  --status-source data/review_export_all_processed.csv \
  --output data/review_export_all_next.csv
```

- `--lemma-stats` — raw frequencies (all tokens)
- `--score-source` — cleaned parquet with current `score` (by default a neighbouring `*_lemma_stats.parquet` is used)
- `--status-source` — previous CSV with manual labels; statuses and tags are carried over
- If a word ends in `-ing`, is not used as a verb in the corpus, and the base form exists, the export sets `status=ignore` and tag `auto_ing_variant` — such forms can be skipped during review
- If a lemma is marked by spaCy as a stopword, the export adds tag `auto_stopword` for quick filtering (status left empty, can be set manually)

The output is the full list of words and phrases (including stopwords) with correct `book_freq`/`global_zipf` and scores only for “clean” candidates.

### Known words storage

The tool supports two ways to store the list of known words:

#### CSV files (default)
- Format: `lemma,status,item_type,tags` (see `known_words.csv.example`)
- Supported statuses: `known`, `ignore`, `learning`, `maybe` (filter uses the first two)
- Usage: `--known-words data/known_words.csv`

#### Google Sheets (for convenient editing)
- Automatic sync across devices
- Easy editing in the browser
- Usage: `--known-words gsheets://SPREADSHEET_ID/WORKSHEET_NAME`

**Google Sheets setup:**

1. Create a Google Service Account:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a project (or select an existing one)
   - Enable Google Sheets API and Google Drive API
   - Create a Service Account and download the JSON key

2. Configure access:
   - Open the JSON file and copy `client_email`
   - Open your Google Sheet
   - Share the sheet with this email (role: Editor)

3. Set credentials path:
   - Set env var: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json`
   - Or pass the path when creating the backend (in code)

4. Use in CLI:
   ```bash
   python -m eng_words.pipeline \
     --known-words gsheets://YOUR_SPREADSHEET_ID/Sheet1 \
     ...
   ```

**Google Sheets format:** The sheet must have headers: `lemma,status,item_type,tags` (same as CSV). If the worksheet does not exist, it will be created.

### Frequency filtering and ranking
- Book threshold (`--min-book-freq`), global min (`--min-zipf`) and max (`--max-zipf`) help drop too rare or too frequent words
- After filtering, `score` is added: normalized frequency × rarity boost (centered around Zipf ≈ 4.0)
- Output parquet is sorted by `score` (most useful words first)

### Phrasal verb detection
- Flag `--detect-phrasals` enables detection of `verb + particle` via spaCy dependency parsing
- You can specify a separate model (`--phrasal-model`); otherwise `--model-name` is used
- Results:
  - `*_phrasal_verbs.parquet` — found occurrences (`book`, `sentence_id`, `phrasal`, `verb`, `particle`, `sentence_text`)
  - `*_phrasal_verb_stats.parquet` — aggregated stats with known-words filter and ranking

### Word Sense Disambiguation (WSD)

The tool supports disambiguating word meanings via WSD:

```bash
python -m eng_words.pipeline \
  --book-path data/raw/book.epub \
  --book-name my_book \
  --output-dir data/processed \
  --enable-wsd \
  --min-sense-freq 5 \    # Minimum sense frequency
  --max-senses 3          # Max senses per word
```

**What WSD does:**
- Determines the specific sense of a word in context (e.g. "bank" as financial institution vs. river bank)
- Groups senses into 43 categories (supersenses): `noun.person`, `verb.motion`, `verb.social`, etc.
- Lets you filter and study words by sense

**Output files:**
- `{book}_sense_tokens.parquet` — tokens with sense annotations (`synset_id`, `supersense`, `sense_confidence`)
- `{book}_supersense_stats.parquet` — (lemma, supersense) statistics with frequencies and shares

**Export for manual review:**
```bash
python scripts/export_for_review.py \
  --supersense-stats data/processed/my_book_supersense_stats.parquet \
  --sense-tokens data/processed/my_book_sense_tokens.parquet \
  --output review_export_supersenses.csv
```

**Example result:**
| lemma | supersense | sense_freq | definition | example | status |
|-------|------------|------------|------------|---------|--------|
| run | verb.motion | 17 | move fast by using legs | He ran quickly. | |
| run | verb.social | 10 | be in charge of | She runs the company. | |

**Detailed docs:** See [docs/WSD_GOLD_DATASET_USAGE.md](docs/WSD_GOLD_DATASET_USAGE.md) and related docs.

### Smart card generation (LLM-powered)

Automatic generation of Anki cards using an LLM:

```bash
python -m eng_words.pipeline \
  --book-path data/raw/book.epub \
  --book-name my_book \
  --output-dir data/processed \
  --enable-wsd \
  --smart-cards \
  --smart-cards-provider gemini \
  --top-n 500
```

**What the Smart Card Generator does:**
- Picks 1–2 best in-context examples from the book per word
- Detects and excludes examples with the wrong sense (WSD check)
- Generates a simple definition (B1–B2 level)
- Produces a translation for the specific sense
- Generates an extra example sentence

**LLM providers:**
- `gemini` (default) — Gemini 3 Flash
- `openai` — GPT-5-mini
- `anthropic` — Claude Haiku 4.5

**Output files:**
- `{book}_smart_cards.json` — full card data
- `anki_exports/{book}_smart_anki.csv` — ready CSV for Anki

**Example card:**
```json
{
  "lemma": "break",
  "supersense": "verb.change",
  "selected_examples": ["The glass will break if you drop it."],
  "excluded_examples": ["Don't break the rules."],
  "simple_definition": "to separate into pieces",
  "translation_ru": "to break, to smash",
  "generated_example": "Be careful not to break the vase."
}
```

**Requirements:**
- `--enable-wsd` is required for WSD annotations
- Provider API key in `.env` (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- Caching: repeated runs do not make new API calls for the same inputs

## Data model

### Known words storage

Table for your personal vocabulary (CSV or Google Sheets):

| lemma | item_type | status | tags | created_at | last_seen_in_book | examples_count | notes |
|-------|-----------|--------|------|------------|-------------------|----------------|-------|
| run | word | known | A2, basic_verbs | ... | ... | 5 | |
| give up | phrasal_verb | learning | B1, phrasal | ... | ... | 3 | |

Where:
- `status`: `known`, `learning`, `ignore`, `maybe`
- `item_type`: `word`, `phrasal_verb`, `ngram`
- `tags`: levels, books, topics

### Intermediate files

- `tokens.parquet`: all tokens with metadata
- `lemma_stats_full.parquet`: all lemmas with raw `book_freq` and `global_zipf`
- `lemma_stats.parquet`: filtered candidates (filters, known words, `score`)

## Technologies

- **spaCy**: NLP processing, lemmatization, dependency parsing
- **pandas**: Data handling
- **wordfreq**: Global word frequency
- **pyarrow**: Parquet files
- **pytest**: Testing
- **sentence-transformers**: Embeddings for WSD (optional)
- **nltk**: WordNet for word senses (optional)

## License

Personal project for learning English.
