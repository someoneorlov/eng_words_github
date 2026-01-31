# Onboarding: Project Overview & Refactor Context

Short reference for an LLM agent (or human) picking up this repo on a **different machine** (e.g. VM with Vertex AI). What the project is, where to look, what to copy, and how LLM access works.

---

## 1. What this project is

**English Words** — a pipeline that turns **books** (epub/text) into **Anki flashcards** for English learning (B1–B2).

- **Input:** Text from books (tokens, sentences, frequency stats).
- **Output:** JSON/CSV of “smart cards”: lemma, definition, translation, 3 examples (from book + generated).
- **Core:** LLM is used for: sense disambiguation (WSD), synset aggregation, example validation, card generation (definitions, translations, example selection).

High-level flow:

```
Book text → Tokens/Stats → WSD (sense per word) → Synset aggregation → Card generation (LLM) → Anki export
```

Details: `docs/WORD_FAMILY_PIPELINE.md`, `docs/QUALITY_FILTERING_PLAN.md`, `README.md`.

---

## 2. Where to look (structure)

| Area | Path | What it is |
|------|------|------------|
| **Pipeline entry** | `scripts/run_synset_card_generation.py` | Main script: aggregation + card generation. |
| **LLM layer** | `src/eng_words/llm/` | `base.py` (interface), `providers/` (OpenAI, Anthropic, **Gemini**), `response_cache.py`, `smart_card_generator.py`, `retry.py`. |
| **Aggregation** | `src/eng_words/aggregation/` | LLM groups WordNet synsets into one card per meaning. |
| **WSD** | `src/eng_words/wsd/` | Word sense disambiguation (LLM + WordNet). |
| **Validation** | `src/eng_words/validation/` | Validates book examples for a synset (LLM). |
| **Config / pricing** | `src/eng_words/constants/llm_pricing.py`, `defaults.py`, `files.py` | Models, paths, cost estimation. |
| **Docs** | `docs/` | `WORD_FAMILY_PIPELINE.md`, `QUALITY_FILTERING_PLAN.md`, `REFACTOR_AND_BACKUP_PLAN.md`, `GENERATION_INSTRUCTIONS.md`, `LLM_API_KEYS_SETUP.md`. |
| **Archive** | `src/eng_words/_archive/`, `scripts/_archive/`, `tests/_archive/`, `docs/archive/` | Old code/scripts/docs; safe to ignore during refactor unless needed. |

**Current “result” of the pipeline:** generated cards live in data outputs (e.g. `synset_smart_cards_final.json`). This repo does **not** commit `data/`, `logs/`, `reports/` (see `.gitignore`). So “current result” exists only on the machine where the pipeline was run (or in backups). For refactoring you only need the code and tests.

---

## 3. What to copy when moving to the other machine

| Need | What to copy |
|------|----------------|
| **Only refactor (no runs)** | Nothing. Clone repo + `uv sync`; tests use mocks. |
| **Run pipeline on VM (Vertex)** | Input data (see §3.1) + Vertex env (§5). No cache. |
| **Run pipeline on VM (no API, from cache)** | Input data + **LLM cache** (§3.2). Same prompts → cache hits → no calls. |

**For refactoring you don’t need to copy any data.** The repo (code + tests) is self-contained; tests mock LLM calls.

### 3.1 Run full pipeline on a small sample (what to copy)

The main script is `scripts/run_synset_card_generation.py`. It expects:

| What | Path | Description |
|------|------|--------------|
| **Aggregated cards** | `data/synset_aggregation_full/aggregated_cards.parquet` | Pre-computed synset groups (from aggregation step). You can trim to first N rows for a small sample. |
| **Tokens (book)** | `data/processed/american_tragedy_tokens.parquet` | Book tokens to reconstruct sentences for examples. Script has `BOOK_NAME = "american_tragedy"` and `TOKENS_PATH`; keep paths consistent. |

Create these dirs on the VM and copy the files (or a subset). Run with a limit, e.g. `uv run python scripts/run_synset_card_generation.py 20` (first argument = max number of cards).

### 3.2 Two ways to run: with Vertex vs without API (replay cache)

**Option A — With Vertex AI (real LLM calls)**  
- Copy only the **input data** above (e.g. `aggregated_cards.parquet` + `american_tragedy_tokens.parquet` into `data/synset_aggregation_full/` and `data/processed/`).  
- Set Vertex env (§5).  
- Run the script (e.g. with a small `limit`). No cache needed; every prompt will call Vertex.

**Option B — Without any API (replay from cache)**  
- Copy the **same input data** as above (same small sample, e.g. first 20 cards).  
- Copy the **LLM cache** from the machine where you already ran that same sample:  
  **`data/synset_cards/llm_cache/`** — this is the dir used by `run_synset_card_generation.py` (see `CACHE_DIR = OUTPUT_DIR / "llm_cache"` → `data/synset_cards/llm_cache`).  
- On the VM, run the script with the same limit (e.g. 20). The prompts will be identical → cache key = hash(model, prompt, temperature) will match → cache hits, no LLM calls.

So: **to run the full pipeline on a small sample without calling the LLM**, copy (1) the input parquet files for that sample, and (2) the contents of `data/synset_cards/llm_cache/` from a run that used that same sample. You can also copy the whole cache from a larger run; extra files don’t hurt, and the needed keys will be there.

---

## 4. What we want to do (refactor goals)

- **Stabilise and simplify** the codebase for future work (no big feature scope in “refactor”).
- **Keep behaviour:** same inputs → same outputs; tests (with mocks) must stay green.
- **Single LLM boundary:** all real calls go through `LLMProvider` (`complete` / `complete_json`). Refactor should not change that contract; only internal structure, naming, file layout, docs.
- **Reference:** `docs/REFACTOR_AND_BACKUP_PLAN.md` describes the last cleanup (archives, script count, doc count). Use it to understand what’s “active” vs archived.

No need to refactor archive code unless you have a concrete reason.

---

## 5. LLM access: this machine vs the other (Vertex AI)

**Current machine (e.g. your laptop):**  
We use **Google AI Studio** (Gemini API) with an **API key** in `GOOGLE_API_KEY`.  
Code path: `src/eng_words/llm/providers/gemini.py` → `genai.Client(api_key=...)`.

**Other machine (e.g. VM):**  
You want to use **Vertex AI** (same Gemini models, but GCP project, no API key; auth via Application Default Credentials).

**Where it’s implemented:**

- **File:** `src/eng_words/llm/providers/gemini.py`
- **Constructor:** `GeminiProvider(..., vertexai=True, project="...", location="...")` or via env.
- **Env vars for Vertex (no API key):**
  - `GOOGLE_GENAI_USE_VERTEXAI=true`
  - `GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>`
  - `GOOGLE_CLOUD_LOCATION=us-central1` (or your region)
- **Factory:** `get_provider("gemini", vertexai=True, project="...", location="...")` in `src/eng_words/llm/base.py`.

So: **same codebase;** on the VM set the Vertex env vars (or pass `vertexai=True` + project/location), and the same pipeline will call Gemini via Vertex AI. No second “Google” backend — it’s one provider with two ways to create the client (API key vs Vertex).

---

## 6. Quick checklist for the other machine

1. Clone the repo (this GitHub repo).
2. Install deps: `uv sync` (or `pip install -e ".[llm]"` etc. as in README).
3. For Vertex AI: set `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`; ensure ADC works (e.g. service account or `gcloud auth application-default login`).
4. Run tests: `pytest` (they use mocks; no LLM calls).
5. Optionally run one pipeline: e.g. `scripts/run_synset_card_generation.py` with a small input to confirm Vertex works.

After that you can refactor with confidence: same interfaces, tests green, LLM behind one abstraction and configurable per environment.
