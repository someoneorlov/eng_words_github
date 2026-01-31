# Repository cleanup and push plan (GitHub, other machine)

**Goal:** Keep the current project as-is in Russian. Create a **new** project by copying the repo, then in the copy: remove data/archive, translate everything to English, and push to GitHub with **no history** (single initial commit).

---

## Workflow: copy → clean → translate → fresh repo → push

- **Current project (this folder):** Leave **completely unchanged**. No edits, no git changes. It remains your full Russian project with data and history.
- **New project:** A separate folder (e.g. `eng_words_github` or `eng_words_en`). Created by copying this project, then in the copy only: delete excluded paths, update .gitignore, translate to English, `git init`, one initial commit, push to a new GitHub repo. Result: new repo with zero history.

Steps below (sections 1–4, 7) apply **only to the new copy**, not to the current project.

---

## 1. What to EXCLUDE in the new copy (do not push)

| Category | Paths / patterns | Reason |
|----------|------------------|--------|
| **Data** | `data/` (all contents) | Parquet, JSONL, books, experiment results — you’ll move as archive separately |
| **Logs & reports** | `logs/`, `reports/` | Temporary outputs |
| **Archive docs** | `docs/archive/` | Old plans/results; not needed for refactoring |
| **Books / raw inputs** | `*.epub`, `*.parquet`, `*.csv` (except fixtures) | Already in `.gitignore`; ensure no tracked |
| **Secrets / env** | `.env`, `known_words.csv` | Already in `.gitignore` |
| **Large caches** | `data/synset_cards/llm_cache/`, `backups/` | Already in `.gitignore` |
| **Batch outputs** | `data/experiment/batch/` (requests/results JSONL) | Generated; can re-run on other machine |
| **Gold labels / WSD data** | `data/wsd_gold/*.jsonl`, `data/wsd_gold/labels*/`, `data/wsd_gold/batch_*` | Not needed for refactoring; see below |

**In the new copy:** Either do not copy these paths when copying the project, or delete them in the copy before `git init`. Also add them to `.gitignore` in the copy so they never get committed.

**Gold dataset:** Tests do **not** depend on `data/wsd_gold/` — they use temporary files (`tmp_path / "gold.jsonl"`) and in-memory data. For refactoring you only need the code and the documented JSONL format; the full gold dataset does not need to be in the repo. If later you want to run WSD eval on the VM (e.g. `evaluate_wsd_on_gold("data/wsd_gold/gold_dev.jsonl", ...)`), copy `gold_dev.jsonl` (or similar) from your backup into the new copy’s `data/wsd_gold/` after clone. So: **do not push the golden dataset**; it is not required for refactoring or for tests to pass.

---

## 2. What to KEEP in the new copy (push to GitHub)

| Category | Contents |
|----------|----------|
| **Source** | `src/eng_words/` (including `_archive` if you use it for reference) |
| **Tests** | `tests/` (and `tests/fixtures/` — keep small JSON/CSV used by tests) |
| **Scripts** | `scripts/` (except heavy one-off scripts in `_archive` — optional keep/remove) |
| **Active docs** | `docs/` **excluding** `docs/archive/` (see list below) |
| **Config / project** | `.env.example`, `known_words.csv.example`, `Makefile`, `pyproject.toml`, `uv.lock` |
| **Root docs** | `README.md`, `QUICK_START.md`, `QUICK_START_GSHEETS.md`, `START_GENERATION.sh` |
| **Cursor rules** | `.cursor/rules/` (optional; useful for AI-assisted refactoring) |

**Active docs to keep (then translate to English):**

- `docs/WORD_FAMILY_PIPELINE.md` — main pipeline/experiment doc  
- `docs/QUALITY_FILTERING_PLAN.md`  
- `docs/REFACTOR_AND_BACKUP_PLAN.md`  
- `docs/DEVELOPMENT_HISTORY.md`  
- `docs/GENERATION_INSTRUCTIONS.md`  
- `docs/LLM_API_KEYS_SETUP.md`  
- `docs/CREDENTIALS_EXPLANATION.md`  
- `docs/GOOGLE_SHEETS_SETUP.md`  
- `docs/WSD_GOLD_DATASET_USAGE.md`  
- `docs/BACKLOG_IDEAS.md`  
- Pricing (reference): `docs/google_pricing.md`, `docs/claude_pricing.md`, `docs/openai_pricing.txt`  

Remove or don’t track: `docs/archive/` (entire tree).

---

## 3. .gitignore changes (add/ensure)

Add or confirm these so nothing below gets committed:

```gitignore
# Data and generated outputs (do not push)
data/
logs/
reports/

# Archive documentation (do not push)
docs/archive/
```

Keep existing ignores for: `*.parquet`, `*.csv` (except fixtures), `.env`, `*.epub`, `backups/`, `data/synset_cards/llm_cache/`, etc.

---

## 4. Translation to English

### 4.1 Documentation (Markdown)

- **README.md** — full translation (currently mixed RU/EN).  
- **QUICK_START.md**, **QUICK_START_GSHEETS.md** — translate any Russian.  
- **docs/** (all kept files above) — translate to English:  
  - `WORD_FAMILY_PIPELINE.md` (most content is RU),  
  - `QUALITY_FILTERING_PLAN.md`,  
  - `REFACTOR_AND_BACKUP_PLAN.md`,  
  - `DEVELOPMENT_HISTORY.md`,  
  - `GENERATION_INSTRUCTIONS.md`,  
  - `BACKLOG_IDEAS.md`,  
  - and any other kept docs that still have Russian.

Keep technical terms consistent (e.g. “lemma”, “synset”, “pipeline”, “batch API”).

### 4.2 Docstrings and code comments

- **src/eng_words/** — all module/class/function docstrings in English.  
- Inline comments — English.  
- **tests/** — docstrings and comments in English.  
- **scripts/** (non-archived) — docstrings and comments in English.

You can do this in passes: first `src/eng_words/` and `docs/`, then `tests/` and `scripts/`.

---

## 5. Order of operations (all steps in the **new copy** only)

1. **Copy the project**  
   - Copy the entire project folder to a new directory (e.g. `../eng_words_github` or `../eng_words_en`).  
   - Do **not** copy the `.git` folder (so the copy has no history).  
   - Optionally: exclude `data/`, `logs/`, `reports/`, `docs/archive/` from the copy so they are not present at all.

2. **In the new copy: remove excluded paths (if they were copied)**  
   - Delete: `data/`, `logs/`, `reports/`, `docs/archive/`.  
   - Keep: `tests/fixtures/` and any small test data under `tests/` if needed.

3. **In the new copy: update .gitignore**  
   - Add or confirm: `data/`, `logs/`, `reports/`, `docs/archive/`.  
   - Keep existing ignores (e.g. `*.parquet`, `.env`, `*.epub`, etc.).

4. **In the new copy: translate to English**  
   - Translate README and root Markdown (QUICK_START, etc.).  
   - Translate all kept docs under `docs/`.  
   - Translate docstrings and comments in `src/`, `tests/`, and active `scripts/`.

5. **In the new copy: create fresh repo and push**  
   - `cd <new-copy>`  
   - `git init`  
   - `git add .`  
   - `git commit -m "Initial commit: English Words pipeline (refactor-ready)"`  
   - Create a new empty repo on GitHub (no README, no .gitignore).  
   - `git remote add origin <github-url>`  
   - `git branch -M main`  
   - `git push -u origin main`  

   Result: one commit, no history from the original project.

6. **Verify**  
   - In the new copy: run tests (e.g. `uv run pytest` or `make test`).  
   - After push: clone the new repo elsewhere and confirm docs/code are in English and nothing sensitive or large is present.

---

## 6. Copy command (example)

From the parent of your current project:

```bash
# Copy project but exclude .git and optionally big folders
rsync -a --exclude='.git' --exclude='data' --exclude='logs' --exclude='reports' --exclude='docs/archive' /path/to/eng_words /path/to/eng_words_github
```

Or: copy the whole folder, then in the copy delete `.git`, `data/`, `logs/`, `reports/`, `docs/archive/` manually.  
Keep `tests/fixtures/` in the copy so tests can run (small JSON/CSV used by tests).

---

## 7. Checklist before push (in the new copy)

- [ ] `.gitignore` includes `data/`, `logs/`, `reports/`, `docs/archive/`.  
- [ ] No `data/`, `logs/`, `reports/`, or `docs/archive/` in `git status`.  
- [ ] README and all kept docs are in English.  
- [ ] Docstrings and comments in `src/`, `tests/`, active `scripts/` are in English.  
- [ ] `.env` and `known_words.csv` are not tracked; examples (`.example`) are.  
- [ ] Tests pass.  
- [ ] Backup/archive of data and archive docs exists if you need it on the other machine.

After this, the **new copy** is the only thing pushed to GitHub (single initial commit, no history). Your **current project** stays untouched in Russian with all data and history. You can do the copy and cleanup on this machine or on the VM; the VM can then do the translation and push.
