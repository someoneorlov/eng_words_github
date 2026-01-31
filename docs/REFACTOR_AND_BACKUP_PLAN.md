# Safe cleanup and refactoring plan

> **Status:** ✅ COMPLETED (2026-01-21)  
> **Goal:** tidy the repository, preserve artifacts, and keep the pipeline working.

---

## Refactoring summary

### What was done

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| Python files (excluding archive) | 150 | 110 | -40 |
| Scripts | 51 | 10 | -41 |
| Documents | 60+ | 12 | -48 |
| Data | ~200 MB | 156 MB | -44 MB |

### Commits (8)

```
190fa5f chore: archive 41 more scripts, cleanup temp data
60684fb add cursor rules
fbdf51c chore: update .gitignore with temp files
7e5b06a test: add tests for new modules
453f3c5 feat: add smart card generation pipeline with Stage 2.5 filtering
93e5171 chore: cleanup tracked files, update modules
063e840 docs: archive 55 outdated documents, add new plans
e263b21 refactor: archive deprecated modules and scripts
```

---

## Final project structure

### Scripts (10)

| Script | Purpose |
|--------|---------|
| `run_synset_card_generation.py` | Main card generation pipeline |
| `compare_cards.py` | Result comparison (regression tests) |
| `run_full_generation.sh` | Run full generation |
| `check_test_progress.sh` | Progress monitoring |
| `monitor_generation.sh` | Generation monitoring |
| `run_gold_labeling.py` | Labeling for Golden Dataset |
| `eval_wsd_on_gold.py` | WSD evaluation on Golden Dataset |
| `freeze_gold_dataset.py` | Freeze Golden Dataset |
| `verify_gold_checksum.py` | Verify checksum |
| `benchmark_wsd.py` | WSD benchmark |

### Documentation (12)

| Document | Purpose |
|----------|---------|
| `DEVELOPMENT_HISTORY.md` | Development history |
| `QUALITY_FILTERING_PLAN.md` | Card quality improvement plan |
| `REFACTOR_AND_BACKUP_PLAN.md` | This plan (completed) |
| `BACKLOG_IDEAS.md` | Future ideas |
| `GENERATION_INSTRUCTIONS.md` | Generation instructions |
| `WSD_GOLD_DATASET_USAGE.md` | Golden Dataset usage |
| `CREDENTIALS_EXPLANATION.md` | Credentials explanation |
| `GOOGLE_SHEETS_SETUP.md` | Google Sheets setup |
| `LLM_API_KEYS_SETUP.md` | API keys setup |
| `claude_pricing.md` | Claude pricing reference |
| `google_pricing.md` | Gemini pricing reference |
| `openai_pricing.txt` | OpenAI pricing reference |

### Archives

- `src/eng_words/_archive/` — deprecated modules (cache.py, card_generator.py, evaluator.py, prompts.py, fallback.py)
- `scripts/_archive/` — 53 one-off scripts
- `tests/_archive/` — tests for archived modules
- `docs/archive/2026-01-21/` — 55 completed documents

---

## Regression tests

### How to run

**Replay (no LLM calls, with cache):**
```bash
rm -f data/synset_cards/synset_smart_cards_*.json
uv run python scripts/run_synset_card_generation.py 100
uv run python scripts/compare_cards.py \
  --expected backups/2026-01-19/benchmark_100/ \
  --actual data/synset_cards/
```

**Baseline:** `backups/2026-01-19/benchmark_100/` — 54 cards

### Last run results

- ✅ Unit tests: 697 passed
- ✅ Replay 100: 54 cards identical to baseline
- ✅ Live smoke 10: 5 cards, 100% with examples

---

## Next steps

See `QUALITY_FILTERING_PLAN.md` — remaining:
1. Run on full dataset (7,872 cards)
2. Analyze results
3. Final documentation
