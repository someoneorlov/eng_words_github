# Удалённый архив (_archive)

Папки **`src/eng_words/_archive/`**, **`scripts/_archive/`** и **`tests/_archive/`** удалены из репозитория. Код и скрипты из архива сохранены в **истории git**; здесь — список удалённого и как искать в истории.

---

## Зачем удалили

Архив содержал старый код и разовые скрипты (Pipeline A, эксперименты, gold-разметка, тесты под удалённые модули). Для текущего пайплайна (Stage 1 + Pipeline B) он не нужен; оставлен только актуальный код. Чтобы не путать и не тащить лишнее, архив удалён с пометкой в документации.

---

## Что было в `src/eng_words/_archive/`

- **`aggregation/`** — fallback-логика для агрегации (использовалась в экспериментах).
- **`experiment/`** — заготовки экспериментов.
- **`llm/`** — cache.py, card_generator.py, evaluator.py, prompts.py (старый LLM card generator, evaluator; не Pipeline B).
- **`pipeline_v2/`** — card_generator.py, data_models.py, meaning_extractor.py, pipeline.py (альтернативный пайплайн, не используется).

---

## Что было в `scripts/_archive/`

Скрипты для gold-разметки, анализа WSD, экспериментов с батчами, пилотов и т.п., например:

- `aggregate_gold_labels.py`, `analyze_gold_dataset.py`, `analyze_validation_results.py`, `analyze_wsd_results.py`, `auto_analyze_wsd_errors.py`
- `collect_full_gold_dataset.py`, `collect_gold_pilot.py`, `evaluate_wsd.py`, `export_for_review.py`, `generate_cards.py`, `generate_pilot_report.py`
- `run_anthropic_batch.py`, `run_gemini_batch.py`, `run_smart_labeling.py`, `test_batch_approaches.py`, `test_gold_pipeline_real_data.py`, `test_gsheets.py`, `test_pipeline_backends.py`, `test_retry_direct.py`, `test_retry_logic.py`
- `upload_review_to_gsheets.py`, `wsd_demo.py`, `check_regeneration_status.py`, `freeze_gold_dataset.py` (если был там), `monitor_generation.sh`, `view_mermaid_diagram.sh`, `conftest.py`
- **`experiment/prepare_sample.py`** — старая версия подготовки выборки; актуальный скрипт: **`scripts/prepare_pipeline_b_sample.py`**.

---

## Что было в `tests/_archive/`

Тесты под удалённые или архивные модули:

- `test_llm_card_generator.py`, `test_llm_cache.py`, `test_llm_evaluator.py`, `test_llm_prompts.py` — под `_archive/llm/`
- `test_word_family_clusterer.py` — под класс `WordFamilyClusterer` (Batch-скрипт его не использует)
- `test_export_for_review.py`, `test_fallback.py`, `conftest.py`

---

## Как найти код в git

В коммите с удалением архива в сообщении должно быть что-то вроде:  
**«Remove _archive (src, scripts, tests); see docs/REMOVED_ARCHIVE.md»**.

Примеры:

```bash
# Список файлов в src/eng_words/_archive до удаления
git show <commit>^:src/eng_words/_archive/

# Содержимое конкретного файла
git show <commit>^:src/eng_words/_archive/llm/card_generator.py
git show <commit>^:scripts/_archive/experiment/prepare_sample.py
```

---

## Актуальные замены

| Было в архиве | Сейчас использовать |
|---------------|----------------------|
| `scripts/_archive/experiment/prepare_sample.py` | `scripts/prepare_pipeline_b_sample.py` |
| In-process кластеризация (WordFamilyClusterer) | `scripts/run_pipeline_b_batch.py` (Batch API, без класса кластеризатора) |
