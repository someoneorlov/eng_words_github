# Обзор пайплайнов: где парсим книгу, где генерируем карточки

Краткая шпаргалка по текущему состоянию репозитория.

---

## 1. Парсим книгу (Stage 1)

**Где:** `eng_words.pipeline` → `process_book_stage1()`

**Что делает:**
- Читает книгу (EPUB) → текст
- Токенизация и лемматизация (spaCy)
- Статистика по леммам, фильтрация (known words, zipf)
- Опционально: phrasal verbs

**Выход:** `tokens.parquet`, `lemma_stats`, phrasal (если включено), Anki CSV

**Примечание:** WSD (разметка смыслов) убран из Stage 1 — использовался только Pipeline A. Модуль `eng_words.wsd` остаётся (gold/eval скрипты; в основном пайплайне не используется).

**Как запустить:**
```bash
uv run python -m eng_words.pipeline --book-path ... --book-name ... --output-dir ...
# --no-phrasals — отключить phrasal verbs
```

---

## 2. Генерация карточек: Pipeline B (Word Family, Batch)

**Единственный путь генерации карточек в репозитории.** Pipeline A (WSD + Synset + SmartCardGenerator) удалён; история и причины — в **`docs/HISTORY_PIPELINE_A.md`**.

**Идея:** Без WSD. Группируем все примеры по лемме, один запрос к LLM на лемму — он кластеризует примеры по смыслу и возвращает несколько карточек (1–3 на лемму).

**Где живёт логика:** модуль **`eng_words.word_family`** (пакеты `batch`, `batch_io`, `batch_api`, `batch_core`, `batch_qc`, `batch_schemas`). Скрипт **`scripts/run_pipeline_b_batch.py`** — тонкий CLI: парсит аргументы, собирает `BatchConfig`, вызывает функции модуля, печатает вывод.

| Этап | Команда / модуль | Что делает |
|------|------------------|------------|
| Подготовка выборки | `scripts/prepare_pipeline_b_sample.py` | tokens.parquet + sentences → `tokens_sample.parquet`, `sentences_sample.parquet` |
| Offline: запросы | `run_pipeline_b_batch.py render-requests` | Группировка по лемме, запись `requests.jsonl` и `lemma_examples.json` (без сети) |
| Создание батча | `run_pipeline_b_batch.py create` | Загрузка запросов в Gemini Batch API, создание job, запись `batch_info.json` |
| Статус / ожидание | `status`, `wait` | Опрос API до SUCCEEDED/FAILED |
| Скачивание + retry | `run_pipeline_b_batch.py download` | Скачивание ответов, парсинг, подстановка примеров, retry при пустых/fallback (с кэшем в `retry_cache.jsonl`) |
| Offline: только парсинг | `run_pipeline_b_batch.py parse-results` | Парсинг существующего `results.jsonl` → карточки (без скачивания) |
| Кандидаты на retry | `run_pipeline_b_batch.py list-retry-candidates` | Список лемм с пустыми/fallback-примерами по `results.jsonl` (без API) |

Промпт — `CLUSTER_PROMPT_TEMPLATE` в `eng_words.word_family.clusterer`; формат ответа — JSON с индексами примеров. QC «лемма в примере» и пороги — в `batch_qc`; при превышении порога (strict) пайплайн падает с ошибкой.

### Артефакты Pipeline B (по умолчанию в `data/experiment/` и `data/experiment/batch_b/`)

| Файл | Назначение |
|------|------------|
| `batch_b/requests.jsonl` | Один JSON-объект на строку: ключ леммы + промпт (для Batch API) |
| `batch_b/lemma_examples.json` | Лемма → список примеров (v1: строки, v2: `{sentence_id, text}`) |
| `batch_b/batch_info.json` | Имя job, модель, число лемм, schema_version |
| `batch_b/results.jsonl` | Ответы Batch API (key + response) |
| `batch_b/retry_cache.jsonl` | Кэш ответов Standard API для retry (key + response) |
| `batch_b/download_log.json` | Лог download: ошибки, retry_log, cards_lemma_not_in_example |
| `cards_B_batch.json` | Итоговые карточки (pipeline, stats, cards, validation_errors) |

---

## 3. Сводка

| Вопрос | Ответ |
|--------|--------|
| Где парсим книгу? | `eng_words.pipeline` → Stage 1 (`process_book_stage1`) |
| Где генерируем карточки? | **Pipeline B:** логика в `eng_words.word_family` (batch, batch_io, batch_api); CLI — `scripts/run_pipeline_b_batch.py` |
| Что было с Pipeline A? | Удалён. Описание и история — `docs/HISTORY_PIPELINE_A.md`; код можно найти в истории git. |

---

## 4. Файлы на выходе

| Пайплайн | Пример выходного файла |
|----------|------------------------|
| B (batch) | `data/experiment/cards_B_batch.json` |
| Stage 1 (без карточек) | `data/processed/<book>/tokens.parquet`, `lemma_stats.parquet`, Anki CSV по леммам |

---

## 5. Что не входит в основной пайплайн (Stage 1 + Pipeline B)

Эти модули и скрипты не используются при «книга → карточки»; нужны только для eval или бенчмарков.

| Назначение | Модули / скрипты |
|------------|------------------|
| WSD gold (разметка, eval) | `eng_words.wsd_gold/`, `scripts/run_gold_labeling.py`, `eval_wsd_on_gold.py`, `benchmark_wsd.py`, `freeze_gold_dataset.py`, `verify_gold_checksum.py` |
| WSD (WordNet) | `eng_words.wsd/` — используется в gold-скриптах. В **Pipeline B (batch)** не используется: скрипт `run_pipeline_b_batch.py` берёт из `word_family` только `group_examples_by_lemma` и `CLUSTER_PROMPT_TEMPLATE`, отправляет промпты в Gemini Batch API напрямую (без `eng_words.llm` и без WSD). Merge карточек по embeddings удалён (косинусное сходство работало плохо). |
| LLM (Standard API) | `eng_words.llm/` — не используется в Batch-скрипте. Нужен только если запускать кластеризацию в процессе через класс `WordFamilyClusterer`. |

**Удалённый архив:** Папки `src/eng_words/_archive/`, `scripts/_archive/`, `tests/_archive/` удалены. Список удалённого и как искать в git — в **`docs/REMOVED_ARCHIVE.md`**.
