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

| Этап | Скрипт/модуль | Что делает |
|------|----------------|------------|
| Подготовка выборки | `scripts/prepare_pipeline_b_sample.py` | tokens.parquet + sentences → `tokens_sample.parquet`, `sentences_sample.parquet` |
| Создание батча | `scripts/run_pipeline_b_batch.py create` | Группировка по лемме, сбор промптов, загрузка в Gemini Batch API |
| Скачивание + retry | `scripts/run_pipeline_b_batch.py download` | Скачивание ответов, парсинг JSON (definition_en, definition_ru, selected_example_indices), подстановка примеров из нашего списка, retry при пустых примерах |

Свой промпт (`CLUSTER_PROMPT_TEMPLATE` в `eng_words.word_family.clusterer`), свой формат ответа (JSON с индексами примеров). Проверка «лемма в примере» — в `run_pipeline_b_batch.py` (пост-проверка после download).

---

## 3. Сводка

| Вопрос | Ответ |
|--------|--------|
| Где парсим книгу? | `eng_words.pipeline` → Stage 1 (`process_book_stage1`) |
| Где генерируем карточки? | **Pipeline B:** `scripts/run_pipeline_b_batch.py` (Batch API, кластеризация по лемме) |
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
