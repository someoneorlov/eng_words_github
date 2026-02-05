# Pipeline B refactor — baseline (этап 0.5)

Заморозка текущего поведения и метрик для сравнения после рефакторинга.

---

## Команды запуска и параметры

### Предусловия

1. **Stage 1** уже выполнен для книги (есть `data/processed/<book>_tokens.parquet`, `<book>_sentences.parquet`, `stage1_manifest.json`).
2. **Переменная окружения:** `GOOGLE_API_KEY` (для Batch API и retry).

**Книга для baseline:** `data/raw/theodore-dreiser_an-american-tragedy.epub`, `--book-name american_tragedy`.

**Проверка наличия файлов:** каталог `data/raw/` может быть в .gitignore; чтобы убедиться, что книга на месте, используйте в терминале `ls -la data/raw/` (а не только просмотр через IDE).

### Шаг 1: Stage 1 (если ещё не выполнен)

```bash
uv run python -m eng_words.pipeline --book-path data/raw/theodore-dreiser_an-american-tragedy.epub --book-name american_tragedy --output-dir data/processed --no-phrasals
```

### Шаг 2: подготовка выборки

```bash
uv run python scripts/prepare_pipeline_b_sample.py --book american_tragedy --size 500
```

- Вход: `data/processed/<book>_tokens.parquet`, `data/processed/<book>_sentences.parquet`
- Выход: `data/experiment/tokens_sample.parquet`, `data/experiment/sentences_sample.parquet`, `data/experiment/sample_stats.json`

### Шаг 3: создание batch (лимит 10 лемм)

```bash
uv run python scripts/run_pipeline_b_batch.py create --limit 10 --max-examples 50
# Если batch_b уже существует — добавьте --overwrite
uv run python scripts/run_pipeline_b_batch.py create --limit 10 --max-examples 50 --overwrite
```

- Выход: `data/experiment/batch_b/requests.jsonl`, `data/experiment/batch_b/lemma_examples.json`, `data/experiment/batch_b/batch_info.json`

### Шаг 4: ожидание и загрузка результатов

```bash
uv run python scripts/run_pipeline_b_batch.py wait
```

- `wait` ждёт завершения batch и вызывает `download` (может занять несколько минут).
- Если batch уже был создан ранее, при новом `create` используйте `--overwrite`.
- Выход: `data/experiment/batch_b/results.jsonl`, `data/experiment/cards_B_batch.json`, `data/experiment/batch_b/download_log.json`

### Шаг 5: отчёт качества (автоматический)

```bash
uv run python scripts/run_quality_investigation.py --cards data/experiment/cards_B_batch.json --output data/experiment/investigation_report.md
```

- Выход: `data/experiment/investigation_report.md`

### Шаг 6: ручная проверка качества (30–50 карточек)

Инструкция для LLM-агента по ручному QC (стратифицированный сэмпл, чеклист, метрики, отчёт): **[PIPELINE_B_MANUAL_QC_INSTRUCTIONS.md](PIPELINE_B_MANUAL_QC_INSTRUCTIONS.md)**.

---

## Baseline метрики (прогон 2026-02-01, american_tragedy, limit 10)

| Метрика | Значение |
|--------|----------|
| Число лемм (limit) | 10 |
| Число карточек | 20 |
| Ошибки парсинга (errors) | 0 |
| Retry (успешно / не успешно) | 0 / 0 |
| lemma_not_in_example (count) | 0 |
| lemmas_with_zero_cards | 0 |
| cards_with_empty_examples | 0 |
| cards_with_examples_fallback | 0 |

**Quality investigation (B4):** 0 записей «пример не содержит лемму» (скрипт использует `example_validator._get_word_forms` и `_word_in_text`, учитываются неправильные формы: thought→think, went→go, said→say, came→come, knew→know).

Источник: `data/experiment/cards_B_batch.json` (config/stats), `data/experiment/batch_b/download_log.json`, `data/experiment/investigation_report.md`.

---

## Expected behavior (кратко)

- Batch создаётся с 10 леммами, запросы уходят в Gemini Batch API.
- После завершения batch: скачиваются `results.jsonl`, парсятся ответы, пишутся карточки в `cards_B_batch.json`.
- Retry по пустым/fallback примерам (Standard API) уменьшает число карточек с пустыми примерами.
- Quality investigation проверяет B4 (lemma in example), A2/A4, B3, E1, F1/F2, G1 и пишет отчёт.

---

## Критерии приёмки этапа 0.5

- [x] pytest зелёный
- [x] baseline артефакты получены (create → wait → download → cards_B_batch.json, download_log.json)
- [x] baseline отчёт качества сохранён (data/experiment/investigation_report.md)
- [x] команды и параметры записаны (этот документ)

**Исправление в скрипте:** при вызове `download()` из `wait()` Typer не подставлял значения по умолчанию, в JSON попадали объекты OptionInfo. В `wait()` добавлен явный вызов `download(retry_empty=True, retry_thinking=False, from_file=False, skip_validation=False)`.

---

## Прогон 2026-02-04 (american_tragedy, limit 10)

| Шаг | Команда | Результат |
|-----|---------|-----------|
| Stage 1 | `--book-path data/raw/theodore-dreiser_an-american-tragedy.epub --book-name american_tragedy --output-dir data/processed --no-phrasals` | Успех (~40 с). Выходы: tokens, sentences, lemma_stats, anki_csv. |
| Sample | `prepare_pipeline_b_sample.py --book american_tragedy --size 500` | 500 предложений, 1466 уникальных лемм (content). |
| Create | `run_pipeline_b_batch.py create --limit 10 --max-examples 50 --overwrite` | Batch создан, 10 лемм. |
| Wait | `run_pipeline_b_batch.py wait` | Завершено (~4 мин), download выполнен. |
| Quality | `run_quality_investigation.py --cards data/experiment/cards_B_batch.json --output data/experiment/investigation_report.md` | Отчёт записан. |

**Метрики (limit 10):** 11 карточек, 0 ошибок парсинга, 0 validation_errors, 0 cards_lemma_not_in_example. Логи: `logs/stage1_american_tragedy.log`, `logs/prepare_sample.log`, `logs/batch_create.log`, `logs/batch_wait.log`, `logs/quality_investigation.log`.

---

## Как получить 50 карточек для ручного QC

Ручная проверка по [PIPELINE_B_MANUAL_QC_INSTRUCTIONS.md](PIPELINE_B_MANUAL_QC_INSTRUCTIONS.md) рассчитана на 30–50 карточек. Чтобы получить ~50 карточек:

1. **Создать batch на 50 лемм** (если ещё не создан):
   ```bash
   uv run python scripts/run_pipeline_b_batch.py create --limit 50 --max-examples 50 --overwrite
   ```

2. **Дождаться завершения и скачать результаты** (Batch API может обрабатывать 50 запросов 15–40 минут):
   ```bash
   uv run python scripts/run_pipeline_b_batch.py wait
   ```
   Или в фоне с опросом раз в 2 минуты:
   ```bash
   uv run python scripts/wait_and_download_batch.py --poll-interval 120
   ```

3. **Проверить число карточек:**
   ```bash
   python3 -c "import json; d=json.load(open('data/experiment/cards_B_batch.json')); print('Cards:', len(d['cards']))"
   ```

4. Когда карточек ≥30–50 — провести ручной QC по инструкции (или попросить агента провести анализ).
