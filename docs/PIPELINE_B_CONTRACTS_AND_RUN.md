# Pipeline B — контракты режимов, роли Stage 1 vs LLM, команды запуска

Документ фиксирует роли компонентов, контракты режимов (word/phrasal/mwe), политику извлечения кандидатов и команды запуска после завершения PIPELINE_B_FIXES_PLAN_2 (этапы 0–7).

---

## 1. Роли Stage 1 vs LLM

**Важно: не «ходить по кругу»** — семантика и карточки рождаются только в Pipeline B (LLM).

| Компонент | Роль | Что делает | Чего не делает |
|-----------|------|------------|----------------|
| **Stage 1** | Детерминированная экстракция и нормализация | Токены, предложения, леммы, статистика; списки кандидатов (phrasal/MWE); manifest, sentences. Всё без семантики. | Не кластеризует по смыслу, не генерирует определения, не выбирает «лучшие» примеры по смыслу. |
| **Pipeline B (LLM)** | Единственное место генерации карточек и семантики | Кластеризация по смыслу, определения, перевод, выбор примеров по смыслу. Контракт карточки (поля, 1-based indices), QC (lemma/headword в примерах, POS, дубликаты). | Не подменяет Stage 1: не переопределяет токены/леммы; примеры для запроса приходят из Stage 1 (lemma_examples). |

Итог: Stage 1 готовит входы и кандидатов; LLM только интерпретирует примеры и выдаёт структурированные карточки в рамках контракта.

---

## 2. Контракты режимов (word / phrasal / mwe)

Режим задаётся в `BatchConfig.mode` и определяет, **какую единицу обучения** считаем валидной и как собираем запросы.

| Режим | Единица обучения | Группировка запросов | Ограничения QC |
|-------|-------------------|----------------------|----------------|
| **word** (default) | Одно слово (lemma = headword или headword подтверждён примерами) | По лемме (`lemma:run`). Примеры из `lemma_examples.json`. | Headword должен быть одним словом; MWE/phrasal в word mode → карточка отбрасывается (headword_invalid_for_mode). |
| **phrasal** | Фразовый глагол (например `look up`, `take off`) | По headword (`headword:look up`). Примеры из предложений, где headword встречается. | Headword — несколько слов; тип должен быть phrasal_verb; headword обязан быть в каждом примере. |
| **mwe** | Устойчивое выражение (fixed expression, adverbial phrase) | По headword. Аналогично phrasal, но тип MWE. | Отдельные правила (строже), обычно меньший top-K. |

- **lemma** — ключ группировки в word mode; в phrasal/mwe — единица это **headword**, примеры подбираются по нему.
- **headword** — строка на карточке (может быть `look up`, `by the way`); не обязан равняться lemma; в strict должен быть подтверждён примерами (не «выдуман»).

Контракт и QC: `src/eng_words/word_family/headword.py`, `batch_qc.py` (get_cards_failing_headword_invalid_for_mode, get_cards_failing_lemma_in_example), `batch_schemas.py` (BatchConfig.mode).

---

## 3. Политика извлечения кандидатов (precision-first)

- **Precision-first:** лучше выкинуть кандидата или карточку и перезапустить/retry, чем записать мусор. Fail-fast: при нарушении контракта или strict QC пайплайн падает до записи финального output.
- **Candidate extraction (Stage 1):**
  - Phrasal/MWE-кандидаты извлекаются детерминированно (например, dependency parsing, частоты). Единый артефакт — например `_mwe_candidates.parquet` с колонками headword, type (phrasal_verb / mwe), частота.
  - Фильтрация: min_freq, cap (top-K); без семантики. Решение «брать Stage 1 или LLM как детектор» зафиксировано в плане (Stage 1 при достаточной точности).
- **Использование кандидатов в Pipeline B:**
  - Word deck: как сейчас — по леммам из tokens/sentences, без phrasal/MWE списка.
  - Phrasal/MWE deck: только add-on — берём top-K из candidate list, собираем примеры по headword из sentences, вызываем LLM только для этих K. Word deck при этом не пересчитывается.

Код: `src/eng_words/mwe_candidates.py`, `pipeline.py` (extract_mwe_candidates), `batch_io.py` (load_mwe_candidates_for_deck, render_requests при mode=phrasal/mwe).

---

## 4. Команды запуска

Все команды — из корня проекта: `cd /path/to/eng_words`.

### 4.1 Word deck (create → download → QC gate)

Типичный прогон по леммам из `tokens_sample.parquet` / `sentences_sample.parquet`:

```bash
# 1) Создать батч (requests + lemma_examples), загрузить в API
uv run python scripts/run_pipeline_b_batch.py create [--limit N] [--max-examples 50]

# 2) Проверить статус (опционально)
uv run python scripts/run_pipeline_b_batch.py status

# 3) Дождаться завершения и скачать (или скачать с уже готовыми results)
uv run python scripts/run_pipeline_b_batch.py download [--from-file] [--retry-empty] [--run-gate]
# --from-file: не качать results с API, использовать существующий results.jsonl
# --run-gate: после записи карточек прогнать QC gate; exit 1 при FAIL
```

Либо ждать в фоне с опросом, затем download:

```bash
uv run python scripts/wait_and_download_batch.py [--poll-interval 120]
```

Итог: `data/experiment/cards_B_batch.json` (или путь из `BatchConfig.output_cards_path`). В strict режиме перед записью выполняется QC-gate: при превышении порогов (validation_errors) пайплайн падает и файл не пишется.

### 4.2 Проверка QC gate по готовому файлу карточек

```bash
uv run python scripts/run_quality_investigation.py --gate --cards data/experiment/cards_B_batch.json --output data/experiment/qc_gate_report.md
# Exit code 1 при FAIL.
```

### 4.3 Регрессия 49/49 (Stage 7)

```bash
uv run python scripts/run_regression_49.py --cards data/experiment/cards_B_batch_2.json --output data/experiment/regression_49_report.md
# Exit code 0 = PASS, 1 = FAIL (чеклист в отчёте).
```

### 4.4 Phrasal deck (top-K, add-on)

Phrasal deck строится **отдельно** от word deck: берутся кандидаты из Stage 1 (MWE/phrasal), для них собираются примеры и вызывается LLM только для top-K.

Текущий CLI не выставляет `mode=phrasal` и `top_k`; это делается через `BatchConfig` в коде:

1. Убедиться, что Stage 1 сгенерировал артефакт кандидатов (например `_mwe_candidates.parquet` или путь из pipeline).
2. Задать конфиг: `mode="phrasal"`, `top_k=200` (или иной cap), `candidates_path=Path(...)`.
3. Вызвать `render_requests(config)` → создадутся запросы с ключами `headword:...` и примеры из предложений.
4. Дальше как обычно: create_batch / upload → batch job → download_batch (или Standard API для малого K).

Пример конфига (псевдокод):

```python
from eng_words.word_family.batch_schemas import BatchConfig
from eng_words.word_family.batch_io import render_requests, create_batch, download_batch

config = BatchConfig(
    tokens_path=...,
    sentences_path=...,
    batch_dir=...,
    output_cards_path=...,  # например cards_phrasal.json
    mode="phrasal",
    top_k=200,
    candidates_path=Path("data/.../_mwe_candidates.parquet"),
)
render_requests(config)
# затем create_batch(config) и т.д.
```

После записи — тот же QC gate по `output_cards_path` (и при необходимости регрессия по своим критериям для phrasal).

---

## 5. Где что лежит в коде

| Тема | Файлы |
|------|--------|
| Контракт карточки, hard errors | `src/eng_words/word_family/contract.py` |
| Strict/relaxed, типы ошибок | `src/eng_words/word_family/qc_types.py` |
| QC: lemma/headword в примерах, POS, дубликаты | `src/eng_words/word_family/batch_qc.py` |
| QC gate (пороги, PASS/FAIL по validation_errors) | `src/eng_words/word_family/qc_gate.py` |
| Регрессия 49/49 (критерии PASS/FAIL) | `src/eng_words/word_family/regression.py` |
| Headword, режимы word/phrasal/mwe | `src/eng_words/word_family/headword.py`, `batch_schemas.py` (mode) |
| Кандидаты MWE/phrasal | `src/eng_words/mwe_candidates.py`, `pipeline.py` |
| Batch IO: render_requests, parse_results, download_batch | `src/eng_words/word_family/batch_io.py` |

---

## 6. Где лежит JSON со всеми результатами

- **После оценки уже готовых карточек** (без нового прогона LLM):  
  `data/experiment/pipeline_b_run_result.json`  
  Создаётся скриптом:  
  `uv run python scripts/run_manual_qc_and_collect.py --cards data/experiment/cards_B_batch_2.json --output data/experiment/pipeline_b_run_result.json`  
  В нём: gate, regression_49, пути к отчётам, чеклист deliverables. Карточки при этом не пересчитываются.

- **После полного прогона Stage 1 + Stage 2 (LLM)** (с тем же сидом и 50 леммами / 50 примерами):  
  `data/experiment/full_run_result.json`  
  Создаётся скриптом:  
  `uv run python scripts/run_full_pipeline_b_and_collect_json.py --book american_tragedy --seed 42 --limit 50 --max-examples 50 --output-json data/experiment/full_run_result.json`  
  В нём: stage1 (sample stats, пути к tokens_sample/sentences_sample), stage2 (cards_path, stats, validation_errors_count), gate, regression_49, deliverables.  

  Для полного прогона нужны: артефакты Stage 1 по книге (`data/processed/{book}_tokens.parquet`, `data/processed/{book}_sentences.parquet`), затем скрипт сам делает prepare_pipeline_b_sample (seed 42), render_requests (limit 50, max_examples 50), create, wait, download и в конце пишет `full_run_result.json`. Без данных Stage 1 в репозитории полный прогон не запускался — только оценка существующего `cards_B_batch_2.json`.

---

## 7. Проверка (верификация)

После изменений по плану PIPELINE_B_FIXES_PLAN_2 (этапы 0–7):

- **Тесты:** `uv run pytest tests/ -v` — все тесты должны проходить (на момент фиксации: 759 passed).
- **Регрессия 49/49:** `uv run python scripts/run_regression_49.py --cards data/experiment/cards_B_batch_2.json` — ожидается **PASS** (по уже записанным карточкам). (valid_schema 100%, lemma/headword_in_example 100%, pos_consistency 100%, major_or_invalid 0%). Exit code 0 = PASS, 1 = FAIL (чеклист в выводе или в `--output`).
