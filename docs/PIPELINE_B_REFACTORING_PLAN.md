# План рефакторинга Pipeline B (Batch): `eng_words.word_family.batch` + thin CLI

**Формат:** рассчитан на выполнение LLM-агентом, маленькими шагами с TDD, проверками, метриками и жёсткими контрактами.

---

## A) Принципы разработки (обновлено: fail-fast + strict/relaxed)

### A.1. Принципы

1. **Fail-fast по инвариантам (по умолчанию)**
   Если нарушен контракт данных/артефактов/схемы/детерминизма/согласованности артефактов — **падать** с понятной ошибкой и инструкцией “как исправить”.
   **Важное правило:** в пайплайне на тысячи лемм “warnings как канал внимания” не работает, поэтому по умолчанию мы не “замалчиваем”.

2. **Strict-by-default + осознанный relaxed**

   * **Strict** (default): всё контрактное и большинство QC-сбоев → error.
   * **Relaxed** (опциональный режим `--relaxed`): часть QC-сбоев записываются как warnings (но с порогами).
   * Пороги, чтобы warnings не превращались в мусор:

     * `--max-warning-rate` (например 0.1%–1%)
     * `--max-warnings-absolute` (например 10–50)
       Превышение порога → **error** даже в relaxed.

3. **Контракты артефактов**
   Форматы `batch_info.json`, `lemma_examples.json`, `results.jsonl`, `cards_B_batch.json`, `download_log.json` закреплены схемами + `schema_version`. Чтение старого — возможно, запись — всегда в актуальную схему.

4. **Модульность**
   Чёткая граница между:

   * **pure core** (детерминированная логика без сети/FS)
   * **IO/интеграцией** (файлы/сеть/клиенты)

5. **TDD и meaningful tests**
   Тесты пишутся параллельно с кодом и реально ловят регрессии (парсинг JSON, индексы, merge retry, idempotency, offline режимы), а не “2+2”.

6. **Поэтапность**
   Разработка маленькими блоками с прогоном тестов и мини-прогоном на sample после каждого блока.

7. **Измеримость качества**
   На sample-выборке до полного запуска: метрики, отчёт, сравнение с baseline.

8. **Простота и поддерживаемость**
   Код должен быть понятен через год: прозрачные имена, dataclass-контракты, минимум “магии”.

9. **Экономия LLM calls**
   Retry только по нужным условиям; `max_examples` ограничивает стоимость/токены; batch-режим — основной.

10. **Кэширование (обязательно)**
    Все Standard API ответы для retry кэшируются локально в `batch_dir` и переиспользуются при повторных запусках.

11. **Precision-first**
    Лучше пропустить карточку, чем создать невалидную: невалидная карточка в strict-mode = **ошибка** (и не попадает в output).

12. **Обратная совместимость CLI**
    `create/status/download/wait` и дефолтные пути остаются рабочими.

---

## B) “Ритуал” работы агента над каждой задачей (обязателен)

Для **каждой** задачи/подзадачи:

1. Выбрать задачу текущего этапа
2. Зафиксировать контракт: входы/выходы/ошибки/инварианты
3. Написать тесты (TDD)
4. Реализовать минимально, чтобы тесты прошли
5. Прогнать `pytest`
6. Прогнать на sample (limit 10–50)
7. Зафиксировать метрики/выводы в отчёте (коротко)
8. Рефакторить/упростить, если нужно
9. Ещё раз `pytest`
10. Только после этого переходить дальше

---

## C) Общие критерии готовности этапа (quality gate)

Этап завершён, когда:

* все задачи этапа выполнены,
* все тесты проходят,
* на sample измерено качество и задокументировано (минимум: counts ошибок/ретраев/невалидных карточек),
* покрытие тестами **≥ 90% для нового/изменённого модуля** (и тесты meaningful),
* код соответствует strict-mode политике.

---

## D) Исходные вводные (важные наблюдения из текущей базы)

Это важно, чтобы план был реалистичным и не “сломал всё”:

* Сейчас логика batch живёт в `scripts/run_pipeline_b_batch.py`, а тесты исторически импортируют скрипт напрямую через sys.path-хаки → перенос нужно делать аккуратно (wrapper-first).
* Текущий `wait` делает side-effect: после ожидания вызывает `download`. В модуле лучше разделить `wait_for_batch` и `download_batch`, а CLI может сохранить прежний UX.
* В batch-режиме код ходит в Gemini напрямую (не через общий `eng_words.llm`), а in-process кластеризация/кэш живут отдельно — при рефакторинге допускается минимальный “адаптер” под Gemini внутри batch-модуля.
* В репо уже есть “quality investigation” скрипт — используем как gate для sample-прогонов.
* В `prepare_pipeline_b_sample.py` уже используется фиксированный seed → детерминизм можно усилить дальше.

---

# Этап 0 — Маленький апдейт Stage 1 (быстро и полезно) + быстрый smoke-run

*(Это добавленный “Шаг 0”, как ты попросил. Он маленький, но помогает всему остальному.)*

**Цель:** сделать так, чтобы Pipeline B получал стабильные входы: `tokens.parquet` + `sentences.parquet` (sentence_id→text) и manifest, чтобы не было реконструкций и проще делать QC/дебаг.

## 0.1. Добавить стандартный артефакт Stage 1: `<book>_sentences.parquet`

**Задачи**

* [ ] В Stage 1 (в функции обработки книги) добавить запись файла:

  * `data/processed/<book>/<book>_sentences.parquet`
  * Колонки: `sentence_id: int`, `text: str`
  * (Опционально: `chapter_id`, `offset`, если уже есть — но не усложнять.)
* [ ] Убедиться, что `sentence_id` согласован с тем, что используется в tokens (`tokens.parquet`).

**Инварианты (fail fast)**

* [ ] `sentence_id` уникальны
* [ ] `tokens.sentence_id` ⊆ `sentences.sentence_id`
  Нарушение → error с подсказкой “пересобери stage1 outputs”.

**TDD**

* [ ] Тест на минимальном фикстуре (короткий текст/мини-book) что:

  * файл создаётся
  * `sentence_id` unique
  * join-инвариант соблюдён

## 0.2. Добавить `stage1_manifest.json`

**Задачи**

* [ ] Писать `data/processed/<book>/stage1_manifest.json` с полями:

  * `schema_version`
  * `created_at`
  * `pipeline_version` (git sha, если доступно)
  * `spacy_model` (если доступно)
  * `token_count`, `sentence_count`
  * `random_seed` (если есть этапы со случайностью)

**TDD**

* [ ] Тест: файл создаётся, `schema_version` есть, базовые поля присутствуют.

## 0.3. Обновить `prepare_pipeline_b_sample.py` (упрощение, без реконструкции)

**Задачи**

* [ ] Перевести `prepare_pipeline_b_sample.py` на использование Stage1 `sentences.parquet`:

  * sample sentence_id
  * фильтровать tokens по sentence_id
  * фильтровать sentences по sentence_id
* [ ] Seed сохраняем (уже есть) + документируем.

**Инварианты**

* [ ] После семпла: все sentence_id из tokens_sample есть в sentences_sample.

**TDD**

* [ ] Тест на синтетике/фикстуре: после семпла всё согласовано.

## 0.4. Быстрый smoke-run Stage 0

**Задачи**

* [ ] Прогнать Stage 1 на маленьком input/фикстуре.
* [ ] Проверить наличие:

  * `<book>_tokens.parquet`
  * `<book>_sentences.parquet`
  * `stage1_manifest.json`
* [ ] Прогнать `prepare_pipeline_b_sample.py` и проверить, что sample файлы корректны.

### Критерии приёмки этапа 0

* Тесты зелёные
* Smoke-run занимает минуты
* Pipeline B может читать `sentences.parquet` напрямую (без “восстановлений”)

---

# Этап 0.5 — Baseline / golden run (без рефакторинга)

**Цель:** заморозить текущее поведение и метрики, чтобы сравнивать после каждого большого шага.

## 0.5.1. Baseline команды

* [ ] Запустить `pytest` (всё зелёное).
* [ ] Прогнать Pipeline B на маленьком лимите (например `--limit 10`), получить артефакты в `data/experiment/...`.
* [ ] Прогнать `run_quality_investigation.py` (или текущий quality-скрипт) и сохранить репорт в `data/experiment/investigation_report.md` (или аналог).

## 0.5.2. Зафиксировать baseline метрики

* [ ] В `docs/pipeline_b_refactor_baseline.md` записать:

  * команды запуска и параметры
  * число лемм, карточек, ошибок парсинга, retries, lemma_not_in_example count (или аналоги)
  * “expected behavior” (коротко)

### Критерии приёмки этапа 0.5

* baseline артефакты получены
* baseline отчёт качества сохранён
* команды и параметры записаны

---

# Этап 1 — “Wrapper first”: новый модуль, логика пока в скрипте

**Цель:** минимальным изменением создать “дом” для Pipeline B и перевести импорты, не меняя поведение.

## 1.1. Создать модуль `src/eng_words/word_family/batch.py` как временную обёртку

**Задачи**

* [ ] Добавить `batch.py`, который **реэкспортирует** функции из текущего скрипта (временно).
* [ ] Добавить `__all__`, комментарии “TEMP WRAPPER — migrate function-by-function”.

**Напоминания агенту (принципы)**

* Не менять поведение
* Только прокидывать вызовы
* Никаких новых зависимостей

## 1.2. Переключить тесты на модуль (не ломая CI)

**Задачи**

* [ ] Создать `tests/test_word_family_batch.py` и перенести тесты так, чтобы импортировать `eng_words.word_family.batch`.
* [ ] Старые тесты можно оставить временно, но цель — чтобы новый модуль покрывался тестами.

## 1.3. Переключить CLI на модуль

**Задачи**

* [ ] В `scripts/run_pipeline_b_batch.py` заменить внутренние вызовы на вызовы через `eng_words.word_family.batch` (пока это тот же код).

### Критерии приёмки этапа 1

* `pytest` зелёный
* baseline прогон (этап 0.5) даёт те же артефакты/метрики
* CLI UX не поменялся

---

# Этап 2 — Paths/Config/Schemas (без изменения логики)

**Цель:** закрепить контракты, убрать магию путей, подготовить почву для переноса логики в модуль.

## 2.1. Dataclasses: `BatchPaths`, `BatchConfig`, схемы

**Задачи**

* [ ] `BatchPaths.from_dir(batch_dir)`:

  * `requests.jsonl`, `results.jsonl`, `lemma_examples.json`, `batch_info.json`, `download_log.json`
* [ ] `BatchConfig`:

  * `tokens_path`, `sentences_path`, `batch_dir`, `output_cards_path`
  * `limit/max_examples`, `model`, `retry_policy`
  * flags: `strict/relaxed`, warning thresholds
* [ ] `BatchInfo` (schema_version + поля из документа)
* [ ] `ErrorEntry` (lemma, stage, error_type, message)
* [ ] `CardRecord` (lemma, pos, definitions, examples, indices, source, warnings)

**TDD**

* [ ] Тест сериализации/десериализации `BatchInfo` (schema_version обязателен).

## 2.2. `schema_version` + backward compatible read

**Задачи**

* [ ] Уметь читать старый `batch_info.json`/`cards_B_batch.json` без schema_version:

  * трактовать как `"0"`, мигрировать на лету
* [ ] Писать всегда в schema_version="1" (или актуальную).

**Напоминания агенту**

* Backward compatible read, forward write
* Любая несовместимость — error с подсказкой

### Критерии приёмки этапа 2

* тесты на схемы зелёные
* старые файлы читаются, новые пишутся с schema_version
* baseline прогон совпадает по смыслу (допустимо: новые поля в json, но не менять содержание cards)

---

# Этап 3 — Перенос pure-core функций в модуль (реальная миграция логики)

**Цель:** вынести большую часть логики из скрипта в модуль, сохранив поведение и повышая тестируемость.

## 3.1. Перенести + покрыть TDD (pure core)

Перенести из скрипта в модуль и покрыть тестами:

1. `build_prompt(lemma, examples)`
2. `build_retry_prompt(lemma, examples)`
3. `extract_json_from_response_text(text)` (JSON может быть в ```json блоке или с мусором вокруг)
4. `parse_one_result(key, response_dict, lemma_examples)`
   **Тест-кейсы (обязательные):**

   * чистый JSON
   * JSON внутри `json … `
   * мусор + JSON
   * невалидный JSON → error
   * индексы 1-based vs 0-based
   * out-of-range индексы
   * пустые/неинтовые `selected_example_indices`
5. `validate_card(card, lemma)` → list ошибок (без исключений внутри)
6. `choose_retry_candidates(parsed_results, policy)`
7. `merge_retry_results(base_cards, retry_cards)` (с закреплённым контрактом merge)
8. `compute_stats(...)`

## 3.2. Precision-first в strict-mode

**Правила**

* [ ] Невалидная карточка по схеме в strict-mode:

  * не попадает в output
  * создаётся `ErrorEntry`
  * если “контрактно критично” (например все карточки леммы невалидны) — можно падать (или считать это fail согласно политике)

### Критерии приёмки этапа 3

* pure core живёт в модуле
* покрытие по pure core ≥ 90%
* baseline прогон даёт тот же результат (кроме новых полей/структуры ошибок)

---

# Этап 4 — IO слой: чтение данных, render-requests, parse-results (оффлайн)

**Цель:** сделать удобный “offline pipeline” без сети для отладки и тестов.

## 4.1. IO функции чтения/записи

**Задачи**

* [ ] `read_tokens(tokens_path)`
* [ ] `read_sentences(sentences_path)`
* [ ] `write_json`, `write_jsonl` с atomic write (temp + rename)

**TDD**

* [ ] отсутствует файл → понятный FileNotFoundError
* [ ] atomic write: при исключении не оставлять битый итоговый файл

## 4.2. `render_requests(config)` (без сети)

**Задачи**

* [ ] Читать tokens+sentences
* [ ] Группировать примеры по lemma (используя `group_examples_by_lemma`)
* [ ] Строить prompts/requests
* [ ] Писать `requests.jsonl` + `lemma_examples.json`

**Детерминизм (обязателен)**

* [ ] Сортировать леммы перед `limit`
* [ ] Стабилизировать порядок примеров внутри леммы (например по `sentence_id` asc) **и документировать**

**TDD**

* [ ] создаются файлы
* [ ] порядок лемм стабилен
* [ ] max_examples обрезает стабильно

## 4.3. `lemma_examples.json` v2 формат (must-have) + backward compat

**Задачи**

* [ ] Новый формат:

  ```json
  {"run":[{"sentence_id":10,"text":"..."}, ...]}
  ```
* [ ] Поддержка чтения старого формата `{lemma: [text, ...]}`.
* [ ] Это необходимо для POS-QC и дебага; не требует изменения Stage 1 (но Stage 0 уже сделал sentences).

## 4.4. `parse_results(config)` (без сети и без retry)

**Задачи**

* [ ] Читать `results.jsonl` + `lemma_examples.json`
* [ ] `parse_one_result` + `validate_card`
* [ ] Писать `cards_B_batch.json` + `download_log.json`

**Fail-fast**

* [ ] Если results.jsonl отсутствует → error
* [ ] Если schema mismatch → error

### Критерии приёмки этапа 4

* есть offline режим: `render-requests` + `parse-results`
* lemma_examples v2 внедрён и backward compatible
* baseline прогон всё ещё работает (в смысле end-to-end через старые команды)

---

# Этап 5 — Сеть: create/status/download/wait через модуль (сохраняем UX)

**Цель:** вынести сетевые вызовы из скрипта в модуль, CLI оставить тонким.

## 5.1. Минимальный API адаптер

**Задачи**

* [ ] `get_client(api_key=None)` (читает env)
* [ ] `upload_requests_file(...)`
* [ ] `create_batch_job(...)`
* [ ] `download_batch_results(...)`
* [ ] `call_standard_retry(...)`

**TDD**

* [ ] мок клиента: create/status/download/standard generate

## 5.2. `create_batch(config)`

**Задачи**

* [ ] Внутри вызвать `render_requests` (или использовать существующие файлы, если позже добавится режим “reuse”)
* [ ] Идемпотентность:

  * если `batch_info.json` есть и нет `--overwrite` → error
* [ ] Писать `batch_info.json` schema_version=1

## 5.3. `get_batch_status(batch_dir)`

**Задачи**

* [ ] читать batch_info
* [ ] вызывать API
* [ ] возвращать структурированный статус

## 5.4. `wait_for_batch(batch_dir, poll_interval_sec=60, timeout_sec=None)`

**Задачи**

* [ ] В модуле wait не вызывает download автоматически (чистое ожидание)
* [ ] CLI-команда `wait` может сохранить старое поведение: дождался → вызвал download

## 5.5. `download_batch(config)`

**Задачи**

* [ ] `from_file=True` → не скачивать batch results
  *(но retry может требовать сеть — это явно документируем)*
* [ ] `from_file=False` → скачать results.jsonl
* [ ] затем `parse_results`
* [ ] затем опционально retry

### Критерии приёмки этапа 5

* команды CLI работают как раньше
* сетевые функции тестируются моками
* baseline прогон совпадает (контент cards не поменялся)

---

# Этап 6 — Retry + кэширование retry (экономия + воспроизводимость)

**Цель:** retry управляемый, дешёвый и воспроизводимый; без лишних сетевых вызовов.

## 6.1. `RetryPolicy` как структура

**Задачи**

* [ ] `RetryPolicy(modes=[standard, thinking], only_if=[...], max_attempts=2)`
* [ ] Тесты: `choose_retry_candidates` даёт ожидаемый список

**Триггеры (закрепить в коде)**

* `json_parse_error`
* `schema_error/missing_required_fields`
* `empty_selected_example_indices`
* `out_of_range_indices`
* `examples_fallback_used`
* (опционально позже) `pos_mismatch_high`

## 6.2. Кэширование Standard API retry (обязательно)

**Задачи**

* [ ] В `batch_dir` хранить `retry_cache.jsonl` (key = lemma + mode + prompt_hash)
* [ ] Перед Standard API:

  * искать в кэше → если есть, использовать
* [ ] В `download_log.json` писать:

  * какие леммы ретраили
  * причина ретрая
  * источник ответа (cache/network)
  * итог

## 6.3. Merge retry результатов (закрепить контракт)

**Задачи**

* [ ] Выбрать и закрепить один контракт merge:

  * A) retry заменяет все карточки леммы (проще), **или**
  * B) retry заменяет только fallback-карточки (сложнее)
* [ ] Тесты на выбранный контракт

### Критерии приёмки этапа 6

* повторный запуск `download --from-file --retry-*` не делает лишних сетевых вызовов
* retry воспроизводим, логируется
* метрики на sample не ухудшились (а лучше улучшились)

---

# Этап 7 — QC и “quality gate” (fail-fast по умолчанию, warnings только осознанно)

**Цель:** качество контролируется автоматически и **не требует ручного чтения тысячи warnings**.

## 7.1. lemma_not_in_example: по умолчанию — error или пороговый gate

**Задачи**

* [ ] Реализовать проверку lemma/form in example (словоформы + границы слова)
* [ ] В strict-mode:

  * либо считать это error сразу, **либо**
  * считать QC-метрикой и падать при превышении порога (`max_warning_rate/absolute`)
* [ ] В relaxed-mode:

  * записывать в warnings, но с порогами

## 7.2. POS distribution hint в prompt (вариант 1+)

**Задачи**

* [ ] Добавить в prompt блок POS distribution по occurrences леммы
* [ ] Сделать это детерминированно и тестируемо

**TDD**

* [ ] синтетический пример: pos_hint корректно строится

## 7.3. POS-QC mismatch (если включено)

**Задачи**

* [ ] Благодаря v2 `lemma_examples` с sentence_id, сверять POS заявленный карточкой vs POS occurrences в sentence_id
* [ ] В strict-mode: падать при превышении порога mismatch
* [ ] В relaxed-mode: warnings + пороги

## 7.4. Интеграция quality investigation как gate

**Задачи**

* [ ] На каждом sample-прогоне запускать quality-скрипт/репорт
* [ ] В PR/логах фиксировать: стало лучше/хуже и почему

### Критерии приёмки этапа 7

* QC превращён в автоматический gate (fail-fast по порогам)
* warnings не “накапливаются в никуда”
* sample quality измеряется и сохраняется

---

# Этап 8 — Финальная чистка: thin CLI, тесты не зависят от scripts, docs

**Цель:** “как будет через год”: модуль — единственный дом логики, CLI — обёртка.

## 8.1. Упростить `scripts/run_pipeline_b_batch.py`

**Задачи**

* [ ] CLI только:

  * парсит args
  * собирает `BatchConfig`
  * вызывает функции модуля
  * печатает stdout
* [ ] Команды:

  * сохранить `create/status/download/wait/list-retry-candidates`
  * добавить `render-requests` и `parse-results`

## 8.2. Тесты

**Задачи**

* [ ] Удалить/переписать тесты, импортирующие `scripts/` напрямую
* [ ] Все тесты целят в `eng_words.word_family.batch`
* [ ] Добить покрытие ≥ 90% (meaningful tests)

## 8.3. Документация

**Задачи**

* [ ] Обновить `docs/PIPELINES_OVERVIEW.md` / README
* [ ] В docstring batch-модуля описать:

  * артефакты
  * state machine
  * strict/relaxed политику
  * пороги QC

### Критерии приёмки этапа 8

* CLI работает, модуль — единственный источник логики
* тесты не зависят от `scripts/`
* baseline прогон совпадает по смыслу
* качество на sample измерено и записано

---

## E) Decision points (ничего не потеряно, + добавлены новые для strict/Stage0)

Эти решения нужно явно зафиксировать (лучше в начале этапа 2), чтобы агент не “дрейфовал”:

1. **Merge retry**: retry заменяет все карточки леммы или только fallback?
2. **parse-results**: отдельная команда или алиас `download --from-file --no-retry`?
3. **--resume**: делать ли вообще “универсальный” флаг или ограничиться `--from-file`, `--overwrite`, `--no-retry`? (рекомендация: меньше магии)
4. **lemma_examples schema**: внедряем v2 сейчас, читаем v1 backward compatible (рекомендация: да)
5. **Errors vs warnings**: строгая таксономия `errors[]` + `warnings[]`, какие типы считаются контрактными
6. **QC пороги**: значения по умолчанию для `max_warning_rate/max_warnings_absolute` и mismatch thresholds
7. **Stage1 manifest**: минимальный набор полей, что считаем обязательным
8. **Детерминизм примеров**: фиксируем сортировку примеров внутри леммы (sentence_id ascending или иной стабильный порядок)

---

## F) Опциональные mini-улучшения Stage 1 (сохранено из предыдущего плана)

Это не обязательно для рефакторинга, но можно сделать позже:

* Если хочется ещё сильнее упростить стык Stage1→B:
  добавить/поддержать `sentences.parquet` как “первоклассный” артефакт (мы это уже сделали в Этапе 0),
  и дальше `prepare_pipeline_b_sample.py` просто фильтрует sentences по sentence_id (без реконструкции).
* Если появятся доп. метаданные (позиции в тексте), можно улучшить “порядок примеров = порядок в книге”.

---

## G) Итог “как агент должен прогонять этапы быстро”

* Stage 0: быстрый smoke-run Stage1+sample
* Stage 0.5: baseline на limit 10–50 + quality report
* Далее: каждый этап — маленькие PR/коммиты с `pytest` + sample run + короткий отчёт.

---

Если хочешь, я могу дополнительно (в следующем сообщении) сделать “версию для агента” в виде **чеклиста задач с шаблоном**:

* **Task:** …
* **Files touched:** …
* **Tests to add:** …
* **Commands to run:** …
* **Done when:** …

Это удобно прямо копировать в issue/план в репозитории и выполнять пункт за пунктом.
