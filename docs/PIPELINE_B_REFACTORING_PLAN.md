# План рефакторинга Pipeline B: модуль + тонкий CLI

**Цель документа:** дать полное понимание Pipeline B (что это, зачем, откуда данные, куда что пишется) и план рефакторинга (вынести логику батча в модуль `eng_words`, скрипт оставить тонким CLI). Документ рассчитан на ревью: после прочтения должно быть ясно, что и зачем делается, какие входы/выходы, где что хранится и как всё связано.

**Цель рефакторинга:** один «дом» для Pipeline B (batch + in-process), единые входы/выходы, тестируемая библиотека, скрипт — только обёртка над модулем.

**Как читать документ (для ревью):**

- **§ 0** — контекст: что такое Pipeline B, зачем Batch API, откуда исходные данные; полный поток данных по шагам (prepare → create → status/wait → download) и справочник «кто что создаёт/читает» и где хранится. Достаточно для понимания всего пайплайна и данных.
- **§ 1** — текущее состояние: где лежит код, текущие входы/выходы, проблемы.
- **§ 2** — целевое состояние после рефакторинга: модуль + тонкий CLI, предлагаемые API и пути; учёт ревью (pure vs IO, BatchPaths, схемы, идемпотентность, детерминизм, POS, retry).
- **§ 3** — плюсы и минусы рефакторинга.
- **§ 4** — пошаговый план переноса (в т.ч. «сначала обёртка», новые команды CLI).
- **§ 5–6** — сводные таблицы и итог.
- **§ 7** — ответы на вопросы ревью по текущей реализации (как сейчас работает).
- **§ 8** — учёт замечаний ревью: что принимаем, что уточняем, что откладываем.
- **§ 9** — POS: принятое решение (вариант 1+) и шаги внедрения.

---

## 0. Контекст: что такое Pipeline B и зачем он нужен

### 0.1. Общая цепочка «книга → карточки»

В репозитории два этапа:

1. **Stage 1 (парсинг книги):** EPUB → текст → токенизация (spaCy) → леммы, предложения, статистика. Выход: `tokens.parquet`, `lemma_stats.parquet` и др. в `data/processed/<book>/`. Реализация: `eng_words.pipeline` (`process_book_stage1`).
2. **Pipeline B (генерация карточек):** по токенам и предложениям строим для каждой леммы список примеров из книги, отправляем в LLM (Gemini Batch API); модель возвращает 1–3 «карточки» на лемму (определение, перевод, часть речи, выбранные примеры). Выход: `cards_B_batch.json` с готовыми карточками для Anki/других форматов.

Pipeline B не использует WSD (разметку смыслов): мы не разбиваем лемму по synset’ам, а даём модели все примеры по лемме и просим её самой сгруппировать по смыслу и выдать несколько карточек. Такой подход (Word Family) выиграл в эксперименте против Pipeline A (WSD + Synset); подробнее — в `docs/HISTORY_PIPELINE_A.md`.

### 0.2. Зачем именно Batch API

- **Объём:** на полную книгу — тысячи лемм; синхронные запросы по одной лемме заняли бы часы и были бы дороже.
- **Batch API Gemini:** загружаем один файл с запросами (по строке на лемму), задание выполняется на стороне Google, потом скачиваем один файл с ответами. Дешевле и не держит наш процесс занятым.
- **Retry:** часть ответов может содержать некорректные индексы примеров (пустые или out-of-range). После скачивания мы повторно вызываем Standard API только по таким леммам (с напоминанием в промпте) и подставляем исправленные карточки в итог.

### 0.3. Где заканчивается «исходные данные» и начинается Pipeline B

**Исходные данные для Pipeline B** — это уже обработанная книга:

- **Откуда берутся:** либо вывод Stage 1 по одной книге (`data/processed/<book>/<book>_tokens.parquet` и восстановленные предложения), либо подготовленная выборка для экспериментов.
- **Что именно нужно Pipeline B:** таблица токенов с колонками `lemma`, `sentence_id`, `pos`, `is_alpha`, `is_stop` и таблица предложений `sentence_id` → `text`. По ним мы группируем примеры по лемме (content words: NOUN, VERB, ADJ, ADV) и формируем промпты.

То есть «исходные данные» для ревью — это **уже токены и предложения** (файлы, которые создаёт Stage 1 или `prepare_pipeline_b_sample.py`). Сама книга (EPUB) и Stage 1 в этом документе не рефакторятся; мы только явно фиксируем, откуда Pipeline B читает и куда пишет.

---

## 0.4. Полный поток данных: от входов до карточек

Ниже — пошагово, кто что читает/пишет и в каком формате. Все пути даны относительно корня репозитория, если не указано иное.

**Шаг 0. Предусловие (вне Pipeline B):**

- Есть результат Stage 1 по книге: `data/processed/<book>/<book>_tokens.parquet` (и при необходимости восстановленные предложения).
- Либо уже подготовлена выборка: `data/experiment/tokens_sample.parquet` и `data/experiment/sentences_sample.parquet` (см. ниже).

**Шаг 1. Подготовка выборки (если работаем не с полной книгой):**

- **Скрипт:** `scripts/prepare_pipeline_b_sample.py`
- **Читает:** `data/processed/<book>_tokens.parquet` (путь задаётся `--book`, по умолчанию `american_tragedy`).
- **Делает:** семплирует N предложений (по умолчанию 2000), сохраняет токены только этих предложений и восстанавливает текст предложений.
- **Пишет:**
  - `data/experiment/tokens_sample.parquet` — токены по выбранным предложениям (те же колонки, что в Stage 1).
  - `data/experiment/sentences_sample.parquet` — колонки `sentence_id`, `text`.
  - `data/experiment/sample_stats.json` — базовая статистика по выборке.
- **Зачем:** чтобы не гонять полную книгу в батч при тестах или ограниченных экспериментах; для продакшена можно готовить сэмпл из полного `_tokens.parquet` или передать полный файл.

**Шаг 2. Create (создание батча):**

- **Скрипт:** `scripts/run_pipeline_b_batch.py create [--limit N] [--max-examples 50]`
- **Читает:**
  - `data/experiment/tokens_sample.parquet`
  - `data/experiment/sentences_sample.parquet`
- **Делает:**
  - Группирует примеры по лемме (`group_examples_by_lemma` из `eng_words.word_family`): для каждой леммы (только content words) список предложений, в которых она встречается.
  - Для каждой леммы строит промпт (шаблон `CLUSTER_PROMPT_TEMPLATE` + нумерованные примеры + напоминание про 1-based индексы).
  - Формирует JSONL: одна строка на лемму — `{"key": "lemma:<lemma>", "request": { "model": "...", "contents": [...], "generationConfig": {...} }}`.
  - Загружает этот файл в Gemini (Files API), создаёт задание Batch API, получает имя задания (batch name).
- **Пишет:**
  - `data/experiment/batch_b/requests.jsonl` — запросы (локальная копия, для отладки/повтора).
  - `data/experiment/batch_b/lemma_examples.json` — словарь `{ "lemma": ["sentence1", ...] }`; нужен при download, чтобы подставить текст примеров по индексам из ответа LLM.
  - `data/experiment/batch_b/batch_info.json` — `batch_name`, `model`, `lemmas_count`, `created_at`, `uploaded_file` (имя загруженного файла в Gemini).
- **Внешний сервис:** Gemini: загружен файл, создано batch-задание (состояние PENDING → RUNNING → SUCCEEDED/FAILED).

**Шаг 3. Status / Wait (опционально):**

- **Скрипт:** `run_pipeline_b_batch.py status` или `wait`
- **Читает:** `data/experiment/batch_b/batch_info.json`
- **Делает:** по `batch_name` запрашивает у Gemini статус задания; `wait` крутит опрос до SUCCEEDED или FAILED/CANCELLED.
- **Ничего не пишет** в репозиторий (только stdout).

**Шаг 4. Download (скачивание ответов и сбор карточек):**

- **Скрипт:** `scripts/run_pipeline_b_batch.py download [--retry-empty] [--retry-thinking] [--from-file]`
- **Читает:**
  - `data/experiment/batch_b/batch_info.json` (чтобы взять `batch_name` и при необходимости скачать результат).
  - Если не `--from-file`: по API скачивается файл с ответами (Gemini кладёт его в указанное при создании место; мы читаем через API).
  - `data/experiment/batch_b/lemma_examples.json` — чтобы по `selected_example_indices` из ответа LLM подставить реальные предложения.
- **Пишет:**
  - `data/experiment/batch_b/results.jsonl` — сырые ответы API (одна строка на лемму: `key`, `response` с кандидатами и текстом).
  - `data/experiment/cards_B_batch.json` — **итоговый артефакт пайплайна:** pipeline, config, stats, массив `cards`, ошибки, леммы с нулём карточек.
  - `data/experiment/batch_b/download_log.json` — детали retry, пустые примеры, fallback, «лемма не в примере» (для отладки и ревью качества).
- **Делает:**
  - Парсит каждую строку results.jsonl: извлекает JSON из ответа, проверяет обязательные поля карточки, по индексам подставляет примеры из `lemma_examples.json`; при невалидных индексах помечает карточку (examples_fallback) или оставляет примеры пустыми.
  - При включённом retry: для лемм с пустыми/fallback примерами повторно вызывает Standard API с тем же промптом плюс напоминание про индексы; при необходимости второй проход с Thinking-моделью.
  - Проверка «лемма в примере»: помечает карточки, где ни один пример не содержит лемму (или её словоформу), в download_log для ручной проверки.
  - Собирает все карточки в один список и пишет в `cards_B_batch.json`.

**Итоговая цепочка хранилища:**

- **Исходные данные для Pipeline B:** `tokens_sample.parquet` + `sentences_sample.parquet` (или полные tokens/sentences после Stage 1).
- **Промежуточные (батч):** `batch_b/requests.jsonl`, `batch_b/lemma_examples.json`, `batch_b/batch_info.json`, `batch_b/results.jsonl`, `batch_b/download_log.json`.
- **Выход пайплайна:** `data/experiment/cards_B_batch.json` — готовые карточки для дальнейшего использования (импорт в Anki, конвертация и т.д.).

### 0.5. Справочник: кто что создаёт и кто что читает

| Путь | Кто создаёт | Кто читает | Формат / назначение |
|------|-------------|------------|----------------------|
| `data/processed/<book>/<book>_tokens.parquet` | Stage 1 (`eng_words.pipeline`) | prepare_pipeline_b_sample | Parquet: lemma, sentence_id, pos, is_alpha, is_stop, book, … |
| `data/experiment/tokens_sample.parquet` | prepare_pipeline_b_sample | run_pipeline_b_batch create | Подмножество токенов (по выбранным sentence_id) |
| `data/experiment/sentences_sample.parquet` | prepare_pipeline_b_sample | run_pipeline_b_batch create | Parquet: sentence_id, text |
| `data/experiment/sample_stats.json` | prepare_pipeline_b_sample | — | Статистика выборки (размер, распределение лемм) |
| `data/experiment/batch_b/requests.jsonl` | run_pipeline_b_batch create | Gemini (upload), человек (отладка) | Одна строка на лемму: key + request (model, contents, generationConfig) |
| `data/experiment/batch_b/lemma_examples.json` | run_pipeline_b_batch create | run_pipeline_b_batch download | { "lemma": ["sentence1", ...] } — подстановка примеров по индексам |
| `data/experiment/batch_b/batch_info.json` | run_pipeline_b_batch create | status, wait, download | batch_name, model, lemmas_count, created_at, uploaded_file |
| `data/experiment/batch_b/results.jsonl` | Gemini (ответы) → download скачивает | run_pipeline_b_batch download | Одна строка на лемму: key, response (candidates с text) |
| `data/experiment/batch_b/download_log.json` | run_pipeline_b_batch download | Человек (ревью качества) | Ошибки парсинга, retry, empty/fallback, lemma_not_in_example |
| `data/experiment/cards_B_batch.json` | run_pipeline_b_batch download | Следующие этапы (Anki, экспорт) | JSON: pipeline, config, stats, cards[], errors[], lemmas_with_zero_cards |

Переменные окружения: `GOOGLE_API_KEY` — нужен для create, status, wait, download (и для retry при download).

---

## 1. Текущее состояние

### 1.1. Где что лежит

| Компонент | Расположение | Назначение |
|-----------|--------------|------------|
| Логика батча (create, status, download, wait, retry, парсинг) | `scripts/run_pipeline_b_batch.py` (~710 строк) | Единственная реализация Pipeline B в продакшене |
| Группировка по лемме + промпт | `src/eng_words/word_family/clusterer.py` | `group_examples_by_lemma()`, `CLUSTER_PROMPT_TEMPLATE` |
| In-process кластеризация (не используется в batch) | `src/eng_words/word_family/clusterer.py` | Класс `WordFamilyClusterer` + `eng_words.llm` |
| Подготовка сэмпла для батча | `scripts/prepare_pipeline_b_sample.py` | Читает tokens.parquet → пишет tokens_sample.parquet, sentences_sample.parquet |
| Валидация «лемма в примере» | `src/eng_words/validation/example_validator.py` | `_get_word_forms`, `_word_in_text` (опционально в скрипте) |

### 1.2. Входы и выходы (текущие)

**Входы (для полного цикла Pipeline B):**

| Вход | Откуда | Формат |
|------|--------|--------|
| Токены | Stage 1 или `prepare_pipeline_b_sample.py` | `data/experiment/tokens_sample.parquet` (lemma, sentence_id, pos, is_alpha, is_stop, …) |
| Предложения | Тот же prepare | `data/experiment/sentences_sample.parquet` (sentence_id, text) |
| API-ключ | Окружение | `GOOGLE_API_KEY` |

**Выходы:**

| Выход | Куда | Формат |
|-------|-----|--------|
| Запросы батча | `data/experiment/batch_b/requests.jsonl` | Одна строка на лемму: `{"key": "lemma:X", "request": {...}}` |
| Примеры по леммам | `data/experiment/batch_b/lemma_examples.json` | `{ "lemma": ["sentence1", ...] }` |
| Метаданные батча | `data/experiment/batch_b/batch_info.json` | batch_name, model, lemmas_count, created_at, uploaded_file |
| Сырые ответы API | Скачивание из Gemini | `data/experiment/batch_b/results.jsonl` (после download) |
| Карточки | `data/experiment/cards_B_batch.json` | JSON: pipeline, config, stats, cards[], errors[], lemmas_with_zero_cards |
| Лог загрузки | `data/experiment/batch_b/download_log.json` | Детали retry, empty/fallback, lemma_not_in_example |

**Жёстко зашитые пути в скрипте:**

- `DATA_DIR = PROJECT_ROOT / "data" / "experiment"`
- `BATCH_DIR = DATA_DIR / "batch_b"`
- `TOKENS_PATH`, `SENTENCES_PATH`, `BATCH_INFO_PATH`, `REQUESTS_PATH`, `RESULTS_PATH`, `LEMMA_EXAMPLES_PATH`, `OUTPUT_CARDS_PATH`

### 1.3. Проблемы текущей организации

- **Вся логика в скрипте:** нельзя вызвать «создать батч» или «скачать результат» из другого кода/ноутбука без копирования.
- **Нет единого API:** пути и конфиг размазаны по константам и функциям внутри одного файла.
- **Тесты зависят от скрипта:** `tests/test_pipeline_b_batch.py` импортирует `run_pipeline_b_batch as batch` и тестирует функции скрипта (build_prompt, _parse_one_result и т.д.) — тесты должны жить рядом с библиотекой.
- **Два «лица» Pipeline B:** batch — в скрипте, in-process — в `word_family.clusterer`; концептуально это один пайплайн с разными режимами выполнения.
- **Сложно переиспользовать:** например, «только построить промпты и сохранить в файл» или «только распарсить готовый results.jsonl» без запуска всего скрипта неудобно.

---

## 2. Предлагаемое состояние

### 2.1. Архитектура

- **Модуль `eng_words.word_family.batch`** (новый пакет или файл `word_family/batch.py`):
  - Вся логика: загрузка данных, построение промптов, создание батча в API, статус, скачивание, парсинг, retry (empty + thinking), валидация, запись карточек и логов.
  - Входы/выходы задаются аргументами и конфигом (пути, лимиты, модель, флаги retry), без жёсткой привязки к `data/experiment/batch_b`.
- **Скрипт `scripts/run_pipeline_b_batch.py`**:
  - Только парсинг аргументов (Typer: create / status / download / wait / list-retry-candidates и, по ревью, render-requests / parse-results) и вызов соответствующих функций из `eng_words.word_family.batch` с дефолтными путями и опциями.
- **Общее для Pipeline B** остаётся в `word_family`: `group_examples_by_lemma`, `CLUSTER_PROMPT_TEMPLATE` — используются и батчем, и (при необходимости) in-process кластеризатором.

### 2.1.1. Нарезка «pure core» vs «side effects» (по ревью)

Чтобы модуль не превратился в монолит на 800+ строк и чтобы 90% логики покрывалось unit-тестами без сети:

**Pure / тестируемые (без сети и файловой системы):**

- `build_prompt(lemma, examples)` → str  
- `build_retry_prompt(lemma, examples)` → str  
- `parse_one_result(key, response_dict, lemma_examples)` → (lemma, cards, error)  
- `validate_card(card, lemma)` → list[str] (ошибки)  
- `choose_retry_candidates(parsed_results, ...)` — какие леммы идти в retry (empty / out_of_range)  
- `merge_retry_results(base_cards, retry_cards)` — подстановка исправленных карточек вместо старых по лемме  
- `compute_stats(...)` — итоговые счётчики для результата  

**IO / интеграция (моки только клиент + ответы API):**

- Чтение/запись: `read_tokens`, `read_sentences`, `write_json`, `write_jsonl`  
- `create_batch(...)` — upload file + batches.create  
- `download_results(...)` — batches.get + files.download (сырой results.jsonl)  
- `call_standard_api_for_retry(...)` — один запрос Standard API  

Публичный API модуля: `create_batch`, `get_batch_status`, `download_batch`, `wait_for_batch`, `list_retry_candidates`, плюс при необходимости `render_requests` (только запись requests.jsonl + lemma_examples.json без API) и `parse_results` (только парсинг results.jsonl → карточки, без retry и без записи в API).

### 2.1.2. Пути: BatchPaths (по ревью)

Вместо передачи 5–7 путей в каждую функцию:

- Передаётся **`batch_dir`** (и при необходимости `tokens_path`, `sentences_path`, `output_cards_path` снаружи).
- Внутри используется **`BatchPaths.from_dir(batch_dir)`**, дающий стандартные имена файлов:
  - `paths.requests_jsonl`, `paths.lemma_examples_json`, `paths.batch_info_json`, `paths.results_jsonl`, `paths.download_log_json`
- Точечное переопределение путей — только если реально понадобится (редко).

### 2.1.3. Схемы артефактов и schema_version (по ревью)

Закрепить форматы кодом (dataclass + `.to_dict()` / `.from_dict()`), чтобы через месяцы однозначно понимать, что в файле.

**batch_info.json:**

- `schema_version` (например `"1"`)
- `pipeline_version` или git commit (опционально)
- `created_at`, `model`, `thinking_model` (даже если thinking выключен)
- `request_count` / `lemmas_count`
- `tokens_path`, `sentences_path` (как были заданы при create)
- `max_examples`, `limit`
- `uploaded_file_name` / `batch_name` (как возвращает API)

**cards_B_batch.json:**

- `schema_version`
- `pipeline="B"`
- `config`: model, max_examples, limit, retry flags
- `stats`: lemmas_total, lemmas_succeeded, cards_total, retries_total, parse_errors_total
- `cards[]`: lemma, part_of_speech, definition_en, definition_ru, examples[], selected_example_indices, source (batch | retry_standard | retry_thinking), warnings[]
- `errors[]`: структурированные (lemma, stage, error_type, message)

### 2.1.4. Идемпотентность и resume (по ревью)

Правила зафиксировать в доке и в коде:

**create:**

- Если `batch_info.json` уже есть — по умолчанию **ошибка** (не перезаписывать случайно). Разрешить перезапись только с флагом `--overwrite`.
- Если только `requests.jsonl` и `lemma_examples.json` есть, а batch_info нет — можно разрешить «только перегенерировать файлы, не создавать job» (режим `render-requests`).

**download:**

- Если `results.jsonl` уже есть и передан `--from-file` — **никогда не трогать сеть**, только парсинг и retry (если нужно).
- Если `output_cards_path` уже есть — по умолчанию **перезаписать** (текущее поведение); при желании позже можно добавить `--no-overwrite` и выходить с ошибкой.

**retry:**

- Сохранять ли сырые ответы retry для отладки: опционально писать в `batch_b/retry_results.jsonl` или включать в download_log — зафиксировать в плане и при реализации решить (хотя бы в download_log список лемм с retry и исходный текст ответа при ошибке).

### 2.1.5. Детерминизм (по ревью)

- **Сортировка лемм:** перед применением `limit` леммы **сортировать** (`sorted(lemmas)` или `lemma_groups.sort_values("lemma")` затем `.head(limit)`), иначе при каждом прогоне в лимит могут попадать разные леммы (порядок в DataFrame от группировки не гарантирован).
- **Семплирование примеров:** если в `group_examples_by_lemma` или при обрезке до `max_examples` появится случайность — зафиксировать `seed` для воспроизводимости.

### 2.1.6. Retry: оформление как стратегия (по ревью)

Вместо двух разрозненных флагов внутри кода держать структуру (можно собирать из флагов CLI):

- `retry_policy = { "max_attempts": 2, "modes": ["standard", "thinking"], "only_if": ["empty_examples", "out_of_range_indices"] }`

Так проще расширять (например, третий режим или условие «только pos_mismatch») и тестировать.

### 2.1.7. «lemma_not_in_example» — warning, не error (по ревью)

- Сейчас это уже **не блокирует** результат: карточки попадают в вывод, а факт «лемма/словоформа не найдена в примере» пишется в `download_log.json` в `cards_lemma_not_in_example`.
- В плане и в коде явно описать: это **warning** для ручной проверки; возможны false positives (другая форма, апостроф, пунктуация). В документе и в download_log указать **что именно проверяем**: словоформы леммы (включая неправильные формы из словаря) + вхождение в текст как **целое слово** (regex `\b...\b`, case-insensitive). См. § 7.

### 2.2. Предлагаемые входы/выходы модуля

**Конфиг батча (общий для create/status/download):**

- `tokens_path`, `sentences_path` — откуда читать токены и предложения.
- `batch_dir` — каталог батча (batch_info.json, requests.jsonl, results.jsonl, lemma_examples.json, download_log.json).
- `output_cards_path` — куда писать итоговый `cards_B_batch.json` (или None, если только парсинг без записи).
- `limit` (0 = все леммы), `max_examples` — как сейчас.
- `model` — модель Gemini.

**create_batch(...)**

- Входы: конфиг выше (+ опционально client).
- Выходы: `BatchInfo` (batch_name, model, lemmas_count, created_at, uploaded_file) и запись в `batch_dir`: requests.jsonl, lemma_examples.json, batch_info.json.
- Исключения: FileNotFoundError если нет tokens/sentences, ValueError если нет API key.

**get_batch_status(batch_dir, client=None)**

- Входы: путь к batch_dir, опционально клиент.
- Выходы: объект/словарь с полями state, batch_name, lemmas_count (из batch_info), при необходимости error.

**download_batch(...)**

- Входы: batch_dir, output_cards_path, skip_validation, retry_empty, retry_thinking, from_file (использовать существующий results.jsonl), опционально client.
- Выходы: словарь результата (как текущий result: pipeline, config, stats, cards, errors, lemmas_with_zero_cards) + запись output_cards_path и download_log.json в batch_dir.

**wait_for_batch(batch_dir, poll_interval_sec=60, client=None)**

- Входы: batch_dir, интервал опроса, опционально client.
- Выходы: по завершении — то же, что get_batch_status (state=SUCCEEDED); при ошибке — исключение или код ошибки.

**Вспомогательные (публичные для тестов и переиспользования):**

- `load_lemma_groups(tokens_path, sentences_path, limit=None, max_examples=50)` → (lemma_groups_df, lemma_examples_dict).
- `build_prompt(lemma, examples)` → str.
- `build_retry_prompt(lemma, examples)` → str.
- `parse_one_result(key, response_dict, lemma_examples)` → (lemma, cards, error).
- `validate_card(card, lemma)` → list[str] (ошибки).
- Опционально: `write_requests_and_lemma_examples(lemma_groups_df, lemma_examples, requests_path, lemma_examples_path, model)` — чтобы тестировать «только создание файлов» без API.

Все пути — аргументы функций или конфиг-объект, без глобальных констант внутри библиотеки.

### 2.3. Скрипт после рефакторинга

- Читает из CLI: команда (create / status / download / wait / list-retry-candidates и, по ревью, **render-requests** / **parse-results**), опции (--limit, --max-examples, --model, --retry-empty, --retry-thinking, --from-file, --skip-validation, **--overwrite**, **--resume** и т.д.).
- Формирует дефолтные пути: например `data/experiment`, `data/experiment/batch_b`, `data/experiment/cards_B_batch.json` (можно задать через переменные окружения или флаги, например `--data-dir`, `--batch-dir`, `--output`).
- Вызывает:
  - `create` → `batch.create_batch(...)` (при наличии batch_info без --overwrite — ошибка).
  - **render-requests** → только построение и запись `requests.jsonl` + `lemma_examples.json` **без** вызова Gemini (отладка/тесты без API).
  - `status` → `batch.get_batch_status(...)`
  - `download` → `batch.download_batch(...)`
  - `wait` → `batch.wait_for_batch(...)` затем `batch.download_batch(...)`
  - **parse-results** → взять существующий `results.jsonl` и собрать `cards_B_batch.json` (парсинг + подстановка примеров, без retry и без скачивания из API).
  - `list-retry-candidates` → функция в модуле, которая читает results.jsonl + lemma_examples.json и возвращает список лемм (скрипт только печатает).
- Вся логика и проверки — в модуле; скрипт не дублирует условия и форматирование.

### 2.4. Тесты (по ревью)

**Unit (без сети):**

1. **build_prompt** — snapshot/golden test с небольшим набором примеров.
2. **parse_one_result**: корректный JSON; JSON внутри блока \`\`\`json … \`\`\`; мусор + JSON; невалидный JSON; out-of-range индексы (1-based и 0-based).
3. **validate_card**: отсутствующие ключи; пустые examples; selected_example_indices пустые или не int.
4. **merge_retry_results** — подстановка исправленных карточек по лемме.
5. **compute_stats** — совпадение счётчиков с ожидаемыми.

**Интеграция (с мокнутым клиентом):**

- Мокать методы клиента: upload file → create batch → get status → download result file → standard generate для retry.
- Проверять: те же файлы создаются, stats/errors совпадают, идемпотентность/resume работает.

**Контракт на формат артефактов:**

- Тест: BatchInfo сериализуется/десериализуется; DownloadResult (или аналог) — то же; поля schema_version присутствуют и совместимы.

**Скрипт:**

- Отдельно: вызов с --help; create с --limit 0 (ошибка «no lemmas»); при необходимости render-requests без API.

---

## 3. Плюсы и минусы рефакторинга

### Плюсы

- **Один модуль Pipeline B (batch):** вся логика в `eng_words.word_family.batch`, скрипт — только CLI. Легко найти код и документировать.
- **Переиспользование:** вызов из других скриптов, ноутбуков, DAG без копирования кода.
- **Тестируемость:** функции с явными входами/выходами удобно покрывать unit-тестами; моки только для `genai.Client`.
- **Гибкие пути и конфиг:** разные окружения (dev/prod, другие каталоги) без правки кода библиотеки.
- **Единый «вход» в Pipeline B:** batch и in-process живут в одном пакете `word_family` (batch — batch.py, in-process — clusterer.py), проще объяснять и поддерживать.
- **Меньше дублирования в будущем:** общие куски (формат промпта, парсинг ответа) в одном месте.

### Минусы / риски

- **Объём переноса:** нужно аккуратно вынести ~500+ строк из скрипта в модуль, сохранив поведение (в т.ч. retry, логи, форматы файлов).
- **Зависимости:** модуль будет зависеть от `google.genai` (и опционально от `eng_words.validation`). Либо оставляем как есть, либо выносим «адаптер» API за интерфейс, если захотим менять провайдера позже — на первом этапе можно не усложнять.
- **Обратная совместимость:** текущие пользователи запускают только скрипт; после рефакторинга CLI и пути по умолчанию должны остаться такими же (те же команды, те же файлы в `data/experiment/batch_b` и `cards_B_batch.json`).

---

## 4. Пошаговый план рефакторинга

### Шаг 0. Снижение риска (по ревью): сначала обёртка

- Сделать `eng_words.word_family.batch` **обёрткой**, которая импортирует функции из текущего скрипта (временно: скрипт остаётся «толстым», модуль просто реэкспортирует).
- Скрипт переключить на вызовы через `eng_words.word_family.batch`. Убедиться, что старый сценарий (create → status → download) даёт те же артефакты.
- Дальше переносить функции **по одной** из скрипта в модуль, на каждом шаге прогонять тесты и при необходимости сравнивать артефакты.

### Шаг 1. Подготовка

- Создать `src/eng_words/word_family/batch.py` (или пакет `word_family/batch/` с `__init__.py`).
- Определить минимальный публичный API: например `create_batch`, `get_batch_status`, `download_batch`, `wait_for_batch`, плюс вспомогательные `load_lemma_groups`, `build_prompt`, `build_retry_prompt`, `parse_one_result`, `validate_card`.
- Описать типы/датаклассы для конфига и результата (BatchInfo, DownloadResult, BatchPaths) в том же модуле или в `word_family/__init__.py`; ввести schema_version в артефакты.

### Шаг 2. Перенос без изменения поведения

- Перенести из скрипта в `batch.py`:
  - Константы (MODEL, THINKING_MODEL, REQUIRED_CARD_KEYS, HOMONYM_EXCLUDE_FORMS) — как аргументы по умолчанию или конфиг.
  - `load_lemma_groups` (с параметрами paths, limit, max_examples).
  - `build_prompt`, `build_retry_prompt`.
  - `_parse_one_result` → `parse_one_result` (публичный).
  - `_validate_card` → `validate_card`.
  - `_call_standard_api_for_retry` → внутренняя или публичная с client.
  - `_cards_lemma_not_in_example` — внутренняя, с опциональной зависимостью от validation.
- Везде заменить жёсткие пути на аргументы (tokens_path, sentences_path, batch_dir, output_cards_path).
- Оставить в скрипте только: импорт из `eng_words.word_family.batch`, парсинг Typer, формирование путей по умолчанию и вызов функций модуля. Убедиться, что текущие тесты `test_pipeline_b_batch.py` переключены на импорт из `eng_words.word_family.batch` и проходят.

### Шаг 3. Функции create / status / download / wait

- **create_batch(tokens_path, sentences_path, batch_dir, output_requests_path, output_lemma_examples_path, batch_info_path, limit=0, max_examples=50, model=...)**  
  Внутри: load_lemma_groups → build запросы → запись requests.jsonl и lemma_examples.json → upload file → client.batches.create → запись batch_info.json. Возврат BatchInfo.
- **get_batch_status(batch_dir, client=None)**  
  Чтение batch_info.json, при необходимости создание client, client.batches.get(name) → возврат статуса.
- **download_batch(batch_dir, output_cards_path, lemma_examples_path, results_path, batch_info_path, skip_validation=False, retry_empty=True, retry_thinking=False, from_file=False, model=..., thinking_model=..., client=None)**  
  Логика как в текущем download: при необходимости скачать results.jsonl из API, парсинг, retry, валидация, запись cards и download_log. Возврат словаря результата.
- **wait_for_batch(batch_dir, poll_interval_sec=60, client=None)**  
  Цикл get_batch_status до SUCCEEDED/FAILED/CANCELLED; при SUCCEEDED можно не вызывать download автоматически, а позволить скрипту вызвать download отдельно (как сейчас wait → download).

Скрипт по очереди переводится на вызовы этих функций с дефолтными путями (например `batch_dir = Path("data/experiment/batch_b")` и т.д.).

### Шаг 4. CLI и пути по умолчанию

- В скрипте оставить только: Typer app, определение дефолтных путей (от PROJECT_ROOT или от env), вызов batch.create_batch(...), get_batch_status(...), download_batch(...), wait_for_batch(...).
- Добавить при необходимости флаги `--data-dir`, `--batch-dir`, `--output` для переопределения путей.
- Команда `list-retry-candidates`: вызов новой функции модуля, например `list_retry_candidates(batch_dir)` → список лемм; скрипт печатает его.

### Шаг 5. Тесты и экспорт

- Перенести тесты из `tests/test_pipeline_b_batch.py` в `tests/test_word_family_batch.py` (или аналог), тестировать функции из `eng_words.word_family.batch`.
- Добавить в `word_family/__init__.py` экспорт при необходимости: например `from eng_words.word_family.batch import create_batch, download_batch, ...` чтобы можно было вызывать из кода без знания внутренней структуры пакета.
- Прогнать полный сценарий: create (с --limit 2), status, download (или wait → download) на реальных данных и убедиться, что выходы (cards_B_batch.json, download_log.json) совпадают с текущим поведением.

### Шаг 6. Документация

- Обновить `docs/PIPELINES_OVERVIEW.md`: указать, что Pipeline B (batch) реализован в `eng_words.word_family.batch`, скрипт — CLI.
- В docstring модуля batch описать входы/выходы каждой функции и формат конфига/путей.
- При необходимости кратко описать в README или в PIPELINE_B_REFACTORING_PLAN.md итоговую структуру (какой файл за что отвечает).

---

## 5. Сводная таблица: входы и выходы (предлагаемые)

| Функция / этап | Входы | Выходы |
|----------------|-------|--------|
| **load_lemma_groups** | tokens_path, sentences_path, limit, max_examples | (DataFrame lemma_groups, dict lemma → list[str] examples) |
| **build_prompt** | lemma, examples | str (prompt) |
| **build_retry_prompt** | lemma, examples | str (prompt + reminder) |
| **create_batch** | tokens_path, sentences_path, batch_dir, limit, max_examples, model, (client) | BatchInfo; файлы: requests.jsonl, lemma_examples.json, batch_info.json |
| **get_batch_status** | batch_dir, (client) | { state, batch_name, lemmas_count, error? } |
| **download_batch** | batch_dir, output_cards_path, skip_validation, retry_empty, retry_thinking, from_file, model, thinking_model, (client) | Result dict (cards, errors, stats, config); файлы: output_cards_path, batch_dir/download_log.json; при необходимости results.jsonl в batch_dir |
| **wait_for_batch** | batch_dir, poll_interval_sec, (client) | — (до завершения); при успехе state=SUCCEEDED |
| **list_retry_candidates** | batch_dir (results.jsonl + lemma_examples.json) | list[str] lemmas (или структура с empty/fallback) |

**Скрипт (после рефакторинга):**

- Входы: команда (create/status/download/wait/list-retry-candidates), CLI-опции.
- Выходы: те же файлы и тот же stdout, что и сейчас; при необходимости выходной путь переопределяется флагом.

---

## 6. Итог

- **Текущее состояние:** один большой скрипт с жёсткими путями и всей логикой; word_family даёт только группировку и шаблон промпта.
- **Целевое состояние:** логика батча в `eng_words.word_family.batch` с явными входами/выходами и путями; скрипт — тонкий CLI; тесты против модуля; один пакет `word_family` для двух режимов Pipeline B (batch и in-process).
- Рефакторинг делается пошагово с сохранением текущего поведения и обратной совместимости CLI и форматов файлов.

---

## 7. Ответы на вопросы ревью: текущая реализация

Ниже — как **сейчас** устроено в коде (инфраструктура и поведение), чтобы ревьюер мог сверить замечания с фактом.

### 7.1. Идемпотентность и resume

- **create:** не проверяет наличие `batch_info.json`; всегда перезаписывает `requests.jsonl`, `lemma_examples.json`, `batch_info.json`. Режима resume/overwrite нет.
- **download:** если передан `--from-file`, использует существующий `results.jsonl` и не вызывает API для скачивания. Файл `cards_B_batch.json` и `download_log.json` всегда перезаписываются. Отдельного флага «не перезаписывать output» нет.

### 7.2. Детерминизм (limit и порядок лемм)

- **limit:** в `load_lemma_groups` применяется как `lemma_groups.head(limit)`. Порядок строк в `lemma_groups` задаётся функцией `group_examples_by_lemma` (итерация по `content["lemma"].unique()` — порядок первого появления в данных). **Сортировки по лемме нет**, поэтому при одном и том же `--limit N` в разные прогоны могут попадать разные наборы лемм. Это стоит исправить при рефакторинге (сортировать леммы перед обрезкой).
- **retry:** итерация по `sorted(lemmas_to_retry)` и `sorted(lemmas_still_empty)` — порядок retry детерминирован.

### 7.3. Структура batch_info.json и cards_B_batch.json (сейчас)

- **batch_info.json:** поля `batch_name`, `model`, `lemmas_count`, `created_at`, `uploaded_file`. Нет: schema_version, pipeline_version, tokens_path, sentences_path, max_examples, limit, thinking_model.
- **cards_B_batch.json:** корень — `pipeline`, `source`, `timestamp`, `config` (batch_name, lemmas_count, retry_*, cards_lemma_not_in_example_count), `stats` (lemmas_processed, cards_generated, errors, lemmas_with_zero_cards, validation_issues), `cards`, `errors`, `lemmas_with_zero_cards`. В карточках есть `lemma`, `definition_en`, `definition_ru`, `part_of_speech`, `examples`, `selected_example_indices`, `total_lemma_examples`, `source="pipeline_b_batch"`; при fallback — `examples_fallback`. Поля schema_version и структурированные errors[] (lemma, stage, error_type, message) пока нет.

### 7.4. Проверка «lemma not in example»

- **Где:** `_cards_lemma_not_in_example(all_cards)` в скрипте; используется `_get_word_forms(lemma)` и `_word_in_text(form, example_text)` из `eng_words.validation.example_validator`.
- **Что проверяем:** для каждой карточки и каждого примера — есть ли в примере **хотя бы одна словоформа леммы** (включая неправильные формы из словаря: ran, runs, running для run и т.д.). Исключаются «гомонимичные» формы (например hoped/hoping для леммы hop). Вхождение проверяется как **целое слово**: regex `\b` + escaped(word) + `\b`, поиск без учёта регистра (`re.IGNORECASE`). Частичные совпадения (например «running» в «overrunning») не считаются совпадением из‑за границ слова.
- **Куда пишется:** в `download_log.json` в массив `cards_lemma_not_in_example`; в консоль выводится предупреждение. Карточки при этом **не отбрасываются** — это именно warning для ручной проверки. Возможны false positives (апостроф, пунктуация, опечатка, форма не из словаря).

### 7.5. POS (part-of-speech) в текущем промпте и карточках

- В **промпте** (`CLUSTER_PROMPT_TEMPLATE` в clusterer.py) уже есть: «MANDATORY SPLITS: Different parts of speech (noun vs verb vs adjective)» и в JSON-шаблоне поле `part_of_speech`: "noun/verb/adj/adv". То есть модель уже просим различать POS и возвращать его на карточке.
- **Распределения POS по примерам** в промпт не передаётся; подсказки вида «Most occurrences are VERB» нет. Проверки соответствия примеров заявленному POS (QC mismatch) после ответа нет — это планируется в рамках варианта 1+ (см. § 9).

---

## 8. Учёт замечаний ревью: что вносим в план

| Замечание | Решение | Открытые вопросы / примечания |
|-----------|---------|-------------------------------|
| **(A) Pure core vs side effects** | Принимаем. В целевом состоянии явно разделяем pure-функции (build_prompt, parse_one_result, validate_card, choose_retry_candidates, merge_retry_results, compute_stats) и IO/integration (create_batch, download_results, call_standard_api_for_retry). | — |
| **(B) Схемы артефактов (BatchInfo, DownloadResult, schema_version)** | Принимаем. Вводим dataclass’ы и schema_version в batch_info и cards; расширяем поля по списку из ревью (tokens_path, sentences_path, max_examples, limit, thinking_model; структурированные errors). | Точный формат errors[] (stage, error_type) уточнить при реализации. |
| **(C) BatchPaths** | Принимаем. Один `batch_dir` + `BatchPaths.from_dir(batch_dir)` со стандартными именами файлов; точечное переопределение при необходимости. | — |
| **(D) Идемпотентность / resume** | Принимаем. create: при наличии batch_info без --overwrite — ошибка. download: при --from-file не трогать сеть; перезапись output — по умолчанию да, при желании позже --no-overwrite. Retry: опционально сохранять сырые ответы в download_log или retry_results.jsonl. | Сохранять ли retry raw в файл — решить при реализации (как минимум в download_log список лемм + статус retry). |
| **(E) Детерминизм** | Принимаем. Сортировать леммы перед применением limit; при любом семплировании — фиксированный seed. | — |
| **(F) POS** | Принимаем вариант 1+ (lemma-only + POS как hint + QC). Не делаем жёсткий split по lemma×POS; добавляем POS distribution в промпт, pos в карточке уже есть; после download — QC (mismatch) и warnings в download_log. Детали в § 9. | Пороги и формат pos_hint в промпте уточнить при реализации. |
| **Retry как стратегия** | Принимаем. Внутри модуля держать структуру retry_policy (modes, only_if); CLI-флаги мапятся в неё. | — |
| **lemma_not_in_example как warning** | Уже так: не блокируем, пишем в download_log. В доке и в коде явно описать проверку: словоформы + целое слово (regex \b, case-insensitive). § 7.4. | — |
| **CLI: render-requests, parse-results** | Принимаем. render-requests — только запись requests.jsonl + lemma_examples.json без API. parse-results — парсинг results.jsonl → cards_B_batch.json без retry и без сети. | parse-results может быть алиасом download --from-file --no-retry или отдельной командой; решить при реализации. |
| **CLI: --overwrite, --resume** | Принимаем. --overwrite для create (разрешить перезапись при существующем batch_info). --resume — использовать существующие артефакты, не пересоздавать (конкретика: при create не создавать новый job, если уже есть файлы?). | Точную семантику --resume зафиксировать при реализации (например: create с --resume только перегенерировать локальные файлы, не вызывать API). |
| **Тест-стратегия** | Принимаем. Unit: build_prompt (snapshot), parse_one_result (JSON/блок/мусор/индексы), validate_card, merge_retry_results, compute_stats. Интеграция: мок клиента, проверка артефактов и idempotency. Контракт: сериализация BatchInfo/DownloadResult и schema_version. | — |
| **Миграция: сначала обёртка** | Принимаем. Шаг 0: модуль как обёртка над скриптом; скрипт переходит на вызов модуля; затем перенос функций по одной с тестами и сравнением артефактов. | — |
| **limit в модуле: int \| None** | Принимаем. В модуле `limit: int | None` (None = все); в CLI 0 → передаём None. | — |
| **lemma_examples.json размер** | Откладываем. Сейчас храним полный текст примеров. В будущем можно хранить sentence_id и подставлять текст из sentences.parquet при сборке карточек — уменьшит дублирование. | Не в рамках первого рефакторинга. |

**Открытые решения на реализацию (уточнить при переносе кода):**

- Точный формат `errors[]` в cards_B_batch.json (поля stage, error_type, message и перечисление допустимых значений).
- Сохранять ли сырые ответы retry в файл (retry_results.jsonl или фрагмент в download_log) и в каком объёме.
- parse-results: отдельная команда или алиас `download --from-file` с флагом «без retry».
- Семантика --resume для create: «не создавать job в API, только перегенерировать локальные requests.jsonl + lemma_examples.json» или иной вариант.
- Формат pos_hint и POS distribution в промпте (шаблон фразы, пороги «most/some»).

---

## 9. POS (part-of-speech): принятое решение (вариант 1+)

После обсуждения вариантов (lemma-only, жёсткий split по lemma×POS, гибридный split) принят **вариант 1+**: lemma-only + POS как подсказка + QC-проверки.

### Почему не жёсткий split по POS (вариант 2)

- Ошибки POS-тэггера (Stage 1 / spaCy) приводят к тому, что примеры попадают «не в ту» корзину → модель генерирует карточки не по тем примерам (ошибка маршрутизации усиливается).
- Меньше контекста на запрос → выше нестабильность и галлюцинации.
- Больше запросов → дороже и сложнее retry/дебаг.

### Что делаем в варианте 1+

1. **Один запрос на лемму** (как сейчас) — не делим корзины по POS.
2. **В промпт добавляем POS distribution** по примерам (например: «In this set, most occurrences are VERB; some NOUN») и при необходимости краткий pos_hint («Primary POS in examples: verb»). Модель уже просим возвращать `part_of_speech` на каждой карточке — оставляем.
3. **После download — QC:** для каждой карточки проверять (по токенам), что выбранные примеры соответствуют заявленному POS; считать mismatch_rate, писать предупреждения в `download_log.json` (например `pos_mismatch_warnings`).
4. **Дальше по логам решить:** достаточно ли предупреждений для ручной выборки или нужны локальный ремап примеров (без LLM) / retry по pos_mismatch. Жёсткий split по lemma×POS не вводим на первом этапе.

### Практические шаги внедрения (без риска)

1. Добавить в промпт POS distribution (и при необходимости pos_hint); требование поля `pos` в ответе уже есть.
2. В этапе download добавить шаг QC (сверка примеров с POS по токенам), запись в download_log.
3. После просмотра логов решить: нужен ли ремап/retry по pos_mismatch и в каком виде.
