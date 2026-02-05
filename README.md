# English Words — инструмент для изучения слов

Персональное приложение для извлечения слов из книг, фильтрации уже известных и создания карточек Anki для изучения английского языка.

## Описание проекта

Инструмент помогает изучать английский через чтение книг:

1. **Извлечение слов** из текстовых файлов книг
2. **Анализ частотности** — находит наиболее часто встречающиеся слова
3. **Фильтрация**:
   - Исключает слова, которые вы уже знаете
   - Убирает слишком частые (базовые) слова
   - Убирает слишком редкие или устаревшие слова
4. **Ранжирование** кандидатов для изучения
5. **Генерация карточек Anki** (с интеграцией LLM)

## Архитектура

Проект построен как модульный пайплайн обработки текста:

```
Текст книги → Токенизация → Лемматизация → Статистика → Фильтрация → Ранжирование → Экспорт
```

### Ключевые компоненты

- **Токенизация и лемматизация**: spaCy для обработки текста
- **Частотный анализ**: комбинация локальной частоты (в книге) и глобальной (в языке)
- **Фильтрация**: исключение известных слов через CSV или Google Sheets
- **Фразовые глаголы**: отдельная обработка фразовых глаголов как особого типа сущностей
- **Экспорт**: генерация CSV для импорта в Anki

## Верхнеуровневый план

### Фаза 1: Базовая обработка текста
- Токенизация и лемматизация с помощью spaCy
- Извлечение статистики по словам
- Сохранение промежуточных результатов (parquet)

### Фаза 2: Фильтрация и ранжирование
- Интеграция с глобальной частотностью (wordfreq)
- Фильтрация по известным словам (CSV)
- Ранжирование кандидатов

### Фаза 3: Фразовые глаголы
- Детекция фразовых глаголов через dependency parsing
- Обработка и фильтрация фразовых глаголов

### Фаза 4: Экспорт и примеры
- Извлечение примеров предложений для каждого слова
- Экспорт в формат Anki (CSV)

### Фаза 5: Интеграция с LLM
- Генерация определений и переводов через LLM API
- Улучшенные карточки Anki с контекстом

## Принципы разработки

### 1. Модульность и тестируемость
- **Всё разбито на функции** — никаких монолитных скриптов
- **У каждой функции одна ответственность**
- **Функции легко тестировать изолированно**

### 2. Покрытие тестами
- **Каждая функция покрыта тестами**
- **Интеграции между функциями тоже тестируются**
- **Тесты пишутся параллельно с кодом**

### 3. Поэтапная разработка
- **Разработка ведётся небольшими логическими блоками**
- **После каждого блока: тестирование и отладка**
- **Только после стабильной работы блока переходим к следующему**
- **После каждого блока**: `git status` → `git add` → осмысленный коммит (Git-First Workflow)

### 4. Сохранение промежуточных результатов
- **Токены сохраняются в parquet** для быстрого доступа
- **Статистика по леммам сохраняется отдельно**
- **Позволяет переиспользовать результаты без переобработки**

### 5. Простота и ясность
- **Код должен быть понятен через год**
- **Пайплайн линейный и прозрачный**
- **Минимум магии, максимум явности**
- **Длинные тексты обрабатываются чанками** (~250k символов) с нормализацией (кавычки, апострофы, невидимые символы)
- **Известные слова отфильтровываются** из кандидатов через CSV
- **Частотная фильтрация и ранжирование** встроены в Stage 1
- **Фразовые глаголы** пишутся в отдельный parquet при необходимости
- **Кандидаты фильтруются по частоте** и ранжируются по score

## Структура проекта

```
eng_words/
├── src/eng_words/          # Код приложения
│   ├── __init__.py
│   ├── text_processing.py  # Токенизация, лемматизация
│   ├── statistics.py       # Частоты, статистика
│   ├── filtering.py       # Фильтрация по известным словам
│   ├── phrasal_verbs.py    # Обработка фразовых глаголов
│   └── ...
├── tests/                  # Тесты
├── data/                   # Данные
│   ├── raw/                # Исходные тексты книг
│   └── processed/          # Промежуточные результаты (parquet)
├── anki_exports/           # Экспортированные карточки
├── pyproject.toml         # Конфигурация проекта
└── README.md
```

## Входные данные

- Основной формат книг — **EPUB**
- Файлы лежат в `data/raw/` (например `data/raw/theodore-dreiser_an-american-tragedy.epub`)
- Поддержка других форматов (PDF и т.д.) будет добавляться по мере необходимости

## Установка и использование

### Требования
- Python 3.10+
- uv (для управления зависимостями)

### Установка

```bash
# Базовые зависимости (обязательно)
uv sync

# С опциональными экстрами (рекомендуется для полного функционала)
# - dev: тесты и инструменты разработки
# - wsd: Word Sense Disambiguation (sentence-transformers)
# - llm: провайдеры LLM для генерации умных карточек
uv sync --extra dev --extra llm --extra wsd
```

**Примечание:** Модель spaCy (`en_core_web_sm`) ставится автоматически — отдельно качать не нужно.

### Проверка установки

```bash
# Проверка доступа к модулю пайплайна
uv run python -m eng_words.pipeline --help

# Быстрый прогон тестов (если установлен dev)
uv run pytest tests/ -v -x

# Проверка импорта LLM (если установлен llm)
uv run python -c "from eng_words.llm import get_provider; print('✓ LLM модуль OK')"

# Генерация карточек Pipeline B (требуется GOOGLE_API_KEY)
uv run python scripts/run_pipeline_b_batch.py --help
```

### Использование

#### Полный пайплайн (текст → Anki CSV)

```bash
uv run python -m eng_words.pipeline \
  --book-path data/raw/theodore-dreiser_an-american-tragedy.epub \
  --book-name american_tragedy \
  --output-dir data/processed \
  --known-words data/known_words.csv \
  --min-book-freq 3 \
  --min-zipf 2.0 \
  --max-zipf 5.3 \
  --top-n 150 \
  --phrasal-model en_core_web_sm
```

Полезные флаги:
- `--top-n` — сколько лемм/фраз попадает в итоговый Anki CSV
- `--no-phrasals` — отключить обработку фразовых глаголов
- `--phrasal-model` — отдельная модель spaCy для фразовых (по умолчанию: `--model-name`)

Выходные файлы:
- `data/processed/american_tragedy_tokens.parquet`
- `data/processed/american_tragedy_lemma_stats_full.parquet`
- `data/processed/american_tragedy_lemma_stats.parquet`
- `data/processed/american_tragedy_phrasal_verbs.parquet` (если не `--no-phrasals`)
- `data/processed/american_tragedy_phrasal_verb_stats.parquet`
- `data/processed/anki_exports/american_tragedy_anki.csv` — готовый CSV для Anki

#### Генерация карточек с LLM (Pipeline B)

Генерация определений и примеров для карточек через Gemini Batch API: один запрос на лемму, кластеризация по смыслу. Логика — в модуле `eng_words.word_family`; CLI — `scripts/run_pipeline_b_batch.py`.

```bash
# Подготовка выборки (tokens_sample.parquet, sentences_sample.parquet)
uv run python scripts/prepare_pipeline_b_sample.py --book american_tragedy --size 100

# Создание батча, ожидание, скачивание (нужен GOOGLE_API_KEY)
uv run python scripts/run_pipeline_b_batch.py create [--limit N]
uv run python scripts/run_pipeline_b_batch.py status
uv run python scripts/run_pipeline_b_batch.py wait
uv run python scripts/run_pipeline_b_batch.py download
```

Подробнее: **[docs/PIPELINES_OVERVIEW.md](docs/PIPELINES_OVERVIEW.md)** — артефакты, команды (render-requests, parse-results, list-retry-candidates), strict/relaxed и QC.

### Обработка чанками и нормализация
- Текст автоматически режется на чанки (~250k символов), чтобы не упираться в лимиты spaCy и экономить память
- Перед токенизацией выполняется нормализация: удаляются невидимые символы, длинные тире и «умные» кавычки/апострофы заменяются на обычные
- Параметр `max_chars` для чанков можно менять в коде (`tokenize_text_in_chunks`)

### Экспорт для ручной проверки

Чтобы обновить CSV для ревью (например перед выгрузкой в Google Sheets):

```bash
uv run python scripts/export_for_review.py \
  --lemma-stats data/processed/american_tragedy_lemma_stats_full.parquet \
  --score-source data/processed/american_tragedy_lemma_stats.parquet \
  --status-source data/review_export_all_processed.csv \
  --output data/review_export_all_next.csv
```

- `--lemma-stats` — сырые частоты (все токены)
- `--score-source` — parquet с текущим `score`
- `--status-source` — предыдущий CSV с ручными метками; статусы и теги переносятся
- Если слово оканчивается на -ing, не используется как глагол в корпусе и базовая форма есть, экспорт ставит `status=ignore` и тег `auto_ing_variant`
- Если лемма помечена spaCy как стоп-слово, экспорт добавляет тег `auto_stopword`

На выходе — полный список слов и фраз с актуальными `book_freq`/`global_zipf` и score только для «чистых» кандидатов.

### Хранение известных слов

Поддерживаются два способа:

#### CSV (по умолчанию)
- Формат: `lemma,status,item_type,tags` (см. `known_words.csv.example`)
- Статусы: `known`, `ignore`, `learning`, `maybe` (фильтр использует первые два)
- Использование: `--known-words data/known_words.csv`

#### Google Sheets (для удобного редактирования)
- Синхронизация между устройствами
- Редактирование в браузере
- Использование: `--known-words gsheets://SPREADSHEET_ID/WORKSHEET_NAME`

**Настройка:** см. [docs/GOOGLE_SHEETS_SETUP.md](docs/GOOGLE_SHEETS_SETUP.md)

Кратко:
1. Создать Service Account в Google и скачать JSON-ключ
2. Создать таблицу с заголовками: `lemma,status,item_type,tags`
3. Дать доступ на редактирование email сервисного аккаунта
4. Задать переменную окружения `GOOGLE_APPLICATION_CREDENTIALS`
5. Указать в пайплайне `gsheets://SPREADSHEET_ID/WORKSHEET_NAME`

### Частотная фильтрация и ранжирование
- Пороги по книге (`--min-book-freq`), по глобальной частоте (`--min-zipf`, `--max-zipf`) отсекают слишком редкие и слишком частые слова
- После фильтрации добавляется `score`: нормализованная частота × бонус за редкость (центр около Zipf ≈ 4.0)
- Выходной parquet отсортирован по `score`

### Детекция фразовых глаголов
- Флаг `--detect-phrasals` включает поиск «глагол + частица» через dependency parsing spaCy
- Можно указать отдельную модель (`--phrasal-model`), иначе используется `--model-name`
- Результаты: `*_phrasal_verbs.parquet`, `*_phrasal_verb_stats.parquet`

### Word Sense Disambiguation (WSD)

Поддерживается различение значений слов в контексте:

```bash
uv run python -m eng_words.pipeline \
  --book-path data/raw/book.epub \
  --book-name my_book \
  --output-dir data/processed \
  --enable-wsd \
  --min-sense-freq 5 \
  --max-senses 3
```

**Что делает WSD:**
- Определяет конкретное значение слова в контексте (например «bank» = банк или берег)
- Группирует значения в 43 категории (supersenses): `noun.person`, `verb.motion` и т.д.
- Позволяет фильтровать и учить слова по значению

**Выходные файлы:**
- `{book}_sense_tokens.parquet` — токены с разметкой значения (`synset_id`, `supersense`, `sense_confidence`)
- `{book}_supersense_stats.parquet` — статистика (lemma, supersense) с частотами и долями

**Экспорт для ревью:** см. пример в документации по `scripts/export_for_review.py` с `--supersense-stats` и `--sense-tokens`.

**Подробнее:** [docs/WSD_GOLD_DATASET_USAGE.md](docs/WSD_GOLD_DATASET_USAGE.md)

### Генерация умных карточек (LLM)

Автоматическое создание карточек Anki с помощью LLM:

```bash
uv run python -m eng_words.pipeline \
  --book-path data/raw/book.epub \
  --book-name my_book \
  --output-dir data/processed \
  --enable-wsd \
  --smart-cards \
  --smart-cards-provider gemini \
  --top-n 500
```

**Что делает Smart Card Generator:**
- Выбирает 1–2 лучших примера из книги на слово
- Отсекает примеры с неподходящим значением (проверка WSD)
- Генерирует простое определение (уровень B1–B2)
- Даёт перевод для данного значения
- Генерирует дополнительное примерное предложение

**Провайдеры LLM:**
- `gemini` (по умолчанию) — Gemini 3 Flash
- `openai` — GPT-5-mini
- `anthropic` — Claude Haiku 4.5

**Выходные файлы:**
- `{book}_smart_cards.json` — полные данные карточек
- `anki_exports/{book}_smart_anki.csv` — готовый CSV для Anki

**Требования:**
- Для разметки значений нужен `--enable-wsd`
- API-ключ провайдера в `.env` (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- Кэширование: повторные запуски не делают лишних запросов к API для тех же входов

## Модель данных

### Хранение известных слов

Таблица личного словаря (CSV или Google Sheets):

| lemma | item_type | status | tags | ... |
|-------|-----------|--------|------|-----|
| run | word | known | A2, basic_verbs | ... |
| give up | phrasal_verb | learning | B1, phrasal | ... |

- `status`: `known`, `learning`, `ignore`, `maybe`
- `item_type`: `word`, `phrasal_verb`, `ngram`
- `tags`: уровни, книги, темы

### Промежуточные файлы

- `tokens.parquet` — все токены с метаданными
- `lemma_stats_full.parquet` — все леммы с сырыми `book_freq` и `global_zipf`
- `lemma_stats.parquet` — отфильтрованные кандидаты с `score`

## Технологии

- **spaCy**: NLP, лемматизация, dependency parsing
- **pandas**: работа с данными
- **wordfreq**: глобальная частотность слов
- **pyarrow**: Parquet
- **pytest**: тесты
- **sentence-transformers**: эмбеддинги для WSD (опционально)
- **nltk**: WordNet для значений слов (опционально)

## Лицензия

Личный проект для изучения английского.
