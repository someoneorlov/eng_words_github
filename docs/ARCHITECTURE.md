# Обзор архитектуры

Краткий обзор архитектуры пайплайна English Words: структура кодовой базы и поток данных.

## Поток данных (высокий уровень)

```
┌─────────────┐
│  EPUB книга │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Этап 1: Токенизация и статистика   │
│  - Извлечение текста из EPUB        │
│  - Токенизация и лемматизация (spaCy)│
│  - Расчёт частотной статистики       │
│  - Фильтрация по Zipf/известным словам│
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  WSD: Снятие неоднозначности смысла  │
│  - Разметка токенов synset'ами       │
│    WordNet (опционально)              │
│  - Агрегация по supersense           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Агрегация: группировка synset'ов   │
│  - Группировка synset'ов по смыслу   │
│  - Семантическая кластеризация (LLM)│
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Генерация карточек                  │
│  - Проверка примеров                 │
│  - Определения/переводы (LLM)        │
│  - Выбор и фильтрация примеров       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Anki CSV   │
└─────────────┘
```

## Ответственность модулей

### Ядро пайплайна (`src/eng_words/pipeline.py`)
- **Оркестрация этапа 1**: токенизация, расчёт статистики, фильтрация
- **Полный пайплайн**: от EPUB до Anki CSV (опционально)
- **Точка входа**: CLI `python -m eng_words.pipeline`

### Слой LLM (`src/eng_words/llm/`)
- **Провайдеры** (`llm/providers/`): абстракция для Gemini, OpenAI, Anthropic
- **Smart Card Generator** (`smart_card_generator.py`): генерация карточек с определениями, переводами, примерами
- **Response Cache** (`response_cache.py`): кэш ответов LLM для снижения затрат на API
- **Retry Logic** (`retry.py`): обработка временных сбоев и лимитов

### WSD (`src/eng_words/wsd/`)
- **WordNet Backend** (`wordnet_backend.py`): разметка токенов synset'ами WordNet
- **LLM WSD** (`llm_wsd.py`): использование LLM для снятия неоднозначности при неоднозначности WordNet
- **Candidate Selector** (`candidate_selector.py`): выбор кандидатов synset по эмбеддингам
- **Aggregator** (`aggregator.py`): агрегация статистики по supersense

### Агрегация (`src/eng_words/aggregation/`)
- **Synset Aggregator** (`synset_aggregator.py`): группировка synset'ов WordNet по частоте/статистике
- **LLM Aggregator** (`llm_aggregator.py`): семантическая группировка synset'ов в «семейства значений» с помощью LLM

### Валидация (`src/eng_words/validation/`)
- **Example Validator** (`example_validator.py`): проверка соответствия примеров из книги значению synset'а
- **Synset Validator** (`synset_validator.py`): проверка примеров для групп synset'ов с помощью LLM

### Хранение (`src/eng_words/storage/`)
- **Backends** (`backends.py`): бэкенды CSV и Google Sheets для списка известных слов
- **Loader** (`loader.py`): единый интерфейс загрузки/сохранения известных слов

### Константы (`src/eng_words/constants/`)
- **Конфигурация**: имена моделей, шаблоны файлов, колонки, значения по умолчанию
- **LLM Pricing** (`llm_pricing.py`): оценка стоимости по провайдерам/моделям
- **Supersenses** (`supersenses.py`): таксономия supersense WordNet

### Вспомогательные модули
- **Text Processing** (`text_processing.py`): токенизация, реконструкция предложений, интеграция со spaCy
- **Statistics** (`statistics.py`): расчёт частот, Zipf-скор
- **Filtering** (`filtering.py`): фильтрация по частоте, известным словам, supersense
- **Examples** (`examples.py`): извлечение и выбор примеров предложений из книги
- **Anki Export** (`anki_export.py`): экспорт карточек в формат Anki CSV
- **EPUB Reader** (`epub_reader.py`): извлечение текста из EPUB

## Поток данных

### Вход
- **Книги**: `data/raw/*.epub` — исходные EPUB
- **Известные слова**: `data/known_words.csv` или URL Google Sheets

### Промежуточные файлы (`data/processed/`)
- `{book_name}_tokens.parquet` — токены с леммами, POS, ID предложений
- `{book_name}_lemma_stats.parquet` — частотная статистика по леммам
- `{book_name}_sense_tokens.parquet` — токены с synset'ами WordNet (если включён WSD)
- `{book_name}_supersense_stats.parquet` — агрегированная статистика по supersense

### Выход
- **Карточки synset**: `data/synset_cards/synset_smart_cards_final.json` — сгенерированные карточки в JSON
- **Anki CSV**: `data/synset_cards/synset_anki.csv` или `anki_exports/` — колода для импорта в Anki
- **Кэш LLM**: `data/synset_cards/llm_cache/` — кэш ответов LLM

## Точки входа

### Полный сценарий для новой книги

```bash
# Шаг 1: Токенизация, статистика, WSD
uv run python -m eng_words.pipeline \
  --book-path data/raw/book.epub \
  --book-name my_book \
  --enable-wsd

# Шаг 2: Агрегация по synset (LLM)
uv run python scripts/run_full_synset_aggregation.py

# Шаг 3: Генерация карточек и экспорт в Anki
uv run python scripts/run_synset_card_generation.py
```

### Основные скрипты

**Этап 1: Токенизация и WSD** (`python -m eng_words.pipeline`):
```bash
uv run python -m eng_words.pipeline --book-path data/raw/book.epub --book-name book --enable-wsd
```
- Извлечение текста из EPUB
- Токенизация и расчёт частот
- Разметка synset'ами WordNet (WSD)
- Выход: `{book}_tokens.parquet`, `{book}_sense_tokens.parquet`

**Этап 2: Агрегация synset'ов** (`scripts/run_full_synset_aggregation.py`):
```bash
uv run python scripts/run_full_synset_aggregation.py
```
- Группировка токенов по (lemma, synset_id)
- Кластеризация связанных synset'ов с помощью LLM
- Выход: `aggregated_cards.parquet`

**Этап 3: Генерация карточек** (`scripts/run_synset_card_generation.py`):
```bash
uv run python scripts/run_synset_card_generation.py [limit]
```
- Загрузка агрегированных карточек
- Генерация умных карточек с LLM
- Проверка примеров
- Экспорт в Anki CSV

**Только этап 1** (`python -m eng_words.pipeline`):
```bash
uv run python -m eng_words.pipeline \
  --book-path data/raw/book.epub \
  --book-name book_name \
  --output-dir data/processed \
  --known-words data/known_words.csv \
  --enable-wsd
```
- Токенизация и статистика
- Опциональная WSD-разметка
- Извлечение примеров
- Экспорт в Anki (при полном пайплайне)

### Тестирование
```bash
make check          # Формат + линт + тесты
make test           # Все тесты
make test-wsd       # Только тесты WSD
```

### Прочие скрипты
- `scripts/eval_wsd_on_gold.py` — оценка точности WSD на gold-датасете
- `scripts/verify_gold_checksum.py` — проверка целостности gold-датасета
- `scripts/benchmark_wsd.py` — бенчмарк WSD

## Ключевые решения

1. **Модульность**: этапы (токенизация, WSD, агрегация, генерация) независимы и могут запускаться по отдельности
2. **Абстракция LLM**: интерфейс, не привязанный к провайдеру, позволяет переключаться между Gemini, OpenAI, Anthropic
3. **Кэширование**: ответы LLM кэшируются для снижения затрат и возможности возобновления генерации
4. **Формат Parquet**: промежуточные данные в Parquet для эффективного I/O и возобновляемости
5. **Валидация по этапам**: многоступенчатая проверка качества карточек (соответствие synset'у, длина примеров, детекция спойлеров)

## Зависимости

- **spaCy**: токенизация, POS-разметка, лемматизация
- **WordNet**: снятие неоднозначности смысла
- **sentence-transformers**: выбор кандидатов по эмбеддингам (опционально)
- **LLM API**: Gemini, OpenAI, Anthropic (через абстракцию провайдеров)
- **pandas**: обработка данных и I/O Parquet
