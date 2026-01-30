# История разработки v1.0

Документ содержит полную историю разработки проекта English Words Learning Tool, включая все реализованные этапы 0-8 и последующие улучшения.

## Общие принципы разработки

1. **Модульность**: Каждая функция решает одну задачу
2. **Тестирование**: Тесты пишутся параллельно с кодом
3. **Поэтапность**: Разработка небольшими блоками с тестированием после каждого
4. **Сохранение результатов**: Промежуточные данные сохраняются в parquet
5. **Простота**: Код должен быть понятен через год

---

## Этап 0: Подготовка инфраструктуры ✅

**Статус:** Завершен

- [x] Инициализация git репозитория
- [x] Создание pyproject.toml
- [x] Создание виртуального окружения (uv venv)
- [x] Создание структуры директорий
- [x] Настройка .gitignore
- [x] Создание README.md
- [x] Создание DEVELOPMENT_PLAN.md

---

## Этап 1: Базовая обработка текста ✅

**Статус:** Завершен

### Задача 1.1: Загрузка и предобработка текста ✅

**Реализованные функции:**
- `load_text_from_file(file_path: Path, encoding: str = "utf-8") -> str`
- `load_book_text(file_path: Path, *, normalize_newlines: bool = True) -> str`
- `preprocess_text(text: str, normalize_newlines: bool = True) -> str`
- `normalize_headers(text: str) -> str` (добавлено позже)

**Реализованные возможности:**
- Загрузка текста из файла с обработкой ошибок
- Поддержка EPUB и TXT форматов
- Предобработка: удаление BOM, нормализация newlines
- Нормализация пунктуации (умные кавычки, тире, апострофы)
- Нормализация заголовков (Book I, Chapter 1, Part II → Book I., Chapter 1., Part II.)
- Удаление невидимых Unicode символов

**Тесты:** `tests/test_text_io.py`
- Загрузка текста из файла
- Обработка пустого файла
- Обработка файла с разными кодировками
- Нормализация заголовков
- Нормализация пунктуации

**Файлы:**
- `src/eng_words/text_io.py`
- `src/eng_words/epub_reader.py`

---

### Задача 1.2: Токенизация и лемматизация с spaCy ✅

**Реализованные функции:**
- `initialize_spacy_model(model_name: str = "en_core_web_sm") -> Language`
- `tokenize_and_lemmatize(text: str, nlp: Language, *, sentence_offset: int = 0) -> List[TokenDict]`
- `create_tokens_dataframe(tokens: List[TokenDict], book_name: str) -> pd.DataFrame`
- `tokenize_text_in_chunks(text: str, nlp: Language, *, max_chars: int = 250_000) -> List[TokenDict]`
- `iterate_book_chunks(text: str, max_chars: int = 250_000) -> Iterator[str]`

**Структура DataFrame:**
```python
columns = [
    'book',           # название книги
    'sentence_id',    # ID предложения
    'position',       # позиция в предложении
    'surface',        # реальная форма слова
    'lemma',          # лемма (lowercase)
    'pos',            # часть речи
    'is_stop',        # стоп-слово или нет
    'is_alpha',       # только буквы
    'whitespace',     # пробелы после токена (добавлено позже)
]
```

**Реализованные возможности:**
- Токенизация и лемматизация с помощью spaCy
- Обработка больших текстов через chunking (≈250k символов)
- Сохранение whitespace информации для точной реконструкции
- Поддержка sentence_offset для обработки чанков

**Тесты:** `tests/test_text_processing.py`
- Токенизация простого предложения
- Корректность лемматизации
- Обработка стоп-слов и пунктуации
- Создание DataFrame с правильной структурой
- Обработка больших текстов через chunking
- Сохранение whitespace

**Файлы:**
- `src/eng_words/text_processing.py`

---

### Задача 1.3: Сохранение токенов в parquet ✅

**Реализованные функции:**
- `save_tokens_to_parquet(tokens_df: pd.DataFrame, output_path: Path) -> None`
- `load_tokens_from_parquet(file_path: Path) -> pd.DataFrame`

**Тесты:** `tests/test_text_processing.py`
- Сохранение и загрузка DataFrame
- Сохранение пустого DataFrame
- Корректность данных после загрузки

**Файлы:**
- `src/eng_words/text_processing.py`

---

## Этап 2: Статистика и частотный анализ ✅

**Статус:** Завершен

### Задача 2.1: Подсчет частоты лемм в книге ✅

**Реализованные функции:**
- `calculate_lemma_frequency(tokens_df: pd.DataFrame) -> pd.DataFrame`

**Структура результата:**
```python
columns = [
    'lemma',          # лемма
    'book_freq',      # частота в книге
    'doc_count',      # количество предложений, где встречается
    'verb_count',     # сколько раз встретилось как VERB (добавлено в Stage 9)
    'other_pos_count',# сколько раз встретилось как не-VERB (добавлено в Stage 9)
    'stopword_count', # сколько раз встретилось как стоп-слово (добавлено в Stage 9)
]
```

**Фильтры:**
- Исключение стоп-слов (`is_stop == True`)
- Исключение имен собственных (`pos == 'PROPN'`)
- Только буквенные токены (`is_alpha == True`)

**Реализованные возможности:**
- Эффективный подсчет частоты с использованием Counter
- Корректный подсчет doc_count (уникальные предложения)
- Поддержка chunked processing
- Учёт части речи и стоп-слов для последующей фильтрации

**Тесты:** `tests/test_statistics.py`
- Подсчет частоты для простого текста
- Корректность фильтрации стоп-слов
- Корректность фильтрации имен собственных
- Группировка по леммам
- Корректность doc_count
- Корректность POS-счетчиков

**Файлы:**
- `src/eng_words/statistics.py`

---

### Задача 2.2: Интеграция глобальной частотности (wordfreq) ✅

**Реализованные функции:**
- `add_global_frequency(lemma_df: pd.DataFrame) -> pd.DataFrame`

**Добавляемая колонка:**
- `global_zipf`: Zipf частота слова в английском языке (0-7)

**Реализованные возможности:**
- Интеграция с библиотекой wordfreq
- Обработка слов, отсутствующих в wordfreq (NaN или 0)

**Тесты:** `tests/test_statistics.py`
- Добавление глобальной частоты для известных слов
- Обработка слов, которых нет в wordfreq
- Корректность значений Zipf частоты

**Файлы:**
- `src/eng_words/statistics.py`

---

### Задача 2.3: Сохранение статистики в parquet ✅

**Реализованные функции:**
- `save_lemma_stats_to_parquet(stats_df: pd.DataFrame, output_path: Path) -> None`
- `load_lemma_stats_from_parquet(file_path: Path) -> pd.DataFrame`

**Тесты:** `tests/test_statistics.py`
- Сохранение и загрузка статистики
- Корректность данных после загрузки

**Файлы:**
- `src/eng_words/statistics.py`

---

## Этап 3: Фильтрация по известным словам ✅

**Статус:** Завершен

### Задача 3.1: Загрузка списка известных слов из CSV ✅

**Реализованные функции:**
- `load_known_words_from_csv(csv_path: Path) -> pd.DataFrame`

**Ожидаемая структура CSV:**
```csv
lemma,status,item_type,tags
run,known,word,A2
give up,learning,phrasal_verb,B1
```

**Реализованные возможности:**
- Загрузка CSV с правильной структурой
- Обработка пустого CSV
- Обработка CSV с отсутствующими колонками
- Обработка дубликатов
- Нормализация лемм (lowercase)

**Тесты:** `tests/test_filtering.py`
- Загрузка CSV с правильной структурой
- Обработка пустого CSV
- Обработка CSV с отсутствующими колонками
- Обработка дубликатов

**Файлы:**
- `src/eng_words/filtering.py`

---

### Задача 3.2: Фильтрация кандидатов по известным словам ✅

**Реализованные функции:**
- `filter_known_words(candidates_df: pd.DataFrame, known_df: pd.DataFrame) -> pd.DataFrame`
- `filter_by_status(candidates_df: pd.DataFrame, known_df: pd.DataFrame, exclude_statuses: List[str] = ["known", "ignore"]) -> pd.DataFrame`

**Логика:**
- Исключение слов со статусом `known`, `ignore`
- Оставление слов со статусом `learning`, `maybe` или отсутствующих в списке

**Тесты:** `tests/test_filtering.py`
- Фильтрация известных слов
- Фильтрация по разным статусам
- Обработка слов, отсутствующих в списке известных
- Корректность join операции

**Файлы:**
- `src/eng_words/filtering.py`

---

## Этап 4: Фильтрация по частоте и ранжирование ✅

**Статус:** Завершен

### Задача 4.1: Фильтрация по частоте в книге и глобальной частоте ✅

**Реализованные функции:**
- `filter_by_frequency(stats_df: pd.DataFrame, min_book_freq: int = 3, min_zipf: float = 2.0, max_zipf: float = 5.5) -> pd.DataFrame`

**Параметры:**
- `min_book_freq`: минимальная частота в книге (по умолчанию 3)
- `min_zipf`: минимальная глобальная частота (исключает очень редкие)
- `max_zipf`: максимальная глобальная частота (исключает очень частые)

**Тесты:** `tests/test_filtering.py`
- Фильтрация по минимальной частоте в книге
- Фильтрация по глобальной частоте
- Комбинированная фильтрация
- Граничные случаи

**Файлы:**
- `src/eng_words/filtering.py`

---

### Задача 4.2: Ранжирование кандидатов ✅

**Реализованные функции:**
- `calculate_rarity_score(zipf_freq: float, target_zipf: float = 4.0) -> float`
- `rank_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame`

**Логика ранжирования:**
```python
# Нормализация частоты в книге
book_freq_norm = book_freq / max(book_freq)

# Буст для "умеренно редких" слов (пик около zipf ~ 4.0)
rarity_boost = exp(-((zipf - 4.0)^2) / 2)

# Итоговый score
score = book_freq_norm * rarity_boost
```

**Добавляемая колонка:**
- `score`: итоговый score для ранжирования

**Тесты:** `tests/test_filtering.py`
- Расчет rarity score для разных значений
- Ранжирование по score
- Корректность нормализации

**Файлы:**
- `src/eng_words/filtering.py`

---

## Этап 5: Фразовые глаголы ✅

**Статус:** Завершен

### Задача 5.1: Детекция фразовых глаголов ✅

**Реализованные функции:**
- `initialize_phrasal_model(model_name: str = "en_core_web_sm") -> Language`
- `detect_phrasal_verbs(tokens_df: pd.DataFrame, nlp: Language) -> pd.DataFrame`

**Логика:**
- Использование dependency parsing spaCy
- Поиск токенов с `pos == "VERB"` и дочерними токенами с `dep == "prt"` (particle)
- Формирование ключа как `"{lemma_verb} {particle}"`

**Структура результата:**
```python
columns = [
    'book',
    'sentence_id',
    'phrasal',        # "give up", "turn on"
    'verb',           # "give", "turn"
    'particle',       # "up", "on"
    'sentence_text',  # текст предложения
]
```

**Реализованные возможности:**
- Точная реконструкция предложений с использованием whitespace
- Удаление дубликатов

**Тесты:** `tests/test_phrasal_verbs.py`
- Детекция простых фразовых глаголов
- Обработка предложений без фразовых глаголов
- Корректность формирования ключа

**Файлы:**
- `src/eng_words/phrasal_verbs.py`

---

### Задача 5.2: Статистика по фразовым глаголам ✅

**Реализованные функции:**
- `calculate_phrasal_verb_stats(phrasal_df: pd.DataFrame) -> pd.DataFrame`

**Структура результата:**
```python
columns = [
    'phrasal',        # "give up"
    'book_freq',      # частота в книге
    'item_type',      # "phrasal_verb"
]
```

**Тесты:** `tests/test_phrasal_verbs.py`
- Подсчет частоты фразовых глаголов
- Группировка одинаковых фраз

**Файлы:**
- `src/eng_words/phrasal_verbs.py`

---

### Задача 5.3: Фильтрация и ранжирование фразовых глаголов ✅

**Реализованные функции:**
- `filter_phrasal_verbs(phrasal_stats_df: pd.DataFrame, known_df: pd.DataFrame | None, *, min_freq: int = 2) -> pd.DataFrame`
- `rank_phrasal_verbs(phrasal_stats_df: pd.DataFrame) -> pd.DataFrame`

**Логика:**
- Аналогично обычным словам, но с более низким порогом частоты (фразовые глаголы встречаются реже)
- Фильтрация по известным словам (join по `phrasal`)
- Ранжирование по частоте в книге

**Тесты:** `tests/test_phrasal_verbs.py`
- Фильтрация известных фразовых глаголов
- Ранжирование по частоте
- Комбинирование с обычными словами

**Файлы:**
- `src/eng_words/phrasal_verbs.py`

---

## Этап 6: Извлечение примеров предложений ✅

**Статус:** Завершен

### Задача 6.1: Извлечение предложений из текста ✅

**Реализованные функции:**
- `extract_sentences(text: str, nlp: Language) -> List[str]`
- `create_sentences_dataframe(sentences: List[str]) -> pd.DataFrame`
- `reconstruct_sentences_from_tokens(tokens_df: pd.DataFrame) -> List[str]` (добавлено позже)

**Структура DataFrame:**
```python
columns = [
    'sentence_id',
    'sentence',
]
```

**Реализованные возможности:**
- Реконструкция предложений из токенов с использованием whitespace для точного восстановления
- Извлечение предложений из текста через spaCy (альтернативный метод)

**Тесты:** `tests/test_sentences.py`, `tests/test_text_processing.py`
- Извлечение предложений из простого текста
- Обработка текста с разными типами предложений
- Корректность нумерации
- Реконструкция из токенов с whitespace

**Файлы:**
- `src/eng_words/text_processing.py`

---

### Задача 6.2: Сопоставление примеров с леммами ✅

**Реализованные функции:**
- `get_examples_for_lemmas(candidates_df: pd.DataFrame, tokens_df: pd.DataFrame, sentences_df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame`
- `_select_optimal_sentence(sentence_ids: List[int], sentence_lookup: Dict[int, str], *, min_length: int = 50, max_length: int = 150, fallback_min: int = 20, fallback_max: int = 300) -> Optional[str]`
- `_normalize_quote_spacing(sentence: str) -> str` (добавлено позже)

**Логика:**
- Для каждой леммы найти предложения, где она встречается
- Выбрать предложение оптимальной длины (50-150 символов предпочтительно)
- Нормализация пробелов после кавычек
- Добавить колонку `example` в candidates_df

**Реализованные возможности:**
- Выбор оптимальной длины вместо самого короткого предложения
- Fallback логика для случаев, когда нет предложений в предпочтительном диапазоне
- Автоматическая нормализация пробелов после кавычек

**Тесты:** `tests/test_examples.py`
- Сопоставление примеров для известных лемм
- Обработка лемм без примеров
- Выбор оптимальной длины предложения
- Нормализация пробелов после кавычек

**Файлы:**
- `src/eng_words/examples.py`

---

### Задача 6.3: Примеры для фразовых глаголов ✅

**Реализованные функции:**
- `get_examples_for_phrasal_verbs(phrasal_df: pd.DataFrame, sentences_df: pd.DataFrame) -> pd.DataFrame`

**Логика:**
- Аналогично обычным словам, но поиск по фразе целиком
- Использование той же логики выбора оптимальной длины

**Тесты:** `tests/test_examples.py`
- Извлечение примеров для фразовых глаголов
- Корректность поиска фразы в предложении

**Файлы:**
- `src/eng_words/examples.py`

---

## Этап 7: Экспорт в Anki ✅

**Статус:** Завершен

### Задача 7.1: Формирование данных для Anki ✅

**Реализованные функции:**
- `prepare_anki_export(candidates_df: pd.DataFrame, book_name: str) -> pd.DataFrame`

**Структура результата:**
```python
columns = [
    'front',      # слово/фраза
    'back',       # пример предложения
    'tags',       # название книги
]
```

**Тесты:** `tests/test_anki_export.py`
- Формирование правильной структуры
- Обработка пустого DataFrame
- Корректность тегов

**Файлы:**
- `src/eng_words/anki_export.py`

---

### Задача 7.2: Экспорт в CSV для Anki ✅

**Реализованные функции:**
- `export_to_anki_csv(anki_df: pd.DataFrame, output_path: Path) -> None`

**Формат CSV:**
- Простой CSV с колонками `front`, `back`, `tags`
- UTF-8 кодировка
- Совместим с импортом Anki

**Тесты:** `tests/test_anki_export.py`
- Экспорт в CSV
- Корректность формата
- Чтение обратно для проверки

**Файлы:**
- `src/eng_words/anki_export.py`

---

## Этап 8: Интеграция всего пайплайна ✅

**Статус:** Завершен

### Задача 8.1: Главный скрипт пайплайна ✅

**Реализованные функции:**
- `process_book_stage1` и `process_book` (полная сигнатура в коде)

**Логика:**
1. Загрузка текста
2. Токенизация и лемматизация (с chunking для больших текстов)
3. Сохранение токенов
4. Подсчет статистики (полной и фильтрованной)
5. Добавление глобальной частоты
6. Сохранение статистики (полной и фильтрованной)
7. Загрузка известных слов
8. Фильтрация по известным словам
9. Фильтрация по частоте
10. Ранжирование
11. Детекция фразовых глаголов (опционально)
12. Статистика по фразовым глаголам (опционально)
13. Реконструкция предложений из токенов
14. Извлечение примеров
15. Экспорт в Anki

**Реализованные возможности:**
- Полный end-to-end пайплайн
- Сохранение всех промежуточных результатов
- Поддержка опциональной детекции фразовых глаголов
- Реконструкция предложений из токенов для точного whitespace

**Тесты:** `tests/test_pipeline_stage1.py`, `tests/test_integration.py`
- Интеграционный тест всего пайплайна
- Обработка ошибок на разных этапах
- Обработка больших текстов
- Детекция фразовых глаголов

**Файлы:**
- `src/eng_words/pipeline.py`

---

### Задача 8.2: CLI интерфейс ✅

**Реализованные функции:**
- `run_stage1_cli() -> None`
- `run_full_pipeline_cli() -> None`

**Параметры CLI:**
- `--book-path`: путь к книге (EPUB или TXT)
- `--book-name`: название книги
- `--output-dir`: директория для результатов
- `--known-words`: путь к CSV с известными словами
- `--model-name`: spaCy модель для токенизации (по умолчанию en_core_web_sm)
- `--phrasal-model`: spaCy модель для детекции фразовых глаголов
- `--min-book-freq`: минимальная частота в книге (по умолчанию 3)
- `--min-zipf`: минимальная глобальная частота (по умолчанию 2.0)
- `--max-zipf`: максимальная глобальная частота (по умолчанию 5.5)
- `--top-n`: количество топ кандидатов для Anki (по умолчанию 100)
- `--detect-phrasals`: включить детекцию фразовых глаголов (для stage1)
- `--no-phrasals`: отключить детекцию фразовых глаголов (для full pipeline)

**Тесты:** `tests/test_pipeline_stage1.py`
- Парсинг аргументов
- Вызов пайплайна с разными параметрами

**Файлы:**
- `src/eng_words/pipeline.py`

---

## Этап 9: Улучшения UI и данных (Google Sheets & Review Export) ✅

**Статус:** Завершен

### Задача 9.1: Google Sheets Integration ✅

**Реализованные возможности:**
- Поддержка Google Sheets как бэкенда для `known_words`.
- Абстракция `KnownWordsBackend` (CSV/GSheets).
- Автоматическое определение типа источника по URL (`gsheets://...`) или пути файла.
- Документация по настройке сервисного аккаунта.

**Файлы:**
- `src/eng_words/storage/` (модуль хранения)
- `docs/GOOGLE_SHEETS_SETUP.md`

### Задача 9.2: Экспорт для ручного ревью (Review Export) ✅

**Реализованные возможности:**
- Скрипт `scripts/export_for_review.py` для создания CSV со всеми токенами.
- Сохранение "сырой" статистики (`*_lemma_stats_full.parquet`) в пайплайне:
  - Включает стоп-слова, имена собственные, редкие слова.
  - Содержит честные `book_freq` и `global_zipf`.
  - Содержит счетчики POS (`verb_count`, `other_pos_count`, `stopword_count`).
- Автоматические теги:
  - `auto_ing_variant`: для форм на -ing, которые не являются глаголами (и есть базовая форма).
  - `auto_stopword`: для стоп-слов.
- Автоматический перенос статусов из предыдущей разметки.

**Файлы:**
- `scripts/export_for_review.py`
- `src/eng_words/statistics.py` (обновлен расчет частот)

---

## Этап 10: Word Sense Disambiguation (WSD) ✅

**Статус:** Завершен

### Задача 10.1: Базовая инфраструктура WSD ✅

**Реализованные модули:**
- `src/eng_words/wsd/base.py` — абстракция `SenseBackend` и `SenseAnnotation`
- `src/eng_words/wsd/embeddings.py` — работа с Sentence-Transformers
- `src/eng_words/wsd/wordnet_utils.py` — утилиты для работы с WordNet
- `src/eng_words/constants/supersenses.py` — 43 supersense категории

**Ключевые компоненты:**
- `EmbeddingModel` — singleton для модели эмбеддингов (all-mpnet-base-v2)
- `DefinitionEmbeddingCache` — кэширование эмбеддингов определений WordNet
- Маппинг WordNet lexname → supersense (43 категории)

**Тесты:** `tests/test_wsd_base.py`, `tests/test_wsd_embeddings.py`, `tests/test_wsd_wordnet.py`

### Задача 10.2: WordNetSenseBackend ✅

**Реализованные возможности:**
- `disambiguate_word()` — определение значения одного слова
- `disambiguate_batch()` — batch обработка для эффективности
- `annotate()` — аннотация DataFrame токенов с progress bar и checkpoint
- `aggregate()` — агрегация статистики по (lemma, supersense)

**Алгоритм:**
1. Получение эмбеддинга предложения
2. Получение всех synsets для леммы из WordNet
3. Вычисление эмбеддингов определений (с кэшированием)
4. Выбор synset с максимальным косинусным сходством
5. Маппинг на supersense категорию

**Тесты:** `tests/test_wsd_wordnet_backend.py` (27 тестов)

### Задача 10.3: Интеграция в пайплайн ✅

**Реализованные возможности:**
- Флаг `--enable-wsd` в CLI
- Параметры `--min-sense-freq` и `--max-senses` для фильтрации
- Сохранение `{book}_sense_tokens.parquet` и `{book}_supersense_stats.parquet`
- Progress bar и checkpoint для длительных операций

**Тесты:** `tests/test_pipeline_wsd.py` (9 тестов), `tests/test_wsd_integration.py` (10 тестов)

### Задача 10.4: Фильтрация и экспорт ✅

**Реализованные возможности:**
- `filter_by_supersense()` — фильтрация по частоте значения и ограничение количества значений
- `export_for_review_with_supersenses()` — экспорт для ручной разметки с группировкой по значениям
- Поддержка определений WordNet в экспорте

**Тесты:** `tests/test_filtering_supersenses.py` (8 тестов), обновлены тесты `test_export_for_review.py`

**Результаты тестирования на 5 главах:**
- 17,667 токенов обработано за ~35 секунд
- 6,684 контентных слов аннотировано
- 3,060 уникальных пар (lemma, supersense)
- 430 многозначных слов обнаружено
- Максимум: 9 значений на слово (GET, MAKE)

**Файлы:**
- `src/eng_words/wsd/` — весь модуль WSD
- `src/eng_words/wsd/aggregator.py` — агрегация статистики
- `docs/WSD_USAGE.md` — подробная документация

---

## Дополнительные улучшения, реализованные в процессе разработки

### Улучшения парсинга

1. **Сохранение whitespace в токенах**
   - Добавлена колонка `whitespace` в токены для точной реконструкции предложений
   - Решает проблему с пробелами после кавычек и другой пунктуации

2. **Реконструкция предложений из токенов**
   - Функция `reconstruct_sentences_from_tokens` для точной реконструкции с использованием whitespace
   - Используется вместо повторного парсинга текста

3. **Нормализация заголовков**
   - Функция `normalize_headers` для обработки заголовков типа "Book I", "Chapter 1"
   - Автоматическое добавление точек в конце заголовков

4. **Выбор оптимальной длины примеров**
   - Заменен выбор самого короткого предложения на выбор оптимальной длины (50-150 символов)
   - Более информативные примеры для изучения

5. **Нормализация пробелов после кавычек**
   - Функция `_normalize_quote_spacing` для исправления пробелов после открывающих кавычек
   - Автоматически применяется ко всем примерам

### Обработка больших текстов

- Реализован chunking для обработки больших текстов (≈250k символов на чанк)
- Корректная обработка sentence_id при chunked processing
- Увеличен max_length для spaCy моделей до 2.5M символов

### Тестирование

- Полное покрытие тестами всех модулей
- Интеграционные тесты для всего пайплайна
- 71 тест, все проходят успешно

---

## Итоговая структура проекта

```
eng_words/
├── src/eng_words/
│   ├── __init__.py
│   ├── anki_export.py          # Экспорт в Anki
│   ├── constants/              # Константы (модульная структура)
│   ├── epub_reader.py          # Чтение EPUB
│   ├── examples.py             # Извлечение примеров
│   ├── filtering.py            # Фильтрация и ранжирование
│   ├── phrasal_verbs.py        # Фразовые глаголы
│   ├── pipeline.py             # Главный пайплайн
│   ├── statistics.py           # Статистика и частоты
│   ├── storage/                # Бэкенды хранения (CSV/GSheets)
│   ├── text_io.py              # Загрузка и предобработка
│   ├── text_processing.py      # Токенизация и лемматизация
│   └── wsd/                    # Word Sense Disambiguation
│       ├── __init__.py
│       ├── base.py             # Абстракции (SenseBackend, SenseAnnotation)
│       ├── embeddings.py       # Sentence-Transformers интеграция
│       ├── wordnet_utils.py    # Утилиты WordNet
│       ├── wordnet_backend.py  # Основной WSD backend
│       └── aggregator.py       # Агрегация статистики
├── scripts/
│   ├── export_for_review.py    # Экспорт для ручной разметки
│   ├── upload_review_to_gsheets.py
│   └── ...
├── tests/
│   ├── ...                     # Тесты для всех модулей
├── data/
│   ├── raw/                    # Исходные книги
│   └── processed/              # Промежуточные результаты
├── anki_exports/               # Экспортированные карточки
├── notebooks/                  # Jupyter ноутбуки
├── docs/                       # Документация
├── pyproject.toml
└── README.md
```

---

## Статистика проекта

- **Модулей:** 15+
- **Тестов:** 246+ (213 passed, 26 skipped)
- **Этапов:** 11 (все завершены)
- **Задач:** 40+ (все выполнены)

### WSD модуль

- **Файлов:** 6 модулей + тесты
- **Тестов:** 60+ (base, embeddings, wordnet, backend, integration, filtering, export)
- **Supersenses:** 43 категории (25 noun, 15 verb, 2 adj, 1 adv)
- **Производительность:** ~500-1000 токенов/сек

---

## Этап 12: LLM интеграция для WSD Evaluation и Card Generation ✅

**Статус:** Завершен

### Задача 12.1: Базовая инфраструктура LLM ✅

**Реализованные модули:**
- `src/eng_words/llm/base.py` — абстракция `LLMProvider` и `LLMResponse`
- `src/eng_words/llm/providers/openai.py` — OpenAI provider (GPT-4o-mini, GPT-4o, GPT-4.1-mini)
- `src/eng_words/llm/providers/anthropic.py` — Anthropic provider (Claude Haiku)
- `src/eng_words/constants/llm_config.py` — конфигурация LLM (модели, температуры, промпт-версии)

**Реализованные возможности:**
- Абстракция провайдеров LLM для поддержки разных API
- Поддержка JSON-ответов с валидацией схемы
- Подсчет токенов и стоимости запросов
- Детерминизм через `seed=42` и `temperature=0`
- Версионирование промптов для кэширования

**Тесты:** `tests/test_llm_base.py`, `tests/test_llm_providers.py`

---

### Задача 12.2: WSD Evaluation через LLM Jury ✅

**Реализованные модули:**
- `src/eng_words/llm/evaluator.py` — WSD Evaluator с jury voting
- `src/eng_words/llm/prompts.py` — промпты для evaluation
- `scripts/evaluate_wsd.py` — CLI для запуска evaluation

**Реализованные возможности:**
- Blind evaluation (LLM не видит assigned synset, только candidates A/B/C)
- Jury voting (2-3 модели голосуют независимо)
- Confidence на основе согласия jury (не self-reported)
- Стратифицированная выборка по supersense
- Агрегация метрик: accuracy, coverage, uncertainty rate

**Результаты:**
- Проанализировано 394 примера (2 батча по 197)
- Найдено 5 ошибок (~1.27% error rate)
- LLM detection rate: ~66.7% (2/3 ошибок найдено)

**Тесты:** `tests/test_llm_evaluator.py`, `tests/test_llm_prompts.py`

---

### Задача 12.3: Генерация карточек через LLM ✅

**Реализованные модули:**
- `src/eng_words/llm/card_generator.py` — генератор карточек
- `src/eng_words/llm/cache.py` — кэширование результатов
- `scripts/generate_cards.py` — CLI для генерации карточек

**Реализованные возможности:**
- Batch генерация карточек (40 synsets за раз)
- Простые определения (B1 level)
- Переводы на русский
- Generic примеры (универсальные)
- Book примеры с фильтрацией спойлеров
- Кэширование результатов в JSON
- Retry логика для неполных ответов
- Guardrails: ограничение длины примеров, количества примеров

**Результаты:**
- Генерация карточек для всех synsets из книги
- Фильтрация спойлеров через `spoiler_risk` оценку
- Экспорт в Anki CSV формат

**Тесты:** `tests/test_llm_card_generator.py`, `tests/test_llm_cache.py`

---

### Задача 12.4: Улучшения и исправления ✅

**Исправления:**
- Исправлен баг с пустой `definition` при добавлении assigned_synset в candidates
- Улучшен промпт для evaluation с примерами и правилами
- Добавлен `seed=42` для детерминизма
- Увеличено `max_tokens` до 16384 для больших JSON ответов
- Интеграция spaCy для корректной обработки irregular verbs в примерах
- Добавлен spaCy model как direct dependency в `pyproject.toml`

**Обновления зависимостей:**
- `openai >= 2.11.0,<3`
- `anthropic >= 0.75.0,<1`
- `pydantic >= 2.12.5`
- `spacy`, `wordfreq`, `sentence-transformers` обновлены до последних версий

---

## Этап 13: Анализ ошибок WSD ✅

**Статус:** Завершен

**Документ:** `docs/WSD_ERRORS_ANALYSIS.md`

### Задача 13.1: Ручной анализ WSD результатов ✅

**Выполнено:**
- Запущена evaluation на 394 примерах (2 батча по 197)
- Ручной анализ каждого примера с классификацией:
  - True Positive (TP)
  - False Positive (FP)
  - True Negative (TN)
  - False Negative (FN)
  - Low Quality (LQ)

**Результаты:**
- Найдено 5 ошибок (1.27% error rate):
  1. `over` → `all_over.r.01` (должно быть `over.r.01`)
  2. `go` → `rifle.v.02` (должно быть `go.v.01`)
  3. `air` → `breeze.n.01` (должно быть `air.n.03`)
  4. `go` → `travel.v.01` (конструкция "going to")
  5. `go` → `travel.v.01` (конструкция "went on")

**Паттерны ошибок:**
- 4/5 ошибок связаны с грамматическими конструкциями ("going to", "went on", "over there")
- 1/5 ошибок связана с коллокациями ("jocular air" → breeze вместо manner)

**Анализ причин:**
- Cosine similarity не учитывает грамматический контекст
- Embedding модели не понимают грамматические конструкции
- LLM evaluation видит только 5 candidates из всех возможных synsets

**Артефакты:**
- `data/llm_cache/evaluations/wsd_errors_registry.json`
- `data/llm_cache/evaluations/wsd_errors_registry_batch2.json`

---

## Этап 14: План улучшений WSD ✅

**Статус:** Завершен (план готов к реализации)

**Документ:** `docs/WSD_IMPROVEMENTS_PLAN.md`

### Задача 14.1: Создание плана улучшений ✅

**Структура плана:**
- Разделение на Track A (WSD prediction) и Track B (LLM evaluation)
- Метрики: targeted-fix success, silver accuracy, stability
- Гипотезно-ориентированный подход с A/B тестированием
- 6 этапов разработки с четкими критериями приемки

**Основные направления улучшений:**
1. Улучшение LLM evaluation (увеличение candidates, smart selection, two-stage)
2. Детекция грамматических конструкций (regex + dependency parsing)
3. Улучшение scoring функции (contextual embeddings, collocation cues)
4. Post-processing фильтры
5. Улучшение промптов для evaluation

**Цели:**
- Accuracy ≥ 99.5% (сейчас ~98.73%)
- LLM detection rate ≥ 90% (сейчас ~66.7%)
- Исправить все 5 известных ошибок без регрессий

**Артефакты:**
- `docs/WSD_IMPROVEMENTS_PLAN.md` — детальный план (v4, release-ready)

---

## Этап 15: WSD Gold Dataset ✅ ЗАВЕРШЕН

**Даты:** 2026-01-08 — 2026-01-09

### Цель
Создать фиксированный (locked) gold test set для честного сравнения улучшений WSD.

### Что сделано

#### Сбор данных
- Обработано 4 книги: American Tragedy, Game of Thrones, On the Edge, Lever of Riches
- Извлечено 3000 примеров (750 из каждой книги)
- Стратификация по POS, сложности и источникам

#### Разметка LLM (Smart Aggregation)
- **Anthropic** (claude-opus-4-5-20251101): основной судья
- **Gemini** (gemini-3-flash-preview): второй судья
- **OpenAI** (gpt-5.2): referee для разногласий
- Agreement rate: 87% (Anthropic + Gemini)
- Referee вызовов: 13% (389 из 3000)

#### Split и заморозка
- **Dev set**: 1500 примеров (American Tragedy + On the Edge)
- **Test locked**: 1500 примеров (Game of Thrones + Lever of Riches)
- SHA256 checksum для защиты test set
- `make verify-gold` для CI проверки

#### Baseline метрики WSD
| Метрика | Значение |
|---------|----------|
| Overall Accuracy | 47.5% |
| ADJ | 56.8% |
| ADV | 53.0% |
| NOUN | 50.6% |
| VERB | 34.0% |
| Easy | 80.8% |
| Medium | 46.4% |
| Hard | 25.7% |

### Созданные модули
- `src/eng_words/wsd_gold/` — основной пакет
  - `models.py` — dataclasses (GoldExample, ModelOutput, etc.)
  - `collect.py` — сбор примеров
  - `sample.py` — стратификация
  - `providers/` — LLM провайдеры (OpenAI, Anthropic, Gemini)
  - `aggregate.py` — базовая агрегация
  - `smart_aggregate.py` — умная агрегация с referee
  - `eval.py` — оценка WSD
  - `cache.py` — кэш LLM ответов

### Скрипты
- `scripts/collect_gold_pilot.py` — пилотный сбор
- `scripts/run_gold_labeling.py` — разметка
- `scripts/aggregate_gold_labels.py` — агрегация
- `scripts/freeze_gold_dataset.py` — заморозка
- `scripts/eval_wsd_on_gold.py` — оценка

### CI команды
```bash
make eval-wsd-gold        # Полная оценка
make eval-wsd-gold-quick  # Быстрая (100 примеров)
make verify-gold          # Проверка checksum
```

### Артефакты
- `data/wsd_gold/gold_dev.jsonl` — dev set
- `data/wsd_gold/gold_test_locked.jsonl` — locked test set
- `data/wsd_gold/gold_manifest.json` — метаданные
- `docs/WSD_GOLD_DATASET_USAGE.md` — документация

---

## Этап 16: WSD Improvements (Завершено)

**Цель:** Улучшить WSD accuracy используя gold dataset.
**Документ:** `docs/WSD_IMPROVEMENTS_IMPLEMENTATION_PLAN.md`

### Реализованные улучшения:

| Компонент | Описание | Эффект на accuracy |
|-----------|----------|-------------------|
| Construction Detector | 31 regex паттерн для грамматических конструкций | 0% |
| Phrasal Verb Detector | 50+ фразовых глаголов с spaCy dependency parsing | 0% |
| Smart Candidate Selection | Context keyword boost в scoring | **+0.5%** |

### Финальные результаты:

**Dev Set (gold_dev.jsonl):**
- Synset Accuracy: 47.5% → **48.0%** (+0.5%)
- Supersense Accuracy: 74.2% → 74.3% (+0.1%)

**Test Locked (gold_test_locked.jsonl):**
- Synset Accuracy: **51.2%**
- By POS: ADJ 60.8%, NOUN 54.6%, ADV 53.1%, VERB 38.8%
- By difficulty: easy 84.7%, medium 48.8%, hard 29.5%

### Ключевые выводы:

1. **Construction/Phrasal Verb detection НЕ улучшает accuracy** — LLM судьи выбирали literal meanings
2. **Context keyword matching дает +0.5%** — скромное улучшение
3. **Целевые метрики (52%+) НЕ достигнуты** — embedding-based подход имеет потолок
4. **VERB — основная проблема** — 38.8% vs 60.8% для ADJ

### Артефакты:
- `src/eng_words/wsd/constructions.py` — 25 тестов
- `src/eng_words/wsd/phrasal_verbs.py` — 20 тестов
- `src/eng_words/wsd/candidate_selector.py` — 15 тестов

---

## Следующие шаги

### Потенциальные направления улучшения WSD:

1. **Fine-tuned модели** — обучить на WSD-специфичных данных
2. **LLM-based WSD** — использовать LLM напрямую для disambiguation
3. **Ensemble методы** — комбинировать несколько подходов
4. **Verb-specific обработка** — отдельная логика для глаголов

---

## Обновленная статистика проекта

- **Модулей:** 35+
- **Тестов:** 250+ (все проходят)
- **Этапов:** 16 (все завершены)
- **Задач:** 90+ (все выполнены)

### WSD Gold Dataset модуль
- **Файлов:** 10 модулей + тесты
- **Тестов:** 190+ (models, collect, sample, providers, aggregate, eval, cache)
- **Провайдеры:** OpenAI (GPT-5.2), Anthropic (Claude Opus 4.5), Gemini (3 Flash)
- **Примеров:** 3000 (1500 dev + 1500 test_locked)
- **Agreement:** 87% между LLM судьями

### LLM модуль
- **Файлов:** 8 модулей + тесты
- **Провайдеры:** OpenAI, Anthropic
- **Функции:** WSD Evaluation, Card Generation, Caching

### WSD модуль
- **Файлов:** 9 модулей + тесты
- **Supersenses:** 43 категории (25 noun, 15 verb, 2 adj, 1 adv)
- **Производительность:** ~500-1000 токенов/сек
- **Accuracy:** Dev 48.0%, Test 51.2%
- **Новые компоненты:**
  - `constructions.py` — 31 regex паттерн
  - `phrasal_verbs.py` — 50+ фразовых глаголов
  - `candidate_selector.py` — context boost scoring

---

## Этап 17: Smart Card Pipeline — Stage 2.5 (Завершено)

**Цель:** Фильтрация длины и спойлеров до генерации, детерминированная логика выбора примеров, снижение объёма и повышение качества карточек.

**Основные изменения (код):**
- `mark_examples_by_length` теперь с `min_words=6` (фильтр “слишком короткие”).
- `check_spoilers` батчами, до генерации.
- `select_examples_for_generation` с дедупликацией, паттерны 2+1 / 1+2 / 0+3 и подсчётом `generate_count`.
- `SMART_CARD_PROMPT` обновлён под pre-checks, явный `generate_count`.
- `run_synset_card_generation.py` интегрирован со Stage 2.5 (длина, спойлеры, выбор), убран `enable_fallback`, добавлена статистика.

**Тесты и результаты:**
- Unit/интеграционные тесты: 19/19 (включая новые интеграционные для pipeline).
- Пробный прогон 10 карточек: 5 сгенерировано, 0 без примеров, все с переводом.
- Прогон 100 карточек: 54 финальных, 0 без примеров, 100% с переводом; распределение паттернов 2+1 / 1+2 / 0+3 = 41 / 13 / 10; среднее 2.98 примера/карту; стоимость ~$0.0457, время ~15.5 мин.
- Статистика фильтрации (на 100): 189 too long, 16 too short, 688 ok; 276 со спойлерами.

**Артефакты и документы:**
- `docs/REFACTOR_AND_BACKUP_PLAN.md` — план уборки/рефакторинга + регресс (replay + live smoke).
- `docs/STAGE2_5_INTEGRATION_TEST_RESULTS.md` — тест на 10 карточках.
- `docs/STAGE2_5_INTEGRATION_TEST_100_RESULTS.md` — тест на 100 карточках.
- `scripts/check_test_progress.sh` — мониторинг долгих прогонов.

---

## Этап 18: Рефакторинг и уборка репозитория ✅ (2026-01-21)

**Цель:** Навести порядок в репозитории, архивировать устаревший код и документацию, не сломав pipeline.

### Выполнено

**Архивация кода:**
- Устаревшие модули → `src/eng_words/_archive/`:
  - `llm/cache.py`, `card_generator.py`, `evaluator.py`, `prompts.py`
  - `aggregation/fallback.py`
- Устаревшие скрипты → `scripts/_archive/` (53 файла)
- Устаревшие тесты → `tests/_archive/`

**Архивация документации:**
- 55 документов → `docs/archive/2026-01-21/`
- Оставлено 12 актуальных документов
- Восстановлен `WSD_GOLD_DATASET_USAGE.md` (важен для Golden Dataset)

**Чистка данных:**
- Удалены временные папки: `data/fallback_test/`, `stage2_5_test/`, `test_*_cache/`, `output/`, `comparison/`
- Удалены устаревшие файлы: `DEVELOPMENT_PLAN.md`, `QUICK_TEST_VALIDATION.md`
- Обновлён `.gitignore`: logs, backups, temp data

### Результаты

| Компонент | Было | Стало |
|-----------|------|-------|
| Python файлов | 150 | 110 |
| Скриптов | 51 | 10 |
| Документов | 60+ | 12 |
| Data | ~200 MB | 156 MB |

### Регресс-тесты

- ✅ Unit-тесты: 697 passed
- ✅ Replay 100: 54 карточки идентичны эталону
- ✅ Pipeline работает корректно

### Коммиты

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

*Документ обновлен: 2026-01-21*
*Версия: 1.7 (Рефакторинг завершён)*
