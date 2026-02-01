# Word Family Pipeline

> **Цель:** Создать качественные Anki карточки для изучения английского (B1-B2)  
> **Подход:** LLM-based кластеризация примеров по смыслам (Word Family)  
> **Статус:** Pipeline v2 реализован и протестирован ✅

---

## Оглавление

1. [Принципы разработки](#принципы-разработки)
2. [История: Эксперимент A vs B](#история-эксперимент-a-vs-b)
3. [История: Улучшение промпта](#история-улучшение-промпта)
4. [Выявленные проблемы](#выявленные-проблемы)
5. [План v2: Двухэтапный пайплайн](#план-v2-двухэтапный-пайплайн)
6. [Текущий статус](#текущий-статус)
7. [Осталось по плану](#осталось-по-плану)
8. [Как подставлять WordNet hints в промпт C](#как-подставлять-wordnet-definitions-hints-в-промпт-c)

---

## Принципы разработки

1. **Модульность**: Каждая функция решает одну задачу
2. **Тестирование**: Тесты пишутся параллельно с кодом (TDD)
3. **Поэтапность**: Разработка небольшими блоками с тестированием после каждого
4. **Измеримость**: Качество оценивается на выборке перед полным запуском
5. **Простота**: Код должен быть понятен через год
6. **Экономия**: Минимизировать стоимость LLM calls без потери качества
7. **Кэширование**: Все LLM ответы кэшируются для повторного использования
8. **Precision-first**: Лучше пропустить, чем создать невалидную карточку

---

# История: Эксперимент A vs B

## Контекст (2026-01-22)

**Проблема:** Текущий WSD-based pipeline (A) имел низкое покрытие и сложную архитектуру.

**Гипотеза:** Простой LLM-based подход (Word Family) даст лучшее качество и покрытие.

## Дизайн эксперимента

### Сравниваемые подходы

| ID | Подход | Описание |
|----|--------|----------|
| A | **Текущий Pipeline** | WSD → Synset Aggregation → Card Generation |
| B | **Word Family** | Группировка по лемме → LLM кластеризация |
| C | **Word Family + hints** | То же + WordNet definitions как hint |

### Архитектура

```
                    tokens.parquet
                    (sample: 2000 предложений)
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
         PIPELINE A                      PIPELINE B
         ┌────────┐                    ┌────────┐
         │  WSD   │                    │Группир.│
         │Analysis│                    │по лемме│
         └───┬────┘                    └───┬────┘
         ┌───┴────┐                    ┌───┴────┐
         │ Synset │                    │  LLM   │
         │  Aggr. │                    │Cluster │
         └───┬────┘                    └───┬────┘
         ┌───┴────┐                    ┌───┴────┐
         │  Card  │                    │  Card  │
         │  Gen.  │                    │  Gen.  │
         └───┬────┘                    └───┬────┘
              │                               │
              ▼                               ▼
         cards_A.json                   cards_B.json
```

## Результаты запусков (2026-01-23)

| Pipeline | Cards | Lemmas | Cost | Time | Status |
|----------|-------|--------|------|------|--------|
| **A** | 902 | 645 | $1.77 | 6.7h | ✅ Готов |
| **B** | 4,515 | 3,229 | $2.89 | ~1.5h | ✅ Готов |
| **C** | 148 | 145 | $0.005 | 7m | ❌ Rate limit |

## Coverage Analysis

| Метрика | Pipeline A | Pipeline B |
|---------|------------|------------|
| **Coverage** | 19.7% | **98.8%** |
| Lemmas | 645 | 3,229 |
| Cards | 902 | 4,515 |
| Avg cards/lemma | 1.40 | 1.40 |

**Venn Diagram:**
- A ∩ B (оба): 600 лемм
- Only A: 45 лемм (в основном function words — stop words включены по ошибке)
- Only B: 2,629 лемм (реальные content words)

## Quality Evaluation (20 лемм из пересечения)

| Winner | Count | % |
|--------|-------|---|
| **A лучше** | 10 | 50% |
| **B лучше** | 7 | 35% |
| **Tie** | 3 | 15% |

### Находки по Pipeline A

**Проблемы:**
- Пропускает значения (realization: "осознание", stone: "кусок", story: "ложь/этаж")
- Баг с дубликатами (pause: две одинаковые карточки)
- Включает function words (still, however, besides) по ошибке

**"Only A" (45 лемм):**
- ~80% function words (is, however, besides, more, less) — НЕ нужны
- ~20% полезные (still, quite, rather) — можно добавить

### Находки по Pipeline B

**Проблемы:**
- Избыточное дробление (reach: 4 vs 2, opposition: 2 vs 1)
- Иногда неточные определения (mother: "забеременеть")
- Выделяет фразы как отдельные значения (long: "no longer")

**"Only B" (2629 лемм):**
- 100% реальные content words (advice, happy, die, wash, guitar)
- Pipeline A пропустил из-за строгих фильтров WSD

## Вердикт: Pipeline B WINS

| Критерий | Winner | Reason |
|----------|--------|--------|
| **Coverage** | B | 5x больше лемм (3229 vs 645) |
| **Quality** | A (slightly) | 50% vs 35% на пересечении |
| **Efficiency** | B | $2.89/1.5h vs $1.77/6.7h |
| **Bugs** | B | A имеет дубликаты |
| **Overall** | **B** | Покрытие важнее |

---

# История: Улучшение промпта

## Контекст (2026-01-23)

**Проблема:** Pipeline B создаёт избыточные карточки для некоторых лемм.

**Цель:** Уменьшить oversplit при сохранении правильного разделения смыслов.

## Примеры проблемы

### OVERSPLIT (избыточно раздроблено)

| Lemma | Cards B | Cards A | Проблема |
|-------|---------|---------|----------|
| reach | 4 | 2 | "прибыть" и "достичь уровня" — то же самое |
| opposition | 2 | 1 | Оба значения очень похожи |
| particular | 2 | 1 | Практически дублируют |
| time | 6 | - | Слишком много оттенков |

### CORRECT (правильно разделено)

| Lemma | Cards B | Cards A | Комментарий |
|-------|---------|---------|-------------|
| realization | 2 | 1 | ✅ "осознание" и "воплощение" |
| story | 3 | 1 | ✅ "рассказ", "ложь", "этаж" |
| mean | 5 | - | ✅ verb/noun/adjective — всё разное |

## Тестовый набор

**Эвристика:** Same POS = likely oversplit, Different POS = likely correct

- **OVERSPLIT candidates**: 591 лемм (один POS)
- **CORRECT candidates**: 313 лемм (разные POS)

Для тестов: 10 OVERSPLIT + 10 CORRECT

## Итерации промпта

| Version | OVERSPLIT reduction | CORRECT preservation | Notes |
|---------|---------------------|----------------------|-------|
| **V1** (baseline) | -17% ❌ | 114% ✅ | Слишком дробит |
| **V2** (strict) | **41.5%** ✅ | 81% ❌ | Схлопывает нужное |
| **V3** (balanced) | -9.8% ❌ | **119%** ✅ | Не уменьшает oversplit |
| **V4** (final) | 22% ❌ | 86.5% ❌ | Компромисс |

### Ключевые изменения в промптах

**V1 → V2 (strict):**
```
- "PREFER FEWER CARDS: Most words have 1-2 core meanings"
- "SAME PART OF SPEECH = likely same meaning"
- "DO NOT create separate cards for slight nuances"
```
Результат: Сильно уменьшил oversplit, но схлопнул и нужные различия.

**V2 → V3 (balanced):**
```
- Добавил примеры GOOD vs BAD splitting
- "GOOD: mean → verb/noun/adjective = 4 cards"
- "BAD: reach → arrive/achieve = should be 2 max"
```
Результат: Сохранил correct, но перестал уменьшать oversplit.

## Выводы

1. **Trade-off**: Нельзя одновременно достичь ≥30% reduction и ≥90% preservation
2. **V2 лучший** для production — 41.5% reduction важнее
3. **Причина**: LLM с трудом различает "похожие но разные" vs "одинаковые"

---

# Выявленные проблемы

## 1. Phrasal verbs

**Проблема в промпте V2:**
```
- Phrasal verbs with similar core meaning (e.g., "go away" and "go out" = both "leave")
```

**Почему неправильно:** Phrasal verbs очень важны для изучения! "go away", "go out", "go on" — это разные значения, которые нужно учить отдельно.

**Решение:** Убрать это правило из промпта.

## 2. Спойлеры и структура данных

**Проблема в промпте:**
```
- IGNORE examples where: Contains major plot spoilers
```

**Почему неправильно:** 
- Смысл нужно извлекать даже из примеров со спойлерами
- Просто не использовать спойлерные примеры в карточках
- Иначе теряем редкие значения

**Решение:** Разделить извлечение смыслов и генерацию карточек (двухэтапный пайплайн).

---

# План v2: Двухэтапный пайплайн

> **Решения приняты 2026-01-23:**
> - Два этапа LLM (стоимость ~$5-6 — приемлемо)
> - Спойлеры: извлекать смысл, но не использовать как пример
> - Phrasal verbs: ОТДЕЛЬНЫЕ карточки (go away ≠ go)
> - Структура данных: трёхуровневая (all → source → clean)

## Архитектура

```
              tokens.parquet
                    │
                    ▼
         ┌─────────────────────┐
         │ Группировка по лемме│
         └──────────┬──────────┘
                    │
                    ▼
    ════════════════════════════════════════════════════════
    ЭТАП 1: ИЗВЛЕЧЕНИЕ СМЫСЛОВ (MeaningExtractor)
    ════════════════════════════════════════════════════════
    │
    │ Input:  Лемма + ВСЕ примеры (включая спойлеры)
    │ Output: Список смыслов с source_examples
    │
    │ Ключевые задачи:
    │ - Определить все значения слова
    │ - Выделить phrasal verbs как ОТДЕЛЬНЫЕ значения
    │ - Пометить примеры со спойлерами (но НЕ исключать)
    │ - Указать откуда извлечён каждый смысл
    │
    └──────────┬───────────────────────────────────────────
               │
               ▼
    ════════════════════════════════════════════════════════
    ЭТАП 2: ГЕНЕРАЦИЯ КАРТОЧЕК (CardGenerator)
    ════════════════════════════════════════════════════════
    │
    │ Input:  Смыслы + source_examples + all_examples
    │ Output: Финальные карточки с clean_examples
    │
    │ Ключевые задачи:
    │ - Выбрать 2-3 чистых примера (без спойлеров, 10-30 слов)
    │ - Сгенерировать примеры если чистых нет
    │ - Перевод определения на русский
    │ - Финальное форматирование
    │
    └──────────┬───────────────────────────────────────────
               │
               ▼
         meanings_extracted.json → cards_final.json
```

## Структура данных

### После Этапа 1 (meanings_extracted.json)

```python
{
  "lemma": "go",
  "all_sentence_ids": [1, 2, 5, 8, 12, 15, 20, 25, 30],
  
  "meanings": [
    {
      "meaning_id": 1,
      "definition_en": "to move or travel from one place to another",
      "part_of_speech": "verb",
      "is_phrasal": false,
      "phrasal_form": null,
      "source_examples": [
        {"index": 1, "has_spoiler": false},
        {"index": 5, "has_spoiler": true, "spoiler_type": "character_death"}
      ]
    },
    {
      "meaning_id": 2,
      "definition_en": "to leave a place permanently",
      "part_of_speech": "phrasal verb",
      "is_phrasal": true,
      "phrasal_form": "go away",
      "source_examples": [
        {"index": 12, "has_spoiler": false},
        {"index": 15, "has_spoiler": false}
      ]
    },
    {
      "meaning_id": 3,
      "definition_en": "to continue doing something",
      "part_of_speech": "phrasal verb",
      "is_phrasal": true,
      "phrasal_form": "go on",
      "source_examples": [
        {"index": 20, "has_spoiler": false}
      ]
    }
  ]
}
```

### После Этапа 2 (cards_final.json)

```python
{
  "lemma": "go",
  "cards": [
    {
      "card_id": "go_1",
      "meaning_id": 1,
      "lemma_display": "go",  # для карточки
      "definition_en": "to move or travel from one place to another",
      "definition_ru": "идти, ехать, двигаться",
      "part_of_speech": "verb",
      "is_phrasal": false,
      
      # Traceability
      "all_sentence_ids": [1, 2, 5, 8, 12, 15, 20, 25, 30],
      "source_sentence_ids": [1, 5],
      
      # Финальные примеры
      "clean_examples": [
        {
          "sentence_id": 1,
          "text": "He decided to go to the city center.",
          "source": "book"
        },
        {
          "sentence_id": null,
          "text": "We need to go now or we'll miss the train.",
          "source": "generated"
        }
      ]
    },
    {
      "card_id": "go_away_2",
      "meaning_id": 2,
      "lemma_display": "go away",  # phrasal verb как отдельная карточка!
      "definition_en": "to leave a place permanently",
      "definition_ru": "уходить, уезжать (насовсем)",
      "part_of_speech": "phrasal verb",
      "is_phrasal": true,
      "phrasal_form": "go away",
      
      "source_sentence_ids": [12, 15],
      "clean_examples": [...]
    }
  ]
}
```

## Промпты

### Этап 1: MeaningExtractor

```
You are analyzing word usage to identify distinct meanings.

LEMMA: {lemma}

EXAMPLES FROM BOOK (numbered):
{numbered_examples}

TASK:
Identify ALL distinct meanings of "{lemma}" found in these examples.

CRITICAL RULES:

1. PHRASAL VERBS are SEPARATE meanings:
   - "go away" ≠ "go out" ≠ "go on" ≠ "go"
   - Each phrasal verb gets its own meaning entry
   - Set is_phrasal=true and phrasal_form="go away"

2. SPOILERS - identify but DON'T exclude:
   - Mark has_spoiler=true if example reveals plot (deaths, crimes, twists)
   - Still use the example to identify the meaning
   - We filter spoilers in the next step

3. DIFFERENT POS = SEPARATE meanings:
   - Noun "run" ≠ Verb "run"

4. DO NOT over-merge:
   - If meanings feel different to a learner, keep them separate

OUTPUT (strict JSON):
{
  "meanings": [
    {
      "meaning_id": 1,
      "definition_en": "clear definition",
      "part_of_speech": "verb/noun/adj/adv/phrasal verb",
      "is_phrasal": false,
      "phrasal_form": null,
      "source_examples": [
        {"index": 1, "has_spoiler": false},
        {"index": 5, "has_spoiler": true, "spoiler_type": "character death"}
      ]
    }
  ]
}
```

### Этап 2: CardGenerator

```
You are creating Anki flashcards for English learners (B1-B2).

LEMMA: {lemma}
MEANING: {definition_en}
PART OF SPEECH: {pos}
{phrasal_info}

SOURCE EXAMPLES (some may have spoilers):
{source_examples_with_spoiler_flags}

ALL AVAILABLE EXAMPLES:
{all_examples}

TASK:
Create a flashcard for this meaning.

RULES:
1. Select 2-3 CLEAN examples:
   - No spoilers (skip examples marked has_spoiler=true)
   - Length: 10-30 words
   - Clear context for the meaning

2. If not enough clean examples:
   - Generate simple, natural examples
   - Mark source="generated"

3. Translation:
   - B1-B2 level Russian
   - Include common synonyms

OUTPUT (strict JSON):
{
  "definition_ru": "перевод",
  "clean_examples": [
    {"sentence_id": 5, "text": "example from book", "source": "book"},
    {"sentence_id": null, "text": "generated example", "source": "generated"}
  ]
}
```

## Реализация

### Файлы

```
src/eng_words/pipeline_v2/
├── __init__.py
├── meaning_extractor.py   # Этап 1
├── card_generator.py      # Этап 2
├── data_models.py         # Dataclasses
└── pipeline.py            # Orchestration
```

### API

```python
# Этап 1
extractor = MeaningExtractor(provider, cache)
meanings = extractor.extract(lemma, examples, sentence_ids)

# Этап 2
generator = CardGenerator(provider, cache)
cards = generator.generate(lemma, meanings, all_examples)

# Полный pipeline
pipeline = WordFamilyPipelineV2(provider, cache)
results = pipeline.process_lemmas(lemma_groups)
```

---

# Текущий статус

## Что сделано ✅

### Эксперимент A vs B
- [x] Создан sample из 2000 предложений
- [x] Запущен Pipeline A → 902 карточки, 645 лемм
- [x] Запущен Pipeline B → 4,515 карточек, 3,229 лемм
- [x] Проведён анализ покрытия
- [x] Проведена ручная проверка (20 лемм)
- [x] **Вывод: Pipeline B победил**

### Улучшение промпта
- [x] Создан тестовый набор (10 OVERSPLIT + 10 CORRECT)
- [x] Протестированы 4 версии промпта
- [x] **Вывод: V2 лучший (41.5% reduction)**

### Выявленные проблемы
- [x] Phrasal verbs не должны схлопываться
- [x] Спойлеры не должны блокировать извлечение смыслов
- [x] Нужен двухэтапный пайплайн

### Pipeline v2 (двухэтапный) — РЕАЛИЗОВАН ✅ (2026-01-23)
- [x] Создана структура `src/eng_words/pipeline_v2/`
- [x] `MeaningExtractor` — Stage 1 (извлечение смыслов)
- [x] `CardGenerator` — Stage 2 (генерация карточек, per-meaning)
- [x] Трёхуровневая структура данных (`all` → `source` → `clean`)
- [x] Phrasal verbs как отдельные карточки
- [x] Spoiler detection (помечает, но не блокирует извлечение)

**Архитектурное решение: per-meaning vs batched**

Тестировали два подхода для Stage 2:

| Подход | Success rate | Время | Стоимость | Решение |
|--------|--------------|-------|-----------|---------|
| Batched (все meanings за 1 вызов) | 71.4% | 1.9 мин | $0.09 | ❌ Потеря данных |
| **Per-meaning** (1 вызов на meaning) | **100%** | 3.2 мин | $0.21 | ✅ Выбрано |

**Причина отказа от batched:** При большом числе meanings (например, "come" = 22) LLM генерировал слишком длинный JSON, который обрывался. Потеря 22 карточек неприемлема.

**Результаты теста (10 лемм, per-meaning):**
| Метрика | Значение |
|---------|----------|
| Meanings | 71 |
| Карточки | **71 (100%)** |
| Phrasal verbs | **17** |
| API calls | 81 (10 extraction + 71 generation) |
| Время | 3.2 мин |
| Стоимость | $0.21 |

**Примеры phrasal verbs:**
- `go on`, `go out`, `go over`
- `look to`, `look up`, `look forward to`
- `make one's way`, `lead the way`

**Распределение карточек по леммам:**
- way: 13, come: 11, look: 10, know: 9, time: 9
- go: 7, girl: 5, little: 3, think: 2, want: 2

## Тест 100 лемм — ГОТОВ ✅ (2026-01-24)

### Улучшение промпта для Gemini 3

Проблема: gemini-3-flash-preview oversplit'ил (102 карточки на 10 лемм, 31 phrasal verb).

Решение: улучшили правила в `MeaningExtractor`:
- Только TRUE idiomatic phrasal verbs (не "look at", "go to")
- PREFER FEWER MEANINGS — группировать похожие значения
- Ask: "Would a learner need different translations?"

**Результат улучшения (10 лемм):**
| Model | Before | After |
|-------|--------|-------|
| Cards | 102 | **49** |
| Phrasal | 31 | **7** |

### Результаты 100 лемм (gemini-3-flash-preview)

| Метрика | Значение |
|---------|----------|
| Карточек | **375** |
| Карточек/лемму | **3.75** (разумно!) |
| Regular | 341 |
| Phrasal verbs | **34** (только идиомы) |
| Время | 19.4 мин |
| Стоимость | **$0.70** |

**Качество phrasal verbs (примеры):**
- ✅ `go on` (continue), `take up`, `take aback`
- ✅ `turn down`, `turn on`, `turn over`, `turn up`
- ✅ `look after`, `look up`, `look forward to`
- ✅ `bring up`, `find out`, `get away with`

**Распределение карточек на лемму:**
- 1 карт: 7 лемм (marry, sir, lake...)
- 2-4 карт: 66 лемм
- 5-7 карт: 19 лемм
- 8-14 карт: 8 лемм (take: 14, turn: 10)

**Экстраполяция на 3000 лемм:**
- Стоимость: ~$21
- Время: ~10 часов

### Файл с результатами
`data/experiment/cards_v2_gemini3_100.json`

## Осталось по плану

**Политика:** Полный прогон (~3000 лемм) — **только через Batch API**. Эксперименты (малые сэмплы, тесты промпта) — через Standard API (`run_pipeline_b.py`).

### 1. Эксперимент B vs C — закрыт, остаёмся на B

- [x] **Baseline C (2026-01-31):** прогнан на 20 леммах, отчёт: `data/experiment/quality_evaluation_C_baseline.md`
  - 97 карточек (4.85 карт/лемму vs ~1.40 у B). OVERSPLIT: time 8, know 7, think 6, look 6… CORRECT: go, come, girl, want. Phrasal verbs не выделены отдельно.
- [x] **Поэкспериментировать с промптом C (2026-01-31):** прогнаны варианты max_hints=15, extra_instruction=use_wordnet/merge_fewer, hints_only_multi. Результаты: 95–97 карт (baseline 97); B на тех же 20 леммах — 59. Ни один вариант не приблизил C к B.
- [x] **Решение:** остаёмся на **Pipeline B**. Дальнейшие эксперименты с C отложены.

### 2. Полный датасет (~3000 лемм) — только Batch API (Pipeline B)

- [x] **Batch API для Pipeline B реализован:** `scripts/run_pipeline_b_batch.py` (create / status / download / wait). Тесты: `tests/test_pipeline_b_batch.py`. Тот же промпт и формат вывода, что у `run_pipeline_b.py`; результат — `data/experiment/cards_B_batch.json`.
- [x] **Прогнать полный датасет через Batch API (2026-01-31):** 3268 лемм, 4218 карточек, 0 ошибок. Результат: `data/experiment/cards_B_batch.json`.
- [ ] **Проверка качества:** отчёт `uv run python scripts/check_quality_b_batch.py` → `data/experiment/quality_report_B_batch.md` (статистика + случайная выборка карточек для ручной оценки). После просмотра сэмпла — сравнить с предыдущими результатами B по покрытию и качеству.

### 3. Дополнительно (не в плане, но полезно)

- [ ] **Stage 2 (card generation) через Batch API**
  - Сейчас Stage 2 в v2 работает через стандартный API
  - Добавить batch для Stage 2, чтобы снизить стоимость
- [ ] **Обработка ошибок Batch API**
  - 13 ошибок из 100 лемм (MAX_TOKENS)
  - Повторить проблемные через стандартный API или увеличить лимит

**Скрипты:** `scripts/run_pipeline_b_batch.py` — прод, Batch API для полного прогона (тесты: `tests/test_pipeline_b_batch.py`). Подготовка выборки: `scripts/prepare_pipeline_b_sample.py`. Скрипты сравнения A/B/C (Standard API) — в `legacy/pipeline_a/scripts/experiment/` (`run_pipeline_a.py`, `run_pipeline_b.py`, `run_pipeline_c.py`). Код кластеризации — `src/eng_words/word_family/` (модуль `clusterer`).

---

### Как подставлять WordNet definitions (hints) в промпт C

**Цель:** Перед прогоном C на полном датасете — поэкспериментировать с форматом и объёмом подсказок, чтобы LLM стабильно давал хорошее качество (меньше oversplit, без потери нужных смыслов).

**Текущая реализация** (`legacy/pipeline_a/scripts/experiment/run_pipeline_c.py` + `eng_words.word_family.clusterer`):

- **Откуда берём подсказки:** WordNet (NLTK) — `get_synsets_with_definitions(lemma, pos)`.
- **Какие POS:** только те, что встретились в книге для этой леммы (`pos_variants` из группировки).
- **Объём:** до 5 синонимичных наборов на POS, не более 10 всего; дедупликация по тексту определения.
- **Формат в промпте:** блок «WORDNET REFERENCE (for guidance only):», затем список строк вида `- {synset_id}: {definition}`.
- **Место в промпте:** после LEMMA, перед EXAMPLES; затем блок инструкций «About WordNet Reference» (primary = examples, WordNet = hint, не создавать карточки для смыслов без примеров, создавать для смыслов из примеров даже если нет в WordNet).

**Варианты для экспериментов:**

| Аспект | Варианты | Зачем |
|--------|----------|--------|
| **Объём** | (a) Как сейчас: 5 per POS, max 10 total; (b) Все синонимичные наборы по этим POS (без лимита 10); (c) Топ-N по частоте в WordNet | Меньше шума vs полная «карта» смыслов; контроль токенов |
| **Формат** | (a) `- synset_id: definition`; (b) Нумерованный список с явным POS: `1. [v] run.v.01: move fast`; (c) Только definitions без synset_id (убрать привязку к WordNet в ответе) | Читаемость, влияние на oversplit и выравнивание с WordNet |
| **Порядок в промпте** | (a) Сейчас: WordNet → инструкции → Examples; (b) Examples → WordNet (как справка после контекста); (c) WordNet в конце как «Reference» | Что LLM считает приоритетом |
| **Инструкции** | (a) Как сейчас (examples = primary); (b) Явно: «When a meaning clearly matches a WordNet sense, use that sense as the main definition»; (c) «Prefer merging into fewer cards when WordNet has one broad sense» | Баланс: меньше oversplit vs сохранение смыслов из книги |
| **Когда давать hints** | (a) Всегда для C; (b) Только если у леммы в WordNet >1 sense (однозначные слова — без подсказок) | Экономия токенов, меньше путаницы для простых слов |

**Предлагаемый порядок экспериментов:**

1. На 10–20 леммах (смесь простых и многозначных) прогнать текущий C, зафиксировать baseline (карточек/лемму, субъективное качество).
2. Вариант объёма: снять лимит 10 или поднять до 15 — посмотреть, не ухудшается ли oversplit.
3. Вариант инструкций: добавить явное «when meaning matches WordNet sense, use it» — проверить, стабилизируется ли число карточек и качество определений.
4. Вариант «hints только для многозначных»: если у леммы в WordNet 1 sense — не подставлять блок WordNet; сравнить с «всегда с hints».

После нескольких итераций зафиксировать выбранный вариант и только потом гнать C на полном датасете через Batch API.

---

### Batch API (50% скидка) — РЕАЛИЗОВАН ✅

Google Gemini Batch API: асинхронная обработка, 50% дешевле.

**Скрипты:**
- **Pipeline B (полный прогон):** `scripts/run_pipeline_b_batch.py` — create / status / download / wait. Вход: `tokens_sample.parquet`, `sentences_sample.parquet` (готовятся через `scripts/prepare_pipeline_b_sample.py`); выход: `data/experiment/cards_B_batch.json`. В download доступна проверка карточек (--skip-validation отключает).
- **Pipeline v2 Stage 1:** `scripts/_archive/run_gemini_batch.py` — create / status / download / wait / estimate
- `scripts/_archive/test_batch_approaches.py` — тест подходов для high-frequency лемм

**Тест MAX_TOKENS (5 проблемных лемм: now, well, up, about, after):**

| Подход | max_examples | maxOutputTokens | Success | Ошибки |
|--------|--------------|-----------------|---------|--------|
| all_examples | ∞ | 16384 | 2/5 | now, well, up |
| **limit_50** | **50** | **16384** | **5/5** | **0** |
| limit_100 | 100 | 16384 | 3/5 | well, up |

**Вывод:** `max_examples=50` даёт 100% успех; при 100+ примеров ответ обрезается (MAX_TOKENS).

**Рекомендуемые настройки:**
- `max_examples=50` (по умолчанию в `run_gemini_batch.py create`)
- `maxOutputTokens=16384`
- Стоп-слова и пунктуация исключаются

**Экономия на 3000 лемм:**
| Подход | Стоимость |
|--------|-----------|
| Standard API | ~$21 |
| **Batch API** | **~$10.50** |

## Файлы

| Файл | Описание |
|------|----------|
| `data/experiment/tokens_sample.parquet` | Sample 2000 предложений |
| `data/experiment/sentences_sample.parquet` | Тексты предложений |
| `data/experiment/cards_A.json` | Результат Pipeline A (WSD) |
| `data/experiment/cards_B.json` | Результат Pipeline B (Standard API) |
| `data/experiment/cards_B_batch.json` | Результат Pipeline B через Batch API |
| `data/experiment/batch_b/` | Входы/выходы Batch API для B (requests.jsonl, results.jsonl, lemma_examples.json) |
| `data/experiment/cards_v2_test.json` | Результат Pipeline v2 (10 лемм, per-meaning) |
| `data/experiment/cards_v2_gemini3_100.json` | Результат Pipeline v2 (100 лемм, gemini-3, улучшенный промпт) |
| `src/eng_words/_archive/pipeline_v2/` | Двухэтапный pipeline (v2), в архиве |
| `legacy/pipeline_a/scripts/experiment/run_pipeline_a.py`, `run_pipeline_b.py`, `run_pipeline_c.py` | Запуск A, B, C (Standard API), история сравнения |
| `scripts/prepare_pipeline_b_sample.py` | Подготовка выборки для Pipeline B (tokens_sample, sentences_sample) |
| `scripts/run_pipeline_b_batch.py` | **Pipeline B через Batch API** (полный прогон), тесты: `tests/test_pipeline_b_batch.py` |
| `legacy/pipeline_a/scripts/experiment/run_pipeline_v2_test.py` | Тест Pipeline v2 |
| `scripts/_archive/run_gemini_batch.py` | Batch API для Pipeline v2 Stage 1 |

### Структура Pipeline v2 (в _archive)

```
src/eng_words/_archive/pipeline_v2/
├── __init__.py           # Экспорты
├── data_models.py        # SourceExample, ExtractedMeaning, FinalCard, etc.
├── meaning_extractor.py  # Stage 1: извлечение смыслов
├── card_generator.py     # Stage 2: генерация карточек (per-meaning)
└── pipeline.py           # WordFamilyPipelineV2 orchestration
```

---

*Документ создан: 2026-01-23*  
*Последнее обновление: 2026-02 — раздел «Осталось по плану», эксперимент B vs C, пути _archive*  
*Объединён из: WORD_FAMILY_EXPERIMENT_PLAN.md + PROMPT_REFINEMENT_PLAN.md*
