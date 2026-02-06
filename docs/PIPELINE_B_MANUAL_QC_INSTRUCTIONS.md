# Инструкция для LLM-агента: ручная проверка качества карточек Pipeline B (30–50 примеров)

После прогона Pipeline B выполни **ручной QC-анализ** на 30–50 карточках: что проверять, как сравнивать, какие метрики считать, как оформить отчёт. Это «человеческая» проверка качества, а не только автоматические счётчики.

---

## 0) Автоматический gate vs ручной QC

- **Автоматический gate (stop/go):** перед ручной проверкой или после download запусти gate:  
  `uv run python scripts/run_quality_investigation.py --gate --cards data/experiment/cards_B_batch.json`  
  или при download: `download --run-gate`.  
  Gate читает `validation_errors` из output и пороги из `qc_gate.QCGateThresholds`; при превышении — **FAIL**, exit code 1. Это и есть критерий «прогон прошёл / не прошёл».
- **Ручной QC** — для **диагностики** и углублённого разбора: типы ошибок, примеры, рекомендации по промпту/парсеру/QC. Критерии ручного чеклиста (lemma в каждом примере, POS, дубликаты смыслов, схема) **соответствуют** тем же проверкам, что делают автоматические gate (lemma_not_in_example, pos_mismatch, duplicate_sense, validation). Ручной отчёт дополняет gate, а не подменяет его.

Подробнее про правила и пороги: `docs/PIPELINE_B_QUALITY_DECISIONS.md`.

---

## 1) Цель и результат

Цель — оценить **качество итоговых карточек** (`cards_B_batch.json`) на сэмпле 30–50 карточек: найти типичные ошибки, посчитать метрики и дать **конкретные рекомендации** (промпт, парсинг, QC-gate).

В конце выдай **QC-отчёт** со:

- таблицей метрик (counts / %);
- списком примеров с ошибками (с цитатами полей карточки);
- топ-3–5 категорий проблем;
- конкретными action items;
- финальным вердиктом **PASS / FAIL** (см. § 7).

---

## 2) Входные данные (актуальные пути)

### 2.1. Must-have (должны существовать после прогона Pipeline B)

| Назначение | Путь |
|------------|------|
| Основной output карточек | `data/experiment/cards_B_batch.json` |
| Лог download / retry / QC | `data/experiment/batch_b/download_log.json` |
| Примеры по леммам (для проверки индексов/подстановок) | `data/experiment/batch_b/lemma_examples.json` |
| Сырые ответы Batch API | `data/experiment/batch_b/results.jsonl` |
| Метаданные batch | `data/experiment/batch_b/batch_info.json` |

> **Проверка:** `cards_B_batch.json` — действительно дефолтный вход для авто-репорта `scripts/check_quality_b_batch.py`.  
> См. `INPUT_PATH = data/experiment/cards_B_batch.json`.  

### 2.2. Optional (может отсутствовать; если отсутствует — честно отметь в отчёте)

| Назначение | Путь | Зачем |
|------------|------|------|
| Выборка токенов (POS/lemma/QC) | `data/experiment/tokens_sample.parquet` | Для POS/QC-проверок на сэмпле |
| Выборка предложений (опционально) | `data/experiment/sentences_sample.parquet` | Для проверок границ предложений/источника примеров |
| Статистика выборки | `data/experiment/sample_stats.json` | Контекст про семпл |
| Авто-отчёт расследования ошибок | `data/experiment/investigation_report.md` | Готовый список «подозрительных» мест |
| Авто-отчёт с рандомным сэмплом | `data/experiment/quality_report_B_batch.md` | Быстрый markdown для ручного чтения |

> `tokens_sample.parquet` / `sentences_sample.parquet` создаёт `scripts/prepare_pipeline_b_sample.py` (см. его docstring с перечислением артефактов).

### 2.3. Stage 1 (полные данные книги): что реально есть сейчас

В текущем репозитории Stage 1 гарантированно пишет **tokens parquet** (и статистику по леммам), обычно в `data/processed/` с именем вида `<book>_tokens.parquet`.  
Схема «`data/processed/<book>/<book>_tokens.parquet`» и `stage1_manifest.json` **не являются текущим стандартом** (если ты их не добавлял отдельным Stage 0 апдейтом).

Если тебе дали полный прогон (не sample), используй то, что реально присутствует в `data/processed/`, например:

- `data/processed/<book>_tokens.parquet`
- `data/processed/<book>_lemma_stats.parquet` (если есть)

Если в этом прогоне были добавлены дополнительные артефакты Stage 1 (например `*_sentences.parquet` или manifest) — **используй их**, но **не требуй их существования** по умолчанию.

### 2.4. Кэширование LLM-ответов

**Не ожидай** отдельного артефакта `data/experiment/batch_b/retry_cache.jsonl`, если его явно не добавляли в код.  
В репозитории есть общий кэш ответов LLM в каталоге:

- `data/cache/llm_responses/`

Если нужно проверить, были ли обращения к Standard API/retry и что именно вернулось — в первую очередь смотри `download_log.json`, а при наличии — файлы кэша в `data/cache/llm_responses/`.

---

## 2.5. Автоматические отчёты (сгенерируй перед ручным QC)

Сначала сделай «быстрый авто-прогон» — он облегчает ручную проверку.

### A) Рандомный сэмпл карточек в markdown

```bash
uv run python scripts/check_quality_b_batch.py --sample 50 --output data/experiment/quality_report_B_batch.md
```

### B) Отчёт расследования типовых ошибок

```bash
uv run python scripts/run_quality_investigation.py \
  --cards data/experiment/cards_B_batch.json \
  --output data/experiment/investigation_report.md
```

### C) Итог прогона в одном JSON (gate + регрессия 49 + пути к отчётам + чеклист артефактов)

Скрипт оценивает QC gate и регрессию 49/49 по файлу карточек и записывает результат в JSON (PASS/FAIL, метрики, пути к отчётам, чеклист «что сдать»).

```bash
# Сначала сгенерировать отчёты (A, B выше), затем:
uv run python scripts/run_manual_qc_and_collect.py --cards data/experiment/cards_B_batch_2.json --output data/experiment/pipeline_b_run_result.json
```

В JSON: `gate` (passed, message, summary), `regression_49` (passed, rates, checklist), `reports` (пути к md-отчётам), `deliverables` (code, reports, docs, commands), `overall_verdict` (PASS/FAIL). Exit code 0 при PASS, 1 при FAIL.

---

## 3) Как выбрать 30–50 карточек (sampling)

Сделай **стратифицированный** отбор: цель — не «рандом», а покрытие рисковых зон.

### 3.1. Обязательно включи группы

- ~10 карточек **без ошибок / без warning-флагов** (если видишь, что такие есть);
- ~10 карточек из **проблемных**: retry / fallback / validation errors (ориентируйся на `download_log.json`);
- ~10 карточек по **частотным леммам** (много примеров: см. `total_lemma_examples` или размер списка в `lemma_examples.json`);
- ~10 карточек по **редким леммам** (1–3 примера);
- ~5 карточек с `part_of_speech=verb` и ~5 с `part_of_speech=noun` (или эквивалентные значения).

Если карточек мало — уменьши числа пропорционально, но **сохрани разнообразие**.

### 3.2. Как выбирать внутри группы

- Если есть `--seed` (или фиксированный seed в скриптах) — используй его.
- Если seed нет — выбирай детерминированно: каждый N‑й элемент или фиксированный алфавитный диапазон лемм.

В отчёте **обязательно** укажи: как выбирал и сколько попало в каждую группу.

---

## 4) Чеклист качества для каждой карточки (ручной QA)

По каждой карточке выставь: **OK / Minor / Major / Invalid**.

### 4.1. Валидность структуры (Schema)

Ожидаемые поля (минимальный набор):

- `lemma`
- `part_of_speech` (согласованный формат)
- `definition_en`
- `definition_ru` (если ожидается в проекте)
- `examples` — массив из 1–3 строк
- `selected_example_indices` (если используется для маппинга)
- опционально: `total_lemma_examples`, `examples_fallback`, `warnings[]`, `source`

**Invalid**, если:

- нет ключевого поля (`lemma`, `definition_en`, `examples`);
- `definition_en` пустой;
- `examples` пустой при требовании «Precision-first»;
- `examples` не массив строк;
- индексы не int или вне диапазона допустимых (см. `lemma_examples.json`).

> **Примечание:** если ваш pipeline допускает «0 примеров» как временный деградированный режим — отметь это отдельно, но **в PASS/FAIL** считай как провал качества (см. § 7).

### 4.2. Соответствие «lemma ↔ examples»

- Лемма (или допустимая словоформа) должна встречаться **в каждом** примере.
- Если лемма отсутствует в примерах — это почти всегда **Major** (или Invalid, если политика strict).
- Если лемма есть, но контекст явно про омоним/другое слово — **Major**.

### 4.3. Качество определения (EN)

- Определение должно соответствовать **контекстам примеров**, а не «среднему словарному».
- Не слишком широкое («a thing», «something» — плохо).
- Без явных фактических ошибок.
- Если на лемму несколько карточек — определения должны быть **разными по смыслу**.

Оценка:
- **OK** — точное и естественное;
- **Minor** — размыто / стиль хромает;
- **Major** — неверный смысл / не по примерам / другой POS.

### 4.4. Качество перевода (RU)

- Перевод соответствует выбранному смыслу (а не «первому значению из головы»).
- Допускаются варианты, но должен быть ясный основной.

**Minor:** стиль/калька.  
**Major:** неверный смысл / неверный POS / перевод «не про то».

### 4.5. POS consistency

- `part_of_speech` соответствует использованию леммы в примерах.
- **Major:** явно неверный POS (например `run` как noun при всех примерах с `to run`).

### 4.6. Смысловая кластеризация (несколько карточек на лемму)

Если по одной лемме несколько карточек:

- карточки должны различаться **смыслами**, не быть дублями;
- примеры и определения логично различаются;
- нет «размазки» одного смысла на 2–3 почти одинаковых карточки.

**Major:** дубликаты по смыслу.

### 4.7. Качество примеров для Anki

- Пример самодостаточен, не слишком длинный.
- Даёт контекст для смысла.
- Без обрезков, мусора, битых символов.

**Minor:** слабый/длинный.  
**Major:** пример не помогает понять смысл или очевидно «не про лемму».

---

## 5) Метрики (на 30–50 карточках)

Считай **по карточкам** и при необходимости **по леммам**.

### 5.1. Основные метрики качества

1. `valid_schema_rate` — % карточек без schema‑проблем.
2. `lemma_in_every_example_rate` — % карточек, где лемма/словоформа есть **в каждом** примере.
3. `definition_ok_or_minor_rate`
4. `translation_ok_or_minor_rate`
5. `pos_consistency_rate`
6. `non_duplicate_senses_rate` — % лемм без дублей смыслов.

### 5.2. Ошибки по категориям

Для каждой карточки с Major/Invalid присвой одну или несколько категорий:

- **E1 Schema/Parsing:** missing fields, broken structure, invalid types.
- **E2 Index/Example mapping:** out-of-range / 0‑based vs 1‑based / examples не соответствуют индексам.
- **E3 Lemma mismatch:** лемма/словоформа отсутствует или другой токен.
- **E4 Wrong sense:** определение не соответствует примерам.
- **E5 Wrong POS:** неверный POS.
- **E6 Bad RU:** неверный перевод.
- **E7 Duplicates:** дубли смыслов.
- **E8 Low-quality examples:** примеры не подходят для Anki.

Считай: `count` и `%` по каждой категории.

---

## 6) Структура QC-отчёта (что именно написать)

### 6.1. Summary

- Сколько карточек проверено.
- Как выбирал (страты, seed/детерминизм).
- Итоговые метрики.
- Топ‑3 проблемы.

### 6.2. Метрики (таблица)

- valid_schema_rate  
- lemma_in_every_example_rate  
- definition_ok_or_minor_rate  
- translation_ok_or_minor_rate  
- pos_consistency_rate  
- non_duplicate_senses_rate

### 6.3. Ошибки по категориям (таблица)

E1…E8: `count`, `%`.

### 6.4. Примеры проблем (10–15 штук)

Для каждого проблемного кейса:

- lemma, part_of_speech;
- definition_en, definition_ru;
- examples (1–3);
- что не так (категория + 1–2 предложения);
- suggested fix (промпт / QC gate / парсер / данные).

### 6.5. Рекомендации (Action items)

Список конкретных действий:

- правка промпта (что именно добавить/убрать);
- QC‑gate: что считать **ошибкой** и где падать;
- retry‑политика: когда ретраить, когда блокировать;
- какие тесты добавить (unit/contract).

---

## 7) Правила «fail fast» (PASS/FAIL)

Считать прогон **неприемлемым (FAIL)**, если выполняется хотя бы одно:

- `valid_schema_rate < 0.98`  *(в strict-режиме требуй 1.0)*  
- `lemma_in_every_example_rate < 0.95`
- `pos_consistency_rate < 0.95`
- `major_or_invalid_rate > 0.05`
- `duplicates_rate` на частотных леммах `> 0.20`

В отчёте явно напиши: **PASS / FAIL** и конкретную причину.

---

## 8) Что сравнивать между прогонами (если есть baseline)

1. Долю ретраев и причины (из `download_log.json`).
2. Долю schema/parse ошибок.
3. lemma_in_every_example_rate.
4. pos_mismatch_rate (если считаете).
5. duplicates_rate.
6. Среднее число карточек на лемму.
7. Стоимость: число вызовов Standard API (если логируется).

---

## 9) Быстрый протокол на одну карточку (~30 сек)

1. Прочитать lemma и POS.
2. Прочитать examples — убедиться, что лемма реально используется.
3. Прочитать definition_en — соответствует ли примерам?
4. Прочитать definition_ru — соответствует ли definition_en?
5. Если карточек на лемму несколько — убедиться, что смыслы различаются.
6. Проставить оценку + категории (если есть).

---

## Связь с другими документами (если присутствуют в проекте)

- **Decision points (правила QC, контракты):** `docs/PIPELINE_B_QUALITY_DECISIONS.md` — правило “target в каждом примере”, контракции, headword, пороги strict/relaxed.
- Baseline и команды прогона: `docs/pipeline_b_refactor_baseline.md`
- Автоматический quality gate (команды, --gate): `docs/PIPELINE_B_QUALITY_GATE.md`
- Обзор пайплайнов и артефактов: `docs/PIPELINES_OVERVIEW.md`
