# PIPELINE B FIXES — супердетальный план задач (LLM-агент)

## 0) Цели, определения, инварианты

### Цели

1. **Fail-fast**: никакой “тысячи warnings”. Если карточки **невалидны по контракту** или **ломают строгий QC**, пайплайн **падает до записи финального output**.
2. **Precision-first**: лучше выкинуть карточку/лемму и перезапустить (или retry), чем записать мусор.
3. **Экономия LLM**: минимизировать повторные вызовы; **не пересчитывать “всё”** ради добавления phrasal/MWE режима — только **add-on** запросы на top-K кандидатов.
4. **Детерминизм**: одинаковый вход → одинаковые артефакты (включая order, seed, manifest, checksums).
5. **Роли компонентов (важно: не “ходить по кругу”)**:

   * **Stage 1 = deterministic extraction/normalization + candidate lists**, без семантики.
   * **Pipeline B (LLM) = единственное место, где рождаются карточки и семантика** (sense clustering, definitions, translation, example selection).

### Ключевые определения (важно зафиксировать в коде и docs)

* **lemma** — ключ группировки (как сейчас).
* **headword** — строка, которую мы показываем/учим на карточке (может быть `look up`, `take off`, `by the way`). Headword **не обязан** равняться lemma.
* **MWE (multi-word expression)** — устойчивое выражение/фраза (в т.ч. adverbial phrase, fixed expressions).
* **phrasal verb** — подтип MWE (`verb + particle`), must-have для изучения.
* **Mode** — режим генерации deck’а:

  * `word` (default): single-word deck (строго)
  * `phrasal`: phrasal verbs deck (отдельно, top-K)
  * `mwe`: fixed expressions deck (опционально позже, top-K)

### Инварианты “что обязано быть true”

#### Contract-инварианты (hard errors, всегда ERROR)

* JSON не парсится / структура сломана
* нет обязательных полей карточки
* `selected_example_indices` не int / out-of-range (по 1-based контракту)
* examples пустые, если в strict они обязательны
* несогласованность batch артефактов (например, нет `lemma_examples.json` когда он обязателен)
* отсутствуют обязательные входные артефакты Stage 1 (tokens/sentences/manifest)

#### QC-инварианты (в strict трактуются как ERROR)

* lemma/headword не находится в примерах (после корректного matching)
* сильный pos mismatch
* дубликаты смыслов (в рамках одной леммы/единицы)
* плохие примеры (обрезки/мусор/битая кодировка/не самостоятельные)
* headword не найден в примерах (если headword присутствует)
* неверная “единица обучения” для текущего режима (например, MWE в `word` mode)

---

## 1) Принципы разработки (обновлённые)

1. **Fail-fast by default**: strict-режим — норма. Warnings допускаются только в явно включённом relaxed-режиме и только ниже порогов.
2. **Контракты > эвристики**: сначала фиксируем контракт (что считаем ошибкой), потом улучшаем эвристики (matching, headword, MWE).
3. **Модульность**: одна функция = одна ответственность; IO отдельно от pure-логики.
4. **TDD**: тесты параллельно (сначала тест → минимальная реализация → рефактор).
5. **Измеримость**: каждый этап заканчивается мини-прогоном на sample + отчёт метрик/дифф.
6. **Экономия LLM**: новые режимы (phrasal/MWE) — **add-on**, не пересчёт word-deck.
7. **Кэширование**: LLM ответы кэшируются; повторные прогоны не должны ходить в сеть без нужды (`--from-file`, `retry_cache.jsonl`).
8. **Понятность через год**: типизированные структуры, явные схемы артефактов, schema_version, стабильные имена файлов.
9. **Детерминизм**: сортировки, seed, manifest/checksums, фиксированные правила нормализации.

---

## 2) Ритуал работы LLM-агента (одинаковый на всех этапах)

Для каждой задачи/подзадачи:

1. Выбрать задачу (самую маленькую, которую можно завершить).
2. Написать тесты (unit → затем интеграционный мок, если нужно).
3. Реализовать минимально, чтобы тест прошёл.
4. Прогнать sample (локально, без LLM где возможно).
5. Зафиксировать метрики/дифф “до/после” (коротко в отчёте этапа).
6. Рефактор (если надо) → все тесты зелёные.
7. Переход к следующей задаче.

---

## 3) Структура этапов (high-level)

* **Этап 0**: Быстрые улучшения Stage 1 (артефакты, manifest, sentences)
* **Этап 1**: Error taxonomy + контракт (hard errors) + strict/relaxed политика
* **Этап 2**: Нормализация текста для matching (апострофы/тире/whitespace + contractions *только для matching*)
* **Этап 3**: Headword контракт + QC для headword, без “выдумывания”
* **Этап 4**: Phrasal/MWE extraction в Stage 1 (candidate lists) + мини-эксперимент “Stage1 detector vs LLM detector”
* **Этап 5**: Отдельный режим “phrasal deck” / “MWE deck” (top-K, add-on, без пересчёта всего)
* **Этап 6**: QC-gate + отчёты + CI-подобный “quality stop”, retry-once strict
* **Этап 7**: Регрессии на твоих 49 карточках (полная проверка 49/49) + критерии PASS/FAIL

---

# Этап 0 — Stage 1: сделать вход стабильным и пригодным для headword/QC

### Цель этапа

Стабильные входные артефакты для Pipeline B + manifest, чтобы дальше можно было делать строгие QC проверки и режимы.

### Задачи

**0.1. Зафиксировать обязательные артефакты Stage 1**
Stage 1 **всегда** пишет:

* `<book>_tokens.parquet`
* `<book>_sentences.parquet`
* `stage1_manifest.json`

**0.2. Manifest: обязательные поля**
В `stage1_manifest.json` добавить/зафиксировать:

* `spacy_model_name`, `spacy_version`
* параметры нормализации текста (если есть)
* параметры токенизации/фильтрации, влияющие на кандидаты
* `created_at`, `book_id` / `book_name`
* (опционально) checksums (tokens/sentences) для детерминизма

**0.3. Согласованность tokens ↔ sentences**

* Все `sentence_id` в tokens должны существовать в sentences.
* sentences должны иметь колонки `sentence_id`, `text`.

### TDD (обязательно)

* Тест: manifest содержит обязательные поля
* Тест: `sentences.parquet` содержит `sentence_id,text` и согласован с tokens по `sentence_id`
* Тест: повторный прогон Stage 1 на одной книге → одинаковые checksums (или одинаковые ключевые поля + статистика)

### Acceptance criteria

* Повторный запуск Stage 1 на одной книге → одинаковые артефакты/manifest (детерминизм)
* `pytest` зелёный

---

# Этап 1 — Fail-fast upgrade: таксономия ошибок + контракт + strict/relaxed policy

### Цель этапа

Железобетонный контракт: pipeline **не пишет финальный output**, если есть hard error/QC-fail в strict.

## 1.1. Типизированные ошибки/QC findings

**Где в коде:** `src/eng_words/word_family/qc_types.py`

### Задачи

* Убедиться, что `ErrorType` покрывает:

  * contract-ошибки (schema/parse/IO invariants)
  * QC-ошибки (lemma/headword/POS/duplicates/examples quality)
* Убедиться, что `QCPolicy.severity_for_finding()`:

  * strict → любой QC finding = ERROR
  * relaxed → допускает warnings до порога, выше порога → exception
* Добавить/уточнить исключение “QCThresholdExceeded” (или аналог) для relaxed-порогов.

### TDD (обязательно)

* сериализация/десериализация `QCFinding`
* strict-policy превращает QC warning в ERROR
* relaxed-policy допускает warnings до порога, выше порога → exception

## 1.2. Контракт-инварианты: точные правила hard errors

**Где в коде:** `src/eng_words/word_family/contract.py`

### Задачи

* Зафиксировать `assert_contract_invariants(...)` со списком hard errors (из §0).
* Сделать сообщения ошибок **actionable**:

  * что сломано (файл/карточка/поле)
  * какой expected формат
  * где смотреть артефакты
  * что сделать (перегенерить requests / проверить lemma_examples / включить --from-file и т.п.)

### TDD

* Минимальный набор тестов на каждый hard error (маленькие фикстуры).

## 1.3. Встроить контракт до записи результата

**Где в коде:** `download_batch()` и/или `parse_results()` (в месте, где формируется итоговый dict)

### Задачи

* Контракт проверяется **до** `write_json(output_cards_path, ...)`.
* Если контракт падает — финальный output **не пишется**.
* Разрешено писать только debug артефакт (например `failed_run_report.json`) при строгом падении.

### Acceptance criteria (этап 1)

* Любой hard error роняет pipeline раньше записи финального output
* Ошибка actionable (понятно, что чинить)

---

# Этап 2 — Нормализация текста для matching (contractions/апострофы/тире)

### Цель этапа

Снизить ложные mismatch’и в lemma/headword matching **без изменения текста примеров в карточке**.

## 2.1. `eng_words/text_norm.py` — pure нормализация

### Задачи

Реализовать `normalize_for_matching(text: str) -> str`:

* Unicode NFKC
* `’ → '`
* `—/– → -`
* normalize whitespace (включая non-breaking)
* детерминизм (никаких “умных” преобразований)

### TDD

* Golden tests “вход → ожидаемая строка” (апострофы/тире/whitespace).
* Тест на детерминизм.

## 2.2. Contractions стратегия (только для matching)

### Задачи

Реализовать `expand_contractions_for_matching(text) -> str`:

* маленький ограниченный словарь частых форм (`don't`, `can't`, `I'm`, `we're`, `it's`, …)
* не пытаться “понимать грамматику”, только замены по словарю
* matching проверяет:

  * `normalize(original)`
  * `normalize(expanded(original))`

### TDD

* Тесты на типовые contractions (включая типографский апостроф).

## 2.3. Подключить в lemma/headword matching

**Где в коде:** QC проверка `lemma/headword in examples` (сейчас через validator/regex)

### Задачи

* Вынести общий матчинг в одну функцию:

  * `word_in_text_for_matching(word, text) -> bool`
* Гарантировать strict поведение:

  * если валидатор недоступен, **не молча пропускать** QC
  * либо (A) падать контракт-ошибкой “validator missing in strict”
  * либо (B) иметь альтернативную реализацию внутри проекта (предпочтительно)

### Acceptance criteria (этап 2)

* На sample реальный спад `lemma_not_in_example` (там, где виноваты типографика/констракции)
* Нет изменений в `examples` как строках карточки (только matching)

---

# Этап 3 — Headword / MWE / phrasal: где вводить и как валидировать (без магии)

### Цель этапа

Сделать headword управляемым: принимаем только подтверждённое примерами, иначе fail-fast.

## 3.1. Контракт: `lemma` vs `headword`

### Решение (фиксируем)

* `lemma` остаётся ключом группировки (word mode).
* `headword` — опциональное поле карточки.
* Если `headword` есть → QC проверяет `headword_in_examples` (строго) + unit-type соответствует mode.

## 3.2. Источник headword (precision-first)

### Политика

* LLM может вернуть headword, но мы **принимаем** его только если:

  * headword действительно найден в примерах после нормализации
  * headword соответствует текущему mode:

    * `word`: single-word (без пробелов/дефисов по политике), иначе ошибка
    * `phrasal/mwe`: multiword разрешён, но должен матчиться

### Задачи

* Реализовать `resolve_headword(card, examples, mode) -> headword|None`:

  * если LLM headword есть и валиден → оставить
  * если нет → None (для word mode)
  * если LLM headword есть, но не найден/невалиден → strict ERROR
* Добавить QC finding:

  * `QC_HEADWORD_NOT_IN_EXAMPLES` (strict-fatal)
  * `QC_HEADWORD_INVALID_FOR_MODE` (strict-fatal)

### TDD

* “look up” принимается только если встречается в примерах
* “made-up headword” → ошибка
* word mode: multiword headword → ошибка
* phrasal mode: multiword разрешён, но обязателен match

### Acceptance criteria (этап 3)

* Нет “выдуманных” headwords в strict: либо подтверждено, либо падение
* Word deck остаётся single-word по контракту

---

# Этап 4 — Stage 1: извлечение кандидатов MWE/phrasal + мини-эксперимент (Stage1 detector vs LLM detector)

### Цель этапа

Не засорять word deck MWE, но **не потерять phrasal verbs**: отделить кандидатов и сделать измеряемый выбор источника кандидатов.

## 4.1. Продуктовое правило (простое)

* **Word mode**: фиксированные выражения/фразы **не допускаются** (strict).
* **Phrasal/MWE mode**: отдельные decks, top-K кандидатов, add-on LLM calls.

## 4.2. Почему фильтр “no phrasal / no fixed expression” полезен (и как не выбросить phrasal verbs)

* Он снижает шум в word deck (например, чтобы `way` не превращался в “by the way”).
* Phrasal verbs не должны “исчезнуть навсегда”:

  * в word mode их можно исключать из headword, но
  * их нужно извлекать в **candidate list** и учить отдельным deck’ом.

## 4.3. Стандартизировать артефакт кандидатов Stage 1

### Задачи

* Добавить артефакт кандидатов:

  * `data/processed/<book>_mwe_candidates.parquet` (или под `stage1_manifest` путь)
  * Схема минимум:

    * `headword` (строка)
    * `type` (`phrasal_verb` | `fixed_expression` | `adverbial_phrase` | `other`)
    * `count`
    * `sample_sentence_ids` (список или отдельная таблица)
    * `source` (`stage1_detector`)
* Добавить флаги Stage 1:

  * `--extract-mwe-candidates`
  * `--no-fixed-expressions` (для word deck)
  * `--no-phrasal-verbs` (только если явно включено; по умолчанию phrasal идут в candidates)

### TDD

* Фикстура текста → извлекаются phrasal + fixed expressions, counts корректны
* Детерминизм: порядок/seed

## 4.4. Mini-этап: сравнение источников кандидатов (Stage1 vs LLM-detector)

### Цель

Проверить “не возвращаемся ли по кругу”: может ли LLM как детектор дать лучше кандидатов, чем Stage 1.

### Протокол

На sample (2000 sentences):

1. `phrasal_candidates_stage1`: из Stage 1 detector, `min_freq>=2`, `cap=200`
2. `phrasal_candidates_llm`: cheap LLM-detector батч:

   * вход: sentences
   * выход: JSON list `["look up", "take off", ...]`
   * затем `min_freq>=2`, `cap=200`
3. Ручной QC на 50 candidates (25+25):

   * precision: реально phrasal и полезно
   * noise: idioms/fixed/free collocations
   * dedupe: дубли/варианты

### Decision rule

* Если `precision(stage1) >= precision(llm) - 0.05` → берём Stage 1 candidates (дёшево, детерминированно)
* Иначе → используем LLM-detector как источник кандидатов **только для phrasal mode** (add-on), а Stage 1 делаем более консервативным

### Артефакты

* `data/experiment/phrasal_candidates_stage1.json`
* `data/experiment/phrasal_candidates_llm.json`
* `data/experiment/phrasal_candidates_comparison.md`

### Acceptance criteria (этап 4)

* Есть единый контракт candidate list’а
* Есть измеримый вывод, какой источник кандидатов выбираем

---

# Этап 5 — Отдельный “phrasal deck / MWE deck” без пересчёта всего

### Цель этапа

Сделать phrasal/mwe как add-on режимы: **только top-K** LLM calls, отдельные outputs, никакого пересчёта word deck.

## 5.1. Главная идея (экономия LLM)

* Word deck (lemma-based) считается как раньше.
* Phrasal/MWE deck — отдельный батч по top-K:

  * берём кандидатов (`cap=200/500/1000`)
  * собираем примеры по `headword`
  * вызываем LLM только для этих K

## 5.2. Можно ли объединить MWE и phrasal verbs в один класс?

Да как “MultiwordUnit”, но практично:

1. сначала `phrasal` (ROI, регулярность)
2. потом `mwe` (строже, меньше K)

## 5.3. Реализация режима в Pipeline B

Добавить параметр/флаг:

* `--deck=word|phrasal|mwe`
* `--top-k` (для phrasal/mwe)
* `--candidates-path` (если не стандартный)

Поведение:

* `word`: как сейчас (lemma grouping)
* `phrasal/mwe`: grouping по headword, примеры — предложения, где headword матчится

## 5.4. QC в режимах

* `word`: multiword headword запрещён, MWE запрещены
* `phrasal`: headword multiword разрешён, но **обязателен match** и тип должен быть `phrasal_verb`
* `mwe`: отдельные правила (строже), небольшой K

### TDD

* Pure: `build_prompt_for_unit(unit_key, examples, mode)`
* QC: `headword_in_examples` обязателен в phrasal/mwe
* Contract: outputs пишутся в разные файлы

### Acceptance criteria (этап 5)

* Phrasal deck генерируется без изменения word deck
* LLM cost растёт линейно от top-K, а не от размера книги

---

# Этап 6 — QC-gate: автоматический “quality stop” + отчёты + retry-once strict

### Цель этапа

Автоматический стоп качества: если FAIL — падение. Ретраи строго ограничены.

## 6.1. Привязать QC-gate к пайплайну

### Задачи

* После сборки результата (до записи финального output либо сразу после формирования in-memory результата):

  * запустить QC-gate
  * если FAIL → падать (strict)
* В strict режиме: любое QC finding = ERROR (через QCPolicy)

## 6.2. Retry once with super strict prompt

### Политика

* Если FAIL из-за:

  * pos mismatch
  * headword/lemma mismatch
  * invalid indices
  * empty examples
* Выполнить **один** retry (standard или thinking по retry_policy) с суперстрогим промптом:

  * “return only JSON”
  * “indices must be 1..N”
  * “headword must appear verbatim in examples”
* Если после retry всё ещё FAIL → падение (фиксить код/данные, не “варнить”)

## 6.3. Артефакты отчёта

* `download_log.json` включает структурированные причины:

  * какие леммы ретраились
  * почему ретрай
  * результат ретрая (fixed / still_fail)
* (опционально) `retry_results.jsonl` для дебага

### Acceptance criteria (этап 6)

* Нет “warnings как основной сигнал”
* FAIL → стоп до финального output
* Retry строго 1 раз (или по явной retry_policy)

---

# Этап 7 — Регрессия на твоих 49 карточках (49/49) + критерии PASS/FAIL

### Цель этапа

Не “репрезентативно”, а **каждую** карточку проверить автогейтом + ручным QC отчётом.

## 7.1. Полная проверка 49/49

### Задачи

* Прогнать pipeline (в strict) на тех же входах
* Прогнать QC-gate (авто)
* Прогнать ручной QC протокол на всех 49 (как ты требуешь):

  * для каждой карточки: OK/Minor/Major/Invalid + категории ошибок

## 7.2. Сравнение с прошлым прогоном

### Что сравнить (обязательно)

* исчезли ли проблемы с типографикой/апострофами/констракциями (Этап 2)
* стало ли меньше pos_mismatch (Этап 6 + строгий промпт)
* ушли ли “выдуманные headwords” (Этап 3)
* снизилась ли доля dropped cards при сохранении качества (precision-first)

## 7.3. Критерии PASS/FAIL (strict)

* `valid_schema_rate == 1.0`
* `lemma/headword_in_example_rate >= 0.98` (идеально 1.0 на 49)
* `pos_consistency_rate >= 0.98`
* `major_or_invalid_rate == 0` (на 49 это ожидаемо)
* если FAIL → список конкретных карточек + точные причины + какой этап чинит

### Acceptance criteria (этап 7)

* PASS на 49/49 или чёткий чеклист “что чинить” (без размытых warnings)

---

## Куда именно править (с конкретикой по файлам)

Минимальный список точек входа (обязателен для агента — правки должны быть локализованы):

* Контракт hard errors: `src/eng_words/word_family/contract.py`
* Типы и политика strict/relaxed: `src/eng_words/word_family/qc_types.py`
* QC проверки lemma/headword in examples: `src/eng_words/word_family/batch_qc.py` (и/или `example_validator`)
* QC gate evaluator/thresholds: `src/eng_words/word_family/qc_gate.py`
* Нормализация: `src/eng_words/text_norm.py` (или `eng_words/text_norm.py` — выбрать одно место, не дублировать)
* Headword логика: `src/eng_words/word_family/headword.py` (новый модуль) + интеграция в парсер результата
* Stage 1 manifest/sentences/candidates: `src/eng_words/pipeline/stage1.py` (или соответствующее место) + `stage1_manifest.json` writer
* Batch pipeline входы/выходы: `src/eng_words/word_family/batch.py` + `batch_io.py` (если есть) + CLI thin wrapper

---

## Ответы на твои вопросы (встроенные решения)

### Что делать с adverbial phrases / fixed expressions?

* В `word` mode: **запрещать** (иначе шум + сложность + качество падает).
* В `mwe` mode: разрешать **только как отдельный deck**, top-K, с отдельными QC правилами.

**Плюсы отдельного режима**: не ломаем word deck, экономим LLM (add-on), можно ограничить объём (200/500).
**Минусы**: добавляется ещё один режим/QC контракт и candidate list.

### Что такое MWE?

Multi-Word Expression — выражение из нескольких слов, которое ведёт себя как единица обучения (idiom/fixed expression/adverbial phrase/phrasal verb). Оно сложнее single-word, поэтому оно отдельно.

### Что даёт фильтр “no phrasal / no fixed expression”?

* Уменьшает шум в word deck (не превращаем “way” в “by the way”).
* **Важно:** phrasal verbs мы не “удаляем”, мы **выносим в отдельный режим/deck** через candidate list.

### Phrasal verbs — не потеряем?

Не потеряем, если:

* Stage 1 извлекает их в candidates
* Pipeline B генерит phrasal deck по top-K (add-on)
* Word deck остаётся строгим single-word

### Headword: “most frequent surface from among tokens” — что значит?

Это способ выбрать “как показывать слово”:

* lemma может быть абстрактной (`be`), а в книге чаще “is/was/are”
* headword можно выбрать как **самую частую surface-form** среди токенов этой леммы (но только как display, не как ключ группировки)
* Важно: headword принимаем только если он **подтверждён примерами** (Этап 3)

### Почему после ужесточения всё ещё есть некорректные варианты?

Потому что “ужесточили промпт” ≠ “сделали контракт+fail-fast”:

* пока есть типографика/констракции → matching даёт ложные FAIL/псевдо-OK
* пока headword не валидируется → LLM может “придумать”
* пока warnings не превращаются в gate → мусор попадает в output
  Этот план делает именно контракт+gate+retry-once+drop/stop.

---

## Что агент должен сдать в конце (артефакты)

1. Код + тесты (осмысленные) для ключевых функций:

   * contract invariants
   * QCPolicy strict/relaxed
   * normalize_for_matching + contractions expansion
   * headword resolution + validation
   * mode routing (word/phrasal/mwe)
2. Отчёты:

   * `investigation_report.md` (авто-отчёт)
   * QC-gate summary PASS/FAIL
   * регрессия 49/49: таблица по всем карточкам
3. Док-обновление:

   * “Роли Stage1 vs LLM”
   * “Mode contracts”
   * “Candidate extraction policy (precision-first)”
4. Команды запуска (пример):

   * word deck: create/download + QC gate
   * phrasal deck: Stage 1 candidates → create/download top-K → QC gate

---
