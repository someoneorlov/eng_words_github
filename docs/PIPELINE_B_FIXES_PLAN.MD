# План исправлений качества Pipeline B (после рефакторинга): Fail-Fast QC + нормализация текста + headwords/констракшены

**Формат:** рассчитан на выполнение LLM-агентом маленькими шагами с TDD, прогоном тестов и mini-run на sample после каждого блока (как в базовом плане). 

---

## A) Принципы разработки (финальная версия для этого плана)

Берём базовые принципы из твоего плана и **усиливаем “no-warnings-in-the-void”**:

1. **Fail-fast по инвариантам (default)**
   Любое нарушение контракта/схемы/согласованности/детерминизма/QC-порогов → **error** с понятным сообщением и “как исправить”. 

2. **Strict-by-default + осознанный relaxed**
   `strict` — по умолчанию; `relaxed` разрешён только явно и с порогами `--max-warning-rate`, `--max-warnings-absolute`, превышение порогов → **error** даже в relaxed. 

3. **Precision-first**
   Лучше **не выпускать карточку**, чем выпустить “почти нормальную”. В strict-mode невалидная карточка **не попадает в output**. 

4. **Модульность / pure vs IO**
   Максимум логики — в pure-функциях (детерминированно/тестируемо), IO/LLM — тонким слоем. 

5. **TDD и meaningful tests**
   Тесты ловят реальные регрессии: JSON-парсинг, индексы, маппинг примеров, нормализация, QC-gate, idempotency, offline режимы. 

6. **Экономия LLM calls + кэширование retry (обязательно)**
   Retry только по точным условиям; Standard API ответы кэшируются и переиспользуются.  

---

## B) “Ритуал” работы агента над каждой задачей (обязателен)

Для **каждой** подзадачи:

1. Зафиксировать контракт: вход/выход/ошибки/инварианты
2. Написать тесты (TDD)
3. Реализовать минимально
4. `pytest`
5. Mini-run на sample (limit 10–50)
6. Зафиксировать метрики/дельту
7. Рефактор/упрощение
8. Снова `pytest`
9. Только потом дальше 

---

## C) Общие критерии готовности этапа (quality gate)

Этап завершён, когда:

* все задачи этапа выполнены,
* все тесты проходят,
* на sample измерено качество и задокументировано,
* покрытие тестами ≥90% для изменённого модуля,
* strict-mode политика соблюдается. 

---

## D) Что именно чиним (scope этого плана)

Этот план закрывает 4 болевые зоны (то, что ты сейчас перечислил: “где hard errors”, “headwords”, “contractions”, и общий fail-fast):

### D.1. Жёсткая таксономия QC: где warnings → errors

Вместо “записали warning и пошли дальше” вводим **QC-gate**: либо *всегда error*, либо *пороговый gate* (ошибка при превышении). Это прямо соответствует твоему принципу “на тысячи лемм warnings никто не читает” и уже зафиксировано как цель этапа QC в исходном плане.  

### D.2. Нормализация текста для matching (contractions / кавычки / дефисы)

Проблема: карточка может быть “правильная”, но **lemma-matching/QC** валится из-за `don't` vs `do not`, типографских апострофов, длинных тире, weird whitespace, “I’m” и т.д.
Нужно: один детерминированный слой `normalize_text()` для:

* сопоставления lemma ↔ examples,
* извлечения word forms,
* QC-проверок,
  без изменения исходного текста карточек (или с контролируемым режимом).

### D.3. Headwords и Multiword Units (в т.ч. phrasal verbs)

Проблема: для некоторых лемм/карточек “headword” должен быть не просто `lemma`, а:

* `phrasal verb` (“look up”, “take off”),
* “fixed expression”,
* либо “canonical surface form” (если модель вернула “lemma”, но смысл в MWE).

Нужно: чёткое правило, **где мы вводим headword**, кто источник правды (Stage 1 vs LLM), и как это влияет на QC.

### D.4. Fail-fast правила: какие ошибки блокируют output

Нужно зафиксировать **точные условия**:

* что считается “contract error” (немедленно падать),
* что считается “QC error” (порогово/немедленно),
* что допускается только в relaxed.

---

# Этап 0 — Быстрые улучшения Stage 1 (маленькие, но полезные)

Цель: дать Pipeline B **лучшие/более стабильные данные** для QC и headwords.

## 0.1. Stage1 Manifest + sentences parquet (если ещё не всегда)

**Задачи**

* [ ] Убедиться, что Stage 1 стабильно пишет:

  * `*_tokens.parquet`
  * `*_sentences.parquet`
  * `stage1_manifest.json` (версии, модели spaCy, параметры токенизации, язык, дата)
* [ ] В manifest добавить:

  * `spacy_model_name`, `spacy_version`
  * параметры нормализации текста (если Stage 1 применяет)
  * checksum файлов (опционально)

**Тесты**

* [ ] Тест “manifest содержит обязательные поля”
* [ ] Тест “sentences.parquet содержит sentence_id + text и согласован с tokens.parquet”

**Критерии приёмки этапа 0**

* Stage 1 артефакты детерминированы на одной и той же книге (одинаковый checksum при повторном запуске)
* `pytest` зелёный

---

# Этап 1 — Таксономия ошибок + строгие контракты “что error, что gate”

> Это — главный “fail-fast upgrade”.

## 1.1. Ввести типизированные ошибки/статусы QC

**Задачи**

* [ ] Создать `eng_words/word_family/qc_types.py`:

  * `ErrorType` (enum)
  * `QCFinding` (dataclass: lemma, card_id/meaning_id, type, severity, message, context)
  * `QCPolicy` (strict/relaxed + пороги)
* [ ] Разделить:

  * **Contract errors**: schema, parsing, index mapping invariants
  * **QC errors**: lemma_in_example, pos_mismatch, duplicates, bad_examples
* [ ] В `cards_B_batch.json` (или внутреннем результате) хранить:

  * `errors[]` структурировано (как у тебя в плане) 
  * `warnings[]` только если relaxed и не превышены пороги 

**TDD**

* [ ] Тест: сериализация/десериализация `QCFinding`
* [ ] Тест: policy “strict => любой warning-класс превращается в error”
* [ ] Тест: policy “relaxed => warnings разрешены до порога, выше порога => exception”

## 1.2. Где “hard errors” (точные правила)

**Задачи**

* [ ] Зафиксировать список **hard errors** (в коде и в docs):

  1. invalid JSON / parse failure результата
  2. отсутствуют обязательные поля карточки
  3. `selected_example_indices` не int / out-of-range (если индексы обязательны)
  4. examples пустые при обязательности (strict)
  5. несогласованность batch_dir артефактов (нет lemma_examples при parse-results)
* [ ] Вынести это в одну функцию: `assert_contract_invariants(...) -> None | raises`

**TDD**

* [ ] Набор тестов на каждый hard error (минимальные входы)

**Критерии приёмки этапа 1**

* Любой hard error роняет pipeline “раньше”, чем мы пишем финальный output
* Ошибка содержит actionable hint (“как исправить”)

**Реализация (зафиксировано в коде)**

* Список hard errors и функция `assert_contract_invariants()`: `src/eng_words/word_family/contract.py`
* Типы QC и политика strict/relaxed: `src/eng_words/word_family/qc_types.py`
* Вызов контракта перед записью карточек: `batch_io.parse_results()` (и при `download_batch`)

---

# Этап 2 — Нормализация текста (contractions / апострофы / dash) для QC и matching

## 2.1. Ввести `normalize_text()` (pure)

**Задачи**

* [ ] Создать `eng_words/text_norm.py`:

  * `normalize_for_matching(text)`:

    * Unicode NFKC (или NFKD, если надо)
    * унификация апострофов (`’` → `'`)
    * унификация дефисов/тире (`—`, `–` → `-` или пробел-тире) — **строго детерминировано**
    * нормализация whitespace
  * `normalize_contractions(text)` — опционально (см. ниже)
* [ ] Важно: **не менять** примеры, которые пишутся в карточку, без явного флага. Нормализация по умолчанию — только для matching/QC.

**TDD**

* [ ] Тесты на типографские апострофы/тире/whitespace
* [ ] Golden tests: вход → ожидаемая нормализованная строка

## 2.2. Contractions: стратегия (чёткая, без сюрпризов)

Нужно выбрать **один** подход:

* **A) Contractions только для matching (рекомендовано)**
  Мы не переписываем текст примеров, но для проверки lemma_in_example используем expanded-вариант как альтернативу.
* **B) Contractions нормализуются в Stage 1**
  Рискованно: может “испортить” оригинальный текст.

**Задачи**

* [ ] Реализовать A):

  * `expand_contractions_for_matching(text)` (только en, небольшой словарь)
  * matching делает: `normalize(text)` и проверяет на обоих вариантах (original/expanded)
* [ ] Ограничить словарь contractions до наиболее частых (don’t, can’t, I’m, it’s, he’s, we’re, they’ve, won’t, didn’t, isn’t…) — чтобы не “галлюцинировать” разбор.

**TDD**

* [ ] Тесты: `"don't"` матчится по лемме `do` / форме `do not` (в рамках выбранной логики)
* [ ] Тесты: `"I'm"` vs `I am` не ломают поиск “am/be” в зависимости от forms

**Критерии приёмки этапа 2**

* lemma_in_example false positives/negatives снижаются на sample
* Никаких сетевых вызовов; чисто pure

**Реализация (зафиксировано в коде)**

* `eng_words.text_norm`: `normalize_for_matching()`, `expand_contractions_for_matching()`, `word_in_text_for_matching()`
* Проверка lemma-in-example в `batch_qc.cards_lemma_not_in_example()` использует `word_in_text_for_matching` (нормализация + контракции только для matching)

---

# Этап 3 — Headwords / MWEs / phrasal verbs: где вводить и как валидировать

## 3.1. Определить контракт: `lemma` vs `headword`

**Задачи**

* [ ] Зафиксировать правило в коде (и docs):

  * `lemma` — ключ сущности для группировки (как сейчас)
  * `headword` — строка для карточки (может отличаться)
* [ ] Добавить `headword` в карточку **опционально**, но:

  * если `headword` есть → QC проверяет `headword_in_example` (вместо lemma)
  * если нет → QC проверяет lemma/forms

## 3.2. Источник headword

Выбираем “source of truth”:

* **Stage 1** может дать кандидаты MWE/phrasal (если есть модуль)
* **LLM** может вернуть “headword” (но это риск: может выдумать)

**Рекомендуемая стратегия (precision-first):**

* `headword` разрешать только если:

  1. он **встречается** в примерах (после normalize) как substring/токены,
  2. и это **не ухудшает** строгий QC.

**Задачи**

* [ ] Реализовать функцию:

  * `infer_headword(card, lemma_examples) -> headword|None`:

    * если LLM вернул headword — принять только если найден в примерах
    * иначе None
* [ ] Добавить QC finding: `HEADWORD_NOT_IN_EXAMPLES` = hard QC error (strict)

**TDD**

* [ ] Тест: “look up” принимается как headword только если встречается
* [ ] Тест: “made up headword” отклоняется

**Критерии приёмки этапа 3**

* headword не создаёт новых ложных карточек
* строгая политика не пропускает "выдуманные" headwords

**Реализация (зафиксировано в коде)**

* Контракт: `lemma` — ключ группировки, `headword` — опциональная строка для карточки (например phrasal).
* `eng_words.word_family.headword`: `infer_headword(card)`, `headword_in_examples(card)` — headword принимается только если встречается в каждом примере (normalize, substring).
* `batch_core.parse_one_result`: если LLM вернул headword и он не в примерах — поле удаляется (не пишем в output).
* `batch_qc.cards_lemma_not_in_example`: при наличии headword проверяется headword в примерах; иначе lemma/forms. HEADWORD_NOT_IN_EXAMPLES ведёт к попаданию карточки в список (QC error).

---

# Этап 4 — lemma/headword in example: сделать это реальным gate, а не “warning”

Это напрямую продолжает твою мысль “warnings никто не читает”, и соответствует целям QC-этапа в исходном плане. 

## 4.1. Переписать проверку присутствия слова (robust)

Сейчас логика “словоформы + `\b...\b`” может давать ложные срабатывания на пунктуации/апострофах. 

**Задачи**

* [ ] Создать `match_target_in_text(target, text) -> bool`:

  * нормализация текста (этап 2)
  * токенизация на простом regex уровне (не spaCy, чтобы быстро и детерминированно)
  * поддержка:

    * слова с апострофом (`don't`)
    * hyphenated words
* [ ] В strict-mode:

  * если ни lemma/forms, ни headword не найдены в **каждом** выбранном примере → **error** (или пороговый gate, но я бы делал error сразу для выбранных примеров)
* [ ] В relaxed-mode:

  * warnings до порогов; выше порогов → error 

**TDD**

* [ ] Тесты на апостроф/тире/пунктуацию
* [ ] Тесты на “каждый пример vs хотя бы один пример” (зафиксировать правило)

## 4.2. Что делать при провале: drop vs retry

Precision-first говорит “лучше пропустить”. 

**Задачи**

* [ ] Зафиксировать стратегию:

  * default: **drop card** (не писать в output) + error в `errors[]`
  * опционально: retry только если причина “индексы плохие” (у тебя уже есть retry-policy структура) 

**TDD**

* [ ] Тест: card drop не ломает общий output schema
* [ ] Тест: счётчики stats корректны

**Критерии приёмки этапа 4**

* “lemma_not_in_example” перестаёт быть мусорным warning; становится явной причиной FAIL/PASS
* sample прогон либо PASS, либо падает с понятным списком лемм/карточек

**Реализация (зафиксировано в коде)**

* `eng_words.text_norm.match_target_in_text(target, text)`: слово — whole-word (normalize + contractions); фраза (пробел в target) — нормализованная подстрока. Правило: target должен быть в **каждом** примере (проверяет вызывающий код).
* `batch_qc.get_cards_failing_lemma_in_example(all_cards)`: возвращает список карточек, не прошедших проверку (для drop).
* В strict: в `parse_results` и перед финальной записью в `download_batch` карточки из get_cards_failing_lemma_in_example удаляются из output, в `validation_errors` добавляется запись с error_type `lemma_not_in_example`. Схема output и счётчики (cards_generated, stats) остаются корректными.

---

# Этап 5 — POS mismatch: QC-gate (вариант 1+), но fail-fast

У тебя в исходном плане уже есть: pos distribution hint и pos mismatch gate. 

## 5.1. POS distribution hint в prompt (детерминированно)

**Задачи**

* [x] В prompt добавить блок: “POS distribution in occurrences: VERB 80%, NOUN 20% …” 
* [x] Сделать формат стабильным (golden test)

## 5.2. POS-QC mismatch как gate

**Задачи**

* [x] Определить mismatch:

  * по выбранным примерам (sentence_id) смотрим POS токенов леммы
  * если карточка заявляет `verb`, а в примерах 0 occurrences VERB → mismatch
* [x] Strict:

  * mismatch → карточка не попадает в output, в validation_errors запись error_type pos_mismatch
* [ ] Relaxed:

  * warnings с порогами (порог 0.5–1% по карточкам — при превышении error)

**TDD**

* [x] Синтетика: токены с pos=VERB/NOUN, карточка заявляет одно — проверяем finding

**Критерии приёмки этапа 5**

* POS mismatch перестаёт быть “интересной статистикой”, становится автоматическим PASS/FAIL

**Реализация (зафиксировано в коде)**

* **5.1** `batch_core._format_pos_distribution(pos_distribution)` — стабильный формат "NOUN 20%, VERB 80%" (sorted, integer %). `build_prompt(..., pos_distribution=...)` добавляет блок в промпт. Golden-тесты в `test_batch_core`.
* **5.2** `clusterer.group_examples_by_lemma`: добавлено поле `pos_per_example` (list of POS по одному на пример, в порядке sentence_ids). `batch_io.render_requests`: пишет `lemma_pos_per_example.json`. `batch_qc._normalize_card_pos(part_of_speech)` → VERB/NOUN/ADJ/ADV; `get_cards_failing_pos_mismatch(all_cards, lemma_pos_per_example)` — карточки, у которых заявленный POS не встречается в выбранных примерах. В strict в `parse_results` после lemma_in_example эти карточки удаляются из output, в `validation_errors` — запись с error_type `pos_mismatch`. Файл `lemma_pos_per_example.json` опционален (старые батчи без него — POS gate не выполняется).

---

# Этап 6 — Duplicates / clustering sanity: QC-gate на уровне леммы

## 6.1. Определить “duplicate senses” (без LLM)

**Задачи**

* [x] Ввести cheap-heuristic:

  * нормализуем `definition_en` (lower, strip, collapse whitespace)
  * similarity по SequenceMatcher.ratio() ≥ threshold → дубликат
* [x] В strict:

  * дубликаты удаляются из output, в validation_errors — error_type duplicate_sense
* [ ] В relaxed:

  * warnings + порог (при превышении — error)

**TDD**

* [x] Тесты на 2 почти одинаковых definitions → duplicate
* [x] Тесты на разные → ok

**Критерии приёмки этапа 6**

* Дубли перестают тихо проходить

**Реализация (зафиксировано в коде)**

* `batch_qc._normalize_definition_for_similarity(text)`, `_definition_similarity(a, b)`; `DEFAULT_DUPLICATE_SENSE_THRESHOLD = 0.85`; `get_cards_failing_duplicate_sense(all_cards, threshold=...)` — по лемме с ≥2 карточками кластеры по similarity, возвращаются карточки-дубликаты (вторая и далее в кластере). В strict в `parse_results` после Stage 5 эти карточки удаляются, в validation_errors — duplicate_sense.

---

# Этап 7 — Интеграция: quality investigation как gate + единый отчёт

У тебя уже есть “QC превращён в автоматический gate” как критерий этапа 7. 

## 7.1. Сделать `scripts/run_quality_investigation.py` обязательным шагом

**Задачи**

* [x] Pipeline `download` (или отдельная команда) по окончании: отдельная команда `run_quality_investigation.py --gate`; при `download --run-gate` gate запускается после download; пишет отчёт `.md`.
* [x] Если gate не пройден → exit code != 0

## 7.2. PASS/FAIL политика (строго)

**Задачи**

* [x] В одном месте хранить thresholds: `qc_gate.QCGateThresholds` (max_lemma_not_in_example_rate, max_pos_mismatch_rate, max_duplicate_sense_rate, max_validation_rate; по умолчанию 0.0).
* [ ] CLI флаги для порогов (опционально), по умолчанию — строгие (0).

**Критерии приёмки этапа 7**

* На sample pipeline даёт либо PASS с отчётом, либо падает и говорит что чинить

**Реализация:** `qc_gate.py` (QCGateThresholds, evaluate_gate, load_result_and_evaluate_gate); `run_quality_investigation.py --gate`; `download --run-gate`.

---

# Этап 8 — Документация и decision points (чтобы агент не дрейфовал)

В исходном плане есть decision points (merge retry, parse-results, resume, schema, errors vs warnings). 

**Задачи**

* [x] Зафиксировать в одном месте (новый doc):

  1. **Правило “each example must contain target”**: lemma/forms/headword — что именно и в каких режимах
  2. Contractions strategy (только matching vs Stage 1)
  3. Headword контракт и источник правды
  4. QC thresholds и strict/relaxed semantics  
  → создан `docs/PIPELINE_B_QUALITY_DECISIONS.md`
* [x] Обновить `PIPELINE_B_MANUAL_QC_INSTRUCTIONS.md`: раздел «Автоматический gate vs ручной QC», ручной QC для диагноза, авто-gate для stop/go; ссылка на QUALITY_DECISIONS; нумерация секций сдвинута

---

## Итоговые критерии успешности всего плана (Definition of Done)

1. Pipeline на sample (30–50 лемм) **PASS** без ручного чтения предупреждений
2. Любое ухудшение (lemma mismatch, pos mismatch, duplicates, schema) приводит к **понятному FAIL**
3. Retry не раздувает стоимость: работает только по условиям и с кэшем 
4. Тесты meaningful, покрытие ≥90% на изменённых модулях 

---

## Мини-список “что именно агенту сделать первым” (чтобы начать без раздумий)

1. **Этап 1.1–1.2**: QC таксономия + hard errors (это задаёт “скелет”)
2. **Этап 2**: normalize_text + contractions-for-matching
3. **Этап 4**: lemma/headword in example → gate (strict)
4. Потом POS-gate и duplicates-gate