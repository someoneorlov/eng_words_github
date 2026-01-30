# Как пользоваться WSD Gold Dataset

## Что это такое

WSD Gold Dataset — эталонный набор данных для оценки качества алгоритмов Word Sense Disambiguation.

**Состав:**
- 3000 примеров из 4 книг
- Размечено LLM-судьями (Claude, Gemini, GPT)
- Разделено на dev (1500) и test_locked (1500)

---

## Быстрый старт

### Оценить текущий WSD

```bash
# Полная оценка на dev set (~2 минуты)
make eval-wsd-gold

# Быстрая оценка (100 примеров)
make eval-wsd-gold-quick
```

### Проверить целостность test set

```bash
make verify-gold
# ✅ Checksum verified: 8e642428413d582c...
```

---

## Правила использования

### ✅ МОЖНО

1. **Использовать dev set для разработки**
   ```python
   from eng_words.wsd_gold.eval import load_gold_examples
   
   dev_examples = load_gold_examples("data/wsd_gold/gold_dev.jsonl")
   for ex in dev_examples:
       print(f"Word: {ex['target']['lemma']}")
       print(f"Context: {ex['context_window']}")
       print(f"Gold answer: {ex['gold_synset_id']}")
   ```

2. **Смотреть на ошибки в dev set**
   ```bash
   uv run python scripts/eval_wsd_on_gold.py --show-errors --top-errors 20
   ```

3. **Подбирать параметры по dev set**
   - Threshold уверенности
   - Веса для разных POS
   - Правила для конструкций

### ❌ НЕЛЬЗЯ

1. **НЕ смотреть на `gold_test_locked.jsonl`!**
   - Только для финального сравнения
   - Посмотрел = скомпрометировал результаты

2. **НЕ подбирать параметры по test set**
   - Это data leakage
   - Результаты будут нечестными

3. **НЕ редактировать test set**
   - CI проверяет checksum
   - Любое изменение — сломает проверку

---

## Структура данных

### Пример из датасета

```json
{
  "example_id": "book:american_tragedy_wsd|sent:12008|tok:1",
  "source_id": "american_tragedy_wsd",
  "context_window": "I went to the bank to deposit money.",
  "target": {
    "surface": "bank",
    "lemma": "bank", 
    "pos": "NOUN",
    "char_span": [15, 19]
  },
  "candidates": [
    {
      "synset_id": "bank.n.01",
      "gloss": "financial institution",
      "examples": ["The bank raised interest rates."]
    },
    {
      "synset_id": "bank.n.02", 
      "gloss": "sloping land beside water",
      "examples": ["They sat on the river bank."]
    }
  ],
  "metadata": {
    "wn_sense_count": 10,
    "baseline_top1": "bank.n.01",
    "baseline_margin": 0.25,
    "is_multiword": false
  },
  "gold_synset_id": "bank.n.01",
  "gold_confidence": 1.0,
  "gold_agreement": 1.0,
  "gold_flags": []
}
```

### Ключевые поля

| Поле | Описание |
|------|----------|
| `context_window` | Предложение с целевым словом |
| `target.lemma` | Лемма целевого слова |
| `target.pos` | Часть речи (NOUN, VERB, ADJ, ADV) |
| `candidates` | Возможные значения из WordNet |
| `gold_synset_id` | Правильный ответ |
| `gold_confidence` | Уверенность (1.0 = все LLM согласились) |
| `gold_agreement` | Доля согласных LLM (0.67 = 2 из 3) |
| `metadata.baseline_top1` | Ответ baseline WSD |
| `metadata.baseline_margin` | Отрыв от второго кандидата |

---

## Типичный workflow

### 1. Установить baseline

```bash
make eval-wsd-gold
# Overall Accuracy: 47.5%
```

### 2. Внести изменения в WSD

```python
# src/eng_words/wsd/wordnet_backend.py
# ... ваши изменения ...
```

### 3. Проверить на dev set

```bash
make eval-wsd-gold
# Overall Accuracy: 52.3%  # Улучшение!
```

### 4. Повторить 2-3 пока не достигли цели

### 5. Финальное сравнение на test set

```bash
uv run python scripts/eval_wsd_on_gold.py \
  --gold-path data/wsd_gold/gold_test_locked.jsonl
```

⚠️ **Делать только ОДИН раз в конце!**

---

## Метрики

### По частям речи

```
ADJ  (прилагательные): 56.8%
ADV  (наречия):        53.0%
NOUN (существительные): 50.6%
VERB (глаголы):        34.0%  ← Слабое место!
```

### По сложности

```
Easy   (≤2 значения, margin ≥0.3):  80.8%
Medium (3-5 значений):              46.4%
Hard   (≥6 значений, margin <0.15): 25.7%
```

**Вывод**: Фокусироваться на глаголах и сложных словах.

---

## API

### Загрузка примеров

```python
from eng_words.wsd_gold import load_gold_examples

examples = load_gold_examples("data/wsd_gold/gold_dev.jsonl")
print(f"Loaded {len(examples)} examples")
```

### Оценка одного примера

```python
from eng_words.wsd_gold import evaluate_single

result = evaluate_single(
    predicted_synset="bank.n.01",
    gold_synset="bank.n.01"
)
print(result["is_correct"])  # True
```

### Полная оценка

```python
from eng_words.wsd import WordNetSenseBackend
from eng_words.wsd_gold import evaluate_wsd_on_gold

backend = WordNetSenseBackend()
results = evaluate_wsd_on_gold(
    gold_path="data/wsd_gold/gold_dev.jsonl",
    backend=backend
)

print(f"Accuracy: {results['metrics']['accuracy']:.1%}")
print(f"By POS: {results['by_pos']}")
print(f"By Difficulty: {results['by_difficulty']}")
```

### Использование кэша LLM

```python
from eng_words.wsd_gold import LLMCache

cache = LLMCache(cache_dir="data/wsd_gold/cache")

# Проверить в кэше
cached = cache.get("example_id", "gpt-5.2")

# Сохранить в кэш
cache.set("example_id", "gpt-5.2", model_output)

# Статистика
print(cache.stats)  # {"hits": 100, "misses": 5, "hit_rate": 0.95}
```

---

## FAQ

### Почему split по книгам, а не 375 из каждой?

Для предотвращения **data leakage**:
- Модель может запомнить стиль автора
- Частотность слов в книге
- Контекст персонажей

Split по книгам — более строгий тест.

### Можно ли добавить новые примеры?

Да, но нужно:
1. Не трогать `gold_test_locked.jsonl`
2. Добавлять только в dev set
3. Пересчитать checksum если меняли test

### Что делать если LLM ошибся в разметке?

Примерно 2-5% примеров могут иметь неточные gold-метки. Это нормально:
- На большом датасете шум усредняется
- Важнее общая тенденция, чем отдельные примеры
- Можно создать "quarantine" список для спорных случаев

