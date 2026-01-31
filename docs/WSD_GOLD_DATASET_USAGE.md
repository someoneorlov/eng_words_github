# WSD Gold Dataset usage

## What it is

The WSD Gold Dataset is a reference dataset for evaluating Word Sense Disambiguation algorithms.

**Contents:**
- 3000 examples from 4 books
- Labeled by LLM judges (Claude, Gemini, GPT)
- Split into dev (1500) and test_locked (1500)

---

## Quick start

### Evaluate current WSD

```bash
# Full evaluation on dev set (~2 minutes)
make eval-wsd-gold

# Quick evaluation (100 examples)
make eval-wsd-gold-quick
```

### Verify test set integrity

```bash
make verify-gold
# ✅ Checksum verified: 8e642428413d582c...
```

---

## Usage rules

### ✅ ALLOWED

1. **Use dev set for development**
   ```python
   from eng_words.wsd_gold.eval import load_gold_examples
   
   dev_examples = load_gold_examples("data/wsd_gold/gold_dev.jsonl")
   for ex in dev_examples:
       print(f"Word: {ex['target']['lemma']}")
       print(f"Context: {ex['context_window']}")
       print(f"Gold answer: {ex['gold_synset_id']}")
   ```

2. **Inspect errors on dev set**
   ```bash
   uv run python scripts/eval_wsd_on_gold.py --show-errors --top-errors 20
   ```

3. **Tune parameters on dev set**
   - Confidence threshold
   - Weights for different POS
   - Rules for constructions

### ❌ NOT ALLOWED

1. **Do NOT look at `gold_test_locked.jsonl`!**
   - Only for final comparison
   - Looking at it compromises results

2. **Do NOT tune parameters on test set**
   - That is data leakage
   - Results would be unfair

3. **Do NOT edit test set**
   - CI checks the checksum
   - Any change will break the check

---

## Data structure

### Example from the dataset

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

### Key fields

| Field | Description |
|------|-------------|
| `context_window` | Sentence containing the target word |
| `target.lemma` | Lemma of the target word |
| `target.pos` | Part of speech (NOUN, VERB, ADJ, ADV) |
| `candidates` | Possible senses from WordNet |
| `gold_synset_id` | Correct answer |
| `gold_confidence` | Confidence (1.0 = all LLMs agreed) |
| `gold_agreement` | Fraction of agreeing LLMs (0.67 = 2 of 3) |
| `metadata.baseline_top1` | Baseline WSD answer |
| `metadata.baseline_margin` | Margin over second candidate |

---

## Typical workflow

### 1. Establish baseline

```bash
make eval-wsd-gold
# Overall Accuracy: 47.5%
```

### 2. Change WSD code

```python
# src/eng_words/wsd/wordnet_backend.py
# ... your changes ...
```

### 3. Evaluate on dev set

```bash
make eval-wsd-gold
# Overall Accuracy: 52.3%  # Improvement!
```

### 4. Repeat 2–3 until goal is met

### 5. Final comparison on test set

```bash
uv run python scripts/eval_wsd_on_gold.py \
  --gold-path data/wsd_gold/gold_test_locked.jsonl
```

⚠️ **Do this only ONCE at the end!**

---

## Metrics

### By part of speech

```
ADJ  (adjectives):  56.8%
ADV  (adverbs):     53.0%
NOUN (nouns):       50.6%
VERB (verbs):       34.0%  ← Weak spot!
```

### By difficulty

```
Easy   (≤2 senses, margin ≥0.3):  80.8%
Medium (3–5 senses):             46.4%
Hard   (≥6 senses, margin <0.15): 25.7%
```

**Conclusion**: Focus on verbs and hard words.

---

## API

### Loading examples

```python
from eng_words.wsd_gold import load_gold_examples

examples = load_gold_examples("data/wsd_gold/gold_dev.jsonl")
print(f"Loaded {len(examples)} examples")
```

### Evaluating a single example

```python
from eng_words.wsd_gold import evaluate_single

result = evaluate_single(
    predicted_synset="bank.n.01",
    gold_synset="bank.n.01"
)
print(result["is_correct"])  # True
```

### Full evaluation

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

### Using LLM cache

```python
from eng_words.wsd_gold import LLMCache

cache = LLMCache(cache_dir="data/wsd_gold/cache")

# Check cache
cached = cache.get("example_id", "gpt-5.2")

# Save to cache
cache.set("example_id", "gpt-5.2", model_output)

# Stats
print(cache.stats)  # {"hits": 100, "misses": 5, "hit_rate": 0.95}
```

---

## FAQ

### Why split by book instead of 375 from each?

To avoid **data leakage**:
- The model may memorize author style
- Word frequency in the book
- Character context

Splitting by book is a stricter test.

### Can I add new examples?

Yes, but:
1. Do not touch `gold_test_locked.jsonl`
2. Add only to the dev set
3. Recompute checksum if you changed the test set

### What if an LLM made a labeling mistake?

About 2–5% of examples may have noisy gold labels. That is normal:
- On a large dataset, noise averages out
- Overall trend matters more than individual examples
- You can maintain a "quarantine" list for borderline cases
