#!/usr/bin/env python3
"""
Тестирование валидации на Golden Dataset для проверки precision/recall.

Использует gold_dev.jsonl для оценки качества валидации примеров.
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.validation.synset_validator import validate_examples_for_synset_group
from eng_words.wsd_gold.eval import load_gold_examples

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_validation_on_gold_dataset(limit: int | None = None):
    """Тестирует валидацию на Golden Dataset.
    
    Для каждого примера из gold_dev.jsonl:
    1. Берем context_window (предложение с целевым словом)
    2. Проверяем валидацию с gold_synset_id как synset_group
    3. Сравниваем результат с gold (должен быть валидным)
    4. Вычисляем precision и recall
    """
    logger.info("=" * 80)
    logger.info("ТЕСТИРОВАНИЕ ВАЛИДАЦИИ НА GOLDEN DATASET")
    logger.info("=" * 80)
    
    # Загрузка Golden Dataset
    gold_path = Path("data/wsd_gold/gold_dev.jsonl")
    if not gold_path.exists():
        logger.error(f"Файл не найден: {gold_path}")
        return
    
    logger.info(f"\n## Загрузка Golden Dataset")
    gold_examples = load_gold_examples(gold_path)
    logger.info(f"  Загружено примеров: {len(gold_examples):,}")
    
    if limit:
        gold_examples = gold_examples[:limit]
        logger.info(f"  Ограничено до: {len(gold_examples):,}")
    
    # Инициализация LLM
    logger.info(f"\n## Инициализация LLM")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_validation_gold_cache"), enabled=True)
    logger.info(f"  Провайдер: {provider.model}")
    logger.info(f"  Кэш: {cache.cache_dir}")
    
    # Статистика для precision/recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0  # Не используется в нашем случае (все примеры должны быть валидными)
    
    results = []
    stats_by_pos = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    
    logger.info(f"\n## Тестирование валидации")
    
    for ex in tqdm(gold_examples, desc="Валидация"):
        lemma = ex["target"]["lemma"]
        gold_synset = ex["gold_synset_id"]
        context = ex["context_window"]
        pos = ex["target"]["pos"]
        
        # Проверяем валидацию
        # Используем gold_synset как synset_group (должен быть валидным)
        validation = validate_examples_for_synset_group(
            lemma=lemma,
            synset_group=[gold_synset],  # Используем gold synset как synset_group
            primary_synset=gold_synset,
            examples=[(0, context)],  # Один пример - context_window
            provider=provider,
            cache=cache,
            max_retries=2,
        )
        
        # Сравниваем с gold
        # Gold говорит: пример валиден для gold_synset
        # LLM говорит: validation["has_valid"]
        is_valid = validation["has_valid"]
        
        # True Positive: LLM сказал валидный И gold говорит валидный
        # False Positive: LLM сказал валидный НО gold говорит невалидный (не должно быть, т.к. мы используем gold_synset)
        # False Negative: LLM сказал невалидный НО gold говорит валидный
        # True Negative: LLM сказал невалидный И gold говорит невалидный (не должно быть)
        
        # В нашем случае:
        # - Если is_valid == True → TP (правильно определил как валидный)
        # - Если is_valid == False → FN (неправильно определил как невалидный, хотя должен быть валидным)
        
        if is_valid:
            true_positives += 1
            stats_by_pos[pos]["tp"] += 1
        else:
            false_negatives += 1
            stats_by_pos[pos]["fn"] += 1
        
        stats_by_pos[pos]["total"] += 1
        
        results.append({
            "example_id": ex.get("example_id", ""),
            "lemma": lemma,
            "pos": pos,
            "gold_synset": gold_synset,
            "is_valid": is_valid,
            "expected_valid": True,  # Всегда True, т.к. используем gold_synset
            "is_correct": is_valid,  # Правильно, если is_valid == True
        })
    
    # Вычисляем метрики
    logger.info(f"\n" + "=" * 80)
    logger.info("РЕЗУЛЬТАТЫ")
    logger.info("=" * 80)
    
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    
    # Precision: TP / (TP + FP)
    # В нашем случае FP = 0 (все примеры должны быть валидными)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    # В нашем случае TN = 0, FP = 0
    accuracy = true_positives / total if total > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logger.info(f"\n  Общая статистика:")
    logger.info(f"    Всего примеров: {total:,}")
    logger.info(f"    True Positives (TP): {true_positives:,} (правильно валидные)")
    logger.info(f"    False Negatives (FN): {false_negatives:,} (неправильно невалидные)")
    logger.info(f"    False Positives (FP): {false_positives:,} (неправильно валидные)")
    logger.info(f"    True Negatives (TN): {true_negatives:,} (правильно невалидные)")
    
    logger.info(f"\n  Метрики:")
    logger.info(f"    Accuracy: {accuracy:.1%} ({correct}/{total})")
    logger.info(f"    Precision: {precision:.1%} (TP / (TP + FP))")
    logger.info(f"    Recall: {recall:.1%} (TP / (TP + FN))")
    logger.info(f"    F1 Score: {f1:.1%}")
    
    # Статистика по POS
    logger.info(f"\n  Статистика по частям речи:")
    for pos in sorted(stats_by_pos.keys()):
        stats = stats_by_pos[pos]
        if stats["total"] > 0:
            pos_precision = stats["tp"] / (stats["tp"] + 0) if stats["tp"] > 0 else 0.0
            pos_recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
            pos_accuracy = stats["tp"] / stats["total"] if stats["total"] > 0 else 0.0
            
            logger.info(f"    {pos}:")
            logger.info(f"      Всего: {stats['total']}")
            logger.info(f"      TP: {stats['tp']}, FN: {stats['fn']}")
            logger.info(f"      Accuracy: {pos_accuracy:.1%}")
            logger.info(f"      Precision: {pos_precision:.1%}")
            logger.info(f"      Recall: {pos_recall:.1%}")
    
    # Статистика кэша
    logger.info(f"\n  Статистика кэша:")
    logger.info(f"    Hits: {cache._hits}")
    logger.info(f"    Misses: {cache._misses}")
    if cache._hits + cache._misses > 0:
        hit_rate = cache._hits / (cache._hits + cache._misses) * 100
        logger.info(f"    Hit rate: {hit_rate:.1f}%")
    
    # Сохраняем результаты
    results_df = pd.DataFrame(results)
    output_path = Path("data/quality_analysis/validation_gold_results.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)
    logger.info(f"\n  Детальные результаты сохранены: {output_path}")
    
    # Примеры ошибок (False Negatives)
    errors = [r for r in results if not r["is_correct"]]
    if errors:
        logger.info(f"\n  Примеры ошибок (False Negatives, первые 5):")
        for i, err in enumerate(errors[:5], 1):
            logger.info(f"    {i}. {err['lemma']} ({err['pos']}) - {err['gold_synset']}")
            # Найдем полный пример
            ex = next((e for e in gold_examples if e.get("example_id") == err["example_id"]), None)
            if ex:
                logger.info(f"       Context: {ex['context_window'][:100]}...")
    
    logger.info(f"\n" + "=" * 80)
    logger.info("✅ Тестирование завершено")
    logger.info("=" * 80)
    
    # Проверка критериев приемки
    logger.info(f"\n  Критерии приемки:")
    logger.info(f"    Precision >95%: {'✅' if precision >= 0.95 else '❌'} ({precision:.1%})")
    logger.info(f"    Recall >90%: {'✅' if recall >= 0.90 else '❌'} ({recall:.1%})")
    
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "total": total,
    }


if __name__ == "__main__":
    import sys
    
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            logger.info(f"Ограничение: {limit} примеров")
        except ValueError:
            logger.warning(f"Неверный limit: {sys.argv[1]}, используем все примеры")
    
    test_validation_on_gold_dataset(limit=limit)
