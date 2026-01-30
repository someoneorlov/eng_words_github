#!/usr/bin/env python3
"""
Тестирование валидации на 100 карточках с расчетом метрик.

Метрики:
- Precision: доля правильно определенных валидных примеров
  (требует ручной проверки или ground truth)
- Recall: доля найденных валидных примеров от всех валидных
  (требует ручной проверки или ground truth)
- Статистика: количество валидных/невалидных, процент карточек с валидными примерами
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation.synset_validator import validate_examples_for_synset_group

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_synset_group(synset_group):
    """Нормализует synset_group в список."""
    import numpy as np
    
    if isinstance(synset_group, (np.ndarray, list)):
        return list(synset_group) if len(synset_group) > 0 else []
    elif isinstance(synset_group, str):
        try:
            synset_group = json.loads(synset_group)
            if not isinstance(synset_group, list):
                synset_group = [synset_group]
            return synset_group
        except (json.JSONDecodeError, TypeError):
            return [synset_group] if synset_group else []
    elif synset_group is None:
        return []
    else:
        try:
            if pd.isna(synset_group):
                return []
            else:
                return [synset_group]
        except (ValueError, TypeError):
            return [synset_group]


def test_validation_100():
    """Тестирует валидацию на 100 карточках."""
    logger.info("=" * 70)
    logger.info("ТЕСТИРОВАНИЕ ВАЛИДАЦИИ НА 100 КАРТОЧКАХ")
    logger.info("=" * 70)
    
    # Загрузка данных
    logger.info("\n## Загрузка данных")
    aggregated_path = Path("data/synset_aggregation_full/aggregated_cards.parquet")
    tokens_path = Path("data/processed/american_tragedy_tokens.parquet")
    
    if not aggregated_path.exists():
        logger.error(f"Файл не найден: {aggregated_path}")
        return
    
    if not tokens_path.exists():
        logger.error(f"Файл не найден: {tokens_path}")
        return
    
    cards_df = pd.read_parquet(aggregated_path)
    logger.info(f"  Загружено карточек: {len(cards_df):,}")
    
    tokens_df = pd.read_parquet(tokens_path)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))
    logger.info(f"  Загружено предложений: {len(sentences_lookup):,}")
    
    # Выбираем 100 карточек для тестирования
    logger.info("\n## Выборка для тестирования")
    test_cards = cards_df.head(100)
    logger.info(f"  Выбрано карточек для теста: {len(test_cards)}")
    
    # Инициализация провайдера и кэша
    logger.info("\n## Инициализация LLM")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_validation_cache"), enabled=True)
    logger.info(f"  Провайдер: {provider.model}")
    logger.info(f"  Кэш: {cache.cache_dir}")
    
    # Тестирование на каждой карточке
    logger.info("\n## Тестирование валидации")
    results = []
    stats_by_pos = defaultdict(lambda: {"total": 0, "has_valid": 0, "valid_examples": 0, "invalid_examples": 0})
    
    for idx, row in tqdm(test_cards.iterrows(), total=len(test_cards), desc="Валидация"):
        lemma = row["lemma"]
        synset_group = row.get("synset_group", [])
        primary_synset = row.get("primary_synset", "")
        pos = row.get("pos", "UNKNOWN")
        sentence_ids = row.get("sentence_ids", [])
        
        # Нормализация synset_group
        synset_group = normalize_synset_group(synset_group)
        
        # Получаем примеры (берем все доступные, не ограничиваем)
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids
            if sid in sentences_lookup and sentences_lookup.get(sid, "")
        ]
        
        if not examples:
            results.append({
                "lemma": lemma,
                "pos": pos,
                "primary_synset": primary_synset,
                "synset_group": synset_group,
                "has_valid": False,
                "valid_count": 0,
                "invalid_count": 0,
                "total_examples": 0,
                "error": "No examples available"
            })
            stats_by_pos[pos]["total"] += 1
            continue
        
        # Валидация
        try:
            validation = validate_examples_for_synset_group(
                lemma=lemma,
                synset_group=synset_group,
                primary_synset=primary_synset,
                examples=examples,
                provider=provider,
                cache=cache,
            )
            
            valid_count = len(validation["valid_sentence_ids"])
            invalid_count = len(validation["invalid_sentence_ids"])
            
            results.append({
                "lemma": lemma,
                "pos": pos,
                "primary_synset": primary_synset,
                "synset_group": synset_group,
                "has_valid": validation["has_valid"],
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "total_examples": len(examples),
                "error": None
            })
            
            # Статистика по POS
            stats_by_pos[pos]["total"] += 1
            if validation["has_valid"]:
                stats_by_pos[pos]["has_valid"] += 1
            stats_by_pos[pos]["valid_examples"] += valid_count
            stats_by_pos[pos]["invalid_examples"] += invalid_count
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка при валидации {lemma}: {e}")
            results.append({
                "lemma": lemma,
                "pos": pos,
                "primary_synset": primary_synset,
                "synset_group": synset_group,
                "has_valid": False,
                "valid_count": 0,
                "invalid_count": 0,
                "total_examples": len(examples),
                "error": str(e)
            })
            stats_by_pos[pos]["total"] += 1
    
    # Итоговая статистика
    logger.info("\n" + "=" * 70)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("=" * 70)
    
    total = len(results)
    has_valid = sum(1 for r in results if r["has_valid"])
    total_valid = sum(r["valid_count"] for r in results)
    total_invalid = sum(r["invalid_count"] for r in results)
    total_examples = sum(r["total_examples"] for r in results)
    errors = sum(1 for r in results if r["error"])
    
    logger.info(f"\n  Общая статистика:")
    logger.info(f"    Всего карточек: {total}")
    logger.info(f"    С валидными примерами: {has_valid} ({has_valid/total*100:.1f}%)")
    logger.info(f"    Без валидных примеров: {total - has_valid} ({(total-has_valid)/total*100:.1f}%)")
    logger.info(f"    Всего примеров проверено: {total_examples}")
    logger.info(f"    Валидных примеров: {total_valid} ({total_valid/total_examples*100:.1f}% от всех)")
    logger.info(f"    Невалидных примеров: {total_invalid} ({total_invalid/total_examples*100:.1f}% от всех)")
    logger.info(f"    Ошибок: {errors}")
    
    # Статистика по POS
    logger.info(f"\n  Статистика по частям речи:")
    for pos in sorted(stats_by_pos.keys()):
        stats = stats_by_pos[pos]
        if stats["total"] > 0:
            has_valid_pct = stats["has_valid"] / stats["total"] * 100
            valid_pct = stats["valid_examples"] / (stats["valid_examples"] + stats["invalid_examples"]) * 100 if (stats["valid_examples"] + stats["invalid_examples"]) > 0 else 0
            logger.info(f"    {pos}:")
            logger.info(f"      Карточек: {stats['total']}")
            logger.info(f"      С валидными: {stats['has_valid']} ({has_valid_pct:.1f}%)")
            logger.info(f"      Валидных примеров: {stats['valid_examples']} ({valid_pct:.1f}% от всех)")
            logger.info(f"      Невалидных примеров: {stats['invalid_examples']}")
    
    # Статистика кэша
    logger.info(f"\n  Статистика кэша:")
    logger.info(f"    Hits: {cache._hits}")
    logger.info(f"    Misses: {cache._misses}")
    if cache._hits + cache._misses > 0:
        hit_rate = cache._hits / (cache._hits + cache._misses) * 100
        logger.info(f"    Hit rate: {hit_rate:.1f}%")
    
    # Метрики (объяснение)
    logger.info(f"\n  Метрики валидации:")
    logger.info(f"    Precision (точность):")
    logger.info(f"      = True Positives / (True Positives + False Positives)")
    logger.info(f"      = Доля правильно определенных валидных примеров")
    logger.info(f"      ⚠️  Требует ручной проверки или ground truth для расчета")
    logger.info(f"    Recall (полнота):")
    logger.info(f"      = True Positives / (True Positives + False Negatives)")
    logger.info(f"      = Доля найденных валидных примеров от всех валидных")
    logger.info(f"      ⚠️  Требует ручной проверки или ground truth для расчета")
    logger.info(f"    Статистика валидации:")
    logger.info(f"      Валидных найдено: {total_valid} ({total_valid/total_examples*100:.1f}%)")
    logger.info(f"      Невалидных отфильтровано: {total_invalid} ({total_invalid/total_examples*100:.1f}%)")
    logger.info(f"      Карточек с валидными примерами: {has_valid} ({has_valid/total*100:.1f}%)")
    
    # Сохраняем детальные результаты
    results_df = pd.DataFrame(results)
    output_path = Path("data/quality_analysis/validation_100_results.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)
    logger.info(f"\n  Детальные результаты сохранены: {output_path}")
    
    # Топ карточек без валидных примеров
    no_valid = [r for r in results if not r["has_valid"] and not r["error"]]
    if no_valid:
        logger.info(f"\n  Топ-10 карточек без валидных примеров:")
        for i, r in enumerate(no_valid[:10], 1):
            logger.info(f"    {i}. {r['lemma']} ({r['pos']}) - {r['primary_synset']} - примеров: {r['total_examples']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Тестирование завершено")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    test_validation_100()
