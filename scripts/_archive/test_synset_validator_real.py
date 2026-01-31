#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функции validate_examples_for_synset_group.

Проверяет работу функции на реальных данных из aggregated_cards.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation.synset_validator import validate_examples_for_synset_group

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def test_validation_on_sample():
    """Тестирует валидацию на выборке из aggregated_cards."""
    logger.info("=" * 70)
    logger.info("ТЕСТИРОВАНИЕ ВАЛИДАЦИИ ПРИМЕРОВ")
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

    # Выбираем несколько карточек для тестирования
    logger.info("\n## Выборка для тестирования")
    test_cards = cards_df.head(5)  # Первые 5 карточек
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

    for idx, row in test_cards.iterrows():
        lemma = row["lemma"]
        synset_group = row.get("synset_group", [])
        primary_synset = row.get("primary_synset", "")
        sentence_ids = row.get("sentence_ids", [])

        logger.info(f"\n  Карточка {idx + 1}: {lemma} ({primary_synset})")
        logger.info(f"    Synset group: {synset_group}")
        logger.info(f"    Примеров в исходных данных: {len(sentence_ids)}")

        # Обработка synset_group (может быть list, string, или numpy array)
        import json

        import numpy as np

        # Сначала проверяем тип, потом обрабатываем
        if isinstance(synset_group, (np.ndarray, list)):
            # Convert numpy array or list to list
            synset_group = list(synset_group) if len(synset_group) > 0 else []
        elif isinstance(synset_group, str):
            try:
                synset_group = json.loads(synset_group)
                if not isinstance(synset_group, list):
                    synset_group = [synset_group]
            except (json.JSONDecodeError, TypeError):
                synset_group = [synset_group] if synset_group else []
        elif synset_group is None:
            synset_group = []
        else:
            # Попытка проверить через pd.isna только для скалярных значений
            try:
                if pd.isna(synset_group):
                    synset_group = []
                else:
                    synset_group = [synset_group]
            except (ValueError, TypeError):
                # Если не получается проверить, считаем что это валидное значение
                synset_group = [synset_group]

        # Получаем примеры
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids[:10]  # Ограничиваем до 10 для теста
            if sid in sentences_lookup and sentences_lookup.get(sid, "")
        ]

        if not examples:
            logger.warning("    ⚠️  Нет доступных примеров")
            results.append(
                {
                    "lemma": lemma,
                    "has_valid": False,
                    "valid_count": 0,
                    "invalid_count": 0,
                    "error": "No examples available",
                }
            )
            continue

        logger.info(f"    Примеров для валидации: {len(examples)}")

        # Валидация
        try:
            validation = validate_examples_for_synset_group(
                lemma=lemma,
                synset_group=synset_group,  # Уже обработан выше
                primary_synset=primary_synset,
                examples=examples,
                provider=provider,
                cache=cache,
            )

            valid_count = len(validation["valid_sentence_ids"])
            invalid_count = len(validation["invalid_sentence_ids"])

            logger.info(f"    ✅ Валидных: {valid_count}")
            logger.info(f"    ❌ Невалидных: {invalid_count}")
            logger.info(f"    Has valid: {validation['has_valid']}")

            if validation["has_valid"]:
                logger.info("    Примеры валидных предложений:")
                for sid in validation["valid_sentence_ids"][:3]:
                    example_text = sentences_lookup.get(sid, "")[:80]
                    logger.info(f"      - {sid}: {example_text}...")

            results.append(
                {
                    "lemma": lemma,
                    "has_valid": validation["has_valid"],
                    "valid_count": valid_count,
                    "invalid_count": invalid_count,
                    "error": None,
                }
            )

        except Exception as e:
            logger.error(f"    ❌ Ошибка при валидации: {e}")
            results.append(
                {
                    "lemma": lemma,
                    "has_valid": False,
                    "valid_count": 0,
                    "invalid_count": 0,
                    "error": str(e),
                }
            )

    # Итоговая статистика
    logger.info("\n" + "=" * 70)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("=" * 70)

    total = len(results)
    has_valid = sum(1 for r in results if r["has_valid"])
    total_valid = sum(r["valid_count"] for r in results)
    total_invalid = sum(r["invalid_count"] for r in results)
    errors = sum(1 for r in results if r["error"])

    logger.info(f"  Всего карточек: {total}")
    logger.info(f"  С валидными примерами: {has_valid} ({has_valid/total*100:.1f}%)")
    logger.info(f"  Всего валидных примеров: {total_valid}")
    logger.info(f"  Всего невалидных примеров: {total_invalid}")
    logger.info(f"  Ошибок: {errors}")

    logger.info("\n  Статистика кэша:")
    logger.info(f"    Hits: {cache._hits}")
    logger.info(f"    Misses: {cache._misses}")
    if cache._hits + cache._misses > 0:
        hit_rate = cache._hits / (cache._hits + cache._misses) * 100
        logger.info(f"    Hit rate: {hit_rate:.1f}%")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Тестирование завершено")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    test_validation_on_sample()
