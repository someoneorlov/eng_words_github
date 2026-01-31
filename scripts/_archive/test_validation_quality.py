#!/usr/bin/env python3
"""
Детальный анализ качества валидации на реальных данных.

Проверяет качество валидации, показывая примеры валидных/невалидных предложений
для ручной проверки.
"""

import json
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


def analyze_validation_quality():
    """Анализирует качество валидации на выборке."""
    logger.info("=" * 70)
    logger.info("АНАЛИЗ КАЧЕСТВА ВАЛИДАЦИИ")
    logger.info("=" * 70)

    # Загрузка данных
    logger.info("\n## Загрузка данных")
    aggregated_path = Path("data/synset_aggregation_full/aggregated_cards.parquet")
    tokens_path = Path("data/processed/american_tragedy_tokens.parquet")

    cards_df = pd.read_parquet(aggregated_path)
    tokens_df = pd.read_parquet(tokens_path)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))

    logger.info(f"  Загружено карточек: {len(cards_df):,}")
    logger.info(f"  Загружено предложений: {len(sentences_lookup):,}")

    # Выбираем 3 карточки для детального анализа
    logger.info("\n## Детальный анализ 3 карточек")
    test_cards = cards_df.head(3)

    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_validation_cache"), enabled=True)

    for idx, row in test_cards.iterrows():
        lemma = row["lemma"]
        synset_group = row.get("synset_group", [])
        primary_synset = row.get("primary_synset", "")
        sentence_ids = row.get("sentence_ids", [])

        # Обработка synset_group
        import numpy as np

        if isinstance(synset_group, (np.ndarray, list)):
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
            try:
                if pd.isna(synset_group):
                    synset_group = []
                else:
                    synset_group = [synset_group]
            except (ValueError, TypeError):
                synset_group = [synset_group]

        logger.info(f"\n{'='*70}")
        logger.info(f"Карточка: {lemma} ({primary_synset})")
        logger.info(f"Synset group: {synset_group}")
        logger.info(f"{'='*70}")

        # Получаем примеры
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids[:10]
            if sid in sentences_lookup and sentences_lookup.get(sid, "")
        ]

        if not examples:
            logger.warning("  Нет доступных примеров")
            continue

        # Валидация
        validation = validate_examples_for_synset_group(
            lemma=lemma,
            synset_group=synset_group,
            primary_synset=primary_synset,
            examples=examples,
            provider=provider,
            cache=cache,
        )

        logger.info("\n  Результаты валидации:")
        logger.info(f"    Валидных: {len(validation['valid_sentence_ids'])}")
        logger.info(f"    Невалидных: {len(validation['invalid_sentence_ids'])}")

        # Показываем валидные примеры
        logger.info("\n  ✅ ВАЛИДНЫЕ примеры:")
        for i, (sid, sentence) in enumerate(examples, 1):
            if sid in validation["valid_sentence_ids"]:
                logger.info(f"    {i}. [{sid}] {sentence[:100]}...")

        # Показываем невалидные примеры
        logger.info("\n  ❌ НЕВАЛИДНЫЕ примеры:")
        for i, (sid, sentence) in enumerate(examples, 1):
            if sid in validation["invalid_sentence_ids"]:
                logger.info(f"    {i}. [{sid}] {sentence[:100]}...")

        # Показываем детали валидации если есть
        if validation.get("validation_details"):
            logger.info("\n  Детали валидации:")
            for ex_idx, details in validation["validation_details"].items():
                if isinstance(details, dict):
                    reason = details.get("reason", "N/A")
                    valid = details.get("valid", "N/A")
                    logger.info(f"    Пример {ex_idx}: valid={valid}, reason={reason}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Анализ завершен")
    logger.info("=" * 70)


if __name__ == "__main__":
    analyze_validation_quality()
