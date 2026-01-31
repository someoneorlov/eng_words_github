#!/usr/bin/env python3
"""
Тестирование модуля retry на различных карточках.

Проверяет работу retry логики на карточках с разным количеством примеров.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation.synset_validator import validate_examples_for_synset_group

load_dotenv()


def normalize_synset_group(synset_group):
    """Нормализует synset_group в список."""
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
        return [synset_group]


def test_retry_on_various_cards():
    """Тестирует retry на карточках с разным количеством примеров."""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ RETRY НА РАЗЛИЧНЫХ КАРТОЧКАХ")
    print("=" * 80)

    # Загрузка данных
    tokens_path = Path("data/processed/american_tragedy_tokens.parquet")
    aggregated_path = Path("data/synset_aggregation_full/aggregated_cards.parquet")

    tokens_df = pd.read_parquet(tokens_path)
    cards_df = pd.read_parquet(aggregated_path)

    # Восстановление предложений
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))

    # Инициализация LLM
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_validation_cache"), enabled=True)

    # Выбираем карточки с разным количеством примеров
    test_cards = [
        # Мало примеров (< 20)
        cards_df[cards_df["lemma"] == "abandon"].iloc[0],
        # Среднее количество (20-50)
        cards_df[cards_df["lemma"] == "absence"].iloc[0],
        # Много примеров (> 50) - able
        cards_df[(cards_df["lemma"] == "able") & (cards_df["primary_synset"] == "able.a.01")].iloc[
            0
        ],
    ]

    print(f"\nТестируем {len(test_cards)} карточек с разным количеством примеров\n")

    results = []

    for idx, row in enumerate(test_cards, 1):
        lemma = row["lemma"]
        primary_synset = row["primary_synset"]
        synset_group = normalize_synset_group(row.get("synset_group", []))
        sentence_ids = row.get("sentence_ids", [])

        # Получаем примеры
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids
            if sid in sentences_lookup and sentences_lookup.get(sid, "")
        ]

        print(f"\n{'─' * 80}")
        print(f"Карточка {idx}: {lemma} ({primary_synset})")
        print(f"  Примеров: {len(examples)}")
        print(f"  Synset group: {synset_group}")

        # Валидация с retry
        try:
            validation = validate_examples_for_synset_group(
                lemma=lemma,
                synset_group=synset_group,
                primary_synset=primary_synset,
                examples=examples,
                provider=provider,
                cache=cache,
                max_retries=2,
            )

            valid_count = len(validation["valid_sentence_ids"])
            invalid_count = len(validation["invalid_sentence_ids"])

            print("  ✅ Успешно обработано")
            print(f"     Валидных: {valid_count}")
            print(f"     Невалидных: {invalid_count}")
            print(f"     Has valid: {validation['has_valid']}")

            results.append(
                {
                    "lemma": lemma,
                    "primary_synset": primary_synset,
                    "total_examples": len(examples),
                    "valid_count": valid_count,
                    "invalid_count": invalid_count,
                    "has_valid": validation["has_valid"],
                    "error": None,
                }
            )

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            results.append(
                {
                    "lemma": lemma,
                    "primary_synset": primary_synset,
                    "total_examples": len(examples),
                    "valid_count": 0,
                    "invalid_count": 0,
                    "has_valid": False,
                    "error": str(e),
                }
            )

    # Итоговая статистика
    print(f"\n{'=' * 80}")
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)

    total = len(results)
    successful = sum(1 for r in results if r["error"] is None)
    errors = sum(1 for r in results if r["error"] is not None)

    print(f"  Всего карточек: {total}")
    print(f"  Успешно обработано: {successful} ({successful/total*100:.1f}%)")
    print(f"  Ошибок: {errors}")

    if successful > 0:
        total_examples = sum(r["total_examples"] for r in results if r["error"] is None)
        total_valid = sum(r["valid_count"] for r in results if r["error"] is None)
        total_invalid = sum(r["invalid_count"] for r in results if r["error"] is None)

        print(f"\n  Всего примеров: {total_examples}")
        print(f"  Валидных: {total_valid} ({total_valid/total_examples*100:.1f}%)")
        print(f"  Невалидных: {total_invalid} ({total_invalid/total_examples*100:.1f}%)")

    # Статистика кэша
    print("\n  Статистика кэша:")
    print(f"    Hits: {cache._hits}")
    print(f"    Misses: {cache._misses}")
    if cache._hits + cache._misses > 0:
        hit_rate = cache._hits / (cache._hits + cache._misses) * 100
        print(f"    Hit rate: {hit_rate:.1f}%")

    print(f"\n{'=' * 80}")
    print("✅ Тестирование завершено")
    print("=" * 80)

    return results


if __name__ == "__main__":
    test_retry_on_various_cards()
