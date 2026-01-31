#!/usr/bin/env python3
"""
Тестирование нового промпта на 150 карточках.

Оценивает:
- Качество примеров (количество, длина, спойлеры)
- Качество определений и переводов
- Статистику генерации
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import SmartCardGenerator
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


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
        try:
            if pd.isna(synset_group):
                return []
            else:
                return [synset_group]
        except (ValueError, TypeError):
            return [synset_group]


def count_words(text: str) -> int:
    """Подсчитывает количество слов в тексте."""
    return len(text.split())


def analyze_card_quality(card) -> dict:
    """Анализирует качество карточки."""
    stats = {
        "selected_examples_count": len(card.selected_examples),
        "generated_examples_count": len(card.generated_examples),
        "total_examples_count": len(card.selected_examples) + len(card.generated_examples),
        "quality_scores_count": len(card.quality_scores),
        "selected_examples_lengths": [],
        "generated_examples_lengths": [],
        "definition_length": count_words(card.simple_definition),
        "has_translation": bool(card.translation_ru),
    }

    # Анализ длины примеров
    for ex in card.selected_examples:
        stats["selected_examples_lengths"].append(count_words(ex))

    for ex in card.generated_examples:
        stats["generated_examples_lengths"].append(count_words(ex))

    # Статистика по длине
    if stats["selected_examples_lengths"]:
        stats["selected_examples_avg_length"] = np.mean(stats["selected_examples_lengths"])
        stats["selected_examples_max_length"] = np.max(stats["selected_examples_lengths"])
        stats["selected_examples_min_length"] = np.min(stats["selected_examples_lengths"])
    else:
        stats["selected_examples_avg_length"] = 0
        stats["selected_examples_max_length"] = 0
        stats["selected_examples_min_length"] = 0

    if stats["generated_examples_lengths"]:
        stats["generated_examples_avg_length"] = np.mean(stats["generated_examples_lengths"])
        stats["generated_examples_max_length"] = np.max(stats["generated_examples_lengths"])
    else:
        stats["generated_examples_avg_length"] = 0
        stats["generated_examples_max_length"] = 0

    # Проверка на слишком длинные примеры (>50 слов)
    stats["selected_examples_too_long"] = sum(
        1 for l in stats["selected_examples_lengths"] if l > 50
    )
    stats["generated_examples_too_long"] = sum(
        1 for l in stats["generated_examples_lengths"] if l > 50
    )

    # Проверка на идеальную длину (10-30 слов)
    stats["selected_examples_ideal_length"] = sum(
        1 for l in stats["selected_examples_lengths"] if 10 <= l <= 30
    )
    stats["generated_examples_ideal_length"] = sum(
        1 for l in stats["generated_examples_lengths"] if 10 <= l <= 30
    )

    return stats


def test_new_prompt_150(limit: int = 150):
    """Тестирует новый промпт на выборке карточек."""
    logger.info("=" * 80)
    logger.info("ТЕСТИРОВАНИЕ НОВОГО ПРОМПТА НА 150 КАРТОЧКАХ")
    logger.info("=" * 80)

    # Загрузка данных
    logger.info("\n## Загрузка данных")
    aggregated_path = Path("data/synset_aggregation_full/aggregated_cards.parquet")
    tokens_path = Path("data/processed/american_tragedy_tokens.parquet")

    if not aggregated_path.exists():
        logger.error(f"Файл не найден: {aggregated_path}")
        return

    cards_df = pd.read_parquet(aggregated_path)
    logger.info(f"  Загружено карточек: {len(cards_df):,}")

    # Выбираем первые 150 карточек (или можно случайно)
    test_cards = cards_df.head(limit).copy()
    logger.info(f"  Выбрано для тестирования: {len(test_cards):,}")

    # Восстановление предложений
    logger.info("\n## Восстановление предложений")
    tokens_df = pd.read_parquet(tokens_path)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))
    logger.info(f"  Предложений доступно: {len(sentences_df):,}")

    # Инициализация генератора
    logger.info("\n## Инициализация SmartCardGenerator")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_new_prompt_cache"), enabled=True)
    generator = SmartCardGenerator(
        provider=provider,
        cache=cache,
        book_name="american_tragedy",
        max_retries=2,
    )
    logger.info(f"  Провайдер: {provider.model}")
    logger.info(f"  Кэш: {cache.cache_dir}")

    # Генерация карточек
    logger.info("\n## Генерация карточек")
    generated_cards = []
    failed_cards = []
    card_stats = []

    for idx, row in tqdm(test_cards.iterrows(), total=len(test_cards), desc="Генерация"):
        lemma = row["lemma"]
        primary_synset = row.get("primary_synset", "")
        synset_group = normalize_synset_group(row.get("synset_group", []))
        sentence_ids = row.get("sentence_ids", [])

        # Получаем примеры
        examples = [
            sentences_lookup.get(sid, "")
            for sid in sentence_ids
            if sid in sentences_lookup and sentences_lookup.get(sid, "")
        ]

        if not examples:
            logger.warning(f"  Нет примеров для '{lemma}' - пропускаем")
            failed_cards.append(
                {
                    "lemma": lemma,
                    "primary_synset": primary_synset,
                    "reason": "no_examples",
                }
            )
            continue

        # Генерируем карточку
        try:
            card = generator.generate_card(
                lemma=lemma,
                pos=row.get("pos", "noun"),
                supersense=row.get("supersense", ""),
                wn_definition=row.get("wn_definition", ""),
                examples=examples[:10],  # Ограничиваем до 10 примеров
                synset_group=synset_group,
                primary_synset=primary_synset,
            )

            if card is None:
                failed_cards.append(
                    {
                        "lemma": lemma,
                        "primary_synset": primary_synset,
                        "reason": "generation_failed",
                    }
                )
                continue

            generated_cards.append(card)

            # Анализ качества
            stats = analyze_card_quality(card)
            stats["lemma"] = lemma
            stats["primary_synset"] = primary_synset
            card_stats.append(stats)

        except Exception as e:
            logger.error(f"  Ошибка при генерации '{lemma}': {e}")
            failed_cards.append(
                {
                    "lemma": lemma,
                    "primary_synset": primary_synset,
                    "reason": f"error: {str(e)}",
                }
            )

    # Статистика генерации
    logger.info("\n" + "=" * 80)
    logger.info("РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ")
    logger.info("=" * 80)

    total = len(test_cards)
    successful = len(generated_cards)
    failed = len(failed_cards)

    logger.info("\n  Общая статистика:")
    logger.info(f"    Всего карточек: {total}")
    logger.info(f"    Успешно сгенерировано: {successful} ({successful/total*100:.1f}%)")
    logger.info(f"    Не удалось сгенерировать: {failed} ({failed/total*100:.1f}%)")

    if failed_cards:
        logger.info("\n  Причины ошибок:")
        reasons = defaultdict(int)
        for fc in failed_cards:
            reasons[fc["reason"]] += 1
        for reason, count in reasons.items():
            logger.info(f"    {reason}: {count}")

    # Статистика качества
    if card_stats:
        logger.info("\n  Статистика качества карточек:")

        # Количество примеров
        total_examples = [s["total_examples_count"] for s in card_stats]
        selected_examples = [s["selected_examples_count"] for s in card_stats]
        generated_examples = [s["generated_examples_count"] for s in card_stats]

        logger.info("\n    Количество примеров:")
        logger.info(f"      Всего примеров (среднее): {np.mean(total_examples):.1f}")
        logger.info(
            f"      Всего примеров (мин/макс): {np.min(total_examples)} / {np.max(total_examples)}"
        )
        logger.info(f"      Из книги (среднее): {np.mean(selected_examples):.1f}")
        logger.info(f"      Сгенерированных (среднее): {np.mean(generated_examples):.1f}")

        # Карточки с нужным количеством примеров (3-5)
        cards_with_3_5_examples = sum(1 for s in card_stats if 3 <= s["total_examples_count"] <= 5)
        logger.info(
            f"      Карточек с 3-5 примерами: {cards_with_3_5_examples} ({cards_with_3_5_examples/len(card_stats)*100:.1f}%)"
        )

        # Длина примеров
        all_selected_lengths = []
        all_generated_lengths = []
        for s in card_stats:
            all_selected_lengths.extend(s["selected_examples_lengths"])
            all_generated_lengths.extend(s["generated_examples_lengths"])

        if all_selected_lengths:
            logger.info("\n    Длина примеров из книги:")
            logger.info(f"      Средняя: {np.mean(all_selected_lengths):.1f} слов")
            logger.info(
                f"      Мин/макс: {np.min(all_selected_lengths)} / {np.max(all_selected_lengths)} слов"
            )
            logger.info(
                f"      Идеальная длина (10-30): {sum(1 for l in all_selected_lengths if 10 <= l <= 30)} / {len(all_selected_lengths)} ({sum(1 for l in all_selected_lengths if 10 <= l <= 30)/len(all_selected_lengths)*100:.1f}%)"
            )
            logger.info(
                f"      Слишком длинные (>50): {sum(1 for l in all_selected_lengths if l > 50)} / {len(all_selected_lengths)} ({sum(1 for l in all_selected_lengths if l > 50)/len(all_selected_lengths)*100:.1f}%)"
            )

        if all_generated_lengths:
            logger.info("\n    Длина сгенерированных примеров:")
            logger.info(f"      Средняя: {np.mean(all_generated_lengths):.1f} слов")
            logger.info(
                f"      Мин/макс: {np.min(all_generated_lengths)} / {np.max(all_generated_lengths)} слов"
            )
            logger.info(
                f"      Идеальная длина (10-30): {sum(1 for l in all_generated_lengths if 10 <= l <= 30)} / {len(all_generated_lengths)} ({sum(1 for l in all_generated_lengths if 10 <= l <= 30)/len(all_generated_lengths)*100:.1f}%)"
            )
            logger.info(
                f"      Слишком длинные (>50): {sum(1 for l in all_generated_lengths if l > 50)} / {len(all_generated_lengths)} ({sum(1 for l in all_generated_lengths if l > 50)/len(all_generated_lengths)*100:.1f}%)"
            )

        # Определения
        definition_lengths = [s["definition_length"] for s in card_stats]
        logger.info("\n    Определения:")
        logger.info(f"      Средняя длина: {np.mean(definition_lengths):.1f} слов")
        logger.info(
            f"      Мин/макс: {np.min(definition_lengths)} / {np.max(definition_lengths)} слов"
        )
        logger.info(
            f"      Соответствуют требованию (≤15 слов): {sum(1 for l in definition_lengths if l <= 15)} / {len(definition_lengths)} ({sum(1 for l in definition_lengths if l <= 15)/len(definition_lengths)*100:.1f}%)"
        )

        # Переводы
        cards_with_translation = sum(1 for s in card_stats if s["has_translation"])
        logger.info("\n    Переводы:")
        logger.info(
            f"      Карточек с переводом: {cards_with_translation} / {len(card_stats)} ({cards_with_translation/len(card_stats)*100:.1f}%)"
        )

        # Quality scores
        cards_with_scores = sum(1 for s in card_stats if s["quality_scores_count"] > 0)
        logger.info("\n    Quality scores:")
        logger.info(
            f"      Карточек с оценками качества: {cards_with_scores} / {len(card_stats)} ({cards_with_scores/len(card_stats)*100:.1f}%)"
        )

    # Статистика LLM
    stats = generator.stats()
    logger.info("\n  Статистика LLM:")
    logger.info(f"    Всего карточек обработано: {stats['total_cards']}")
    logger.info(f"    Успешно: {stats['successful']}")
    logger.info(f"    Ошибок: {stats['failed']}")
    logger.info(f"    Всего токенов: {stats['total_tokens']:,}")
    logger.info(f"    Стоимость: ${stats['total_cost']:.4f}")

    logger.info("\n  Статистика кэша:")
    logger.info(f"    Hits: {cache._hits}")
    logger.info(f"    Misses: {cache._misses}")
    if cache._hits + cache._misses > 0:
        hit_rate = cache._hits / (cache._hits + cache._misses) * 100
        logger.info(f"    Hit rate: {hit_rate:.1f}%")

    # Сохранение результатов
    output_dir = Path("data/quality_analysis/new_prompt_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем карточки
    cards_json = [card.to_dict() for card in generated_cards]
    with open(output_dir / "generated_cards.json", "w", encoding="utf-8") as f:
        json.dump(cards_json, f, ensure_ascii=False, indent=2)
    logger.info(f"\n  Сохранено карточек: {output_dir / 'generated_cards.json'}")

    # Сохраняем статистику
    stats_df = pd.DataFrame(card_stats)
    stats_df.to_parquet(output_dir / "card_stats.parquet")
    logger.info(f"  Сохранена статистика: {output_dir / 'card_stats.parquet'}")

    # Сохраняем информацию об ошибках
    if failed_cards:
        failed_df = pd.DataFrame(failed_cards)
        failed_df.to_parquet(output_dir / "failed_cards.parquet")
        logger.info(f"  Сохранена информация об ошибках: {output_dir / 'failed_cards.parquet'}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Тестирование завершено")
    logger.info("=" * 80)

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "card_stats": card_stats,
        "failed_cards": failed_cards,
    }


if __name__ == "__main__":
    import sys

    limit = 150
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Неверный limit: {sys.argv[1]}, используем 150")

    test_new_prompt_150(limit=limit)
