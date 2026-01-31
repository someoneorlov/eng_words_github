#!/usr/bin/env python3
"""
Тестирование Этапа 2.5: Фильтрация длины и спойлеров ДО генерации.

Тестирует:
- Разметку по длине (mark_examples_by_length)
- Проверку спойлеров (check_spoilers)
- Логику выбора примеров (select_examples_for_generation)
- Генерацию карточек с новой логикой

Оценивает:
- Качество примеров (количество, длина, спойлеры)
- Статистику разметки (сколько too_long, сколько has_spoiler)
- Статистику выбора (2+1, 1+2, 0+3)
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
from eng_words.llm.smart_card_generator import (
    SmartCardGenerator,
    check_spoilers,
    mark_examples_by_length,
    select_examples_for_generation,
)
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation import validate_examples_for_synset_group

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
BOOK_NAME = "american_tragedy"
AGGREGATED_CARDS_PATH = Path("data/synset_aggregation_full/aggregated_cards.parquet")
TOKENS_PATH = Path(f"data/processed/{BOOK_NAME}_tokens.parquet")
OUTPUT_DIR = Path("data/stage2_5_test")
CACHE_DIR = OUTPUT_DIR / "llm_cache"

TEST_SIZE = 200  # Количество карточек для тестирования


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


def analyze_card_quality(card, selection_info: dict | None = None) -> dict:
    """Анализирует качество карточки."""
    stats = {
        "selected_examples_count": len(card.selected_examples),
        "generated_examples_count": len(card.generated_examples),
        "total_examples_count": len(card.selected_examples) + len(card.generated_examples),
        "quality_scores_count": len(card.quality_scores),
        "selected_examples_lengths": [],
        "generated_examples_lengths": [],
        "definition_length": count_words(card.simple_definition) if card.simple_definition else 0,
        "has_translation": bool(card.translation_ru),
        "skip_reason": card.skip_reason if hasattr(card, "skip_reason") else None,
    }

    # Добавляем информацию о выборе примеров
    if selection_info:
        stats["generate_count_requested"] = selection_info.get("generate_count", 0)
        stats["selected_from_book_count"] = len(selection_info.get("selected_from_book", []))

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

    return stats


def main():
    """Основная функция тестирования."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ТЕСТИРОВАНИЕ ЭТАПА 2.5: Фильтрация длины и спойлеров ДО генерации")
    logger.info("=" * 70)

    # Загрузка данных
    logger.info("\n## Загрузка данных")
    cards_df = pd.read_parquet(AGGREGATED_CARDS_PATH)
    logger.info(f"  Загружено {len(cards_df):,} карточек")

    tokens_df = pd.read_parquet(TOKENS_PATH)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))
    logger.info(f"  Загружено {len(sentences_df):,} предложений")

    # Инициализация провайдеров
    logger.info("\n## Инициализация провайдеров")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)
    generator = SmartCardGenerator(
        provider=provider, cache=cache, book_name=BOOK_NAME, max_retries=2
    )
    logger.info("  Провайдеры инициализированы")

    # Выборка для тестирования
    test_df = cards_df.head(TEST_SIZE)
    logger.info(f"\n## Тестирование на {len(test_df)} карточках")

    # Статистика
    all_stats = []
    length_stats = defaultdict(int)
    spoiler_stats = defaultdict(int)
    selection_stats = defaultdict(int)
    skipped_cards = []  # Список пропущенных карточек с причиной (вместо счетчика)
    skipped_with_reason = 0  # Карточки с skip_reason (отдельная категория)
    json_parse_errors = 0  # JSON parsing errors (должно быть 0%)
    api_errors = 0  # API errors
    other_errors = 0  # Другие неожиданные ошибки

    # Обработка карточек
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Генерация карточек"):
        try:
            lemma = row["lemma"]
            pos = row["pos"]
            supersense = row["supersense"]
            wn_definition = row.get("definition", row.get("wn_definition", ""))
            synset_group = normalize_synset_group(row.get("synset_group", []))
            primary_synset = row.get("primary_synset", "")

            # Получение примеров
            sentence_ids = row.get("sentence_ids", [])
            if isinstance(sentence_ids, str):
                sentence_ids = json.loads(sentence_ids)

            examples = [
                (sid, sentences_lookup.get(sid, ""))
                for sid in sentence_ids
                if sid in sentences_lookup
            ]

            if not examples:
                skipped_cards.append(
                    {
                        "lemma": lemma,
                        "pos": pos,
                        "skip_reason": "no_examples",
                        "skip_reason_detail": "Нет примеров в sentences_lookup",
                    }
                )
                logger.debug(f"  Пропущено '{lemma}': нет примеров")
                continue

            # Шаг 1: Валидация synset_group
            validation_result = validate_examples_for_synset_group(
                lemma=lemma,
                synset_group=synset_group,
                primary_synset=primary_synset,
                examples=examples,
                provider=provider,
                cache=cache,
            )

            if not validation_result["has_valid"]:
                # Сохраняем информацию о пропущенной карточке с деталями валидации
                skipped_info = {
                    "lemma": lemma,
                    "pos": pos,
                    "skip_reason": "no_valid_examples_for_synset_group",
                    "skip_reason_detail": "После валидации synset_group не найдено валидных примеров",
                    "synset_group": synset_group,
                    "primary_synset": primary_synset,
                    "total_examples": len(examples),
                    "invalid_examples_count": len(
                        validation_result.get("invalid_sentence_ids", [])
                    ),
                    "validation_details": validation_result.get("validation_details", {}),
                }
                skipped_cards.append(skipped_info)
                logger.debug(
                    f"  Пропущено '{lemma}': нет валидных примеров для synset_group (всего примеров: {len(examples)})"
                )
                continue

            valid_examples = [
                (sid, sentences_lookup[sid])
                for sid in validation_result["valid_sentence_ids"]
                if sid in sentences_lookup
            ]

            # Шаг 2: Разметка по длине
            length_flags = mark_examples_by_length(valid_examples, max_words=50)
            too_long_count = sum(1 for v in length_flags.values() if not v)
            length_stats["too_long"] += too_long_count
            length_stats["appropriate_length"] += len(length_flags) - too_long_count

            # Шаг 3: Проверка спойлеров
            spoiler_flags = check_spoilers(
                examples=valid_examples,
                provider=provider,
                cache=cache,
                book_name=BOOK_NAME,
            )
            spoiler_count = sum(1 for v in spoiler_flags.values() if v)
            spoiler_stats["has_spoiler"] += spoiler_count
            spoiler_stats["no_spoiler"] += len(spoiler_flags) - spoiler_count

            # Шаг 4: Выбор примеров для генерации
            selection = select_examples_for_generation(
                all_examples=valid_examples,
                length_flags=length_flags,
                spoiler_flags=spoiler_flags,
                target_count=3,
            )

            # Статистика выбора
            selected_count = len(selection["selected_from_book"])
            generate_count = selection["generate_count"]
            if selected_count == 2 and generate_count == 1:
                selection_stats["2+1"] += 1
            elif selected_count >= 1 and generate_count >= 1:
                selection_stats[f"{selected_count}+{generate_count}"] += 1
            elif selected_count == 0:
                selection_stats["0+3"] += 1

            # Шаг 5: Генерация карточки
            selected_examples_text = [ex for _, ex in selection["selected_from_book"]]
            card = generator.generate_card(
                lemma=lemma,
                pos=pos,
                supersense=supersense,
                wn_definition=wn_definition,
                examples=selected_examples_text,
                synset_group=synset_group,
                primary_synset=primary_synset,
                generate_count=generate_count,
            )

            if card is None:
                other_errors += 1
                logger.warning(f"  Не удалось сгенерировать '{lemma}': card is None")
                continue

            if hasattr(card, "skip_reason") and card.skip_reason:
                skipped_with_reason += 1
                logger.info(f"  Пропущено '{lemma}' (skip_reason: {card.skip_reason})")
                continue

            # Анализ качества
            stats = analyze_card_quality(card, selection_info=selection)
            stats["lemma"] = lemma
            stats["pos"] = pos
            stats["row_index"] = row.name  # Сохраняем индекс для точного восстановления

            # Добавляем полную информацию о карточке для ручной проверки
            stats["card_full"] = {
                "selected_examples": card.selected_examples,
                "generated_examples": card.generated_examples,
                "simple_definition": card.simple_definition,
                "translation_ru": card.translation_ru,
                "wn_definition": wn_definition,
                "synset_group": synset_group,
                "primary_synset": primary_synset,
            }

            all_stats.append(stats)

        except json.JSONDecodeError as e:
            json_parse_errors += 1
            logger.error(f"  JSON parsing error для '{row.get('lemma', 'unknown')}': {e}")
        except Exception as e:
            # Проверяем тип ошибки
            error_str = str(e).lower()
            if any(
                keyword in error_str for keyword in ["api", "timeout", "rate limit", "503", "429"]
            ):
                api_errors += 1
                logger.error(f"  API error для '{row.get('lemma', 'unknown')}': {e}")
            else:
                other_errors += 1
                logger.error(
                    f"  Неожиданная ошибка при обработке '{row.get('lemma', 'unknown')}': {e}"
                )
                logger.error(f"  Тип ошибки: {type(e).__name__}")
                import traceback

                logger.debug(f"  Traceback: {traceback.format_exc()}")

    # Вывод статистики
    logger.info("\n" + "=" * 70)
    logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    logger.info("=" * 70)

    logger.info("\n## Общая статистика")
    logger.info(f"  Всего карточек: {len(test_df)}")
    logger.info(
        f"  Успешно сгенерировано: {len(all_stats)} ({len(all_stats)/len(test_df)*100:.1f}%)"
    )
    logger.info(
        f"  Пропущено (нет валидных примеров для synset_group): {len(skipped_cards)} ({len(skipped_cards)/len(test_df)*100:.1f}%)"
    )
    logger.info(
        f"  Пропущено (skip_reason): {skipped_with_reason} ({skipped_with_reason/len(test_df)*100:.1f}%)"
    )

    # Детальная статистика по пропущенным карточкам
    if skipped_cards:
        skip_reasons = defaultdict(int)
        for card in skipped_cards:
            skip_reasons[card.get("skip_reason", "unknown")] += 1
        logger.info("\n## Детализация пропущенных карточек:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {reason}: {count}")

    logger.info("\n## Ошибки генерации:")
    logger.info(
        f"  JSON parsing errors: {json_parse_errors} ({json_parse_errors/len(test_df)*100:.1f}%) - ДОЛЖНО БЫТЬ 0%"
    )
    logger.info(f"  API errors: {api_errors} ({api_errors/len(test_df)*100:.1f}%)")
    logger.info(f"  Другие ошибки: {other_errors} ({other_errors/len(test_df)*100:.1f}%)")
    logger.info(
        f"  Всего ошибок: {json_parse_errors + api_errors + other_errors} ({(json_parse_errors + api_errors + other_errors)/len(test_df)*100:.1f}%)"
    )

    logger.info("\n## Статистика разметки по длине")
    logger.info(f"  Слишком длинные (>50 слов): {length_stats['too_long']}")
    logger.info(f"  Подходящая длина (<=50 слов): {length_stats['appropriate_length']}")

    logger.info("\n## Статистика проверки спойлеров")
    logger.info(f"  Со спойлерами: {spoiler_stats['has_spoiler']}")
    logger.info(f"  Без спойлеров: {spoiler_stats['no_spoiler']}")

    logger.info("\n## Статистика выбора примеров")
    for pattern, count in sorted(selection_stats.items()):
        logger.info(f"  {pattern}: {count}")

    if all_stats:
        stats_df = pd.DataFrame(all_stats)

        logger.info("\n## Качество карточек")
        logger.info(f"  Среднее количество примеров: {stats_df['total_examples_count'].mean():.2f}")
        logger.info(
            f"  Среднее количество из книги: {stats_df['selected_examples_count'].mean():.2f}"
        )
        logger.info(
            f"  Среднее количество сгенерированных: {stats_df['generated_examples_count'].mean():.2f}"
        )
        logger.info(
            f"  Средняя длина примеров из книги: {stats_df['selected_examples_avg_length'].mean():.2f}"
        )
        logger.info(
            f"  Средняя длина сгенерированных примеров: {stats_df['generated_examples_avg_length'].mean():.2f}"
        )
        logger.info(f"  Средняя длина определения: {stats_df['definition_length'].mean():.2f}")
        logger.info(
            f"  Карточек с переводом: {stats_df['has_translation'].sum()}/{len(stats_df)} ({stats_df['has_translation'].mean()*100:.1f}%)"
        )

        # Сохранение результатов
        output_file = OUTPUT_DIR / "test_results.json"
        stats_df.to_json(output_file, orient="records", indent=2, force_ascii=False)
        logger.info(f"\n  Результаты сохранены в {output_file}")

    # Сохранение информации о пропущенных карточках
    if skipped_cards:
        skipped_file = OUTPUT_DIR / "skipped_cards.json"
        with open(skipped_file, "w", encoding="utf-8") as f:
            json.dump(skipped_cards, f, indent=2, ensure_ascii=False)
        logger.info(
            f"\n  Информация о пропущенных карточках сохранена в {skipped_file} ({len(skipped_cards)} карточек)"
        )

    logger.info("\n" + "=" * 70)
    logger.info("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
