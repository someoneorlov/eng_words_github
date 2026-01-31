#!/usr/bin/env python3
"""
Регенерирует карточки с полными данными для ручной проверки.

Загружает результаты тестирования и регенерирует карточки с сохранением
полной информации (selected_examples, generated_examples, definition, translation).
"""

import json
import logging
import sys
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
TEST_RESULTS_PATH = Path("data/stage2_5_test/test_results.json")
OUTPUT_DIR = Path("data/stage2_5_test")
CACHE_DIR = OUTPUT_DIR / "llm_cache"
FULL_CARDS_OUTPUT = OUTPUT_DIR / "test_results_with_full_cards.json"


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


def main():
    """Регенерирует карточки с полными данными."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("РЕГЕНЕРАЦИЯ КАРТОЧЕК С ПОЛНЫМИ ДАННЫМИ")
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

    # Загружаем результаты тестирования
    if not TEST_RESULTS_PATH.exists():
        logger.error(f"Файл результатов не найден: {TEST_RESULTS_PATH}")
        sys.exit(1)

    results_df = pd.read_json(TEST_RESULTS_PATH)
    logger.info(f"  Загружено {len(results_df)} результатов тестирования")

    # Инициализация провайдеров
    logger.info("\n## Инициализация провайдеров")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)
    generator = SmartCardGenerator(
        provider=provider, cache=cache, book_name=BOOK_NAME, max_retries=2
    )

    # Регенерируем карточки
    logger.info("\n## Регенерация карточек с полными данными")
    full_results = []

    for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Регенерация"):
        lemma = row["lemma"]
        pos = row["pos"]

        # Находим исходную строку в aggregated_cards
        # Используем сохраненный индекс для точного восстановления (исправление проблемы с пропущенными карточками)
        row_index = row.get("row_index")
        if row_index is not None and row_index in cards_df.index:
            card_row = cards_df.loc[row_index]
            logger.debug(f"  Использован сохраненный индекс {row_index} для '{lemma}'")
        else:
            # Fallback на текущую логику (для старых файлов без row_index)
            card_row = cards_df[(cards_df["lemma"] == lemma) & (cards_df["pos"] == pos)].iloc[0]
            logger.debug(f"  Использован fallback поиск для '{lemma}'")

        # Получаем данные
        synset_group = normalize_synset_group(card_row.get("synset_group", []))
        primary_synset = card_row.get("primary_synset", "")
        wn_definition = card_row.get("definition", "")
        supersense = card_row.get("supersense", "")

        # Получаем примеры
        sentence_ids = card_row.get("sentence_ids", [])
        if isinstance(sentence_ids, str):
            sentence_ids = json.loads(sentence_ids)

        examples = [
            (sid, sentences_lookup.get(sid, "")) for sid in sentence_ids if sid in sentences_lookup
        ]

        if not examples:
            logger.warning(f"  Нет примеров для '{lemma}', пропускаем")
            continue

        # Валидация synset_group
        validation_result = validate_examples_for_synset_group(
            lemma=lemma,
            synset_group=synset_group,
            primary_synset=primary_synset,
            examples=examples,
            provider=provider,
            cache=cache,
        )

        if not validation_result["has_valid"]:
            logger.warning(f"  Нет валидных примеров для '{lemma}', пропускаем")
            continue

        valid_examples = [
            (sid, sentences_lookup[sid])
            for sid in validation_result["valid_sentence_ids"]
            if sid in sentences_lookup
        ]

        # Разметка по длине
        length_flags = mark_examples_by_length(valid_examples, max_words=50)

        # Проверка спойлеров
        spoiler_flags = check_spoilers(
            examples=valid_examples,
            provider=provider,
            cache=cache,
            book_name=BOOK_NAME,
        )

        # Выбор примеров
        selection = select_examples_for_generation(
            all_examples=valid_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        # Генерация карточки
        selected_examples_text = [ex for _, ex in selection["selected_from_book"]]
        card = generator.generate_card(
            lemma=lemma,
            pos=pos,
            supersense=supersense,
            wn_definition=wn_definition,
            examples=selected_examples_text,
            synset_group=synset_group,
            primary_synset=primary_synset,
            generate_count=selection["generate_count"],
        )

        if card is None:
            logger.warning(f"  Не удалось сгенерировать '{lemma}'")
            continue

        # Создаем полный результат
        full_result = row.to_dict()
        full_result["card_full"] = {
            "selected_examples": card.selected_examples,
            "generated_examples": card.generated_examples,
            "simple_definition": card.simple_definition,
            "translation_ru": card.translation_ru,
            "wn_definition": wn_definition,
            "synset_group": synset_group,
            "primary_synset": primary_synset,
            "supersense": supersense,
        }

        full_results.append(full_result)

    # Сохраняем результаты
    logger.info("\n## Сохранение результатов")
    with open(FULL_CARDS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    logger.info(f"  Сохранено {len(full_results)} карточек с полными данными в {FULL_CARDS_OUTPUT}")
    logger.info("\n" + "=" * 70)
    logger.info("РЕГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
