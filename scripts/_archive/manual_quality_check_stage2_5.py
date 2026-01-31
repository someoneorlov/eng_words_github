#!/usr/bin/env python3
"""
Ручная проверка качества карточек Этапа 2.5.

Проверяет все 115 успешно сгенерированных карточек на соответствие требованиям:
1. Ровно 3 примера
2. Все примеры без спойлеров (уже проверено ДО генерации)
3. Все примеры нормальной длины (<=50 слов)
4. Определение краткое (<=15 слов)
5. Есть перевод
6. Примеры соответствуют synset_group (уже проверено ДО генерации)
7. Примеры из книги + сгенерированные правильного качества

Проверка выполняется батчами по 10 карточек для удобства.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
BOOK_NAME = "american_tragedy"
AGGREGATED_CARDS_PATH = Path("data/synset_aggregation_full/aggregated_cards.parquet")
TOKENS_PATH = Path(f"data/processed/{BOOK_NAME}_tokens.parquet")
TEST_RESULTS_PATH = Path("data/stage2_5_test/test_results.json")
FULL_CARDS_PATH = Path("data/stage2_5_test/test_results_with_full_cards.json")
OUTPUT_DIR = Path("data/stage2_5_test")
CACHE_DIR = OUTPUT_DIR / "llm_cache"

BATCH_SIZE = 10  # Проверять по 10 карточек за раз


def count_words(text: str) -> int:
    """Подсчитывает количество слов."""
    return len(text.split()) if text else 0


def check_card_quality(card_data: dict, card_index: int) -> dict:
    """Проверяет качество одной карточки.

    Returns:
        dict with quality checks:
        - has_3_examples: bool
        - all_examples_appropriate_length: bool
        - definition_short: bool
        - has_translation: bool
        - issues: list[str] - список проблем
    """
    issues = []
    checks = {
        "card_index": card_index,
        "lemma": card_data.get("lemma", ""),
        "pos": card_data.get("pos", ""),
    }

    # Проверка 1: Ровно 3 примера
    total_count = card_data.get("total_examples_count", 0)
    if total_count == 3:
        checks["has_3_examples"] = True
    else:
        checks["has_3_examples"] = False
        issues.append(f"Неправильное количество примеров: {total_count} (ожидается 3)")

    # Проверка 2: Все примеры нормальной длины (<=50 слов)
    max_length = card_data.get("selected_examples_max_length", 0)
    if max_length <= 50:
        checks["all_examples_appropriate_length"] = True
    else:
        checks["all_examples_appropriate_length"] = False
        issues.append(f"Есть примеры длиннее 50 слов: {max_length}")

    # Проверка сгенерированных примеров
    gen_max_length = card_data.get("generated_examples_max_length", 0)
    if gen_max_length > 50:
        checks["all_examples_appropriate_length"] = False
        issues.append(f"Есть сгенерированные примеры длиннее 50 слов: {gen_max_length}")

    # Проверка 3: Определение краткое (<=15 слов)
    def_length = card_data.get("definition_length", 0)
    if def_length <= 15:
        checks["definition_short"] = True
    else:
        checks["definition_short"] = False
        issues.append(f"Определение слишком длинное: {def_length} слов (лимит 15)")

    # Проверка 4: Есть перевод
    has_translation = card_data.get("has_translation", False)
    checks["has_translation"] = has_translation
    if not has_translation:
        issues.append("Нет перевода")

    # Проверка 5: Баланс примеров
    selected_count = card_data.get("selected_examples_count", 0)
    generated_count = card_data.get("generated_examples_count", 0)
    checks["selected_count"] = selected_count
    checks["generated_count"] = generated_count

    # Проверка 6: Средняя длина примеров разумная
    avg_length = card_data.get("selected_examples_avg_length", 0)
    if avg_length > 50:
        issues.append(f"Средняя длина примеров слишком большая: {avg_length:.1f} слов")

    checks["issues"] = issues
    checks["has_issues"] = len(issues) > 0

    return checks


def display_card_details(card_data: dict, check_result: dict, sentences_lookup: dict):
    """Выводит детальную информацию о карточке для проверки."""
    lemma = card_data.get("lemma", "")
    pos = card_data.get("pos", "")
    card_full = card_data.get("card_full", {})

    print(f"\n{'='*80}")
    print(f"Карточка #{check_result['card_index'] + 1}: {lemma} ({pos})")
    print(f"{'='*80}")

    # Информация о слове
    if card_full.get("synset_group"):
        print("\n📖 Информация о слове:")
        print(f"  - Synset Group: {', '.join(card_full.get('synset_group', []))}")
        print(f"  - Primary Synset: {card_full.get('primary_synset', '')}")
        print(f"  - WordNet Definition: {card_full.get('wn_definition', '')[:100]}...")

    # Определение и перевод
    print("\n📝 Определение и перевод:")
    print(f"  - Простое определение: {card_full.get('simple_definition', 'N/A')}")
    print(f"    (Длина: {count_words(card_full.get('simple_definition', ''))} слов)")
    print(f"  - Перевод: {card_full.get('translation_ru', 'N/A')}")

    # Примеры из книги
    selected_examples = card_full.get("selected_examples", [])
    if selected_examples:
        print(f"\n📚 Примеры из книги ({len(selected_examples)}):")
        for i, ex in enumerate(selected_examples, 1):
            word_count = count_words(ex)
            status = "✅" if word_count <= 50 else "❌"
            print(f"  {status} Пример {i} ({word_count} слов):")
            print(f'      "{ex}"')
    elif card_full.get("valid_examples"):
        # Если selected_examples нет, показываем valid_examples (первые из валидных)
        valid_examples = card_full.get("valid_examples", [])
        print(f"\n📚 Валидные примеры из книги (первые {len(valid_examples)}):")
        for i, ex in enumerate(valid_examples[:3], 1):  # Показываем первые 3
            word_count = count_words(ex)
            status = "✅" if word_count <= 50 else "❌"
            print(f"  {status} Пример {i} ({word_count} слов):")
            print(f'      "{ex}"')
        if len(valid_examples) > 3:
            print(f"  ... и еще {len(valid_examples) - 3} примеров")

    # Сгенерированные примеры
    generated_examples = card_full.get("generated_examples", [])
    if generated_examples:
        print(f"\n✨ Сгенерированные примеры ({len(generated_examples)}):")
        for i, ex in enumerate(generated_examples, 1):
            word_count = count_words(ex)
            status = "✅" if word_count <= 50 else "❌"
            print(f"  {status} Пример {i} ({word_count} слов):")
            print(f'      "{ex}"')
    else:
        print(
            "\n✨ Сгенерированные примеры: не найдены в данных (нужно регенерировать карточки с сохранением card_full)"
        )

    # Статистика
    print("\n📊 Статистика:")
    print(f"  - Примеров из книги: {card_data.get('selected_examples_count', 0)}")
    print(f"  - Сгенерированных примеров: {card_data.get('generated_examples_count', 0)}")
    print(f"  - Всего примеров: {card_data.get('total_examples_count', 0)}")
    print(
        f"  - Средняя длина из книги: {card_data.get('selected_examples_avg_length', 0):.1f} слов"
    )
    print(f"  - Макс. длина из книги: {card_data.get('selected_examples_max_length', 0)} слов")
    print(
        f"  - Средняя длина сгенерированных: {card_data.get('generated_examples_avg_length', 0):.1f} слов"
    )
    print(f"  - Длина определения: {card_data.get('definition_length', 0)} слов")
    print(f"  - Есть перевод: {card_data.get('has_translation', False)}")

    # Проверки
    print("\n✅ Проверки:")
    checks = [
        ("Ровно 3 примера", check_result.get("has_3_examples", False)),
        (
            "Все примеры нормальной длины (<=50 слов)",
            check_result.get("all_examples_appropriate_length", False),
        ),
        ("Определение краткое (<=15 слов)", check_result.get("definition_short", False)),
        ("Есть перевод", check_result.get("has_translation", False)),
    ]

    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")

    # Проблемы
    if check_result.get("issues"):
        print("\n⚠️  Проблемы:")
        for issue in check_result["issues"]:
            print(f"  - {issue}")
    else:
        print("\n✅ Проблем не обнаружено")

    print(f"\n{'='*80}")


def manual_review_batch(cards_batch: list[dict], batch_num: int, total_batches: int):
    """Интерактивная проверка батча карточек.

    Args:
        cards_batch: Список карточек для проверки
        batch_num: Номер батча (начиная с 1)
        total_batches: Всего батчей
    """
    print(f"\n{'='*80}")
    print(f"БАТЧ {batch_num}/{total_batches} - {len(cards_batch)} карточек")
    print(f"{'='*80}")

    # Загружаем sentences_lookup (хотя он не нужен, если card_full уже есть)
    tokens_df = pd.read_parquet(TOKENS_PATH)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))

    # Проверяем каждую карточку
    batch_results = []
    for idx, card_data in enumerate(cards_batch):
        card_index = (batch_num - 1) * BATCH_SIZE + idx
        check_result = check_card_quality(card_data, card_index)
        batch_results.append(check_result)

        display_card_details(card_data, check_result, sentences_lookup)

        # Дополнительная ручная проверка
        print("\n🔍 Дополнительные вопросы для проверки:")
        print("  1. Соответствуют ли примеры из книги synset_group?")
        print("  2. Нет ли спойлеров в примерах?")
        print("  3. Качественные ли примеры (ясные, естественные)?")
        print("  4. Правильный ли перевод?")
        print("  5. Правильное ли определение?")

        # Пауза между карточками
        if idx < len(cards_batch) - 1:
            input("\nНажмите Enter для следующей карточки...")

    # Итоги батча
    print(f"\n{'='*80}")
    print(f"ИТОГИ БАТЧА {batch_num}/{total_batches}")
    print(f"{'='*80}")

    total_in_batch = len(cards_batch)
    cards_with_issues = sum(1 for r in batch_results if r.get("has_issues", False))
    cards_ok = total_in_batch - cards_with_issues

    print(f"\nВсего карточек в батче: {total_in_batch}")
    print(f"✅ Без проблем: {cards_ok}")
    print(f"⚠️  С проблемами: {cards_with_issues}")

    if cards_with_issues > 0:
        print("\nКарточки с проблемами:")
        for result in batch_results:
            if result.get("has_issues", False):
                print(f"  - {result['lemma']} ({result['pos']}): {', '.join(result['issues'])}")

    return batch_results


def load_card_full_data(
    card_data: dict, cards_df: pd.DataFrame, sentences_lookup: dict, provider, cache
):
    """Загружает полные данные карточки для отображения.

    Восстанавливает примеры из aggregated_cards и sentences_lookup.
    Если card_full уже есть в данных, использует его.
    """
    lemma = card_data.get("lemma", "")
    pos = card_data.get("pos", "")

    # Если card_full уже есть, используем его
    if "card_full" in card_data and card_data["card_full"]:
        return card_data["card_full"]

    # Иначе загружаем из aggregated_cards
    row = (
        cards_df[cards_df["lemma"] == lemma].iloc[0] if lemma in cards_df["lemma"].values else None
    )
    if row is None:
        return {}

    # Получаем synset_group
    synset_group = row.get("synset_group", [])
    if isinstance(synset_group, str):
        try:
            synset_group = json.loads(synset_group)
        except:
            synset_group = [synset_group] if synset_group else []

    primary_synset = row.get("primary_synset", "")
    wn_definition = row.get("definition", "")

    # Получаем примеры
    sentence_ids = row.get("sentence_ids", [])
    if isinstance(sentence_ids, str):
        sentence_ids = json.loads(sentence_ids)

    examples = [
        (sid, sentences_lookup.get(sid, "")) for sid in sentence_ids if sid in sentences_lookup
    ]

    # Валидация synset_group
    if examples:
        from eng_words.validation import validate_examples_for_synset_group

        validation_result = validate_examples_for_synset_group(
            lemma=lemma,
            synset_group=synset_group,
            primary_synset=primary_synset,
            examples=examples,
            provider=provider,
            cache=cache,
        )

        valid_examples = [
            sentences_lookup[sid]
            for sid in validation_result["valid_sentence_ids"]
            if sid in sentences_lookup
        ]
    else:
        valid_examples = []

    # Генерируем карточку для получения полных данных
    # Но это может быть долго, поэтому пока вернем то, что есть
    return {
        "synset_group": synset_group,
        "primary_synset": primary_synset,
        "wn_definition": wn_definition,
        "valid_examples": valid_examples[:5],  # Первые 5 для показа
    }


def main():
    """Основная функция для ручной проверки."""
    print("=" * 80)
    print("РУЧНАЯ ПРОВЕРКА КАЧЕСТВА КАРТОЧЕК: ЭТАП 2.5")
    print("=" * 80)

    # Загружаем данные
    logger.info("Загрузка данных...")
    cards_df = pd.read_parquet(AGGREGATED_CARDS_PATH)
    tokens_df = pd.read_parquet(TOKENS_PATH)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))

    # Загружаем результаты тестирования (предпочитаем файл с полными данными)
    if FULL_CARDS_PATH.exists():
        logger.info(f"Загружаем карточки с полными данными из {FULL_CARDS_PATH}")
        results_df = pd.read_json(FULL_CARDS_PATH)
    elif TEST_RESULTS_PATH.exists():
        logger.info(f"Загружаем статистику из {TEST_RESULTS_PATH}")
        logger.warning("В файле нет полных данных карточек. Для полной проверки запустите:")
        logger.warning("  uv run python scripts/regenerate_cards_for_check.py")
        results_df = pd.read_json(TEST_RESULTS_PATH)
    else:
        logger.error(f"Файлы результатов не найдены: {TEST_RESULTS_PATH} или {FULL_CARDS_PATH}")
        sys.exit(1)

    logger.info(f"Загружено {len(results_df)} успешно сгенерированных карточек")

    # Конвертируем в список словарей
    cards_list = results_df.to_dict("records")

    # Инициализируем провайдер для загрузки полных данных (если нужно)
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)

    # Добавляем полные данные для карточек, где их нет
    logger.info("Загрузка полных данных карточек...")
    for card_data in tqdm(cards_list, desc="Загрузка данных"):
        if "card_full" not in card_data or not card_data.get("card_full"):
            card_full = load_card_full_data(card_data, cards_df, sentences_lookup, provider, cache)
            card_data["card_full"] = card_full

    # Разбиваем на батчи
    total_batches = (len(cards_list) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nВсего карточек для проверки: {len(cards_list)}")
    print(f"Батчей: {total_batches} (по {BATCH_SIZE} карточек)")
    print("\nБудете проверять карточки батчами. После каждого батча будет показана статистика.")

    input("\nНажмите Enter для начала проверки...")

    all_results = []

    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(cards_list))
        batch = cards_list[start_idx:end_idx]

        batch_results = manual_review_batch(batch, batch_num, total_batches)
        all_results.extend(batch_results)

        if batch_num < total_batches:
            print(f"\nБатч {batch_num} завершен. Переходим к следующему батчу...")
            input("Нажмите Enter для следующего батча...")

    # Финальная статистика
    print(f"\n{'='*80}")
    print("ФИНАЛЬНАЯ СТАТИСТИКА ПРОВЕРКИ")
    print(f"{'='*80}")

    total_cards = len(all_results)
    cards_ok = sum(1 for r in all_results if not r.get("has_issues", False))
    cards_with_issues = total_cards - cards_ok

    print(f"\nВсего проверено карточек: {total_cards}")
    print(f"✅ Без проблем: {cards_ok} ({cards_ok/total_cards*100:.1f}%)")
    print(f"⚠️  С проблемами: {cards_with_issues} ({cards_with_issues/total_cards*100:.1f}%)")

    # Детальная статистика по проблемам
    if cards_with_issues > 0:
        issue_counts = {}
        for result in all_results:
            for issue in result.get("issues", []):
                issue_type = issue.split(":")[0]  # Берем тип проблемы
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        print("\nРаспределение проблем:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {issue_type}: {count}")

        print("\nКарточки с проблемами:")
        for result in all_results:
            if result.get("has_issues", False):
                print(f"  - {result['lemma']} ({result['pos']}) - {', '.join(result['issues'])}")

    # Сохраняем результаты проверки
    check_results_file = OUTPUT_DIR / "manual_check_results.json"
    with open(check_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты проверки сохранены в: {check_results_file}")
    print(f"\n{'='*80}")
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
