#!/usr/bin/env python3
"""
Критическая ручная проверка качества карточек.

Проверяет каждую карточку отдельно на:
- Соответствие lemma и примеров
- Качество определения
- Качество перевода
- Смысловую согласованность всех компонентов

Usage:
    uv run python scripts/manual_quality_review.py --n 200 --output data/comparison/manual_review.json
"""

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eng_words.llm.smart_card_generator import SmartCard
from eng_words.validation.example_validator import validate_card_examples

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class QualityReview:
    """Результат проверки одной карточки."""
    card_id: int
    lemma: str
    pos: str
    synset_id: str
    
    # Оценки (1-5)
    examples_quality: int  # Соответствие примеров lemma/synset
    definition_quality: int  # Точность и ясность определения
    translation_quality: int  # Точность перевода
    overall_quality: int  # Общее качество карточки
    
    # Проблемы
    issues: list[str]  # Список найденных проблем
    comments: str  # Комментарии проверяющего
    
    # Данные карточки (для контекста)
    examples: list[str]
    definition: str
    translation: str


def load_cards(path: Path) -> list[dict]:
    """Загрузить карточки из JSON."""
    logger.info(f"Loading cards from {path}")
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)
    logger.info(f"  Loaded {len(cards)} cards")
    return cards


def sample_cards(cards: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Выбрать случайную выборку карточек."""
    random.seed(seed)
    sampled = random.sample(cards, min(n, len(cards)))
    logger.info(f"Sampled {len(sampled)} cards for review")
    return sampled


def print_card_for_review(card: dict, index: int, total: int) -> None:
    """Вывести карточку для ручной проверки."""
    print("\n" + "=" * 80)
    print(f"КАРТОЧКА {index + 1} / {total}")
    print("=" * 80)
    print()
    
    lemma = card.get("lemma", "")
    pos = card.get("pos", "")
    synset_id = card.get("primary_synset", "")
    
    print(f"📝 ЛЕММА: {lemma} ({pos})")
    print(f"🔖 SYNSET: {synset_id}")
    print()
    
    # Примеры
    examples = card.get("selected_examples", [])
    print(f"📚 ПРИМЕРЫ ({len(examples)}):")
    if examples:
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex}")
    else:
        print("  ❌ НЕТ ПРИМЕРОВ!")
    print()
    
    # Определение
    definition = card.get("simple_definition", "")
    wn_definition = card.get("wn_definition", "")
    print(f"📖 ОПРЕДЕЛЕНИЕ:")
    if definition:
        print(f"  {definition}")
    else:
        print("  ❌ НЕТ ОПРЕДЕЛЕНИЯ!")
    if wn_definition and wn_definition != definition:
        print(f"  (WordNet: {wn_definition})")
    print()
    
    # Перевод
    translation = card.get("translation_ru", "")
    print(f"🌐 ПЕРЕВОД:")
    if translation:
        print(f"  {translation}")
    else:
        print("  ❌ НЕТ ПЕРЕВОДА!")
    print()
    
    # Валидация
    try:
        smart_card = SmartCard(
            lemma=lemma,
            pos=pos,
            supersense=card.get("supersense", ""),
            selected_examples=examples,
            excluded_examples=card.get("excluded_examples", []),
            simple_definition=definition,
            translation_ru=translation,
            generated_example=card.get("generated_example", ""),
            wn_definition=wn_definition,
            book_name=card.get("book_name", "american_tragedy"),
            primary_synset=synset_id,
            synset_group=card.get("synset_group", [synset_id]),
        )
        validation = validate_card_examples(smart_card)
        
        print(f"✅ ВАЛИДАЦИЯ:")
        print(f"  Валидность: {'✅ ДА' if validation.is_valid else '❌ НЕТ'}")
        if validation.found_forms:
            print(f"  Найденные формы: {', '.join(validation.found_forms)}")
        if validation.invalid_examples:
            print(f"  ❌ Невалидные примеры: {validation.invalid_examples}")
    except Exception as e:
        print(f"⚠️  Ошибка валидации: {e}")
    print()


def collect_review(card: dict, index: int, total: int) -> QualityReview:
    """Собрать оценку карточки от проверяющего."""
    print_card_for_review(card, index, total)
    
    print("=" * 80)
    print("ОЦЕНКА КАЧЕСТВА (1-5, где 5 = отлично):")
    print("=" * 80)
    print()
    
    # Оценки
    try:
        examples_q = int(input("Примеры (1-5): ").strip() or "3")
        definition_q = int(input("Определение (1-5): ").strip() or "3")
        translation_q = int(input("Перевод (1-5): ").strip() or "3")
        overall_q = int(input("Общее качество (1-5): ").strip() or "3")
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️  Используются значения по умолчанию (3)")
        examples_q = definition_q = translation_q = overall_q = 3
    
    # Проблемы
    print()
    print("ПРОБЛЕМЫ (введите через запятую или Enter для пропуска):")
    issues_input = input().strip()
    issues = [i.strip() for i in issues_input.split(",") if i.strip()] if issues_input else []
    
    # Комментарии
    print()
    print("КОММЕНТАРИИ (Enter для пропуска):")
    comments = input().strip()
    
    return QualityReview(
        card_id=index + 1,
        lemma=card.get("lemma", ""),
        pos=card.get("pos", ""),
        synset_id=card.get("primary_synset", ""),
        examples_quality=examples_q,
        definition_quality=definition_q,
        translation_quality=translation_q,
        overall_quality=overall_q,
        issues=issues,
        comments=comments,
        examples=card.get("selected_examples", []),
        definition=card.get("simple_definition", ""),
        translation=card.get("translation_ru", ""),
    )


def generate_summary(reviews: list[QualityReview]) -> dict[str, Any]:
    """Сгенерировать сводку проверки."""
    total = len(reviews)
    
    # Средние оценки
    avg_examples = sum(r.examples_quality for r in reviews) / total if total > 0 else 0
    avg_definition = sum(r.definition_quality for r in reviews) / total if total > 0 else 0
    avg_translation = sum(r.translation_quality for r in reviews) / total if total > 0 else 0
    avg_overall = sum(r.overall_quality for r in reviews) / total if total > 0 else 0
    
    # Распределение оценок
    examples_dist = {i: sum(1 for r in reviews if r.examples_quality == i) for i in range(1, 6)}
    definition_dist = {i: sum(1 for r in reviews if r.definition_quality == i) for i in range(1, 6)}
    translation_dist = {i: sum(1 for r in reviews if r.translation_quality == i) for i in range(1, 6)}
    overall_dist = {i: sum(1 for r in reviews if r.overall_quality == i) for i in range(1, 6)}
    
    # Проблемные карточки (оценка ≤ 2)
    problematic = [r for r in reviews if r.overall_quality <= 2]
    
    # Все проблемы
    all_issues = []
    for r in reviews:
        all_issues.extend(r.issues)
    
    # Уникальные проблемы
    unique_issues = {}
    for issue in all_issues:
        unique_issues[issue] = unique_issues.get(issue, 0) + 1
    
    return {
        "total_reviewed": total,
        "average_scores": {
            "examples": round(avg_examples, 2),
            "definition": round(avg_definition, 2),
            "translation": round(avg_translation, 2),
            "overall": round(avg_overall, 2),
        },
        "score_distribution": {
            "examples": examples_dist,
            "definition": definition_dist,
            "translation": translation_dist,
            "overall": overall_dist,
        },
        "problematic_cards": len(problematic),
        "problematic_cards_pct": len(problematic) / total * 100 if total > 0 else 0,
        "issue_frequency": dict(sorted(unique_issues.items(), key=lambda x: x[1], reverse=True)),
        "reviews": [asdict(r) for r in reviews],
    }


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Критическая ручная проверка качества карточек")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synset_cards/synset_smart_cards_partial.json"),
        help="Путь к файлу с карточками",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Количество карточек для проверки",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для случайной выборки",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/comparison/manual_review.json"),
        help="Путь для сохранения результатов",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("КРИТИЧЕСКАЯ РУЧНАЯ ПРОВЕРКА КАЧЕСТВА КАРТОЧЕК")
    print("=" * 80)
    print()
    print(f"Загружаю карточки из: {args.input}")
    print(f"Будут проверены: {args.n} случайных карточек")
    print(f"Результаты будут сохранены в: {args.output}")
    print()
    
    # Загрузка карточек
    cards = load_cards(args.input)
    
    if len(cards) < args.n:
        print(f"⚠️  В файле только {len(cards)} карточек, будет проверено {len(cards)}")
        args.n = len(cards)
    
    # Выборка
    sampled = sample_cards(cards, args.n, args.seed)
    
    print("\n" + "=" * 80)
    print("НАЧИНАЕМ ПРОВЕРКУ")
    print("=" * 80)
    print()
    print("Для каждой карточки:")
    print("  1. Оцените качество (1-5, где 5 = отлично)")
    print("  2. Укажите проблемы (если есть)")
    print("  3. Добавьте комментарии (если нужно)")
    print()
    print("Нажмите Enter для продолжения...")
    input()
    
    # Проверка
    reviews = []
    for i, card in enumerate(sampled):
        try:
            review = collect_review(card, i, len(sampled))
            reviews.append(review)
            
            print()
            print(f"✅ Карточка {i + 1}/{len(sampled)} проверена")
            print()
            print("─" * 80)
            print()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Проверка прервана пользователем")
            break
    
    # Генерация сводки
    if reviews:
        summary = generate_summary(reviews)
        
        # Сохранение
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Вывод сводки
        print("\n" + "=" * 80)
        print("СВОДКА ПРОВЕРКИ")
        print("=" * 80)
        print()
        print(f"Проверено карточек: {summary['total_reviewed']}")
        print()
        print("Средние оценки:")
        print(f"  Примеры:       {summary['average_scores']['examples']:.2f}/5")
        print(f"  Определение:   {summary['average_scores']['definition']:.2f}/5")
        print(f"  Перевод:       {summary['average_scores']['translation']:.2f}/5")
        print(f"  Общее:         {summary['average_scores']['overall']:.2f}/5")
        print()
        print(f"Проблемных карточек (≤2): {summary['problematic_cards']} ({summary['problematic_cards_pct']:.1f}%)")
        print()
        
        if summary['issue_frequency']:
            print("Частые проблемы:")
            for issue, count in list(summary['issue_frequency'].items())[:10]:
                print(f"  - {issue}: {count} раз(а)")
        
        print()
        print(f"Результаты сохранены в: {args.output}")
        print()
        print("=" * 80)
    else:
        print("\n❌ Не было проверено ни одной карточки")


if __name__ == "__main__":
    main()

