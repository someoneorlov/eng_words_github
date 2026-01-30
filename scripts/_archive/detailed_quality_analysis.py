#!/usr/bin/env python3
"""
Детальная автоматизированная проверка качества карточек.

Проводит критический анализ каждой карточки:
- Наличие всех компонентов
- Валидность примеров
- Согласованность компонентов
- Качество определения
- Качество перевода
- Качество примеров

Usage:
    uv run python scripts/detailed_quality_analysis.py --n 200
"""

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nltk.corpus import wordnet as wn

from eng_words.llm.smart_card_generator import SmartCard
from eng_words.validation.example_validator import validate_card_examples

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CardAnalysis:
    """Результат анализа одной карточки."""
    card_index: int
    lemma: str
    pos: str
    synset_id: str
    
    # Наличие компонентов
    has_examples: bool
    has_definition: bool
    has_translation: bool
    has_all_components: bool
    
    # Валидность примеров
    examples_valid: bool
    examples_count: int
    invalid_examples: list[str]
    found_forms: list[str]
    
    # Качество компонентов (1-5)
    examples_quality_score: int  # Релевантность, разнообразие
    definition_quality_score: int  # Полнота, ясность
    translation_quality_score: int  # Точность, естественность
    overall_quality_score: int  # Общее качество
    
    # Проблемы
    issues: list[str]
    critical_issues: list[str]  # Критические проблемы
    
    # Данные карточки
    examples: list[str]
    definition: str
    translation: str
    
    # Анализ
    definition_length: int
    translation_length: int
    examples_diversity: float  # Уникальность примеров (0-1)
    
    # Согласованность
    examples_match_definition: bool
    translation_matches_definition: bool
    all_components_aligned: bool


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
    if len(cards) < n:
        logger.warning(f"Only {len(cards)} cards available, using all")
        return cards
    sampled = random.sample(cards, n)
    logger.info(f"Sampled {len(sampled)} cards for analysis")
    return sampled


def check_component_presence(card: dict) -> dict[str, bool]:
    """Проверить наличие всех компонентов."""
    has_examples = bool(card.get("selected_examples"))
    has_definition = bool(card.get("simple_definition") or card.get("wn_definition"))
    has_translation = bool(card.get("translation_ru"))
    has_all = has_examples and has_definition and has_translation
    
    return {
        "has_examples": has_examples,
        "has_definition": has_definition,
        "has_translation": has_translation,
        "has_all_components": has_all,
    }


def validate_examples(card: dict) -> dict[str, Any]:
    """Валидировать примеры."""
    try:
        smart_card = SmartCard(
            lemma=card.get("lemma", ""),
            pos=card.get("pos", ""),
            supersense=card.get("supersense", ""),
            selected_examples=card.get("selected_examples", []),
            excluded_examples=card.get("excluded_examples", []),
            simple_definition=card.get("simple_definition", ""),
            translation_ru=card.get("translation_ru", ""),
            generated_example=card.get("generated_example", ""),
            wn_definition=card.get("wn_definition", ""),
            book_name=card.get("book_name", "american_tragedy"),
            primary_synset=card.get("primary_synset", ""),
            synset_group=card.get("synset_group", [card.get("primary_synset", "")]),
        )
        validation = validate_card_examples(smart_card)
        
        return {
            "examples_valid": validation.is_valid,
            "examples_count": len(card.get("selected_examples", [])),
            "invalid_examples": validation.invalid_examples,
            "found_forms": validation.found_forms,
        }
    except Exception as e:
        logger.warning(f"Failed to validate examples for {card.get('lemma')}: {e}")
        return {
            "examples_valid": False,
            "examples_count": len(card.get("selected_examples", [])),
            "invalid_examples": card.get("selected_examples", []),
            "found_forms": [],
        }


def analyze_examples_quality(examples: list[str], lemma: str) -> int:
    """Оценить качество примеров (1-5)."""
    if not examples:
        return 1
    
    # Критерии:
    # 5: 5+ примеров, разнообразные, lemma присутствует во всех
    # 4: 3-4 примера, разнообразные, lemma присутствует
    # 3: 2-3 примера, lemma присутствует
    # 2: 1-2 примера или проблемы
    # 1: Нет примеров или невалидные
    
    lemma_lower = lemma.lower()
    lemma_in_all = all(lemma_lower in ex.lower() for ex in examples)
    
    # Проверка разнообразия (примеры не слишком похожи)
    unique_starts = len(set(ex[:30].lower() for ex in examples))
    diversity = unique_starts / len(examples) if examples else 0
    
    if len(examples) >= 5 and lemma_in_all and diversity > 0.8:
        return 5
    elif len(examples) >= 3 and lemma_in_all and diversity > 0.7:
        return 4
    elif len(examples) >= 2 and lemma_in_all:
        return 3
    elif len(examples) >= 1 and lemma_in_all:
        return 2
    else:
        return 1


def analyze_definition_quality(definition: str, wn_definition: str = "") -> int:
    """Оценить качество определения (1-5)."""
    if not definition:
        return 1
    
    # Критерии:
    # 5: Полное, ясное определение (20+ символов), отличается от WordNet
    # 4: Хорошее определение (15-20 символов)
    # 3: Приемлемое определение (10-15 символов)
    # 2: Короткое или слишком простое (5-10 символов)
    # 1: Очень короткое или отсутствует
    
    length = len(definition.strip())
    
    # Проверка на полноту (не просто одно слово)
    words = definition.split()
    is_simple = len(words) <= 3 or length < 10
    
    if length >= 20 and not is_simple:
        return 5
    elif length >= 15 and not is_simple:
        return 4
    elif length >= 10:
        return 3
    elif length >= 5:
        return 2
    else:
        return 1


def analyze_translation_quality(translation: str, definition: str) -> int:
    """Оценить качество перевода (1-5)."""
    if not translation:
        return 1
    
    # Критерии:
    # 5: Полный, естественный перевод (10+ символов)
    # 4: Хороший перевод (8-10 символов)
    # 3: Приемлемый перевод (5-8 символов)
    # 2: Короткий перевод (3-5 символов)
    # 1: Очень короткий или отсутствует
    
    length = len(translation.strip())
    
    # Проверка на наличие русских букв
    has_russian = bool(re.search(r'[а-яё]', translation, re.IGNORECASE))
    
    if not has_russian:
        return 1
    
    if length >= 10:
        return 5
    elif length >= 8:
        return 4
    elif length >= 5:
        return 3
    elif length >= 3:
        return 2
    else:
        return 1


def check_alignment(card: dict) -> dict[str, bool]:
    """Проверить согласованность компонентов."""
    examples = card.get("selected_examples", [])
    definition = card.get("simple_definition", "") or card.get("wn_definition", "")
    translation = card.get("translation_ru", "")
    lemma = card.get("lemma", "").lower()
    
    # Примеры соответствуют определению (эвристика: lemma присутствует в примерах)
    examples_match = bool(examples) and all(lemma in ex.lower() for ex in examples)
    
    # Перевод соответствует определению (эвристика: перевод не пустой и на русском)
    translation_matches = bool(translation) and bool(re.search(r'[а-яё]', translation, re.IGNORECASE))
    
    # Все компоненты согласованы
    all_aligned = examples_match and translation_matches and bool(definition)
    
    return {
        "examples_match_definition": examples_match,
        "translation_matches_definition": translation_matches,
        "all_components_aligned": all_aligned,
    }


def calculate_examples_diversity(examples: list[str]) -> float:
    """Вычислить разнообразие примеров (0-1)."""
    if not examples or len(examples) == 1:
        return 0.0
    
    # Сравниваем начальные части примеров
    starts = [ex[:50].lower().strip() for ex in examples]
    unique_starts = len(set(starts))
    
    return unique_starts / len(examples)


def identify_issues(card: dict, analysis: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Выявить проблемы карточки."""
    issues = []
    critical = []
    
    # Критические проблемы
    if not analysis["has_examples"]:
        critical.append("Нет примеров")
    if not analysis["has_definition"]:
        critical.append("Нет определения")
    if not analysis["has_translation"]:
        critical.append("Нет перевода")
    if not analysis["examples_valid"]:
        critical.append(f"Невалидные примеры: {analysis['invalid_examples']}")
    
    # Проблемы качества
    if analysis["examples_count"] == 0:
        issues.append("Нет примеров")
    elif analysis["examples_count"] == 1:
        issues.append("Только один пример")
    elif analysis["examples_count"] < 3:
        issues.append(f"Мало примеров ({analysis['examples_count']})")
    
    if analysis["examples_quality_score"] <= 2:
        issues.append("Низкое качество примеров")
    
    if analysis["definition_quality_score"] <= 2:
        issues.append("Низкое качество определения")
    
    if analysis["translation_quality_score"] <= 2:
        issues.append("Низкое качество перевода")
    
    if not analysis["examples_match_definition"]:
        issues.append("Примеры не соответствуют определению")
    
    if not analysis["translation_matches_definition"]:
        issues.append("Перевод не соответствует определению")
    
    if analysis["examples_diversity"] < 0.5:
        issues.append("Низкое разнообразие примеров")
    
    # Проверка на слишком короткие компоненты
    if analysis["definition_length"] < 10:
        issues.append("Очень короткое определение")
    
    if analysis["translation_length"] < 5:
        issues.append("Очень короткий перевод")
    
    return issues, critical


def analyze_card(card: dict, index: int) -> CardAnalysis:
    """Детально проанализировать одну карточку."""
    logger.debug(f"Analyzing card {index + 1}: {card.get('lemma')}")
    
    # Наличие компонентов
    components = check_component_presence(card)
    
    # Валидация примеров
    validation = validate_examples(card)
    
    # Качество компонентов
    examples = card.get("selected_examples", [])
    definition = card.get("simple_definition", "") or card.get("wn_definition", "")
    translation = card.get("translation_ru", "")
    lemma = card.get("lemma", "")
    
    examples_quality = analyze_examples_quality(examples, lemma)
    definition_quality = analyze_definition_quality(definition, card.get("wn_definition", ""))
    translation_quality = analyze_translation_quality(translation, definition)
    
    # Общее качество (среднее с учетом критических проблем)
    base_overall = (examples_quality + definition_quality + translation_quality) / 3
    if not components["has_all_components"] or not validation["examples_valid"]:
        overall_quality = max(1, base_overall - 2)  # Штраф за критические проблемы
    else:
        overall_quality = base_overall
    
    # Согласованность
    alignment = check_alignment(card)
    
    # Разнообразие примеров
    diversity = calculate_examples_diversity(examples)
    
    # Проблемы
    analysis_data = {
        "has_examples": components["has_examples"],
        "has_definition": components["has_definition"],
        "has_translation": components["has_translation"],
        "examples_valid": validation["examples_valid"],
        "examples_count": validation["examples_count"],
        "invalid_examples": validation["invalid_examples"],
        "examples_quality_score": examples_quality,
        "definition_quality_score": definition_quality,
        "translation_quality_score": translation_quality,
        "examples_match_definition": alignment["examples_match_definition"],
        "translation_matches_definition": alignment["translation_matches_definition"],
        "examples_diversity": diversity,
        "definition_length": len(definition),
        "translation_length": len(translation),
    }
    
    issues, critical = identify_issues(card, analysis_data)
    
    return CardAnalysis(
        card_index=index + 1,
        lemma=lemma,
        pos=card.get("pos", ""),
        synset_id=card.get("primary_synset", ""),
        has_examples=components["has_examples"],
        has_definition=components["has_definition"],
        has_translation=components["has_translation"],
        has_all_components=components["has_all_components"],
        examples_valid=validation["examples_valid"],
        examples_count=validation["examples_count"],
        invalid_examples=validation["invalid_examples"],
        found_forms=validation["found_forms"],
        examples_quality_score=examples_quality,
        definition_quality_score=definition_quality,
        translation_quality_score=translation_quality,
        overall_quality_score=round(overall_quality),
        issues=issues,
        critical_issues=critical,
        examples=examples,
        definition=definition,
        translation=translation,
        definition_length=len(definition),
        translation_length=len(translation),
        examples_diversity=diversity,
        examples_match_definition=alignment["examples_match_definition"],
        translation_matches_definition=alignment["translation_matches_definition"],
        all_components_aligned=alignment["all_components_aligned"],
    )


def generate_report(analyses: list[CardAnalysis]) -> dict[str, Any]:
    """Сгенерировать отчет об анализе."""
    total = len(analyses)
    
    # Средние оценки
    avg_examples = sum(a.examples_quality_score for a in analyses) / total if total > 0 else 0
    avg_definition = sum(a.definition_quality_score for a in analyses) / total if total > 0 else 0
    avg_translation = sum(a.translation_quality_score for a in analyses) / total if total > 0 else 0
    avg_overall = sum(a.overall_quality_score for a in analyses) / total if total > 0 else 0
    
    # Распределение оценок
    def score_dist(score_func):
        return {i: sum(1 for a in analyses if score_func(a) == i) for i in range(1, 6)}
    
    # Наличие компонентов
    has_all = sum(1 for a in analyses if a.has_all_components)
    has_examples = sum(1 for a in analyses if a.has_examples)
    has_definition = sum(1 for a in analyses if a.has_definition)
    has_translation = sum(1 for a in analyses if a.has_translation)
    
    # Валидность
    valid_examples = sum(1 for a in analyses if a.examples_valid)
    aligned = sum(1 for a in analyses if a.all_components_aligned)
    
    # Проблемы
    problematic = sum(1 for a in analyses if a.overall_quality_score <= 2)
    critical = sum(1 for a in analyses if a.critical_issues)
    
    # Частые проблемы
    all_issues = []
    for a in analyses:
        all_issues.extend(a.issues)
    
    issue_freq = {}
    for issue in all_issues:
        issue_freq[issue] = issue_freq.get(issue, 0) + 1
    
    # Статистика по примерам
    avg_examples_count = sum(a.examples_count for a in analyses) / total if total > 0 else 0
    cards_with_many_examples = sum(1 for a in analyses if a.examples_count >= 5)
    cards_with_few_examples = sum(1 for a in analyses if a.examples_count < 3)
    
    return {
        "summary": {
            "total_analyzed": total,
            "average_scores": {
                "examples": round(avg_examples, 2),
                "definition": round(avg_definition, 2),
                "translation": round(avg_translation, 2),
                "overall": round(avg_overall, 2),
            },
            "score_distribution": {
                "examples": score_dist(lambda a: a.examples_quality_score),
                "definition": score_dist(lambda a: a.definition_quality_score),
                "translation": score_dist(lambda a: a.translation_quality_score),
                "overall": score_dist(lambda a: a.overall_quality_score),
            },
            "components_presence": {
                "all_components": has_all,
                "all_components_pct": has_all / total * 100 if total > 0 else 0,
                "has_examples": has_examples,
                "has_examples_pct": has_examples / total * 100 if total > 0 else 0,
                "has_definition": has_definition,
                "has_definition_pct": has_definition / total * 100 if total > 0 else 0,
                "has_translation": has_translation,
                "has_translation_pct": has_translation / total * 100 if total > 0 else 0,
            },
            "quality_metrics": {
                "valid_examples": valid_examples,
                "valid_examples_pct": valid_examples / total * 100 if total > 0 else 0,
                "aligned_components": aligned,
                "aligned_components_pct": aligned / total * 100 if total > 0 else 0,
                "problematic_cards": problematic,
                "problematic_cards_pct": problematic / total * 100 if total > 0 else 0,
                "critical_issues": critical,
                "critical_issues_pct": critical / total * 100 if total > 0 else 0,
            },
            "examples_statistics": {
                "average_count": round(avg_examples_count, 2),
                "cards_with_many": cards_with_many_examples,
                "cards_with_many_pct": cards_with_many_examples / total * 100 if total > 0 else 0,
                "cards_with_few": cards_with_few_examples,
                "cards_with_few_pct": cards_with_few_examples / total * 100 if total > 0 else 0,
            },
            "issue_frequency": dict(sorted(issue_freq.items(), key=lambda x: x[1], reverse=True)),
        },
        "analyses": [asdict(a) for a in analyses],
    }


def print_detailed_card(card: dict, analysis: CardAnalysis, index: int, total: int):
    """Вывести детальную информацию о карточке."""
    print("\n" + "=" * 100)
    print(f"КАРТОЧКА {index + 1} / {total}")
    print("=" * 100)
    print()
    
    print(f"📝 ЛЕММА: {analysis.lemma} ({analysis.pos})")
    print(f"🔖 SYNSET: {analysis.synset_id}")
    print()
    
    # Статус компонентов
    print("📦 КОМПОНЕНТЫ:")
    print(f"  Примеры:     {'✅' if analysis.has_examples else '❌'} ({analysis.examples_count})")
    print(f"  Определение: {'✅' if analysis.has_definition else '❌'} ({analysis.definition_length} символов)")
    print(f"  Перевод:     {'✅' if analysis.has_translation else '❌'} ({analysis.translation_length} символов)")
    print()
    
    # Оценки
    print("⭐ ОЦЕНКИ КАЧЕСТВА (1-5):")
    print(f"  Примеры:     {analysis.examples_quality_score}/5")
    print(f"  Определение: {analysis.definition_quality_score}/5")
    print(f"  Перевод:     {analysis.translation_quality_score}/5")
    print(f"  ОБЩЕЕ:       {analysis.overall_quality_score}/5")
    print()
    
    # Примеры
    if analysis.examples:
        print(f"📚 ПРИМЕРЫ ({analysis.examples_count}):")
        for i, ex in enumerate(analysis.examples[:5], 1):  # Показываем первые 5
            print(f"  {i}. {ex}")
        if len(analysis.examples) > 5:
            print(f"  ... и еще {len(analysis.examples) - 5}")
        print(f"  Разнообразие: {analysis.examples_diversity:.2f}")
        if analysis.found_forms:
            print(f"  Найденные формы: {', '.join(analysis.found_forms)}")
        if analysis.invalid_examples:
            print(f"  ❌ Невалидные: {analysis.invalid_examples}")
    else:
        print("📚 ПРИМЕРЫ: ❌ НЕТ")
    print()
    
    # Определение
    print(f"📖 ОПРЕДЕЛЕНИЕ:")
    if analysis.definition:
        print(f"  {analysis.definition}")
    else:
        print("  ❌ НЕТ")
    print()
    
    # Перевод
    print(f"🌐 ПЕРЕВОД:")
    if analysis.translation:
        print(f"  {analysis.translation}")
    else:
        print("  ❌ НЕТ")
    print()
    
    # Согласованность
    print("🔗 СОГЛАСОВАННОСТЬ:")
    print(f"  Примеры ↔ Определение: {'✅' if analysis.examples_match_definition else '❌'}")
    print(f"  Перевод ↔ Определение: {'✅' if analysis.translation_matches_definition else '❌'}")
    print(f"  Все согласованы:        {'✅' if analysis.all_components_aligned else '❌'}")
    print()
    
    # Проблемы
    if analysis.critical_issues:
        print("🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
        for issue in analysis.critical_issues:
            print(f"  ❌ {issue}")
        print()
    
    if analysis.issues:
        print("⚠️  ПРОБЛЕМЫ:")
        for issue in analysis.issues:
            print(f"  • {issue}")
        print()
    
    if not analysis.critical_issues and not analysis.issues:
        print("✅ Проблем не найдено")
        print()


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Детальная автоматизированная проверка качества карточек")
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
        help="Количество карточек для анализа",
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
        default=Path("data/comparison/detailed_quality_analysis.json"),
        help="Путь для сохранения результатов",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Выводить детальную информацию о каждой карточке",
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("ДЕТАЛЬНАЯ АВТОМАТИЗИРОВАННАЯ ПРОВЕРКА КАЧЕСТВА КАРТОЧЕК")
    print("=" * 100)
    print()
    
    # Загрузка карточек
    cards = load_cards(args.input)
    
    if len(cards) < args.n:
        logger.warning(f"Only {len(cards)} cards available, analyzing all")
        args.n = len(cards)
    
    # Выборка
    sampled = sample_cards(cards, args.n, args.seed)
    
    # Анализ
    logger.info("Starting detailed analysis...")
    analyses = []
    
    for i, card in enumerate(sampled):
        analysis = analyze_card(card, i)
        analyses.append(analysis)
        
        if args.verbose:
            print_detailed_card(card, analysis, i, len(sampled))
        elif (i + 1) % 20 == 0:
            logger.info(f"Analyzed {i + 1}/{len(sampled)} cards...")
    
    # Генерация отчета
    report = generate_report(analyses)
    
    # Сохранение
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Вывод сводки
    summary = report["summary"]
    print("\n" + "=" * 100)
    print("СВОДКА АНАЛИЗА")
    print("=" * 100)
    print()
    print(f"📊 Проанализировано: {summary['total_analyzed']} карточек")
    print()
    
    print("⭐ СРЕДНИЕ ОЦЕНКИ КАЧЕСТВА (1-5):")
    print(f"  Примеры:       {summary['average_scores']['examples']:.2f}/5")
    print(f"  Определение:   {summary['average_scores']['definition']:.2f}/5")
    print(f"  Перевод:       {summary['average_scores']['translation']:.2f}/5")
    print(f"  ОБЩЕЕ:         {summary['average_scores']['overall']:.2f}/5")
    print()
    
    print("📦 НАЛИЧИЕ КОМПОНЕНТОВ:")
    print(f"  Все компоненты:  {summary['components_presence']['all_components']} ({summary['components_presence']['all_components_pct']:.1f}%)")
    print(f"  С примерами:     {summary['components_presence']['has_examples']} ({summary['components_presence']['has_examples_pct']:.1f}%)")
    print(f"  С определением:  {summary['components_presence']['has_definition']} ({summary['components_presence']['has_definition_pct']:.1f}%)")
    print(f"  С переводом:     {summary['components_presence']['has_translation']} ({summary['components_presence']['has_translation_pct']:.1f}%)")
    print()
    
    print("✅ КАЧЕСТВО:")
    print(f"  Валидные примеры:        {summary['quality_metrics']['valid_examples']} ({summary['quality_metrics']['valid_examples_pct']:.1f}%)")
    print(f"  Согласованные:           {summary['quality_metrics']['aligned_components']} ({summary['quality_metrics']['aligned_components_pct']:.1f}%)")
    print(f"  Проблемные (≤2):         {summary['quality_metrics']['problematic_cards']} ({summary['quality_metrics']['problematic_cards_pct']:.1f}%)")
    print(f"  С критическими ошибками: {summary['quality_metrics']['critical_issues']} ({summary['quality_metrics']['critical_issues_pct']:.1f}%)")
    print()
    
    print("📚 ПРИМЕРЫ:")
    print(f"  Среднее количество:  {summary['examples_statistics']['average_count']:.1f}")
    print(f"  Много примеров (≥5): {summary['examples_statistics']['cards_with_many']} ({summary['examples_statistics']['cards_with_many_pct']:.1f}%)")
    print(f"  Мало примеров (<3):  {summary['examples_statistics']['cards_with_few']} ({summary['examples_statistics']['cards_with_few_pct']:.1f}%)")
    print()
    
    if summary['issue_frequency']:
        print("⚠️  ЧАСТЫЕ ПРОБЛЕМЫ (топ-10):")
        for issue, count in list(summary['issue_frequency'].items())[:10]:
            print(f"  • {issue}: {count} раз(а)")
        print()
    
    # Показываем топ-5 проблемных карточек
    problematic = sorted(analyses, key=lambda a: a.overall_quality_score)[:5]
    if problematic:
        print("🚨 ТОП-5 ПРОБЛЕМНЫХ КАРТОЧЕК:")
        for a in problematic:
            print(f"  {a.lemma} ({a.pos}): {a.overall_quality_score}/5 - {', '.join(a.critical_issues[:2])}")
        print()
    
    print(f"💾 Результаты сохранены в: {args.output}")
    print()
    print("=" * 100)
    print("✅ Анализ завершен!")
    print("=" * 100)


if __name__ == "__main__":
    main()

