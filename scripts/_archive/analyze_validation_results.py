#!/usr/bin/env python3
"""
Детальный анализ результатов валидации.

Показывает конкретные примеры валидных/невалидных для ручной проверки.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens

load_dotenv()


def analyze_validation_results():
    """Анализирует результаты валидации и показывает примеры."""
    
    # Загрузка результатов
    results_path = Path("data/quality_analysis/validation_100_results.parquet")
    tokens_path = Path("data/processed/american_tragedy_tokens.parquet")
    aggregated_path = Path("data/synset_aggregation_full/aggregated_cards.parquet")
    
    results_df = pd.read_parquet(results_path)
    tokens_df = pd.read_parquet(tokens_path)
    cards_df = pd.read_parquet(aggregated_path)
    
    # Восстановление предложений
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))
    
    # Создаем lookup для sentence_ids из aggregated_cards
    cards_lookup = {}
    for idx, row in cards_df.iterrows():
        lemma = row["lemma"]
        primary_synset = row.get("primary_synset", "")
        key = (lemma, primary_synset)
        cards_lookup[key] = row.get("sentence_ids", [])
    
    print("=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ВАЛИДАЦИИ")
    print("=" * 80)
    
    # Выбираем карточки для детального анализа
    # 1. Карточки с валидными примерами (хорошие)
    valid_cards = results_df[results_df["has_valid"] == True].head(5)
    
    # 2. Карточки без валидных примеров (проблемные)
    invalid_cards = results_df[results_df["has_valid"] == False].head(5)
    
    print("\n" + "=" * 80)
    print("1. КАРТОЧКИ С ВАЛИДНЫМИ ПРИМЕРАМИ (5 примеров)")
    print("=" * 80)
    
    for idx, row in valid_cards.iterrows():
        lemma = row["lemma"]
        pos = row["pos"]
        primary_synset = row["primary_synset"]
        synset_group = row["synset_group"]
        valid_count = row["valid_count"]
        invalid_count = row["invalid_count"]
        total = row["total_examples"]
        
        print(f"\n{'─' * 80}")
        print(f"Карточка: {lemma} ({pos})")
        print(f"Primary synset: {primary_synset}")
        print(f"Synset group: {synset_group}")
        print(f"Статистика: {valid_count} валидных, {invalid_count} невалидных из {total} всего")
        
        # Получаем sentence_ids из aggregated_cards
        key = (lemma, primary_synset)
        sentence_ids = cards_lookup.get(key, [])
        
        if not sentence_ids:
            print("  ⚠️  Не найдены sentence_ids в aggregated_cards")
            continue
        
        # Получаем примеры
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids[:20]  # Берем первые 20 для анализа
            if sid in sentences_lookup
        ]
        
        if not examples:
            print("  ⚠️  Нет доступных примеров")
            continue
        
        print(f"\n  Примеры (первые {min(10, len(examples))} из {len(examples)}):")
        for i, (sid, sentence) in enumerate(examples[:10], 1):
            # Определяем статус (валидный/невалидный) - нужно проверить через валидацию
            # Для простоты показываем все, но помечаем что нужно проверить
            print(f"    {i}. [{sid}] {sentence[:120]}...")
    
    print("\n" + "=" * 80)
    print("2. КАРТОЧКИ БЕЗ ВАЛИДНЫХ ПРИМЕРОВ (5 примеров)")
    print("=" * 80)
    
    for idx, row in invalid_cards.iterrows():
        lemma = row["lemma"]
        pos = row["pos"]
        primary_synset = row["primary_synset"]
        synset_group = row["synset_group"]
        invalid_count = row["invalid_count"]
        total = row["total_examples"]
        
        print(f"\n{'─' * 80}")
        print(f"Карточка: {lemma} ({pos})")
        print(f"Primary synset: {primary_synset}")
        print(f"Synset group: {synset_group}")
        print(f"Статистика: 0 валидных, {invalid_count} невалидных из {total} всего")
        
        # Получаем sentence_ids
        key = (lemma, primary_synset)
        sentence_ids = cards_lookup.get(key, [])
        
        if not sentence_ids:
            print("  ⚠️  Не найдены sentence_ids в aggregated_cards")
            continue
        
        # Получаем примеры
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids[:20]
            if sid in sentences_lookup
        ]
        
        if not examples:
            print("  ⚠️  Нет доступных примеров")
            continue
        
        print(f"\n  Примеры (первые {min(10, len(examples))} из {len(examples)}):")
        for i, (sid, sentence) in enumerate(examples[:10], 1):
            print(f"    {i}. [{sid}] {sentence[:120]}...")
    
    print("\n" + "=" * 80)
    print("✅ Анализ завершен")
    print("=" * 80)


if __name__ == "__main__":
    analyze_validation_results()
