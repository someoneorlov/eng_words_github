#!/usr/bin/env python3
"""
Ручная проверка качества валидации.

Показывает примеры валидных/невалидных для ручной проверки.
Использует реальные результаты валидации из кэша или повторно валидирует.
"""

import json
import sys
from pathlib import Path

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
    import numpy as np
    import pandas as pd
    
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


def manual_check_samples():
    """Проверяет выборку примеров для ручной оценки."""
    
    # Загрузка данных
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
    
    # Инициализация LLM (используем кэш)
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_validation_cache"), enabled=True)
    
    print("=" * 80)
    print("РУЧНАЯ ПРОВЕРКА КАЧЕСТВА ВАЛИДАЦИИ")
    print("=" * 80)
    print("\nПроверяем выборку из 10 карточек (5 с валидными, 5 без)")
    print("Для каждой карточки показываем валидные и невалидные примеры\n")
    
    # Выбираем карточки
    valid_cards = results_df[results_df["has_valid"] == True].head(5)
    invalid_cards = results_df[results_df["has_valid"] == False].head(5)
    
    all_samples = []
    
    # Обрабатываем карточки с валидными примерами
    print("\n" + "=" * 80)
    print("КАРТОЧКИ С ВАЛИДНЫМИ ПРИМЕРАМИ")
    print("=" * 80)
    
    for idx, row in valid_cards.iterrows():
        lemma = row["lemma"]
        pos = row["pos"]
        primary_synset = row["primary_synset"]
        synset_group = normalize_synset_group(row["synset_group"])
        
        # Находим соответствующую карточку в aggregated_cards
        card_row = cards_df[
            (cards_df["lemma"] == lemma) & 
            (cards_df["primary_synset"] == primary_synset)
        ]
        
        if card_row.empty:
            print(f"\n⚠️  Не найдена карточка: {lemma} ({primary_synset})")
            continue
        
        card_row = card_row.iloc[0]
        sentence_ids = card_row.get("sentence_ids", [])
        
        # Получаем примеры
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids[:15]  # Берем первые 15
            if sid in sentences_lookup and sentences_lookup.get(sid, "")
        ]
        
        if not examples:
            continue
        
        # Валидируем (используем кэш)
        validation = validate_examples_for_synset_group(
            lemma=lemma,
            synset_group=synset_group,
            primary_synset=primary_synset,
            examples=examples,
            provider=provider,
            cache=cache,
        )
        
        print(f"\n{'─' * 80}")
        print(f"Карточка: {lemma} ({pos})")
        print(f"Synset group: {synset_group}")
        print(f"Валидных: {len(validation['valid_sentence_ids'])}, "
              f"Невалидных: {len(validation['invalid_sentence_ids'])}")
        
        # Показываем валидные примеры
        print(f"\n  ✅ ВАЛИДНЫЕ примеры ({len(validation['valid_sentence_ids'])}):")
        for i, (sid, sentence) in enumerate(examples, 1):
            if sid in validation["valid_sentence_ids"]:
                print(f"    {i}. [{sid}] {sentence[:150]}...")
                all_samples.append({
                    "lemma": lemma,
                    "synset": primary_synset,
                    "sentence_id": sid,
                    "sentence": sentence,
                    "status": "valid",
                    "llm_said": "valid"
                })
        
        # Показываем невалидные примеры
        print(f"\n  ❌ НЕВАЛИДНЫЕ примеры ({len(validation['invalid_sentence_ids'])}):")
        for i, (sid, sentence) in enumerate(examples, 1):
            if sid in validation["invalid_sentence_ids"]:
                print(f"    {i}. [{sid}] {sentence[:150]}...")
                all_samples.append({
                    "lemma": lemma,
                    "synset": primary_synset,
                    "sentence_id": sid,
                    "sentence": sentence,
                    "status": "invalid",
                    "llm_said": "invalid"
                })
    
    # Обрабатываем карточки без валидных примеров
    print("\n" + "=" * 80)
    print("КАРТОЧКИ БЕЗ ВАЛИДНЫХ ПРИМЕРОВ")
    print("=" * 80)
    
    for idx, row in invalid_cards.iterrows():
        lemma = row["lemma"]
        pos = row["pos"]
        primary_synset = row["primary_synset"]
        synset_group = normalize_synset_group(row["synset_group"])
        
        # Находим соответствующую карточку
        card_row = cards_df[
            (cards_df["lemma"] == lemma) & 
            (cards_df["primary_synset"] == primary_synset)
        ]
        
        if card_row.empty:
            print(f"\n⚠️  Не найдена карточка: {lemma} ({primary_synset})")
            continue
        
        card_row = card_row.iloc[0]
        sentence_ids = card_row.get("sentence_ids", [])
        
        # Получаем примеры
        examples = [
            (sid, sentences_lookup.get(sid, ""))
            for sid in sentence_ids[:15]
            if sid in sentences_lookup and sentences_lookup.get(sid, "")
        ]
        
        if not examples:
            continue
        
        # Валидируем
        validation = validate_examples_for_synset_group(
            lemma=lemma,
            synset_group=synset_group,
            primary_synset=primary_synset,
            examples=examples,
            provider=provider,
            cache=cache,
        )
        
        print(f"\n{'─' * 80}")
        print(f"Карточка: {lemma} ({pos})")
        print(f"Synset group: {synset_group}")
        print(f"Валидных: {len(validation['valid_sentence_ids'])}, "
              f"Невалидных: {len(validation['invalid_sentence_ids'])}")
        
        # Показываем все примеры (все невалидные)
        print(f"\n  ❌ ВСЕ ПРИМЕРЫ НЕВАЛИДНЫЕ ({len(examples)}):")
        for i, (sid, sentence) in enumerate(examples[:10], 1):  # Показываем первые 10
            print(f"    {i}. [{sid}] {sentence[:150]}...")
            all_samples.append({
                "lemma": lemma,
                "synset": primary_synset,
                "sentence_id": sid,
                "sentence": sentence,
                "status": "invalid",
                "llm_said": "invalid"
            })
    
    # Сохраняем выборку для ручной проверки
    samples_df = pd.DataFrame(all_samples)
    output_path = Path("data/quality_analysis/manual_check_samples.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"✅ Выборка сохранена: {output_path}")
    print(f"   Всего примеров: {len(samples_df)}")
    print(f"   Валидных: {len(samples_df[samples_df['status'] == 'valid'])}")
    print(f"   Невалидных: {len(samples_df[samples_df['status'] == 'invalid'])}")
    print(f"\n   Для ручной проверки откройте CSV и добавьте колонку 'manual_check'")
    print(f"   (valid/invalid) для сравнения с результатами LLM")
    print("=" * 80)


if __name__ == "__main__":
    manual_check_samples()
