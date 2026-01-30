#!/usr/bin/env python3
"""
Тест батчинга на проблемной карточке "able (a)".
"""

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation.synset_validator import validate_examples_for_synset_group

load_dotenv()


def test_able_with_batching():
    """Тестирует валидацию able (a) с батчингом."""
    print("=" * 80)
    print("ТЕСТ БАТЧИНГА НА КАРТОЧКЕ ABLE (A)")
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
    
    # Находим карточку able (a)
    card_row = cards_df[
        (cards_df["lemma"] == "able") & 
        (cards_df["primary_synset"] == "able.a.01")
    ]
    
    if card_row.empty:
        print("❌ Карточка не найдена")
        return
    
    card_row = card_row.iloc[0]
    sentence_ids = card_row.get("sentence_ids", [])
    synset_group = card_row.get("synset_group", [])
    
    # Нормализация synset_group
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
        synset_group = [synset_group]
    
    # Получаем все примеры
    examples = [
        (sid, sentences_lookup.get(sid, ""))
        for sid in sentence_ids
        if sid in sentences_lookup and sentences_lookup.get(sid, "")
    ]
    
    print(f"\nКарточка: able (a)")
    print(f"Synset group: {synset_group}")
    print(f"Всего примеров: {len(examples)}")
    
    # Инициализация LLM
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_validation_cache"), enabled=True)
    
    # Валидация с батчингом
    print(f"\nВалидация с батчингом (max_examples_per_batch=50)...")
    validation = validate_examples_for_synset_group(
        lemma="able",
        synset_group=synset_group,
        primary_synset="able.a.01",
        examples=examples,
        provider=provider,
        cache=cache,
        max_examples_per_batch=50,
    )
    
    print(f"\nРезультаты:")
    print(f"  Валидных: {len(validation['valid_sentence_ids'])}")
    print(f"  Невалидных: {len(validation['invalid_sentence_ids'])}")
    print(f"  Has valid: {validation['has_valid']}")
    
    if validation['has_valid']:
        print(f"\n✅ УСПЕХ! Найдено {len(validation['valid_sentence_ids'])} валидных примеров")
        print(f"\nПервые 10 валидных примеров:")
        for i, sid in enumerate(validation['valid_sentence_ids'][:10], 1):
            sentence = sentences_lookup.get(sid, "")
            print(f"  {i}. [{sid}] {sentence[:100]}...")
    else:
        print(f"\n❌ Все примеры невалидные (возможно, проблема с валидацией)")
    
    print("\n" + "=" * 80)
    print("✅ Тест завершен")
    print("=" * 80)


if __name__ == "__main__":
    test_able_with_batching()
