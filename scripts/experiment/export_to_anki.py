#!/usr/bin/env python3
"""
Export Pipeline B cards to Anki-compatible CSV format.

Creates a CSV file that can be imported directly into Anki.
"""

import csv
import json
import re
from pathlib import Path

# Paths
INPUT_PATH = Path('data/experiment/cards_B.json')
OUTPUT_DIR = Path('data/experiment')


def normalize_pos(pos: str) -> str:
    """Normalize POS tag to standard form."""
    pos = pos.lower().strip()
    
    # Map variations to standard tags
    if pos.startswith('noun') or pos == 'n':
        return 'noun'
    elif pos.startswith('verb') or pos == 'v':
        return 'verb'
    elif pos.startswith('adj') or pos == 'a':
        return 'adj'
    elif pos.startswith('adv'):
        return 'adv'
    elif 'phrasal' in pos:
        return 'phrasal verb'
    elif 'idiom' in pos or 'phrase' in pos:
        return 'phrase'
    else:
        return pos


def format_examples(examples: list[str], max_examples: int = 3) -> str:
    """Format examples for Anki card."""
    if not examples:
        return ""
    
    formatted = []
    for i, ex in enumerate(examples[:max_examples]):
        # Clean up the example
        ex = ex.strip()
        # Limit length
        if len(ex) > 150:
            ex = ex[:147] + "..."
        formatted.append(f"• {ex}")
    
    return "<br>".join(formatted)


def create_anki_card(card: dict) -> dict:
    """Convert a card to Anki format.
    
    Returns dict with:
    - front: Question side (word + examples)
    - back: Answer side (definition + translation + generated example)
    - tags: Space-separated tags
    """
    lemma = card.get('lemma', '')
    pos = normalize_pos(card.get('part_of_speech', 'unknown'))
    definition_en = card.get('definition_en', '')
    definition_ru = card.get('definition_ru', '')
    examples = card.get('examples', [])
    generated = card.get('generated_example', '')
    
    # Front side: word with examples from book
    examples_html = format_examples(examples)
    front = f"<div class='word'><b>{lemma}</b> <span class='pos'>({pos})</span></div>"
    if examples_html:
        front += f"<div class='examples'>{examples_html}</div>"
    
    # Back side: definition + translation + generated example
    back = f"<div class='definition'>{definition_en}</div>"
    back += f"<div class='translation'>🇷🇺 {definition_ru}</div>"
    if generated:
        back += f"<div class='generated'>💡 {generated}</div>"
    
    # Tags
    tags = f"word_family {pos} american_tragedy"
    
    return {
        'front': front,
        'back': back,
        'tags': tags,
    }


def main():
    # Load cards
    print(f"Loading cards from {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        data = json.load(f)
    
    cards = data['cards']
    print(f"Loaded {len(cards)} cards")
    
    # Convert to Anki format
    anki_cards = []
    for card in cards:
        anki_card = create_anki_card(card)
        anki_cards.append(anki_card)
    
    # Save CSV
    csv_path = OUTPUT_DIR / 'anki_cards_word_family.csv'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['front', 'back', 'tags'], 
                                quoting=csv.QUOTE_ALL)
        # No header for Anki import
        writer.writerows(anki_cards)
    
    print(f"\nSaved {len(anki_cards)} cards to {csv_path}")
    
    # Also save full JSON with normalized data
    json_path = OUTPUT_DIR / 'cards_B_normalized.json'
    
    normalized = []
    for card in cards:
        normalized.append({
            'lemma': card.get('lemma', ''),
            'pos': normalize_pos(card.get('part_of_speech', 'unknown')),
            'definition_en': card.get('definition_en', ''),
            'definition_ru': card.get('definition_ru', ''),
            'examples': card.get('examples', []),
            'generated_example': card.get('generated_example', ''),
            'sentence_ids': card.get('sentence_ids', []),
        })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    
    print(f"Saved normalized JSON to {json_path}")
    
    # Stats
    pos_counts = {}
    for card in normalized:
        pos = card['pos']
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    print(f"\n--- Normalized POS distribution ---")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"  {pos}: {count}")
    
    print(f"\n✅ Ready to import into Anki!")
    print(f"   File: {csv_path}")
    print(f"   Import as: Basic (Front/Back)")
    print(f"   Allow HTML: Yes")


if __name__ == '__main__':
    main()
