# -*- coding: utf-8 -*-
"""
Bulaq 1280 AH — Sentence-Level OCR Corruption Engine
=====================================================
Reads the authentic Bulaq text, splits into natural sentences/clauses,
and applies realistic lithographic OCR corruption patterns to produce
full sentence-level pairs: corrupt_ocr → ground_truth.

This fixes the root cause of the failed Colab training:
  - Old dataset: 60% single-word pairs, 32% punctuation-only changes
  - New dataset: Full sentence pairs (20-80 chars), realistic multi-error corruption
"""

import json
import random
import re
import os
import sys

# ============================================================
# 1. OCR CORRUPTION FUNCTIONS (Empirical Bulaq Lithography)
# ============================================================

# Dot/i'jam confusion pairs (characters that differ only by dots)
DOT_CONFUSION = [
    ('ب', 'ت'), ('ب', 'ث'), ('ت', 'ث'), ('ب', 'ن'), ('ت', 'ن'),
    ('ج', 'ح'), ('ج', 'خ'), ('ح', 'خ'),
    ('د', 'ذ'), ('ر', 'ز'),
    ('س', 'ش'), ('ص', 'ض'), ('ط', 'ظ'),
    ('ع', 'غ'), ('ف', 'ق'),
]

# Hamza/alef normalization errors
HAMZA_STRIP = {
    'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ي',
}

# Common lithographic loop substitutions
LITHO_SUBS = {
    'ة': 'ه', 'ى': 'ي', 'ي': 'ى',
}

def corrupt_dot_swap(text, prob=0.08):
    """Swap dotted/undotted letter pairs (i'jam confusion)."""
    chars = list(text)
    for i, c in enumerate(chars):
        if random.random() < prob:
            for a, b in DOT_CONFUSION:
                if c == a:
                    chars[i] = b
                    break
                elif c == b:
                    chars[i] = a
                    break
    return ''.join(chars)

def corrupt_hamza_strip(text, prob=0.35):
    """Strip hamza marks from alef."""
    chars = list(text)
    for i, c in enumerate(chars):
        if c in HAMZA_STRIP and random.random() < prob:
            chars[i] = HAMZA_STRIP[c]
    return ''.join(chars)

def corrupt_space_delete(text, prob=0.12):
    """Delete spaces to simulate fusion collapse."""
    words = text.split()
    if len(words) < 3:
        return text
    result = []
    i = 0
    while i < len(words):
        if i < len(words) - 1 and random.random() < prob:
            result.append(words[i] + words[i+1])
            i += 2
        else:
            result.append(words[i])
            i += 1
    return ' '.join(result)

def corrupt_space_insert(text, prob=0.06):
    """Insert spurious spaces to simulate kerning fracture."""
    words = text.split()
    result = []
    for w in words:
        if len(w) > 4 and random.random() < prob:
            split_pos = random.randint(2, len(w) - 2)
            result.append(w[:split_pos])
            result.append(w[split_pos:])
        else:
            result.append(w)
    return ' '.join(result)

def corrupt_litho_sub(text, prob=0.15):
    """Apply lithographic loop substitutions."""
    chars = list(text)
    for i, c in enumerate(chars):
        if c in LITHO_SUBS and random.random() < prob:
            chars[i] = LITHO_SUBS[c]
    return ''.join(chars)

def corrupt_punct_drop(text, prob=0.5):
    """Drop punctuation (common in raw OCR)."""
    if random.random() < prob:
        text = re.sub(r'[،؛:\.!\?]', '', text)
    return text

def corrupt_sentence(text):
    """Apply a random combination of corruption functions."""
    text = corrupt_hamza_strip(text)
    text = corrupt_punct_drop(text)
    if random.random() < 0.7:
        text = corrupt_dot_swap(text)
    if random.random() < 0.5:
        text = corrupt_space_delete(text)
    if random.random() < 0.3:
        text = corrupt_space_insert(text)
    if random.random() < 0.6:
        text = corrupt_litho_sub(text)
    return text.strip()


# ============================================================
# 2. TEXT EXTRACTION & SENTENCE SPLITTING
# ============================================================

def clean_openiti_line(line):
    """Remove OpenITI markup tags."""
    line = line.strip()
    if line.startswith('#META#') or line.startswith('######'):
        return ''
    line = re.sub(r'^#+\s*\|?\s*', '', line)
    line = re.sub(r'^~~', '', line)
    line = line.strip()
    return line

def split_into_sentences(text, min_len=15, max_len=120):
    """Split text into natural sentence/clause chunks."""
    parts = re.split(r'(?<=[.،؛:!؟])\s+', text)
    sentences = []
    current = ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current:
            candidate = current + ' ' + part
        else:
            candidate = part
        if len(candidate) > max_len:
            if len(current) >= min_len:
                sentences.append(current.strip())
            current = part
        else:
            current = candidate
    if current and len(current) >= min_len:
        sentences.append(current.strip())
    return sentences

def extract_sentences(source_path, min_len=15, max_len=120):
    """Extract clean sentences from the Bulaq OpenITI text."""
    all_sentences = []
    with open(source_path, 'r', encoding='utf-8') as f:
        buffer = ''
        for line in f:
            cleaned = clean_openiti_line(line)
            if not cleaned:
                if buffer:
                    sents = split_into_sentences(buffer, min_len, max_len)
                    all_sentences.extend(sents)
                    buffer = ''
                continue
            buffer += ' ' + cleaned
        if buffer:
            sents = split_into_sentences(buffer, min_len, max_len)
            all_sentences.extend(sents)
    return all_sentences


# ============================================================
# 3. DATASET GENERATION
# ============================================================

def generate_dataset(source_path, output_path, augmentation_factor=3):
    print(f"Reading source text: {source_path}")
    sentences = extract_sentences(source_path)
    print(f"Extracted {len(sentences):,} sentences (15-120 chars)")

    print("\nSample sentences:")
    for s in sentences[:5]:
        print(f"  [{len(s):3d} chars] {s}")

    pairs = []
    skipped = 0

    for sent in sentences:
        for _ in range(augmentation_factor):
            corrupt = corrupt_sentence(sent)
            if corrupt != sent and len(corrupt) >= 10:
                pairs.append({
                    'corrupt_ocr': corrupt,
                    'ground_truth': sent,
                })
            else:
                skipped += 1

    random.seed(42)
    random.shuffle(pairs)

    print(f"\nGenerated {len(pairs):,} sentence-level pairs ({skipped} skipped as identical)")

    src_lens = [len(p['corrupt_ocr']) for p in pairs]
    tgt_lens = [len(p['ground_truth']) for p in pairs]
    print(f"Source length: min={min(src_lens)}, median={sorted(src_lens)[len(src_lens)//2]}, max={max(src_lens)}")
    print(f"Target length: min={min(tgt_lens)}, median={sorted(tgt_lens)[len(tgt_lens)//2]}, max={max(tgt_lens)}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"Written to: {output_path}")

    print("\n=== Sample Pairs ===")
    for p in pairs[:10]:
        print(f"  ❌ {p['corrupt_ocr']}")
        print(f"  ✅ {p['ground_truth']}")
        print()

    return pairs


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_candidates = [
        os.path.join(script_dir, 'alwaraq_bulaq_full.txt'),
        os.path.join(script_dir, '..', 'alf', 'alwaraq_bulaq_full.txt'),
        os.path.join(script_dir, 'data', 'alwaraq_bulaq_full.txt'),
    ]
    source_path = None
    for c in source_candidates:
        if os.path.exists(c):
            source_path = c
            break
    if not source_path:
        print("ERROR: Cannot find alwaraq_bulaq_full.txt")
        sys.exit(1)

    output_path = os.path.join(script_dir, 'data', 'bulaq_sentence_pairs.jsonl')
    generate_dataset(source_path, output_path, augmentation_factor=3)
