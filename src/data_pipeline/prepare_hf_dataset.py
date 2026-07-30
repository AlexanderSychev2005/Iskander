import json
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.tokenizer import CharacterTokenizer

# --- LABEL VOCABULARIES ---
PERIOD_LABELS = [
    'Neo-Assyrian', 'Middle Assyrian', 'Old Babylonian', 'Old Assyrian', 
    'Neo/Late Babylonian', 'Third Millennium', 'Late Antiquity', 'Other'
]

GENRE_LABELS = [
    'Royal Inscription', 'Lexical & Scholarly', 'Administrative', 
    'Legal', 'Literary & Ritual', 'Letter', 'Other'
]

def label_to_idx(label_str, label_list):
    if not label_str or label_str.lower() == 'unknown':
        return -100
    try:
        return label_list.index(label_str)
    except ValueError:
        return -100

def load_and_deduplicate(files):
    print("Loading and deduplicating datasets...")
    unique_lines = {}
    
    for file_path in files:
        print(f"Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data = json.loads(line)
                    signs = data.get('signs', [])
                    if not signs:
                        continue
                        
                    sign_str = "".join(signs).strip()
                    if not sign_str:
                        continue
                        
                    # Keep the first occurrence. Oracc comes first, so we keep Oracc metadata!
                    if sign_str not in unique_lines:
                        unique_lines[sign_str] = data
                except Exception as e:
                    pass
                    
    print(f"Total unique lines across datasets: {len(unique_lines)}")
    return list(unique_lines.values())

def process_records(records, tokenizer, max_length=128):
    processed = []
    for data in tqdm(records, desc="Tokenizing"):
        signs = data.get('signs', [])
        
        # If it's a list of characters, we can join them or encode directly. 
        # Tokenizer encode handles strings.
        signs_text = "".join(signs) if isinstance(signs, list) else signs
        
        input_ids = tokenizer.encode(signs_text, add_special_tokens=True, max_length=max_length)
        
        if len(input_ids) > 0:
            # Note: prepare_oracc.py produced 'period' and 'genre'. We use those directly.
            processed.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "period_labels": label_to_idx(data.get('period'), PERIOD_LABELS),
                "genre_labels": label_to_idx(data.get('genre'), GENRE_LABELS)
            })
    return processed

def main():
    base_dir = Path(r"C:\Programming\akkadian\data")
    prepared_dir = base_dir / "prepared"
    os.makedirs(prepared_dir, exist_ok=True)
    
    files_to_merge = [
        prepared_dir / "oracc.jsonl",
        prepared_dir / "cuneiml.jsonl"
    ]
    
    # 1. Deduplicate
    all_unique_records = load_and_deduplicate(files_to_merge)
    
    # 2. Save combined to disk for vocab building
    combined_path = prepared_dir / "combined_unique.jsonl"
    print(f"Saving combined dataset to {combined_path}...")
    with open(combined_path, 'w', encoding='utf-8') as f:
        for r in all_unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 3. Build Tokenizer Vocab
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(str(combined_path), min_freq=2) # drop signs appearing only once
    vocab_path = prepared_dir / "vocab.json"
    tokenizer.save(str(vocab_path))
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    # 4. Stratify into 4 buckets
    print("Stratifying records by metadata presence...")
    buckets = {
        'both': [],
        'period_only': [],
        'genre_only': [],
        'neither': []
    }
    
    for r in all_unique_records:
        has_period = r.get('period', 'Unknown').lower() != 'unknown'
        has_genre = r.get('genre', 'Unknown').lower() != 'unknown'
        
        if has_period and has_genre:
            buckets['both'].append(r)
        elif has_period:
            buckets['period_only'].append(r)
        elif has_genre:
            buckets['genre_only'].append(r)
        else:
            buckets['neither'].append(r)
            
    print(f"Bucket sizes: Both={len(buckets['both'])}, PeriodOnly={len(buckets['period_only'])}, GenreOnly={len(buckets['genre_only'])}, Neither={len(buckets['neither'])}")
    
    # 5. Split each bucket (90/5/5)
    random.seed(42)
    train_raw, val_raw, test_raw = [], [], []
    
    for b_name, b_data in buckets.items():
        random.shuffle(b_data)
        n = len(b_data)
        test_end = int(n * 0.05)
        val_end = int(n * 0.10)
        
        test_raw.extend(b_data[:test_end])
        val_raw.extend(b_data[test_end:val_end])
        train_raw.extend(b_data[val_end:])
        
    random.shuffle(train_raw)
    random.shuffle(val_raw)
    random.shuffle(test_raw)
    print(f"Raw Split sizes: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
    
    # 6. Save Test split un-tokenized
    test_path = out_dir / "test.jsonl"
    print(f"Saving un-tokenized test set to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for r in test_raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 7. Process Train and Val
    print("Processing Train records...")
    train_processed = process_records(train_raw, tokenizer, max_length=128)
    print("Processing Val records...")
    val_processed = process_records(val_raw, tokenizer, max_length=128)
    
    # 8. Create HuggingFace Dataset
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_processed),
        "validation": Dataset.from_list(val_processed)
    })
    
    hf_dir = out_dir / "hf_dataset"
    print(f"Saving Arrow dataset to {hf_dir}...")
    dataset_dict.save_to_disk(str(hf_dir))
    print("Done!")

if __name__ == "__main__":
    main()
