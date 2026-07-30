import json
import os
import sys
from pathlib import Path
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
    if not label_str or label_str == 'Unknown':
        return -100
    try:
        return label_list.index(label_str)
    except ValueError:
        return -100

def process_file(jsonl_path, tokenizer, max_length=128):
    print(f"Processing {jsonl_path}...")
    records = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                signs_text = data.get('signs', '')
                if not signs_text:
                    continue
                
                input_ids = tokenizer.encode(signs_text, add_special_tokens=True, max_length=max_length)
                
                if len(input_ids) > 0:
                    records.append({
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids), # Needed for HF, overridden by collator
                        "period_labels": label_to_idx(data.get('period_mapped'), PERIOD_LABELS),
                        "genre_labels": label_to_idx(data.get('genre_mapped'), GENRE_LABELS)
                    })
            except Exception as e:
                continue
                
    print(f"Processed {len(records)} records.")
    return records

def main():
    base_dir = Path(r"C:\Programming\akkadian\data\prepated_oracc")
    vocab_path = base_dir / "vocab.json"
    train_path = base_dir / "train.jsonl"
    val_path = base_dir / "val.jsonl"
    out_dir = base_dir / "hf_dataset"
    
    tokenizer = CharacterTokenizer()
    tokenizer.load(str(vocab_path))
    
    train_records = process_file(train_path, tokenizer, max_length=128)
    val_records = process_file(val_path, tokenizer, max_length=128)
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records)
    })
    
    print(f"Saving Arrow dataset to {out_dir}...")
    dataset_dict.save_to_disk(str(out_dir))
    print("Done!")

if __name__ == "__main__":
    main()
