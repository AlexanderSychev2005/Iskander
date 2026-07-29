import os
import json
import random

def split_dataset():
    input_file = "../../data/oracc_dataset/oracc_processed.jsonl"
    train_file = "../../data/oracc_dataset/train.jsonl"
    val_file = "../../data/oracc_dataset/val.jsonl"
    test_file = "../../data/oracc_dataset/test.jsonl"
    
    # Check if run from project root or data_pipeline dir
    if not os.path.exists(input_file):
        input_file = "data/oracc_dataset/oracc_processed.jsonl"
        train_file = "data/oracc_dataset/train.jsonl"
        val_file = "data/oracc_dataset/val.jsonl"
        test_file = "data/oracc_dataset/test.jsonl"
        
    print(f"Reading from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} lines.")
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(lines)
    
    # 90% train, 5% val, 5% test
    train_idx = int(len(lines) * 0.90)
    val_idx = train_idx + int(len(lines) * 0.05)
    
    train_lines = lines[:train_idx]
    val_lines = lines[train_idx:val_idx]
    test_lines = lines[val_idx:]
    
    print(f"Writing {len(train_lines)} to train.jsonl...")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
        
    print(f"Writing {len(val_lines)} to val.jsonl...")
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
        
    print(f"Writing {len(test_lines)} to test.jsonl...")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
        
    print("Done!")

if __name__ == "__main__":
    split_dataset()
