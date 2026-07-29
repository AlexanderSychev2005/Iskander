import json
import random
import os

JSON_FILE = "../../data/CuneiMLv1.2.json"
DATA_DIR = "../../data/prepared_datasets"
os.makedirs(DATA_DIR, exist_ok=True)

def contains_lacuna(raw_text):
    text = raw_text.lower()
    return (" x " in text or text.startswith("x ") or text.endswith(" x") or text == "x" or 
            "[" in text or "]" in text or "..." in text or "-x-" in text or "x-" in text or "-x" in text)

def main():
    print("Loading JSON...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    clean_lines = []
    lacuna_lines = []
    
    # Extract lines
    for item in data:
        img_url = item.get("img_url")
        item_id = item.get("id")
        bboxes = item.get("bboxes")
        
        text_data = item.get("text", {})
        for side, lines in text_data.items():
            for line in lines:
                if not isinstance(line, dict):
                    continue
                
                raw = line.get("raw", "")
                signs = line.get("sign", [])
                
                if not raw or not signs:
                    continue
                
                record = {
                    "id": item_id,
                    "img_url": img_url,
                    "bboxes": bboxes,
                    "side": side,
                    "num": line.get("num", ""),
                    "raw": raw,
                    "signs": signs
                }
                
                if contains_lacuna(raw):
                    lacuna_lines.append(record)
                else:
                    clean_lines.append(record)
                    
    print(f"Extracted {len(clean_lines)} clean lines.")
    print(f"Extracted {len(lacuna_lines)} lacuna lines (Test B).")
    
    # Shuffle clean lines deterministically
    random.seed(42)
    random.shuffle(clean_lines)
    
    # Split
    total_clean = len(clean_lines)
    train_size = int(total_clean * 0.9)
    val_size = int(total_clean * 0.05)
    
    train_data = clean_lines[:train_size]
    val_data = clean_lines[train_size:train_size+val_size]
    test_a_data_raw = clean_lines[train_size+val_size:]
    
    # Mask Test A
    MASK_TOKEN = "[MASK]"
    test_a_data = []
    for record in test_a_data_raw:
        signs = record["signs"]
        if len(signs) < 2:
            continue # Hard to mask if only 1 sign
            
        # Mask 1 random sign deterministically per record
        # (Since we seeded random at the start, this sequence of random.randint is also deterministic)
        mask_idx = random.randint(0, len(signs)-1)
        target = signs[mask_idx]
        
        masked_signs = list(signs)
        masked_signs[mask_idx] = MASK_TOKEN
        
        new_record = dict(record)
        new_record["masked_signs"] = masked_signs
        new_record["target_index"] = mask_idx
        new_record["target"] = target
        test_a_data.append(new_record)
        
    # Save datasets
    def save_jsonl(data_to_save, filename):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for d in data_to_save:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Saved {len(data_to_save)} records to {path}")

    save_jsonl(train_data, "train.jsonl")
    save_jsonl(val_data, "val.jsonl")
    save_jsonl(test_a_data, "test_a.jsonl")
    save_jsonl(lacuna_lines, "test_b.jsonl")

if __name__ == "__main__":
    main()
