import os
import json
from tqdm import tqdm

def main():
    INPUT_FILE = "../../data/raw/cuneiml/CuneiMLv1.2.json"
    OUTPUT_FILE = "../../data/cleaned/cuneiml.jsonl"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please place CuneiML data in data/raw/cuneiml/")
        return
        
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} tablets. Extracting signs...")
    
    total_lines = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for tablet in tqdm(data, desc="Processing tablets"):
            if 'text' in tablet and isinstance(tablet['text'], dict):
                for face in ['obverse', 'reverse', 'left', 'right', 'top', 'bottom']:
                    if face in tablet['text'] and isinstance(tablet['text'][face], list):
                        for line_obj in tablet['text'][face]:
                            if not isinstance(line_obj, dict):
                                continue
                            signs = line_obj.get("sign", [])
                            raw_text = line_obj.get("raw", "")
                            
                            if signs and len(signs) > 1: # Require at least 2 signs
                                out_obj = {
                                    "raw": raw_text,
                                    "signs": signs,
                                    "period": "unknown",
                                    "genre": "unknown"
                                }
                                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                                total_lines += 1
                            
    print(f"Done! Extracted {total_lines} lines to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
