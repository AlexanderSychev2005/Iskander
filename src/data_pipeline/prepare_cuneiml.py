import os
import json
import csv
from tqdm import tqdm

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_FILE = os.path.join(base_dir, "data", "raw", "cuneiml", "CuneiMLv1.2.json")
    CDLI_CAT_CSV = os.path.join(base_dir, "data", "raw", "cdli_data", "cdli_cat.csv")
    OUTPUT_FILE = os.path.join(base_dir, "data", "interim", "cuneiml.jsonl")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    print("Loading CDLI catalog...")
    cdli_dict = {}
    try:
        with open(CDLI_CAT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Indexing CDLI"):
                id_text = str(row.get('id_text', '')).strip()
                if id_text:
                    cdli_dict[id_text] = row
    except Exception as e:
        print(f"Error loading {CDLI_CAT_CSV}: {e}")
        return
        
        
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} tablets. Extracting signs and metadata...")
    
    total_lines = 0
    enriched_lines = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for tablet in tqdm(data, desc="Processing tablets"):
            t_id = tablet.get("id")
            if not t_id:
                continue
                
            t_id_str = str(t_id).strip()
            metadata = cdli_dict.get(t_id_str, {})
            # Normalize to the "P######" CDLI convention so overlap with ORACC
            # (which carries real P-numbers) can be detected downstream.
            tablet_id = "P" + t_id_str.zfill(6) if t_id_str.isdigit() else t_id_str
            
            if metadata:
                enriched_lines += 1
                
            period = metadata.get("period", "unknown")
            genre = metadata.get("genre", "unknown")
            provenience = metadata.get("provenience", "unknown")
            language = metadata.get("language", "unknown")
            material = metadata.get("material", "unknown")
            object_type = metadata.get("object_type", "unknown")
            
            # Clean up empty strings
            period = period if period else "unknown"
            genre = genre if genre else "unknown"
            provenience = provenience if provenience else "unknown"
            language = language if language else "unknown"
            material = material if material else "unknown"
            object_type = object_type if object_type else "unknown"

            if 'text' in tablet and isinstance(tablet['text'], dict):
                for face in ['obverse', 'reverse', 'left', 'right', 'top', 'bottom']:
                    if face in tablet['text'] and isinstance(tablet['text'][face], list):
                        for line_obj in tablet['text'][face]:
                            if not isinstance(line_obj, dict):
                                continue
                            signs = line_obj.get("sign", [])
                            raw_text = line_obj.get("raw", "")
                            
                            if signs and len(signs) > 1:
                                out_obj = {
                                    "raw": raw_text,
                                    "signs": signs,
                                    "tablet_id": tablet_id,
                                    "period": period,
                                    "genre": genre,
                                    "provenience": provenience,
                                    "language": language,
                                    "material": material,
                                    "object_type": object_type
                                }
                                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                                total_lines += 1
                                
    print(f"Done! Extracted {total_lines} lines to {OUTPUT_FILE}")
    print(f"Successfully enriched {enriched_lines} out of {len(data)} tablets ({enriched_lines/len(data)*100:.1f}%).")

if __name__ == "__main__":
    main()
