import os
import json
import re
from tqdm import tqdm

def load_osl_dict(osl_path):
    sign_dict = {}
    with open(osl_path, 'r', encoding='utf-8') as f:
        current_unicode = None
        for line in f:
            line = line.strip()
            if line.startswith('@sign '):
                current_unicode = None
            elif line.startswith('@ucun '):
                parts = line.split()
                if len(parts) > 1:
                    current_unicode = parts[1]
            elif line.startswith('@list') and 'U+' in line:
                match = re.search(r'U\+([0-9A-Fa-f]{4,5})', line)
                if match:
                    try:
                        current_unicode = chr(int(match.group(1), 16))
                    except ValueError:
                        pass
            elif line.startswith('@v '):
                if current_unicode:
                    parts = line.split()
                    if len(parts) > 1:
                        val = parts[1]
                        sign_dict[val] = current_unicode
            elif line.startswith('@uname '):
                if current_unicode:
                    name = line.split('CUNEIFORM SIGN ')[-1]
                    sign_dict[name.upper()] = current_unicode
                    sign_dict[name.lower()] = current_unicode
                    
    # Numeric fallbacks
    for i in range(1, 10):
        sign_dict[f"{i}(disz)"] = chr(0x12079)
        sign_dict[f"{i}(u)"] = chr(0x1230B)
        sign_dict[f"{i}(gesz2)"] = chr(0x120FC)
        sign_dict[f"{i}(asz)"] = chr(0x12038)
        sign_dict[f"{i}(N01)"] = "𒁹"
        sign_dict[f"{i}(N14)"] = "𒌋"
        
    # Common ones
    sign_dict["lugal"] = "𒈗"
    sign_dict["mu"] = "𒈬"
    sign_dict["gal"] = "𒃲"
    sign_dict["an"] = "𒀭"
    
    return sign_dict

def clean_token(token):
    token = re.sub(r'[#\[\]\?!<>]', '', token)
    token = re.sub(r'~[a-z0-9A-Z]*', '', token)
    return token

def transliterate(text, sign_dict):
    tokens = re.split(r'([\s-])', text)
    out = []
    for t in tokens:
        if t in [' ', '-']:
            continue
        ct = clean_token(t)
        if not ct:
            continue
            
        if ct in sign_dict:
            out.append(sign_dict[ct])
        elif ct.upper() in sign_dict:
            out.append(sign_dict[ct.upper()])
        else:
            # Drop unknown signs to keep data clean, or keep them?
            # Oracc and CuneiML drop unknown. We will skip the word if unknown, 
            # but wait, it's better to just skip the line if it has too many unknown signs
            out.append(None)
    return out

def main():
    OSL_FILE = "../../data/raw/osl/00lib/osl.asl"
    CDLI_FILE = "../../data/raw/cdli/cdli_data/cdliatf_unblocked.atf"
    OUTPUT_FILE = "../../data/cleaned/cdli.jsonl"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    if not os.path.exists(CDLI_FILE):
        print(f"Error: {CDLI_FILE} not found.")
        return
        
    print(f"Loading OSL mapping from {OSL_FILE}...")
    sign_dict = load_osl_dict(OSL_FILE)
    print(f"Loaded {len(sign_dict)} mappings.")
    
    print("Processing CDLI ATF...")
    total_lines = 0
    with open(CDLI_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Parsing ATF"):
            line = line.strip()
            if re.match(r"^\d+'?\.", line):
                text_part = line.split('.', 1)[1].strip()
                signs = transliterate(text_part, sign_dict)
                
                # Filter out None and keep valid signs
                valid_signs = [s for s in signs if s is not None]
                
                if len(valid_signs) >= 3 and len(valid_signs) >= len(signs) * 0.7:
                    # Require at least 3 signs and 70% success rate in translating
                    out_obj = {
                        "raw": text_part,
                        "signs": valid_signs,
                        "period": "unknown",
                        "genre": "unknown"
                    }
                    f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    total_lines += 1
                    
    print(f"Done! Extracted {total_lines} lines to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
