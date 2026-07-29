import os
import json
import glob
from tqdm import tqdm

def load_cuneiml_signs(dataset_dir):
    """Loads all sign sequences from CuneiML datasets into a set for deduplication."""
    existing_signs = set()
    for split in ['train', 'val', 'test_a', 'test_b']:
        filepath = os.path.join(dataset_dir, f"{split}.jsonl")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'signs' in data:
                            # Join array of signs into a single string
                            sign_str = "".join(data['signs'])
                            if sign_str:
                                existing_signs.add(sign_str)
                    except Exception:
                        pass
    return existing_signs

catalogue_cache = {}

def parse_oracc_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    textid = data.get('textid', '')
    
    # Try to load metadata from catalogue.json
    metadata = {}
    catalogue_path = os.path.join(os.path.dirname(os.path.dirname(filepath)), 'catalogue.json')
    if textid and os.path.exists(catalogue_path):
        if catalogue_path not in catalogue_cache:
            try:
                with open(catalogue_path, 'r', encoding='utf-8') as cat_f:
                    catalogue_cache[catalogue_path] = json.load(cat_f)
            except Exception:
                catalogue_cache[catalogue_path] = {}
                
        catalogue = catalogue_cache.get(catalogue_path, {})
        members = catalogue.get('members', {})
        if textid in members:
            member_data = members[textid]
            
            # Core metadata
            metadata['period'] = member_data.get('period', 'unknown')
            metadata['genre'] = member_data.get('genre', 'unknown')
            metadata['provenience'] = member_data.get('provenience', 'unknown')
            
            # Expanded metadata
            metadata['language'] = member_data.get('language', 'unknown')
            metadata['dialect'] = member_data.get('dialect', 'unknown')
            metadata['material'] = member_data.get('material', 'unknown')
            metadata['object_type'] = member_data.get('object_type', 'unknown')
            metadata['script'] = member_data.get('script', 'unknown')
            metadata['ruler'] = member_data.get('ruler', 'unknown')
            
    # Rest of the function...
            
    lines = []
    current_line_raw = []
    current_line_signs = []
    current_line_num = ""
    
    def traverse(node):
        nonlocal current_line_raw, current_line_signs, current_line_num
        
        # Line start
        if node.get('node') == 'd' and node.get('type') == 'line-start':
            # Save previous line if exists
            if current_line_raw or current_line_signs:
                line_obj = {
                    "raw": " ".join(current_line_raw),
                    "signs": current_line_signs.copy()
                }
                if metadata:
                    line_obj.update(metadata)
                lines.append(line_obj)
            current_line_raw.clear()
            current_line_signs.clear()
            current_line_num = node.get('n', '')
            
        # Word node
        elif node.get('node') == 'l':
            frag = node.get('frag', '')
            if frag:
                current_line_raw.append(frag)
                
            f_dict = node.get('f', {})
            gdl = f_dict.get('gdl', [])
            
            def extract_utf8(gdl_list):
                for g in gdl_list:
                    if 'utf8' in g:
                        current_line_signs.append(g['utf8'])
                    elif 'seq' in g:
                        extract_utf8(g['seq'])
                    elif 'group' in g:
                        extract_utf8(g['group'])
            
            extract_utf8(gdl)
            
        # Recursive traverse
        if 'cdl' in node:
            for child in node['cdl']:
                traverse(child)
                
    traverse(data)
    # Save the last line
    if current_line_raw or current_line_signs:
        line_obj = {
            "raw": " ".join(current_line_raw),
            "signs": current_line_signs.copy()
        }
        if metadata:
            line_obj.update(metadata)
        lines.append(line_obj)
        
    return lines

def main():
    CUNEIML_DIR = "../../data/prepared_datasets"
    ORACC_UNZIPPED_DIR = "../../data/oracc_raw/unzipped"
    OUTPUT_FILE = "../../data/oracc_dataset/oracc_unique.jsonl"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("Initializing empty set for deduplication (Oracc-internal only)...")
    existing_signs = set()
    
    print("Finding Oracc JSON files...")
    json_files = glob.glob(os.path.join(ORACC_UNZIPPED_DIR, "**", "corpusjson", "*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files in Oracc.")
    
    total_lines = 0
    unique_lines = 0
    skipped_empty = 0
    skipped_dupes = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for filepath in tqdm(json_files, desc="Parsing Oracc JSONs"):
            try:
                parsed_lines = parse_oracc_json(filepath)
                
                for line in parsed_lines:
                    total_lines += 1
                    
                    sign_str = "".join(line['signs'])
                    
                    # 1. Skip empty lines
                    if not sign_str.strip():
                        skipped_empty += 1
                        continue
                        
                    # 2. Skip duplicates
                    if sign_str in existing_signs:
                        skipped_dupes += 1
                        continue
                        
                    # Write unique line
                    out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    unique_lines += 1
                    
                    # Add to set so we don't duplicate within Oracc itself!
                    existing_signs.add(sign_str)
                    
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
                
    print("\n--- Parsing Complete ---")
    print(f"Total lines parsed: {total_lines}")
    print(f"Skipped (Empty): {skipped_empty}")
    print(f"Skipped (Duplicates): {skipped_dupes}")
    print(f"Saved Unique Lines: {unique_lines}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Ensure script runs with correct paths if run from root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
