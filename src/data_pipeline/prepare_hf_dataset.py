import json
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.training.tokenizer import CharacterTokenizer

# --- LABEL ENGINEERING V2.0 MAPPINGS ---

def map_language(l):
    if not l: return 'Unknown'
    l = l.lower()
    if l in ['akkadian', 'middle assyrian', 'old assyrian', 'standard babylonian'] or 'akkadian (with' in l: return 'Akkadian'
    if l == 'sumerian' or l == 'sumerian ?': return 'Sumerian'
    if 'bilingual' in l or ('sumerian' in l and 'akkadian' in l): return 'Bilingual'
    if l in ['urartian', 'hittite', 'eblaite', 'elamite', 'old persian'] or 'urartian/assyrian' in l: return 'Peripheral/Other'
    return 'Unknown'

def map_period(p):
    if not p: return 'Unknown'
    p = p.lower()
    if p in ['neo-assyrian', 'neo-assyrian (ca. 911-612 bc)', 'neo assyrian']: return 'Neo-Assyrian'
    if p in ['ur iii (ca. 2100-2000 bc)', 'ur iii']: return 'Ur III'
    if p in ['old babylonian', 'old babylonian (ca. 1900-1600 bc)', 'early old babylonian (ca. 2000-1900 bc)']: return 'Old Babylonian'
    if 'middle assyrian' in p: return 'Middle Assyrian'
    if 'middle babylonian' in p: return 'Middle Babylonian'
    if p in ['ed iiib (ca. 2500-2340 bc)', 'ed iiib', 'old akkadian (ca. 2340-2200 bc)', 'old akkadian', 'ed iiia', 'lagaš ii', 'ebla']: return 'Third Millennium'
    if p in ['seleucid', 'achaemenid', 'hellenistic']: return 'Late Antiquity'
    return 'Unknown'

def map_genre(g):
    if not g: return 'Unknown'
    g = g.lower()
    if g in ['administrative', 'administrative letter', 'administrative record', 'administrative ?']: return 'Administrative'
    if 'lexical' in g: return 'Lexical'
    if g in ['royal inscription', 'royal/monumental', 'royal stone inscription']: return 'Royal Inscriptions'
    if g in ['literary', 'literary work', 'scholarly letter', 'astrological report', 'omen', 'school']: return 'Literary & Scholarly'
    if g in ['legal', 'legal transaction']: return 'Legal'
    if g == 'letter': return 'Letters'
    return 'Unknown'

def map_provenience(p):
    if not p: return 'Unknown'
    p = p.lower()
    if p in ['kuyunjik (nineveh)', 'nineveh', 'nineveh (mod. kuyunjik)']: return 'Nineveh'
    if p in ['umma (mod. tell jokha)', 'umma']: return 'Umma'
    if p in ['girsu (mod. tello)', 'girsu']: return 'Girsu'
    if p in ['nippur', 'nippur (mod. nuffar)']: return 'Nippur'
    if p in ['puzriš-dagan (mod. drehem)', 'puzriš-dagan']: return 'Puzriš-Dagan'
    if p in ['kanesh (mod. kültepe)', 'kanesh']: return 'Kanesh'
    if p in ['aššur (mod. qalʿat sherqat)', 'assur', 'qalat sherqat (assur)']: return 'Assur'
    return 'Unknown'

LANGUAGE_LABELS = ['Akkadian', 'Sumerian', 'Bilingual', 'Peripheral/Other']
PERIOD_LABELS = ['Neo-Assyrian', 'Ur III', 'Old Babylonian', 'Middle Assyrian', 'Middle Babylonian', 'Third Millennium', 'Late Antiquity']
GENRE_LABELS = ['Administrative', 'Lexical', 'Royal Inscriptions', 'Literary & Scholarly', 'Legal', 'Letters']
PROVENIENCE_LABELS = ['Nineveh', 'Umma', 'Girsu', 'Nippur', 'Puzriš-Dagan', 'Kanesh', 'Assur']

def label_to_idx(label_str, label_list):
    if not label_str or label_str == 'Unknown':
        return -100
    try:
        return label_list.index(label_str)
    except ValueError:
        return -100

def load_and_deduplicate_v2(files):
    print("Loading and deduplicating datasets (v2)...")
    unique_lines = {}
    
    for file_path in files:
        print(f"Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data = json.loads(line)
                    signs = data.get('signs', [])
                    if not signs: continue
                    sign_str = "".join(signs).strip()
                    if not sign_str: continue
                    
                    # Merge metadata
                    if sign_str in unique_lines:
                        existing = unique_lines[sign_str]
                        existing['provenience'] = data.get('provenience', existing.get('provenience', 'unknown'))
                        existing['language'] = data.get('language', existing.get('language', 'unknown'))
                        if existing.get('period', 'unknown').lower() == 'unknown':
                            existing['period'] = data.get('period', 'unknown')
                        if existing.get('genre', 'unknown').lower() == 'unknown':
                            existing['genre'] = data.get('genre', 'unknown')
                    else:
                        unique_lines[sign_str] = data
                except Exception:
                    pass
                    
    print(f"Total unique lines across datasets: {len(unique_lines)}")
    return list(unique_lines.values())

def process_records(records, tokenizer, max_length=128):
    processed = []
    for data in tqdm(records, desc="Tokenizing"):
        signs = data.get('signs', [])
        signs_text = "".join(signs) if isinstance(signs, list) else signs
        input_ids = tokenizer.encode(signs_text, add_special_tokens=True, max_length=max_length)
        
        if len(input_ids) > 0:
            p_val = map_period(data.get('period'))
            g_val = map_genre(data.get('genre'))
            l_val = map_language(data.get('language'))
            prov_val = map_provenience(data.get('provenience'))
            
            processed.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "period_labels": label_to_idx(p_val, PERIOD_LABELS),
                "genre_labels": label_to_idx(g_val, GENRE_LABELS),
                "language_labels": label_to_idx(l_val, LANGUAGE_LABELS),
                "provenience_labels": label_to_idx(prov_val, PROVENIENCE_LABELS)
            })
    return processed

def main():
    base_dir = Path(r"C:\Programming\akkadian\data")
    prepared_dir = base_dir / "prepared"
    os.makedirs(prepared_dir, exist_ok=True)
    
    files_to_merge = [
        prepared_dir / "oracc.jsonl",
        base_dir / "cleaned" / "cuneiml.jsonl"
    ]
    
    # 1. Deduplicate & Merge
    all_unique_records = load_and_deduplicate_v2(files_to_merge)
    
    # 2. Build Tokenizer Vocab
    combined_path = prepared_dir / "combined_unique_v2.jsonl"
    with open(combined_path, 'w', encoding='utf-8') as f:
        for r in all_unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(str(combined_path), min_freq=2)
    vocab_path = prepared_dir / "vocab.json"
    tokenizer.save(str(vocab_path))
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    # 3. Random Split (90/5/5)
    random.seed(42)
    random.shuffle(all_unique_records)
    n = len(all_unique_records)
    test_end = int(n * 0.05)
    val_end = int(n * 0.10)
    
    test_raw = all_unique_records[:test_end]
    val_raw = all_unique_records[test_end:val_end]
    train_raw = all_unique_records[val_end:]
    
    print(f"Split sizes: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
    
    # 4. Save Test split un-tokenized
    test_path = prepared_dir / "test.jsonl"
    print(f"Saving un-tokenized test set to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for r in test_raw:
            # save mapped fields for easy eval later
            r['period_mapped'] = map_period(r.get('period'))
            r['genre_mapped'] = map_genre(r.get('genre'))
            r['language_mapped'] = map_language(r.get('language'))
            r['provenience_mapped'] = map_provenience(r.get('provenience'))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 5. Process Train and Val
    print("Processing Train records...")
    train_processed = process_records(train_raw, tokenizer, max_length=128)
    print("Processing Val records...")
    val_processed = process_records(val_raw, tokenizer, max_length=128)
    
    # Define features
    features = Features({
        'input_ids': Sequence(Value('int32')),
        'attention_mask': Sequence(Value('int8')),
        'period_labels': Value('int64'),
        'genre_labels': Value('int64'),
        'language_labels': Value('int64'),
        'provenience_labels': Value('int64'),
    })
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_processed, features=features),
        "validation": Dataset.from_list(val_processed, features=features)
    })
    
    hf_dir = prepared_dir / "hf_dataset"
    print(f"Saving Arrow dataset to {hf_dir}...")
    dataset_dict.save_to_disk(str(hf_dir))
    
    # Save label dictionaries for model config
    label_dicts = {
        'period': {
            'labels': PERIOD_LABELS,
            'id2label': {i: l for i, l in enumerate(PERIOD_LABELS)},
            'label2id': {l: i for i, l in enumerate(PERIOD_LABELS)}
        },
        'genre': {
            'labels': GENRE_LABELS,
            'id2label': {i: l for i, l in enumerate(GENRE_LABELS)},
            'label2id': {l: i for i, l in enumerate(GENRE_LABELS)}
        },
        'language': {
            'labels': LANGUAGE_LABELS,
            'id2label': {i: l for i, l in enumerate(LANGUAGE_LABELS)},
            'label2id': {l: i for i, l in enumerate(LANGUAGE_LABELS)}
        },
        'provenience': {
            'labels': PROVENIENCE_LABELS,
            'id2label': {i: l for i, l in enumerate(PROVENIENCE_LABELS)},
            'label2id': {l: i for i, l in enumerate(PROVENIENCE_LABELS)}
        }
    }
    with open(prepared_dir / "label_configs.json", "w", encoding='utf-8') as f:
        json.dump(label_dicts, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
