import json
import os
import re
import sys
import random
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.training.tokenizer import CharacterTokenizer

# One unified dataset serves both training pipelines: 'signs' (pre-segmented
# cuneiform, tokenized with our own CharacterTokenizer in train.py) and
# 'text' (cleaned Latin transliteration, tokenized with mBERT's own
# WordPiece tokenizer in train_mbert.py). Neither side is pre-tokenized here
# -- each script tokenizes at load time with its own tokenizer/vocab, so the
# same Arrow dataset works for both without duplicating the split logic.

_DETERMINATIVE_RE = re.compile(r"\{[^}]*\}")  # e.g. {m}, {d}, {ki} -- editorial determinatives, dropped entirely (Lazar et al. 2021 do the same with sub/superscripts)
_BRACKET_CHARS = "[]⸢⸣()<>|"  # editorial uncertainty/restoration brackets and ATF sign-separator (|) -- stripped, content kept
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_ASCII_INDEX_DIGIT_RE = re.compile(r"([a-zŋ])([0-9]+)(?![0-9a-z])")

def _normalize_cuneiml_romanization(text):
    """CuneiML and ORACC transliterate the same phonemes with two disjoint
    ASCII/Unicode conventions (measured on 200k lines each: ORACC uses
    Unicode š/Unicode subscripts in ~49%/62% of lines and ASCII 'sz'/
    digit-suffix in ~0%/0.3%; CuneiML is the mirror image -- ASCII 'sz'/
    digit-suffix in ~62%/75%, Unicode š in ~0%). Left unmerged, the same
    word surfaces as two unrelated tokens depending on source, which
    fragments mBERT's WordPiece vocabulary for no linguistic reason. Digit-
    suffix conversion must run before the digraph substitutions below, so
    that a digit right after a still-ASCII 'sz'/'s,'/'t,' (e.g. 'gesz2')
    gets subscripted before the digraph collapses ('gesz2' -> 'gesz₂' ->
    'geš₂', not 'geš2'). ASCII 'h' for ḫ is NOT converted: CuneiML
    never marks that distinction (0 instances found), so it can't be
    recovered from the ASCII side.
    """
    text = _ASCII_INDEX_DIGIT_RE.sub(lambda m: m.group(1) + m.group(2).translate(_SUBSCRIPT_DIGITS), text)
    text = text.replace("sz", "š").replace("SZ", "Š").replace("Sz", "Š")
    text = text.replace("s,", "ṣ").replace("S,", "Ṣ")
    text = text.replace("t,", "ṭ").replace("T,", "Ṭ")
    return text

def clean_transliteration(raw):
    if not raw:
        return ""
    text = _normalize_cuneiml_romanization(raw)
    text = _DETERMINATIVE_RE.sub("", text)
    text = text.translate(str.maketrans("", "", _BRACKET_CHARS))
    # Em-dash used as a name/word-joining mark (e.g. 'GAL—MU', 'qur-di—DINGIR-ma') --
    # not in mBERT's vocab (unlike every other Assyriological diacritic we
    # checked), so normalize it to the plain hyphen already used for the
    # same joining role elsewhere in the transliteration.
    text = text.replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()

# --- LABEL ENGINEERING V3.0 MAPPINGS ---
# Substring-based (not exact-match) on purpose: raw CDLI/ORACC values almost
# always carry a "(ca. NNNN-NNNN BC)" or "(mod. Xxx)" suffix, so exact-match
# lists silently dropped ~30-58% of records with a perfectly good value to
# 'Unknown' (verified against the full 353k-row CDLI catalogue and the
# 639k-record merged corpus -- see conversation for the breakdown).

FAKE_GENRE_MARKERS = ('fake (modern)',)

def map_language(l):
    if not l: return 'Unknown'
    l = l.lower()
    if 'bilingual' in l or ('sumerian' in l and 'akkadian' in l): return 'Bilingual'
    if 'akkadian' in l or 'assyrian' in l or 'babylonian' in l: return 'Akkadian'
    if 'sumerian' in l: return 'Sumerian'
    if any(x in l for x in ['urartian', 'hittite', 'eblaite', 'elamite', 'old persian', 'ugaritic']): return 'Peripheral/Other'
    return 'Unknown'

def map_period(p):
    if not p: return 'Unknown'
    p = p.lower()
    if 'neo-assyrian' in p or 'neo assyrian' in p: return 'Neo-Assyrian'
    if 'ur iii' in p: return 'Ur III'
    if 'old assyrian' in p: return 'Old Assyrian'
    if 'old babylonian' in p: return 'Old Babylonian'
    if 'middle assyrian' in p: return 'Middle Assyrian'
    if 'middle babylonian' in p: return 'Middle Babylonian'
    if 'neo-babylonian' in p or 'neo/late babylonian' in p or 'late babylonian' in p: return 'Neo-Babylonian'
    if any(x in p for x in ['ed iii', 'ed i-ii', 'early dynastic', 'old akkadian', 'lagaš ii', 'lagash ii', 'ebla', 'uruk iii', 'uruk iv']): return 'Third Millennium'
    if any(x in p for x in ['seleucid', 'achaemenid', 'hellenistic']): return 'Late Antiquity'
    return 'Unknown'

def map_genre(g):
    if not g: return 'Unknown'
    g = g.lower()
    if g in FAKE_GENRE_MARKERS: return 'Unknown'
    if 'administrative' in g: return 'Administrative'
    if 'lexical' in g: return 'Lexical'
    if 'royal' in g or 'monumental' in g: return 'Royal Inscriptions'
    if any(x in g for x in ['literary', 'scholarly', 'astrolog', 'astronomical', 'omen', 'school',
                             'ritual', 'incantation', 'extispicy', 'mathematical', 'scientific',
                             'technical procedure', 'prayer']): return 'Literary & Scholarly'
    if any(x in g for x in ['legal', 'treaty', 'grant']): return 'Legal'
    if 'letter' in g: return 'Letters'
    return 'Unknown'

def map_provenience(p):
    if not p: return 'Unknown'
    p = p.lower()
    if 'nineveh' in p or 'kuyunjik' in p: return 'Nineveh'
    if 'umma' in p: return 'Umma'
    if 'girsu' in p or 'tello' in p: return 'Girsu'
    if 'nippur' in p or 'nuffar' in p: return 'Nippur'
    if 'puzriš-dagan' in p or 'puzris-dagan' in p or 'drehem' in p: return 'Puzriš-Dagan'
    if 'kanesh' in p or 'kültepe' in p: return 'Kanesh'
    if 'aššur' in p or 'assur' in p or 'ashur' in p or 'qal' in p and 'sherqat' in p: return 'Assur'
    if 'uruk' in p or 'warka' in p: return 'Uruk'
    if p.startswith('ur ') or p.startswith('ur(') or 'tell muqayyar' in p or p == 'ur': return 'Ur'
    if 'ugarit' in p or 'ras shamra' in p: return 'Ugarit'
    if 'sippar' in p: return 'Sippar'
    if 'nimrud' in p or 'kalhu' in p: return 'Nimrud'
    return 'Unknown'

LANGUAGE_LABELS = ['Akkadian', 'Sumerian', 'Bilingual', 'Peripheral/Other']
PERIOD_LABELS = ['Neo-Assyrian', 'Ur III', 'Old Babylonian', 'Old Assyrian', 'Middle Assyrian', 'Middle Babylonian', 'Neo-Babylonian', 'Third Millennium', 'Late Antiquity']
GENRE_LABELS = ['Administrative', 'Lexical', 'Royal Inscriptions', 'Literary & Scholarly', 'Legal', 'Letters']
PROVENIENCE_LABELS = ['Nineveh', 'Umma', 'Girsu', 'Nippur', 'Puzriš-Dagan', 'Kanesh', 'Assur', 'Uruk', 'Ur', 'Ugarit', 'Sippar', 'Nimrud']

def label_to_idx(label_str, label_list):
    if not label_str or label_str == 'Unknown':
        return -100
    try:
        return label_list.index(label_str)
    except ValueError:
        return -100

def load_and_deduplicate_v2(files):
    print("Loading and deduplicating datasets (v2)...")

    # Cross-source dedup: a small number of CuneiML tablets (~1.5% of them)
    # are the same physical tablet as an ORACC edition (same CDLI P-number).
    # Keep the CuneiML side (it carries images/bboxes for later multimodal
    # heads, and tends to have more lines per tablet) and drop the ORACC
    # duplicate here, before the sign-string dedup below -- otherwise the
    # same tablet contributes near-duplicate lines under two different
    # transliteration styles, and could land in different splits later.
    cuneiml_tablet_ids = set()
    for file_path in files:
        if "cuneiml" in str(file_path).lower():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        tid = json.loads(line).get('tablet_id')
                        if tid:
                            cuneiml_tablet_ids.add(tid)
                    except Exception:
                        pass
    print(f"CuneiML tablet count (preferred on overlap): {len(cuneiml_tablet_ids)}")

    unique_lines = {}
    skipped_cross_source = 0

    for file_path in files:
        print(f"Reading {file_path}...")
        is_oracc = "oracc" in str(file_path).lower()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data = json.loads(line)
                    if is_oracc and data.get('tablet_id') in cuneiml_tablet_ids:
                        skipped_cross_source += 1
                        continue
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

    print(f"Skipped (ORACC lines whose tablet is already in CuneiML): {skipped_cross_source}")
    print(f"Total unique lines across datasets: {len(unique_lines)}")
    return list(unique_lines.values())

def to_examples(records):
    """Untokenized examples: both 'signs' (cuneiform) and 'text' (cleaned
    transliteration) side by side, plus the 4 metadata labels shared by both
    training pipelines."""
    examples = []
    for data in tqdm(records, desc="Building examples"):
        examples.append({
            "signs": data.get('signs', []),
            "text": clean_transliteration(data.get('raw')),
            # Carried through (unused by either tokenizer) so a later training
            # stage can look up a per-tablet image by id -- e.g.
            # train_mbert.py's --use_image vision branch, which needs to know
            # which physical tablet a line belongs to. Empty string (not
            # None) so the Arrow 'string' column stays uniformly typed;
            # ORACC-sourced rows get their own synthetic "oracc:..." id
            # (see prepare_oracc.py), which simply won't match any collected
            # image -- fine, since only CuneiML tablets have photos.
            "tablet_id": data.get('tablet_id') or "",
            "period_labels": label_to_idx(map_period(data.get('period')), PERIOD_LABELS),
            "genre_labels": label_to_idx(map_genre(data.get('genre')), GENRE_LABELS),
            "language_labels": label_to_idx(map_language(data.get('language')), LANGUAGE_LABELS),
            "provenience_labels": label_to_idx(map_provenience(data.get('provenience')), PROVENIENCE_LABELS),
        })
    return examples

def main():
    base_dir = Path(r"C:\Programming\akkadian\data")
    interim_dir = base_dir / "interim"
    processed_dir = base_dir / "processed"
    os.makedirs(processed_dir, exist_ok=True)

    files_to_merge = [
        interim_dir / "oracc.jsonl",
        interim_dir / "cuneiml.jsonl"
    ]
    
    # 1. Deduplicate & Merge
    all_unique_records = load_and_deduplicate_v2(files_to_merge)
    
    # 2. Build Tokenizer Vocabs -- 'text' is stored here (not just derived
    # later in to_examples) so build_vocab(field='text') can read it directly
    # from this same file, the same way it already reads 'signs'.
    combined_path = processed_dir / "combined_unique.jsonl"
    with open(combined_path, 'w', encoding='utf-8') as f:
        for r in all_unique_records:
            r['text'] = clean_transliteration(r.get('raw'))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    signs_tokenizer = CharacterTokenizer()
    signs_tokenizer.build_vocab(str(combined_path), min_freq=2, field='signs')
    vocab_path = processed_dir / "vocab.json"
    signs_tokenizer.save(str(vocab_path))
    print(f"Signs vocab size: {len(signs_tokenizer.vocab)}")

    text_tokenizer = CharacterTokenizer()
    text_tokenizer.build_vocab(str(combined_path), min_freq=2, field='text')
    vocab_translit_path = processed_dir / "vocab_translit.json"
    text_tokenizer.save(str(vocab_translit_path))
    print(f"Transliteration vocab size: {len(text_tokenizer.vocab)}")
    
    # 3. Grouped split (90/5/5) -- grouped by tablet_id, not shuffled per line.
    # Lines from the same physical tablet are highly correlated (same
    # formulae, obviously the same period/genre/provenience), so a per-line
    # random split leaks tablet identity across train/val/test and inflates
    # both MLM and classification metrics.
    groups = {}
    for r in all_unique_records:
        groups.setdefault(r.get('tablet_id') or r.get('signs', [None])[0], []).append(r)

    group_keys = list(groups.keys())
    random.seed(42)
    random.shuffle(group_keys)

    n = len(all_unique_records)
    test_budget = int(n * 0.05)
    val_budget = int(n * 0.05)

    test_raw, val_raw, train_raw = [], [], []
    for key in group_keys:
        recs = groups[key]
        if len(test_raw) < test_budget:
            test_raw.extend(recs)
        elif len(val_raw) < val_budget:
            val_raw.extend(recs)
        else:
            train_raw.extend(recs)

    print(f"Tablet groups: {len(group_keys)}")
    print(f"Split sizes: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
    
    # 4. Save Test split un-tokenized
    test_path = processed_dir / "test.jsonl"
    print(f"Saving un-tokenized test set to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for r in test_raw:
            # save mapped fields for easy eval later
            r['period_mapped'] = map_period(r.get('period'))
            r['genre_mapped'] = map_genre(r.get('genre'))
            r['language_mapped'] = map_language(r.get('language'))
            r['provenience_mapped'] = map_provenience(r.get('provenience'))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 5. Process Train and Val (untokenized -- each training script tokenizes
    # its own 'signs' or 'text' column at load time)
    print("Processing Train records...")
    train_processed = to_examples(train_raw)
    print("Processing Val records...")
    val_processed = to_examples(val_raw)

    # Define features
    features = Features({
        'signs': Sequence(Value('string')),
        'text': Value('string'),
        'tablet_id': Value('string'),
        'period_labels': Value('int64'),
        'genre_labels': Value('int64'),
        'language_labels': Value('int64'),
        'provenience_labels': Value('int64'),
    })
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_processed, features=features),
        "validation": Dataset.from_list(val_processed, features=features)
    })
    
    hf_dir = processed_dir / "hf_dataset"
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
    with open(processed_dir / "label_configs.json", "w", encoding='utf-8') as f:
        json.dump(label_dicts, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
