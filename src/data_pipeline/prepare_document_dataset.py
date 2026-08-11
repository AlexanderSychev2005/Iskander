"""Document-granularity version of the mBERT dataset: one row per physical
tablet (concatenated text of all its surviving lines, in original face/line
order) instead of one row per line. Built to pair naturally with the vision
config (one photo per tablet already) without the line-repetition frequency
skew a line-level dataset has (session finding, 2026-08-06) -- and closer to
Aeneas's own design, where geographic/date attribution use the whole
inscription's embedding, not a per-line one.

Source: data/processed/combined_unique.jsonl (written by
prepare_hf_dataset.py's main() -- the deduplicated, pre-split line pool,
already carrying 'text' (cleaned transliteration) and 'tablet_id'). Lines
are grouped by tablet_id preserving file order, which already matches
original face/line order (prepare_cuneiml.py writes one tablet's lines
consecutively, face by face; the sign-string dedup in
load_and_deduplicate_v2 only ever *drops* colliding lines, never reorders
survivors).

Split: reuses the EXACT same train/validation/test tablet assignment as
data/processed/hf_dataset (same tablet-grouped 90/5/5 split) rather than
drawing a fresh one -- a tablet's document and its line-level rows (and its
vision-config row, if it has a photo) always land on the same side.

max_length: mBERT's absolute position embeddings hard-cap at 512 tokens --
this is not a tunable choice, it's the model's ceiling. Measured on a 3,000-
tablet sample: median 66 tokens, p90=427, p95=711, p99=2326, max=6738 --
so 512 covers the large majority but truncates the long tail (mostly bulky
lexical/omen compendia), same as any BERT-family model working with
multi-paragraph documents.

Output: data/processed/hf_dataset_documents/ (DatasetDict via save_to_disk,
splits train/validation/test), same column schema as hf_dataset (signs,
text, tablet_id, period_labels, genre_labels, language_labels,
provenience_labels) so train_mbert.py needs no code changes -- just point
--data_dir here and raise --max_length.
"""
import json
import os
import sys
from collections import defaultdict

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.prepare_hf_dataset import (
    GENRE_LABELS, LANGUAGE_LABELS, PERIOD_LABELS, PROVENIENCE_LABELS, label_to_idx,
    map_genre, map_language, map_period, map_provenience,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMBINED_PATH = os.path.join(BASE_DIR, "data", "processed", "combined_unique.jsonl")
TEXT_DATASET_DIR = os.path.join(BASE_DIR, "data", "processed", "hf_dataset")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "hf_dataset_documents")


def tablet_split_map():
    text_ds = load_from_disk(TEXT_DATASET_DIR)
    mapping = {}
    for split in ("train", "validation", "test"):
        for tid in set(text_ds[split]["tablet_id"]):
            if tid:
                mapping[tid] = split
    return mapping


def main():
    split_of = tablet_split_map()

    # Group preserving file order == original face/line order (see docstring).
    docs = defaultdict(lambda: {"signs": [], "texts": [], "period": None, "genre": None,
                                 "provenience": None, "language": None})
    with open(COMBINED_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            tid = r.get("tablet_id")
            if not tid:
                continue
            d = docs[tid]
            d["signs"].extend(r.get("signs", []))
            text = r.get("text", "")
            if text:
                d["texts"].append(text)
            # metadata is identical across a tablet's lines; keep first non-empty
            for field in ("period", "genre", "provenience", "language"):
                if not d[field] and r.get(field):
                    d[field] = r.get(field)

    rows = {"train": [], "validation": [], "test": []}
    n_unmatched = 0
    for tid, d in docs.items():
        split = split_of.get(tid)
        if split is None:
            n_unmatched += 1
            continue
        rows[split].append({
            "signs": d["signs"],
            "text": " ".join(d["texts"]),
            "tablet_id": tid,
            "period_labels": label_to_idx(map_period(d["period"]), PERIOD_LABELS),
            "genre_labels": label_to_idx(map_genre(d["genre"]), GENRE_LABELS),
            "language_labels": label_to_idx(map_language(d["language"]), LANGUAGE_LABELS),
            "provenience_labels": label_to_idx(map_provenience(d["provenience"]), PROVENIENCE_LABELS),
        })

    features = Features({
        "signs": Sequence(Value("string")),
        "text": Value("string"),
        "tablet_id": Value("string"),
        "period_labels": Value("int64"),
        "genre_labels": Value("int64"),
        "language_labels": Value("int64"),
        "provenience_labels": Value("int64"),
    })
    ds = DatasetDict({split: Dataset.from_list(split_rows, features=features) for split, split_rows in rows.items()})
    ds.save_to_disk(OUT_DIR)
    print(f"Saved to {OUT_DIR} ({n_unmatched} tablets had no split assignment, skipped -- "
          f"same tablets excluded from hf_dataset's own line-level splits)")
    for split, split_rows in rows.items():
        print(f"  {split}: {len(split_rows)} documents")


if __name__ == "__main__":
    main()
