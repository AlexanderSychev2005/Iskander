"""Assemble the reviewed tablet crops into a proper HF `datasets.Dataset`
(one row per unique tablet, not per text line) and save it locally.

Deliberately NOT one row per line of the main text dataset (which has
tablet_id but is line-granular, ~604k rows): a photo belongs to a tablet,
not to any one of its lines, and a table can have anywhere from 1 to dozens
of lines. Embedding the same image bytes into every line-row of a tablet
would multiply storage for no reason and doesn't match how train_mbert.py
actually consumes it (a tablet_id -> image lookup, joined at collate time).
This is that same join, materialized as its own compact table: one row per
reviewed tablet id, ready to publish as a separate HF dataset config
("vision") alongside the existing line-level "default" config, joinable by
tablet_id.

Split into train/val/test matching the SAME tablet-level assignment as
data/processed/hf_dataset's line-level splits (not a fresh random split of
just the image subset) -- a tablet's photo and its own text lines must
stay on the same side of the split, same as any other tablet-grouped data,
and this keeps the vision config directly comparable to (and joinable
with) the text config's val/test rather than an independently-drawn subset.

Output: data/processed/hf_dataset_vision/ (DatasetDict via save_to_disk,
splits train/validation/test)
  columns: tablet_id, image, x1, y1, x2, y2 (bbox in the ORIGINAL image's
  pixel space, not the saved crop's), period, genre, provenience, language
  (this project's canonical mapped labels, "Unknown" if the raw CDLI field
  didn't map to a known class -- see prepare_hf_dataset.py's map_* functions).
"""
import csv
import json
import os
import sys

from datasets import Dataset, DatasetDict, Features, Image, Value, load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.collect_vision_dataset import load_all_candidates
from src.data_pipeline.prepare_hf_dataset import map_genre, map_language, map_period, map_provenience
from src.data_pipeline.prepare_cdli_bulk import split_for as cdli_bulk_split_for

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROPS_DIR = os.path.join(BASE_DIR, "data", "vision_dataset_final")
BBOX_CSV = os.path.join(CROPS_DIR, "bboxes.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "hf_dataset_vision")
TEXT_DATASET_DIR = os.path.join(BASE_DIR, "data", "processed", "hf_dataset")


def tablet_split_map():
    """tablet_id -> which split (train/validation/test) it belongs to in
    the text dataset, so the vision rows can be assigned consistently."""
    text_ds = load_from_disk(TEXT_DATASET_DIR)
    mapping = {}
    for split in ("train", "validation", "test"):
        for tid in set(text_ds[split]["tablet_id"]):
            if tid:
                mapping[tid] = split
    return mapping


def bulk_backfill_meta():
    """tablet_id -> {period, genre, provenience, language} for tablets added
    by prepare_cdli_bulk.py / the eBL backfill (session 2026-08-12). These
    aren't in CuneiML's JSON (that's WHY they needed a separate source), so
    load_all_candidates() below can't supply their metadata -- read it back
    from the same interim files add_cdli_bulk_documents.py already resolved
    it into. Also doubles as the id set for the split_for() fallback: only
    these get it, unlike the ~221 pre-existing CuneiML-photo orphans with no
    text anywhere, which should stay excluded."""
    meta = {}
    for fname in ("cdli_bulk_documents.jsonl", "ebl_bulk_documents.jsonl", "balance_documents.jsonl"):
        path = os.path.join(BASE_DIR, "data", "interim", fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                meta[r["tablet_id"]] = r
    return meta


def main():
    candidates = load_all_candidates()
    split_of = tablet_split_map()
    bulk_meta = bulk_backfill_meta()

    rows = {"train": [], "validation": [], "test": []}
    n_unmatched = 0
    with open(BBOX_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            img_path = os.path.join(CROPS_DIR, f"{pid}.jpg")
            if not os.path.exists(img_path):
                continue
            tablet_id = "P" + pid.zfill(6) if pid.isdigit() else pid
            if tablet_id in bulk_meta:
                meta = bulk_meta[tablet_id]
            else:
                pair = candidates.get(pid)
                meta = pair[1] if pair else {}
            split = split_of.get(tablet_id)
            if split is None and tablet_id in bulk_meta:
                # Same deterministic hash used in add_cdli_bulk_documents.py,
                # so this always agrees with the documents config.
                split = cdli_bulk_split_for(tablet_id)
            if split is None:
                n_unmatched += 1
                continue
            rows[split].append({
                "tablet_id": tablet_id,
                "image": img_path,
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
                "period": map_period(meta.get("period", "")),
                "genre": map_genre(meta.get("genre", "")),
                "provenience": map_provenience(meta.get("provenience", "")),
                "language": map_language(meta.get("language", "")),
            })

    features = Features({
        "tablet_id": Value("string"),
        "image": Image(),
        "x1": Value("float32"), "y1": Value("float32"),
        "x2": Value("float32"), "y2": Value("float32"),
        "period": Value("string"), "genre": Value("string"),
        "provenience": Value("string"), "language": Value("string"),
    })
    ds = DatasetDict({
        split: Dataset.from_list(split_rows, features=features)
        for split, split_rows in rows.items()
    })
    ds.save_to_disk(OUT_DIR)
    print(f"Saved to {OUT_DIR} ({n_unmatched} tablets had no matching text split, skipped)")
    for split, split_rows in rows.items():
        print(f"  {split}: {len(split_rows)}")


if __name__ == "__main__":
    main()
