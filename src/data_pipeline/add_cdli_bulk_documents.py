"""Append the new tablets from prepare_cdli_bulk.py's output into the
existing data/processed/hf_dataset_documents DatasetDict, WITHOUT touching
any existing tablet's split assignment (see prepare_cdli_bulk.py's docstring
for why the split field it assigned is used as-is here rather than re-
running prepare_hf_dataset.py's random 90/5/5 split over everything).

Run after prepare_cdli_bulk.py. Saves back to the same dir and (optionally)
pushes the updated 'documents' config to the Hub.
"""
import json
import os
import sys

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.prepare_hf_dataset import (
    GENRE_LABELS, LANGUAGE_LABELS, PERIOD_LABELS, PROVENIENCE_LABELS, label_to_idx,
    map_genre, map_language, map_period, map_provenience,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_PATHS = [
    os.path.join(BASE_DIR, "data", "interim", "cdli_bulk_documents.jsonl"),
    os.path.join(BASE_DIR, "data", "interim", "ebl_bulk_documents.jsonl"),
    os.path.join(BASE_DIR, "data", "interim", "balance_documents.jsonl"),
    os.path.join(BASE_DIR, "data", "interim", "text_balance_documents.jsonl"),
]
DOCS_DIR = os.path.join(BASE_DIR, "data", "processed", "hf_dataset_documents")


def main():
    rows = {"train": [], "validation": [], "test": []}
    seen_tablet_ids = set()
    for in_path in IN_PATHS:
        if not os.path.exists(in_path):
            continue
        with open(in_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["tablet_id"] in seen_tablet_ids:
                    continue
                seen_tablet_ids.add(r["tablet_id"])
                rows[r["split"]].append({
                "signs": r.get("signs", []),
                "text": r["text"],
                "tablet_id": r["tablet_id"],
                "period_labels": label_to_idx(map_period(r["period"]), PERIOD_LABELS),
                "genre_labels": label_to_idx(map_genre(r["genre"]), GENRE_LABELS),
                "language_labels": label_to_idx(map_language(r["language"]), LANGUAGE_LABELS),
                "provenience_labels": label_to_idx(map_provenience(r["provenience"]), PROVENIENCE_LABELS),
            })

    ds = load_from_disk(DOCS_DIR)
    for split, new_rows in rows.items():
        if not new_rows:
            continue
        addition = Dataset.from_list(new_rows, features=ds[split].features)
        ds[split] = concatenate_datasets([ds[split], addition])

    ds.save_to_disk(DOCS_DIR + "_with_cdli_bulk")
    print("Saved to", DOCS_DIR + "_with_cdli_bulk")
    for split in ("train", "validation", "test"):
        print(f"  {split}: {len(ds[split])} ({len(rows[split])} new)")


if __name__ == "__main__":
    main()
