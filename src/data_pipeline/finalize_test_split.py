"""Turn data/processed/test.jsonl (raw/untokenized, written by
prepare_hf_dataset.py's main()) into a Dataset with the exact same schema
as the train/validation splits already in data/processed/hf_dataset
(signs, text, tablet_id, 4 int-encoded *_labels columns) -- test.jsonl was
deliberately kept out of that DatasetDict so it wasn't trivially loadable
alongside train/val (discourages casually peeking at it during iteration),
but it was never pushed anywhere at all, which the pushed HF dataset needs
fixed to have a real held-out split. Re-adds it as a third split, both
locally and on the Hub.
"""
import json
import os
import sys

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.prepare_hf_dataset import (
    GENRE_LABELS, LANGUAGE_LABELS, PERIOD_LABELS, PROVENIENCE_LABELS, label_to_idx,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_JSONL = os.path.join(BASE_DIR, "data", "processed", "test.jsonl")
HF_DATASET_DIR = os.path.join(BASE_DIR, "data", "processed", "hf_dataset")


def build_test_dataset():
    rows = []
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "signs": r.get("signs", []),
                "text": r.get("text", ""),
                "tablet_id": r.get("tablet_id") or "",
                "period_labels": label_to_idx(r.get("period_mapped"), PERIOD_LABELS),
                "genre_labels": label_to_idx(r.get("genre_mapped"), GENRE_LABELS),
                "language_labels": label_to_idx(r.get("language_mapped"), LANGUAGE_LABELS),
                "provenience_labels": label_to_idx(r.get("provenience_mapped"), PROVENIENCE_LABELS),
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
    return Dataset.from_list(rows, features=features)


def main():
    test_ds = build_test_dataset()
    print(f"Built test split: {len(test_ds)} rows")

    ds = load_from_disk(HF_DATASET_DIR)
    ds = DatasetDict({"train": ds["train"], "validation": ds["validation"], "test": test_ds})

    # save_to_disk can't overwrite a directory it's memory-mapped from --
    # write to a sibling temp dir, then swap it into place.
    import shutil
    tmp_dir = HF_DATASET_DIR + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    ds.save_to_disk(tmp_dir)
    del ds
    shutil.rmtree(HF_DATASET_DIR)
    os.rename(tmp_dir, HF_DATASET_DIR)
    print(f"Saved train/validation/test to {HF_DATASET_DIR}")


if __name__ == "__main__":
    main()
