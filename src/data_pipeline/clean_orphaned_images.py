"""Remove collected tablet images whose tablet_id has no surviving line in
data/processed/hf_dataset (train/validation/test combined) -- this happens
when prepare_hf_dataset.py's cross-tablet line dedup (same sign string seen
under an earlier tablet_id) absorbs every one of a tablet's lines into a
different tablet's record, leaving that tablet_id with zero rows anywhere.
Such an image can never be joined to any text example, so it's dead weight:
takes disk space, and blocks collect_vision_dataset.py from drawing a real
replacement candidate for that class (it looks "already collected").

Deletes from:
  - data/vision_dataset/<head>/<class>/<id>.jpg (every class folder it's in)
  - data/vision_dataset_final/<id>.jpg + its row in crops_manifest.jsonl
  - its row in data/vision_dataset_final/bboxes.csv

Does NOT touch data/bbox_corrections.jsonl (the manual review record) --
harmless if unused, and re-collecting the exact same id later (unlikely,
but possible if pool composition shifts again) would still benefit from
not having to re-review it.

Re-run collect_vision_dataset.py afterwards to backfill: it treats
whatever's left on disk as "already collected" and draws fresh replacement
candidates from the pool for any class now short of its target.
"""
import csv
import json
import os
import sys

from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT_DATASET_DIR = os.path.join(BASE_DIR, "data", "processed", "hf_dataset")
VISION_DATASET_DIR = os.path.join(BASE_DIR, "data", "vision_dataset")
CROPS_DIR = os.path.join(BASE_DIR, "data", "vision_dataset_final")
CROPS_MANIFEST = os.path.join(CROPS_DIR, "crops_manifest.jsonl")
BBOX_CSV = os.path.join(CROPS_DIR, "bboxes.csv")


def to_tablet_id(pid):
    return "P" + pid.zfill(6) if pid.isdigit() else pid


def main():
    text_ds = load_from_disk(TEXT_DATASET_DIR)
    matched = set()
    for split in ("train", "validation", "test"):
        matched |= set(t for t in text_ds[split]["tablet_id"] if t)

    orphaned = set()
    for head in os.listdir(VISION_DATASET_DIR):
        head_dir = os.path.join(VISION_DATASET_DIR, head)
        if not os.path.isdir(head_dir):
            continue
        for cls in os.listdir(head_dir):
            cls_dir = os.path.join(head_dir, cls)
            for fn in os.listdir(cls_dir):
                pid = fn.rsplit(".", 1)[0]
                if to_tablet_id(pid) not in matched:
                    orphaned.add(pid)
                    os.remove(os.path.join(cls_dir, fn))

    print(f"Removed {len(orphaned)} orphaned ids from per-class folders")

    crop_path = os.path.join(CROPS_DIR, f"{{}}.jpg")
    for pid in orphaned:
        p = crop_path.format(pid)
        if os.path.exists(p):
            os.remove(p)

    if os.path.exists(CROPS_MANIFEST):
        rows = []
        with open(CROPS_MANIFEST, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if str(row["id"]) not in orphaned:
                    rows.append(row)
        with open(CROPS_MANIFEST, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"crops_manifest.jsonl: {len(rows)} rows remain")

    if os.path.exists(BBOX_CSV):
        with open(BBOX_CSV, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        kept = [r for r in reader if r["id"] not in orphaned]
        with open(BBOX_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(kept)
        print(f"bboxes.csv: {len(kept)} rows remain")


if __name__ == "__main__":
    main()
