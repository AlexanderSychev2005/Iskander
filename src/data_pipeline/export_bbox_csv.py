"""Export the finished manual bbox review as a flat CSV, for pushing to HF
alongside the rest of the dataset. One row per tablet id actually in scope
(present in data/vision_dataset/manifest.jsonl AND reviewed "ok" in
data/bbox_corrections.jsonl) -- ids reviewed under an earlier manifest that
no longer includes them (see review_bboxes_gui.py's stale-correction note)
are excluded, since they're not part of the current collected set.

bbox coordinates are in the ORIGINAL full-resolution image's pixel space
(same convention as CuneiML's own bboxes and bbox_corrections.jsonl) --
not the 224x224 crop's space -- so this CSV stays valid regardless of
whatever target resolution finalize_vision_crops.py uses.

Output: data/vision_dataset_final/bboxes.csv
  columns: id, x1, y1, x2, y2
"""
import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_FILE = os.path.join(BASE_DIR, "data", "vision_dataset", "manifest.jsonl")
CORRECTIONS_FILE = os.path.join(BASE_DIR, "data", "bbox_corrections.jsonl")
OUT_CSV = os.path.join(BASE_DIR, "data", "vision_dataset_final", "bboxes.csv")


def main():
    manifest_ids = set()
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                manifest_ids.add(str(json.loads(line)["id"]))
            except Exception:
                pass

    rows = {}
    with open(CORRECTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid = str(row["id"])
            if row["status"] != "ok" or pid not in manifest_ids:
                continue
            (x1, y1), (x2, y2) = row["bbox"]
            rows[pid] = (pid, x1, y1, x2, y2)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x1", "y1", "x2", "y2"])
        for pid in sorted(rows, key=int):
            writer.writerow(rows[pid])

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    missing = manifest_ids - set(rows)
    if missing:
        print(f"Note: {len(missing)} manifest ids have no 'ok' review yet -- not in the CSV.")


if __name__ == "__main__":
    main()
