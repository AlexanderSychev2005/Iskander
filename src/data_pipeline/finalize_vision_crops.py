"""Crop collected vision-dataset images to their tablet bbox and resize to a
fixed size for the ResNet vision branch (see train_mbert_vision.py). Mirrors
Aeneas's own design (Assael et al. 2025, Methods p.148: "The visual inputs
are processed using a ResNet-8 ... concatenated with the relevant textual
embeddings") -- crop first so the CNN only ever sees the tablet face, not
the surrounding photo background/scale bar/collection card that the raw
CuneiML bbox sometimes locks onto instead (session bbox-quality audit,
2026-08-05).

For each id in data/vision_dataset/manifest.jsonl:
  - skipped entirely if marked "no_tablet" in data/bbox_corrections.jsonl.
  - bbox = the correction's saved box if reviewed, else the manifest's own
    (raw CuneiML) bbox if present, else skipped (nothing to crop). Cropped
    exactly as drawn -- no added margin. An earlier version padded the box
    by 8% to avoid clipping signs at the edge, but that could pull a
    deliberately-excluded museum marker/ink mark back into frame after the
    reviewer had specifically cropped it out (session correction,
    2026-08-06), so the crop now trusts the reviewed box exactly.
  - resized to fit within TARGET_SIZE x TARGET_SIZE (224, matching Aeneas's
    own model input exactly -- Assael et al. 2025, Methods p.148: "a
    corresponding greyscale image of size 224 x 224") preserving aspect
    ratio, then letterboxed (centered on a black canvas) to a square. A
    plain squish-to-square would distort tablet shape -- round tablets have
    an already-roughly-square bbox so squish barely affects them, but a
    rectangular/elongated one (e.g. a "pillow"-shaped letter) would get
    stretched toward square, destroying exactly the width:height signal the
    period/genre shape hypothesis (round vs. square vs. elongated, session
    discussion) depends on. Letterbox fill is black to match the photo's
    own backdrop, not an artificial border. No need to hedge the stored
    resolution against a hypothetical future architecture: the actual
    future-proofing is data/raw/cuneiml/images_full/ (full-resolution
    originals, kept locally) plus the recorded bbox in
    data/bbox_corrections.jsonl, which together can regenerate a crop at
    any resolution later without redoing the manual review.

Output: data/vision_dataset_final/<id>.jpg
        data/vision_dataset_final/crops_manifest.jsonl
          {"id", "reviewed": bool} -- reviewed=False means the crop still
          relies on CuneiML's own automated bbox (~58% reliable per the
          session's own 24-sample audit). train_mbert_vision.py defaults to
          reviewed-only for the first pilot; the flag lets that loosen as
          manual review (review_bboxes_gui.py) progresses. Re-running this
          script picks up newly-reviewed ids automatically (skips ids
          already cropped, same as the rest of this pipeline).
"""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.review_bboxes_gui import build_path_index, load_manifest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORRECTIONS_FILE = os.path.join(BASE_DIR, "data", "bbox_corrections.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "data", "vision_dataset_final")
OUT_MANIFEST = os.path.join(OUT_DIR, "crops_manifest.jsonl")
TARGET_SIZE = 224


def load_corrections():
    corrections = {}
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    corrections[str(row["id"])] = row
                except Exception:
                    pass
    return corrections


def crop_and_resize(img, bbox):
    w, h = img.size
    (x1, y1), (x2, y2) = bbox
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = img.crop((x1, y1, x2, y2))

    # Aspect-preserving resize + letterbox to square -- see module docstring
    # (a plain squish-to-square would distort tablet shape).
    cw, ch = crop.size
    scale = TARGET_SIZE / max(cw, ch)
    new_w, new_h = max(1, round(cw * scale)), max(1, round(ch * scale))
    resized = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    canvas.paste(resized, ((TARGET_SIZE - new_w) // 2, (TARGET_SIZE - new_h) // 2))
    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = load_manifest()
    corrections = load_corrections()
    path_index = build_path_index()

    already = {f.rsplit(".", 1)[0] for f in os.listdir(OUT_DIR) if f.endswith(".jpg")}
    done_meta = {}
    if os.path.exists(OUT_MANIFEST):
        with open(OUT_MANIFEST, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_meta[row["id"]] = row
                except Exception:
                    pass

    n_ok, n_skip_no_tablet, n_skip_no_bbox, n_skip_missing_img, n_fail = 0, 0, 0, 0, 0
    for item in manifest:
        pid = str(item["id"])
        corr = corrections.get(pid)
        if corr and corr["status"] == "no_tablet":
            n_skip_no_tablet += 1
            continue

        reviewed = bool(corr and corr["status"] == "ok")
        bbox = corr["bbox"] if reviewed else item.get("bbox")
        if not bbox:
            n_skip_no_bbox += 1
            continue

        if pid in already and done_meta.get(pid, {}).get("reviewed") == reviewed:
            done_meta[pid] = {"id": pid, "bbox": bbox, "reviewed": reviewed}  # backfill bbox on older manifests
            n_ok += 1
            continue

        img_path = path_index.get(pid)
        if img_path is None:
            n_skip_missing_img += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            crop = crop_and_resize(img, bbox)
            if crop is None:
                n_skip_no_bbox += 1
                continue
            crop.save(os.path.join(OUT_DIR, f"{pid}.jpg"), quality=90)
            done_meta[pid] = {"id": pid, "bbox": bbox, "reviewed": reviewed}
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"  fail {pid}: {e}")

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        for row in done_meta.values():
            f.write(json.dumps(row) + "\n")

    n_reviewed = sum(1 for r in done_meta.values() if r["reviewed"])
    print(f"Done. {n_ok} crops on disk ({n_reviewed} reviewed, {n_ok - n_reviewed} raw-bbox).")
    print(f"Skipped: {n_skip_no_tablet} no_tablet, {n_skip_no_bbox} no usable bbox, "
          f"{n_skip_missing_img} missing source image, {n_fail} failed.")
    print(f"Manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
