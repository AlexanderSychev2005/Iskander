"""One-off export for manual bbox QA (see docs/thesis_outline.md, section 6:
'Image bounding-box quality is inconsistent'). For a random sample of
CuneiML entries that have a locally-cached photo and a recorded bbox, fetch
the full-resolution CDLI photo (bbox coordinates are recorded against the
full-res image, not the ~4x-downscaled locally cached thumbnail), draw the
box, and append a caption panel with the line-by-line transliteration of the
face the box is assumed to cover ('obverse', falling back to whichever face
has content) so a human reviewer can judge box placement without needing to
read cuneiform. Delete bad files directly from the output folder; nothing
reads this folder back automatically.
"""
import json
import os
import random
import textwrap
import urllib.request
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_FILE = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "CuneiMLv1.2.json")
IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "images")
OUT_DIR = os.path.join(BASE_DIR, "data", "bbox_review")
FONT_PATH = "C:/Windows/Fonts/arial.ttf"
N_SAMPLES = 1000
SEED = 42
MAX_WORKERS = 8

os.makedirs(OUT_DIR, exist_ok=True)


def face_lines(text_obj):
    if not isinstance(text_obj, dict):
        return None
    for face in ("obverse", "reverse", "left", "right", "top", "bottom"):
        lines = text_obj.get(face)
        if lines:
            return face, lines
    return None


def valid_bbox(bb):
    if not bb or len(bb) != 2:
        return False
    (x1, y1), (x2, y2) = bb
    return (x2 - x1) > 10 and (y2 - y1) > 10


def build_caption(face, lines, width_px):
    font = ImageFont.truetype(FONT_PATH, 16)
    header = f"[{face}]"
    wrapped = [header]
    for ln in lines:
        if isinstance(ln, dict):
            raw = (ln.get("raw") or "").strip()
            num = ln.get("num", "")
        else:
            raw = str(ln).strip()
            num = ""
        if raw:
            wrapped.extend(textwrap.wrap(f"{num}. {raw}", width=60) or [f"{num}."])
    line_h = 20
    panel_h = 10 + line_h * len(wrapped) + 10
    panel = Image.new("RGB", (width_px, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    y = 10
    for row in wrapped:
        draw.text((10, y), row, font=font, fill=(0, 0, 0))
        y += line_h
    return panel


def process(item):
    pid = str(item["id"])
    try:
        bb = item.get("bboxes")
        if not valid_bbox(bb):
            return pid, "skip: bad bbox"
        fl = face_lines(item.get("text"))
        if fl is None:
            return pid, "skip: no text"
        face, lines = fl

        url = (item.get("img_url") or "").replace("/tn_photo/", "/photo/")
        if not url:
            return pid, "skip: no url"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw_bytes = urllib.request.urlopen(req, timeout=25).read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        draw = ImageDraw.Draw(img)
        (x1, y1), (x2, y2) = bb
        w, h = img.size
        if x2 <= w and y2 <= h and x1 >= 0 and y1 >= 0:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
        else:
            draw.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 0), width=8)

        img.thumbnail((900, 1300))
        caption = build_caption(face, lines, img.width)
        combined = Image.new("RGB", (img.width, img.height + caption.height), (255, 255, 255))
        combined.paste(img, (0, 0))
        combined.paste(caption, (0, img.height))
        combined.save(os.path.join(OUT_DIR, f"{pid}.jpg"), quality=85)
        return pid, "ok"
    except Exception as e:
        return pid, f"fail: {e}"


def main():
    with open(JSON_FILE, encoding="utf-8") as f:
        data = json.load(f)

    local_ids = set(f.rsplit(".", 1)[0] for f in os.listdir(IMG_DIR))
    seen = {}
    for it in data:
        pid = str(it["id"])
        if pid in local_ids and pid not in seen and valid_bbox(it.get("bboxes")) and face_lines(it.get("text")):
            seen[pid] = it

    candidates = list(seen.values())
    print(f"candidates (unique id, local image, bbox, text): {len(candidates)}")
    random.seed(SEED)
    sample = random.sample(candidates, min(N_SAMPLES, len(candidates)))

    ok, skipped, failed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process, it) for it in sample]
        for i, fut in enumerate(as_completed(futures)):
            pid, status = fut.result()
            if status == "ok":
                ok += 1
            elif status.startswith("skip"):
                skipped += 1
            else:
                failed += 1
                print(pid, status)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(sample)} done (ok={ok}, skipped={skipped}, failed={failed})")

    print(f"Done. ok={ok}, skipped={skipped}, failed={failed}. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
