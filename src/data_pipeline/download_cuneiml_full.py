"""Download full-resolution (untouched, un-resized) CDLI photos for CuneiML
entries that have a usable bbox, transliteration, and complete
(period/genre/language/provenience) metadata via cdli_cat.csv (see
prepare_cuneiml.py for the same join). Estimated at ~128GB for the full
candidate set (session sizing check, 2026-08-05) -- a real risk of filling
the disk, so free space is checked before every write and downloading stops
cleanly once free space drops below FREE_SPACE_FLOOR_GB, rather than running
the disk to 0 and risking the rest of the system. Resumable: skips ids
already present in the manifest.
"""
import json
import os
import csv
import shutil
import urllib.request
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_FILE = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "CuneiMLv1.2.json")
IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "images")
CDLI_CAT_CSV = os.path.join(BASE_DIR, "data", "raw", "cdli_data", "cdli_cat.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "images_full")
MANIFEST_FILE = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "images_full_manifest.jsonl")
MAX_WORKERS = 8
FREE_SPACE_FLOOR_GB = 15

os.makedirs(OUT_DIR, exist_ok=True)


def free_space_gb():
    return shutil.disk_usage(BASE_DIR).free / (1024 ** 3)


def valid_bbox(bb):
    if not bb or len(bb) != 2:
        return False
    (x1, y1), (x2, y2) = bb
    return (x2 - x1) > 10 and (y2 - y1) > 10


def has_lines(t):
    if not isinstance(t, dict):
        return False
    return any(t.get(face) for face in ("obverse", "reverse", "left", "right", "top", "bottom"))


def load_candidates():
    cdli_dict = {}
    with open(CDLI_CAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idt = str(row.get("id_text", "")).strip()
            if idt:
                cdli_dict[idt] = row

    with open(JSON_FILE, encoding="utf-8") as f:
        data = json.load(f)
    local_ids = set(f.rsplit(".", 1)[0] for f in os.listdir(IMG_DIR))

    seen = {}
    for it in data:
        pid = str(it["id"])
        if pid in seen or pid not in local_ids:
            continue
        if not (valid_bbox(it.get("bboxes")) and has_lines(it.get("text"))):
            continue
        meta = cdli_dict.get(pid, {})
        if all((meta.get(k) or "").strip() for k in ("period", "genre", "provenience", "language")):
            seen[pid] = it
    return list(seen.values())


def process(item):
    pid = str(item["id"])
    out_path = os.path.join(OUT_DIR, f"{pid}.jpg")
    if os.path.exists(out_path):
        return pid, "skip: exists", None
    if free_space_gb() < FREE_SPACE_FLOOR_GB:
        return pid, "skip: low disk space", None
    try:
        url = (item.get("img_url") or "").replace("/tn_photo/", "/photo/")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=25).read()
        # Verify it decodes as an image before trusting it, but write the
        # original bytes untouched -- no resize/re-encode this time.
        Image.open(io.BytesIO(raw)).verify()
        with open(out_path, "wb") as f:
            f.write(raw)
        return pid, "ok", {"id": pid, "bboxes": item.get("bboxes"), "bytes": len(raw)}
    except Exception as e:
        return pid, f"fail: {e}", None


def main():
    candidates = load_candidates()
    print(f"candidates (image+bbox+text+full metadata): {len(candidates)}")

    done_ids = set()
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [it for it in candidates if str(it["id"]) not in done_ids]
    print(f"already done: {len(done_ids)}, remaining: {len(todo)}")

    ok, skipped, failed = 0, 0, 0
    with open(MANIFEST_FILE, "a", encoding="utf-8") as mf, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process, it) for it in todo]
        for i, fut in enumerate(as_completed(futures)):
            pid, status, manifest_row = fut.result()
            if status == "ok":
                ok += 1
                mf.write(json.dumps(manifest_row) + "\n")
                mf.flush()
            elif status.startswith("skip"):
                skipped += 1
            else:
                failed += 1
                print(pid, status)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(todo)} done (ok={ok}, skipped={skipped}, failed={failed}), free space: {free_space_gb():.1f}GB")

    low_disk = free_space_gb() < FREE_SPACE_FLOOR_GB
    print(f"Done. ok={ok}, skipped={skipped}, failed={failed}. Output: {OUT_DIR}")
    if low_disk:
        print(f"STOPPED EARLY: free space hit the {FREE_SPACE_FLOOR_GB}GB floor -- re-run after freeing more space to continue (resumable via manifest).")


if __name__ == "__main__":
    main()
