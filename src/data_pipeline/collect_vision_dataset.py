"""Assemble a per-head, roughly-balanced RAW image dataset for the vision-
conditioned heads (provenience, genre, period -- language excluded, no
plausible visual signal, see session discussion 2026-08-05). Only classes
clearing a minimum-count floor are included (default 50, matching CuneiML's
own precedent for discarding rare classes -- Chen et al. 2023, section 5);
classes below the floor are left unsupported by images and keep working via
text as before, rather than being force-fit with too little data.

Unlike the first version of this script, a recorded bbox is NOT required to
collect an id -- the review tool (review_bboxes_gui.py) now draws boxes from
scratch for images that don't have one, instead of only correcting existing
ones. Images are saved untouched (native resolution) so bbox coordinates
(existing or hand-drawn later) never need rescaling.

Per class, up to --max_per_class images are gathered: reused from
data/raw/cuneiml/images_full if already downloaded, otherwise fetched fresh
(same disk-space safety floor as download_cuneiml_full.py). No physical
oversampling/duplication is done for classes short of the target -- handle
that with a class-balanced sampler at training time instead.

ids already marked "no_tablet" in data/bbox_corrections.jsonl (from a prior
review pass) are skipped -- that signal still holds regardless of which
class folder would otherwise include them.

Output:
  data/vision_dataset/<head>/<class>/<id>.jpg        -- one copy per class
    (a given id can legitimately appear under multiple heads/classes, e.g.
    the same tablet is both a "Girsu" provenience example and an
    "Administrative" genre example)
  data/vision_dataset/manifest.jsonl                  -- one line per UNIQUE
    id actually collected: {"id", "bbox" or null}. review_bboxes_gui.py
    reads this (not the per-class folders) so each id is reviewed once.
"""
import argparse
import csv
import io
import json
import os
import random
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_FILE = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "CuneiMLv1.2.json")
CDLI_CAT_CSV = os.path.join(BASE_DIR, "data", "raw", "cdli_data", "cdli_cat.csv")
FULL_IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "cuneiml", "images_full")
CORRECTIONS_FILE = os.path.join(BASE_DIR, "data", "bbox_corrections.jsonl")
OUT_ROOT = os.path.join(BASE_DIR, "data", "vision_dataset")
MANIFEST_FILE = os.path.join(OUT_ROOT, "manifest.jsonl")
FREE_SPACE_FLOOR_GB = 15
MAX_WORKERS = 8


def map_provenience(p):
    if not p:
        return "Unknown"
    p = p.lower()
    if "nineveh" in p or "kuyunjik" in p: return "Nineveh"
    if "umma" in p: return "Umma"
    if "girsu" in p or "tello" in p: return "Girsu"
    if "nippur" in p or "nuffar" in p: return "Nippur"
    if "puzriš-dagan" in p or "puzris-dagan" in p or "drehem" in p: return "Puzriš-Dagan"
    if "kanesh" in p or "kültepe" in p: return "Kanesh"
    if "aššur" in p or "assur" in p or "ashur" in p or ("qal" in p and "sherqat" in p): return "Assur"
    if "uruk" in p or "warka" in p: return "Uruk"
    if p.startswith("ur ") or p.startswith("ur(") or "tell muqayyar" in p or p == "ur": return "Ur"
    if "ugarit" in p or "ras shamra" in p: return "Ugarit"
    if "sippar" in p: return "Sippar"
    if "nimrud" in p or "kalhu" in p: return "Nimrud"
    return "Unknown"


def map_genre(g):
    if not g:
        return "Unknown"
    g = g.lower()
    if "administrative" in g: return "Administrative"
    if "lexical" in g: return "Lexical"
    if "royal" in g or "monumental" in g: return "Royal Inscriptions"
    if any(x in g for x in ["literary", "scholarly", "astrolog", "astronomical", "omen", "school",
                             "ritual", "incantation", "extispicy", "mathematical", "scientific",
                             "technical procedure", "prayer"]):
        return "Literary & Scholarly"
    if any(x in g for x in ["legal", "treaty", "grant"]): return "Legal"
    if "letter" in g: return "Letters"
    return "Unknown"


def map_period(p):
    if not p:
        return "Unknown"
    p = p.lower()
    if "neo-assyrian" in p or "neo assyrian" in p: return "Neo-Assyrian"
    if "ur iii" in p: return "Ur III"
    if "old assyrian" in p: return "Old Assyrian"
    if "old babylonian" in p: return "Old Babylonian"
    if "middle assyrian" in p: return "Middle Assyrian"
    if "middle babylonian" in p: return "Middle Babylonian"
    if "neo-babylonian" in p or "late babylonian" in p: return "Neo-Babylonian"
    if any(x in p for x in ["ed iii", "ed i-ii", "early dynastic", "old akkadian", "lagash ii",
                             "ebla", "uruk iii", "uruk iv"]):
        return "Third Millennium"
    if any(x in p for x in ["seleucid", "achaemenid", "hellenistic"]): return "Late Antiquity"
    return "Unknown"


# Classes below cleared the >=50-example floor under the relaxed (single-
# field) filter, per the session's counting pass (2026-08-05). Ugarit/
# Nimrud/Uruk (provenience), Neo-Babylonian/Late Antiquity (period) excluded.
HEADS = {
    "provenience": {
        "field": "provenience", "mapper": map_provenience,
        "classes": ["Umma", "Puzriš-Dagan", "Girsu", "Nippur", "Kanesh", "Ur", "Nineveh", "Sippar", "Assur"],
    },
    "genre": {
        "field": "genre", "mapper": map_genre,
        "classes": ["Administrative", "Letters", "Legal", "Literary & Scholarly", "Royal Inscriptions", "Lexical"],
    },
    "period": {
        "field": "period", "mapper": map_period,
        "classes": ["Ur III", "Third Millennium", "Old Babylonian", "Old Assyrian", "Neo-Assyrian", "Middle Babylonian", "Middle Assyrian"],
    },
}


def free_space_gb():
    return shutil.disk_usage(BASE_DIR).free / (1024 ** 3)


def load_corrections():
    corrections = {}
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    corrections[row["id"]] = row
                except Exception:
                    pass
    return corrections


def load_all_candidates():
    cdli_dict = {}
    with open(CDLI_CAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idt = str(row.get("id_text", "")).strip()
            if idt:
                cdli_dict[idt] = row

    with open(JSON_FILE, encoding="utf-8") as f:
        data = json.load(f)

    def has_lines(t):
        if not isinstance(t, dict):
            return False
        return any(t.get(face) for face in ("obverse", "reverse", "left", "right", "top", "bottom"))

    seen = {}
    for it in data:
        pid = str(it["id"])
        if pid in seen or not it.get("img_url") or not has_lines(it.get("text")):
            continue
        seen[pid] = (it, cdli_dict.get(pid, {}))
    return seen


def fetch_and_save(pid, img_url, out_path):
    local_path = os.path.join(FULL_IMG_DIR, f"{pid}.jpg")
    try:
        if os.path.exists(local_path):
            shutil.copy(local_path, out_path)
            return pid, "ok"
        if free_space_gb() < FREE_SPACE_FLOOR_GB:
            return pid, "skip: low disk space"
        url = img_url.replace("/tn_photo/", "/photo/")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=25).read()
        Image.open(io.BytesIO(raw)).verify()
        with open(out_path, "wb") as f:
            f.write(raw)
        return pid, "ok"
    except Exception as e:
        return pid, f"fail: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int, default=300)
    parser.add_argument("--min_class_count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corrections = load_corrections()
    print(f"loaded {len(corrections)} manual bbox corrections")
    all_candidates = load_all_candidates()
    print(f"total raw candidates (image_url+text present): {len(all_candidates)}")

    random.seed(args.seed)
    selected_ids = {}  # pid -> (img_url, bbox-or-None)

    for head_name, cfg in HEADS.items():
        by_class = defaultdict(list)
        for pid, (it, meta) in all_candidates.items():
            corr = corrections.get(pid)
            if corr and corr["status"] == "no_tablet":
                continue
            raw_val = meta.get(cfg["field"], "")
            cls = cfg["mapper"](raw_val)
            if cls in cfg["classes"]:
                by_class[cls].append(pid)

        print(f"\n=== {head_name} ===")
        for cls in cfg["classes"]:
            pool = by_class.get(cls, [])
            print(f"  {cls}: {len(pool)} available" + (" (below floor, skipped)" if len(pool) < args.min_class_count else ""))
            if len(pool) < args.min_class_count:
                continue
            target = min(args.max_per_class, len(pool))
            out_dir = os.path.join(OUT_ROOT, head_name, cls.replace("/", "-"))
            os.makedirs(out_dir, exist_ok=True)

            pool_set = set(pool)
            already = {f.rsplit(".", 1)[0] for f in os.listdir(out_dir) if f.endswith(".jpg")} & pool_set
            # backfill queue: pool members never yet attempted (e.g. dead CDLI
            # links found on a prior run stay excluded since they're not on
            # disk but we don't retry the exact same ones -- draw fresh ones
            # from the rest of the pool instead until target is met or the
            # pool runs out).
            queue = [pid for pid in pool if pid not in already]
            random.shuffle(queue)

            have = set(already)
            ok, failed = 0, 0
            while len(have) < target and queue:
                batch, queue = queue[: MAX_WORKERS * 4], queue[MAX_WORKERS * 4 :]
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futures = {ex.submit(fetch_and_save, pid, all_candidates[pid][0]["img_url"], os.path.join(out_dir, f"{pid}.jpg")): pid
                               for pid in batch}
                    for fut in as_completed(futures):
                        pid, status = fut.result()
                        if status == "ok":
                            ok += 1
                            have.add(pid)
                        else:
                            failed += 1

            for pid in have:
                it = all_candidates[pid][0]
                corr = corrections.get(pid)
                bbox = corr["bbox"] if corr else it.get("bboxes")
                selected_ids[pid] = (it["img_url"], bbox)
            print(f"    -> {len(have)}/{target} on disk ({ok} new, {len(already)} already present, {failed} attempts failed) -> {out_dir}")

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        for pid, (url, bbox) in selected_ids.items():
            f.write(json.dumps({"id": pid, "bbox": bbox}) + "\n")
    print(f"\nDone. {len(selected_ids)} unique ids collected. Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
