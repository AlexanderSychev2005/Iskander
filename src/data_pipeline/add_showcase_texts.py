"""Pull well-known named works (Gilgamesh, Enuma Elish, Atrahasis, Hammurabi's
code) from eBL into the corpus, forced into the TEST split regardless of
prepare_cdli_bulk.py's normal hash-based assignment (session 2026-08-12):
these exist specifically as recognizable qualitative-demo examples for the
thesis/paper, so they must be held out of training -- a model that
memorized Gilgamesh during training and then "restores" it on a masked-
span demo isn't demonstrating anything.

Same verified path as reprocess_bulk_documents.py: CuneiML's own ATF parser,
dedup against the main corpus's sign-string keys. Photos fetched for the
subset CDLI's catalog flags as having one.
"""
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.collect_vision_dataset import map_provenience
from src.data_pipeline.cuneiform_unicode import atf_to_lines
from src.data_pipeline.prepare_hf_dataset import clean_transliteration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EBL_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "ebl_fragments.json")
CDLI_CAT_CSV = os.path.join(BASE_DIR, "data", "raw", "cdli_data", "cdli_cat.csv")
COMBINED_PATH = os.path.join(BASE_DIR, "data", "processed", "combined_unique.jsonl")
OUT_DOCS_PATH = os.path.join(BASE_DIR, "data", "interim", "showcase_documents.jsonl")
VISION_BASE = os.path.join(BASE_DIR, "data", "vision_dataset", "provenience")
MANIFEST = os.path.join(BASE_DIR, "data", "vision_dataset", "manifest.jsonl")

WORKS = {
    "Gilgamesh": "gilgame",
    "Enuma Elish": "enuma eli",
    "Atrahasis": "atrahasis",
    "Hammurabi": "hammurabi",
}


def load_existing_sign_keys():
    keys = set()
    with open(COMBINED_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = "".join(r.get("signs") or []).strip()
            if k:
                keys.add(k)
    return keys


def main():
    frags = json.load(open(EBL_PATH, encoding="utf-8"))
    cdli_meta = {}
    with open(CDLI_CAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idt = str(row.get("id_text", "")).strip()
            if idt:
                cdli_meta[idt] = row

    existing_keys = load_existing_sign_keys()
    seen_tids = set()

    n_written, n_empty, n_photo_targets = 0, 0, 0
    photo_targets = []
    with open(OUT_DOCS_PATH, "w", encoding="utf-8") as out:
        for work, needle in WORKS.items():
            for f in frags:
                if needle not in json.dumps(f, ensure_ascii=False).lower():
                    continue
                cdli = (f.get("externalNumbers") or {}).get("cdliNumber")
                tid = cdli or f"ebl:{f.get('_id')}"
                if tid in seen_tids:
                    continue
                seen_tids.add(tid)

                parsed, misses, tok = atf_to_lines(f.get("atf", ""))
                tablet_signs, tablet_texts = [], []
                for ln in parsed:
                    signs = [s for s in ln["signs"] if s and s != "<S>"]
                    if len(signs) < 2:
                        continue
                    sign_key = "".join(signs)
                    if sign_key in existing_keys:
                        continue
                    existing_keys.add(sign_key)
                    tablet_signs.extend(signs)
                    text = clean_transliteration(ln["raw"])
                    if text:
                        tablet_texts.append(text)
                text = re.sub(r"\s+", " ", " ".join(tablet_texts)).strip()
                if not text or not tablet_signs:
                    n_empty += 1
                    continue

                idt = cdli[1:].lstrip("0") if cdli else None
                meta = (cdli_meta.get(idt) or cdli_meta.get(str(int(idt))) if idt else {}) or {}
                out.write(json.dumps({
                    "tablet_id": tid,
                    "work": work,
                    "text": text,
                    "signs": tablet_signs,
                    "period": meta.get("period", ""),
                    "genre": meta.get("genre", ""),
                    "provenience": meta.get("provenience", ""),
                    "language": meta.get("language", ""),
                    "split": "test",  # forced -- see module docstring
                }, ensure_ascii=False) + "\n")
                n_written += 1

                if cdli and meta.get("photo_up", "").strip():
                    photo_targets.append((idt, cdli, map_provenience(meta.get("provenience", ""))))
                    n_photo_targets += 1

    print(f"wrote {n_written} showcase documents ({n_empty} empty after parsing/dedup)")
    print(f"photo targets: {n_photo_targets}")

    existing_manifest = set()
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            existing_manifest.add(json.loads(line)["id"])

    ok, failed = 0, []
    with open(MANIFEST, "a", encoding="utf-8") as mf:
        for idt, tid, cls in photo_targets:
            if idt in existing_manifest:
                continue
            out_dir = os.path.join(VISION_BASE, cls if cls != "Unknown" else "Showcase")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{idt}.jpg")
            url = f"https://cdli.mpiwg-berlin.mpg.de/dl/photo/{tid}.jpg"
            got = False
            for _ in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    raw = urllib.request.urlopen(req, timeout=60).read()
                    Image.open(io.BytesIO(raw)).verify()
                    with open(out_path, "wb") as fimg:
                        fimg.write(raw)
                    got = True
                    break
                except Exception as e:
                    last = str(e)
                    time.sleep(2)
            if got:
                ok += 1
                mf.write(json.dumps({"id": idt, "bbox": None}) + "\n")
            else:
                failed.append((idt, last))

    print(f"photos: ok={ok} failed={len(failed)}")
    for f in failed:
        print(" fail:", f)


if __name__ == "__main__":
    main()
