"""Balance backfill (session 2026-08-12): bring under-represented provenience
classes up toward the current top tier (~650 images, matching Umma/Kanesh/
Nippur) using the same CDLI-catalog-photo_up + (ATF dump | eBL) text
cross-reference already used for the Uruk/Nimrud rescue. Per-class caps below
were computed from that cross-reference and confirmed with the user before
running -- NOT "grab everything available" (Nineveh alone has 14k+
candidates; downloading and hand-annotating bboxes for all of them isn't
feasible for a thesis timeline).

Writes data/interim/balance_documents.jsonl (text, same schema as
prepare_cdli_bulk.py's output) and downloads the matching photos into
data/vision_dataset/provenience/<class>/, appending manifest.jsonl for
review_bboxes_gui.py exactly like the earlier backfills.
"""
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.collect_vision_dataset import map_provenience, JSON_FILE
from src.data_pipeline.prepare_cdli_bulk import parse_atf_body, split_for
from src.data_pipeline.prepare_hf_dataset import clean_transliteration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CDLI_CAT_CSV = os.path.join(BASE_DIR, "data", "raw", "cdli_data", "cdli_cat.csv")
ATF_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "cdliatf_unblocked.atf")
EBL_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "ebl_fragments.json")
VISION_BASE = os.path.join(BASE_DIR, "data", "vision_dataset", "provenience")
MANIFEST = os.path.join(BASE_DIR, "data", "vision_dataset", "manifest.jsonl")
OUT_DOCS = os.path.join(BASE_DIR, "data", "interim", "balance_documents.jsonl")

# Confirmed with the user (session 2026-08-12): target ~650, capped by real
# availability with text.
CAPS = {
    "Nineveh": 242,
    "Puzriš-Dagan": 257,
    "Girsu": 178,
    "Ur": 163,
    "Nippur": 50,
    "Sippar": 114,
    "Assur": 121,
}


def already_have_pids():
    have = set()
    data = json.load(open(JSON_FILE, encoding="utf-8"))
    have |= {str(it["id"]) for it in data if it.get("img_url")}
    for fname in ("cdli_bulk_documents.jsonl", "ebl_bulk_documents.jsonl"):
        path = os.path.join(BASE_DIR, "data", "interim", fname)
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                have.add(str(int(json.loads(line)["tablet_id"][1:])))
    return have


def build_atf_text_index():
    content = open(ATF_PATH, encoding="utf-8", errors="replace").read()
    chunks = re.split(r"(?m)^&(P\d{6})", content)
    idx = {}
    for i in range(1, len(chunks), 2):
        idx[chunks[i]] = chunks[i + 1] if i + 1 < len(chunks) else ""
    return idx


def build_ebl_text_index():
    frags = json.load(open(EBL_PATH, encoding="utf-8"))
    idx = {}
    for f in frags:
        cdli = (f.get("externalNumbers") or {}).get("cdliNumber")
        if cdli:
            idx[cdli] = f.get("atf", "")
    return idx


def fetch_photo(pid_numeric, tablet_id, cls):
    out_dir = os.path.join(VISION_BASE, cls)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pid_numeric}.jpg")
    if os.path.exists(out_path):
        return pid_numeric, "already"
    url = f"https://cdli.mpiwg-berlin.mpg.de/dl/photo/{tablet_id}.jpg"
    last = "unknown"
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            Image.open(io.BytesIO(raw)).verify()
            with open(out_path, "wb") as f:
                f.write(raw)
            return pid_numeric, "ok"
        except Exception as e:
            last = str(e)
            time.sleep(2)
    return pid_numeric, f"fail: {last}"


def main():
    have = already_have_pids()
    print(f"already have {len(have)} pids across all sources")

    cdli_meta_by_id = {}
    by_class = {}
    with open(CDLI_CAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idt = str(row.get("id_text", "")).strip()
            if idt:
                cdli_meta_by_id[idt] = row
            if not row.get("photo_up", "").strip():
                continue
            cls = map_provenience(row.get("provenience", ""))
            if cls in CAPS and idt and idt not in have:
                by_class.setdefault(cls, []).append(idt)

    print("building ATF text index (large file, ~1 min)...")
    atf_idx = build_atf_text_index()
    print("building eBL text index...")
    ebl_idx = build_ebl_text_index()

    selected = []  # (pid_numeric, tablet_id, cls, raw_text_source)
    for cls, cap in CAPS.items():
        pool = by_class.get(cls, [])
        n = 0
        for idt in pool:
            if n >= cap:
                break
            pid_p = "P" + idt.zfill(6)
            body = atf_idx.get(pid_p)
            source = "atf"
            if body is None:
                body = ebl_idx.get(pid_p)
                source = "ebl"
            if body is None:
                continue
            raw_lines = parse_atf_body(body)
            if not raw_lines:
                continue
            text = re.sub(r"\s+", " ", " ".join(clean_transliteration(l) for l in raw_lines)).strip()
            if not text:
                continue
            selected.append((idt, pid_p, cls, text, source))
            n += 1
        print(f"{cls}: selected {n}/{cap}")

    # 1. Write documents jsonl
    with open(OUT_DOCS, "w", encoding="utf-8") as out:
        for idt, pid_p, cls, text, source in selected:
            meta = cdli_meta_by_id.get(idt, {})
            out.write(json.dumps({
                "tablet_id": pid_p,
                "text": text,
                "period": meta.get("period", ""),
                "genre": meta.get("genre", ""),
                "provenience": meta.get("provenience", ""),
                "language": meta.get("language", ""),
                "split": split_for(pid_p),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} rows to {OUT_DOCS}")

    # 2. Download photos, gently (3 workers, matches earlier session finding
    # that CDLI's photo endpoint chokes on 8 concurrent connections)
    existing_manifest = set()
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            existing_manifest.add(json.loads(line)["id"])

    tasks = [(idt, pid_p, cls) for idt, pid_p, cls, _, _ in selected if idt not in existing_manifest]
    print(f"downloading {len(tasks)} photos...")
    ok, failed = 0, []
    with open(MANIFEST, "a", encoding="utf-8") as mf:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(fetch_photo, idt, pid_p, cls): (idt, cls) for idt, pid_p, cls in tasks}
            for fut in as_completed(futs):
                pid_numeric, status = fut.result()
                if status in ("ok", "already"):
                    ok += 1
                    mf.write(json.dumps({"id": pid_numeric, "bbox": None}) + "\n")
                else:
                    failed.append((pid_numeric, status))
                if (ok + len(failed)) % 50 == 0:
                    print(f"progress: {ok} ok, {len(failed)} failed, {len(tasks) - ok - len(failed)} left", flush=True)

    print(f"DONE. ok={ok} failed={len(failed)}")
    for f in failed[:20]:
        print(" fail:", f)


if __name__ == "__main__":
    main()
