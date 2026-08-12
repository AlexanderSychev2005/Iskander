"""Text-only balance backfill (session 2026-08-12): bring under-represented
period/genre/language classes up toward a mid-tier target, using the same
CDLI-catalog metadata + (ATF dump | eBL) text cross-reference as
backfill_balance.py did for provenience -- but no images needed here, so no
manual bbox review gate. Goes straight through the same verified path as
reprocess_bulk_documents.py: CuneiML's own ATF->Unicode-sign parser, dedup
against the main corpus's sign-string keys, clean_transliteration for text.

Per-class caps confirmed with the user (session 2026-08-12).

Writes data/interim/text_balance_documents.jsonl, same schema as the other
bulk-backfill interim files, ready for add_cdli_bulk_documents.py.
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.cuneiform_unicode import atf_to_lines
from src.data_pipeline.prepare_cdli_bulk import split_for
from src.data_pipeline.prepare_hf_dataset import (
    clean_transliteration, map_genre, map_language, map_period,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CDLI_CAT_CSV = os.path.join(BASE_DIR, "data", "raw", "cdli_data", "cdli_cat.csv")
ATF_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "cdliatf_unblocked.atf")
EBL_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "ebl_fragments.json")
COMBINED_PATH = os.path.join(BASE_DIR, "data", "processed", "combined_unique.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "data", "interim", "text_balance_documents.jsonl")

# task -> {class: cap}, confirmed with the user
CAPS = {
    "period": {"Old Assyrian": 1080, "Neo-Babylonian": 810, "Late Antiquity": 443},
    "genre": {"Letters": 1905, "Legal": 1316},
    "language": {"Bilingual": 383, "Peripheral/Other": 1465},
}
MAPPERS = {"period": map_period, "genre": map_genre, "language": map_language}


def already_have_ids():
    have = set()
    for fname in os.listdir(os.path.join(BASE_DIR, "data", "interim")):
        if fname.endswith("_documents.jsonl"):
            path = os.path.join(BASE_DIR, "data", "interim", fname)
            for line in open(path, encoding="utf-8"):
                have.add(json.loads(line)["tablet_id"])
    from datasets import load_from_disk
    docs = load_from_disk(os.path.join(BASE_DIR, "data", "processed", "hf_dataset"))
    for split in ("train", "validation", "test"):
        have.update(t for t in docs[split]["tablet_id"] if t)
    return have


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


def build_atf_body_index():
    content = open(ATF_PATH, encoding="utf-8", errors="replace").read()
    chunks = re.split(r"(?m)^&(P\d{6})", content)
    return {chunks[i]: (chunks[i + 1] if i + 1 < len(chunks) else "") for i in range(1, len(chunks), 2)}


def build_ebl_atf_index():
    frags = json.load(open(EBL_PATH, encoding="utf-8"))
    idx = {}
    for f in frags:
        cdli = (f.get("externalNumbers") or {}).get("cdliNumber")
        if cdli:
            idx[cdli] = f.get("atf", "")
    return idx


def parse_tablet(tid, body, existing_keys):
    parsed, misses, tok = atf_to_lines(body)
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
    return text, tablet_signs, sum(misses.values()), tok


def main():
    have = already_have_ids()
    print(f"already have {len(have)} tablet ids across the corpus")

    atf_idx = build_atf_body_index()
    ebl_idx = build_ebl_atf_index()
    existing_keys = load_existing_sign_keys()

    cdli_meta_by_id = {}
    by_task_class = {task: {cls: [] for cls in caps} for task, caps in CAPS.items()}
    with open(CDLI_CAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idt = str(row.get("id_text", "")).strip()
            if not idt:
                continue
            tid = "P" + idt.zfill(6)
            if tid in have:
                continue
            cdli_meta_by_id[idt] = row
            for task, caps in CAPS.items():
                v = MAPPERS[task](row.get(task, ""))
                if v in caps:
                    by_task_class[task][v].append(idt)

    selected = {}  # idt -> (tid, meta_row)
    for task, caps in CAPS.items():
        for cls, cap in caps.items():
            pool = by_task_class[task][cls]
            n = 0
            for idt in pool:
                if n >= cap:
                    break
                if idt in selected:
                    continue
                tid = "P" + idt.zfill(6)
                body = atf_idx.get(tid, ebl_idx.get(tid))
                if body is None:
                    continue
                selected[idt] = tid
                n += 1
            print(f"{task}/{cls}: selected up to {n}/{cap}")

    print(f"total candidate tablets to parse: {len(selected)}")
    n_written, n_empty, n_miss, n_tok = 0, 0, 0, 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for idt, tid in selected.items():
            body = atf_idx.get(tid, ebl_idx.get(tid))
            text, signs, misses, tok = parse_tablet(tid, body, existing_keys)
            n_miss += misses
            n_tok += tok
            if not text or not signs:
                n_empty += 1
                continue
            meta = cdli_meta_by_id.get(idt, {})
            out.write(json.dumps({
                "tablet_id": tid,
                "text": text,
                "signs": signs,
                "period": meta.get("period", ""),
                "genre": meta.get("genre", ""),
                "provenience": meta.get("provenience", ""),
                "language": meta.get("language", ""),
                "split": split_for(tid),
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"wrote {n_written} tablets ({n_empty} empty after parsing/dedup)")
    print(f"sign-resolution miss rate: {n_miss}/{n_tok} = {100*n_miss/max(n_tok,1):.1f}%")


if __name__ == "__main__":
    main()
