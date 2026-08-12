"""Pull specific tablets out of CDLI's own bulk ATF dump (github.com/cdli-gh/
data, last updated 2022) that are NOT present in our CuneiML export at all --
found via cdli_cat.csv's 'photo_up' field, which flags CDLI-hosted photos
independently of what CuneiML happened to bundle (session 2026-08-11: 1486
Uruk + 642 Nimrud + 8 Ugarit candidates have a photo per CDLI's own catalogue
but zero overlap with CuneiML; only a fraction of those also have a
transliteration in the 2022 ATF dump).

Does NOT touch data/processed/hf_dataset or re-run prepare_hf_dataset.py's
main() -- that reshuffles the ENTIRE train/val/test split (random.shuffle
over all group_keys), which would invalidate every checkpoint and eval
result produced so far. Instead this assigns the new tablets their own
independent, deterministic 90/5/5 split (hash of tablet_id) and the two
downstream scripts (add_cdli_bulk_documents.py, build_vision_hf_dataset.py)
are taught to fall back to this map for ids the main text dataset doesn't
know about -- an additive patch, not a rebuild.

Only extracts 'text' (cleaned Latin transliteration) -- the 'signs' (Unicode
cuneiform) field CuneiML derives is dropped entirely by train_mbert.py before
training (see MBertMultiTask's tokenize path), so replicating CuneiML's
ATF-to-Unicode-sign mapping isn't needed here.

Output: data/interim/cdli_bulk_documents.jsonl, one row per tablet:
  {tablet_id, text, period, genre, provenience, language, split}
"""
import csv
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.prepare_hf_dataset import clean_transliteration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATF_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "cdliatf_unblocked.atf")
CDLI_CAT_CSV = os.path.join(BASE_DIR, "data", "raw", "cdli_data", "cdli_cat.csv")
TARGETS_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "targets.json")
OUT_PATH = os.path.join(BASE_DIR, "data", "interim", "cdli_bulk_documents.jsonl")

_LINE_RE = re.compile(r"^\s*[0-9]+'*\.\s*(.*)$")
_SKIP_PREFIXES = ("@", "#", "$", ">>", "=")


def split_for(tablet_id):
    """Deterministic, independent of the main pipeline's random split --
    doesn't touch or depend on any existing tablet's assignment."""
    h = int(hashlib.sha256(tablet_id.encode("utf-8")).hexdigest(), 16) % 20
    if h == 0:
        return "test"
    if h == 1:
        return "validation"
    return "train"


def parse_atf_body(body):
    lines = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(_SKIP_PREFIXES):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        content = m.group(1).strip()
        if content:
            lines.append(content)
    return lines


def main():
    targets = json.load(open(TARGETS_PATH, encoding="utf-8"))
    want_ids = set()
    for ids in targets.values():
        want_ids.update("P" + i.zfill(6) for i in ids)
    print(f"Looking for {len(want_ids)} target tablet ids in the ATF dump...")

    with open(ATF_PATH, encoding="utf-8", errors="replace") as f:
        content = f.read()
    chunks = re.split(r"(?m)^&(P\d{6})", content)

    cdli_meta = {}
    with open(CDLI_CAT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idt = str(row.get("id_text", "")).strip()
            if idt:
                cdli_meta[idt] = row

    n_written, n_empty = 0, 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i in range(1, len(chunks), 2):
            pid = chunks[i]
            if pid not in want_ids:
                continue
            body = chunks[i + 1] if i + 1 < len(chunks) else ""
            raw_lines = parse_atf_body(body)
            if not raw_lines:
                n_empty += 1
                continue
            text = " ".join(clean_transliteration(l) for l in raw_lines).strip()
            text = re.sub(r"\s+", " ", text)
            if not text:
                n_empty += 1
                continue
            meta = cdli_meta.get(pid[1:].lstrip("0") or "0", {}) or cdli_meta.get(str(int(pid[1:])), {})
            out.write(json.dumps({
                "tablet_id": pid,
                "text": text,
                "period": meta.get("period", ""),
                "genre": meta.get("genre", ""),
                "provenience": meta.get("provenience", ""),
                "language": meta.get("language", ""),
                "split": split_for(pid),
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} tablets to {OUT_PATH} ({n_empty} had no usable content lines)")


if __name__ == "__main__":
    main()
