"""Qualitative side-by-side demo: for a handful of real test-split tablets,
show the masked input and what each final checkpoint predicts, text-only vs
vision (provenience) model on the identical input -- complements
evaluate_mbert.py's aggregate/per-class numbers with concrete examples for
the diploma writeup.

Masking here always shows [MASK] at every chosen position (not the real
80/10/10 BERT recipe DataCollatorForLanguageModeling uses during actual
training/eval) -- clearer to read, and the reported metrics still come from
evaluate_mbert.py's real collator, not from this file. Both models see the
exact same masked positions (one shared RNG draw per example) so restoration
quality is comparable position-by-position, not just in aggregate.

The image only ever reaches provenience_head (see MBertMultiTask.forward --
mlm_logits is computed straight from BERT's own hidden states, before any
image concatenation), so the two models' MLM restoration differs only
because they're separately trained weights, not because of the image
directly. The place the image can actually change an answer is the
provenience row in each example's metadata table.

--fetch_cdli_info (optional, needs network) additionally pulls, per tablet,
from whichever of eBL/CDLI actually has it: a one-line description (genre/
period/collection/publication), and a line-by-line table of cuneiform +
transliteration + English translation -- a genuine per-line parse of the
tablet's own raw ATF (local bulk dump / eBL export first, live API as
fallback), NOT derived from this project's own flattened 'text'/'signs'
columns, which have no per-line structure left after corpus-wide merging.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training.train_mbert import (
    MBertMultiTask, mark_damage_signals, build_tablet_image_index_from_hf, IMG_TRANSFORM_EVAL,
)
from src.data_pipeline.review_bboxes_gui import build_path_index
from src.data_pipeline.cuneiform_unicode import atf_to_lines, _FACE_KEYS

TASKS = ["period", "genre", "language", "provenience"]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORD_CACHE_PATH = os.path.join(BASE_DIR, "data", "interim", "cdli_record_cache.json")
CDLI_ATF_DUMP_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "cdliatf_unblocked.atf")
EBL_LOCAL_PATH = os.path.join(BASE_DIR, "data", "raw", "cdli_bulk", "ebl_fragments.json")
SHOWCASE_PATH = os.path.join(BASE_DIR, "data", "interim", "showcase_documents.jsonl")


# ---------------------------------------------------------------------------
# Local-source indexes (built once, no network) -- checked before falling
# back to a live API call for a given tablet_id.
# ---------------------------------------------------------------------------

def build_cdli_atf_dump_index():
    if not os.path.exists(CDLI_ATF_DUMP_PATH):
        return {}
    content = open(CDLI_ATF_DUMP_PATH, encoding="utf-8", errors="replace").read()
    chunks = re.split(r"(?m)^&(P\d{6})", content)
    idx = {}
    for i in range(1, len(chunks), 2):
        idx["P" + chunks[i]] = chunks[i + 1] if i + 1 < len(chunks) else ""
    return idx


def build_ebl_local_index():
    """(by_cdli, by_id): eBL fragments keyed by their CDLI cross-reference
    (for P###### tablet_ids whose actual transliteration source is eBL, not
    CDLI -- true for most literary/library fragments, see session notes on
    K.3375) and by eBL's own fragment id (for "ebl:<id>" tablet_ids, the
    convention add_showcase_texts.py uses when no CDLI number exists)."""
    if not os.path.exists(EBL_LOCAL_PATH):
        return {}, {}
    frags = json.load(open(EBL_LOCAL_PATH, encoding="utf-8"))
    by_cdli, by_id = {}, {}
    for f in frags:
        cdli = (f.get("externalNumbers") or {}).get("cdliNumber")
        if cdli:
            by_cdli[cdli] = f
        eid = f.get("_id")
        if eid:
            by_id[eid] = f
    return by_cdli, by_id


def load_showcase_work_map():
    if not os.path.exists(SHOWCASE_PATH):
        return {}
    out = {}
    with open(SHOWCASE_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("work"):
                out[r["tablet_id"]] = r["work"]
    return out


# ---------------------------------------------------------------------------
# Record cache (disk) -- one live network call per new tablet_id, ever.
# ---------------------------------------------------------------------------

def load_record_cache():
    if os.path.exists(RECORD_CACHE_PATH):
        with open(RECORD_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_record_cache(cache):
    os.makedirs(os.path.dirname(RECORD_CACHE_PATH), exist_ok=True)
    with open(RECORD_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_cdli_json(numeric_id):
    req = urllib.request.Request(
        f"https://cdli.earth/artifacts/{numeric_id}",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    raw = urllib.request.urlopen(req, timeout=20).read()
    data = json.loads(raw)
    return data[0] if isinstance(data, list) else data


def fetch_ebl_json(museum_number):
    for base in ("https://www.ebl.lmu.de/api/fragments/", "https://www.ebl.uni-muenchen.de/api/fragments/"):
        try:
            req = urllib.request.Request(base + museum_number, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            raw = urllib.request.urlopen(req, timeout=20).read()
            return json.loads(raw)
        except Exception:
            continue
    return None


def describe_from_cdli(rec):
    parts = []
    if rec.get("designation"):
        parts.append(rec["designation"])
    period = (rec.get("period") or {}).get("name")
    genres = rec.get("genres") or []
    genre = (genres[0].get("genre") or {}).get("genre") if genres else None
    prov = (rec.get("provenience") or {}).get("provenience")
    bits = [b for b in [genre, period, prov] if b]
    if bits:
        parts.append(", ".join(bits))
    colls = rec.get("collections") or []
    if colls:
        cname = (colls[0].get("collection") or {}).get("collection")
        if cname:
            parts.append(cname)
    pubs = rec.get("publications") or []
    primary = next((p for p in pubs if p.get("publication_type") == "primary"), None)
    if primary:
        pub = primary.get("publication") or {}
        title = pub.get("title") or pub.get("designation")
        authors = pub.get("authors") or []
        last = authors[0]["author"]["last"] if authors and authors[0].get("author") else None
        cite = ", ".join(b for b in [last, pub.get("year")] if b)
        if title:
            parts.append(f"published in {title}" + (f" ({cite})" if cite else ""))
    return " -- ".join(parts) if parts else None


def describe_from_ebl(frag, work=None):
    parts = []
    if work:
        parts.append(f"{work} fragment")
    genres = frag.get("genres") or []
    if genres:
        cat = (genres[0].get("category") or [])
        if cat and not work:
            parts.append(" > ".join(cat))
        elif cat:
            parts.append(cat[-1])
    script = frag.get("script") or {}
    if script.get("period"):
        parts.append(script["period"])
    museum = frag.get("museum")
    if museum:
        parts.append(museum)
    return " -- ".join(parts) if parts else None


def parse_translations_by_line(raw_atf):
    """(face, line_num) -> English translation, from #tr.en: comment lines
    immediately following the numbered content line they annotate. Mirrors
    atf_to_lines's own face-tracking/line-number rules exactly so its output
    can be joined to this by the same (face, num) key."""
    translations = {}
    curr_face = "default"
    pending = None
    sep = "\n"
    if "\\n" in raw_atf and "\n" not in raw_atf:
        sep = "\\n"
    for line in raw_atf.split(sep):
        line = line.strip()
        if not line:
            continue
        if line.startswith("@"):
            key = line[1:].strip().strip("?")
            if key in _FACE_KEYS:
                curr_face = key
            pending = None
            continue
        if line.startswith("#tr.en:"):
            if pending:
                translations[pending] = line[len("#tr.en:"):].strip()
            continue
        if line.startswith(("#", ">>", "$", "&")):
            continue
        m = re.match(r"(\S+)\.\s+(.*)", line)
        pending = (curr_face, m.group(1)) if m else None
    return translations


def build_line_table(raw_atf):
    """[{'face','num','signs','translit','translation'}, ...] -- a real
    per-line parse (not this project's own flattened, whole-document
    'text'/'signs' columns, which lose line boundaries in the corpus merge).
    Returns [] if raw_atf is empty/unparseable."""
    if not raw_atf:
        return []
    parsed, _misses, _tok = atf_to_lines(raw_atf)
    tr_by_line = parse_translations_by_line(raw_atf)
    by_key = {}
    order = []
    for ln in parsed:
        signs = [s for s in ln["signs"] if s and s != "<S>"]
        if not signs and not ln["raw"].strip():
            continue
        key = (ln["face"], ln["num"])
        row = {
            "face": ln["face"], "num": ln["num"],
            "signs": " ".join(signs), "translit": ln["raw"],
            "translation": tr_by_line.get(key),
        }
        # Some multi-piece joins repeat the same (face, num) for a different
        # physical fragment CDLI's ATF doesn't mark with a face/object tag
        # atf_to_lines recognizes -- keep the richer entry (more signs), not
        # the first one, so a stub "x" line doesn't shadow the real content.
        if key not in by_key or len(row["signs"]) > len(by_key[key]["signs"]):
            if key not in by_key:
                order.append(key)
            by_key[key] = row
    return [by_key[k] for k in order]


def fetch_record(tablet_id, cache, cdli_dump_idx, ebl_by_cdli, ebl_by_id, work_map):
    """{'description': str|None, 'lines': [...]} -- eBL preferred when it has
    this tablet (richer genre/script metadata, and it's the actual
    transliteration source for most literary fragments in this corpus);
    else CDLI's own record; else nothing. Local sources checked before any
    network call; live APIs only as fallback for coverage gaps in the local
    snapshots."""
    if tablet_id in cache:
        return cache[tablet_id]

    result = {"description": None, "lines": []}
    work = work_map.get(tablet_id)
    ebl_frag = ebl_by_cdli.get(tablet_id) or (ebl_by_id.get(tablet_id.split(":", 1)[1]) if tablet_id.startswith("ebl:") else None)

    if ebl_frag is not None:
        result["description"] = describe_from_ebl(ebl_frag, work)
        result["lines"] = build_line_table(ebl_frag.get("atf", ""))
    elif tablet_id.startswith("ebl:"):
        museum_number = tablet_id.split(":", 1)[1]
        data = fetch_ebl_json(museum_number)
        time.sleep(0.3)
        if data:
            result["description"] = describe_from_ebl(data, work)
            result["lines"] = build_line_table(data.get("atf", ""))
    elif tablet_id.startswith("P") and tablet_id[1:].isdigit():
        numeric = tablet_id[1:]
        raw_atf = cdli_dump_idx.get(tablet_id)
        cdli_rec = None
        try:
            cdli_rec = fetch_cdli_json(numeric)
            time.sleep(0.3)
        except Exception:
            pass
        if cdli_rec:
            result["description"] = describe_from_cdli(cdli_rec) or (f"{work} fragment" if work else None)
            atf = (cdli_rec.get("inscription") or {}).get("atf") or raw_atf
            result["lines"] = build_line_table(atf)
        elif raw_atf:
            result["lines"] = build_line_table(raw_atf)
            if work:
                result["description"] = f"{work} fragment"

    cache[tablet_id] = result
    return result


def load_model(checkpoint, model_name, num_labels, use_image, vision_init):
    model = MBertMultiTask(model_name, num_period=num_labels["period"], num_genre=num_labels["genre"],
                            num_language=num_labels["language"], num_provenience=num_labels["provenience"],
                            use_image=use_image, vision_init=vision_init)
    state_dict = load_file(os.path.join(checkpoint, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mask_positions(input_ids, banned_ids, mlm_probability, rng):
    eligible = [i for i, t in enumerate(input_ids) if t not in banned_ids]
    n_mask = max(1, round(len(eligible) * mlm_probability))
    return sorted(rng.sample(eligible, min(n_mask, len(eligible))))


def topk_at(logits, position, banned, tokenizer, k=3):
    row = logits[0, position].clone()
    row[list(banned)] = float("-inf")
    top = torch.topk(row, k=k).indices.tolist()
    return [tokenizer.convert_ids_to_tokens([t])[0] for t in top]


def format_metadata_table(label_configs, truth, text_pred, vision_pred):
    lines = ["| head | ground truth | text-only prediction | vision prediction |", "|---|---|---|---|"]
    for task in TASKS:
        names = label_configs[task]["labels"]
        t_idx, t_conf = text_pred[task]
        v_idx, v_conf = vision_pred[task]
        truth_name = names[truth[task]] if truth[task] != -100 and truth[task] < len(names) else "(no label)"
        t_name = f"{names[t_idx]} ({t_conf:.2f})"
        v_name = f"{names[v_idx]} ({v_conf:.2f})"
        marker = " **<- differs**" if t_idx != v_idx else ""
        lines.append(f"| {task} | {truth_name} | {t_name} | {v_name}{marker} |")
    return "\n".join(lines)


def format_side_by_side(safe_id, tablet_id, has_crop, has_full, line_rows):
    """Raw HTML (not a markdown table nested in one) -- photo(s) on the
    left, a real per-line cuneiform/transliteration/translation table on the
    right, so it renders identically everywhere without depending on a
    renderer's markdown-inside-HTML support."""
    img_cell = ""
    if has_crop:
        img_cell += f'<img src="demo_images/{safe_id}.jpg" width="220"><br><sub>model input (224x224)</sub><br><br>'
    if has_full:
        img_cell += f'<img src="demo_images/{safe_id}_full.jpg" width="220"><br><sub>full photo (reference)</sub>'
    if not img_cell:
        return None

    if line_rows:
        table_rows = "".join(
            f"<tr><td>{r['num']}</td><td>{r['face']}</td><td>{r['signs']}</td>"
            f"<td>{r['translit']}</td><td>{r['translation'] or '&mdash;'}</td></tr>"
            for r in line_rows
        )
        line_cell = (
            '<table><tr><th>#</th><th>face</th><th>cuneiform</th><th>transliteration</th><th>translation</th></tr>'
            + table_rows + "</table>"
        )
    else:
        line_cell = "<sub>(no line-by-line ATF available for this tablet)</sub>"

    return (
        f'<table><tr><td valign="top" width="240">{img_cell}</td>'
        f'<td valign="top">{line_cell}</td></tr></table>\n'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_checkpoint", type=str, default=r"C:\Programming\akkadian\checkpoints_final_text\final_model")
    parser.add_argument("--vision_checkpoint", type=str, default=r"C:\Programming\akkadian\checkpoints_final_vision\final_model")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/Iskander-Dataset")
    parser.add_argument("--hf_config", type=str, default="documents")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--label_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--n_examples", type=int, default=20)
    parser.add_argument("--context_char_max", type=int, default=850)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tablet_ids", type=str, default=None,
                         help="Comma-separated tablet_id list to use instead of random sampling (order preserved). "
                              "Overrides --n_examples.")
    parser.add_argument("--output_file", type=str, default="predictions_demo.md")
    parser.add_argument("--embed_images", action="store_true",
                         help="Save each example's real photo (if any) next to output_file and embed it in the "
                              "markdown, instead of only noting has-photo True/False")
    parser.add_argument("--full_photo_max_side", type=int, default=900,
                         help="Downscale the full-resolution reference photo so its longer side is at most this "
                              "many pixels (aspect ratio preserved) -- CDLI originals run 2500-5000px+ and are "
                              "several MB each, wasted size for a reference image at markdown-viewer scale")
    parser.add_argument("--fetch_cdli_info", action="store_true",
                         help="Look up each tablet's description (genre/period/collection/publication) and a real "
                              "line-by-line cuneiform+transliteration+translation table from eBL/CDLI (local dump "
                              "first, live API fallback, cached to data/interim/cdli_record_cache.json). Also "
                              "enables the whole-document English translation line. Not every tablet has all of "
                              "this -- eBL has no translations, CDLI has no inscription record for most literary "
                              "fragments (see session notes on K.3375).")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Loading tokenizer + label config...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_checkpoint, use_fast=False)
    banned_ids = set(tokenizer.all_special_ids)

    label_config_path = args.label_config
    if label_config_path is None:
        from huggingface_hub import hf_hub_download
        label_config_path = hf_hub_download(repo_id=args.data_dir, filename="configs/label_configs.json", repo_type="dataset")
    with open(label_config_path, encoding="utf-8") as f:
        label_configs = json.load(f)
    num_labels = {task: len(label_configs[task]["labels"]) for task in TASKS}

    print(f"Loading {args.split} split ({args.hf_config})...")
    ds = load_dataset(args.data_dir, args.hf_config)[args.split]
    print(f"  {len(ds)} rows")

    print("Loading tablet image index (for the vision model)...")
    image_index = build_tablet_image_index_from_hf(args.data_dir)
    zero_image = torch.zeros(3, 224, 224)

    full_photo_index = {}
    if args.embed_images:
        print("Loading full-resolution photo index (local cache, for display only -- not fed to the model)...")
        full_photo_index = build_path_index()

    record_cache = {}
    cdli_dump_idx, ebl_by_cdli, ebl_by_id, work_map = {}, {}, {}, {}
    if args.fetch_cdli_info:
        record_cache = load_record_cache()
        print("Loading local CDLI ATF dump + eBL export indexes (checked before any live API call)...")
        cdli_dump_idx = build_cdli_atf_dump_index()
        ebl_by_cdli, ebl_by_id = build_ebl_local_index()
        work_map = load_showcase_work_map()

    print("Loading text-only model...")
    text_model = load_model(args.text_checkpoint, args.model_name, num_labels, use_image=False, vision_init="scratch")
    print("Loading vision model...")
    vision_model = load_model(args.vision_checkpoint, args.model_name, num_labels, use_image=True, vision_init="finetune")

    if args.tablet_ids:
        wanted = [t.strip() for t in args.tablet_ids.split(",") if t.strip()]
        id_to_idx = {ds[i]["tablet_id"]: i for i in range(len(ds))}
        indices = []
        for tid in wanted:
            if tid not in id_to_idx:
                print(f"  WARNING: {tid} not found in {args.split} split, skipping")
                continue
            indices.append(id_to_idx[tid])
        print(f"Selected {len(indices)}/{len(wanted)} requested tablets")
    else:
        # Prefer examples that actually have a real photo, so the vision model's
        # provenience row isn't just running on the same all-zero placeholder as
        # text-only every time -- otherwise most of the demo would show no
        # possible difference by construction.
        has_photo = [i for i in range(len(ds)) if ds[i]["tablet_id"] in image_index]
        no_photo = [i for i in range(len(ds)) if ds[i]["tablet_id"] not in image_index]
        rng.shuffle(has_photo)
        rng.shuffle(no_photo)
        n_photo = min(len(has_photo), max(1, args.n_examples * 2 // 3))
        indices = has_photo[:n_photo] + no_photo[:args.n_examples - n_photo]
        rng.shuffle(indices)
        print(f"Selected {len(indices)} examples ({n_photo} with a real photo, {len(indices) - n_photo} without)")

    selection_note = (f"{len(indices)} hand-picked tablet(s) (`--tablet_ids`)" if args.tablet_ids
                       else f"{len(indices)} random test-split tablets, seed={args.seed}")
    out = []
    out.append("# Prediction demo: text-only vs vision (provenience) model\n")
    out.append(f"{selection_note}. Both models see the exact same "
               f"masked positions per example (`[MASK]` shown at every chosen position, {args.mlm_probability:.0%} "
               "of eligible tokens) -- differences in restoration come only from the two models' separately "
               "trained weights, not from the image itself (the image only reaches `provenience_head`, see module "
               "docstring). The metadata table's `provenience` row is where the image can actually change an answer.\n")

    n_with_lines = 0
    for n, idx in enumerate(indices, 1):
        row = ds[idx]
        tablet_id = row["tablet_id"]
        text = row["text"][:args.context_char_max]
        marked = mark_damage_signals(text)
        enc = tokenizer(marked, truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"]

        positions = mask_positions(input_ids, banned_ids, args.mlm_probability, rng)
        masked_ids = list(input_ids)
        true_tokens = [input_ids[p] for p in positions]
        for p in positions:
            masked_ids[p] = tokenizer.mask_token_id

        input_tensor = torch.tensor([masked_ids])
        attn = torch.tensor([[1] * len(masked_ids)])
        img = image_index.get(tablet_id)
        pixel_values = IMG_TRANSFORM_EVAL(img).unsqueeze(0) if img is not None else zero_image.unsqueeze(0)

        with torch.no_grad():
            text_out = text_model(input_ids=input_tensor, attention_mask=attn)
            vision_out = vision_model(input_ids=input_tensor, attention_mask=attn, pixel_values=pixel_values)

        truth = {task: row[f"{task}_labels"] for task in TASKS}
        text_pred = {}
        vision_pred = {}
        for task in TASKS:
            for name, out_dict, pred_dict in [("text", text_out, text_pred), ("vision", vision_out, vision_pred)]:
                probs = torch.softmax(out_dict[f"{task}_logits"][0], dim=-1)
                conf, cls = probs.max(dim=-1)
                pred_dict[task] = (cls.item(), conf.item())

        masked_display = tokenizer.decode(masked_ids[1:-1], skip_special_tokens=False)
        original_display = tokenizer.decode(input_ids[1:-1], skip_special_tokens=False)

        out.append(f"## Example {n} — `{tablet_id}` (has photo: {img is not None})\n")

        record = {"description": None, "lines": []}
        if args.fetch_cdli_info:
            record = fetch_record(tablet_id, record_cache, cdli_dump_idx, ebl_by_cdli, ebl_by_id, work_map)
            if record["description"]:
                out.append(f"*{record['description']}*\n")
            if record["lines"]:
                n_with_lines += 1

        safe_id = tablet_id.replace(":", "_").replace(",", "_")
        has_crop = has_full = False
        if img is not None and args.embed_images:
            img_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_file)) or ".", "demo_images")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{safe_id}.jpg")
            img.convert("RGB").save(img_path, quality=90)
            has_crop = True

            # Full original (all photographed faces, not just the
            # cropped/letterboxed 224x224 the model actually sees) -- for human
            # reference in the writeup, pulled from local cache (same source
            # finalize_vision_crops.py crops from), not the model's own input.
            # Downscaled (aspect-preserving, PIL's own thumbnail()) to
            # --full_photo_max_side -- these are reference images, not model
            # input, so full CDLI resolution (often 2500x5000+, multiple MB
            # each) is wasted size for no visible gain at markdown-viewer scale.
            full_id = tablet_id[1:] if tablet_id.startswith("P") else None
            full_path = full_photo_index.get(full_id) if full_id else None
            if full_path and os.path.exists(full_path):
                from PIL import Image as _Image
                full_out = os.path.join(img_dir, f"{safe_id}_full.jpg")
                full_img = _Image.open(full_path).convert("RGB")
                full_img.thumbnail((args.full_photo_max_side, args.full_photo_max_side), _Image.LANCZOS)
                full_img.save(full_out, quality=85)
                has_full = True

        side_by_side = format_side_by_side(safe_id, tablet_id, has_crop, has_full, record["lines"]) if has_crop else None
        if side_by_side:
            out.append(side_by_side)
        elif has_crop:
            out.append(f"![{tablet_id}](demo_images/{safe_id}.jpg)\n")

        out.append(f"**Original text (transliteration):**\n> {original_display}\n")
        if row.get("signs"):
            out.append(f"**Cuneiform (Unicode signs, whole document, not position-aligned to the text above):**\n> {' '.join(row['signs'])}\n")
        if args.fetch_cdli_info:
            all_translations = [r["translation"] for r in record["lines"] if r.get("translation")]
            if all_translations:
                out.append(f"**English translation (CDLI, whole document, line-by-line above is the exact alignment):**\n> {' '.join(all_translations)}\n")
        out.append(f"**Masked input ({len(positions)} positions):**\n> {masked_display}\n")

        out.append("### Restoration (masked-token predictions)\n")
        out.append("| # | true token | text-only top-1 | text-only top-3 | vision top-1 | vision top-3 | text-only correct | vision correct |")
        out.append("|---|---|---|---|---|---|---|---|")
        text_correct = vision_correct = 0
        for i, (pos, true_id) in enumerate(zip(positions, true_tokens), 1):
            true_tok = tokenizer.convert_ids_to_tokens([true_id])[0]
            t_top3 = topk_at(text_out["logits"], pos, banned_ids, tokenizer)
            v_top3 = topk_at(vision_out["logits"], pos, banned_ids, tokenizer)
            t_ok = t_top3[0] == true_tok
            v_ok = v_top3[0] == true_tok
            text_correct += t_ok
            vision_correct += v_ok
            out.append(f"| {i} | `{true_tok}` | `{t_top3[0]}` | {', '.join(f'`{t}`' for t in t_top3)} | "
                       f"`{v_top3[0]}` | {', '.join(f'`{t}`' for t in v_top3)} | {'✅' if t_ok else '❌'} | {'✅' if v_ok else '❌'} |")
        n_pos = len(positions)
        out.append(f"\nTop-1 accuracy on this example: text-only {text_correct}/{n_pos} "
                   f"({text_correct / n_pos:.0%}), vision {vision_correct}/{n_pos} ({vision_correct / n_pos:.0%})\n")

        out.append("### Metadata predictions\n")
        out.append(format_metadata_table(label_configs, truth, text_pred, vision_pred))
        out.append("\n---\n")

        if n % 5 == 0:
            print(f"  {n}/{len(indices)} done")

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Saved to {args.output_file}")

    if args.fetch_cdli_info:
        save_record_cache(record_cache)
        n_desc = sum(1 for i in indices if record_cache.get(ds[i]["tablet_id"], {}).get("description"))
        print(f"Descriptions found: {n_desc}/{len(indices)}, line-by-line tables: {n_with_lines}/{len(indices)}")


if __name__ == "__main__":
    main()
