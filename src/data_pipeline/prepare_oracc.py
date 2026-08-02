import os
import json
import glob
import zipfile
from tqdm import tqdm

# ORACC projects that are lexical lists / gazetteers / catalogue indexes with
# no actual corpusjson text -- skipping them up front avoids opening 47 zips
# just to find nothing.
METADATA_FIELDS = ["period", "genre", "provenience", "language", "dialect", "material", "object_type", "script", "ruler"]


def extract_utf8(gdl_list, out):
    for g in gdl_list:
        if g.get("gdl_type") == "diszless" and "group" in g:
            # A numeral value (e.g. "15", "2/3") -- ORACC's own utf8 field on
            # this wrapper node is a generic "X" placeholder, not damage: the
            # real, fully-known sign glyph(s) are one level down in 'group'
            # (e.g. 15 -> 𒌋+𒐊). Recovering it keeps distinct numerals
            # distinguishable instead of collapsing them all onto the same
            # token used for genuinely illegible signs.
            extract_utf8(g["group"], out)
        elif "utf8" in g:
            out.append(g["utf8"])
        elif g.get("x") == "ellipsis":
            # An unknown-length gap ("..." in the transliteration) -- unlike
            # a single damaged sign ("x", which has its own utf8=="x" node
            # and is handled by the branch above), this node carries no
            # utf8/seq/group and was previously skipped entirely, silently
            # deleting the gap and making its neighbors look adjacent.
            # Represent it with the same compressed-gap token the training
            # collator already uses for synthetic damage, so real and
            # simulated gaps share one vocabulary entry.
            out.append("[#]")
        elif "seq" in g:
            extract_utf8(g["seq"], out)
        elif "group" in g:
            extract_utf8(g["group"], out)


def parse_corpus_json(data, metadata):
    lines = []
    current_raw, current_signs, current_num = [], [], ""

    def flush():
        if current_raw or current_signs:
            # Collapse consecutive "[#]" markers into one: ORACC sometimes
            # records the same physical gap via ellipsis nodes on both the
            # word before and after it, which would otherwise inflate one
            # real lacuna into two adjacent gap tokens.
            signs = []
            for s in current_signs:
                if s == "[#]" and signs and signs[-1] == "[#]":
                    continue
                signs.append(s)
            line_obj = {"raw": " ".join(current_raw), "signs": signs, "num": current_num}
            if metadata:
                line_obj.update(metadata)
            lines.append(line_obj)

    def traverse(node):
        nonlocal current_raw, current_signs, current_num
        if not isinstance(node, dict):
            return
        if node.get("node") == "d" and node.get("type") == "line-start":
            flush()
            current_raw, current_signs = [], []
            current_num = node.get("n", "")
        elif node.get("node") == "l":
            frag = node.get("frag", "")
            if frag:
                current_raw.append(frag)
            extract_utf8(node.get("f", {}).get("gdl", []), current_signs)
        for child in node.get("cdl", []):
            traverse(child)

    traverse(data)
    flush()
    return lines


def catalogue_metadata(members, textid):
    member = members.get(textid)
    if not member:
        return {}, None
    meta = {field: member.get(field, "unknown") or "unknown" for field in METADATA_FIELDS}
    cdli_id = member.get("cdli_id")
    if not cdli_id and str(member.get("id_text", "")).startswith("P"):
        cdli_id = member["id_text"]
    return meta, cdli_id


def process_zip(zip_path, out_f, seen_signs):
    stats = {"total": 0, "written": 0, "skipped_empty": 0, "skipped_dupe": 0, "cdli_ids": set()}
    try:
        z = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        return stats
    names = z.namelist()
    corpus_files = [n for n in names if "/corpusjson/" in n and n.endswith(".json")]
    if not corpus_files:
        return stats

    catalogue_name = next((n for n in names if n.endswith("catalogue.json")), None)
    members = {}
    if catalogue_name:
        try:
            with z.open(catalogue_name) as f:
                members = json.load(f).get("members", {})
        except (json.JSONDecodeError, KeyError):
            members = {}

    for cf in corpus_files:
        try:
            with z.open(cf) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        textid = data.get("textid", "")
        metadata, cdli_id = catalogue_metadata(members, textid)
        if cdli_id:
            stats["cdli_ids"].add(cdli_id)
        # A stable grouping key for train/val/test splitting: the real CDLI
        # P-number when we have one (comparable across ORACC and CuneiML), or
        # an ORACC-local fallback so ungrouped texts still split as a whole.
        tablet_id = cdli_id or f"oracc:{data.get('project', textid)}:{textid}"

        for line in parse_corpus_json(data, metadata):
            stats["total"] += 1
            signs = [s for s in line["signs"] if s]
            if len(signs) < 2:
                stats["skipped_empty"] += 1
                continue
            sign_key = "".join(signs)
            if sign_key in seen_signs:
                stats["skipped_dupe"] += 1
                continue
            seen_signs.add(sign_key)
            line["signs"] = signs
            line["textid"] = textid
            line["cdli_id"] = cdli_id
            line["tablet_id"] = tablet_id
            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(base_dir, "data", "raw", "oracc")
    OUTPUT_FILE = os.path.join(base_dir, "data", "interim", "oracc.jsonl")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    zip_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.zip")))
    print(f"Found {len(zip_paths)} ORACC project archives in {RAW_DIR}")

    seen_signs = set()
    all_cdli_ids = set()
    totals = {"total": 0, "written": 0, "skipped_empty": 0, "skipped_dupe": 0}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for zp in tqdm(zip_paths, desc="ORACC projects"):
            stats = process_zip(zp, out_f, seen_signs)
            all_cdli_ids |= stats["cdli_ids"]
            for k in totals:
                totals[k] += stats[k]

    print(f"\nParsed lines: {totals['total']}")
    print(f"Written unique lines: {totals['written']}")
    print(f"Skipped (fewer than 2 signs): {totals['skipped_empty']}")
    print(f"Skipped (duplicate sign string): {totals['skipped_dupe']}")
    print(f"Distinct CDLI IDs cross-referenced: {len(all_cdli_ids)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
