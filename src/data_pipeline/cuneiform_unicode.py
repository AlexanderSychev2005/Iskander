"""ATF (line-numbered, @obverse/@reverse-tagged) -> {raw transliteration,
Unicode cuneiform signs} per line. Ported from CuneiML's own converter
(github.com/taineleau/CuneiML, CC0), the same tool used to build CuneiML's
own 'signs' field -- so tablets we pull from the raw CDLI ATF dump / eBL get
signs through the identical process as the rest of the corpus, instead of
sitting with an empty 'signs' column (session 2026-08-12: user asked to
unify signs+transliteration across all sources rather than leaving the
bulk-backfilled tablets text-only).

Vendored sign list: data/raw/cuneiform_unicode_vocab/{token.tsv,
cuneiform_vocab.txt} (8200 entries total, copied verbatim from the CuneiML
repo, same CC0 license). No network calls, no external API.
"""
import os
import re
from collections import Counter

VOCAB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "cuneiform_unicode_vocab",
)

_FACE_KEYS = ("obverse", "reverse", "left", "right", "top", "down", "surface a")


def _load_vocab():
    text2sign = {}
    for fname in ("cuneiform_vocab.txt", "token.tsv"):
        path = os.path.join(VOCAB_DIR, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                try:
                    k, s = line.split("\t")
                except ValueError:
                    continue
                text2sign[k] = s
    return text2sign


_TEXT2SIGN = _load_vocab()

_S_TOKENS = ("<B>", "<M>", "<S>", "<D>", "<munus>", "<ansze>", "<ki>", "<disz>", "x")


def _remove_at(x):
    if x.endswith("@c)") or x.endswith("@t)"):
        return x[:-3] + ")"
    return None


def _remove_spaces(signs):
    out = []
    for item in signs:
        if item == "<S>" and out and out[-1] == "<S>":
            continue
        out.append(item)
    return out


def atf_to_lines(raw_text):
    """raw_text: a tablet's full ATF body (line-numbered, with @face/#atf/$
    structural markers) -- e.g. one &P###### chunk's body from the CDLI bulk
    dump, or an eBL fragment's 'atf' field, unmodified.

    Returns a list of {'raw': str, 'signs': [str], 'num': str, 'face': str}
    in file order, plus an unknown-token miss count for QA (CuneiML's own
    paper reports ~1% of tokens fail to resolve to a sign)."""
    lines_out = []
    curr_face = "default"
    misses = Counter()
    total_tokens = 0

    sep = "\n"
    if "\\n" in raw_text and "\n" not in raw_text:
        sep = "\\n"

    for line in raw_text.split(sep):
        line = line.strip()
        if not line:
            continue
        if line.startswith("&") or line.startswith("'&"):
            continue
        if line.startswith("#atf"):
            continue
        if line.startswith("#") or line.startswith(">>"):
            continue
        if line.startswith("$"):
            continue
        if line.startswith("@"):
            key = line[1:].strip().strip("?")
            if key in _FACE_KEYS:
                curr_face = key
            continue

        line = line.replace("{d}", "<D>")
        for x in re.findall(r"\{.*?\}", line):
            line = line.replace(x, " " + x[1:-1] + " ")
        line = line.replace("($ blank space $)", "<S>")
        line = line.replace("_", " ")
        line = line.replace("#", "").replace("?", "").replace("!", "")
        for x in re.findall(r"\[.*?\]", line):
            line = line.replace(x, "")

        parts = line.split(". ")
        if len(parts) < 2:
            continue
        if len(parts) > 2:
            parts = parts[0], ". ".join(parts[1:])
        line_num, text = parts

        tokens = text.split(" ")
        signs = []
        for i, t in enumerate(tokens):
            if i > 0 and len(signs) > 0:
                signs.append("<S>")
            if "-" in t:
                for x in t.split("-"):
                    x = x.strip()
                    if not x:
                        continue
                    total_tokens += 1
                    if x in _TEXT2SIGN:
                        signs.append(_TEXT2SIGN[x])
                    else:
                        alt = _remove_at(x)
                        if alt and alt in _TEXT2SIGN:
                            signs.append(_TEXT2SIGN[alt])
                        else:
                            misses[x] += 1
            elif t in _TEXT2SIGN:
                total_tokens += 1
                signs.append(_TEXT2SIGN[t])
            elif t in _S_TOKENS:
                total_tokens += 1
                signs.append(t)
            elif t.strip():
                total_tokens += 1
                alt = _remove_at(t)
                if alt and alt in _TEXT2SIGN:
                    signs.append(_TEXT2SIGN[alt])
                else:
                    misses[t] += 1

        signs = _remove_spaces(signs)
        if text.strip():
            lines_out.append({"raw": text.strip(), "signs": signs, "num": line_num.strip(), "face": curr_face})

    return lines_out, misses, total_tokens
