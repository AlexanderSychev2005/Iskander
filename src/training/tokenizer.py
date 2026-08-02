import json
import os
import re
from collections import Counter

_ELLIPSIS_RE = re.compile(r"\.\.\.+")

def collapse_ellipsis_gaps(text):
    """Collapses an unknown-length gap ('...' in ATF transliteration) to the
    single reserved '[#]' token, mirroring how prepare_oracc.py already does
    this for the sign-level 'signs' field (see extract_utf8's 'ellipsis'
    branch). Must run before build_vocab(field='text') or encode() on
    transliteration text, so a real gap shares one vocab slot with the
    collator's own synthetic gaps instead of 3+ literal dots being counted
    (and potentially masked) as ordinary characters."""
    return _ELLIPSIS_RE.sub(" [#] ", text)

class CharacterTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.hash_token = '[#]'
        self.x_token = 'x'
        # Uppercase 'X' is the third real-damage signal (a sign present but
        # unidentifiable, distinct from lowercase 'x' and from '[#]') -- it
        # was already excluded from masking ad hoc in the training collator,
        # but never had a reserved vocab slot like its two siblings. Making
        # it a proper special token keeps all three consistent: fixed index,
        # immune to build_vocab's frequency cutoff, skipped by decode(...,
        # skip_special_tokens=True).
        self.X_token = 'X'

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.mask_token,
            self.cls_token,
            self.sep_token,
            self.hash_token,
            self.x_token,
            self.X_token
        ]
        
        # Initialize vocab with special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

    def build_vocab(self, jsonl_path, min_freq=1, field='signs'):
        """Builds vocabulary from a JSONL dataset file. field='signs' counts
        the pre-segmented cuneiform sign list (list of glyph strings);
        field='text' counts the cleaned Latin transliteration string
        character-by-character (with '...' collapsed to '[#]' first, and
        '[#]' itself kept atomic via _units_from_text) -- these are two
        disjoint vocabularies for the two training tracks, never merged."""
        print(f"Building vocabulary from {jsonl_path} (field={field})...")
        char_counter = Counter()

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    units = data.get(field, '')
                    if not units:
                        continue
                    if field == 'text':
                        char_counter.update(self._units_from_text(collapse_ellipsis_gaps(units)))
                    else:
                        char_counter.update(units)
                except Exception:
                    pass
                    
        # Sort characters by frequency. Special tokens (e.g. "[#]", now also
        # produced by the ORACC extractor for real unknown-length gaps) are
        # excluded here so a real-data occurrence can never reassign their
        # fixed reserved index.
        valid_chars = [char for char, count in char_counter.most_common() if count >= min_freq and char not in self.vocab]
        print(f"Found {len(valid_chars)} unique characters with frequency >= {min_freq}")
        
        # Add to vocab
        start_idx = len(self.special_tokens)
        for i, char in enumerate(valid_chars):
            idx = start_idx + i
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            
    def save(self, vocab_path):
        """Saves vocabulary to a JSON file."""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {vocab_path} (Size: {len(self.vocab)})")
        
    def save_pretrained(self, save_directory):
        """Hugging Face Trainer compatibility."""
        os.makedirs(save_directory, exist_ok=True)
        self.save(os.path.join(save_directory, "vocab.json"))
        
    def load(self, vocab_path):
        """Loads vocabulary from a JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        print(f"Vocabulary loaded from {vocab_path} (Size: {len(self.vocab)})")
        
    def _markup_regex(self):
        """Matches the small closed set of multi-char determinative/markup
        tags (e.g. <D>, <ki>) plus the '[#]' gap sentinel, all of which
        appear as vocab entries. Never matches compound cuneiform sign
        clusters -- those are ambiguous to reconstruct from a flat string
        and must go through encode_signs() with the original per-sign
        segmentation instead. '[#]' needs to be here (not just in
        special_tokens) because plain encode() is used on the transliteration
        track, where an unknown-length gap is marked '...' in the source and
        collapsed to the literal substring '[#]' before encoding -- without
        this, '[', '#', ']' would tokenize as three unrelated characters
        instead of the one reserved gap token."""
        if not hasattr(self, "_markup_re"):
            import re
            tags = [k for k in self.vocab if re.fullmatch(r"<\w+>", k)]
            if self.hash_token in self.vocab:
                tags.append(self.hash_token)
            tags.sort(key=len, reverse=True)
            pattern = "|".join(re.escape(t) for t in tags) if tags else r"(?!)"
            self._markup_re = re.compile(pattern)
        return self._markup_re

    def _units_from_text(self, text):
        """Splits a plain string into vocab units: known markup tags stay
        atomic, everything else is one unit per character. This is the only
        safe way to segment a string with no known sign boundaries -- it
        never merges adjacent characters into a compound token."""
        regex = self._markup_regex()
        units = []
        pos = 0
        for m in regex.finditer(text):
            if m.start() > pos:
                units.extend(text[pos:m.start()])
            units.append(m.group())
            pos = m.end()
        units.extend(text[pos:])
        return units

    def _ids_from_units(self, units):
        ids = []
        for unit in units:
            if unit in self.vocab:
                ids.append(self.vocab[unit])
            else:
                # Unknown multi-char unit (e.g. a compound sign that never made
                # the frequency cutoff): fall back to its individual characters
                # instead of collapsing straight to [UNK].
                for ch in unit:
                    ids.append(self.vocab.get(ch, self.vocab[self.unk_token]))
        return ids

    def _finalize(self, token_ids, add_special_tokens, max_length):
        if add_special_tokens:
            token_ids = [self.vocab[self.cls_token]] + token_ids + [self.vocab[self.sep_token]]
        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                if add_special_tokens:
                    token_ids[-1] = self.vocab[self.sep_token]
            else:
                pad_len = max_length - len(token_ids)
                token_ids = token_ids + [self.vocab[self.pad_token]] * pad_len
        return token_ids

    def encode(self, text, add_special_tokens=True, max_length=None):
        """Encodes a free-form string (no known sign boundaries) into token IDs.
        One unit = one character, except for known markup tags. Use
        encode_signs() instead whenever the original per-sign list is available
        (e.g. CuneiML's 'signs' field) -- it preserves the source's real
        segmentation instead of guessing it back from a flat string."""
        if text is None:
            text = ""
        ids = self._ids_from_units(self._units_from_text(text))
        return self._finalize(ids, add_special_tokens, max_length)

    def encode_signs(self, signs, add_special_tokens=True, max_length=None):
        """Encodes a sequence of pre-segmented signs (a list, as produced by
        the data pipeline) or a plain string. Each list element is treated as
        one atomic unit if it's in the vocab -- this preserves compound signs
        (e.g. numeral clusters) and markup tags exactly as segmented by the
        source dataset, instead of re-deriving boundaries via greedy matching."""
        if signs is None:
            signs = []
        if isinstance(signs, str):
            units = self._units_from_text(signs)
        else:
            units = list(signs)
        ids = self._ids_from_units(units)
        return self._finalize(ids, add_special_tokens, max_length)
        
    def decode(self, token_ids, skip_special_tokens=False):
        """Decodes a list of token IDs back into a string."""
        chars = []
        for idx in token_ids:
            char = self.inverse_vocab.get(idx, self.unk_token)
            if skip_special_tokens and char in self.special_tokens:
                continue
            chars.append(char)
        return "".join(chars)

# Helper for testing/building
def _demo():
    """Self-check: encode_signs must respect the caller's own sign
    boundaries, never merge them via coincidental greedy matching."""
    tok = CharacterTokenizer()
    tok.vocab = {
        "[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3, "[SEP]": 4,
        "𒌋": 5, "𒌋𒌋": 6, "<D>": 7, "𒀭": 8,
    }
    tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}

    # Two SEPARATE signs that happen to concatenate into a known compound
    # token -- must stay two tokens, not collapse into one.
    ids = tok.encode_signs(["𒌋", "𒌋"], add_special_tokens=False)
    assert ids == [5, 5], f"adjacent signs got merged: {ids}"

    # A genuine compound sign from the source list must stay atomic.
    ids = tok.encode_signs(["𒌋𒌋", "𒀭"], add_special_tokens=False)
    assert ids == [6, 8], f"compound sign was split: {ids}"

    # A markup tag inside a plain string must stay atomic; everything else
    # falls back to one token per character.
    ids = tok.encode("𒀭<D>𒀭", add_special_tokens=False)
    assert ids == [8, 7, 8], f"markup tag not preserved: {ids}"

    print("tokenizer self-check passed")


if __name__ == "__main__":
    _demo()
