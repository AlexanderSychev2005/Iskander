# Final test-split evaluation (session 2026-08-13, updated after adding P387407)

First and only time either checkpoint touched the `test` split — every
number quoted earlier in the session (validation, used for checkpoint
selection during training) is a different, slightly less strict split. This
is the number to actually cite.

## Files

- `metrics_text.json` — `checkpoints_final_text` on `test` (3107 tablets):
  aggregate metrics (MLM MRR/Hit@k, per-head accuracy + macro-F1) and a
  full per-class precision/recall/F1/support breakdown for all 4 metadata
  heads.
- `metrics_vision.json` — same, `checkpoints_final_vision` (provenience
  image-conditioned), 377 tablets in the `vision` config's test split.
- `metrics_untrained.json` — the zero-finetuning baseline: plain
  `bert-base-multilingual-cased`'s own pretrained weights (untouched) +
  freshly random-initialized metadata heads, evaluated the same way on the
  same `test` split. Uses `checkpoints_final_text/final_model`'s tokenizer
  (for the injected Akkadian WordPiece tokens + damage sentinels, so
  masking-eligibility and vocab fragmentation match the trained runs
  exactly) but none of its trained weights (`evaluate_mbert.py --untrained`).
  This is the same kind of comparison Lazar et al. 2021 report (their
  Table 2, finetuned vs. non-finetuned mBERT) -- shows how much of the
  result is from Akkadian-specific finetuning vs. mBERT's own multilingual
  pretraining.
- `predictions_demo.md` — 20 random test-split tablets. `predictions_demo_showcase.md`
  — 8 hand-picked tablets (Gilgamesh/Enuma Elish/Atrahasis/Hammurabi + P387407).
  Each example has:
  - a one-line **description** (genre/period/collection/publication, pulled
    from whichever of eBL/CDLI actually has this tablet);
  - a **side-by-side block**: the 224x224 crop the model actually sees + the
    full original photo (all photographed faces, downscaled to 900px longer
    side) on the left, and a genuine **line-by-line table** (cuneiform +
    transliteration + English translation, by ATF line number/face) on the
    right -- a real per-line parse of the tablet's own raw ATF, not this
    project's own flattened whole-document `text`/`signs` columns (which
    lose line boundaries in the corpus merge);
  - the flattened original transliteration, whole-document cuneiform signs,
    and whole-document translation (same content as the line table, joined)
    for reference/searchability;
  - the masked input and both models' top-1/top-3 restoration guess per
    masked token, and both models' metadata predictions (confidence) vs.
    ground truth -- provenience rows where the two models disagree are
    flagged `<- differs`.

  Description/translation/line-table coverage is honestly uneven, not a
  bug: `predictions_demo_showcase.md` got a description+line-table for all
  8, but only 1 (P387407, an ordinary letter) has any translation -- the
  other 7 are eBL-sourced literary fragments (Gilgamesh etc.) with no CDLI
  inscription record at all and no translation field in eBL's own API
  either (checked both live). `predictions_demo.md` did better on
  translations (3/20, ordinary administrative/lexical texts) and got line
  tables for 18/20. Same wall this session already hit with K.3375: open
  translations for literary works don't really exist outside copyrighted
  scholarly editions.
- `demo_images/` — `<tablet_id>.jpg` (model-input crop) and
  `<tablet_id>_full.jpg` (full original, reference only) for every example
  with a photo in either demo file.

## Headline numbers (test split, not validation)

| | untrained (no finetuning) | text-only | vision (provenience) |
|---|---|---|---|
| mlm_mrr | 0.512 | 0.797 | 0.799 |
| mlm_top5_acc | 0.561 | 0.867 | 0.867 |
| period macro-F1 | 0.062 | 0.838 | 0.854 |
| genre macro-F1 | 0.053 | 0.840 | 0.855 |
| language macro-F1 | 0.071 | 0.846 | 0.841 |
| **provenience macro-F1** | 0.023 | **0.727** | **0.768** |

The untrained column confirms neither result is free: mBERT's own
pretraining gets restoration to a non-trivial MRR 0.51 zero-shot (matching
Lazar et al. 2021's own point that multilingual pretraining transfers
usefully to Akkadian on its own), but the metadata heads are exactly what
random Linear-layer init on a 4-96-class problem looks like -- effectively
chance, several classes collapsed to 0 F1 (see `metrics_untrained.json`'s
per-class breakdown). All four heads' real signal comes entirely from this
project's finetuning, not from the backbone alone.

Vision vs. text-only matches the validation-split finding closely (+0.042
here vs. +0.039-0.041 on validation) — the provenience effect holds on
genuinely held-out data, not just the split used for checkpoint selection
during training. Period/
genre/language moving by a few points here (in both directions) is the same
ordinary run-to-run noise already characterized via the controlled ablation
in `docs/final_results.md` -- not a new finding, don't read a single-run
test-split number as reopening that question.

## P387407 addition

Added this session (found by the user on cdli.earth, has both a real photo
and full ATF transliteration in the official CDLI bulk dump -- see
`docs/final_results.md`'s git history / commit `31e450b`). Forced into
`test`, hence the split growing from 3106->3107 documents / 376->377 vision
tablets and this file's numbers moving slightly from the previous version
(one additional tablet out of 3107 has negligible effect on aggregate
metrics, but is now included for consistency).

## Reading the demo files

- Masking always shows literal `[MASK]` at every chosen position (not
  BERT's real 80/10/10 masking recipe) -- for legibility; the metrics above
  come from the real collator, these files are illustration only.
- The image only reaches `provenience_head` (see
  `src/training/train_mbert.py`'s `MBertMultiTask.forward` -- `mlm_logits`
  is computed before any image concatenation), so restoration differences
  between the two models come only from being separately trained weights,
  not from the image directly. The `provenience` row in each example's
  metadata table is where the image can actually change an answer.
- The full-resolution photo is pulled from the local raw-download cache
  (same source `finalize_vision_crops.py` crops from), not from the model's
  actual input -- it's there for human reference/the diploma writeup, the
  model only ever sees the 224x224 crop next to it.
- `predictions_demo.md`: 13/20 examples used a tablet with a real photo
  (biased toward this on purpose -- otherwise the vision model would run on
  the same all-zero placeholder as text-only most of the time, showing no
  possible difference by construction). Of the examples with a real
  ground-truth provenience label, the two models mostly agree (matching or
  both wrong), disagreeing on a small minority -- consistent with the
  aggregate result being a real but modest average effect, not an
  every-single-example win.

## Reproduce

```bash
uv run python src/analysis/evaluate_mbert.py \
  --checkpoint checkpoints_final_text/final_model \
  --data_dir AlexSychovUN/Iskander-Dataset --hf_config documents --split test \
  --context_char_max 850 --max_length 512 --batch_size 4 \
  --output_file results_final/metrics_text.json

uv run python src/analysis/evaluate_mbert.py \
  --checkpoint checkpoints_final_vision/final_model \
  --data_dir AlexSychovUN/Iskander-Dataset --hf_config documents --split test \
  --context_char_max 850 --max_length 512 --batch_size 4 \
  --use_image --vision_init finetune --images_from_hf \
  --output_file results_final/metrics_vision.json

uv run python src/analysis/evaluate_mbert.py \
  --checkpoint checkpoints_final_text/final_model --untrained \
  --data_dir AlexSychovUN/Iskander-Dataset --hf_config documents --split test \
  --context_char_max 850 --max_length 512 --batch_size 4 \
  --output_file results_final/metrics_untrained.json

uv run python src/analysis/demo_predictions.py \
  --text_checkpoint checkpoints_final_text/final_model \
  --vision_checkpoint checkpoints_final_vision/final_model \
  --data_dir AlexSychovUN/Iskander-Dataset --hf_config documents --split test \
  --n_examples 20 --context_char_max 850 --max_length 512 --embed_images --fetch_cdli_info --seed 42 \
  --output_file results_final/predictions_demo.md

uv run python src/analysis/demo_predictions.py \
  --text_checkpoint checkpoints_final_text/final_model \
  --vision_checkpoint checkpoints_final_vision/final_model \
  --data_dir AlexSychovUN/Iskander-Dataset --hf_config documents --split test \
  --tablet_ids "P273207,P285823,P273223,P402919,ebl:BM.42004,P404643,P402685,P387407" \
  --context_char_max 850 --max_length 512 --embed_images --fetch_cdli_info \
  --output_file results_final/predictions_demo_showcase.md
```

`--batch_size 4` is sized for this machine's 4GB GPU (the training runs
used a larger box) -- bump it up if running elsewhere. `--embed_images`
needs the local raw-image cache (`data/vision_dataset/`,
`data/raw/cuneiml/images_full/`) for the full-resolution photos -- the
224x224 crops alone come from the HF `vision` config regardless.
