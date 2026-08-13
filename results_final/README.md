# Final test-split evaluation (session 2026-08-13)

First and only time either checkpoint touched the `test` split — every
number quoted earlier in the session (validation, used for checkpoint
selection during training) is a different, slightly less strict split. This
is the number to actually cite.

## Files

- `metrics_text.json` — `checkpoints_final_text` on `test` (3106 tablets):
  aggregate metrics (MLM MRR/Hit@k, per-head accuracy + macro-F1) and a
  full per-class precision/recall/F1/support breakdown for all 4 metadata
  heads.
- `metrics_vision.json` — same, `checkpoints_final_vision` (provenience
  image-conditioned).
- `predictions_demo.md` — 20 real test-split tablets: original text, masked
  input (`[MASK]` at every chosen position), both models' top-1/top-3
  restoration guess per masked token side by side, and both models'
  metadata predictions (with confidence) vs. ground truth. Provenience rows
  where the two models disagree are flagged `<- differs`.

## Headline numbers (test split, not validation)

| | text-only | vision (provenience) |
|---|---|---|
| mlm_mrr | 0.798 | 0.799 |
| mlm_top5_acc | 0.869 | 0.868 |
| period macro-F1 | 0.837 | 0.855 |
| genre macro-F1 | 0.843 | 0.854 |
| language macro-F1 | 0.830 | 0.831 |
| **provenience macro-F1** | **0.729** | **0.768** |

Matches the validation-split finding closely (+0.039 here vs. +0.039-0.041
on validation) — the provenience effect holds on genuinely held-out data,
not just the split used for checkpoint selection during training.

## Reading `predictions_demo.md`

- Masking always shows literal `[MASK]` at every chosen position (not
  BERT's real 80/10/10 masking recipe) -- for legibility; the metrics above
  come from the real collator, this file is illustration only.
- The image only reaches `provenience_head` (see
  `src/training/train_mbert.py`'s `MBertMultiTask.forward` -- `mlm_logits`
  is computed before any image concatenation), so restoration differences
  between the two models come only from being separately trained weights,
  not from the image directly. The `provenience` row in each example's
  metadata table is where the image can actually change an answer.
- 13/20 examples used a tablet with a real photo (biased toward this on
  purpose -- otherwise the vision model would run on the same all-zero
  placeholder as text-only most of the time, showing no possible
  difference by construction).
- Of the 14 examples with a real ground-truth provenience label, the two
  models agree 13 times (matching or both wrong) and disagree once (a case
  text-only got right and vision got wrong) -- consistent with the
  aggregate result being a real but modest average effect, not a
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

uv run python src/analysis/demo_predictions.py \
  --text_checkpoint checkpoints_final_text/final_model \
  --vision_checkpoint checkpoints_final_vision/final_model \
  --data_dir AlexSychovUN/Iskander-Dataset --hf_config documents --split test \
  --n_examples 20 --context_char_max 850 --max_length 512 \
  --output_file results_final/predictions_demo.md
```

`--batch_size 4` is sized for this machine's 4GB GPU (the training runs
used a larger box) -- bump it up if running elsewhere.
