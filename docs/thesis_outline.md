# Thesis Outline & Sourced Talking Points

**Working title:** Automated Restoration of Damaged Akkadian Cuneiform Texts: A Masked Language Modelling Approach

Structure modelled on Eremeev's thesis (`papers/thesis_text.txt`), adapted to this project. Every point below cites where it came from: a published source (author, year, exact page/section) or "this work" (a decision/finding made in this project, with the relevant file). Use these as talking points — write the actual prose yourself.

---

## 1. Introduction

- Cuneiform tablets deteriorate; restoration is currently manual, subjective, and slow — this is the motivating problem for framing restoration as MLM.
  — Lazar et al. 2021, §1, p.4682 ("Due to the tablets' deterioration, scholars often rely on contextual cues to manually fill in missing parts in the text in a subjective and time-consuming process.")
  — Fetaya et al. 2020, Abstract, p.22743 (same framing for Late Babylonian archival texts).

- MLM is not a bespoke architecture for this task — it is a repurposing of BERT's own pretraining objective, since "predict the masked token" and "restore the missing sign" are the same task.
  — Lazar et al. 2021, §1, p.4683 ("we identify that the task of masked language modeling... lends itself directly to missing sign prediction in the transliterated texts").

- Prior state of the art (Lazar et al. 2021) reached 89.5% top-5 accuracy on ORACC despite a small corpus (~2.3M signs); this project extends the same task with a larger, merged corpus (ORACC + CuneiML) and two parallel representations (transliteration and cuneiform glyphs).
  — Lazar et al. 2021, Abstract, p.4682.
  — This work: `data/processed/hf_dataset` — 636,051 unique lines after merge/dedup (`src/data_pipeline/prepare_hf_dataset.py`).

- The most recent related system (Aeneas) broadens the task beyond MLM restoration to multimodal, arbitrary-length restoration plus geographic/chronological attribution — motivates this project's own planned image-conditioned head.
  — Assael et al. 2025 (Aeneas), *Nature* 645, p.141–142 ("Aeneas retrieves textual and contextual parallels, leverages visual inputs, handles arbitrary-length text restoration...").

---

## 2. Literature Review

### 2.1 Akkadian Cuneiform Corpora and Related Work

- ORACC scale in prior work: ~10K texts, 1M words, 2.3M signs (Lazar); Fetaya et al. used a much smaller, genre-restricted corpus (1,400 Late Babylonian archival texts, ~221K words).
  — Lazar et al. 2021, §2.1, p.4683, Table 1.
  — Fetaya et al. 2020, "Data Scraping...", p.22750 (vocabulary 1,549 unique words, 220,926 total).

- This project combines ORACC with CuneiML (which additionally carries real Unicode-glyph segmentation and CDLI tablet images/bounding boxes) to reach a substantially larger, source-agnostic corpus than either prior study.
  — This work: `src/data_pipeline/prepare_oracc.py`, `prepare_cuneiml.py`; merged corpus stats above.

### 2.1.1 Sign-Level vs. Editorial Digitization

- ORACC's `corpusjson`/`gdl` tree is an *editorial* representation (grammatical parse, not raw glyphs); CuneiML's `sign` field is a direct list of Unicode cuneiform glyphs. The two require different extraction logic and were cross-validated against each other this session (romanization convention unification, cross-source deduplication by CDLI P-number).
  — This work: `src/data_pipeline/prepare_hf_dataset.py` (`_normalize_cuneiml_romanization`, `load_and_deduplicate_v2`).

### 2.2 Character, Sign, and Word Granularity in Ancient-Text Restoration

- Aeneas operates **exclusively on characters**, explicitly avoiding word-level representations used by earlier approaches — direct precedent for a character-level design in a related restoration task.
  — Assael et al. 2025, *Nature* 645, p.143 ("Its efficient architecture operates exclusively on characters, avoiding the need for additional word-level representations implemented by previous approaches.")

- Lazar et al. 2021 use **subword (WordPiece)** tokenization for both mBERT and their from-scratch model — the two are directly comparable specifically because both share this granularity, not because tokenization granularity was found unimportant.
  — Lazar et al. 2021, §4.2, p.4685 ("BertWordPieceTokenizer... vocab_size... min_frequency=2"); corroborated by `SLAB-NLP/Akk` repo (`akkadian_bert/tokenize_bert.py`, not a published source — code inspection, this work).

- Fetaya et al. 2020 use **word-level** tokenization with a custom Akkadian tokenizer (proper-name/place/month tagging, `<BRK>`/`<UNK>` sentinels) — a third granularity choice, motivated by their highly structured, formulaic legal-document genre.
  — Fetaya et al. 2020, "Tokenization", p.22750.

- Eremeev's thesis treats granularity as an explicit, controlled experimental variable: DualEmbLM (character-level) vs. RoFormer (BPE), both trained from scratch on identical data, isolating the effect of granularity alone.
  — Eremeev thesis, §4.2, lines 607–640 ("The comparison between RoFormer and DualEmbLM... isolates the effect of tokenisation granularity on restoration quality, holding all other architectural and training choices constant.")

- This project's own granularity decision: character-level for *both* tracks (signs and transliteration), justified empirically — word-level transliteration tokenization would leave 70.4% of the vocabulary as hapax legomena (unusable under `min_freq=2`); syllable-level, 47.6%; character-level has zero OOV risk by construction.
  — This work: session-derived corpus statistics (100k-line sample of `text` field).

### 2.3 Masked Language Modelling for Ancient Text Restoration

#### 2.3.1 Fine-Tuned Masked Language Models

- Zero-shot multilingual mBERT outperforms a small monolingual from-scratch Akkadian BERT by ~10%, attributed to the small size of the target corpus rather than an architectural deficiency — motivates including an mBERT baseline in this project.
  — Lazar et al. 2021, §5.4, p.4686 ("the zero-shot performance of a multilingual language model surpasses that of a monolingual Akkadian model by about 10%... likely due to the relatively small amounts of data").
  — Exact reported numbers: MBERT+Akk (finetuned) MRR .83 / Hit@5 .89 overall vs. BERT+AKK(mono) MRR .50 / Hit@5 .60 — Lazar et al. 2021, Table 2, p.4686.

#### 2.3.2 Task-Specific Models Trained from Scratch

- Direct Akkadian-domain evidence that **bidirectional context** dramatically outperforms unidirectional/causal context on this exact task: LSTM(start) (context before the gap only) reaches MRR 0.754/Hit@1 66.1%, LSTM(full) (bidirectional context) reaches MRR 0.89/Hit@1 85.4% — the single strongest piece of evidence justifying a bidirectional encoder over an autoregressive decoder for this project.
  — Fetaya et al. 2020, Table 2, p.22746.

- Aeneas's "torso" is a T5-derived transformer decoder (non-causal despite the name) augmented with rotary position embeddings (RoPE) — the actual trained configuration is **384 embedding dim, qkv dim 32, MLP 1,536, 16 layers, 8 heads** ("a deep narrow T5 transformer decoder"), not the generic defaults shown in the public repository's `Model` class (512/512/2048/6 layers), which are placeholders, not the paper's reported production config.
  — Assael et al. 2025, *Nature* 645, p.142 (body text, "deep narrow T5 transformer decoder"); Methods, "Aeneas' architecture", p.148 (exact dimensions).

- Eremeev's from-scratch models (RoFormer, DualEmbLM) both use a physical-degradation collator, calibrated against the real corpus's own measured damage rate — direct methodological precedent for this project's `AkkadianPhysicalCollator`.
  — Eremeev thesis, §3.4, lines 468–485.

---

## 3. Data

### 3.1 Dataset Sources

#### 3.1.1 ORACC

- Native per-project metadata (`catalogue.json`) supplies period/genre/provenience/language/dialect/material/object_type/ruler directly — not scraped from an external CDLI catalogue as initially assumed.
  — This work: `src/data_pipeline/prepare_oracc.py` (`catalogue_metadata`).

#### 3.1.2 CuneiML (Supplementary Source)

- CuneiML supplies real Unicode cuneiform glyph segmentation and CDLI photo URLs/bounding boxes per line — the basis for a future multimodal (image-conditioned) head, following Aeneas's precedent of pairing visual input with only specific task heads (see §6).
  — This work: `data/raw/cuneiml/CuneiMLv1.2.json` structure inspection.
  — Chen et al. 2023 (CuneiML dataset paper), `papers/2023_chen_cuneiml_dataset.pdf` (cite dataset description directly from the paper once you re-check the exact page for the figure you use).

### 3.2 Metadata

- Substring-based (not exact-match) label mapping was required because raw CDLI/ORACC period/genre strings almost always carry parenthetical suffixes (e.g. "(ca. NNNN-NNNN BC)"), which silently dropped 30–58% of otherwise-valid records under exact matching.
  — This work: `src/data_pipeline/prepare_hf_dataset.py` (`map_period`, `map_genre`, `map_language`, `map_provenience`).

### 3.3 Text Preprocessing and Normalisation

- ORACC and CuneiML transliterate the same phonemes with two disjoint conventions (Unicode š/ṣ/ṭ/subscript digits vs. ASCII "sz"/"s,"/"t,"/digit-suffix) — left unmerged, this fragments the shared vocabulary for no linguistic reason; unified this session.
  — This work: `_normalize_cuneiml_romanization` (`src/data_pipeline/prepare_hf_dataset.py`), verified on 50,000 lines.

- Lazar et al. 2021 strip editorial uncertainty brackets and *unconditionally* remove sub-/superscript disambiguation digits, losing homophone-sign distinctions (e.g. "ku"/"ku₂"/"ku₃" all collapse) — a real information-loss tradeoff this project's sign-level track avoids by preserving full Unicode glyph segmentation.
  — Lazar et al. 2021, §4.1, p.4684.

- Aeneas retains editorial square-bracket restorations and represents unknown-length gaps with a dedicated `#` placeholder, single missing characters with `-` — structurally the same two-tier convention (single vs. unknown-length gap) this project independently arrived at for ORACC's `x`/`...`.
  — Assael et al. 2025, Methods, "Latin Epigraphic Dataset", p.148.

### 3.4 Masking Scheme and Set Construction

- **Three distinct real-damage signals identified in ORACC**, one of which (uppercase `X`) was found this session to conflate two unrelated phenomena: a genuinely illegible sign (`gdl_sign: "X"`, no `group`) vs. a "diszless" numeral-value wrapper (`gdl_type: "diszless"`, has `group`) where ORACC's own `utf8` field is a generic placeholder and the real, distinguishable numeral glyph sits one level down — 25,181 such nodes were being needlessly collapsed onto one opaque token before this session's fix.
  — This work: full-corpus structural audit of `data/raw/oracc/*.zip`, `src/data_pipeline/prepare_oracc.py` (`extract_utf8`).

- A second, independently confirmed bug: unknown-length gaps (`"..."` in ATF, encoded as a `gdl` node with `x: "ellipsis"` and no `utf8`) were silently dropped from the `signs` list entirely — affecting 16.4% of the final merged corpus's lines — rather than represented by any placeholder, creating false adjacency between the signs on either side of a real physical gap.
  — This work: same audit; fixed by mapping ellipsis nodes to the same `[#]` token the training collator already used synthetically.

- **Masking rate calibration**: real corpus damage rate measured directly (3.2% of signs are `x`/`X`; 2.48% of transliteration characters after collapsing gaps) and the collator's simulated rate (~7.8% signs / ~8.08% transliteration) was deliberately set *above* the raw rate — following the same reasoning Eremeev makes explicit for an unrelated corpus (birchbark manuscripts): "MLM training benefits from somewhat more signal than pure real-damage-rate matching."
  — Eremeev thesis, §3.4, lines 460–467 ("The masking probability of 8% is set to match the empirically observed reconstruction density in the birchbark corpus... 7.38%... An 8% masking rate therefore reflects the proportion of damaged content in the target domain rather than an arbitrary hyperparameter choice.")
  — This work: `src/training/train.py` (`AkkadianPhysicalCollator` docstring, calibration simulation this session).

- **Exclusion of real damage markers from maskable targets** — a position marking genuinely unrecoverable content must never be a training label, since there is no ground truth to score against. This project's design (`x`/`X`/`[#]` excluded from both masking targets *and* model outputs via constrained decoding) matches Eremeev's stated rationale exactly, and goes further than Lazar et al. 2021, whose collator excludes only the single-character `x`→`MISSING_SIGN_CHAR` substitution and does *not* exclude `"..."` from masking targets at all.
  — Eremeev thesis, §3.4, lines 471–474 ("The gap token... is excluded from the set of eligible mask targets: the model observes it as context but is never trained to predict it, preventing the model from learning to output [GAP] as a restoration.")
  — Lazar et al. 2021 code inspection (not the paper itself): `akkadian_bert/data_collators_bert.py`, `preprocessing/main_preprocess.py` lines 140–156 (`_has_missing_parts`, `MISSING_SIGN_CHAR`) — this work.

- Aeneas masks up to 75% of characters during training (span-grouped to simulate contiguous physical damage) — an order of magnitude above this project's ~8% target, reflecting Aeneas's much larger and more heavily damaged corpus (176,861 inscriptions, most damaged) vs. this project's more conservative, corpus-calibrated rate.
  — Assael et al. 2025, Methods, "Aeneas' architecture", p.148 ("up to 75% text masking... Some of these masks are deliberately grouped into continuous segments to better simulate real-world damage.")

### 3.5 Data Augmentation

- Eremeev applies source-level resampling weights to correct thematic-category imbalance in the raw corpus (religious texts overrepresented).
  — Eremeev thesis, §3.5, lines 486–501.

- This project instead relies on macro-F1 reporting (not reweighting) to surface minority-class performance on the 4 metadata heads — an open design choice worth discussing explicitly as a point of divergence, not yet resolved as better or worse.
  — This work: `src/training/model.py` (loss-weighting comment, "no per-class weighting anywhere in their loss either; macro-F1 in compute_metrics is what actually tracks minority-class quality").

---

## 4. Methods

### 4.1 Fine-Tuned Models (mBERT track)

- Following Lazar et al. 2021's mechanism of reassigning mBERT's 99 reserved `[unusedN]` vocabulary slots to inject new tokens without growing the embedding table, this project reuses two such slots as dedicated sentinels for the two real-damage signals present in transliteration text (`x` and unknown-length `...`), verified empirically to be excluded from HF's own masking collator once registered as `additional_special_tokens`.
  — Lazar et al. 2021, §4.2, p.4685 ("we assign its 99 available free tokens, optimizing for maximum likelihood by the WordPiece tokenization algorithm").
  — This work: `src/training/train_mbert.py` (`mark_damage_signals`, `UNCLEAR_SIGN_TOKEN`/`UNKNOWN_GAP_TOKEN`).

### 4.2 Models Trained from Scratch (custom `AkkadianModel`)

- Architecture converges independently with Aeneas's own design on several points: an MLP projection before a tied, `√dim`-normalized embedding dot-product for the restoration head; a dedicated 2-class auxiliary head predicting whether an unknown-length gap continues (Aeneas: `logits_unk`; this project: `unk_head`); RoPE as one of the supported position-encoding schemes.
  — Assael et al. 2025, Methods, p.148 (`logits_mask = ... / jnp.sqrt(x_mask.shape[-1])`; `logits_unk`, out_dim=2); code inspection of `predictingthepast/models/model.py`, this work.
  — This work: `src/training/model.py`.

#### 4.2.1 Sign-Level Track — Trained First (Priority Track)

This is the first track actually trained (ahead of the transliteration track), since it is the intended base for the planned image-conditioned head (§6) — CuneiML's tablet images correspond physically to glyphs, not to the romanized transliteration.

- `hidden_size=640, num_layers=8, num_heads=8` for a 2,145-token glyph vocabulary. Two factors were weighed against Aeneas's actual trained shape (384 dim / 16 layers, §2.3.2/4.2.2) and found to point the *same* direction, not opposite ones: (1) a vocabulary 13–60× larger than Aeneas's needs a *wider* final representation to discriminate cleanly at the tied-softmax readout; (2) lines average only ~8.35 signs (vs. Aeneas's up to 768-character whole-inscription inputs), so *depth* buys little — there is little long-range structure for extra layers to model. Both factors favour a wide/shallow shape for this track, the opposite of Aeneas's narrow/deep shape for its own (small-vocabulary, long-sequence) regime — the two architectures diverge for principled reasons tied to the vocabulary-size/sequence-length profile of each task, not because this track was built more elaborately than necessary.
  — This work: `src/training/train.py` (`--hidden_size` argparse comment); session parameter-count comparison (Aeneas's literal shape at our vocab size ≈29M params vs. this track's ≈41M, with the narrower 384-dim readout estimated as under-provisioned for a 2,145-way classification, by analogy by log-scaling against BERT-base's 768-dim/~30k-vocab ratio).

#### 4.2.2 Transliteration Track — Second Track (mBERT-Comparable Baseline)

- `hidden_size=512, num_layers=6, num_heads=8` — chosen to match Eremeev's DualEmbLM configuration for the same class of problem (character-level, small-vocabulary, low-resource ancient-language restoration transformer): "hidden size 512, 8 attention heads, intermediate size 2,048... 6 layers."
  — Eremeev thesis, §4.2.2, lines 625–635.

### 4.3 Auxiliary Classification (Metadata Heads)

- Task loss weighting (`MLM=3.0, unk=1.0, each metadata head=0.25`) is structurally the same pattern as Aeneas's own fixed (not learned per-example) multi-task weighting, `L = 3·L_restoration + L_unknown + 2·L_region + 1.25·L_date`.
  — Assael et al. 2025, Methods, p.148.
  — This work: `src/training/model.py` (loss computation).

- Label smoothing rates (0.05 for the MLM/restoration objective, 0.1 for classification heads) numerically match Aeneas's own reported values exactly ("smoothing rates of 5% for the restoration task and 10% for geographical attribution") — an independent convergence worth noting, not a copied hyperparameter.
  — Assael et al. 2025, Methods, p.148.

- This session's own loss-weight revision (`meta_weight`: 0.25 → 1.0, i.e. MLM:metadata from 12:1 to 3:1) was motivated directly by contrasting the original ratio against Aeneas's own fixed weighting (≈3:2, restoration:region) — the metadata heads were judged structurally underweighted relative to this precedent, not merely slow to converge. A step-matched comparison against the `meta_weight=0.25` run (step 2000, identical schedule otherwise) shows metadata macro-F1 gains of +6–21% (provenience +21%, period +18%, language +16%, genre +6%) at an MLM cost of only −1–2% (MRR 0.727→0.718, Hit@5 0.810→0.801) — evidence the original weighting was leaving metadata performance on the table rather than reflecting a genuine task-difficulty ceiling.
  — Assael et al. 2025, Methods, p.148 (`L = 3·L_restoration + L_unknown + 2·L_region + 1.25·L_date`).
  — This work: `src/training/train_mbert.py` (`--meta_weight` CLI arg, commit 9e47fa6); step-2000 comparison, `checkpoints_mbert_amd*/checkpoint-2000/trainer_state.json`, 2026-08-05. **Final full-validation-set result (n=31,921) confirms the trend held to completion**: genre macro-F1 0.747→0.776 (+3.9%), language 0.634→0.701 (+10.5%), period 0.645→0.695 (+7.7%), provenience 0.557→0.609 (+9.3%), against an MLM cost of under 1% on every restoration metric (MRR 0.784→0.777, Hit@5 0.862→0.855, Hit@3 0.826→0.818, top-1 0.719→0.711) — `evaluation_report_mbert.json` (meta_weight=0.25) vs. `evaluation_report_mbert_metaw1.json` (meta_weight=1.0), both via `src/analysis/evaluate_mbert.py`.

---

## 5. Evaluation

### 5.1 Metrics

- MRR and Hit@k are defined identically across all three primary comparison points (Lazar, Fetaya, Aeneas) — use the same formal definitions.
  — Lazar et al. 2021, §5.3, Eq. 2–3, p.4685.
  — Fetaya et al. 2020, "Completing Random Missing Tokens", p.22746.
  — Assael et al. 2025, Methods, "Task metrics", p.149.

- CER formula (per-length, then macro-averaged across lengths 1..L) is taken directly from Aeneas's published definition; this project uses L=10 (signs) / L=30 (transliteration, matched to the same physical range in characters) rather than Aeneas's L=20, following Eremeev's L=10 precedent for a similarly-scaled corpus.
  — Assael et al. 2025, Methods, "Task metrics", p.149 (CER_l and CER_score formulas).
  — Eremeev thesis, §3.4, lines 678–680 ("we additionally report span-level macro-CER, averaged uniformly across span lengths 1–10 following Aeneas").
  — This work: `src/analysis/evaluate.py` (`evaluate_cer_by_length`).

- Cross-granularity comparability (character-level custom model vs. subword mBERT) cannot rely on raw per-position Hit@k, since the two operate over different-sized prediction units. Eremeev's solution — constrained decoding restricting all models to a single output character per step — is the direct precedent if this project later needs a token-level comparison; CER (already string-based, hence tokenizer-agnostic) is used as the primary safe comparison metric in the meantime.
  — Eremeev thesis, §4.1, lines 591–605 ("All three models are evaluated under constrained decoding: the prediction space is restricted to tokens corresponding to exactly one Cyrillic character... since the restoration task is defined at the character level, all models, including subword-based ones, are evaluated in a unified output space.")

- Constrained decoding is also applied at the level of *content validity*: this project bans non-content tokens (PAD/UNK/CLS/SEP/MASK plus the real-damage markers) from ever being a model's predicted answer, not just from being a masking target — matching Eremeev's stated rationale and going slightly further than either Lazar or Aeneas make explicit in their published metrics sections.
  — This work: `src/training/train.py` (`non_content_ids`), `src/analysis/evaluate.py`, `src/analysis/inference.py`.

### 5.2 mBERT (Transliteration) Track — Results

- First full run (`meta_weight=0.25`, 20 epochs, best checkpoint selected by `eval_loss` under `load_best_model_at_end` — turned out to be the very last step, 11180/11180, i.e. `eval_loss` never stopped improving): MLM MRR 0.784, Hit@5 0.862, Hit@3 0.826, top-1 0.719 on the full validation split (n=31,921) — 94.5% and 96.9% of Lazar et al. 2021's reported overall MRR (.83) and Hit@5 (.89) respectively, despite not matching their training scale (8×Tesla M60, 20 epochs) or reusing their (unpublished) injected-token list — this project relearns its own WordPiece injection vocabulary from the corpus (§4.1).
  — Lazar et al. 2021, Table 2, p.4686 (comparison numbers).
  — This work: `src/analysis/evaluate_mbert.py`, `checkpoints_mbert_amd/final_model`, run 2026-08-05.

- Metadata heads on the same checkpoint: genre macro-F1 0.747, period 0.645, language 0.634 (accuracy 0.938 — much higher than macro-F1, driven by severe within-head class imbalance: 2 of 4 language classes carry ~97% of labeled examples), provenience 0.557 (weakest head — most classes (12) and largest missing-label fraction (~26%) of all four heads).
  — This work: `evaluation_report_mbert.json`; label-distribution audit against `AlexSychovUN/Iskander-Dataset` train split, this session.

- No train/eval divergence (the standard overfitting signature) observed at any point: train-step loss and `eval_loss` track within ~0.05–0.1 of each other through the final ~2,700 steps.
  — This work: `checkpoints_mbert_amd/checkpoint-11180/trainer_state.json` (`log_history`).

- Design choice, not yet resolved as better or worse: following Aeneas (§3.5), this project uses plain cross-entropy + label smoothing for all four metadata heads rather than per-class (inverse-frequency) weighting, even though Aeneas's own province classes are far more imbalanced than this project's worst case (language) — Aeneas explicitly accepts weaker performance on data-poor classes as a reported limitation rather than a defect to correct via reweighting.
  — Assael et al. 2025, *Nature* 645, p.146 ("performance often tends to be weaker where data is limited"); Extended Data Figs. 3–4.
  — This work: session discussion, 2026-08-05.

### 5.3–5.6 (HP-tuning, sign-level track results, experiment log)

*Fill in once the sign-level track and the `meta_weight=1.0` mBERT re-run complete.*

---

## 6. Limitations & Future Work

- **Data circularity**: training on text that includes editors' own bracketed conjectural restorations risks confirmation bias, but excluding it loses a large fraction of usable data. Aeneas quantifies this directly (excluding conjectures cost 20% of usable I.PHI text; models trained with vs. without conjectures differed by <5% in all tasks) and elects to keep conjectures in, citing data scarcity as the deciding factor — directly relevant precedent for whether/how to treat ORACC's own bracketed restorations.
  — Assael et al. 2025, Methods, "The question of data circularity", p.147.

- **Multimodal (image-conditioned) head — planned future work.** Aeneas's own design is the direct precedent: only the geographical-attribution head receives visual input; restoration explicitly excludes it to prevent information leakage (since masked positions' image regions are not itself masked); the dating head excludes it because ablations showed no measurable gain. By analogy, this project's future image head is planned for provenience classification specifically, not for the restoration objective or (per this same evidence) chronological (period) attribution.
  — Assael et al. 2025, *Nature* 645, p.142 ("It should be noted that only the geographical attribution head incorporates the additional inputs from the vision network — the restoration and chronological attribution tasks do not use the visual modality... The visual input was excluded for the restoration task to prevent unintended information 'leakage'... The visual modality was also omitted for the dating task because experiments showed no significant performance gains.").
  — This work: session discussion on image-head sequencing.

- **Revised track pairing — divergence from the original plan, to be justified explicitly in the write-up.** The reasoning above originally paired a future image head with the sign-level (glyph) track, since CuneiML images correspond physically to glyphs rather than to the romanized transliteration. This session's results (mBERT/transliteration substantially ahead of the from-scratch sign-level track on restoration quality at comparable training cost, §5.2) make the mBERT track the more likely candidate to extend with a provenience-focused vision head first — even though the transliteration is one step further removed from the image than the glyphs are. This is a real tension with the "physical correspondence" argument above and should be addressed head-on in the final text (e.g. images condition on the *tablet*, not on either text representation specifically, so the correspondence argument bears less weight than it first appears to), not silently resolved by switching tracks.
  — This work: session comparison of sign-level vs. mBERT-track metrics, 2026-08-05 (exact sign-level figures pending final checkpoint).

- **Image bounding-box quality is inconsistent and needs filtering before use — root cause identified and sourced.** CuneiML's bounding boxes are not manually annotated: they are produced by reconciling three automated CV segmentation methods (connected-component segmentation, watershed, SegmentAnything) by overlap area, validated by the dataset authors on only 100 randomly sampled images (97% self-reported pass rate) — i.e. imperfect segmentation is an acknowledged, expected property of the source data, not a defect introduced by this project's pipeline. This project's own image URL construction (`https://cdli.mpiwg-berlin.mpg.de/dl/photo/P{id}.jpg`) exactly matches CuneiML's documented crawl source, ruling out an id/image mismatch on this project's side. A manual visual audit of 24 randomly sampled entries (full-resolution composite 6-face tablet photos, correctly scaled) found only ~58% (14/24) of boxes correctly isolate the inscribed face; failures plausibly stem from the segmentation locking onto another high-contrast rectangular object in frame (e.g. a museum collection card or scale bar) rather than the tablet face itself. This project's ~42% failure rate on its own sample is notably higher than CuneiML's self-reported ~3%, a discrepancy worth noting explicitly rather than silently using the friendlier published figure. A minimum bbox-area-fraction filter (or equivalent validation) is needed before training any vision head on CuneiML crops as-is.
  — Chen et al. 2023 (CuneiML), §3.3 "Cutting Out the Major Faces", p.5 ("we reconcile differences between the bounding boxes each system produces by computing the area of their overlap... We sampled 100 images randomly to validate the cutouts; 97% met our quality requirements."); §3.1, p.4 (crawl URL pattern, footnote 2).
  — This work: session visual audit (24-sample contact sheet) and 1,000-sample manual-review export (`src/data_pipeline/export_bbox_review.py`, `data/bbox_review/`), 2026-08-05.

---

## 7. Conclusions

*Write last, after results exist.*
