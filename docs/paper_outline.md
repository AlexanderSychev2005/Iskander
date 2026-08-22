# Paper Outline & Sourced Talking Points

Target: a short NLP-conference paper (Introduction / Related Work / Data / Methods / Results / Conclusion — the standard shape, e.g. Lazar et al. 2021's own EMNLP paper, not a diploma). Length target: dense, like Aeneas/Ithaca/Lazar/Pythia — one paragraph per point, not a page. Checked directly against `aeneas.pdf`/`akkadian.pdf`: neither leans on large in-text tables — a dataset gets one dense paragraph with the essential numbers inline, and a full breakdown (if any) goes in supplementary material, not the main text. Follow that density here.

**Format:** each point below is `- Thesis. (Source Year, p.X)` — one claim, one citation, nothing else. Expand into prose only when actually drafting the paper.

**What we actually have to report:** two mBERT models (text-only; text+image where the image reaches only `provenience_head`) and an untrained baseline. No from-scratch architecture in the final results — abandoned as unproductive relative to mBERT, worth one sentence, not a section.

---

## 1. Introduction

- Cuneiform damage is the field's central bottleneck: about half of the ~500,000 excavated tablets remain untransliterated. (Cobanoglu et al. 2024, p.3, citing Streck 2010)
- Restoration is currently manual and subjective. (Lazar et al. 2021, §1, p.4682)
- MLM is not a bespoke architecture for this task — it *is* BERT's own pretraining objective, applied directly. (Lazar et al. 2021, §1, p.4683; Devlin et al. 2019)
- Motivation chain: Pythia → Ithaca (restoration + geographic/chronological attribution) → Aeneas (+ vision, arbitrary-length gaps) is a Greek/Latin lineage; Lazar et al. 2021 brought restoration alone to Akkadian, no attribution, no vision. This paper is the missing combination — multi-task attribution (period, genre, language, provenience) *and* vision, for Akkadian, for the first time. (This work — no prior Akkadian system combines the two)
- Historian+model synergy is the field's own framing for why this matters, not just a metrics exercise. (Assael et al. 2022, Abstract, p.280 — 25%→72% with model assistance; Assael et al. 2025, p.141-142)

---

## 2. Related Work

### 2.1 Fine-tuned BERT for Akkadian

- mBERT fine-tuned on Akkadian outperforms a from-scratch monolingual Akkadian BERT by ~10%. (Lazar et al. 2021, §5.4, p.4686)
- Mechanism: mBERT's 99 free `[unusedN]` slots reassigned to inject task tokens without growing the embedding table — this project reuses the same mechanism (2 slots for damage sentinels, plus 97 corpus-learned Akkadian WordPiece tokens in the rest). (Lazar et al. 2021, §4.2, p.4685; this work — `train_mbert.py`)
- Bidirectional context beats unidirectional on this exact task (LSTM full-context MRR 0.89 vs. context-before-only MRR 0.754) — the direct justification for a BERT-style encoder over an autoregressive decoder. (Fetaya et al. 2020, Table 2, p.22746)

### 2.2 Pythia → Ithaca → Aeneas

- Pythia: first deep-learning ancient-text restoration, BiLSTM, beat expert epigraphists (CER 30.1% vs. 57.3%). (Assael et al. 2019, p.6370)
- Ithaca: adds geographic (84 regions, 70.8% top-1/82.1% top-3) and chronological (median 3-year error) attribution to restoration (26.3% CER, 61.8% top-1) — one shared torso, per-task shallow feedforward heads. Direct precedent for this project's own multi-head design. (Assael et al. 2022, Abstract p.280, Table 1 p.282)
- Aeneas: T5+RoPE torso, vision branch (ResNet-8) concatenated with text — **only into the geographic-attribution head**; excluded from restoration (leakage risk) and dating (no measured gain). Direct precedent for scoping this project's own vision branch to `provenience_head` only — and this project independently re-derived the same conclusion via its own ablation, not just by copying the choice (§5). (Assael et al. 2025, Methods p.148)

### 2.3 Vision + cuneiform

- Tablet *shape* alone (no text), classified by ResNet50 on 94,936 CDLI images, predicts historical period at 61% macro-F1 (vs. 8% for a decision-tree height/width baseline) — real period signal exists in images alone. Contrast with this project's own null result for `period_head` in a *joint* text+image model: text likely already saturates the period signal a fused image feature could add, unlike a vision-only setting where the image is the only signal available. (Kapon, Fire & Gordin 2024, Abstract p.1, §1 p.2; this work — `docs/final_results.md`, period macro-F1 stays inside the unconditioned noise range [0.856, 0.876] whether or not the image is attached)
- Weakly-supervised cuneiform sign detection from tablet photos (aligning transliterations to images) is an established task, not a novel domain transfer. (Dencker et al. 2020, Abstract p.1)
- AI-assisted Gilgamesh restoration succeeds on formulaic/parallel-attested lines, struggles on rare lexemes and damaged proper names — matches this project's own showcase-vs-random accuracy gap (53-55% vs. 63-64% top-1). (Mahmood & Panok 2025, Abstract p.99; this work — `results_final/`)

### 2.4 Other Akkadian NLP (brief — different tasks, cited for field context only)

- Sign→word transliteration from Unicode glyphs, RNN, up to 97% accuracy. (Gordin et al. 2020, Abstract p.1)
- Akkadian→English NMT, both from glyphs and from transliteration (BLEU4 36.5/37.5). (Gutherz et al. 2023, Abstract p.1)
- A 2025 shared task on lemmatization/token-prediction for Akkadian and Sumerian confirms this is an active benchmarking target for the field, not a one-off framing choice. (Gordin, Sahala, Spencer & Klein 2025, Abstract p.164)
- Most recent (2026): bidirectional Akkadian↔English NMT + cuneiform rendering — different task/audience (composition, not restoration), doesn't overlap with this project's contribution. (Wang 2026, Abstract p.132)

### 2.5 Data sources

- CuneiML: supplementary source, Unicode glyph segmentation + CDLI photo/bbox data — basis of this project's vision branch. (Chen et al. 2023)
- eBL: formal citation for the Zenodo transliteration snapshot (`fragments.json`, DOI 10.5281/zenodo.10018951) used directly for text backfill and showcase fragments; ~25,000 tablets, 350,000+ lines, CC BY-NC-SA 4.0. (Cobanoglu et al. 2024, Abstract p.1, §2.2 p.4, §Reuse Potential p.5)

---

## 3. Data

- Only Test A. This project cannot build a Kyivan-style Test B (hide *real* editorial reconstructions, evaluate recovery): ORACC's raw markup distinguishes an editor's bracketed conjecture from a plainly-attested reading, but this project's parsing (`prepare_oracc.py`'s `extract_utf8`) does not preserve that distinction into the final corpus — by the time text reaches training, the two are indistinguishable. Retrofitting it is a new pipeline stage, out of scope. Report Test A only: standard 15% MLM masking on a held-out `test` split untouched during training/checkpoint selection. (This work — `src/data_pipeline/prepare_oracc.py`)
- Editorial conjectures are kept in training text, not stripped — same choice as Aeneas, same reason (data scarcity). (Assael et al. 2025, Methods, "data circularity", p.147)

**Corpus.** 60,050 documents: 51,974 from the primary ORACC + supplementary CuneiML merge, plus 8,076 from this project's own targeted backfill (CDLI bulk ATF/eBL rescue for tablets with text but no usable prior transliteration, balancing for underrepresented period/genre/language/provenience classes, and a curated Gilgamesh/Enuma Elish/Atrahasis/Hammurabi showcase set forced into the test split). 7,049 documents (11.7%) additionally have a reviewed photograph; the rest get an all-zero image placeholder at train time, the same design Aeneas uses for its own 95%-without-images majority. **The image is fed to the model for every one of these 7,049 tablets, but reaches only `provenience_head`** — say this explicitly here, not only in Methods. (This work — `docs/data_layout.md`)

**Metadata categories.** Raw CDLI/ORACC period/genre/provenience/language values are free text with inconsistent suffixes (e.g. `"Old Babylonian (ca. 1900-1600 BC)"`); mapped by substring match, not exact match, into a small fixed set per head — exact match silently dropped 30-58% of otherwise-valid records to "Unknown". Coarsened deliberately to keep each head's class count trainable at this corpus size, not to preserve every distinction CDLI itself makes (e.g. genre's "Literary & Scholarly" absorbs astrological/omen/ritual/mathematical/prayer texts alike):
  - Period (9): Neo-Assyrian, Ur III, Old Babylonian, Old Assyrian, Middle Assyrian, Middle Babylonian, Neo-Babylonian, Third Millennium, Late Antiquity.
  - Genre (6): Administrative, Lexical, Royal Inscriptions, Literary & Scholarly, Legal, Letters.
  - Language (4): Akkadian, Sumerian, Bilingual, Peripheral/Other.
  - Provenience (12): Nineveh, Umma, Girsu, Nippur, Puzriš-Dagan, Kanesh, Assur, Uruk, Ur, Ugarit, Sippar, Nimrud.

(This work — `prepare_hf_dataset.py`'s `map_period`/`map_genre`/`map_language`/`map_provenience`)

- A value matching no known category (or absent from the record) is `Unknown`, mapped to `-100` and excluded from both that head's loss and its accuracy/F1 — never defaulted. Current missing rate: period 2.0%, genre 11.5%, language 18.1%, provenience 23.5% (provenience has both the most classes and the least consistently filled catalogue field). (This work — `prepare_hf_dataset.py`'s `label_to_idx`)
- Two damage sentinels in the text itself, distinct from missing metadata: a single unclear sign (`x`, present in 32.3% of documents / 9.26% of word-tokens) and a gap of unknown length (`...`, 16.5% of documents / 2.01% of characters) — mapped to reserved mBERT token slots, excluded from masking targets and outputs alike. Same two-tier convention as Aeneas's own `-`/`#`. (Assael et al. 2025, Methods, p.148; this work — `train_mbert.py`'s `mark_damage_signals`)

---

## 4. Methods

- Torso: `bert-base-multilingual-cased`, fine-tuned, not from scratch — same reasoning as Lazar et al. 2021's own choice (corpus too small for a from-scratch transformer), confirmed empirically here too (untrained MLM MRR 0.512 → fine-tuned 0.797). (This work — `results_final/metrics_untrained.json` vs `metrics_text.json`)
- Four metadata heads (period/genre/language/provenience) jointly trained with the MLM objective on one shared encoder — extends Ithaca/Aeneas's 2-head design (region, date) to 4; no genre/language equivalent exists in their Greek/Latin setting. (This work — `MBertMultiTask`)
- Loss weighting MLM=3.0, each metadata head=1.0 — close to, not copied from, Aeneas's `3·restoration + 2·region + 1.25·date`. (Assael et al. 2025, Methods, p.148; this work — `--meta_weight`)
- Vision branch: ResNet18 (ImageNet-init, jointly fine-tuned), LayerNorm, concatenated with `[CLS]`, into `provenience_head` only — same principle as Aeneas's ResNet-8→geography design, different backbone (ResNet18/finetune vs. their ResNet8/from-scratch — this project's corpus is two orders of magnitude smaller, a from-scratch CNN overfit) and lighter augmentation (Aeneas's own 30°/10° rotation/shear clipped real content on this project's tight, human-reviewed crops; capped at 15°/5° after a visual check). (This work — `train_mbert.py`)

---

## 5. Results

**Metrics to report** (decided): MLM top-1/top-3/top-5 accuracy + MRR for restoration (matches Lazar et al. 2021's own metric choice exactly, direct comparability); accuracy + macro-F1 for each of the 4 metadata heads (macro-F1 is load-bearing given severe class imbalance — accuracy alone hides collapsed minority classes, see the untrained baseline). **Not** adding CER: it fits a character-generation setup (Ithaca/Aeneas/Kyivan all produce open-length text); this project's task is single-position top-1 prediction over a fixed vocabulary, already exactly Lazar's own MRR/Hit@k framing — CER would need new code (the from-scratch track's CER implementation was deleted with that track) for a metric that duplicates what MRR/Hit@k already show here.

- Headline: untrained / text-only / vision(provenience), all four heads + MLM MRR/Hit@k, test split. (This work — `results_final/README.md`)
- Provenience vision effect reproduced across 2 independent runs, clean of the noise floor established via `language_head` (never image-conditioned). The one genuinely new result beyond replicating prior work: independent re-derivation of Aeneas's own provenience-only finding, on a corpus ~30x smaller. (This work — `docs/final_results.md`)
- Per-class breakdown: the provenience gain concentrates in the weakest classes (Nimrud +0.241, Ur +0.102, Assur +0.034) while language stays flat to ±0.002 per class — finer, stronger evidence than the aggregate number. (This work — `results_final/metrics_{text,vision}.json`)
- Qualitative examples: `results_final/predictions_demo_showcase.md` (Gilgamesh, real photo + line-by-line cuneiform/transliteration/translation) is strong figure material, parallel to Ithaca's own Fig. 1/2 worked examples.

---

## 6. Conclusion

- First system combining Ithaca/Aeneas-style multi-task attribution with restoration *and* a vision-conditioned head, for Akkadian. Lazar et al. 2021 did restoration only; Aeneas did vision+attribution for Latin, at far larger scale, without genre/language.
- The provenience-vision result is cross-corpus, cross-architecture confirmation of Aeneas's own finding (~5.3k vs. ~176k images, ResNet18/finetune vs. ResNet8/from-scratch) — evidence the effect is a real property of the data, not one paper's pipeline artifact.
- State plainly: the from-scratch character-level architecture was tried and abandoned as unproductive relative to fine-tuned mBERT at comparable cost.

---

## Still open

1. **Dialect/date-bin framing vs. Kyivan's bins.** Not this paper's concern directly, but worth confirming in Methods that this project's categorical CDLI-derived buckets (not a from-scratch uniform binning) are stated as a deliberate choice, not left ambiguous.
2. **Venue/length target.** Determines how much of the above survives — decide before drafting full prose.
