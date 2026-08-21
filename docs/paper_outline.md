# Paper Outline & Sourced Talking Points

**Target:** a Kyivan-style short paper (see `Slavic Text Restoration`, Eremeev & Sychov, Overleaf — structure: Introduction / Related Work / Data / Methods / Results / Conclusion), not a diploma. This project's deliverable is a paper, not a thesis — `docs/thesis_outline*.md` were removed for this reason.

Every point below cites where it came from: a published source (author, year, exact page/section) or "this work" (a decision/finding made in this project, with the relevant file). Use these as talking points — write the actual prose yourself. Follows the same citation discipline as the (removed) diploma outline.

**What we actually have to report** (so the outline below stays grounded in real results, not aspirational ones):
- Two trained models, both mBERT (`bert-base-multilingual-cased`) fine-tuned jointly on masked-token restoration + 4 metadata heads (period, genre, language, provenience): a text-only model and a text+image model where the image reaches **only** `provenience_head`.
- A zero-shot / no-finetuning mBERT baseline (`results_final/metrics_untrained.json`).
- A controlled ablation establishing that the image only helps `provenience`, not `period`/`genre` (`docs/final_results.md`, `docs/ablation_runs/`).
- No custom from-scratch architecture in the final results — that track was built, compared, and abandoned as unproductive relative to mBERT at comparable cost (worth one sentence in Related Work/Discussion, not a whole section, since it did not ship).

---

## 1. Introduction

- Cuneiform's damage/loss problem is the same shape as the birch-bark problem Kyivan opens with (Zaliznyak 2004) — physical damage to a writing support creates lacunae that scholars currently fill in manually and subjectively. Akkadian's version of this: cuneiform tablets are clay, damaged by breakage, erosion, and in Nineveh's case the library's own destruction; CDLI/eBL both catalogue tens of thousands of tablets never fully transliterated because of exactly this bottleneck.
  — Cobanoglu et al. 2024, p.3 ("About 50 percent of the roughly half a million cuneiform tablets which have been excavated so far have not yet been transliterated or published" (Streck, 2010)).
  — Lazar et al. 2021, §1, p.4682 (manual, subjective, time-consuming restoration — same framing Kyivan already uses for Zaliznyak).

- MLM (masked language modeling) reframes "restore the missing sign" as BERT's own pretraining objective — not a bespoke architecture. Same argument Kyivan already makes for BERT/birch-bark; direct precedent specifically for Akkadian.
  — Lazar et al. 2021, §1, p.4683 ("we identify that the task of masked language modeling... lends itself directly to missing sign prediction").
  — Devlin et al. 2019, throughout (the MLM pretraining objective itself).

- Motivation chain to state explicitly (this is the paper's actual novelty claim): Pythia (Assael et al. 2019) → Ithaca (Assael et al. 2022, adds geographic + chronological attribution to restoration) → Aeneas (Assael et al. 2025, adds vision + arbitrary-length gaps) is a Greek/Latin epigraphy lineage; Lazar et al. 2021 is the one prior attempt to bring the *restoration* half to Akkadian, but never added attribution heads or vision. This project is the missing combination: Ithaca/Aeneas-style multi-task heads (restoration + period + genre + language + provenience) **and** Aeneas-style vision conditioning, applied to Akkadian for the first time.
  — This work: no prior Akkadian system combines multi-task attribution with a vision-conditioned head; Lazar et al. 2021 is restoration-only (confirmed by reading their Table 2 and Methods, §4-5, p.4684-4686 — no attribution task at all).

- Historian+model synergy framing (Kyivan's own closing paragraph of §1) — directly reusable, both Ithaca and Aeneas report it, worth restating for Akkadian even though this project itself didn't run a human-in-the-loop study.
  — Assael et al. 2022, Abstract, p.280 ("the use of Ithaca by historians improved their accuracy from 25% to 72%").
  — Assael et al. 2025, p.141-142 (Aeneas's own framing of historian-AI collaboration).

---

## 2. Related Work

### 2.1 MLM with fine-tuned BERT (direct Akkadian precedent)

- Lazar et al. 2021: mBERT fine-tuned on Akkadian transliteration outperforms a monolingual from-scratch Akkadian BERT by ~10%, and reassigns mBERT's 99 free `[unusedN]` slots to inject task-specific tokens without growing the embedding table — this project reuses that exact mechanism (2 slots for damage sentinels, not 99, since we don't need a full injected vocabulary the same way — see 2.2).
  — Lazar et al. 2021, §5.4, p.4686 ("the zero-shot performance of a multilingual language model surpasses... by about 10%"); §4.2, p.4685 (`[unusedN]` mechanism).
  — This work: `src/training/train_mbert.py` (`UNCLEAR_SIGN_TOKEN`/`UNKNOWN_GAP_TOKEN`, `learn_akkadian_tokens`/`inject_akkadian_tokens` — we actually go further than Lazar and inject 97 corpus-learned Akkadian WordPiece tokens into the remaining free slots, not just the 2 damage sentinels).

- Fetaya et al. 2020: direct Akkadian-domain evidence that bidirectional context beats unidirectional context on this exact task (LSTM with full context: MRR 0.89 vs. context-before-only: MRR 0.754) — the strongest single justification for a BERT-style encoder over an autoregressive decoder.
  — Fetaya et al. 2020, Table 2, p.22746.

### 2.2 Architecture-from-scratch lineage (Pythia → Ithaca → Aeneas)

- Pythia (Assael et al. 2019): first deep-learning ancient-text restoration system, BiLSTM seq2seq, Greek epigraphy, character+word level. Outperformed expert epigraphists (CER 30.1% vs. 57.3%) — establishes the field's own baseline for "does ML beat human experts on this task," which Ithaca/Aeneas both later beat again.
  — matches Kyivan's own §2.2 wording exactly; verify exact page in `papers/pythia.pdf` before citing.

- Ithaca (Assael et al. 2022): adds geographic attribution (84 regions, 70.8% top-1 / 82.1% top-3 accuracy) and chronological attribution (median 3 years from ground truth) to restoration (26.3% CER, 61.8% top-1, 78.3% top-20 accuracy) — the direct architectural precedent for this project's own multi-head design (restoration + period + genre + language + provenience, jointly trained on one torso). Torso: stacked transformer blocks with sparse multihead attention, position info concatenated into the input representation; three shallow feedforward task heads read off the torso's final output.
  — Assael et al. 2022, Abstract p.280, Table 1 p.282 (exact numbers above), p.281 (architecture, "the torso... three different task heads").
  — This work: `src/training/train_mbert.py`'s `MBertMultiTask` — same principle (one shared encoder, per-task linear heads), Aeneas's fixed multi-task loss weighting reused directly (see 2.3).

- Aeneas (Assael et al. 2025): replaces Ithaca's torso with a deep narrow T5 decoder + rotary position embeddings, adds a vision branch (ResNet-8) concatenated with text embeddings **only for the geographical attribution head** — restoration explicitly excludes vision (leakage risk: masked positions' own image region isn't itself masked), dating excludes it because ablations showed no gain. This is the direct precedent for this project's own decision to scope the vision branch to `provenience_head` only, and — going further than just following the precedent — this project independently re-derived the same conclusion via its own controlled ablation (§5 below), on a corpus ~30x smaller and a different vision backbone (ResNet18/ImageNet-finetune vs. their ResNet8/from-scratch).
  — Assael et al. 2025, Methods p.148 ("only the geographical attribution head incorporates the additional inputs from the vision network... excluded for the restoration task to prevent unintended information 'leakage'... omitted for the dating task because experiments showed no significant performance gains").
  — This work: `docs/final_results.md` (own ablation, provenience macro-F1 0.725→0.768 reproduced across 2 runs; period/genre stayed inside the noise floor across 4 runs — see §5).

### 2.3 Vision + cuneiform (new for this paper vs. the Kyivan draft — no image modality in Kyivan at all)

- Kapon, Fire & Gordin 2024: 94,936 CDLI tablet images classified by physical *shape* alone (no text) into historical period via ResNet50, reaching 61% macro-F1 (vs. 8% for a height/width-ratio decision-tree baseline) — direct evidence that tablet **shape carries real period signal** through a purely visual channel. Worth an explicit discussion-section contrast: their shape-only model finds a real period signal from images; this project's *joint* text+image model finds none for `period_head`. Plausible reconciliation: text alone (via language/orthography/genre cues mBERT already reads) may already saturate what period-signal exists, leaving nothing incremental for a jointly-fused image feature to add on top — unlike Kapon et al.'s pure-vision setting, where image is the *only* signal available, so pulling out even weak shape-period correlation shows up as a real gain there but not in a model that already has a much stronger channel (text) providing the same information.
  — Kapon, Fire & Gordin 2024 (arXiv:2406.04039), Abstract p.1 (61% macro-F1, ResNet50, 94,936 images); §1 p.2 (8% decision-tree baseline, height-to-width ratio).
  — This work: `docs/final_results.md` (period macro-F1 unconditioned range [0.856, 0.876] across 4 runs vs. conditioned 0.871/0.868 — inside the noise floor, no effect).

- Dencker, Klinkisch, Maul & Ommer 2020 (PLOS ONE): weakly-supervised cuneiform *sign detection* in tablet photographs, aligning transliterations to images to bootstrap bounding-box training data without manual annotation — establishes that CDLI-scale tablet photography is usable for real computer-vision tasks on cuneiform specifically, not just a generic-object-detection domain transfer. Useful precedent for the image side of this project's own pipeline (ResNet on tablet photo crops), and a candidate future-work citation (their sign-localization output could in principle feed sign-level, not just tablet-level, visual features).
  — Dencker et al. 2020, Abstract p.1, §Introduction p.1 (weak supervision via transliteration alignment).

- Mahmood & Panok 2025 (Iraqi Literary and Cultural Review): AI-assisted reconstruction/translation specifically of Standard Babylonian Gilgamesh — reports the system "performs strongly on relatively formulaic lines and on passages with close parallels elsewhere in the epic... struggles... with rare lexical items, damaged proper names, and lines whose syntactic structure is uncertain." This independently corroborates this project's own qualitative finding in `results_final/predictions_demo_showcase.md`: restoration top-1 accuracy on the Gilgamesh/showcase set (53-55%) is meaningfully lower than on the random administrative-heavy test sample (63-64%), for exactly this reason (literary/damaged vs. formulaic/administrative).
  — Mahmood & Panok 2025, Abstract p.99 ("performs strongly on relatively formulaic lines... struggles... with rare lexical items, damaged proper names").
  — This work: `results_final/README.md` / `results_final/predictions_demo.md` vs. `predictions_demo_showcase.md` (aggregate top-1 accuracy figures).

### 2.4 Data sources (cite as data provenance, not "related work" per se)

- Chen et al. 2023 (CuneiML): supplementary source for Unicode cuneiform glyph segmentation and CDLI photo/bounding-box data — the basis for this project's vision branch.
  — This work: `data/raw/cuneiml/CuneiMLv1.2.json`; `src/data_pipeline/prepare_cuneiml.py`.

- Cobanoglu et al. 2024 (eBL data paper, *Journal of Open Humanities Data*): formal citation for the eBL Zenodo transliteration snapshot (`fragments.json`, Zenodo DOI 10.5281/zenodo.10018951) this project uses directly for text backfill and several showcase (Gilgamesh/Enuma Elish/Atrahasis/Hammurabi) fragments. ~25,000 tablets transliterated, 350,000+ lines, CC BY-NC-SA 4.0.
  — Cobanoglu et al. 2024, Abstract p.1, §2.2 p.4 (dataset description), §Reuse Potential p.5 (350,000 lines, license).
  — This work: `data/raw/cdli_bulk/ebl_fragments.json`; `src/data_pipeline/add_showcase_texts.py`.

---

## 3. Data

- Two-source corpus, same "combine an editorial-transcription source with a supplementary glyph/image source" shape as Kyivan's own multi-collection table (NKRYA/birch-bark/Epigraphica/Polotsk/other) — worth a similar table here: ORACC (primary, catalogue metadata) + CuneiML (supplementary, glyphs/photos) + this session's CDLI-bulk-ATF/eBL backfill (targeted weak-class balancing, forced-split showcase texts). Final corpus size, per-split counts, per-category (period/genre/language/provenience) distributions: pull fresh from `docs/final_results.md` / `docs/data_layout.md` rather than reusing any number from the deleted diploma outline (corpus grew substantially after that was written).
  — This work: `docs/data_layout.md`, `docs/final_results.md`.

- Damage-marker convention: this project's `x` (one unclear sign) / `[#]`-equivalent (unknown-length gap, via `UNKNOWN_GAP_TOKEN`) is structurally the same two-tier convention Kyivan uses (`[-]` single character, `[#]` unknown-length gap) and that Aeneas uses (`-` / `#`) — independently convergent design across three unrelated ancient-language restoration projects, worth stating as such rather than "we copied it."
  — Assael et al. 2025, Methods, "Latin Epigraphic Dataset", p.148.
  — This work: `src/training/train_mbert.py` (`mark_damage_signals`, `ELLIPSIS_RE`, `LONE_X_RE`).

- Unlike Kyivan's Test A/Test B split (synthetic masking vs. real editorial-reconstruction masking), this project only reports one test regime: standard random 15% MLM masking, evaluated once on a genuinely held-out `test` split never touched during training or checkpoint selection (`results_final/`). Worth deciding explicitly whether the paper should add a Kyivan-style Test B (hide real ORACC/CDLI editorial bracket-restorations specifically, not synthetic masks) — this project has the raw material for it (bracketed conjectural restorations are already flagged during ORACC parsing, see the data-circularity discussion below) but has not built that second eval regime. **Open question, not yet decided — see "What we still need to decide" below.**

- Data circularity (training on text that includes editors' own bracketed conjectures) — Aeneas quantifies this directly for their own corpus and elects to keep conjectures in, citing data scarcity. Directly relevant if this paper adds a Test-B-style real-reconstruction eval, since editorial conjectures are exactly what such an eval would be hiding and re-predicting.
  — Assael et al. 2025, Methods, "The question of data circularity", p.147.

---

## 4. Methods

- Torso: `bert-base-multilingual-cased`, fine-tuned (not from scratch) — this is the one significant divergence from Kyivan's own from-scratch character-level torso, and worth explicitly justifying rather than glossing over: Akkadian's available corpus (tens of thousands of lines) is far smaller than what a from-scratch transformer needs, matching exactly why Lazar et al. 2021 chose mBERT over a monolingual model (§2.1 above) — this project made and empirically confirmed the same choice for the *same reason*, not by default.
  — This work: `results_final/metrics_untrained.json` vs `metrics_text.json` (MLM MRR 0.512 zero-shot → 0.797 fine-tuned — shows fine-tuning is doing real, substantial work on top of mBERT's own pretraining, consistent with Lazar's framing).

- Four metadata heads (period/genre/language/provenience) trained jointly with the MLM objective, one shared encoder — direct analog of Ithaca/Aeneas's own multi-head torso design (§2.2), extended from their 2 attribution tasks (region, date) to 4 (this project has no genre/language equivalent in Ithaca/Aeneas, since those are Akkadian-specific — Ithaca's ancient Greek corpus doesn't carry the Sumerian/Akkadian bilingual-corpus language-attribution problem, and it has no separate genre head at all).
  — This work: `src/training/train_mbert.py` (`MBertMultiTask`).

- Loss weighting: MLM=3.0, each metadata head=1.0 (`meta_weight`, raised this session from an earlier 0.25 after finding it under-weighted metadata relative to Aeneas's own fixed ratio) — numerically close to, not copied from, Aeneas's own `3·restoration + 2·region + 1.25·date`.
  — Assael et al. 2025, Methods, p.148.
  — This work: `src/training/train_mbert.py` (`--meta_weight`; see `docs/final_results.md` for the before/after metadata macro-F1 comparison).

- Vision branch: ResNet18 (ImageNet-initialized, jointly fine-tuned — `vision_init=finetune`), LayerNorm'd, concatenated with `[CLS]`, feeding **only** `provenience_head` — architecturally identical in spirit to Aeneas's ResNet-8-into-geography-head design (§2.2), differing in backbone choice (ResNet18/finetune vs. their ResNet8/from-scratch, since this project's ~5.3k-image corpus is two orders of magnitude smaller than Aeneas's and a full from-scratch CNN was found to overfit — matches Kapon et al.'s own point that a fully from-scratch shape classifier needs real data volume to work, §2.3) and augmentation strength (calibrated down from Aeneas's own 30°/10° rotation/shear after finding it clipped real tablet content on tight, human-reviewed crops — see `docs/final_results.md`/git history for the visual audit that motivated this).
  — This work: `src/training/train_mbert.py` (`IMG_TRANSFORM_TRAIN`, `MBertMultiTask.forward`); commits on ImageNet normalization/LayerNorm/augmentation tuning (session 2026-08-13).

---

## 5. Results

Pull exact numbers from `results_final/` (test split) — do not reuse validation-split numbers quoted earlier in the project's history, and do not reuse the pre-backfill numbers that were in the deleted diploma outline's old §5.2 (corpus has grown substantially since).

- Headline comparison table: untrained / text-only / vision(provenience), all four metadata heads' accuracy+macro-F1, MLM MRR/Hit@k. Source: `results_final/README.md`'s own table (already built, just needs transcribing into the paper's own table format, Kyivan-style).
- Provenience vision effect, reproduced across 2 independent runs, clean separation from the noise floor established via `language_head` (never image-conditioned in any run) — full evidence and exact noise-floor reasoning in `docs/final_results.md`. This is the paper's one genuinely new empirical result beyond replicating Ithaca/Aeneas/Lazar: not "we followed Aeneas's architecture," but an independent re-derivation of the *same* provenience-only conclusion via a controlled, noise-floor-aware ablation, on a corpus far smaller than Aeneas's own.
- Per-class breakdown (where the provenience gain concentrates: weak/small classes — Nimrud +0.241, Ur +0.102, Assur +0.034 — while language stays flat to ±0.002 per class) is a stronger, finer-grained piece of evidence than the aggregate number alone and is worth its own table or figure, not just prose — already computed, see `docs/final_results.md` §5.4 (in the now-deleted diploma outline; re-pull the underlying numbers from `results_final/metrics_text.json` / `metrics_vision.json`'s per-class sections directly for the paper, don't cite the deleted file).
- Qualitative restoration examples: `results_final/predictions_demo.md` / `predictions_demo_showcase.md` are exactly the kind of worked examples Kyivan's own eventual Table (`tab:testab`) and prose around it want to be paired with — the showcase file in particular (Gilgamesh, real photo + line-by-line cuneiform/transliteration/translation) is strong figure material for the paper, directly parallel to Ithaca's Fig. 1/Fig. 2 worked-example figures and to Kyivan's own planned pipeline figure.

---

## 6. Conclusion (draft talking points, expand into prose last)

- This is, to our knowledge, the first system to combine Ithaca/Aeneas-style multi-task attribution (period, genre, language, provenience) with masked-token restoration **and** a vision-conditioned head, for Akkadian specifically — Lazar et al. 2021 did restoration only; Aeneas did vision+attribution but for Latin, at a much larger data scale, and without a genre/language task.
- The provenience-vision result is not just "we replicated Aeneas's architectural choice" — it is an independent confirmation, via a controlled ablation with an explicit noise floor, on a completely different corpus scale (~5.3k vs. ~176k images) and vision backbone, that the effect Aeneas reported for Latin geography also holds for Akkadian provenience. Cross-corpus, cross-architecture agreement is itself evidence the effect is a real property of the data (tablet material/shape/photographic convention correlating with findspot), not an artifact of one paper's specific pipeline.
- Open honest limitation to state, not hide: the from-scratch character-level architecture (the Kyivan-style track) was tried for this project and abandoned as unproductive relative to fine-tuned mBERT at comparable training cost — worth one sentence acknowledging this was tested, not omitted from consideration.

---

## What we still need to decide (not yet resolved — flagged, not silently assumed)

1. **Test-B-style real-reconstruction eval.** Kyivan reports two test regimes (synthetic masking vs. real editorial reconstructions); this project currently reports only one (synthetic 15% masking on a held-out split). Building a real Test B is feasible (ORACC's own bracketed conjectural restorations are already visible during parsing) but not yet done. Decide: worth building for the paper, or explicitly out of scope?
2. **CER metric.** Kyivan reports CER (Levenshtein-based, per-gap) alongside top-k accuracy; this project's `results_final/` currently reports only top-k/MRR (matching Lazar et al.'s own metric set), not CER. `src/analysis/evaluate.py`'s CER code existed for the now-deleted from-scratch track — check whether it's reusable for the mBERT models or needs rebuilding, and decide whether CER is worth adding to match Kyivan's own metric table shape.
3. **Dialect/date-bin framing.** Kyivan bins dates into 20 fifty-year bins and dialects into 4 macro-dialects, each with an explicit "excluded from loss if unreliable" masking convention. This project's period/provenience/genre/language heads use a different labeling scheme (CDLI/ORACC's own category strings, mapped via `map_period`/`map_genre`/etc., not a from-scratch binning system) — worth a sentence in Methods contrasting the two approaches (categorical CDLI period buckets vs. Kyivan's own uniform 50-year bins) rather than silently presenting them as equivalent.
4. **Author list / venue.** Not this document's call — flagging only because the paper's framing (co-authored, Kyivan-style) implies a specific venue/audience that should shape how much of the above actually gets included vs. trimmed for length.
5. **Papers read but not yet deeply cited above** — `papers/2023_gutherz_akkadian_english_nmt.pdf` (Akkadian→English NMT, a different task, maybe one sentence in Related Work as "other Akkadian NLP work"), `papers/2020_gordin_reading_akkadian_nlp.pdf` (general Akkadian NLP overview), `papers/2025_evacun_shared_task.pdf` (a cuneiform NLP shared task — could support an "the field has organized shared tasks on this exact problem" framing), `papers/2026_tabletcraft_akkadian_nmt.pdf` (very recent, worth a skim to make sure nothing here scoops/duplicates this project's own contribution). Not yet read closely enough to cite with page numbers — say so if used rather than guessing a page.
