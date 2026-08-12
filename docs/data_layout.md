# Data layout

Two real tiers: **raw** (untouched external sources) and **ready for
training** (what `train_mbert.py` actually loads, pushed to
`AlexSychovUN/Iskander-Dataset` on HF Hub). Everything else is regeneratable
build machinery in between.

## raw/ -- sources, never modified by our own code
- `cuneiml/` -- CuneiML v1.2 JSON export (text + image URLs)
- `oracc/` -- ORACC project zips (corpusjson + catalogue per project)
- `cdli_data/cdli_cat.csv` -- CDLI's own metadata catalogue (period/genre/
  provenience/language/photo_up flag), used to enrich CuneiML+bulk entries
- `cdli_bulk/` -- CDLI's official bulk ATF dump (github.com/cdli-gh/data,
  2022 snapshot) + eBL's Zenodo fragment snapshot (2023) -- used to backfill
  tablets CuneiML never captured (session 2026-08-12, see
  prepare_cdli_bulk.py's docstring)
- `cuneiform_unicode_vocab/` -- vendored sign-list from CuneiML's own ATF->
  Unicode converter (CC0), used by `cuneiform_unicode.py`

## interim/ -- per-source parsed, not yet merged (regenerate via
prepare_cuneiml.py / prepare_oracc.py / prepare_cdli_bulk.py / etc.)
- `cuneiml.jsonl`, `oracc.jsonl` -- one row per transliterated line
- `{cdli_bulk,ebl_bulk,balance,text_balance}_documents.jsonl` -- session
  2026-08-12's backfills, one row per *tablet* already (not per line),
  verified through CuneiML's own sign parser + deduped against the main
  corpus (see reprocess_bulk_documents.py)

## processed/ -- merged, split, ready
- `combined_unique.jsonl` -- cuneiml.jsonl + oracc.jsonl merged and
  deduplicated by sign-string (prepare_hf_dataset.py)
- `hf_dataset` -- line-level "default" config. Not used for training
  directly anymore, but IS the split-assignment authority every downstream
  document/vision config reuses (`tablet_split_map()`) -- don't delete this
  even though nothing trains on it directly.
- `hf_dataset_documents` -- one row per tablet, CuneiML+ORACC only (base,
  before this session's backfill)
- `hf_dataset_documents_with_cdli_bulk` -- **the actual training config**
  ("documents" on the Hub): base + the 4 interim backfill files above
- `hf_dataset_vision` -- **the actual training config** ("vision" on the
  Hub): one row per photographed tablet, provenience-only relevant now
  (train_mbert.py only feeds images into the provenience head)
- `label_configs.json` -- head sizes, read by train_mbert.py

## vision_dataset/ -- curated but uncropped (raw tier for images)
Per-class collected photos + `manifest.jsonl` (bbox review state) +
`../bbox_corrections.jsonl`. Keep -- needed if a bbox ever needs
re-reviewing or re-cropping.

## vision_dataset_final/ -- ready
Exact-bbox crops, letterboxed to 224x224. `bboxes.csv` +
`crops_manifest.jsonl` + one `<id>.jpg` per reviewed tablet. This is what
`build_vision_hf_dataset.py` turns into `hf_dataset_vision`.

## Pipeline order (only needed to add more data later)
1. `prepare_cuneiml.py` / `prepare_oracc.py` -> interim/*.jsonl
2. `prepare_hf_dataset.py` -> combined_unique.jsonl + hf_dataset (split authority)
3. `prepare_document_dataset.py` -> hf_dataset_documents
4. `prepare_cdli_bulk.py` / `backfill_balance.py` / `backfill_text_balance.py`
   -> new interim/*_documents.jsonl (raw ATF text + signs=[])
5. `reprocess_bulk_documents.py` -> same files, now with real signs +
   dedup-verified (session 2026-08-12 finding: skip this and ~20% of what
   you added is noise or duplicate)
6. `add_cdli_bulk_documents.py` -> hf_dataset_documents_with_cdli_bulk
7. `collect_vision_dataset.py` -> vision_dataset/ (needs manual bbox review:
   `review_bboxes_gui.py`) -> `export_bbox_csv.py` -> `finalize_vision_crops.py`
   -> vision_dataset_final/
8. `build_vision_hf_dataset.py` -> hf_dataset_vision
9. `push_dataset.py --config_name documents` / `--config_name vision`
