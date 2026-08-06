import os
import random
import re
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import argparse
import json
import logging
from datetime import datetime
from sklearn.metrics import f1_score
import numpy as np
from PIL import Image
from torchvision import models as tv_models
from torchvision import transforms as tv_transforms
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback,
)
from datasets import load_from_disk, load_dataset
from tokenizers import Tokenizer as RawTokenizer
from tokenizers.models import WordPiece as WordPieceModel
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Vision branch (--use_image): only period/genre/provenience get a picture
# (language has no plausible visual signal -- session discussion,
# 2026-08-06). Follows Aeneas's own mechanism (Assael et al. 2025, Methods
# p.148): a CNN feature vector is concatenated with the text embedding
# before the head. Unlike this project's earlier frozen-backbone pilot
# script (train_mbert_vision.py), this is the real joint end-to-end
# training Aeneas actually did -- "batch size of 1,024 text-image pairs"
# (p.9) means every example carried an image slot in a single training run
# over the whole corpus, not a separate post-hoc stage over only the
# image-bearing subset. Tablets without a collected photo (the overwhelming
# majority of the corpus) get an all-zero placeholder image so batch
# tensors stay uniformly shaped -- no explicit missing-modality flag is
# needed, the head just learns a near-constant image contribution for
# those rows since the input carries no information.
IMAGE_HEADS = ("period", "genre", "provenience")
IMG_SIZE = 224  # ResNet18's input size, matches Aeneas's own (Methods p.148) and finalize_vision_crops.py's stored size
# Per-class image counts (~150-300) are the same order of magnitude as
# Aeneas's own average (~8,843 images / 62 provinces =~ 142/class), not
# orders of magnitude smaller -- but a ResNet trained fully from scratch at
# that scale still overfits easily without the augmentation Aeneas explicitly
# used to fight it ("image augmentations such as zooming, rotation, and
# adjustments to brightness and contrast", Methods p.148). No horizontal
# flip: unlike a generic object photo, a mirrored tablet face has reversed
# sign order/orientation, which is not a valid input for this task. Eval
# stays deterministic (no augmentation) so checkpoint comparisons aren't
# noisy -- see TiedWeightSafeTrainer's eval_data_collator override below.
IMG_TRANSFORM_TRAIN = tv_transforms.Compose([
    tv_transforms.Resize((IMG_SIZE, IMG_SIZE)),
    tv_transforms.RandomRotation(10),
    tv_transforms.ColorJitter(brightness=0.2, contrast=0.2),
    tv_transforms.ToTensor(),
])
IMG_TRANSFORM_EVAL = tv_transforms.Compose([tv_transforms.Resize((IMG_SIZE, IMG_SIZE)), tv_transforms.ToTensor()])

# The two real-damage signals that survive into the 'text' column (see
# prepare_hf_dataset.py's clean_transliteration): a standalone 'x' is one
# unclear sign, '...' is a lacuna of unknown length. Neither has its own
# WordPiece token in stock mBERT, so left alone they'd tokenize into ordinary
# maskable subwords -- unlike our sign-level 'x'/'X'/'[#]', mBERT would then
# be trained to "restore" positions that have no real answer. We reuse two of
# mBERT's 99 reserved [unusedN] vocab slots as dedicated sentinels (same
# trick Lazar et al. 2021 use for their own injected tokens) so they get
# registered as genuine special tokens -- HF's own masking collator already
# excludes anything in additional_special_tokens from masking targets, so no
# custom collator logic is needed once the substitution and registration are
# done.
LONE_X_RE = re.compile(r"\bx\b")
ELLIPSIS_RE = re.compile(r"\.\.\.+")
UNCLEAR_SIGN_TOKEN = "[unused1]"
UNKNOWN_GAP_TOKEN = "[unused2]"

def mark_damage_signals(text):
    text = ELLIPSIS_RE.sub(f" {UNKNOWN_GAP_TOKEN} ", text)
    text = LONE_X_RE.sub(UNCLEAR_SIGN_TOKEN, text)
    return re.sub(r"\s+", " ", text).strip()

def learn_akkadian_tokens(texts, existing_vocab, n_tokens=97, target_vocab_size=8000, min_frequency=10):
    """Reproduce Lazar et al. 2021's other free-token trick: "we assign its
    99 available free tokens, optimizing for maximum likelihood by the
    WordPiece tokenization algorithm" -- they never published the exact
    token list, so we relearn it here by training a fresh WordPiece
    vocabulary on our own Akkadian transliteration corpus and keeping the
    highest-frequency pieces mBERT doesn't already have. Without this,
    Akkadian-specific sign sequences get chopped into excessive fragments by
    mBERT's stock (mostly-modern-language) WordPiece vocab."""
    tok = RawTokenizer(WordPieceModel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=target_vocab_size, min_frequency=min_frequency,
        special_tokens=["[UNK]"], continuing_subword_prefix="##",
    )
    tok.train_from_iterator(texts, trainer=trainer)
    learned = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])  # id order ~ frequency rank
    candidates = [t for t, _ in learned if t not in existing_vocab and t != "[UNK]" and t.replace("##", "")]
    return candidates[:n_tokens]

def inject_akkadian_tokens(tokenizer, new_tokens, first_free_slot=3):
    """Rename mBERT's unused [unusedN] vocab slots (N >= first_free_slot,
    since 1-2 are already claimed by our damage sentinels) to the learned
    Akkadian tokens, keeping the same embedding row id -- no vocab growth,
    no resize_token_embeddings() needed, exactly the slot-reuse mechanism
    Lazar et al. 2021 describe.

    In this transformers version BertTokenizer's `.vocab` is a detached
    snapshot dict -- mutating it in place does not reach the actual Rust
    WordPiece tokenizer used for encoding (verified: get_vocab() and real
    tokenization both stay unchanged). The only reliable way to rename a
    slot is to rebuild the tokenizer from a modified vocab dict, so this
    returns a *new* tokenizer instance rather than mutating in place."""
    from transformers import BertTokenizer
    vocab = dict(tokenizer.get_vocab())
    n_injected = 0
    for i, new_tok in enumerate(new_tokens):
        slot = f"[unused{i + first_free_slot}]"
        if slot not in vocab:
            break
        vocab[new_tok] = vocab.pop(slot)
        n_injected += 1

    new_tokenizer = BertTokenizer(
        vocab=vocab, do_lower_case=tokenizer.do_lower_case,
        unk_token=tokenizer.unk_token, sep_token=tokenizer.sep_token,
        pad_token=tokenizer.pad_token, cls_token=tokenizer.cls_token,
        mask_token=tokenizer.mask_token,
        tokenize_chinese_chars=tokenizer.tokenize_chinese_chars,
        strip_accents=tokenizer.strip_accents,
    )
    return new_tokenizer, n_injected

class TiedWeightSafeTrainer(Trainer):
    """BertForMaskedLM ties cls.predictions.decoder.weight/bias to
    bert.embeddings.word_embeddings.weight/bert.embeddings... (standard
    tied-embeddings MLM head), so state_dict() has two keys aliasing the
    same tensor storage. A real PreTrainedModel's own save_pretrained()
    de-duplicates this automatically before writing safetensors;
    MBertMultiTask is a plain nn.Module wrapper, so Trainer routes through
    the generic "not a PreTrainedModel" save path in this transformers
    version, which calls safetensors.torch.save_file() directly on the raw
    state_dict with no format fallback (TrainingArguments.save_safetensors
    no longer exists to opt out). Cloning each tensor gives every key its
    own storage, satisfying safetensors' shared-memory check -- the live
    model's actual weight tying during training is untouched, this only
    affects what gets written to disk."""
    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict = {k: v.clone() for k, v in state_dict.items()}
        super()._save(output_dir, state_dict=state_dict)

    # --use_image trains with augmented crops (see IMG_TRANSFORM_TRAIN) but
    # eval should stay deterministic so checkpoint comparisons (early
    # stopping, with-vs-without-image deltas) aren't noisy from random
    # rotation/color jitter. HF's Trainer only takes one data_collator
    # constructor arg, so swap it in for the duration of eval only.
    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_collator = eval_data_collator

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original

class LogToFileCallback(TrainerCallback):
    # report_to="none" leaves Trainer's default PrinterCallback printing
    # step/eval metrics straight to stdout (bypasses the `logging` module),
    # so the FileHandler on the module logger never sees them -- same fix
    # applied to train.py's LogToFileCallback.
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logging.getLogger(__name__).info(f"step {state.global_step}: {logs}")

# mBERT baseline, following Lazar et al. 2021's finding that a pretrained
# multilingual model finetuned on Akkadian outperforms a from-scratch model
# at their data scale. Trained on the transliteration ('raw') side of the
# corpus (data/processed/hf_dataset_translit), since mBERT's WordPiece
# vocabulary has no cuneiform Unicode signs. Same joint MLM + 4 metadata
# classification heads recipe as AkkadianModel (src/training/train.py), so
# the two runs are comparable apart from the backbone itself.

class MBertMultiTask(nn.Module):
    def __init__(self, model_name, num_period, num_genre, num_language, num_provenience, meta_weight=1.0,
                 use_image=False, vision_init="scratch", img_feat_dim=128):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.use_image = use_image
        self.meta_weight = meta_weight

        head_in = hidden_size + img_feat_dim if use_image else hidden_size
        self.period_head = nn.Linear(head_in, num_period)
        self.genre_head = nn.Linear(head_in, num_genre)
        self.language_head = nn.Linear(hidden_size, num_language)  # never sees the image
        self.provenience_head = nn.Linear(head_in, num_provenience)

        if use_image:
            # scratch (default): random init, fully trainable -- matches
            # Aeneas's own from-scratch ResNet-8 (ref. 82 there is just the
            # general He et al. residual-block paper, not a checkpoint);
            # our image count is the same order of magnitude as theirs
            # (~5.3k vs ~8.8k = 5% of their 176,861-inscription corpus).
            # pretrained: frozen ImageNet ResNet18, only vision_proj trains
            # -- kept as an A/B option, not the default (domain gap between
            # ImageNet photos and tablet macro shots makes transfer uncertain).
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if vision_init == "pretrained" else None
            resnet = tv_models.resnet18(weights=weights)
            if vision_init == "pretrained":
                for p in resnet.parameters():
                    p.requires_grad = False
            resnet.fc = nn.Identity()
            self.vision_cnn = resnet
            self.vision_proj = nn.Linear(512, img_feat_dim)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None,
                period_labels=None, genre_labels=None, language_labels=None, provenience_labels=None):
        bert_out = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = bert_out.last_hidden_state
        mlm_logits = self.backbone.cls(seq)

        cls_embed = seq[:, 0, :]
        if self.use_image:
            img_feat = self.vision_proj(self.vision_cnn(pixel_values))
            head_in = torch.cat([cls_embed, img_feat], dim=-1)
        else:
            head_in = cls_embed
        period_logits = self.period_head(head_in)
        genre_logits = self.genre_head(head_in)
        language_logits = self.language_head(cls_embed)
        provenience_logits = self.provenience_head(head_in)

        loss = None
        if any(l is not None for l in [labels, period_labels, genre_labels, language_labels, provenience_labels]):
            loss_mlm_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
            loss_meta_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = 0.0

            # MLM=3.0 (matches AkkadianModel); meta_weight defaults to 1.0 -- raised
            # from the original 0.25 (MLM:meta = 12:1) after comparing against
            # Aeneas' own multi-task loss weights (restoration=3, region=2,
            # date=1.25 -- roughly 3:2, not 12:1), and noting our metadata heads
            # (esp. provenience, period) were still climbing when MLM had
            # plateaued. Configurable via --meta_weight for further tuning.
            if labels is not None and (labels != -100).any():
                loss += 3.0 * loss_mlm_fct(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))

            for logits, lbl in [(period_logits, period_labels), (genre_logits, genre_labels),
                                 (language_logits, language_labels), (provenience_logits, provenience_labels)]:
                if lbl is not None and (lbl != -100).any():
                    loss += self.meta_weight * loss_meta_fct(logits, lbl)

        return {
            "loss": loss,
            "logits": mlm_logits,
            "period_logits": period_logits,
            "genre_logits": genre_logits,
            "language_logits": language_logits,
            "provenience_logits": provenience_logits,
        }

def build_tablet_image_index(crops_dir, reviewed_only=True):
    """tablet_id (CDLI "P######" form, matching prepare_hf_dataset.py's
    to_examples()) -> PIL.Image (opened eagerly, not a lazy path -- keeps
    _load_image agnostic to whether the index came from local files or
    build_tablet_image_index_from_hf). Only used when --use_image; ids
    outside this index (the overwhelming majority of the corpus -- images
    exist for a small collected subset, not every tablet) fall back to an
    all-zero placeholder in the collator, same as Aeneas's own training
    (every example carries an image slot, real or not)."""
    manifest_path = os.path.join(crops_dir, "crops_manifest.jsonl")
    index = {}
    if not os.path.exists(manifest_path):
        return index
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if reviewed_only and not row.get("reviewed"):
                continue
            raw_id = str(row["id"]).strip()
            tablet_id = "P" + raw_id.zfill(6) if raw_id.isdigit() else raw_id
            path = os.path.join(crops_dir, f"{row['id']}.jpg")
            if os.path.exists(path):
                try:
                    index[tablet_id] = Image.open(path).convert("RGB")
                except Exception:
                    pass
    return index

def build_tablet_image_index_from_hf(repo_id):
    """Same tablet_id -> PIL.Image mapping as build_tablet_image_index, but
    pulled straight from the "vision" HF config instead of a local crops
    folder -- so a training box only needs `git pull` + this script, no
    scp'ing image folders around. All three splits are loaded and merged
    (train/validation/test): the index is purely "does this tablet have a
    photo", split-membership is already handled by --data_dir's own splits."""
    from datasets import load_dataset as _load_dataset
    vision_ds = _load_dataset(repo_id, "vision")
    index = {}
    for split in vision_ds:
        for row in vision_ds[split]:
            index[row["tablet_id"]] = row["image"].convert("RGB")
    return index

def mark_one_line_per_tablet(dataset):
    """Adds an "image_tablet_id" column: equal to "tablet_id" for exactly
    one (the first-encountered) line of each tablet, "" for every other
    line of that same tablet. TRAIN-only fix for a real skew (session
    finding, 2026-08-06): without this, a tablet's photo is shown to the
    model once per LINE it has (this corpus: up to 407, avg 13, median 8),
    and that count varies systematically by class -- e.g. provenience
    Assur averages 23 lines/tablet vs Puzriš-Dagan's 4.3, a ~5x difference
    in effective image-training frequency despite deliberately balanced
    per-class *tablet* counts. Capping it to one real showing per tablet
    per epoch removes that skew entirely without touching line-level MLM
    (Aeneas's own province-only image head, and the earlier
    train_mbert_vision.py pilot, don't have this problem because they
    don't operate at line granularity in the first place)."""
    # Plain sequential pass (not .map(num_proc>1)) -- "first encountered"
    # must follow actual row order, which parallel/batched execution
    # wouldn't guarantee.
    seen = set()
    marked = []
    for tid in dataset["tablet_id"]:
        if not tid or tid in seen:
            marked.append("")
        else:
            seen.add(tid)
            marked.append(tid)
    return dataset.add_column("image_tablet_id", marked)

class MBertCollator:
    """Standard 15% MLM masking (HF's own collator) plus the 4 metadata labels
    carried through -- unlike AkkadianModel's physical-damage collator, mBERT
    isn't being taught the [#]-gap-expansion task, only domain-adapted MLM.
    HF's collator already excludes anything in tokenizer.additional_special_tokens
    from masking targets via get_special_tokens_mask(), so registering
    UNCLEAR_SIGN_TOKEN/UNKNOWN_GAP_TOKEN as special tokens (see train(),
    mark_damage_signals()) is enough to keep them out of the mask targets
    here too -- no extra exclusion logic needed in this collator.

    image_index (tablet_id -> PIL.Image) is None when --use_image is off,
    in which case no pixel_values key is produced at all -- MBertMultiTask
    with use_image=False never looks for one. img_transform selects
    train (augmented) vs eval (deterministic) processing -- see
    IMG_TRANSFORM_TRAIN/IMG_TRANSFORM_EVAL and TiedWeightSafeTrainer's
    eval_data_collator swap.

    context_char_max (set only for the document-granularity dataset, where
    ~5-8% of documents exceed mBERT's hard 512-token position-embedding
    ceiling): instead of always keeping a long document's first max_length
    tokens, follow Aeneas's own approach (predictingthepast/train/
    dataloader.py -- context_char_min=25, context_char_max=768,
    context_char_random=True for their Latin/character-level setup) --
    sample a random character window per example, so the model sees every
    *part* of a long document across training rather than only its
    opening. Requires "text" to still be a raw (untokenized) column on the
    dataset -- tokenization happens here, per batch, not once in train()'s
    .map() step. training=True gives a random start position AND random
    window length each call (a fresh crop every epoch, unlike a one-time
    truncation); training=False (eval) takes a fixed window from the start
    so repeated evaluate() calls stay reproducible -- a deliberate
    departure from Aeneas's own eval-time random start, for the same
    determinism reason IMG_TRANSFORM_EVAL skips augmentation."""
    def __init__(self, tokenizer, mlm_probability=0.15, image_index=None, img_transform=IMG_TRANSFORM_EVAL,
                 context_char_min=None, context_char_max=None, max_length=96, training=False):
        self.tokenizer = tokenizer
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.image_index = image_index
        self.img_transform = img_transform
        self.context_char_min = context_char_min
        self.context_char_max = context_char_max
        self.max_length = max_length
        self.training = training
        self._zero_image = torch.zeros(3, IMG_SIZE, IMG_SIZE)

    def _load_image(self, tablet_id):
        img = self.image_index.get(tablet_id) if tablet_id else None
        if img is None:
            return self._zero_image
        try:
            return self.img_transform(img)
        except Exception:
            return self._zero_image

    def _window(self, text):
        if not self.context_char_max or len(text) <= self.context_char_max:
            return text
        if self.training:
            length = random.randint(min(self.context_char_min, len(text)), self.context_char_max)
            start = random.randint(0, len(text) - length)
            return text[start:start + length]
        return text[:self.context_char_max]

    def _tokenize(self, ex):
        text = mark_damage_signals(self._window(ex["text"]))
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        # MBertMultiTask.forward() only accepts input_ids/attention_mask --
        # drop token_type_ids (BertTokenizer returns it by default), same as
        # the pre-tokenized path already implicitly does by only selecting
        # these two keys out of each example.
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def __call__(self, examples):
        if self.context_char_max is not None:
            pre = [self._tokenize(ex) for ex in examples]
        else:
            pre = [{"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]} for ex in examples]
        batch = self.mlm_collator(pre)
        for task in ["period", "genre", "language", "provenience"]:
            batch[f"{task}_labels"] = torch.tensor([ex[f"{task}_labels"] for ex in examples], dtype=torch.long)
        if self.image_index is not None:
            # "image_tablet_id" (see mark_one_line_per_tablet) is blank for
            # every line of a tablet except one, on the TRAIN split only --
            # falls back to plain "tablet_id" if that column wasn't added
            # (eval collator: every line of an image-bearing tablet gets its
            # real photo, since eval isn't fighting a training-time bias).
            batch["pixel_values"] = torch.stack([
                self._load_image(ex["image_tablet_id"] if "image_tablet_id" in ex else ex.get("tablet_id"))
                for ex in examples
            ])
        return batch

def make_preprocess_logits_for_metrics(banned_ids):
    """banned_ids: PAD/UNK/CLS/SEP/MASK plus the two injected damage
    sentinels -- none of these is ever a valid restoration answer, mirroring
    train.non_content_ids() for the sign-level model."""
    banned = torch.tensor(sorted(banned_ids), dtype=torch.long)

    def preprocess_logits_for_metrics(logits, labels):
        # MBertMultiTask has no HF PretrainedConfig, so Trainer's
        # prediction_step can't filter its output dict via
        # model.config.keys_to_ignore_at_inference -- it converts the dict
        # into a plain positional tuple (dropping "loss") before calling
        # this function. Must index by position, matching
        # MBertMultiTask.forward's dict insertion order: logits,
        # period_logits, genre_logits, language_logits, provenience_logits.
        # Same root cause as AkkadianModel's own config-less-model handling
        # in train.py, which is why that file's version of this function
        # already indexes positionally instead of by key.
        mlm_logits = logits[0].clone()
        mlm_logits[..., banned.to(mlm_logits.device)] = float("-inf")
        mlm_top5 = torch.topk(mlm_logits, k=5, dim=-1).indices

        # Full-vocab rank of the true token, computed here (not in
        # compute_metrics) so we never have to hold the full (B, S, V)
        # logits in the accumulated eval predictions -- same convention as
        # evaluate.py's evaluate_top_k: rank = 1 + count of logits that beat
        # the target's own logit. labels[0] is the primary MLM "labels"
        # tensor (label_names[0]); -100 (unmasked) positions get clamped to
        # a dummy valid index and filtered out downstream via the same mask
        # compute_metrics already applies for mlm_acc/top3/top5.
        mlm_labels = labels[0]
        safe_labels = mlm_labels.clamp(min=0)
        target_logits = mlm_logits.gather(-1, safe_labels.unsqueeze(-1))
        rank = (mlm_logits > target_logits).sum(dim=-1) + 1

        meta_preds = [torch.argmax(logits[i], dim=-1) for i in range(1, 5)]
        return (mlm_top5, rank, *meta_preds)

    return preprocess_logits_for_metrics

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    label_ids = eval_pred.label_ids
    metrics = {}

    task_names = ["period", "genre", "language", "provenience"]
    for i, task in enumerate(task_names):
        task_preds = preds[i + 2].reshape(-1)
        task_labels = label_ids[i + 1].reshape(-1)
        mask = task_labels != -100
        if not mask.any():
            metrics[f"{task}_acc"] = 0.0
            metrics[f"{task}_macro_f1"] = 0.0
            continue
        task_preds, task_labels = task_preds[mask], task_labels[mask]
        metrics[f"{task}_acc"] = float((task_preds == task_labels).mean())
        metrics[f"{task}_macro_f1"] = float(f1_score(task_labels, task_preds, average="macro", zero_division=0))

    mlm_preds = preds[0].reshape(-1, 5)
    mlm_rank = preds[1].reshape(-1)
    mlm_labels = label_ids[0].reshape(-1)
    mlm_mask = mlm_labels != -100
    if mlm_mask.any():
        masked_preds = mlm_preds[mlm_mask]
        masked_labels = mlm_labels[mlm_mask]
        metrics["mlm_acc"] = float((masked_preds[:, 0] == masked_labels).mean())
        metrics["mlm_top3_acc"] = float(np.any(masked_preds[:, :3] == masked_labels[:, None], axis=1).mean())
        metrics["mlm_top5_acc"] = float(np.any(masked_preds == masked_labels[:, None], axis=1).mean())
        # Same metric Lazar et al. 2021 report in their Table 2 (MRR + Hit@5)
        # -- lets us cite a directly comparable number instead of only CER.
        metrics["mlm_mrr"] = float((1.0 / mlm_rank[mlm_mask]).mean())
    else:
        metrics["mlm_acc"] = metrics["mlm_top3_acc"] = metrics["mlm_top5_acc"] = metrics["mlm_mrr"] = 0.0

    return metrics

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Programming\akkadian\data\processed\hf_dataset")
    parser.add_argument("--label_config", type=str, default=None, help="Path to label_configs.json (sizes the metadata heads); auto-resolved from --data_dir if omitted")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--save_dir", type=str, default="checkpoints_mbert")
    # 64 is untested on real hardware -- mBERT (~179M params, 12 layers) is
    # much bigger than AkkadianModel (~41M), so this is a starting point for
    # a 16GB Colab GPU, not a measured value like the signs track's batch
    # size. Test and adjust the same way we tuned the signs track's batch/lr.
    parser.add_argument("--batch_size", type=int, default=64, help="Starting point for a 16GB GPU -- untested, adjust based on actual VRAM usage")
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help="Standard BERT finetuning LR, an order of magnitude below the from-scratch run")
    parser.add_argument("--meta_weight", type=float, default=1.0, help="Loss weight for each metadata head (period/genre/language/provenience); MLM restoration is fixed at 3.0")
    parser.add_argument("--epochs", type=int, default=20, help="Lazar et al. 2021 finetune mBERT for 20 epochs on Akkadian; matched here as the closest precedent")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    # Real token-length distribution (measured against combined_unique.jsonl
    # with mBERT's own WordPiece tokenizer): median=18, p99=72, p99.9=120 --
    # 96 covers 99.7% of examples at little more than half the attention
    # FLOPs of 128.
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16", help="Mixed precision mode -- fp16 for T4/Colab, bf16 for Ampere+ (A100/newer)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint, or 'auto' to resume from the latest one in --save_dir")
    parser.add_argument("--use_image", action="store_true", help="Add the vision branch to period/genre/provenience (Aeneas-style concat) -- off by default, identical behavior to before this flag existed")
    parser.add_argument("--vision_init", type=str, choices=["scratch", "pretrained"], default="scratch", help="scratch: random-init ResNet18, fully trainable (matches Aeneas's own from-scratch ResNet-8). pretrained: frozen ImageNet ResNet18, A/B option")
    parser.add_argument("--crops_dir", type=str, default=r"C:\Programming\akkadian\data\vision_dataset_final", help="Dir with <tablet id>.jpg crops + crops_manifest.jsonl (see finalize_vision_crops.py); ignored if --images_from_hf")
    parser.add_argument("--include_unreviewed", action="store_true", help="Also use tablets whose bbox was never manually reviewed (raw CuneiML bbox, ~58%% reliable) -- off by default, and not meaningful with --images_from_hf (the published vision config is reviewed-only already)")
    parser.add_argument("--images_from_hf", action="store_true", help="Load the vision config straight from --data_dir's HF repo instead of a local --crops_dir -- no scp'ing image folders to a training box")
    parser.add_argument("--hf_config", type=str, default="default", help="Which HF dataset config to load when --data_dir is a Hub repo id (e.g. 'documents' for the tablet-granularity dataset)")
    parser.add_argument("--context_char_min", type=int, default=32, help="Aeneas-style random text windowing (predictingthepast/train/dataloader.py): minimum window length in characters. Only used if --context_char_max is set")
    parser.add_argument("--context_char_max", type=int, default=None, help="Enables random-window sampling of 'text' at collate time instead of always keeping the first --max_length tokens -- for the document-granularity dataset, where some documents badly exceed mBERT's 512-token position-embedding ceiling. None (default) = old behavior, pre-tokenize once and truncate from the start")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(args.save_dir, f"training_log_{timestamp}.txt")), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    # use_fast=False: inject_akkadian_tokens() rebuilds a BertTokenizer from a
    # modified vocab dict, which only the plain-Python slow tokenizer class
    # supports as a constructor argument. WordPiece tokenization itself is
    # identical between the two for BERT.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    logger.info(f"Loading datasets from {args.data_dir} (config={args.hf_config})...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir, args.hf_config)
    else:
        hf_ds = load_from_disk(args.data_dir)

    # Lazar et al. 2021's other free-token trick (see learn_akkadian_tokens
    # docstring) -- reserve slots 3-99 (1-2 go to the damage sentinels below)
    # for WordPiece pieces learned from our own Akkadian corpus, so common
    # sign/word sequences aren't needlessly fragmented by mBERT's stock vocab.
    logger.info("Learning Akkadian-specific WordPiece tokens for mBERT's free vocab slots...")
    akkadian_tokens = learn_akkadian_tokens(hf_ds["train"]["text"], set(tokenizer.get_vocab().keys()), n_tokens=97)
    tokenizer, n_injected = inject_akkadian_tokens(tokenizer, akkadian_tokens, first_free_slot=3)
    logger.info(f"Injected {n_injected} Akkadian tokens into mBERT's unused[3..{2 + n_injected}] slots")

    # Reuse 2 of mBERT's existing [unusedN] embedding rows -- add_special_tokens
    # on a token string already in the vocab only registers it as special
    # (so the tokenizer stops splitting it and the masking collator stops
    # masking it), it does not grow the vocab or add a new row.
    tokenizer.add_special_tokens({"additional_special_tokens": [UNCLEAR_SIGN_TOKEN, UNKNOWN_GAP_TOKEN]})

    # Same dataset as train.py -- we tokenize the 'text' (transliteration)
    # column with mBERT's own WordPiece tokenizer; train.py instead tokenizes
    # the sibling 'signs' column with our CharacterTokenizer.
    #
    # --context_char_max skips pre-tokenization here entirely: MBertCollator
    # tokenizes from raw "text" per batch instead, so it can draw a fresh
    # random character window each time (see MBertCollator._window). "signs"
    # is still unused by this script either way.
    if args.context_char_max is not None:
        hf_ds = hf_ds.remove_columns(["signs"])
    else:
        def tokenize_fn(examples):
            marked = [mark_damage_signals(t) for t in examples["text"]]
            return tokenizer(marked, truncation=True, max_length=args.max_length)
        hf_ds = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text", "signs"], num_proc=max(1, os.cpu_count() - 1))
    train_dataset = hf_ds["train"]
    val_dataset = hf_ds["validation"]
    logger.info(f"Loaded {len(train_dataset)} training samples.")

    image_index = None
    if args.use_image:
        if args.images_from_hf:
            image_index = build_tablet_image_index_from_hf(args.data_dir)
            logger.info(f"Vision branch on ({args.vision_init}): {len(image_index)} tablets have a real photo "
                        f"(loaded from {args.data_dir}'s 'vision' config); everything else gets an all-zero placeholder image")
        else:
            image_index = build_tablet_image_index(args.crops_dir, reviewed_only=not args.include_unreviewed)
            logger.info(f"Vision branch on ({args.vision_init}): {len(image_index)} tablets have a real photo "
                        f"({'including' if args.include_unreviewed else 'excluding'} unreviewed bboxes); "
                        f"everything else gets an all-zero placeholder image")
        # TRAIN only: cap each tablet to one real image showing per epoch
        # (see mark_one_line_per_tablet) -- eval keeps every line's real
        # image, since eval isn't fighting a training-time frequency bias.
        # A no-op at document granularity (already one row per tablet).
        train_dataset = mark_one_line_per_tablet(train_dataset)
        n_marked = sum(1 for t in train_dataset["image_tablet_id"] if t)
        logger.info(f"mark_one_line_per_tablet: {n_marked} rows (of {len(train_dataset)}) keep their real "
                    f"image slot for training, one per tablet")
    collator = MBertCollator(tokenizer, image_index=image_index, img_transform=IMG_TRANSFORM_TRAIN,
                              context_char_min=args.context_char_min, context_char_max=args.context_char_max,
                              max_length=args.max_length, training=True)
    eval_collator = MBertCollator(tokenizer, image_index=image_index, img_transform=IMG_TRANSFORM_EVAL,
                                   context_char_min=args.context_char_min, context_char_max=args.context_char_max,
                                   max_length=args.max_length, training=False) \
        if (args.use_image or args.context_char_max is not None) else None

    if args.label_config:
        label_config_path = args.label_config
    elif os.path.exists(args.data_dir):
        label_config_path = r"C:\Programming\akkadian\data\processed\label_configs.json"
    else:
        from huggingface_hub import hf_hub_download
        label_config_path = hf_hub_download(repo_id=args.data_dir, filename="configs/label_configs.json", repo_type="dataset")
    with open(label_config_path, "r", encoding="utf-8") as f:
        label_configs = json.load(f)
    tasks = ["period", "genre", "language", "provenience"]
    num_labels = {task: len(label_configs[task]["labels"]) for task in tasks}
    logger.info(f"Metadata head sizes from {label_config_path}: {num_labels}")

    logger.info(f"Initializing {args.model_name}...")
    model = MBertMultiTask(
        args.model_name, num_period=num_labels["period"], num_genre=num_labels["genre"],
        num_language=num_labels["language"], num_provenience=num_labels["provenience"],
        meta_weight=args.meta_weight, use_image=args.use_image, vision_init=args.vision_init,
    )

    # TrainingArguments(logging_dir=...) is deprecated in favor of this env
    # var (transformers >= 5.x) -- must be set before the TensorBoardCallback
    # reads it in on_train_begin.
    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(args.save_dir, "runs")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=100,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        fp16=(args.precision == "fp16"),
        bf16=(args.precision == "bf16"),
        dataloader_num_workers=args.num_workers,
        report_to=["tensorboard"],
        label_names=["labels", "period_labels", "genre_labels", "language_labels", "provenience_labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Trainer's default (True) strips any dataset column not in
        # MBertMultiTask.forward()'s signature before the collator ever
        # sees a batch -- breaks both --context_char_max (collator
        # tokenizes from "text" itself) and --use_image (collator looks up
        # "tablet_id"/"image_tablet_id"), neither of which is a forward()
        # parameter. Caught this only via a real Trainer run on the AMD
        # box; direct collator() calls in local smoke tests never
        # exercised Trainer's own column-pruning at all.
        remove_unused_columns=False,
    )

    trainer = TiedWeightSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        eval_data_collator=eval_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=make_preprocess_logits_for_metrics(tokenizer.all_special_ids),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience), LogToFileCallback()],
    )

    logger.info("Starting training with Hugging Face Trainer...")
    resume = True if args.resume_from_checkpoint == "auto" else args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)

    logger.info("Training complete. Saving final state and metrics...")
    trainer.save_model(os.path.join(args.save_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.save_dir, "final_model"))

    with open(os.path.join(args.save_dir, f"training_history_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    logger.info("History saved.")

if __name__ == "__main__":
    train()
