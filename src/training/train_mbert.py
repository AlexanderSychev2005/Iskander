import os
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
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback,
)
from datasets import load_from_disk, load_dataset
from tokenizers import Tokenizer as RawTokenizer
from tokenizers.models import WordPiece as WordPieceModel
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

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
    def __init__(self, model_name, num_period, num_genre, num_language, num_provenience):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.period_head = nn.Linear(hidden_size, num_period)
        self.genre_head = nn.Linear(hidden_size, num_genre)
        self.language_head = nn.Linear(hidden_size, num_language)
        self.provenience_head = nn.Linear(hidden_size, num_provenience)

    def forward(self, input_ids, attention_mask=None, labels=None,
                period_labels=None, genre_labels=None, language_labels=None, provenience_labels=None):
        bert_out = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = bert_out.last_hidden_state
        mlm_logits = self.backbone.cls(seq)

        cls_embed = seq[:, 0, :]
        period_logits = self.period_head(cls_embed)
        genre_logits = self.genre_head(cls_embed)
        language_logits = self.language_head(cls_embed)
        provenience_logits = self.provenience_head(cls_embed)

        loss = None
        if any(l is not None for l in [labels, period_labels, genre_labels, language_labels, provenience_labels]):
            loss_mlm_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
            loss_meta_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = 0.0

            # Same task weighting as AkkadianModel: MLM=3.0, each metadata head=0.25.
            if labels is not None and (labels != -100).any():
                loss += 3.0 * loss_mlm_fct(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))

            meta_weight = 0.25
            for logits, lbl in [(period_logits, period_labels), (genre_logits, genre_labels),
                                 (language_logits, language_labels), (provenience_logits, provenience_labels)]:
                if lbl is not None and (lbl != -100).any():
                    loss += meta_weight * loss_meta_fct(logits, lbl)

        return {
            "loss": loss,
            "logits": mlm_logits,
            "period_logits": period_logits,
            "genre_logits": genre_logits,
            "language_logits": language_logits,
            "provenience_logits": provenience_logits,
        }

class MBertCollator:
    """Standard 15% MLM masking (HF's own collator) plus the 4 metadata labels
    carried through -- unlike AkkadianModel's physical-damage collator, mBERT
    isn't being taught the [#]-gap-expansion task, only domain-adapted MLM.
    HF's collator already excludes anything in tokenizer.additional_special_tokens
    from masking targets via get_special_tokens_mask(), so registering
    UNCLEAR_SIGN_TOKEN/UNKNOWN_GAP_TOKEN as special tokens (see train(),
    mark_damage_signals()) is enough to keep them out of the mask targets
    here too -- no extra exclusion logic needed in this collator."""
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

    def __call__(self, examples):
        batch = self.mlm_collator([{"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]} for ex in examples])
        for task in ["period", "genre", "language", "provenience"]:
            batch[f"{task}_labels"] = torch.tensor([ex[f"{task}_labels"] for ex in examples], dtype=torch.long)
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

    logger.info(f"Loading datasets from {args.data_dir}...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir)
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
    def tokenize_fn(examples):
        marked = [mark_damage_signals(t) for t in examples["text"]]
        return tokenizer(marked, truncation=True, max_length=args.max_length)

    hf_ds = hf_ds.map(tokenize_fn, batched=True, remove_columns=["text", "signs"], num_proc=max(1, os.cpu_count() - 1))
    train_dataset = hf_ds["train"]
    val_dataset = hf_ds["validation"]
    logger.info(f"Loaded {len(train_dataset)} training samples.")

    collator = MBertCollator(tokenizer)

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
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
