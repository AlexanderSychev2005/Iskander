import os
import torch
torch.set_float32_matmul_precision('high')
import argparse
import random
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from datasets import load_from_disk, load_dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.tokenizer import CharacterTokenizer, collapse_ellipsis_gaps
from src.training.model import AkkadianModel

class LogToFileCallback(TrainerCallback):
    # report_to="none" leaves Trainer's default PrinterCallback printing
    # step/eval metrics straight to stdout (bypasses the `logging` module),
    # so the FileHandler on the module logger never sees them -- only the
    # explicit logger.info(...) calls elsewhere in this script land in
    # training_log_*.txt. This forwards every on_log payload (loss,
    # eval_loss, eval metrics, lr) into that same file.
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logging.getLogger(__name__).info(f"step {state.global_step}: {logs}")


class AkkadianPhysicalCollator:
    # Calibrated against the actual corpus, the same way Eremeev calibrates
    # the birchbark collator against its real reconstruction density: the
    # fraction of signs that are literally 'x' (genuinely damaged/missing on
    # the tablet) in data/processed/combined_unique.jsonl is 3.2%. The old
    # defaults (p=0.25, max=0.15, no span cap) averaged 32.7% of a line
    # masked (median 21.7%, p90 = 100% -- entire short lines wiped out with
    # zero context left). These new defaults simulate to mean=7.8%,
    # median=3.8%, p90=25%, matching Eremeev's own 8% target rather than our
    # own much lower 3.2% raw rate -- MLM training benefits from somewhat
    # more signal than pure real-damage-rate matching (BERT's own 15% is far
    # above any natural corruption rate too), but 32.7% was still an order of
    # magnitude too aggressive.
    def __init__(self, tokenizer, char_mask_rate_min=0.0, char_mask_rate_max=0.08, span_mask_ratio=0.3,
                 unk_geometric_p=0.5, span_cap_frac=0.3):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
        self.cls_id = tokenizer.vocab.get(tokenizer.cls_token, 3)
        self.sep_id = tokenizer.vocab.get(tokenizer.sep_token, 4)
        self.unk_id = tokenizer.vocab.get(tokenizer.unk_token, 1)
        self.mask_id = tokenizer.vocab.get(tokenizer.mask_token, 2)
        self.hash_id = tokenizer.vocab.get("[#]", -1)
        self.x_id = tokenizer.vocab.get("x", -1)
        # Uppercase 'X' is a second, distinct real-damage marker in the ORACC
        # extraction (a sign present but unidentifiable, vs lowercase 'x' for
        # a sign confirmed missing) -- found by comparing against Lazar et
        # al. 2021's own preprocessing, which excludes both 'x' and '...'
        # from maskable positions. We were only excluding lowercase 'x';
        # uppercase 'X' (120,559 occurrences in ORACC alone) was being
        # treated as an ordinary sign eligible for synthetic masking, asking
        # the model to predict positions nobody -- including the original
        # editors -- could actually read.
        self.X_id = tokenizer.vocab.get("X", -1)

        self.char_mask_rate_min = char_mask_rate_min
        self.char_mask_rate_max = char_mask_rate_max
        self.span_mask_ratio = span_mask_ratio
        self.unk_geometric_p = unk_geometric_p
        self.span_cap_frac = span_cap_frac

    def _process_one(self, tokens_list):
        if isinstance(tokens_list, torch.Tensor):
            tokens = tokens_list.tolist()
        else:
            tokens = tokens_list
        
        # 1. Identify valid maskable positions
        # self.hash_id ("[#]") is included here too: since prepare_oracc.py
        # now emits it for real unknown-length gaps ("..."), it can appear in
        # the raw input_ids before this collator adds any synthetic damage --
        # it must be treated as a fixed, non-maskable landmark exactly like
        # the other real-damage markers, not a normal predictable sign.
        special_ids = {self.pad_id, self.cls_id, self.sep_id, self.unk_id, self.x_id, self.X_id, self.hash_id}
        valid_indices = [i for i, t in enumerate(tokens) if t not in special_ids]
        
        if not valid_indices:
            return torch.tensor(tokens), torch.tensor([-100]*len(tokens)), torch.tensor([-100]*len(tokens))
            
        # 2. Pick ONE compressed gap [#], capped so a single gap can never
        # consume an entire (often short, median 7 signs) line -- the long
        # tail of the geometric draw was previously the main cause of the
        # p90=100%-masked lines.
        span_len = int(np.random.geometric(self.unk_geometric_p)) - 1
        span_len = min(span_len, max(1, int(self.span_cap_frac * len(valid_indices))))
        gap_positions = []
        if span_len > 0 and len(valid_indices) >= span_len:
            for _ in range(20):
                start_idx = random.choice(range(len(valid_indices) - span_len + 1))
                pos_slice = valid_indices[start_idx : start_idx + span_len]
                if pos_slice[-1] - pos_slice[0] == span_len - 1: # contiguous
                    gap_positions = pos_slice
                    break
                    
        gap_set = set(gap_positions)
        gap_is_multi = len(gap_positions) > 1
        
        # 3. Spend remaining budget on chars and spans
        remaining_valid = [i for i in valid_indices if i not in gap_set]
        
        char_budget_idx = []
        span_budget_idx = []
        
        if remaining_valid:
            rate = random.uniform(self.char_mask_rate_min, self.char_mask_rate_max)
            mask_num_total = int(rate * len(remaining_valid))
            mask_num_span = int(mask_num_total * self.span_mask_ratio)
            mask_num_char = mask_num_total - mask_num_span
            
            pool = set(remaining_valid)
            for _ in range(mask_num_span):
                if not pool: break
                start = random.choice(list(pool))
                if start + 1 in pool:
                    span_budget_idx.extend([start, start+1])
                    pool.remove(start)
                    pool.remove(start+1)
            
            mask_num_char += mask_num_span - len(span_budget_idx)
            char_pool = list(pool)
            if mask_num_char > 0 and char_pool:
                char_budget_idx = random.sample(char_pool, min(mask_num_char, len(char_pool)))
                
        single_mask_idx = set(span_budget_idx) | set(char_budget_idx)
        
        # 4. Construct new sequence
        new_tokens, new_labels_res, new_labels_unk = [], [], []
        i = 0
        while i < len(tokens):
            if i in gap_set and i == min(gap_set):
                new_tokens.append(self.hash_id)
                new_labels_res.append(-100)
                new_labels_unk.append(1 if gap_is_multi else 0)
                i = max(gap_set) + 1
                continue
            if i in gap_set:
                i += 1
                continue
                
            if i in single_mask_idx:
                new_tokens.append(self.mask_id)
                new_labels_res.append(tokens[i])
                new_labels_unk.append(-100)
            else:
                new_tokens.append(tokens[i])
                new_labels_res.append(-100)
                new_labels_unk.append(-100)
            i += 1
            
        pad_len = len(tokens) - len(new_tokens)
        if pad_len > 0:
            new_tokens.extend([self.pad_id] * pad_len)
            new_labels_res.extend([-100] * pad_len)
            new_labels_unk.extend([-100] * pad_len)
            
        return torch.tensor(new_tokens), torch.tensor(new_labels_res), torch.tensor(new_labels_unk)

    def __call__(self, examples):
        b_input_ids, b_labels_res, b_labels_unk = [], [], []
        
        for ex in examples:
            tokens, l_res, l_unk = self._process_one(ex["input_ids"])
            b_input_ids.append(tokens)
            b_labels_res.append(l_res)
            b_labels_unk.append(l_unk)
            
        batch = {
            "input_ids": torch.stack(b_input_ids),
            "labels": torch.stack(b_labels_res),
            "unk_labels": torch.stack(b_labels_unk),
            "period_labels": torch.tensor([ex["period_labels"] for ex in examples], dtype=torch.long),
            "genre_labels": torch.tensor([ex["genre_labels"] for ex in examples], dtype=torch.long),
            "language_labels": torch.tensor([ex["language_labels"] for ex in examples], dtype=torch.long),
            "provenience_labels": torch.tensor([ex["provenience_labels"] for ex in examples], dtype=torch.long),
        }
        
        return batch

def non_content_ids(tokenizer):
    """Every vocab id that is never a valid restoration answer: the
    structural special tokens (PAD/UNK/CLS/SEP/MASK) plus the three
    real-damage markers (x/X/[#]). These can appear as *context* the model
    conditions on, but should never be the model's own prediction for a
    masked content position -- there both because "the answer is [PAD]"
    is meaningless, and because scoring them as answers has no ground
    truth to check against (see AkkadianPhysicalCollator's masking
    exclusion for the training-time half of this same principle)."""
    ids = {
        tokenizer.vocab.get(tokenizer.pad_token, -1),
        tokenizer.vocab.get(tokenizer.unk_token, -1),
        tokenizer.vocab.get(tokenizer.cls_token, -1),
        tokenizer.vocab.get(tokenizer.sep_token, -1),
        tokenizer.vocab.get(tokenizer.mask_token, -1),
        tokenizer.vocab.get(tokenizer.hash_token, -1),
        tokenizer.vocab.get(tokenizer.x_token, -1),
        tokenizer.vocab.get(tokenizer.X_token, -1),
    }
    ids.discard(-1)
    return ids

def make_preprocess_logits_for_metrics(banned_ids):
    banned = torch.tensor(sorted(banned_ids), dtype=torch.long)

    def preprocess_logits_for_metrics(logits, labels):
        # logits: (mlm_logits, unk_logits, emb, period, genre, lang, prov)
        # To save memory, we reduce logits to predictions before accumulating

        # 1. MLM: ban non-content tokens as candidate answers, then keep
        # Top-5 indices (B, S, 5)
        mlm_logits = logits[0].clone()
        mlm_logits[..., banned.to(mlm_logits.device)] = float("-inf")
        mlm_top5 = torch.topk(mlm_logits, k=5, dim=-1).indices

        # 2. UNK: Keep Top-1 (B, S)
        unk_top1 = torch.argmax(logits[1], dim=-1)

        # 3. Metadata: Keep Top-1 (B,)
        meta_preds = []
        for i in range(3, 7): # period, genre, language, provenience
            meta_preds.append(torch.argmax(logits[i], dim=-1))

        return mlm_top5, unk_top1, *meta_preds

    return preprocess_logits_for_metrics

def compute_metrics(eval_pred):
    # eval_pred.predictions are now from preprocess_logits_for_metrics
    # (mlm_top5, unk_top1, period_top1, genre_top1, language_top1, provenience_top1)
    # eval_pred.label_ids: (labels, unk_labels, period_labels, genre_labels, language_labels, provenience_labels)
    preds = eval_pred.predictions
    label_ids = eval_pred.label_ids
    
    metrics = {}
    
    # Names for the 4 metadata tasks
    task_names = ["period", "genre", "language", "provenience"]
    
    # Metadata metrics (preds indices 2 to 5, label_ids indices 2 to 5)
    for i in range(2, 6):
        task_preds = preds[i].reshape(-1)
        task_labels = label_ids[i].reshape(-1)
        
        mask = task_labels != -100
        if not mask.any():
            metrics[f"{task_names[i-2]}_acc"] = 0.0
            metrics[f"{task_names[i-2]}_macro_f1"] = 0.0
            continue
            
        task_preds = task_preds[mask]
        task_labels = task_labels[mask]
        
        metrics[f"{task_names[i-2]}_acc"] = float((task_preds == task_labels).mean())
        metrics[f"{task_names[i-2]}_macro_f1"] = float(f1_score(task_labels, task_preds, average="macro", zero_division=0))
        
    # MLM accuracy (Top-1, Top-3, Top-5)
    mlm_preds = preds[0].reshape(-1, 5) # (B*S, 5)
    mlm_labels = label_ids[0].reshape(-1)
    mlm_mask = mlm_labels != -100
    if mlm_mask.any():
        masked_preds = mlm_preds[mlm_mask]
        masked_labels = mlm_labels[mlm_mask]
        
        # Top-1 is just the first element in the top-5
        mlm_acc = float((masked_preds[:, 0] == masked_labels).mean())
        metrics["mlm_acc"] = mlm_acc
        
        metrics["mlm_top3_acc"] = float(np.any(masked_preds[:, :3] == masked_labels[:, None], axis=1).mean())
        metrics["mlm_top5_acc"] = float(np.any(masked_preds == masked_labels[:, None], axis=1).mean())
    else:
        metrics["mlm_acc"] = 0.0
        metrics["mlm_top3_acc"] = 0.0
        metrics["mlm_top5_acc"] = 0.0
        
    # UNK Head Accuracy and F1
    unk_preds = preds[1].reshape(-1)
    unk_labels = label_ids[1].reshape(-1)
    unk_mask = unk_labels != -100
    if unk_mask.any():
        masked_unk_preds = unk_preds[unk_mask]
        masked_unk_labels = unk_labels[unk_mask]
        
        metrics["unk_acc"] = float((masked_unk_preds == masked_unk_labels).mean())
        metrics["unk_macro_f1"] = float(f1_score(masked_unk_labels, masked_unk_preds, average="macro", zero_division=0))
    else:
        metrics["unk_acc"] = 0.0
        metrics["unk_macro_f1"] = 0.0
        
    return metrics

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Programming\akkadian\data\processed\hf_dataset", help="Path to jsonl datasets or HF Repo ID")
    parser.add_argument("--field", type=str, choices=["signs", "text"], default="signs", help="'signs' trains on cuneiform glyphs, 'text' on Latin transliteration -- two disjoint vocabs/tracks")
    parser.add_argument("--vocab_file", type=str, default=None, help="Path to vocab.json; defaults to vocab.json for --field signs, vocab_translit.json for --field text")
    # Real token-length distribution (measured against combined_unique.jsonl):
    # signs median=9, p99=32, p99.9=54 -- 128 was wasting ~90% of every
    # sequence as PAD, which self-attention burns quadratically. text median=26,
    # p99.9=164, so 128 is already a reasonable fit there and is left alone.
    parser.add_argument("--max_length", type=int, default=None, help="Token sequence length (pad/truncate target); defaults to 64 for --field signs (covers p99.9), 128 for --field text")
    parser.add_argument("--label_config", type=str, default=None, help="Path to label_configs.json (sizes the metadata heads); auto-resolved from --data_dir if omitted")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save checkpoints and logs; defaults to 'checkpoints' for --field signs, 'checkpoints_translit' for --field text")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs (early stopping will usually cut this short)")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate and save every N steps")
    parser.add_argument("--early_stopping_patience", type=int, default=6, help="Stop after N evals with no eval_loss improvement")
    # 640/8/8 (~41M params): a deliberate middle step up from the previous
    # 512/6/8 (~20M) run, well short of full BERT-Base 768/12/12 (~87M).
    # Sequences here average ~9-11 signs, so depth buys little over a few
    # layers -- most of the extra capacity should go to hidden_size, which
    # is what actually represents the (now 2129-token) sign vocabulary.
    parser.add_argument("--hidden_size", type=int, default=640, help="Hidden size for the transformer")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    # T4 (Colab's common free/paid GPU, compute capability 7.5) has no bf16
    # tensor cores -- only Ampere+ (8.0+) does. bf16 there either errors or
    # silently falls back to a slow emulated path; fp16 is the correct choice
    # on T4 and runs at full tensor-core speed. Default to fp16 accordingly;
    # override to bf16 on Ampere+/A100 hardware where it's the safer choice
    # (wider dynamic range, no loss-scaling needed).
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16", help="Mixed precision mode -- fp16 for T4/Colab, bf16 for Ampere+ (A100/newer)")
    # Colab sessions disconnect/recycle well before 30 epochs over 570k+
    # examples finishes -- without this, a disconnect loses all progress
    # despite checkpoints being saved every eval_steps.
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint, or 'auto' to resume from the latest one in --save_dir")
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = "checkpoints_translit" if args.field == "text" else "checkpoints"
    if args.max_length is None:
        args.max_length = 128 if args.field == "text" else 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, f"training_log_{timestamp}.txt")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Using device: {device}")
    
    default_vocab_name = "vocab_translit.json" if args.field == "text" else "vocab.json"
    if args.vocab_file:
        vocab_file = args.vocab_file
    elif os.path.exists(args.data_dir):
        # Local pipeline output (prepare_hf_dataset.py's own directory layout).
        vocab_file = os.path.join(r"C:\Programming\akkadian\data\processed", default_vocab_name)
    else:
        # --data_dir is a Hub dataset repo id (e.g. on Colab, where the local
        # C:\Programming\... layout doesn't exist) -- pull the matching vocab
        # from the same repo instead of requiring a separate manual step.
        from huggingface_hub import hf_hub_download
        vocab_file = hf_hub_download(repo_id=args.data_dir, filename=f"tokenizer/{default_vocab_name}", repo_type="dataset")
    tokenizer = CharacterTokenizer()
    tokenizer.load(vocab_file)

    logger.info(f"Loading datasets from {args.data_dir}... (max_length={args.max_length})")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir)
    else:
        hf_ds = load_from_disk(args.data_dir)

    # 'signs' -> our own CharacterTokenizer over cuneiform glyphs. 'text' ->
    # the same tokenizer, but over Latin transliteration characters (a
    # disjoint vocab, vocab_translit.json) -- '...' is collapsed to '[#]'
    # first so a real unknown-length gap gets the same reserved token the
    # collator uses for its own synthetic gaps, matching how prepare_oracc.py
    # already does this for the 'signs' side.
    def tokenize_signs(example):
        return {"input_ids": tokenizer.encode_signs(example["signs"], add_special_tokens=True, max_length=args.max_length)}
    def tokenize_text(example):
        return {"input_ids": tokenizer.encode(collapse_ellipsis_gaps(example["text"]), add_special_tokens=True, max_length=args.max_length)}
    # num_proc parallelizes this one-time preprocessing pass across CPU cores
    # -- this is pure-Python per-example tokenization (not batched), so on a
    # single core it's a real, avoidable chunk of wall-clock time before
    # training even starts (~636k rows).
    hf_ds = hf_ds.map(tokenize_text if args.field == "text" else tokenize_signs, num_proc=max(1, os.cpu_count() - 1))

    train_dataset = hf_ds["train"]
    val_dataset = hf_ds["validation"]
    logger.info(f"Loaded {len(train_dataset)} training samples.")

    # unk_geometric_p was re-simulated (not just linearly rescaled) against
    # real transliteration line lengths to hit the same target mean/median/p90
    # masked-fraction as the signs track's own calibration (see conversation
    # -- a naive char-per-sign multiplier overshot badly because of how the
    # geometric draw interacts with span_cap_frac on longer sequences).
    unk_geometric_p = 0.4 if args.field == "text" else 0.5
    collator = AkkadianPhysicalCollator(tokenizer, unk_geometric_p=unk_geometric_p)
    
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

    # TrainingArguments(logging_dir=...) is deprecated in favor of this env
    # var (transformers >= 5.x) -- must be set before the TensorBoardCallback
    # reads it in on_train_begin.
    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(args.save_dir, "runs")

    # torch_compile only pays off on Ampere+ (tensor-core generation the
    # compiler actually targets); on T4/Turing it just adds ~30s of upfront
    # compile time and prints "Not enough SMs to use max_autotune_gemm mode"
    # for no steady-state speedup.
    use_compile = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    logger.info("Initializing model...")
    model = AkkadianModel(
        vocab_size=len(tokenizer.vocab), hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
        num_period=num_labels["period"], num_genre=num_labels["genre"],
        num_language=num_labels["language"], num_provenience=num_labels["provenience"],
    )
    
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
        fp16=(args.precision == "fp16"),
        bf16=(args.precision == "bf16"),
        torch_compile=use_compile,
        dataloader_num_workers=args.num_workers,
        report_to=["tensorboard"],
        label_names=["labels", "unk_labels", "period_labels", "genre_labels", "language_labels", "provenience_labels"],
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
        preprocess_logits_for_metrics=make_preprocess_logits_for_metrics(non_content_ids(tokenizer)),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience), LogToFileCallback()],
    )
    
    logger.info("Starting training with Hugging Face Trainer...")
    resume = True if args.resume_from_checkpoint == "auto" else args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume)
    
    logger.info("Training complete. Saving final state and metrics...")
    trainer.save_model(os.path.join(args.save_dir, "final_model"))
    
    with open(os.path.join(args.save_dir, f"training_history_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    logger.info("History saved.")
    
if __name__ == "__main__":
    train()
