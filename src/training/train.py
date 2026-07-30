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
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.tokenizer import CharacterTokenizer
from src.training.model import AkkadianModel

class AkkadianPhysicalCollator:
    def __init__(self, tokenizer, char_mask_rate_min=0.0, char_mask_rate_max=0.15, span_mask_ratio=0.3, unk_geometric_p=0.25):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
        self.cls_id = tokenizer.vocab.get(tokenizer.cls_token, 3)
        self.sep_id = tokenizer.vocab.get(tokenizer.sep_token, 4)
        self.unk_id = tokenizer.vocab.get(tokenizer.unk_token, 1)
        self.mask_id = tokenizer.vocab.get(tokenizer.mask_token, 2)
        self.hash_id = tokenizer.vocab.get("[#]", -1)
        self.x_id = tokenizer.vocab.get("x", -1)
        
        self.char_mask_rate_min = char_mask_rate_min
        self.char_mask_rate_max = char_mask_rate_max
        self.span_mask_ratio = span_mask_ratio
        self.unk_geometric_p = unk_geometric_p

    def _process_one(self, tokens_list):
        if isinstance(tokens_list, torch.Tensor):
            tokens = tokens_list.tolist()
        else:
            tokens = tokens_list
        
        # 1. Identify valid maskable positions
        special_ids = {self.pad_id, self.cls_id, self.sep_id, self.unk_id, self.x_id}
        valid_indices = [i for i, t in enumerate(tokens) if t not in special_ids]
        
        if not valid_indices:
            return torch.tensor(tokens), torch.tensor([-100]*len(tokens)), torch.tensor([-100]*len(tokens))
            
        # 2. Pick ONE compressed gap [#]
        span_len = int(np.random.geometric(self.unk_geometric_p)) - 1
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

def preprocess_logits_for_metrics(logits, labels):
    # logits: (mlm_logits, unk_logits, emb, period, genre, lang, prov)
    # To save memory, we reduce logits to predictions before accumulating
    
    # 1. MLM: Keep Top-5 indices (B, S, 5)
    mlm_top5 = torch.topk(logits[0], k=5, dim=-1).indices
    
    # 2. UNK: Keep Top-1 (B, S)
    unk_top1 = torch.argmax(logits[1], dim=-1)
    
    # 3. Metadata: Keep Top-1 (B,)
    meta_preds = []
    for i in range(3, 7): # period, genre, language, provenience
        meta_preds.append(torch.argmax(logits[i], dim=-1))
        
    return mlm_top5, unk_top1, *meta_preds

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
    parser.add_argument("--data_dir", type=str, default=r"C:\Programming\akkadian\data\ready_for_training\hf_dataset", help="Path to jsonl datasets or HF Repo ID")
    parser.add_argument("--vocab_file", type=str, default=os.path.join(os.path.dirname(__file__), "vocab.json"), help="Path to vocab.json")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate and save every N steps")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size for the transformer")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    args = parser.parse_args()
    
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
    
    vocab_file = args.vocab_file
    tokenizer = CharacterTokenizer()
    tokenizer.load(vocab_file)
    
    logger.info(f"Loading datasets from {args.data_dir}...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir)
    else:
        hf_ds = load_from_disk(args.data_dir)
        
    train_dataset = hf_ds["train"]
    val_dataset = hf_ds["validation"]
    logger.info(f"Loaded {len(train_dataset)} training samples.")
    
    collator = AkkadianPhysicalCollator(tokenizer)
    
    logger.info("Initializing model...")
    model = AkkadianModel(vocab_size=len(tokenizer.vocab), hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads)
    
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
        bf16=True,
        torch_compile=False, # AMD ROCm Inductor can cause inf gradients, better to disable
        dataloader_num_workers=args.num_workers,
        report_to="none",
        label_names=["labels", "unk_labels", "period_labels", "genre_labels", "language_labels", "provenience_labels"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    logger.info("Starting training with Hugging Face Trainer...")
    trainer.train()
    
    logger.info("Training complete. Saving final state and metrics...")
    trainer.save_model(os.path.join(args.save_dir, "final_model"))
    
    with open(os.path.join(args.save_dir, f"training_history_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    logger.info("History saved.")
    
if __name__ == "__main__":
    train()
