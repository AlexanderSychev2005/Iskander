import os
import sys
import json
import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk, load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.train_mbert import (
    MBertMultiTask, MBertCollator, mark_damage_signals,
    make_preprocess_logits_for_metrics, compute_metrics,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Dir with model.safetensors + tokenizer files (e.g. final_model)")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/Iskander-Dataset")
    parser.add_argument("--label_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--output_file", type=str, default="evaluation_report_mbert.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)

    print(f"Loading dataset from {args.data_dir}...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir)
    else:
        hf_ds = load_from_disk(args.data_dir)

    def tokenize_fn(examples):
        marked = [mark_damage_signals(t) for t in examples["text"]]
        return tokenizer(marked, truncation=True, max_length=args.max_length)

    val_dataset = hf_ds["validation"].map(tokenize_fn, batched=True, remove_columns=["text", "signs"])
    print(f"Validation samples: {len(val_dataset)}")

    label_config_path = args.label_config or (
        r"C:\Programming\akkadian\data\processed\label_configs.json" if os.path.exists(args.data_dir)
        else None
    )
    if label_config_path is None:
        from huggingface_hub import hf_hub_download
        label_config_path = hf_hub_download(repo_id=args.data_dir, filename="configs/label_configs.json", repo_type="dataset")
    with open(label_config_path, "r", encoding="utf-8") as f:
        label_configs = json.load(f)
    tasks = ["period", "genre", "language", "provenience"]
    num_labels = {task: len(label_configs[task]["labels"]) for task in tasks}

    print(f"Loading model from {args.checkpoint}...")
    model = MBertMultiTask(
        args.model_name, num_period=num_labels["period"], num_genre=num_labels["genre"],
        num_language=num_labels["language"], num_provenience=num_labels["provenience"],
    )
    state_dict = load_file(os.path.join(args.checkpoint, "model.safetensors"))
    model.load_state_dict(state_dict)

    training_args = TrainingArguments(
        output_dir="/tmp/mbert_eval",
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
        label_names=["labels", "period_labels", "genre_labels", "language_labels", "provenience_labels"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        data_collator=MBertCollator(tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=make_preprocess_logits_for_metrics(tokenizer.all_special_ids),
    )

    print("Running evaluation...")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved report to {args.output_file}")
