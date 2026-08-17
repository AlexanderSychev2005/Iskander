import os
import sys
import json
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk, load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.train_mbert import (
    MBertMultiTask, MBertCollator, mark_damage_signals,
    build_tablet_image_index, build_tablet_image_index_from_hf,
    make_preprocess_logits_for_metrics, compute_metrics,
    IMG_TRANSFORM_EVAL,
)


def per_class_report(preds_by_task, labels_by_task, label_configs):
    """Per-VALUE precision/recall/f1/support for each metadata task --
    compute_metrics (train_mbert.py) only reports the macro average, which
    hides exactly the thing worth knowing when comparing a text-only run
    against a --use_image run: whether a given head's lift (or drop) is
    spread evenly across its classes or concentrated in one or two
    (session discussion, 2026-08-06 -- e.g. the val-set sample-size
    caveats already flagged for Royal Inscriptions/Lexical/Assur)."""
    report = {}
    for task, preds in preds_by_task.items():
        labels = labels_by_task[task]
        mask = labels != -100
        if not mask.any():
            continue
        names = label_configs[task]["labels"]
        present = sorted(set(labels[mask].tolist()) | set(preds[mask].tolist()))
        target_names = [names[i] for i in present]
        report[task] = classification_report(
            labels[mask], preds[mask], labels=present, target_names=target_names,
            output_dict=True, zero_division=0,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Dir with model.safetensors + tokenizer files (e.g. final_model). With --untrained, "
                              "only its tokenizer is used (for the injected Akkadian tokens + damage sentinels, so "
                              "masking-eligibility and vocab fragmentation stay identical to the trained run) -- "
                              "model.safetensors is not loaded, so any checkpoint's tokenizer works")
    parser.add_argument("--untrained", action="store_true",
                         help="Skip loading --checkpoint's weights -- evaluate the backbone at its plain "
                              "AutoModelForMaskedLM.from_pretrained(--model_name) state (no Akkadian finetuning at "
                              "all) with freshly random-initialized metadata heads. The zero-shot/no-finetuning "
                              "baseline Lazar et al. 2021 also report (their Table 2).")
    parser.add_argument("--seed", type=int, default=42, help="Only matters for --untrained (random head init)")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/Iskander-Dataset")
    parser.add_argument("--hf_config", type=str, default="default", help="'default' (line-level) or 'documents' (tablet-level)")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"], help="Use 'validation' while iterating, 'test' only for the final reported number")
    parser.add_argument("--label_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--context_char_min", type=int, default=32)
    parser.add_argument("--context_char_max", type=int, default=None, help="Match whatever the checkpoint was trained with (e.g. 850 for a --hf_config documents run)")
    parser.add_argument("--use_image", action="store_true", help="Must match how the checkpoint was trained")
    parser.add_argument("--vision_init", type=str, choices=["scratch", "pretrained", "finetune"], default="scratch")
    parser.add_argument("--images_from_hf", action="store_true")
    parser.add_argument("--crops_dir", type=str, default=r"C:\Programming\akkadian\data\vision_dataset_final")
    parser.add_argument("--include_unreviewed", action="store_true")
    parser.add_argument("--output_file", type=str, default="evaluation_report_mbert.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)

    print(f"Loading dataset from {args.data_dir} (config={args.hf_config}, split={args.split})...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir, args.hf_config)
    else:
        hf_ds = load_from_disk(args.data_dir)

    # Windowed (document-level) eval: tokenize per-batch in the collator
    # (deterministic, from-the-start window -- see MBertCollator._window),
    # same as train_mbert.py's eval_collator. Otherwise, pre-tokenize once
    # as before.
    if args.context_char_max is not None:
        eval_dataset = hf_ds[args.split].remove_columns(["signs"])
    else:
        def tokenize_fn(examples):
            marked = [mark_damage_signals(t) for t in examples["text"]]
            return tokenizer(marked, truncation=True, max_length=args.max_length)
        eval_dataset = hf_ds[args.split].map(tokenize_fn, batched=True, remove_columns=["text", "signs"])
    print(f"{args.split} samples: {len(eval_dataset)}")

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

    image_index = None
    if args.use_image:
        if args.images_from_hf:
            image_index = build_tablet_image_index_from_hf(args.data_dir)
        else:
            image_index = build_tablet_image_index(args.crops_dir, reviewed_only=not args.include_unreviewed)
        print(f"Vision branch on: {len(image_index)} tablets have a real photo")

    torch.manual_seed(args.seed)
    if args.untrained:
        print(f"Building UNTRAINED model ({args.model_name}'s own pretrained backbone, random-init heads, "
              f"--checkpoint used only for its tokenizer)...")
    else:
        print(f"Loading model from {args.checkpoint}...")
    model = MBertMultiTask(
        args.model_name, num_period=num_labels["period"], num_genre=num_labels["genre"],
        num_language=num_labels["language"], num_provenience=num_labels["provenience"],
        use_image=args.use_image, vision_init=args.vision_init,
    )
    if not args.untrained:
        state_dict = load_file(os.path.join(args.checkpoint, "model.safetensors"))
        model.load_state_dict(state_dict)

    collator = MBertCollator(
        tokenizer, image_index=image_index, img_transform=IMG_TRANSFORM_EVAL,
        context_char_min=args.context_char_min, context_char_max=args.context_char_max,
        max_length=args.max_length, training=False,
    )

    training_args = TrainingArguments(
        output_dir="/tmp/mbert_eval",
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
        label_names=["labels", "period_labels", "genre_labels", "language_labels", "provenience_labels"],
        # See train_mbert.py's train() -- Trainer's default column-pruning
        # strips "text"/"tablet_id" before the collator can use them for
        # on-the-fly windowing/image lookup.
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=make_preprocess_logits_for_metrics(tokenizer.all_special_ids),
    )

    print("Running evaluation...")
    pred_output = trainer.predict(eval_dataset)
    metrics = {k.replace("test_", ""): v for k, v in pred_output.metrics.items()}
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    preds = pred_output.predictions
    label_ids = pred_output.label_ids
    preds_by_task = {task: np.asarray(preds[i + 2]).reshape(-1) for i, task in enumerate(tasks)}
    labels_by_task = {task: np.asarray(label_ids[i + 1]).reshape(-1) for i, task in enumerate(tasks)}
    per_class = per_class_report(preds_by_task, labels_by_task, label_configs)

    print("\nPer-class breakdown:")
    for task, rep in per_class.items():
        print(f"  --- {task} ---")
        for cls, stats in rep.items():
            if cls in ("accuracy", "macro avg", "weighted avg"):
                continue
            print(f"    {cls}: f1={stats['f1-score']:.3f} precision={stats['precision']:.3f} "
                  f"recall={stats['recall']:.3f} support={int(stats['support'])}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_class": per_class}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report to {args.output_file}")
