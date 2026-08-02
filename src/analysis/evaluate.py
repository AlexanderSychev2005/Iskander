import os
import torch
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from sklearn.metrics import precision_recall_fscore_support
import Levenshtein

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.model import AkkadianModel
from src.training.tokenizer import CharacterTokenizer
from src.training.train import AkkadianPhysicalCollator, non_content_ids
from src.training.tokenizer import collapse_ellipsis_gaps
from src.analysis.inference import AkkadianPredictor

def evaluate_top_k(predictor, dataset, batch_size=64):
    """Top-1/3/5 accuracy and MRR over every position masked by the same
    realistic multi-gap AkkadianPhysicalCollator used in training -- not one
    isolated character per example. Real tablets rarely have exactly one
    clean gap in an otherwise intact line, and this also matches how Lazar
    et al. 2021 and Eremeev aggregate Hit@k over every masked position in a
    sequence rather than sampling a single position."""
    print("\n--- Evaluating Top-1, Top-3, Top-5 Accuracy and MRR (realistic masking) ---")
    collator = AkkadianPhysicalCollator(predictor.tokenizer)
    # Never let the model "win" by predicting a non-content token (PAD/UNK/
    # CLS/SEP/MASK/x/X/[#]) -- there's no ground truth to check such a
    # prediction against, so it must be structurally excluded from the
    # candidate pool, not just hoped to score low.
    banned = torch.tensor(sorted(non_content_ids(predictor.tokenizer)), dtype=torch.long)

    correct_1, correct_3, correct_5, total = 0, 0, 0, 0
    mrr_sum = 0.0

    for i in tqdm(range(0, len(dataset), batch_size), desc="Top-K Batch Processing"):
        batch_dict = dataset[i:i+batch_size]
        n = len(batch_dict["input_ids"])
        examples = [{k: batch_dict[k][j] for k in batch_dict} for j in range(n)]

        collated = collator(examples)
        input_ids = collated["input_ids"].to(predictor.device)
        labels = collated["labels"]  # (B, S) on CPU, -100 where not masked

        with torch.no_grad():
            outputs = predictor.model(input_ids, return_dict=False)
            mlm_logits = outputs[0].cpu()  # (B, S, V)
            mlm_logits[..., banned] = float("-inf")

        for b_idx, pos in (labels != -100).nonzero(as_tuple=False).tolist():
            target = labels[b_idx, pos].item()
            logits = mlm_logits[b_idx, pos]
            top5 = torch.topk(logits, k=5).indices.tolist()

            if target == top5[0]:
                correct_1 += 1
            if target in top5[:3]:
                correct_3 += 1
            if target in top5:
                correct_5 += 1
            # Full rank of the true token (not just whether it's in the
            # top 5) -- MRR gives partial credit even on a miss.
            rank = int((logits > logits[target]).sum().item()) + 1
            mrr_sum += 1.0 / rank
            total += 1

    acc_1 = correct_1 / max(total, 1)
    acc_3 = correct_3 / max(total, 1)
    acc_5 = correct_5 / max(total, 1)
    mrr = mrr_sum / max(total, 1)

    print(f"Total Masked-Position Tests: {total}")
    print(f"Top-1 Accuracy: {acc_1:.4f} ({acc_1*100:.1f}%)")
    print(f"Top-3 Accuracy: {acc_3:.4f} ({acc_3*100:.1f}%)")
    print(f"Top-5 Accuracy: {acc_5:.4f} ({acc_5*100:.1f}%)")
    print(f"MRR: {mrr:.4f}")

    return {"top1": acc_1, "top3": acc_3, "top5": acc_5, "mrr": mrr, "total": total}

def evaluate_cer_by_length(predictor, dataset, max_length=10, samples_per_length=200):
    """CER for lacunae of each exact length 1..max_length, plus a macro-CER
    averaged uniformly across lengths. Aeneas's own paper uses L=20; we use
    L=10 to match Eremeev's thesis (papers/thesis_text.txt), the more directly
    relevant precedent for this project. Long (naturally rarer) lacunae count
    as much as short ones in the summary number, instead
    of being drowned out by the natural length distribution."""
    print("\n--- Evaluating Character Error Rate (CER) by Lacuna Length ---")
    long_enough = [ex["input_ids"] for ex in dataset if len(ex["input_ids"]) >= max_length + 10]
    random.shuffle(long_enough)

    results = {}
    for hole_size in tqdm(range(1, max_length + 1), desc="CER by exact length"):
        total_cer, count = 0.0, 0
        for original_ids in long_enough:
            if count >= samples_per_length:
                break

            start_idx = random.randint(1, len(original_ids) - hole_size - 1)

            original_target = original_ids[start_idx : start_idx + hole_size]
            original_target_str = predictor.tokenizer.decode(original_target)

            corrupted_ids = original_ids[:start_idx] + [predictor.hash_id] + original_ids[start_idx + hole_size:]
            predicted_ids = predictor.iterative_decode(corrupted_ids)

            suffix_len = len(original_ids) - (start_idx + hole_size)
            restored_target = predicted_ids[start_idx : max(start_idx, len(predicted_ids) - suffix_len)]
            restored_target_str = predictor.tokenizer.decode(restored_target)

            distance = Levenshtein.distance(original_target_str, restored_target_str)
            cer = distance / max(len(original_target_str), 1)

            total_cer += cer
            count += 1

        mean_cer = total_cer / count if count else 0.0
        results[str(hole_size)] = {"cer": mean_cer, "count": count}
        print(f"[{hole_size} chars] Mean CER: {mean_cer:.4f} (over {count} samples)")

    macro_cer = sum(r["cer"] for r in results.values()) / len(results)
    print(f"Macro-CER (uniform across lengths 1-{max_length}): {macro_cer:.4f}")
    results["macro_cer"] = macro_cer
    return results

def evaluate_classification_heads(predictor, dataset, label_configs, batch_size=64):
    """Per-class precision/recall/F1 (+ macro avg) for period/genre/language/
    provenience, following the per-genre breakdown practice from Lazar et al.
    2021 and the macro-F1 reporting from Lendvai et al. 2023 -- aggregate
    accuracy alone hides which specific classes the model is failing on."""
    print("\n--- Evaluating Metadata Classification Heads ---")
    heads = ["period", "genre", "language", "provenience"]
    preds = {h: [] for h in heads}
    golds = {h: [] for h in heads}

    for i in tqdm(range(0, len(dataset), batch_size), desc="Classification Batch Processing"):
        batch = dataset[i:i+batch_size]
        input_ids = batch["input_ids"]
        max_len = max(len(ids) for ids in input_ids)
        pad_id = predictor.tokenizer.vocab.get(predictor.tokenizer.pad_token)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids]

        t_input = torch.tensor(padded, dtype=torch.long, device=predictor.device)
        with torch.no_grad():
            outputs = predictor.model(t_input, return_dict=True)

        for h in heads:
            head_preds = outputs[f"{h}_logits"].argmax(dim=-1).cpu().tolist()
            for pred, gold in zip(head_preds, batch[f"{h}_labels"]):
                if gold != -100:  # -100 = Unknown, excluded from loss and from eval
                    preds[h].append(pred)
                    golds[h].append(gold)

    results = {}
    for h in heads:
        if not golds[h]:
            print(f"[{h}] No labeled examples in this split, skipping.")
            continue
        labels_present = sorted(set(golds[h]) | set(preds[h]))
        id2label = label_configs[h]["id2label"]
        names = [id2label.get(str(i), id2label.get(i, str(i))) for i in labels_present]

        precision, recall, f1, support = precision_recall_fscore_support(
            golds[h], preds[h], labels=labels_present, zero_division=0
        )
        per_class = {
            names[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(labels_present))
        }
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            golds[h], preds[h], labels=labels_present, average="macro", zero_division=0
        )
        results[h] = {"per_class": per_class, "macro_precision": float(macro_p),
                       "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
                       "total": len(golds[h])}

        print(f"\n[{h}] {len(golds[h])} labeled examples, macro-F1: {macro_f1:.4f}")
        for name, m in per_class.items():
            print(f"    {name:20s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--field", type=str, choices=["signs", "text"], default="signs", help="Which track the checkpoint was trained on")
    parser.add_argument("--vocab_file", type=str, default=None, help="Defaults to vocab.json for --field signs, vocab_translit.json for --field text")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/Iskander-Dataset")
    parser.add_argument("--label_config", type=str, default=r"C:\Programming\akkadian\data\processed\label_configs.json")
    parser.add_argument("--cer_max_length", type=int, default=None, help="Defaults to 10 for --field signs, 30 for --field text (~10 signs' worth of characters, so both cover the same physical lacuna-length range)")
    parser.add_argument("--output_file", type=str, default="evaluation_report.json")
    args = parser.parse_args()

    default_vocab = r"C:\Programming\akkadian\data\processed\vocab_translit.json" if args.field == "text" else r"C:\Programming\akkadian\data\processed\vocab.json"
    cer_max_length = args.cer_max_length or (30 if args.field == "text" else 10)

    # We reuse AkkadianPredictor from inference.py for simplicity
    predictor = AkkadianPredictor(args.checkpoint, vocab_file=args.vocab_file or default_vocab)

    print(f"Loading dataset from {args.data_dir}...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir)
    else:
        hf_ds = load_from_disk(args.data_dir)

    # Dataset stores untokenized 'signs'/'text' -- tokenize with the checkpoint's own vocab/track.
    if args.field == "text":
        val_dataset = hf_ds["validation"].map(
            lambda ex: {"input_ids": predictor.tokenizer.encode(collapse_ellipsis_gaps(ex["text"]), add_special_tokens=True, max_length=128)}
        )
    else:
        val_dataset = hf_ds["validation"].map(
            lambda ex: {"input_ids": predictor.tokenizer.encode_signs(ex["signs"], add_special_tokens=True, max_length=128)}
        )
    print(f"Total validation samples: {len(val_dataset)}")

    # 1. Evaluate Top-K
    topk_results = evaluate_top_k(predictor, val_dataset)

    # 2. Evaluate CER
    cer_results = evaluate_cer_by_length(predictor, val_dataset, max_length=cer_max_length)

    # 3. Evaluate classification heads (period/genre/language/provenience)
    with open(args.label_config, "r", encoding="utf-8") as f:
        label_configs = json.load(f)
    classification_results = evaluate_classification_heads(predictor, val_dataset, label_configs)

    # 4. Compile and save report
    report = {
        "dataset_size": len(val_dataset),
        "top_k_accuracy": topk_results,
        "cer_by_lacuna_length": cer_results,
        "classification_heads": classification_results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n✅ Evaluation complete. Report saved to {args.output_file}")
