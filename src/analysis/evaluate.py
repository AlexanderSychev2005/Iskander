import os
import torch
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
import Levenshtein

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.model import AkkadianModel
from src.training.tokenizer import CharacterTokenizer
from src.analysis.inference import AkkadianPredictor

def evaluate_top_k(predictor, dataset, batch_size=64):
    """Evaluates Top-1, Top-3, and Top-5 accuracy for single missing characters."""
    print("\n--- Evaluating Top-1, Top-3, Top-5 Accuracy ---")
    correct_1, correct_3, correct_5, total = 0, 0, 0, 0
    
    # We will process in batches to be fast
    for i in tqdm(range(0, len(dataset), batch_size), desc="Top-K Batch Processing"):
        batch = dataset[i:i+batch_size]["input_ids"]
        
        corrupted_batch = []
        target_ids = []
        target_positions = []
        
        for ids in batch:
            if len(ids) < 10:
                continue
            # Pick a random character to mask
            mask_idx = random.randint(1, len(ids) - 2) # avoid CLS/SEP
            
            target_ids.append(ids[mask_idx])
            target_positions.append(mask_idx)
            
            corrupted = ids.copy()
            corrupted[mask_idx] = predictor.mask_id
            
            # pad to 128
            if len(corrupted) < 128:
                corrupted.extend([predictor.tokenizer.vocab.get(predictor.tokenizer.pad_token)] * (128 - len(corrupted)))
            corrupted_batch.append(corrupted[:128])
            
        if not corrupted_batch:
            continue
            
        t_input = torch.tensor(corrupted_batch, dtype=torch.long, device=predictor.device)
        with torch.no_grad():
            outputs = predictor.model(t_input, return_dict=False)
            mlm_logits = outputs[0] # (B, S, V)
            
            for b_idx in range(len(corrupted_batch)):
                pos = target_positions[b_idx]
                target = target_ids[b_idx]
                if pos >= 128: continue
                
                logits = mlm_logits[b_idx, pos]
                top5 = torch.topk(logits, k=5).indices.tolist()
                
                if target == top5[0]:
                    correct_1 += 1
                if target in top5[:3]:
                    correct_3 += 1
                if target in top5:
                    correct_5 += 1
                total += 1
                
    acc_1 = correct_1 / max(total, 1)
    acc_3 = correct_3 / max(total, 1)
    acc_5 = correct_5 / max(total, 1)
    
    print(f"Total Single-Char Tests: {total}")
    print(f"Top-1 Accuracy: {acc_1:.4f} ({acc_1*100:.1f}%)")
    print(f"Top-3 Accuracy: {acc_3:.4f} ({acc_3*100:.1f}%)")
    print(f"Top-5 Accuracy: {acc_5:.4f} ({acc_5*100:.1f}%)")
    
    return {"top1": acc_1, "top3": acc_3, "top5": acc_5, "total": total}

def evaluate_cer_by_length(predictor, dataset):
    """Evaluates CER for lacunae of varying lengths."""
    print("\n--- Evaluating Character Error Rate (CER) by Lacuna Length ---")
    
    # We define 4 buckets
    buckets = {
        "1-5": {"range": (1, 5), "total_cer": 0.0, "count": 0},
        "6-10": {"range": (6, 10), "total_cer": 0.0, "count": 0},
        "11-15": {"range": (11, 15), "total_cer": 0.0, "count": 0},
        "16-20": {"range": (16, 20), "total_cer": 0.0, "count": 0},
    }
    
    for ex in tqdm(dataset, desc="CER Sequential Evaluation"):
        original_ids = ex["input_ids"]
        if len(original_ids) < 30:
            continue
            
        # Randomly choose a bucket for this example
        bucket_name = random.choice(list(buckets.keys()))
        min_l, max_l = buckets[bucket_name]["range"]
        
        hole_size = random.randint(min_l, max_l)
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
        
        buckets[bucket_name]["total_cer"] += cer
        buckets[bucket_name]["count"] += 1
        
    results = {}
    for name, data in buckets.items():
        if data["count"] > 0:
            mean_cer = data["total_cer"] / data["count"]
            results[name] = {"cer": mean_cer, "count": data["count"]}
            print(f"[{name} chars] Mean CER: {mean_cer:.4f} (over {data['count']} samples)")
        else:
            results[name] = {"cer": 0.0, "count": 0}
            print(f"[{name} chars] No samples evaluated.")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/akkadian")
    parser.add_argument("--output_file", type=str, default="evaluation_report.json")
    args = parser.parse_args()
    
    # We reuse AkkadianPredictor from inference.py for simplicity
    predictor = AkkadianPredictor(args.checkpoint)
    
    print(f"Loading dataset from {args.data_dir}...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir)
    else:
        hf_ds = load_from_disk(args.data_dir)
        
    val_dataset = hf_ds["validation"]
    print(f"Total validation samples: {len(val_dataset)}")
    
    # 1. Evaluate Top-K
    topk_results = evaluate_top_k(predictor, val_dataset)
    
    # 2. Evaluate CER
    cer_results = evaluate_cer_by_length(predictor, val_dataset)
    
    # 3. Compile and save report
    report = {
        "dataset_size": len(val_dataset),
        "top_k_accuracy": topk_results,
        "cer_by_lacuna_length": cer_results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n✅ Evaluation complete. Report saved to {args.output_file}")
