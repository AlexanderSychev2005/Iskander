"""
Iskander v2.0 Test Evaluation Script.

Evaluates:
- Classification accuracy and Macro-F1 (Period, Genre, Language, Provenience)
- Masked Language Modeling (MLM) accuracy using training-aligned AkkadianPhysicalCollator.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.model import AkkadianModel
from src.training.tokenizer import CharacterTokenizer
from src.training.train import AkkadianPhysicalCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def label_to_id(label_str, mapping):
    if not label_str or label_str == 'Unknown':
        return -100
    return mapping.get(label_str, -100)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Iskander v2.0")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to final model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data dir")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, default="test_results.json", help="Path to save metrics")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None, help="Limit number of test samples for quick testing")
    parser.add_argument("--hidden_size", type=int, default=640, help="Must match the checkpoint's training config")
    parser.add_argument("--num_layers", type=int, default=8, help="Must match the checkpoint's training config")
    parser.add_argument("--num_heads", type=int, default=8, help="Must match the checkpoint's training config")
    args = parser.parse_args()

    set_seed(42)

    data_dir = Path(args.data_dir)
    test_path = data_dir / "test.jsonl"
    vocab_path = Path(args.data_dir) / "vocab.json"
    if not vocab_path.exists():
        vocab_path = Path(args.checkpoint_dir) / "vocab.json"
    label_path = data_dir / "label_configs.json"

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log.info(f"Using device: {device}")

    # Load tokenizer & configs
    tokenizer = CharacterTokenizer()
    tokenizer.load(str(vocab_path))
    label_configs = load_json(label_path)
    
    # Extract label2id
    period_l2i = label_configs['period']['label2id']
    genre_l2i = label_configs['genre']['label2id']
    lang_l2i = label_configs['language']['label2id']
    prov_l2i = label_configs['provenience']['label2id']

    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(args.checkpoint_dir, "model.safetensors"))

    # vocab_size and the metadata head sizes are fully determined by the
    # checkpoint's own tensor shapes; hidden_size/layers/heads are not
    # (dense weight shapes don't encode head count), so those still need args.
    model = AkkadianModel(
        vocab_size=state_dict['char_embeddings.weight'].shape[0],
        hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
        num_period=state_dict['period_head.weight'].shape[0],
        num_genre=state_dict['genre_head.weight'].shape[0],
        num_language=state_dict['language_head.weight'].shape[0],
        num_provenience=state_dict['provenience_head.weight'].shape[0],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load test data
    log.info(f"Loading test data from {test_path}")
    test_records = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_records.append(json.loads(line))
            
    if args.max_test_samples:
        test_records = test_records[:args.max_test_samples]
        log.info(f"Limited test set to {args.max_test_samples} samples")

    # ==========================================
    # 1. Classification Evaluation
    # ==========================================
    log.info("Starting Classification Evaluation...")
    
    true_labels = {'period': [], 'genre': [], 'language': [], 'provenience': []}
    pred_labels = {'period': [], 'genre': [], 'language': [], 'provenience': []}

    for i in tqdm(range(0, len(test_records), args.batch_size), desc="Classification"):
        batch = test_records[i:i + args.batch_size]
        
        signs_list = [r.get('signs', []) for r in batch]
        valid_idx = [j for j, s in enumerate(signs_list) if len(s) > 0]
        if not valid_idx: continue

        signs_list = [signs_list[j] for j in valid_idx]
        batch = [batch[j] for j in valid_idx]

        input_ids = []
        for signs in signs_list:
            input_ids.append(tokenizer.encode_signs(signs, add_special_tokens=True, max_length=128))
            
        # Pad batch
        pad_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = []
        attention_mask = []
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            
        t_input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        t_attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = model(input_ids=t_input_ids)
            
        p_preds = outputs['period_logits'].argmax(dim=-1).cpu().numpy()
        g_preds = outputs['genre_logits'].argmax(dim=-1).cpu().numpy()
        l_preds = outputs['language_logits'].argmax(dim=-1).cpu().numpy()
        prov_preds = outputs['provenience_logits'].argmax(dim=-1).cpu().numpy()
        
        for j, record in enumerate(batch):
            p_true = label_to_id(record.get('period_mapped'), period_l2i)
            g_true = label_to_id(record.get('genre_mapped'), genre_l2i)
            l_true = label_to_id(record.get('language_mapped'), lang_l2i)
            prov_true = label_to_id(record.get('provenience_mapped'), prov_l2i)
            
            if p_true != -100:
                true_labels['period'].append(p_true)
                pred_labels['period'].append(p_preds[j])
            if g_true != -100:
                true_labels['genre'].append(g_true)
                pred_labels['genre'].append(g_preds[j])
            if l_true != -100:
                true_labels['language'].append(l_true)
                pred_labels['language'].append(l_preds[j])
            if prov_true != -100:
                true_labels['provenience'].append(prov_true)
                pred_labels['provenience'].append(prov_preds[j])

    metrics = {}
    metrics['classification'] = {}
    for task in ['period', 'genre', 'language', 'provenience']:
        if len(true_labels[task]) > 0:
            acc = accuracy_score(true_labels[task], pred_labels[task])
            f1 = f1_score(true_labels[task], pred_labels[task], average='macro')
            metrics['classification'][task] = {'accuracy': round(acc, 4), 'macro_f1': round(f1, 4), 'samples': len(true_labels[task])}
            log.info(f"  {task.capitalize()}: Acc {acc:.4f} | F1 {f1:.4f}")

    # ==========================================
    # 2. Restoration Evaluation (Random Masking)
    # ==========================================
    log.info("Starting Restoration Evaluation (training-aligned Random Masking)...")
    
    collator = AkkadianPhysicalCollator(tokenizer)
    processed_records = []
    for r in test_records:
        input_ids = tokenizer.encode_signs(r.get('signs', []), add_special_tokens=True, max_length=128)
        processed_records.append({
            "input_ids": input_ids,
            "period_labels": label_to_id(r.get('period_mapped'), period_l2i),
            "genre_labels": label_to_id(r.get('genre_mapped'), genre_l2i),
            "language_labels": label_to_id(r.get('language_mapped'), lang_l2i),
            "provenience_labels": label_to_id(r.get('provenience_mapped'), prov_l2i)
        })
        
    all_mlm_preds = []
    all_mlm_labels = []
    
    for i in tqdm(range(0, len(processed_records), args.batch_size), desc="Random MLM Eval"):
        batch_records = processed_records[i : i + args.batch_size]
        batch_collated = collator(batch_records)
        
        t_input_ids = batch_collated["input_ids"].to(device)
        t_labels = batch_collated["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=t_input_ids)
            
        logits = outputs['logits']
        logits_masked = logits.clone()
        # Logit masking of service tokens (indices 0 to 6)
        logits_masked[:, :, :7] = -1e9
        
        mlm_top5 = torch.topk(logits_masked, k=5, dim=-1).indices.cpu().numpy()
        t_labels_np = t_labels.cpu().numpy()
        
        mask = t_labels_np != -100
        if mask.any():
            all_mlm_preds.append(mlm_top5[mask])
            all_mlm_labels.append(t_labels_np[mask])
            
    if all_mlm_preds:
        flat_preds = np.concatenate(all_mlm_preds, axis=0)
        flat_labels = np.concatenate(all_mlm_labels, axis=0)
        
        mlm_top1 = float((flat_preds[:, 0] == flat_labels).mean())
        mlm_top3 = float(np.any(flat_preds[:, :3] == flat_labels[:, None], axis=1).mean())
        mlm_top5 = float(np.any(flat_preds == flat_labels[:, None], axis=1).mean())
        
        metrics['restoration'] = {
            'top1': round(mlm_top1, 4),
            'top3': round(mlm_top3, 4),
            'top5': round(mlm_top5, 4),
            'samples': len(flat_labels)
        }
        
        print("\n" + "="*50)
        print("Restoration MLM Metrics (Matches Train/Val):")
        print("-" * 50)
        print(f"Top-1 Acc : {mlm_top1:.4f}")
        print(f"Top-3 Acc : {mlm_top3:.4f}")
        print(f"Top-5 Acc : {mlm_top5:.4f}")
        print(f"Total masked tokens evaluated: {len(flat_labels)}")
        print("="*50)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
