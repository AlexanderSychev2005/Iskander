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

class AkkadianPredictor:
    def __init__(self, checkpoint_path, vocab_file=None, index_file=None, hidden_size=640, num_hidden_layers=8, num_attention_heads=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = CharacterTokenizer()
        if vocab_file is None:
            vocab_file = r"C:\Programming\akkadian\data\processed\vocab.json"
        self.tokenizer.load(vocab_file)

        try:
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
        except (ImportError, FileNotFoundError):
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")

        # hidden_size/num_hidden_layers/num_attention_heads must match how the
        # checkpoint was trained (not recoverable from tensor shapes alone);
        # vocab_size and the metadata head sizes are, so infer those.
        self.model = AkkadianModel(
            vocab_size=state_dict['char_embeddings.weight'].shape[0],
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_period=state_dict['period_head.weight'].shape[0],
            num_genre=state_dict['genre_head.weight'].shape[0],
            num_language=state_dict['language_head.weight'].shape[0],
            num_provenience=state_dict['provenience_head.weight'].shape[0],
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.index = None
        if index_file and os.path.exists(index_file):
            self.index = torch.load(index_file).to(self.device)
            print(f"Loaded embedding index with {self.index.shape[0]} documents.")
            
        self.mask_id = self.tokenizer.vocab.get(self.tokenizer.mask_token)
        self.hash_id = self.tokenizer.vocab.get("[#]", -1)

        # Positions the model fills in during decoding must never resolve to
        # a non-content token (PAD/UNK/CLS/SEP/MASK/x/X/[#]) -- there's no
        # such thing as "the restored sign is [#]". Mirrors the same
        # exclusion applied at training time (AkkadianPhysicalCollator) and
        # in eval (train.non_content_ids).
        from src.training.train import non_content_ids
        self.banned_ids = torch.tensor(sorted(non_content_ids(self.tokenizer)), dtype=torch.long)

    def search_parallels(self, text, top_k=5):
        """Finds the most similar historical texts using cosine similarity of the CLS token."""
        if self.index is None:
            print("No index loaded. Use build_index.py first.")
            return []
            
        token_ids = self.tokenizer.encode(text, max_length=128)
        t_input = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(t_input)
            emb = outputs["emb_context"]  # already pooled to (B, H)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            
            # Cosine similarity is dot product when normalized
            similarities = torch.matmul(self.index, emb.T).squeeze(-1)
            
            top_scores, top_indices = torch.topk(similarities, k=top_k)
            
        return top_indices.tolist(), top_scores.tolist()

    def iterative_decode(self, token_ids):
        """
        Iteratively restores missing characters.
        First, it resolves all '[-]' tokens by greedily picking the most confident prediction.
        Second, it expands '[#]' tokens using the unk head.
        """
        current_ids = list(token_ids)
        
        # 1. Expand [#] into [-] tokens until unk head says stop (max 20)
        # We process one [#] at a time from left to right
        idx = 0
        while idx < len(current_ids):
            if current_ids[idx] == self.hash_id:
                # We replace [#] with [-], and append a new [#] next to it, checking if we should continue
                expansion_length = 0
                while expansion_length < 20:
                    current_ids[idx] = self.mask_id
                    current_ids.insert(idx + 1, self.hash_id)
                    
                    t_input = torch.tensor([current_ids[:128]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        outputs = self.model(t_input, return_dict=False)
                        unk_logits = outputs[1][0] # (S, 2)
                        
                    # Did the model predict to stop (0) or continue (1) expanding at the newly inserted [#]?
                    hash_pos = min(idx + 1, 127)
                    continue_prob = torch.softmax(unk_logits[hash_pos], dim=-1)[1].item()
                    
                    if continue_prob < 0.5:
                        # Stop expanding
                        current_ids.pop(idx + 1)
                        break
                        
                    expansion_length += 1
                    idx += 1 # move to the next inserted mask
                    
                if current_ids[idx] == self.hash_id:
                    current_ids.pop(idx) # Clean up if it hit max 20
            idx += 1
            
        # 2. Fill all [-] tokens iteratively (most confident first)
        mask_indices = [i for i, tid in enumerate(current_ids) if tid == self.mask_id]
        
        with torch.no_grad():
            while mask_indices:
                t_input = torch.tensor([current_ids[:128]], dtype=torch.long, device=self.device)
                outputs = self.model(t_input, return_dict=False)
                mlm_logits = outputs[0][0] # (S, V)
                mlm_logits[:, self.banned_ids] = float("-inf")

                best_idx = -1
                best_prob = -1.0
                best_char_id = -1
                
                for m_idx in mask_indices:
                    if m_idx >= 128: continue # truncated
                    probs = torch.softmax(mlm_logits[m_idx], dim=-1)
                    top_prob, top_char = torch.max(probs, dim=-1)
                    
                    if top_prob.item() > best_prob:
                        best_prob = top_prob.item()
                        best_idx = m_idx
                        best_char_id = top_char.item()
                
                if best_idx == -1: 
                    break # All remaining masks are beyond seq_len
                    
                current_ids[best_idx] = best_char_id
                mask_indices.remove(best_idx)
                
        return current_ids

def evaluate_cer(predictor, data_dir, num_samples=1000):
    """Evaluates Character Error Rate (CER) by punching synthetic holes in validation data."""
    if "/" in data_dir and not os.path.exists(data_dir):
        hf_ds = load_dataset(data_dir)
    else:
        hf_ds = load_from_disk(data_dir)
        
    val_dataset = hf_ds["validation"]
    samples = list(val_dataset)
    random.shuffle(samples)
    samples = samples[:num_samples]
    
    total_cer = 0.0
    valid_evals = 0
    
    for ex in tqdm(samples, desc="Evaluating CER"):
        original_ids = ex["input_ids"]
        
        # Only use texts long enough to punch a hole
        if len(original_ids) < 30:
            continue
            
        # Punch a hole of random size between 1 and 20
        hole_size = random.randint(1, 20)
        start_idx = random.randint(1, len(original_ids) - hole_size - 1)
        
        # Save the original substring that we are removing
        original_target = original_ids[start_idx : start_idx + hole_size]
        original_target_str = predictor.tokenizer.decode(original_target)
        
        # Create corrupted input: replace hole with a single [#]
        corrupted_ids = original_ids[:start_idx] + [predictor.hash_id] + original_ids[start_idx + hole_size:]
        
        # Predict
        predicted_ids = predictor.iterative_decode(corrupted_ids)
        
        # Extract the restored segment
        # We know where the hole started. The length might have changed.
        # Everything before start_idx is identical. Everything after the original hole is shifted.
        # To find the restored part, we take predicted_ids from start_idx up to the point where the remaining suffix starts.
        suffix_len = len(original_ids) - (start_idx + hole_size)
        restored_target = predicted_ids[start_idx : max(start_idx, len(predicted_ids) - suffix_len)]
        restored_target_str = predictor.tokenizer.decode(restored_target)
        
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(original_target_str, restored_target_str)
        cer = distance / max(len(original_target_str), 1)
        
        total_cer += cer
        valid_evals += 1
        
    if valid_evals > 0:
        mean_cer = total_cer / valid_evals
        print(f"Mean CER over {valid_evals} samples (hole size 1-20): {mean_cer:.4f}")
    else:
        print("Not enough samples evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (e.g. checkpoints/checkpoint-2000)")
    parser.add_argument("--vocab_file", type=str, default=None, help="Defaults to vocab.json (signs); pass vocab_translit.json for a --field text checkpoint")
    parser.add_argument("--index_file", type=str, default=None, help="Path to dataset_index.pt")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/Iskander-Dataset")
    parser.add_argument("--eval_cer", action="store_true", help="Run CER evaluation on validation set")
    parser.add_argument("--text", type=str, default=None, help="Text to restore")
    parser.add_argument("--hidden_size", type=int, default=640, help="Must match the checkpoint's training config")
    parser.add_argument("--num_layers", type=int, default=8, help="Must match the checkpoint's training config")
    parser.add_argument("--num_heads", type=int, default=8, help="Must match the checkpoint's training config")
    args = parser.parse_args()

    predictor = AkkadianPredictor(
        args.checkpoint, vocab_file=args.vocab_file, index_file=args.index_file,
        hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
    )
    
    if args.eval_cer:
        evaluate_cer(predictor, args.data_dir)
        
    if args.text:
        token_ids = predictor.tokenizer.encode(args.text, add_special_tokens=True)
        restored_ids = predictor.iterative_decode(token_ids)
        restored_text = predictor.tokenizer.decode(restored_ids)
        print(f"Input:    {args.text}")
        print(f"Restored: {restored_text}")
        
        if args.index_file:
            print("\nFinding parallels...")
            indices, scores = predictor.search_parallels(args.text)
            for idx, score in zip(indices, scores):
                print(f"Doc {idx} - Score: {score:.4f}")
