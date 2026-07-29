import os
import torch
import json
import argparse
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.model import AkkadianModel
from src.training.tokenizer import CharacterTokenizer

def build_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/akkadian", help="HF Dataset or local path")
    parser.add_argument("--vocab_file", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "training", "vocab.json"))
    parser.add_argument("--output_file", type=str, default="dataset_index.pt", help="Where to save the tensor index")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharacterTokenizer()
    tokenizer.load(args.vocab_file)

    print("Loading model...")
    # Load weights (assuming standard 12-layer config for now, can be adjusted)
    model = AkkadianModel(vocab_size=len(tokenizer.vocab), hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
    state_dict = torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loading datasets from {args.data_dir}...")
    if "/" in args.data_dir and not os.path.exists(args.data_dir):
        hf_ds = load_dataset(args.data_dir)
    else:
        hf_ds = load_from_disk(args.data_dir)
        
    train_dataset = hf_ds["train"]
    
    def collate_fn(batch):
        # We just need the input_ids
        max_len = max(len(ex["input_ids"]) for ex in batch)
        padded_ids = []
        for ex in batch:
            ids = ex["input_ids"]
            if len(ids) < max_len:
                ids = ids + [tokenizer.vocab.get(tokenizer.pad_token, 0)] * (max_len - len(ids))
            padded_ids.append(ids)
        return torch.tensor(padded_ids, dtype=torch.long)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)
    
    all_embeddings = []
    
    print("Building embeddings index...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            # Forward pass
            outputs = model(batch)
            # logits, unk, emb_context, ...
            emb_context = outputs[2] # Shape: (B, S, H)
            
            # Use mean pooling over the sequence for the document embedding
            # More accurately, we can use the first token [CLS]
            cls_embeddings = emb_context[:, 0, :] # Shape: (B, H)
            
            # Normalize to cosine similarity can be computed via dot product
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=-1)
            
            all_embeddings.append(cls_embeddings.cpu())
            
    final_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Index built! Shape: {final_tensor.shape}")
    
    torch.save(final_tensor, args.output_file)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    build_index()
