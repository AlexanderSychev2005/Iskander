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
    parser.add_argument("--vocab_file", type=str, default=r"C:\Programming\akkadian\data\processed\vocab.json")
    parser.add_argument("--output_file", type=str, default="dataset_index.pt", help="Where to save the tensor index")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=640, help="Must match the checkpoint's training config")
    parser.add_argument("--num_layers", type=int, default=8, help="Must match the checkpoint's training config")
    parser.add_argument("--num_heads", type=int, default=8, help="Must match the checkpoint's training config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharacterTokenizer()
    tokenizer.load(args.vocab_file)

    print("Loading model...")
    ckpt_path = os.path.join(args.checkpoint, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(args.checkpoint, "model.safetensors"))
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    # Metadata head sizes are fully determined by the checkpoint itself --
    # inferring them here avoids the head-count/hidden-size drifting out of
    # sync with what the checkpoint was actually trained with.
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
            # emb_context is already pooled to (B, H) -- mean of [CLS] and the sequence mean
            emb_context = outputs["emb_context"]

            # Normalize so cosine similarity can be computed via dot product
            cls_embeddings = torch.nn.functional.normalize(emb_context, p=2, dim=-1)
            
            all_embeddings.append(cls_embeddings.cpu())
            
    final_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Index built! Shape: {final_tensor.shape}")
    
    torch.save(final_tensor, args.output_file)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    build_index()
