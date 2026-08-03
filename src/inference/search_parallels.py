import argparse
import torch
import os
import sys
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.model import AkkadianModel
from src.training.tokenizer import CharacterTokenizer

@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Cuneiform query text")
    parser.add_argument("--db", type=str, default="embeddings_db.pt", help="Path to saved embeddings")
    parser.add_argument("--model_dir", type=str, default="checkpoints/final_model")
    parser.add_argument("--k", type=int, default=5, help="Number of parallels to return")
    parser.add_argument("--hidden_size", type=int, default=640, help="Must match the checkpoint's training config")
    parser.add_argument("--num_layers", type=int, default=8, help="Must match the checkpoint's training config")
    parser.add_argument("--num_heads", type=int, default=8, help="Must match the checkpoint's training config")
    args = parser.parse_args()
    
    if not args.query:
        print("Please provide a query: --query '...'")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DB
    print(f"Loading database from {args.db}...")
    db = torch.load(args.db, map_location=device)
    db_embeddings = db["embeddings"].to(device)
    
    # Load Tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.load(os.path.join(args.model_dir, "vocab.json"))
    
    # Load Model
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(args.model_dir, "model.safetensors"))
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
    
    # Embed query
    input_ids = tokenizer.encode(args.query, add_special_tokens=True, max_length=128)
    t_input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    outputs = model(t_input_ids, return_dict=True)
    query_emb = outputs['emb_context']
    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    
    # Compute cosine similarity
    similarities = torch.matmul(db_embeddings, query_emb.T).squeeze(-1)
    
    # Get top K
    topk_sims, topk_indices = torch.topk(similarities, k=args.k)
    
    print("\n" + "="*50)
    print(f"QUERY: {args.query}")
    print("="*50)
    print(f"Top {args.k} Parallels found:\n")
    
    for i, idx in enumerate(topk_indices):
        idx = idx.item()
        sim = topk_sims[i].item()
        print(f"[{i+1}] Similarity: {sim:.4f}")
        print(f"    Raw Latin : {db['raws'][idx]}")
        print(f"    Cuneiform : {db['signs'][idx]}")
        
        meta = db['metadata'][idx]
        print(f"    Metadata  : {meta['period']} | {meta['genre']} | {meta['language']} | {meta['provenience']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
