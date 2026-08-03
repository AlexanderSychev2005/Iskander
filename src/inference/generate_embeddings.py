import os
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.model import AkkadianModel
from src.training.tokenizer import CharacterTokenizer

def load_data(data_path):
    records = []
    path = Path(data_path)
    
    files_to_load = []
    if path.is_file():
        files_to_load.append(path)
    elif path.is_dir():
        for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            p = path / split
            if p.exists():
                files_to_load.append(p)
                
    for f_path in files_to_load:
        print(f"Loading {f_path}...")
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
    return records

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="checkpoints/final_model")
    parser.add_argument("--data_path", type=str, default="data/processed/combined_unique.jsonl")
    parser.add_argument("--output_file", type=str, default="full_embeddings_db.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=640, help="Must match the checkpoint's training config")
    parser.add_argument("--num_layers", type=int, default=8, help="Must match the checkpoint's training config")
    parser.add_argument("--num_heads", type=int, default=8, help="Must match the checkpoint's training config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.load(os.path.join(args.model_dir, "vocab.json"))

    # Load Model
    print(f"Loading model from {args.model_dir}")
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
    
    # Load Data
    records = load_data(args.data_path)
    print(f"Total records to embed: {len(records)}")
    
    all_embeddings = []
    all_raws = []
    all_signs = []
    all_metadata = []
    
    for i in tqdm(range(0, len(records), args.batch_size), desc="Generating Embeddings"):
        batch = records[i:i+args.batch_size]
        
        signs_list = []
        for r in batch:
            signs = r.get('signs', [])
            signs_list.append(signs)

            all_raws.append(r.get('raw', ''))
            all_signs.append("".join(signs) if isinstance(signs, list) else signs)
            all_metadata.append({
                "period": r.get("period_mapped", "Unknown"),
                "genre": r.get("genre_mapped", "Unknown"),
                "language": r.get("language_mapped", "Unknown"),
                "provenience": r.get("provenience_mapped", "Unknown")
            })

        input_ids = []
        for signs in signs_list:
            input_ids.append(tokenizer.encode_signs(signs, add_special_tokens=True, max_length=128))
            
        # Pad batch
        pad_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = []
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            
        t_input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        
        outputs = model(t_input_ids, return_dict=True)
        # Context embeddings from CLS + Sequence Mean
        emb_context = outputs['emb_context']
        
        # Normalize for cosine similarity
        emb_context = torch.nn.functional.normalize(emb_context, p=2, dim=1)
        
        all_embeddings.append(emb_context.cpu())
        
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    db = {
        "embeddings": final_embeddings,
        "raws": all_raws,
        "signs": all_signs,
        "metadata": all_metadata
    }
    
    torch.save(db, args.output_file)
    print(f"Saved {len(final_embeddings)} embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
