import argparse
import os
from huggingface_hub import HfApi
from datasets import load_from_disk
from pathlib import Path

def main():
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "data" / "processed"

    parser = argparse.ArgumentParser(description="Push Iskander dataset and tokenizer to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="Iskander-Dataset", help="Repository name (will be prefixed with your username)")
    args = parser.parse_args()

    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.repo_id}"

    print(f"--- Pushing Dataset to {repo_id} ---")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # 1. Push HF Dataset (Train & Validation)
    print("Loading dataset...")
    ds = load_from_disk(str(base_dir / "hf_dataset"))
    print(f"Pushing dataset splits to {repo_id}...")
    ds.push_to_hub(repo_id)
    print("Dataset splits uploaded successfully!")

    # 2. Upload vocab.json (signs) and vocab_translit.json (transliteration)
    # -- two disjoint vocabularies for the two training tracks.
    for fname in ["vocab.json", "vocab_translit.json"]:
        vocab_path = base_dir / fname
        if vocab_path.exists():
            print(f"Uploading {vocab_path.name}...")
            api.upload_file(
                path_or_fileobj=str(vocab_path),
                path_in_repo=f"tokenizer/{fname}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {fname}"
            )

    # 3. Upload label_configs.json
    label_path = base_dir / "label_configs.json"
    if label_path.exists():
        print(f"Uploading {label_path.name}...")
        api.upload_file(
            path_or_fileobj=str(label_path),
            path_in_repo="configs/label_configs.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload label configs"
        )
        
    print("\nSuccessfully pushed dataset, vocabulary, and configs to Hugging Face Hub!")
    print(f"You can view your dataset here: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
