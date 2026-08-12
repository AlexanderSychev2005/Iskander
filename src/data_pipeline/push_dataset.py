import argparse
import os
from huggingface_hub import HfApi
from datasets import load_from_disk
from pathlib import Path

# Local dir name for each pushed config -- documents/vision are the two
# configs the current pipeline actually trains on (session 2026-08-12);
# 'default' (line-level) stays as split-assignment infrastructure for
# prepare_document_dataset.py/build_vision_hf_dataset.py but isn't itself a
# training target anymore, so it's included here only if explicitly asked for.
CONFIG_DIRS = {
    "documents": "hf_dataset_documents_with_cdli_bulk",
    "vision": "hf_dataset_vision",
    "default": "hf_dataset",
}


def main():
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "data" / "processed"

    parser = argparse.ArgumentParser(description="Push an Iskander dataset config to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="Iskander-Dataset", help="Repository name (will be prefixed with your username)")
    parser.add_argument("--config_name", type=str, default="documents", choices=list(CONFIG_DIRS))
    args = parser.parse_args()

    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.repo_id}"

    print(f"--- Pushing '{args.config_name}' config to {repo_id} ---")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    ds = load_from_disk(str(base_dir / CONFIG_DIRS[args.config_name]))
    ds.push_to_hub(repo_id, config_name=args.config_name)
    print(f"'{args.config_name}' config uploaded successfully!")

    label_path = base_dir / "label_configs.json"
    if label_path.exists():
        api.upload_file(
            path_or_fileobj=str(label_path),
            path_in_repo="configs/label_configs.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload label configs",
        )
        print("label_configs.json uploaded.")

    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
