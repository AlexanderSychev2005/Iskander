"""Qualitative side-by-side demo: for a handful of real test-split tablets,
show the masked input and what each final checkpoint predicts, text-only vs
vision (provenience) model on the identical input -- complements
evaluate_mbert.py's aggregate/per-class numbers with concrete examples for
the diploma writeup.

Masking here always shows [MASK] at every chosen position (not the real
80/10/10 BERT recipe DataCollatorForLanguageModeling uses during actual
training/eval) -- clearer to read, and the reported metrics still come from
evaluate_mbert.py's real collator, not from this file. Both models see the
exact same masked positions (one shared RNG draw per example) so restoration
quality is comparable position-by-position, not just in aggregate.

The image only ever reaches provenience_head (see MBertMultiTask.forward --
mlm_logits is computed straight from BERT's own hidden states, before any
image concatenation), so the two models' MLM restoration differs only
because they're separately trained weights, not because of the image
directly. The place the image can actually change an answer is the
provenience row in each example's metadata table.
"""
import argparse
import os
import random
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.training.train_mbert import (
    MBertMultiTask, mark_damage_signals, build_tablet_image_index_from_hf, IMG_TRANSFORM_EVAL,
)

TASKS = ["period", "genre", "language", "provenience"]


def load_model(checkpoint, model_name, num_labels, use_image, vision_init):
    model = MBertMultiTask(model_name, num_period=num_labels["period"], num_genre=num_labels["genre"],
                            num_language=num_labels["language"], num_provenience=num_labels["provenience"],
                            use_image=use_image, vision_init=vision_init)
    state_dict = load_file(os.path.join(checkpoint, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mask_positions(input_ids, banned_ids, mlm_probability, rng):
    eligible = [i for i, t in enumerate(input_ids) if t not in banned_ids]
    n_mask = max(1, round(len(eligible) * mlm_probability))
    return sorted(rng.sample(eligible, min(n_mask, len(eligible))))


def topk_at(logits, position, banned, tokenizer, k=3):
    row = logits[0, position].clone()
    row[list(banned)] = float("-inf")
    top = torch.topk(row, k=k).indices.tolist()
    return [tokenizer.convert_ids_to_tokens([t])[0] for t in top]


def format_metadata_table(label_configs, truth, text_pred, vision_pred):
    lines = ["| head | ground truth | text-only prediction | vision prediction |", "|---|---|---|---|"]
    for task in TASKS:
        names = label_configs[task]["labels"]
        t_idx, t_conf = text_pred[task]
        v_idx, v_conf = vision_pred[task]
        truth_name = names[truth[task]] if truth[task] != -100 and truth[task] < len(names) else "(no label)"
        t_name = f"{names[t_idx]} ({t_conf:.2f})"
        v_name = f"{names[v_idx]} ({v_conf:.2f})"
        marker = " **<- differs**" if t_idx != v_idx else ""
        lines.append(f"| {task} | {truth_name} | {t_name} | {v_name}{marker} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_checkpoint", type=str, default=r"C:\Programming\akkadian\checkpoints_final_text\final_model")
    parser.add_argument("--vision_checkpoint", type=str, default=r"C:\Programming\akkadian\checkpoints_final_vision\final_model")
    parser.add_argument("--data_dir", type=str, default="AlexSychovUN/Iskander-Dataset")
    parser.add_argument("--hf_config", type=str, default="documents")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--label_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--n_examples", type=int, default=20)
    parser.add_argument("--context_char_max", type=int, default=850)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tablet_ids", type=str, default=None,
                         help="Comma-separated tablet_id list to use instead of random sampling (order preserved). "
                              "Overrides --n_examples.")
    parser.add_argument("--output_file", type=str, default="predictions_demo.md")
    parser.add_argument("--embed_images", action="store_true",
                         help="Save each example's real photo (if any) next to output_file and embed it in the "
                              "markdown, instead of only noting has-photo True/False")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Loading tokenizer + label config...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_checkpoint, use_fast=False)
    banned_ids = set(tokenizer.all_special_ids)

    import json
    label_config_path = args.label_config
    if label_config_path is None:
        from huggingface_hub import hf_hub_download
        label_config_path = hf_hub_download(repo_id=args.data_dir, filename="configs/label_configs.json", repo_type="dataset")
    with open(label_config_path, encoding="utf-8") as f:
        label_configs = json.load(f)
    num_labels = {task: len(label_configs[task]["labels"]) for task in TASKS}

    print(f"Loading {args.split} split ({args.hf_config})...")
    ds = load_dataset(args.data_dir, args.hf_config)[args.split]
    print(f"  {len(ds)} rows")

    print("Loading tablet image index (for the vision model)...")
    image_index = build_tablet_image_index_from_hf(args.data_dir)
    zero_image = torch.zeros(3, 224, 224)

    print("Loading text-only model...")
    text_model = load_model(args.text_checkpoint, args.model_name, num_labels, use_image=False, vision_init="scratch")
    print("Loading vision model...")
    vision_model = load_model(args.vision_checkpoint, args.model_name, num_labels, use_image=True, vision_init="finetune")

    if args.tablet_ids:
        wanted = [t.strip() for t in args.tablet_ids.split(",") if t.strip()]
        id_to_idx = {ds[i]["tablet_id"]: i for i in range(len(ds))}
        indices = []
        for tid in wanted:
            if tid not in id_to_idx:
                print(f"  WARNING: {tid} not found in {args.split} split, skipping")
                continue
            indices.append(id_to_idx[tid])
        print(f"Selected {len(indices)}/{len(wanted)} requested tablets")
    else:
        # Prefer examples that actually have a real photo, so the vision model's
        # provenience row isn't just running on the same all-zero placeholder as
        # text-only every time -- otherwise most of the demo would show no
        # possible difference by construction.
        has_photo = [i for i in range(len(ds)) if ds[i]["tablet_id"] in image_index]
        no_photo = [i for i in range(len(ds)) if ds[i]["tablet_id"] not in image_index]
        rng.shuffle(has_photo)
        rng.shuffle(no_photo)
        n_photo = min(len(has_photo), max(1, args.n_examples * 2 // 3))
        indices = has_photo[:n_photo] + no_photo[:args.n_examples - n_photo]
        rng.shuffle(indices)
        print(f"Selected {len(indices)} examples ({n_photo} with a real photo, {len(indices) - n_photo} without)")

    selection_note = (f"{len(indices)} hand-picked tablet(s) (`--tablet_ids`)" if args.tablet_ids
                       else f"{len(indices)} random test-split tablets, seed={args.seed}")
    out = []
    out.append("# Prediction demo: text-only vs vision (provenience) model\n")
    out.append(f"{selection_note}. Both models see the exact same "
               f"masked positions per example (`[MASK]` shown at every chosen position, {args.mlm_probability:.0%} "
               "of eligible tokens) -- differences in restoration come only from the two models' separately "
               "trained weights, not from the image itself (the image only reaches `provenience_head`, see module "
               "docstring). The metadata table's `provenience` row is where the image can actually change an answer.\n")

    for n, idx in enumerate(indices, 1):
        row = ds[idx]
        tablet_id = row["tablet_id"]
        text = row["text"][:args.context_char_max]
        marked = mark_damage_signals(text)
        enc = tokenizer(marked, truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"]

        positions = mask_positions(input_ids, banned_ids, args.mlm_probability, rng)
        masked_ids = list(input_ids)
        true_tokens = [input_ids[p] for p in positions]
        for p in positions:
            masked_ids[p] = tokenizer.mask_token_id

        input_tensor = torch.tensor([masked_ids])
        attn = torch.tensor([[1] * len(masked_ids)])
        img = image_index.get(tablet_id)
        pixel_values = IMG_TRANSFORM_EVAL(img).unsqueeze(0) if img is not None else zero_image.unsqueeze(0)

        with torch.no_grad():
            text_out = text_model(input_ids=input_tensor, attention_mask=attn)
            vision_out = vision_model(input_ids=input_tensor, attention_mask=attn, pixel_values=pixel_values)

        truth = {task: row[f"{task}_labels"] for task in TASKS}
        text_pred = {}
        vision_pred = {}
        for task in TASKS:
            for name, out_dict, pred_dict in [("text", text_out, text_pred), ("vision", vision_out, vision_pred)]:
                probs = torch.softmax(out_dict[f"{task}_logits"][0], dim=-1)
                conf, cls = probs.max(dim=-1)
                pred_dict[task] = (cls.item(), conf.item())

        masked_display = tokenizer.decode(masked_ids[1:-1], skip_special_tokens=False)
        original_display = tokenizer.decode(input_ids[1:-1], skip_special_tokens=False)

        out.append(f"## Example {n} — `{tablet_id}` (has photo: {img is not None})\n")
        if img is not None and args.embed_images:
            img_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_file)) or ".", "demo_images")
            os.makedirs(img_dir, exist_ok=True)
            safe_id = tablet_id.replace(":", "_").replace(",", "_")
            img_path = os.path.join(img_dir, f"{safe_id}.jpg")
            img.convert("RGB").save(img_path, quality=90)
            out.append(f"![{tablet_id}](demo_images/{safe_id}.jpg)\n")
        out.append(f"**Original text:**\n> {original_display}\n")
        out.append(f"**Masked input ({len(positions)} positions):**\n> {masked_display}\n")

        out.append("### Restoration (masked-token predictions)\n")
        out.append("| # | true token | text-only top-1 | text-only top-3 | vision top-1 | vision top-3 | text-only correct | vision correct |")
        out.append("|---|---|---|---|---|---|---|---|")
        text_correct = vision_correct = 0
        for i, (pos, true_id) in enumerate(zip(positions, true_tokens), 1):
            true_tok = tokenizer.convert_ids_to_tokens([true_id])[0]
            t_top3 = topk_at(text_out["logits"], pos, banned_ids, tokenizer)
            v_top3 = topk_at(vision_out["logits"], pos, banned_ids, tokenizer)
            t_ok = t_top3[0] == true_tok
            v_ok = v_top3[0] == true_tok
            text_correct += t_ok
            vision_correct += v_ok
            out.append(f"| {i} | `{true_tok}` | `{t_top3[0]}` | {', '.join(f'`{t}`' for t in t_top3)} | "
                       f"`{v_top3[0]}` | {', '.join(f'`{t}`' for t in v_top3)} | {'✅' if t_ok else '❌'} | {'✅' if v_ok else '❌'} |")
        n_pos = len(positions)
        out.append(f"\nTop-1 accuracy on this example: text-only {text_correct}/{n_pos} "
                   f"({text_correct / n_pos:.0%}), vision {vision_correct}/{n_pos} ({vision_correct / n_pos:.0%})\n")

        out.append("### Metadata predictions\n")
        out.append(format_metadata_table(label_configs, truth, text_pred, vision_pred))
        out.append("\n---\n")

        if n % 5 == 0:
            print(f"  {n}/{len(indices)} done")

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
