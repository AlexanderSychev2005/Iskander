"""Pilot vision-conditioned fine-tune: does concatenating a ResNet image
feature into the metadata heads actually help, for provenience/genre/period?

Architecture follows Aeneas's own mechanism (Assael et al. 2025, Methods
p.148): visual input goes through a CNN, and the resulting feature vector is
concatenated with the text embedding before the head's classifier.

--vision_init defaults to "scratch" (random-init ResNet18, fully trainable),
matching Aeneas's own choice more closely than an ImageNet-pretrained
backbone would: Aeneas trained its ResNet-8 from scratch too (ref. 82 there
is just the general He et al. residual-block paper, not a pretrained
checkpoint), and its image budget -- images existed for only ~5% of the
176,861-inscription LED corpus, ~8,843 images (Assael et al. 2025, p.2/p.8)
-- is the same order of magnitude as ours (~5,292 collected, growing as
review continues), not orders of magnitude bigger. "Not enough data for a
CNN from scratch" isn't a safe assumption to make unchecked, especially
given the domain gap: ImageNet is everyday-object photos, tablets are
macro shots of incised clay -- pretrained features may simply not transfer.
--vision_init pretrained (frozen ImageNet ResNet18, linear-probe style) is
kept available for an A/B comparison, not deleted.

Second departure from Aeneas: all three of the heads we collected images
for (provenience/genre/period), not just provenience like Aeneas -- we
already have the data, so let metrics decide rather than assume Aeneas's
province-only result (and its own negative date-head ablation) transfers
directly. language stays text-only (no plausible visual signal, see
session discussion).

The mBERT backbone is FROZEN here: only the classification heads (+ image
projection, when --use_image) are trained. This is a linear-probe setup on
top of the already-trained checkpoints_mbert_metaw1 representations, chosen
because ~5k tablets is not enough to safely fine-tune a 179M-param backbone
without either overfitting or quietly degrading the restoration quality it
already has -- and it keeps the with/without-image comparison clean (the
only architectural difference between the two runs is the image branch).

Run the ablation pair to compare:
    python src/training/train_mbert_vision.py --save_dir checkpoints_vision_with_image
    python src/training/train_mbert_vision.py --save_dir checkpoints_vision_no_image --no_image

Data: built directly from CuneiMLv1.2.json + cdli_cat.csv (same join as
collect_vision_dataset.py), restricted to ids present in
data/vision_dataset_final/crops_manifest.jsonl (reviewed-only by default --
see finalize_vision_crops.py's docstring for why). One example per
transliteration line, all lines of a tablet sharing that tablet's single
crop -- matches the granularity the base checkpoint was trained on.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from torchvision import models as tv_models
from torchvision import transforms
from transformers import AutoModelForMaskedLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_pipeline.collect_vision_dataset import load_all_candidates
from src.data_pipeline.prepare_hf_dataset import (
    GENRE_LABELS, LANGUAGE_LABELS, PERIOD_LABELS, PROVENIENCE_LABELS,
    clean_transliteration, label_to_idx, map_genre, map_language, map_period, map_provenience,
)
from src.training.train_mbert import mark_damage_signals

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROPS_DIR = os.path.join(BASE_DIR, "data", "vision_dataset_final")
CROPS_MANIFEST = os.path.join(CROPS_DIR, "crops_manifest.jsonl")
FACES = ("obverse", "reverse", "left", "right", "top", "bottom")
IMAGE_HEADS = ("period", "genre", "provenience")  # language excluded: no visual signal (session finding)

IMG_SIZE = 224  # stored crops are 512x512 (finalize_vision_crops.py); resized here per-model
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class MBertVisionMultiTask(nn.Module):
    def __init__(self, model_name, num_period, num_genre, num_language, num_provenience,
                 meta_weight=1.0, use_image=True, vision_init="pretrained", img_feat_dim=128):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden_size = self.backbone.config.hidden_size
        self.use_image = use_image
        self.vision_init = vision_init
        self.meta_weight = meta_weight

        head_in = hidden_size + img_feat_dim if use_image else hidden_size
        self.period_head = nn.Linear(head_in, num_period)
        self.genre_head = nn.Linear(head_in, num_genre)
        self.language_head = nn.Linear(hidden_size, num_language)  # never sees image
        self.provenience_head = nn.Linear(head_in, num_provenience)

        if use_image:
            # pretrained: ImageNet-frozen ResNet18 as a fixed feature extractor
            # (only vision_proj trains) -- the "safe default" choice.
            # scratch: same ResNet18 *architecture*, random init, fully
            # trainable -- Aeneas's own ResNet-8 was trained from scratch too,
            # and its image count (5% of 176,861 LED inscriptions =~ 8,843,
            # Assael et al. 2025 p.142) is the same order of magnitude as ours
            # (~5,292 once review is complete), not orders of magnitude more --
            # so "not enough data to train a CNN from scratch" isn't a safe
            # assumption here and is worth testing empirically instead of
            # assuming. See train_mbert_vision.py module docstring.
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if vision_init == "pretrained" else None
            resnet = tv_models.resnet18(weights=weights)
            if vision_init == "pretrained":
                for p in resnet.parameters():
                    p.requires_grad = False
            resnet.fc = nn.Identity()
            self.vision_cnn = resnet
            self.vision_proj = nn.Linear(512, img_feat_dim)

    def load_backbone_and_matching_heads(self, checkpoint_path):
        """Warm-start from a text-only checkpoint: backbone always matches;
        heads only match when --no_image (identical shape to the base
        model), in which case this is just a continuation of that
        checkpoint. Shape-mismatched heads (the --use_image run's wider
        concat-input heads) are silently left at their fresh init."""
        state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
        own_state = self.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in own_state and own_state[k].shape == v.shape}
        self.load_state_dict(filtered, strict=False)
        return len(filtered), len(state_dict)

    def forward(self, input_ids, attention_mask=None, pixel_values=None,
                period_labels=None, genre_labels=None, language_labels=None, provenience_labels=None):
        with torch.no_grad():
            bert_out = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = bert_out.last_hidden_state[:, 0, :]

        if self.use_image:
            img_feat = self.vision_proj(self.vision_cnn(pixel_values))
            head_in = torch.cat([cls_embed, img_feat], dim=-1)
        else:
            head_in = cls_embed

        period_logits = self.period_head(head_in)
        genre_logits = self.genre_head(head_in)
        language_logits = self.language_head(cls_embed)
        provenience_logits = self.provenience_head(head_in)

        loss = None
        if any(l is not None for l in [period_labels, genre_labels, language_labels, provenience_labels]):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = 0.0
            for logits, lbl in [(period_logits, period_labels), (genre_logits, genre_labels),
                                 (language_logits, language_labels), (provenience_logits, provenience_labels)]:
                if lbl is not None and (lbl != -100).any():
                    loss += self.meta_weight * loss_fct(logits, lbl)

        return {
            "loss": loss,
            "period_logits": period_logits,
            "genre_logits": genre_logits,
            "language_logits": language_logits,
            "provenience_logits": provenience_logits,
        }


class VisionLineDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length, use_image):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_image = use_image

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex["text"], truncation=True, max_length=self.max_length)
        item = {
            "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "period_labels": ex["period_labels"], "genre_labels": ex["genre_labels"],
            "language_labels": ex["language_labels"], "provenience_labels": ex["provenience_labels"],
        }
        if self.use_image:
            img = Image.open(ex["image_path"]).convert("RGB")
            item["pixel_values"] = IMG_TRANSFORM(img)
        return item


class VisionCollator:
    def __init__(self, tokenizer, use_image):
        self.tokenizer = tokenizer
        self.use_image = use_image

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            [{"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]} for ex in examples],
            return_tensors="pt")
        for task in ["period", "genre", "language", "provenience"]:
            batch[f"{task}_labels"] = torch.tensor([ex[f"{task}_labels"] for ex in examples], dtype=torch.long)
        if self.use_image:
            batch["pixel_values"] = torch.stack([ex["pixel_values"] for ex in examples])
        return batch


def build_examples(ids, crop_dir):
    candidates = load_all_candidates()
    examples = []
    for pid in ids:
        pair = candidates.get(pid)
        if pair is None:
            continue
        it, meta = pair
        labels = {
            "period_labels": label_to_idx(map_period(meta.get("period", "")), PERIOD_LABELS),
            "genre_labels": label_to_idx(map_genre(meta.get("genre", "")), GENRE_LABELS),
            "language_labels": label_to_idx(map_language(meta.get("language", "")), LANGUAGE_LABELS),
            "provenience_labels": label_to_idx(map_provenience(meta.get("provenience", "")), PROVENIENCE_LABELS),
        }
        image_path = os.path.join(crop_dir, f"{pid}.jpg")
        text_dict = it.get("text") or {}
        for face in FACES:
            for line_obj in (text_dict.get(face) or []):
                if not isinstance(line_obj, dict):
                    continue
                raw = line_obj.get("raw", "")
                cleaned = mark_damage_signals(clean_transliteration(raw))
                if not cleaned:
                    continue
                examples.append({"tablet_id": pid, "text": cleaned, "image_path": image_path, **labels})
    return examples


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred.predictions, eval_pred.label_ids
    metrics = {}
    for i, task in enumerate(["period", "genre", "language", "provenience"]):
        preds = np.argmax(logits[i], axis=-1)
        lbl = label_ids[i]
        mask = lbl != -100
        if not mask.any():
            metrics[f"{task}_acc"] = metrics[f"{task}_macro_f1"] = 0.0
            continue
        p, l = preds[mask], lbl[mask]
        metrics[f"{task}_acc"] = float((p == l).mean())
        metrics[f"{task}_macro_f1"] = float(f1_score(l, p, average="macro", zero_division=0))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=r"C:\Programming\akkadian\checkpoints_mbert_metaw1\final_model",
                         help="Text-only base checkpoint to warm-start the backbone (and heads, if --no_image) from")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--save_dir", type=str, default="checkpoints_mbert_vision")
    parser.add_argument("--no_image", action="store_true", help="Ablation control: identical setup minus the vision branch")
    parser.add_argument("--vision_init", type=str, choices=["pretrained", "scratch"], default="scratch",
                         help="scratch (default): random-init ResNet18, fully trainable -- matches Aeneas's own "
                              "from-scratch ResNet-8 more closely; our image count (~5.3k) is the same order of "
                              "magnitude as theirs (~8.8k), not too small to bother. pretrained: frozen ImageNet "
                              "ResNet18, only the projection trains -- kept for an A/B comparison.")
    parser.add_argument("--include_unreviewed", action="store_true",
                         help="Also use ids whose bbox was never manually reviewed (raw CuneiML bbox, ~58%% reliable) -- off by default")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Default suits --vision_init scratch (whole ResNet18 trains, needs a real CNN-training LR). "
                              "Use something higher (e.g. 1e-3) for --vision_init pretrained, where only small heads/proj train.")
    parser.add_argument("--meta_weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_image = not args.no_image

    with open(CROPS_MANIFEST, encoding="utf-8") as f:
        crops = [json.loads(line) for line in f]
    ids = [row["id"] for row in crops if args.include_unreviewed or row["reviewed"]]
    print(f"{len(ids)} tablet ids available ({'including' if args.include_unreviewed else 'excluding'} unreviewed bboxes)")

    import random
    random.seed(args.seed)
    ids = sorted(ids)
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids, train_ids = set(ids[:n_val]), set(ids[n_val:])

    print("Building examples...")
    train_examples = build_examples(sorted(train_ids), CROPS_DIR)
    val_examples = build_examples(sorted(val_ids), CROPS_DIR)
    print(f"Train: {len(train_examples)} lines from {len(train_ids)} tablets | Val: {len(val_examples)} lines from {len(val_ids)} tablets")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    train_dataset = VisionLineDataset(train_examples, tokenizer, args.max_length, use_image)
    val_dataset = VisionLineDataset(val_examples, tokenizer, args.max_length, use_image)
    collator = VisionCollator(tokenizer, use_image)

    with open(r"C:\Programming\akkadian\data\processed\label_configs.json", encoding="utf-8") as f:
        label_configs = json.load(f)
    tasks = ["period", "genre", "language", "provenience"]
    num_labels = {task: len(label_configs[task]["labels"]) for task in tasks}

    model = MBertVisionMultiTask(
        args.model_name, num_period=num_labels["period"], num_genre=num_labels["genre"],
        num_language=num_labels["language"], num_provenience=num_labels["provenience"],
        meta_weight=args.meta_weight, use_image=use_image, vision_init=args.vision_init,
    )
    n_loaded, n_total = model.load_backbone_and_matching_heads(args.checkpoint)
    print(f"Warm-started {n_loaded}/{n_total} tensors from {args.checkpoint} (use_image={use_image}, vision_init={args.vision_init})")

    os.makedirs(args.save_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=20,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        report_to=[],
        label_names=["period_labels", "genre_labels", "language_labels", "provenience_labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")
    with open(os.path.join(args.save_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
