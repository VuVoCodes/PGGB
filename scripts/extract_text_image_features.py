#!/usr/bin/env python
"""
Extract pre-computed BERT (text) + ResNet18 (image) features for text+image datasets.

One-time pre-extraction to speed up subsequent training. Mirrors the
pre-extracted-features pattern used for MOSEI/MOSI.

Output format:
    data/<dataset>/features/<split>.pt with keys:
        - text: (N, 768) BERT [CLS] embeddings
        - image: (N, 512) ResNet18 pooled features
        - label: (N,) class labels

Usage:
    python scripts/extract_text_image_features.py --dataset sarcasm
    python scripts/extract_text_image_features.py --dataset twitter
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_image_encoder():
    """Frozen ResNet18 pretrained on ImageNet → 512-d features."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  # remove classifier, keep 512-d features
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)


def build_text_encoder():
    """Frozen BERT-base-uncased → 768-d [CLS] embeddings."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model.to(DEVICE)


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# Sarcasm (MMSD v1, parquet format from HuggingFace coderchen01/MMSD2.0)
# =============================================================================

class SarcasmRawDataset(Dataset):
    """Loads from parquet files; returns (PIL_image, text_str, label)."""

    def __init__(self, parquet_files):
        import pyarrow.parquet as pq
        tables = [pq.read_table(p) for p in parquet_files]
        # Concatenate
        import pyarrow as pa
        self.table = pa.concat_tables(tables)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1).to_pylist()[0]
        # row['image'] may be bytes or dict {'bytes': ..., 'path': ...}
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes")
        else:
            img_bytes = img_data
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img, str(row["text"]), int(row["label"])


def extract_sarcasm(split):
    """Extract features for one Sarcasm split."""
    base = "/home/main/AsynchronousFunction/data/Sarcasm/mmsd2/mmsd-v1"
    if split == "train":
        files = [f"{base}/train-{i:05d}-of-00004.parquet" for i in range(4)]
    elif split == "valid":
        files = [f"{base}/validation-00000-of-00001.parquet"]
    elif split == "test":
        files = [f"{base}/test-00000-of-00001.parquet"]
    else:
        raise ValueError(f"Unknown split: {split}")

    ds = SarcasmRawDataset(files)
    print(f"Sarcasm {split}: {len(ds)} samples")
    return ds


# =============================================================================
# Twitter15 (TomBERT format, TSV + image directory)
# =============================================================================

class TwitterRawDataset(Dataset):
    """TomBERT Twitter2015 TSV + images. Returns (PIL_image, text_str, label)."""

    def __init__(self, tsv_path, image_dir):
        self.image_dir = image_dir
        self.rows = []
        with open(tsv_path, "r") as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                # cols: index, label, image_id, text_before_target, text_after_target
                idx, label, img_id, text_before, text_after = parts[:5]
                # Combine text — replace $T$ placeholder convention
                text = (text_before + " " + text_after).strip()
                self.rows.append((img_id, text, int(label)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_id, text, label = self.rows[idx]
        img_path = os.path.join(self.image_dir, img_id)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # missing image → use black placeholder
            img = Image.new("RGB", (224, 224))
        return img, text, label


def extract_twitter(split):
    """Extract features for one Twitter15 split."""
    base = "/home/main/AsynchronousFunction/data/Twitter15/text_data/absa_data/twitter2015"
    img_dir = "/home/main/AsynchronousFunction/data/Twitter15/IJCAI2019_data/twitter2015_images"
    split_map = {"train": "train.tsv", "valid": "dev.tsv", "test": "test.tsv"}
    ds = TwitterRawDataset(f"{base}/{split_map[split]}", img_dir)
    print(f"Twitter {split}: {len(ds)} samples")
    return ds


# =============================================================================
# UPMC-Food101 (Wang et al., ICME 2015)
# Format: CSVs with class/text/image_name + image directories per split
# =============================================================================

class Food101RawDataset(Dataset):
    """UPMC-Food101 dataset loader.

    CSV format (no header): image_filename, title_text, class_name
    Image path: <image_dir>/<class_name>/<image_filename>
    """

    def __init__(self, csv_path, image_dir, class_to_idx):
        import csv
        self.image_dir = image_dir
        self.class_to_idx = class_to_idx
        self.rows = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                # Format: image_name, title, class_name
                img_name = row[0].strip()
                text = row[1].strip()
                cls_name = row[-1].strip()
                if cls_name not in class_to_idx:
                    continue
                self.rows.append((cls_name, img_name, text))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        cls_name, img_name, text = self.rows[idx]
        # Image path: <image_dir>/<class>/<image_name>
        img_path = os.path.join(self.image_dir, cls_name, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        label = self.class_to_idx[cls_name]
        return img, text, label


def extract_food101(split):
    """Extract features for one Food101 split (train or test)."""
    base = "/home/main/AsynchronousFunction/data/Food101/UPMC-Food-101"
    train_csv = f"{base}/texts/train_titles.csv"
    test_csv = f"{base}/texts/test_titles.csv"
    train_img = f"{base}/images/train"
    test_img = f"{base}/images/test"
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Food101 train CSV not found: {train_csv}")

    # Build class_to_idx from train directory listing
    class_names = sorted(os.listdir(train_img))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Food101 classes: {len(class_names)}")

    if split == "train":
        ds = Food101RawDataset(train_csv, train_img, class_to_idx)
    elif split == "test":
        ds = Food101RawDataset(test_csv, test_img, class_to_idx)
    else:
        raise ValueError(f"Food101 only supports train/test splits, got: {split}")

    print(f"Food101 {split}: {len(ds)} samples")
    return ds


# =============================================================================
# Unified extraction loop
# =============================================================================

def collate(batch):
    """Custom collate: stack PIL images into tensor, return list of texts."""
    imgs = torch.stack([IMG_TRANSFORM(item[0]) for item in batch])
    texts = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return imgs, texts, labels


@torch.no_grad()
def extract_features(dataset, image_encoder, tokenizer, text_encoder, batch_size=64, max_text_len=128):
    """Run BERT + ResNet18 over dataset, return (text_features, image_features, labels)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate, pin_memory=True)

    all_text, all_img, all_label = [], [], []
    for imgs, texts, labels in tqdm(loader, desc="Extracting"):
        imgs = imgs.to(DEVICE, non_blocking=True)

        # Image: ResNet18 → 512-d
        img_feats = image_encoder(imgs).cpu()  # (B, 512)

        # Text: BERT → 768-d [CLS]
        tokens = tokenizer(texts, padding=True, truncation=True,
                           max_length=max_text_len, return_tensors="pt").to(DEVICE)
        out = text_encoder(**tokens)
        text_feats = out.last_hidden_state[:, 0, :].cpu()  # CLS token (B, 768)

        all_text.append(text_feats)
        all_img.append(img_feats)
        all_label.append(labels)

    return torch.cat(all_text), torch.cat(all_img), torch.cat(all_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sarcasm", "twitter", "food101"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    print("Building image encoder (ResNet18)...")
    image_encoder = build_image_encoder()

    print("Building text encoder (BERT-base-uncased)...")
    tokenizer, text_encoder = build_text_encoder()

    dataset_dir = {"sarcasm": "Sarcasm", "twitter": "Twitter15", "food101": "Food101"}[args.dataset]
    out_dir = Path(f"data/{dataset_dir}/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_fn = {"sarcasm": extract_sarcasm, "twitter": extract_twitter, "food101": extract_food101}[args.dataset]

    # Food101 only has train/test (no valid split)
    splits = ["train", "test"] if args.dataset == "food101" else ["train", "valid", "test"]
    for split in splits:
        print(f"\n=== {args.dataset.upper()} {split} ===")
        ds = extract_fn(split)
        text, image, label = extract_features(
            ds, image_encoder, tokenizer, text_encoder, batch_size=args.batch_size
        )
        out_path = out_dir / f"{split}.pt"
        torch.save({"text": text, "image": image, "label": label}, out_path)
        print(f"Saved {out_path}: text={text.shape}, image={image.shape}, label={label.shape}")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
