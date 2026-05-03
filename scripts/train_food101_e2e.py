#!/usr/bin/env python
"""
End-to-end Food101 training with trainable BERT + ResNet18.

Purpose: test the hypothesis that our method requires representation learning
(gradient flow into feature extractors), not frozen features. If boost+OGM-GE
beats baseline on end-to-end Food101, the hypothesis is confirmed.

Usage:
    python scripts/train_food101_e2e.py --mode baseline --seed 42
    python scripts/train_food101_e2e.py --mode boost_ogm_ge --seed 42
"""
import argparse
import os
import sys
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from transformers import BertModel, BertTokenizer

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Dataset (raw images + text, no pre-extraction)
# =============================================================================

IMG_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
IMG_TRANSFORM_TEST = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Food101RawE2E(Dataset):
    """UPMC-Food101 raw dataset for end-to-end fine-tuning."""

    def __init__(self, split, tokenizer, max_len=50):
        base = "/home/main/AsynchronousFunction/data/Food101/UPMC-Food-101"
        self.image_dir = f"{base}/images/{split}"
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = IMG_TRANSFORM_TRAIN if split == "train" else IMG_TRANSFORM_TEST

        # Build class list from directory structure
        self.classes = sorted(os.listdir(self.image_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Load CSV
        csv_path = f"{base}/texts/{split}_titles.csv"
        self.rows = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                img_name = row[0].strip()
                text = row[1].strip()
                cls_name = row[-1].strip()
                if cls_name not in self.class_to_idx:
                    continue
                self.rows.append((cls_name, img_name, text))

        print(f"Food101 E2E {split}: {len(self.rows)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        cls_name, img_name, text = self.rows[idx]
        img_path = os.path.join(self.image_dir, cls_name, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        img = self.transform(img)
        label = self.class_to_idx[cls_name]
        return img, text, label


def collate_fn(batch, tokenizer, max_len=50):
    imgs = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    tokens = tokenizer(texts, padding=True, truncation=True,
                       max_length=max_len, return_tensors="pt")
    return imgs, tokens, labels


# =============================================================================
# Model (BERT + ResNet18, both trainable)
# =============================================================================

class LSTMTextEncoder(nn.Module):
    """LSTM text encoder trained from scratch. Uses BertTokenizer vocab
    (30522) but RANDOMLY INITIALIZED embedding + LSTM (no BERT weights).
    Matches the "from scratch" regime used for CREMA-D's visual encoder.
    """

    def __init__(self, vocab_size=30522, embed_dim=128, lstm_hidden=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.output_dim = lstm_hidden * 2  # bidirectional

    def forward(self, tokens):
        """tokens: dict with input_ids and attention_mask."""
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        emb = self.embedding(input_ids)  # (B, T, E)
        out, _ = self.lstm(emb)          # (B, T, 2H)
        # Mean-pool over valid tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled  # (B, 2H)


class BertImageTextModel(nn.Module):
    """Joint text + image model with choice of text encoder and image init.

    text_encoder: "bert" (pretrained) or "lstm" (from scratch)
    image_pretrained: True (ImageNet) or False (from scratch)
    """

    def __init__(self, num_classes=101, hidden_dim=512,
                 text_encoder_type="bert", image_pretrained=True):
        super().__init__()
        self.text_encoder_type = text_encoder_type

        if text_encoder_type == "bert":
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
            self.text_proj = nn.Linear(768, hidden_dim)
        elif text_encoder_type == "lstm":
            self.text_encoder = LSTMTextEncoder()
            self.text_proj = nn.Linear(self.text_encoder.output_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown text encoder: {text_encoder_type}")

        if image_pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None  # random init — from scratch
        resnet = models.resnet18(weights=weights)
        resnet.fc = nn.Identity()
        self.image_encoder = resnet
        self.image_proj = nn.Linear(512, hidden_dim)

        # Fusion: concat [text|image] → 2*hidden_dim → hidden_dim → num_classes
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_text(self, tokens):
        """Returns text features (B, hidden_dim)."""
        if self.text_encoder_type == "bert":
            out = self.text_encoder(**{k: v.to(next(self.parameters()).device) for k, v in tokens.items()})
            cls = out.last_hidden_state[:, 0, :]  # (B, 768)
            return self.text_proj(cls)
        else:  # lstm
            feat = self.text_encoder(tokens)  # (B, 2H)
            return self.text_proj(feat)

    def encode_image(self, imgs):
        """Returns image features (B, hidden_dim)."""
        feats = self.image_encoder(imgs)  # (B, 512)
        return self.image_proj(feats)  # (B, hidden_dim)

    def forward(self, imgs, tokens):
        text_feat = self.encode_text(tokens)
        img_feat = self.encode_image(imgs)
        joint = torch.cat([text_feat, img_feat], dim=-1)
        logits = self.fusion(joint)
        return logits, text_feat, img_feat


# =============================================================================
# Probe manager (same pattern as BraTS: split-batch, EMA, K=20, scale_ema=0.3)
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class ProbeManager:
    """Shared ProbeManager-lite: split-batch, EMA accuracy, EMA scales, K=20."""

    def __init__(self, hidden_dim, num_classes, device,
                 probe_lr=1e-3, eval_freq=20, scale_ema_mu=0.3, ema_alpha=0.1,
                 scale_max=2.0):
        self.modalities = ["text", "image"]
        self.eval_freq = eval_freq
        self.scale_ema_mu = scale_ema_mu
        self.ema_alpha = ema_alpha
        self.scale_max = scale_max
        self.probes = {m: LinearProbe(hidden_dim, num_classes).to(device) for m in self.modalities}
        self.optims = {m: optim.Adam(p.parameters(), lr=probe_lr) for m, p in self.probes.items()}
        self.acc_ema = {m: None for m in self.modalities}  # None = not seeded yet
        self.stored_scales = {m: 1.0 for m in self.modalities}

    def train_and_eval(self, features_dict, labels):
        """Split-batch: train probe on first half, eval on second."""
        B = labels.size(0)
        split = B // 2
        if split < 2:
            return {m: 0.5 for m in self.modalities}

        accs = {}
        for m in self.modalities:
            feat = features_dict[m].detach().float()
            # Train half
            probe = self.probes[m]
            self.optims[m].zero_grad()
            pred_train = probe(feat[:split])
            loss = F.cross_entropy(pred_train, labels[:split])
            loss.backward()
            self.optims[m].step()
            # Eval half
            with torch.no_grad():
                pred_eval = probe(feat[split:])
                acc = (pred_eval.argmax(dim=1) == labels[split:]).float().mean().item()
            accs[m] = acc
            # Update EMA (seed from first measurement)
            if self.acc_ema[m] is None:
                self.acc_ema[m] = acc
            else:
                self.acc_ema[m] = self.ema_alpha * acc + (1 - self.ema_alpha) * self.acc_ema[m]
        return accs

    def update_scales(self, alpha=0.75):
        """Compute raw boost scales from EMA accs, apply EMA smoothing."""
        if any(v is None for v in self.acc_ema.values()):
            return
        use_accs = self.acc_ema
        mn, mx = min(use_accs.values()), max(use_accs.values())
        gap = mx - mn + 1e-8
        for m in self.modalities:
            rel_weak = 1.0 - (use_accs[m] - mn) / gap
            raw_scale = min(1.0 + alpha * rel_weak, self.scale_max)
            self.stored_scales[m] = self.scale_ema_mu * raw_scale + (1 - self.scale_ema_mu) * self.stored_scales[m]

    def get_scales(self):
        return self.stored_scales.copy()

    def get_gap(self):
        if any(v is None for v in self.acc_ema.values()):
            return 0.0
        return max(self.acc_ema.values()) - min(self.acc_ema.values())


# =============================================================================
# OGM-GE (2-modality, applied to named encoder params)
# =============================================================================

def apply_ogm_ge(model, text_logits_proxy, img_logits_proxy, targets, alpha=0.8, epoch=0, start=0, end=50):
    """OGM-GE for text+image model.

    Uses per-modality linear classifier outputs as the discriminative proxy.
    Since this model doesn't have auxiliary classifiers, we use the projected
    features directly with the final classifier weights.

    Actually: compute softmax of unimodal predictions (from probes) to match OGM-GE original.
    For simplicity, compute gradient-norm-based ratio here as in BraTS.
    """
    if not (start <= epoch <= end):
        return {"text": 1.0, "image": 1.0}

    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()

    # Compute gradient norms per encoder
    grad_norms = {}
    for mname, key in [("text", "text_encoder"), ("image", "image_encoder")]:
        total = 0.0
        count = 0
        for name, param in model.named_parameters():
            if key in name and param.grad is not None and len(param.grad.shape) >= 2:
                total += param.grad.data.norm().item() ** 2
                count += 1
        grad_norms[mname] = (total ** 0.5) if count > 0 else 0.0

    mean_norm = sum(grad_norms.values()) / len(grad_norms)
    if mean_norm < 1e-8:
        return {"text": 1.0, "image": 1.0}

    coeffs = {}
    for mname in ["text", "image"]:
        ratio = grad_norms[mname] / (mean_norm + 1e-8)
        if ratio > 1.0:
            coeffs[mname] = 1 - tanh(alpha * relu(torch.tensor(ratio))).item()
        else:
            coeffs[mname] = 1.0

    # Apply to encoder parameters
    for mname, key in [("text", "text_encoder"), ("image", "image_encoder")]:
        if coeffs[mname] < 1.0:
            for name, param in model.named_parameters():
                if key in name and param.grad is not None and len(param.grad.shape) >= 2:
                    noise = torch.zeros_like(param.grad).normal_(0, param.grad.std().item() + 1e-8)
                    param.grad.data = param.grad.data * coeffs[mname] + noise
    return coeffs


def apply_boost_scales(model, scales):
    """Scale encoder gradients by per-modality boost factors."""
    for mname, key in [("text", "text_encoder"), ("image", "image_encoder")]:
        s = scales[mname]
        if abs(s - 1.0) < 1e-6:
            continue
        for name, param in model.named_parameters():
            if key in name and param.grad is not None:
                param.grad.data = param.grad.data * s


# =============================================================================
# Training
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "boost_ogm_ge", "ogm_ge", "boost_only"], default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--max-text-len", type=int, default=50)
    parser.add_argument("--ogm-alpha", type=float, default=0.8)
    parser.add_argument("--ogm-start", type=int, default=0)
    parser.add_argument("--ogm-end", type=int, default=15)  # first half of 30 epochs
    parser.add_argument("--boost-alpha", type=float, default=0.75)
    parser.add_argument("--boost-scale-max", type=float, default=2.0)
    parser.add_argument("--text-encoder", choices=["bert", "lstm"], default="bert")
    parser.add_argument("--image-pretrained", type=int, default=1, help="1=ImageNet, 0=from scratch")
    parser.add_argument("--output-dir", type=str, default="outputs/sweep_food101_e2e")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--debug-subset", type=int, default=0, help="If >0, use only N train samples for smoke test")
    args = parser.parse_args()

    set_seed(args.seed)

    exp_name = args.exp_name or f"food101_e2e_{args.mode}_seed{args.seed}"
    out_dir = Path(args.output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Experiment: {exp_name}")
    log(f"Mode: {args.mode}, Seed: {args.seed}")
    log(f"Epochs={args.epochs}, Batch={args.batch_size}, LR encoder={args.lr_encoder}, head={args.lr_head}")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Datasets
    train_ds = Food101RawE2E("train", tokenizer, max_len=args.max_text_len)
    test_ds = Food101RawE2E("test", tokenizer, max_len=args.max_text_len)

    # Optional debug subset
    if args.debug_subset > 0:
        train_ds.rows = train_ds.rows[:args.debug_subset]
        test_ds.rows = test_ds.rows[:min(args.debug_subset, len(test_ds.rows))]
        log(f"DEBUG SUBSET: train={len(train_ds.rows)}, test={len(test_ds.rows)}")

    def _collate(b):
        return collate_fn(b, tokenizer, args.max_text_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=_collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, collate_fn=_collate, pin_memory=True)

    # Model
    num_classes = len(train_ds.classes)
    model = BertImageTextModel(
        num_classes=num_classes,
        text_encoder_type=args.text_encoder,
        image_pretrained=bool(args.image_pretrained),
    ).to(DEVICE)
    log(f"Model: {args.text_encoder.upper()} + ResNet18 (pretrained={bool(args.image_pretrained)}), num_classes={num_classes}, params={sum(p.numel() for p in model.parameters()):,}")

    # Optimizer: different LRs for pretrained vs head
    encoder_params = list(model.text_encoder.parameters()) + list(model.image_encoder.parameters())
    head_params = list(model.text_proj.parameters()) + list(model.image_proj.parameters()) + list(model.fusion.parameters())
    optimizer = optim.Adam([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    # Probe manager for boost modes
    probe_mgr = None
    if args.mode in ("boost_ogm_ge", "boost_only"):
        probe_mgr = ProbeManager(hidden_dim=512, num_classes=num_classes, device=DEVICE,
                                 scale_max=args.boost_scale_max)
        log(f"Probe manager: K={probe_mgr.eval_freq}, scale_ema={probe_mgr.scale_ema_mu}, "
            f"alpha={args.boost_alpha}, scale_max={args.boost_scale_max}")

    # Training
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for i_batch, (imgs, tokens, labels) in enumerate(train_loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            tokens = {k: v.to(DEVICE, non_blocking=True) for k, v in tokens.items()}

            optimizer.zero_grad()
            logits, text_feat, img_feat = model(imgs, tokens)
            loss = criterion(logits, labels)
            loss.backward()

            # ===== Gradient modulation =====
            if args.mode in ("ogm_ge", "boost_ogm_ge"):
                apply_ogm_ge(model, text_feat, img_feat, labels,
                             alpha=args.ogm_alpha, epoch=epoch,
                             start=args.ogm_start, end=args.ogm_end)

            if probe_mgr is not None:
                # Every K batches: refresh probes and update scales
                if (i_batch + 1) % probe_mgr.eval_freq == 0:
                    features = {"text": text_feat, "image": img_feat}
                    probe_mgr.train_and_eval(features, labels)
                    probe_mgr.update_scales(alpha=args.boost_alpha)
                # Every batch: apply stored scales
                apply_boost_scales(model, probe_mgr.get_scales())

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if (i_batch + 1) % 100 == 0:
                gap = probe_mgr.get_gap() if probe_mgr else 0.0
                log(f"  ep{epoch} b{i_batch+1}/{len(train_loader)} loss={loss.item():.3f} gap={gap:.3f}")

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, tokens, labels in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                tokens = {k: v.to(DEVICE, non_blocking=True) for k, v in tokens.items()}
                logits, _, _ = model(imgs, tokens)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        elapsed = time.time() - t0
        log(f"Epoch {epoch}: Train Loss={total_loss/n_batches:.4f}, Test Acc={acc:.2f}% [{elapsed:.0f}s]")

        if acc > best_acc:
            best_acc = acc
            log(f"New best: {best_acc:.2f}%")

    log(f"Training complete. Best accuracy: {best_acc/100:.4f}")
    log_file.close()


if __name__ == "__main__":
    main()
