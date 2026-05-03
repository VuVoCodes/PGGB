#!/usr/bin/env python
"""Post-hoc linear-probe evaluation on CREMA-D 3f checkpoints.

For methods that did NOT log probe accuracy during training (MMPareto, AGM,
G-Blend, InfoReg, MILES), we reload the saved best_model.pt, extract per-
modality features on the full CREMA-D 3f train and test splits, fit a fresh
linear probe per modality on train features, and report test-set accuracy.

Output: JSON with per-method, per-seed {audio, visual, gap} numbers that can
be merged into the existing figure and table.

Usage:
    python scripts/post_hoc_probe_cremad.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import CREMADDataset
from src.models import MultimodalModel


def get_dataset(config, split):
    name = config["dataset"]["name"]
    assert name == "cremad", f"Only CREMA-D supported; got {name}"
    return CREMADDataset(
        root=config["dataset"]["root"],
        split=split,
        fps=config["dataset"].get("fps", 1),
        num_frames=config["dataset"].get("num_frames", 1),
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("/home/main/AsynchronousFunction/outputs/sweep_3f")

# Determinism for reproducibility across runs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
METHODS = ["3f_mmpareto", "3f_agm", "3f_gblend", "3f_inforeg_100ep", "3f_miles_t02"]
SEEDS = [42, 123, 456, 789, 1024]

# Probe training hyperparameters (post hoc; features are frozen so we can be aggressive)
PROBE_LR = 1e-3
PROBE_EPOCHS = 300
PROBE_BATCH = 256


@torch.no_grad()
def extract_features(model, loader, modalities):
    model.eval()
    feats = {m: [] for m in modalities}
    labels = []
    for batch in loader:
        inputs = {m: batch[m].to(DEVICE, non_blocking=True) for m in modalities}
        y = batch["label"]
        _, _, features = model(inputs, return_features=True)
        for m in modalities:
            feats[m].append(features[m].detach().float().cpu())
        labels.append(y)
    return {m: torch.cat(feats[m], dim=0) for m in modalities}, torch.cat(labels, dim=0)


PROBE_SEED = 2026  # fixed across all method/seed combinations for protocol parity


def fit_linear_probe(train_feats, train_labels, test_feats, test_labels,
                     num_classes, epochs=PROBE_EPOCHS, lr=PROBE_LR, batch=PROBE_BATCH):
    # Fixed seed for probe init + data order. Ensures any difference across
    # methods is attributable to encoder features, not probe training randomness.
    g = torch.Generator(device=DEVICE)
    g.manual_seed(PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(PROBE_SEED)

    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, num_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    N = train_feats.shape[0]
    train_feats_d = train_feats.to(DEVICE)
    train_labels_d = train_labels.to(DEVICE)
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEVICE, generator=g)
        for i in range(0, N, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            logits = probe(train_feats_d[idx])
            loss = F.cross_entropy(logits, train_labels_d[idx])
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(test_feats.to(DEVICE))
        preds = logits.argmax(-1).cpu()
        acc = (preds == test_labels).float().mean().item()
    return acc


def load_datasets_from_config(config):
    """Return (train_loader, test_loader) using the repo's dataset factory."""
    train_ds = get_dataset(config, split="train")
    test_ds = get_dataset(config, split="test")
    # Disable shuffling for train so feature extraction is deterministic
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out", default="outputs/post_hoc_probe_cremad.json")
    args = parser.parse_args()

    # Cache loaders by config hash (reuse across methods if possible)
    loader_cache = {}
    results = {}

    for method in args.methods:
        results[method] = {}
        for seed in args.seeds:
            run = ROOT / f"{method}_seed{seed}"
            ckpt_path = run / "best_model.pt"
            cfg_path = run / "config.yaml"
            if not ckpt_path.exists() or not cfg_path.exists():
                print(f"  SKIP {method} seed{seed}: missing files")
                continue

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            # Build model and load weights (mirror train.py setup for CREMA-D)
            modalities = cfg["dataset"]["modalities"]
            num_classes = cfg["dataset"]["num_classes"]
            backbone = cfg["model"]["backbone"]
            encoder_config = {
                m: {"backbone": backbone, "pretrained": cfg["model"].get("pretrained", False)}
                for m in modalities
            }
            model = MultimodalModel(
                modalities=modalities,
                num_classes=num_classes,
                encoder_config=encoder_config,
                fusion_type=cfg["model"]["fusion_type"],
                feature_dim=cfg["model"]["feature_dim"],
                fusion_dim=cfg["model"]["fusion_dim"],
            ).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            # Load data (cache by config sub-dict)
            ds_key = json.dumps(cfg["dataset"], sort_keys=True)
            if ds_key not in loader_cache:
                loader_cache[ds_key] = load_datasets_from_config(cfg)
            train_loader, test_loader = loader_cache[ds_key]

            # Extract features
            train_feats, train_labels = extract_features(model, train_loader, modalities)
            test_feats, test_labels = extract_features(model, test_loader, modalities)

            per_mod = {}
            for m in modalities:
                acc = fit_linear_probe(
                    train_feats[m], train_labels,
                    test_feats[m], test_labels,
                    num_classes=num_classes,
                )
                per_mod[m] = acc
                print(f"  {method} seed{seed} probe[{m}]: {acc:.4f}")
            gap = per_mod.get("audio", 0) - per_mod.get("visual", 0)
            per_mod["gap"] = gap
            results[method][f"seed{seed}"] = per_mod
            print(f"  {method} seed{seed} gap: {gap:.4f}")

            # Free memory
            del model, train_feats, test_feats
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary
    print("\n=== Summary (mean +/- std over seeds) ===")
    print(f"{'Method':<25}{'Audio':<20}{'Visual':<20}{'Gap':<20}")
    for method, seed_results in results.items():
        if not seed_results:
            continue
        aud = np.array([v["audio"] for v in seed_results.values()]) * 100
        vis = np.array([v["visual"] for v in seed_results.values()]) * 100
        gap = aud - vis
        print(f"{method:<25}{aud.mean():.2f} +/- {aud.std():.2f}    "
              f"{vis.mean():.2f} +/- {vis.std():.2f}    "
              f"{gap.mean():.2f} +/- {gap.std():.2f}")


if __name__ == "__main__":
    main()
