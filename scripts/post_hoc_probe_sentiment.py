#!/usr/bin/env python
"""Post-hoc linear-probe evaluation on sentiment-benchmark checkpoints.

Rebuttal follow-up (gN93 Q3): reload saved best_model.pt from genuine-CH-SIMS
(outputs/sweep_chsims_true) or CMU-MOSI (outputs/sweep_mosi) runs, extract
per-modality features on train/test splits, fit a fresh linear probe per
modality on train features, and report test-set accuracy plus the
max-min utilization gap.

Protocol mirrors scripts/post_hoc_probe_cremad.py (fixed probe seed 2026,
Adam lr 1e-3, 300 epochs, batch 256).

Usage:
    python scripts/post_hoc_probe_sentiment.py --dataset chsims_true
    python scripts/post_hoc_probe_sentiment.py --dataset mosi
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import CHSIMSDataset, CMUMOSIDataset
from src.models import MultimodalModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROBE_SEED = 2026
PROBE_LR = 1e-3
PROBE_EPOCHS = 300
PROBE_BATCH = 256

TARGETS = {
    "chsims_true": {
        "root_dir": "outputs/sweep_chsims_true",
        "methods": ["tchsims_baseline", "tchsims_ogm_ge", "tchsims_boost_only",
                    "tchsims_boost_ogm_a075"],
    },
    # CH-SIMS: the dataset actually contained in data/CH-SIMS (see
    # Reviews/rebuttal_plan.md); checkpoints are the original March runs.
    "chsims": {
        "root_dir": "outputs/sweep_chsims",
        "methods": ["chsims_baseline", "chsims_ogm_ge", "chsims_boost_only",
                    "chsims_boost_ogm_a075"],
    },
    "mosi": {
        "root_dir": "outputs/sweep_mosi",
        "methods": ["mosi_baseline", "mosi_ogmge", "mosi_boost_only",
                    "mosi_boost_ogm"],
    },
}
SEEDS = [42, 123, 456, 789, 1024]


def get_dataset(cfg, split):
    name = cfg["dataset"]["name"]
    root = cfg["dataset"]["root"]
    if name == "chsims":
        return CHSIMSDataset(root=root, split=split if split == "train" else "test")
    if name == "mosi":
        return CMUMOSIDataset(root=root, split="test" if split == "test" else split)
    raise ValueError(f"Unsupported dataset {name}")


@torch.no_grad()
def extract_features(model, loader, modalities):
    model.eval()
    feats = {m: [] for m in modalities}
    labels = []
    for batch in loader:
        inputs = {m: batch[m].to(DEVICE, non_blocking=True) for m in modalities}
        _, _, features = model(inputs, return_features=True)
        for m in modalities:
            feats[m].append(features[m].detach().float().cpu())
        labels.append(batch["label"])
    return {m: torch.cat(feats[m], dim=0) for m in modalities}, torch.cat(labels, dim=0)


def fit_linear_probe(train_feats, train_labels, test_feats, test_labels, num_classes):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(PROBE_SEED)

    probe = nn.Linear(train_feats.shape[1], num_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    N = train_feats.shape[0]
    tf, tl = train_feats.to(DEVICE), train_labels.to(DEVICE)
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(N, device=DEVICE, generator=g)
        for i in range(0, N, PROBE_BATCH):
            idx = perm[i:i + PROBE_BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(probe(tf[idx]), tl[idx])
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        preds = probe(test_feats.to(DEVICE)).argmax(-1).cpu()
    return (preds == test_labels).float().mean().item()


def build_model(cfg, modalities):
    backbone = cfg["model"]["backbone"]
    dim_key_map = {"text": "text_dim", "audio": "audio_dim", "vision": "visual_dim",
                   "visual": "visual_dim", "image": "image_dim"}
    encoder_config = {}
    for m in modalities:
        enc = {"backbone": backbone, "pretrained": cfg["model"].get("pretrained", False)}
        if backbone == "mlp":
            enc["input_dim"] = cfg["dataset"].get(dim_key_map.get(m, f"{m}_dim"), 300)
            enc["dropout"] = cfg["model"].get("dropout", 0.3)
        encoder_config[m] = enc
    return MultimodalModel(
        modalities=modalities,
        num_classes=cfg["dataset"]["num_classes"],
        encoder_config=encoder_config,
        fusion_type=cfg["model"]["fusion_type"],
        feature_dim=cfg["model"]["feature_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
    ).to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(TARGETS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    target = TARGETS[args.dataset]
    root = Path(".") / target["root_dir"]
    out_path = Path(f"outputs/post_hoc_probe_{args.dataset}.json")

    loader_cache = {}
    results = {}
    for method in target["methods"]:
        results[method] = {}
        for seed in args.seeds:
            run = root / f"{method}_seed{seed}"
            ckpt_path, cfg_path = run / "best_model.pt", run / "config.yaml"
            if not ckpt_path.exists() or not cfg_path.exists():
                print(f"  SKIP {method} seed{seed}: missing files")
                continue
            cfg = yaml.safe_load(open(cfg_path))
            modalities = cfg["dataset"]["modalities"]

            ds_key = json.dumps(cfg["dataset"], sort_keys=True)
            if ds_key not in loader_cache:
                tr = DataLoader(get_dataset(cfg, "train"), batch_size=256, shuffle=False, num_workers=0)
                te = DataLoader(get_dataset(cfg, "test"), batch_size=256, shuffle=False, num_workers=0)
                loader_cache[ds_key] = (tr, te)
            train_loader, test_loader = loader_cache[ds_key]

            # Saved config.yaml may carry stale template dims; the dataset is
            # the source of truth (mirrors train.py's runtime override).
            train_ds = train_loader.dataset
            if hasattr(train_ds, "text_dim"):
                cfg["dataset"]["text_dim"] = train_ds.text_dim
                cfg["dataset"]["audio_dim"] = train_ds.audio_dim
                cfg["dataset"]["visual_dim"] = train_ds.visual_dim
            elif hasattr(train_ds, "get_dims"):
                for k, v in train_ds.get_dims().items():
                    cfg["dataset"][f"{k}_dim"] = v

            model = build_model(cfg, modalities)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            train_feats, train_labels = extract_features(model, train_loader, modalities)
            test_feats, test_labels = extract_features(model, test_loader, modalities)

            per_mod = {}
            for m in modalities:
                per_mod[m] = fit_linear_probe(
                    train_feats[m], train_labels, test_feats[m], test_labels,
                    num_classes=cfg["dataset"]["num_classes"])
                print(f"  {method} seed{seed} probe[{m}]: {per_mod[m]:.4f}")
            vals = [per_mod[m] for m in modalities]
            per_mod["gap"] = max(vals) - min(vals)
            results[method][f"seed{seed}"] = per_mod
            print(f"  {method} seed{seed} gap: {per_mod['gap']:.4f}")
            del model, train_feats, test_feats
            torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")

    # Summary: per-method mean per-modality accuracy and gap
    import statistics
    print("\n=== Summary (mean over seeds) ===")
    for method, per_seed in results.items():
        if not per_seed:
            continue
        keys = [k for k in next(iter(per_seed.values())) if k != "gap"]
        means = {k: statistics.mean(v[k] for v in per_seed.values()) for k in keys + ["gap"]}
        print(f"{method:26s} " + "  ".join(f"{k}={means[k]*100:.2f}" for k in keys)
              + f"  gap={means['gap']*100:.2f}")


if __name__ == "__main__":
    main()
