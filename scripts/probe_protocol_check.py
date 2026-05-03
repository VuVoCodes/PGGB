"""Protocol-matched probe evaluation: train the probe on frozen encoder features
using the SAME per-batch mini-batch schedule as the live-training probe.

Live-training probe protocol (from train.py lines 2079-2094):
  - Every 20 training batches, take a batch of size 64
  - Split into train half (32) and eval half (32)
  - Call probe_manager.train_probes(train_half, 10 steps)
  - Total training events over 100 epochs = ceil(100 * 105 / 20) ~ 525

If the post-hoc result disagrees because of protocol (not feature quality), this
script should reproduce the live numbers. If it agrees with the 50-epoch post-hoc,
then the discrepancy is genuinely about feature quality at final vs mid training.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets import CREMADDataset
from src.models import MultimodalModel, ProbeManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROBE_SEED = 2026
NUM_EVENTS = 525   # matches live probe total events over 100 epochs
NUM_STEPS = 10     # steps per event
TRAIN_HALF = 32    # half of batch 64
BATCH_SIZE = 64


def load_and_extract(run_dir):
    cfg_path = Path(run_dir) / "config.yaml"
    ckpt_path = Path(run_dir) / "best_model.pt"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    modalities = cfg["dataset"]["modalities"]
    num_classes = cfg["dataset"]["num_classes"]
    encoder_config = {m: {"backbone": cfg["model"]["backbone"],
                          "pretrained": cfg["model"].get("pretrained", False)}
                       for m in modalities}
    model = MultimodalModel(
        modalities=modalities, num_classes=num_classes,
        encoder_config=encoder_config,
        fusion_type=cfg["model"]["fusion_type"],
        feature_dim=cfg["model"]["feature_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_ds = CREMADDataset(root=cfg["dataset"]["root"], split="train",
                             fps=cfg["dataset"]["fps"], num_frames=cfg["dataset"]["num_frames"])
    test_ds = CREMADDataset(root=cfg["dataset"]["root"], split="test",
                            fps=cfg["dataset"]["fps"], num_frames=cfg["dataset"]["num_frames"])
    tloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    eloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    def extract(loader):
        feats = {m: [] for m in modalities}
        labs = []
        with torch.no_grad():
            for batch in loader:
                inputs = {m: batch[m].to(DEVICE) for m in modalities}
                _, _, fdict = model(inputs, return_features=True)
                for m in modalities:
                    feats[m].append(fdict[m].float().cpu())
                labs.append(batch["label"])
        return {m: torch.cat(feats[m], 0) for m in modalities}, torch.cat(labs, 0)

    tf, tl = extract(tloader)
    ef, el = extract(eloader)
    return modalities, num_classes, tf, tl, ef, el


def run_live_style_probe(train_feats, train_labels, test_feats, test_labels,
                          modalities, num_classes):
    """Train probe mimicking live-training: 525 events × 10 steps each,
    using random batches of size 64 split into 32/32 train/eval halves."""
    torch.manual_seed(PROBE_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(PROBE_SEED)

    N = train_labels.size(0)
    # Use same ProbeManager config as training
    pm = ProbeManager(
        modalities=modalities,
        feature_dim=train_feats[modalities[0]].shape[1],
        num_classes=num_classes,
        probe_type="linear",
        probe_lr=1e-3,
        device=DEVICE,
        ema_alpha=0.1,
    )

    train_feats_d = {m: v.to(DEVICE) for m, v in train_feats.items()}
    train_labels_d = train_labels.to(DEVICE)

    g = torch.Generator(device=DEVICE)
    g.manual_seed(PROBE_SEED)

    for event in range(NUM_EVENTS):
        idx = torch.randint(0, N, (BATCH_SIZE,), device=DEVICE, generator=g)
        batch_feats = {m: train_feats_d[m][idx] for m in modalities}
        batch_labels = train_labels_d[idx]
        train_half = {m: v[:TRAIN_HALF] for m, v in batch_feats.items()}
        train_half_labels = batch_labels[:TRAIN_HALF]
        pm.train_probes(train_half, train_half_labels, num_steps=NUM_STEPS)

    # Evaluate on test set
    test_feats_d = {m: v.to(DEVICE) for m, v in test_feats.items()}
    test_labels_d = test_labels.to(DEVICE)
    results = pm.evaluate_probes(test_feats_d, test_labels_d)
    return {m: results[m]["accuracy"] for m in modalities}


ROOT = "/home/main/AsynchronousFunction/outputs/sweep_3f"
for method in ["3f_baseline", "3f_boost_only", "3f_boost_ogm_a075"]:
    print(f"\n=== {method} (5 seeds) ===")
    acc_a, acc_v = [], []
    for seed in [42, 123, 456, 789, 1024]:
        run = f"{ROOT}/{method}_seed{seed}"
        modalities, nc, tf, tl, ef, el = load_and_extract(run)
        r = run_live_style_probe(tf, tl, ef, el, modalities, nc)
        acc_a.append(r["audio"] * 100)
        acc_v.append(r["visual"] * 100)
        print(f"  seed{seed}: audio={r['audio']*100:.2f}  visual={r['visual']*100:.2f}  "
              f"gap={abs(r['audio']-r['visual'])*100:.2f}")
        del tf, ef, tl, el
        torch.cuda.empty_cache()
    a = np.array(acc_a); v = np.array(acc_v)
    print(f"  MEAN: audio={a.mean():.2f} +/- {a.std():.2f}  "
          f"visual={v.mean():.2f} +/- {v.std():.2f}  "
          f"gap={np.abs(a-v).mean():.2f} +/- {np.abs(a-v).std():.2f}")
