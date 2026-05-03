#!/usr/bin/env python
"""Generate probe-stability figure + correlation/RMSE table for Appendix B.6.

Reads outputs/probe_stability_val/probe_stability.json
Outputs:
  - Manuscript/figures/probe_stability/fig_probe_stability.pdf  (2-panel)
  - prints Pearson r, Spearman rho, RMSE per modality
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150,
})

path = "outputs/probe_stability_val/probe_stability.json"
data = json.load(open(path))
print(f"Loaded {len(data)} checkpoints from {path}")

iters = np.array([d["iteration"] for d in data])
modalities = list(data[0]["batch_raw"].keys())

series = {}
for m in modalities:
    series[m] = {
        "batch_raw": np.array([d["batch_raw"][m] for d in data]) * 100,
        "ema":       np.array([d["ema_smoothed"][m] for d in data]) * 100,
        "full":      np.array([d["full_test"][m] for d in data]) * 100,
    }

# ========== Stats ==========
print("\n=== Agreement: EMA vs Full-test ===")
stats_rows = []
for m in modalities:
    ema = series[m]["ema"]; full = series[m]["full"]
    pearson_r, _ = stats.pearsonr(ema, full)
    spearman_r, _ = stats.spearmanr(ema, full)
    rmse = float(np.sqrt(np.mean((ema - full) ** 2)))
    mad = float(np.mean(np.abs(ema - full)))
    print(f"  {m:8s}: Pearson r = {pearson_r:.4f}  Spearman rho = {spearman_r:.4f}  RMSE = {rmse:.2f}pp  MAD = {mad:.2f}pp")
    stats_rows.append((m, pearson_r, spearman_r, rmse, mad))

print("\n=== Batch-raw variance vs EMA variance (last 200 checkpoints) ===")
tail_n = 200
for m in modalities:
    br = series[m]["batch_raw"][-tail_n:]; ema = series[m]["ema"][-tail_n:]; full = series[m]["full"][-tail_n:]
    print(f"  {m:8s}: std(batch_raw)={np.std(br):.2f}pp  std(ema)={np.std(ema):.2f}pp  std(full)={np.std(full):.2f}pp")

# ========== Plot ==========
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)

for i, m in enumerate(modalities):
    ax = axes[i]
    ax.plot(iters, series[m]["batch_raw"], color="#CCCCCC", linewidth=0.7,
            alpha=0.7, label=r"Batch raw $P_m$")
    ax.plot(iters, series[m]["full"],      color="#2E7D32", linewidth=1.8,
            label=r"Full-test $P_m^{\star}$ (ground truth)")
    ax.plot(iters, series[m]["ema"],       color="#D32F2F", linewidth=1.6,
            linestyle="--", label=r"EMA-smoothed $\bar{P}_m$ (scheduler signal)")
    ax.set_title(f"({'ab'[i]}) {m.capitalize()} probe")
    ax.set_xlabel("Training iteration")
    if i == 0:
        ax.set_ylabel("Probe accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

plt.tight_layout()
out_dir = "Manuscript/figures/probe_stability"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/fig_probe_stability.pdf"
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"\nSaved figure: {out_path}")

# ========== Summary for LaTeX ==========
print("\n=== LaTeX table rows ===")
for m, pr, sr, rmse, _ in stats_rows:
    name = {"audio": "Audio", "visual": "Visual"}.get(m, m.capitalize())
    print(f"  {name} & {pr:.4f} & {sr:.4f} & {rmse:.2f} \\\\")
