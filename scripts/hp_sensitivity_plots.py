#!/usr/bin/env python
"""Generate hyperparameter sensitivity plots for the appendix.

Single-seed sweeps on CREMA-D 3-frame for boost+OGM-GE:
- alpha (boost strength) in {0.25, 0.5, 1.0, 1.5}
- K (probe evaluation frequency) in {1, 5, 10, 20, 50, 100}
- s_max (boost cap) in {1.5, 2.0, 3.0}

Outputs: Manuscript/figures/hp/*.pdf
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})


def best_acc(log_path):
    """Return best accuracy (%) from a train.log file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Training complete\. Best accuracy: ([\d.]+)", line)
            if m:
                return float(m.group(1)) * 100
    return None


def collect_alpha():
    values = [0.25, 0.5, 1.0, 1.5]
    accs = []
    for a in values:
        p = f"outputs/sweep_hp/3f_hp_alpha{a}/train.log"
        v = best_acc(p)
        accs.append(v if v is not None else np.nan)
    return values, accs


def collect_smax():
    values = [1.5, 2.0, 3.0]
    accs = []
    for s in values:
        if s == 2.0:
            # 2.0 is the default, pull from alpha1.0 run (default s_max) or main sweep
            p = "outputs/sweep_3f/3f_boost_ogm_a075_seed42/train.log"
        else:
            p = f"outputs/sweep_hp/3f_hp_smax{s}/train.log"
        v = best_acc(p)
        accs.append(v if v is not None else np.nan)
    return values, accs


def collect_k():
    values = [1, 5, 10, 20, 50, 100]
    accs = []
    for k in values:
        if k == 20:
            # Default K=20 is the main sweep run (Table 1 boost+OGM-GE seed 42)
            p = "outputs/sweep_3f/3f_boost_ogm_a075_seed42/train.log"
        else:
            p = f"outputs/sweep_k_ablation/3f_boost_ogm_K{k}_seed42/train.log"
        v = best_acc(p)
        accs.append(v if v is not None else np.nan)
    return values, accs


def plot_alpha(out_path):
    xs, ys = collect_alpha()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(xs, ys, "o-", color="#4C72B0", linewidth=1.8, markersize=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"Boost strength $\alpha$")
    ax.set_ylabel("Accuracy (\%)".replace(r"\%", "%"))
    ax.set_xticks(xs)
    ax.set_ylim(66, 72)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, "CREMA-D 3f, single seed",
            transform=ax.transAxes, fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_k(out_path):
    xs, ys = collect_k()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.semilogx(xs, ys, "o-", color="#DD8452", linewidth=1.8, markersize=7)
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"Probe eval interval $K$ (iterations)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(66, 72)
    ax.grid(True, alpha=0.3, which="both")
    ax.text(0.02, 0.02, "CREMA-D 3f, single seed",
            transform=ax.transAxes, fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_smax(out_path):
    xs, ys = collect_smax()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(xs, ys, "o-", color="#2E7D32", linewidth=1.8, markersize=7)
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"Boost cap $s_{\max}$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(xs)
    ax.set_ylim(66, 72)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, "CREMA-D 3f, single seed",
            transform=ax.transAxes, fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_dir = "Manuscript/figures/hp"
    os.makedirs(out_dir, exist_ok=True)
    plot_alpha(f"{out_dir}/fig_hp_alpha.pdf")
    plot_k(f"{out_dir}/fig_hp_K.pdf")
    plot_smax(f"{out_dir}/fig_hp_smax.pdf")
    print("\nAll HP figures generated.")
    print("\nAlpha sweep values:", collect_alpha())
    print("K sweep values:", collect_k())
    print("s_max sweep values:", collect_smax())
