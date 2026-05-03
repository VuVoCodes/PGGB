#!/usr/bin/env python
"""Generate publication-quality figures for the ablation section.

Outputs:
- figure_ablation_ave_contrast.png  : AVE pretrained vs scratch with error bars + significance
- figure_ablation_conditions.png    : Operating conditions across 4 regimes
- figure_ablation_trajectories.png  : Probe gap + accuracy over epochs (2-panel)
"""
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})


def get_best(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Training complete\. Best accuracy: ([\d.]+)", line)
            if m:
                return float(m.group(1)) * 100
    return None


def get_util_gaps(log_path):
    if not os.path.exists(log_path):
        return []
    out = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Epoch (\d+):.*Util Gap=([\d.]+)", line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def get_acc_series(log_path):
    if not os.path.exists(log_path):
        return []
    out = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Epoch (\d+):.*Test Acc=([\d.]+)", line)
            if m:
                out.append((int(m.group(1)), float(m.group(2)) * 100))
    return out


def collect(base, prefix, method, seeds=(42, 123, 456, 789, 1024)):
    vals = []
    for s in seeds:
        # Try both naming conventions
        for alt in [method, method.replace("_a075", "")]:
            p = f"{base}/{prefix}_{alt}_seed{s}/train.log"
            v = get_best(p)
            if v is not None:
                vals.append(v)
                break
    return vals


# =============================================================================
# FIGURE 1: AVE Pretrained vs Scratch Contrast
# =============================================================================

def fig_ave_contrast(out_path):
    # Collect data
    pretrained = {
        "Baseline": collect("outputs/sweep_ave", "ave", "baseline"),
        "Boost-only": collect("outputs/sweep_ave", "ave", "boost_only"),
        "Boost+OGM-GE": collect("outputs/sweep_ave", "ave", "boost_ogm_a075"),
    }
    scratch = {
        "Baseline": collect("outputs/sweep_ave_scratch", "ave_scratch", "baseline"),
        "Boost-only": collect("outputs/sweep_ave_scratch", "ave_scratch", "boost_only"),
        "Boost+OGM-GE": collect("outputs/sweep_ave_scratch", "ave_scratch", "boost_ogm_a075"),
    }

    labels = list(pretrained.keys())
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    pre_means = [np.mean(pretrained[m]) for m in labels]
    pre_stds = [np.std(pretrained[m]) for m in labels]
    scr_means = [np.mean(scratch[m]) for m in labels]
    scr_stds = [np.std(scratch[m]) for m in labels]

    b1 = ax.bar(x - w/2, pre_means, w, yerr=pre_stds, capsize=2.5,
                label="Pretrained", color="#4C72B0", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, scr_means, w, yerr=scr_stds, capsize=2.5,
                label="From-scratch", color="#DD8452", edgecolor="black", linewidth=0.5)

    for bars, means in [(b1, pre_means), (b2, scr_means)]:
        for rect, m in zip(bars, means):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.0,
                    f"{m:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(55, 95)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# FIGURE 2: Operating Conditions (Method Δ across 4 regimes)
# =============================================================================

def fig_operating_conditions(out_path):
    conditions = [
        ("CREMA-D", "outputs/sweep_3f", "3f"),
        ("AVE (pretrained)", "outputs/sweep_ave", "ave"),
        ("AVE (scratch)", "outputs/sweep_ave_scratch", "ave_scratch"),
        ("Food101 (frozen)", "outputs/sweep_food101", "food101"),
    ]

    data = []
    for name, base, prefix in conditions:
        b = collect(base, prefix, "baseline")
        bog = collect(base, prefix, "boost_ogm_a075")
        if b and bog:
            diff = np.mean(bog) - np.mean(b)
            n = min(len(b), len(bog))
            try:
                _, p = stats.ttest_rel(bog[:n], b[:n])
            except Exception:
                p = np.nan
            data.append((name, diff, p))

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    names = [d[0] for d in data]
    diffs = [d[1] for d in data]
    colors = ["#2E7D32" if d > 1 else ("#D32F2F" if d < -1 else "#9E9E9E") for d in diffs]

    ax.barh(names, diffs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)

    for i, (_, diff, p) in enumerate(data):
        mark = ""
        if not np.isnan(p):
            if p < 0.001: mark = "***"
            elif p < 0.01: mark = "**"
            elif p < 0.05: mark = "*"
        x_pos = diff + (0.25 if diff > 0 else -0.25)
        ax.text(x_pos, i, f"{diff:+.2f}{mark}", va="center",
                ha="left" if diff > 0 else "right", fontsize=9, fontweight="bold")

    ax.set_xlabel(r"$\Delta$ accuracy (pp): Boost+OGM-GE $-$ Baseline", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(-8, 12)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# FIGURE 3: Probe Gap + Accuracy Trajectories (2-panel)
# =============================================================================

def fig_trajectories(out_path):
    runs = [
        ("outputs/sweep_3f/3f_boost_ogm_a075_seed42/train.log", "CREMA-D 3f (wins +9.86pp)", "#2E7D32"),
        ("outputs/sweep_ave/ave_boost_ogm_a075_seed42/train.log", "AVE pretrained (+0.69pp)", "#4C72B0"),
        ("outputs/sweep_ave_scratch/ave_scratch_boost_ogm_seed42/train.log", "AVE scratch (-5.06pp)", "#DD8452"),
        ("outputs/sweep_food101/food101_boost_ogm_a075_seed42/train.log", "Food101 frozen (-1.64pp)", "#9467BD"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for path, label, color in runs:
        gaps = get_util_gaps(path)
        accs = get_acc_series(path)
        if gaps:
            e, v = zip(*gaps)
            ax1.plot(e, v, label=label, color=color, alpha=0.85, linewidth=1.5)
        if accs:
            e, v = zip(*accs)
            ax2.plot(e, v, label=label, color=color, alpha=0.85, linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Probe Utilization Gap $\\delta$")
    ax1.set_title("(a) Probe-detected imbalance during training")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("(b) Test accuracy trajectories (Boost+OGM-GE, seed 42)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_dir = "Manuscript/figures/ablation"
    os.makedirs(out_dir, exist_ok=True)
    fig_ave_contrast(f"{out_dir}/fig_ablation_ave_contrast.png")
    fig_operating_conditions(f"{out_dir}/fig_ablation_conditions.png")
    fig_trajectories(f"{out_dir}/fig_ablation_trajectories.png")
    print("\nAll ablation figures regenerated.")
