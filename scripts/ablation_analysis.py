#!/usr/bin/env python
"""
Ablation study analysis: statistical tests + visualizations.

Generates:
- Stat tests (paired t-test, Wilcoxon) for key contrasts
- Bar plot: method performance across operating conditions
- Line plot: probe gap trajectories during training
- Line plot: boost scale trajectories
- Accuracy vs imbalance scatter (one point per dataset/condition)

Output: docs/experiments/figures/ablation/*.png + stats table
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# Data extraction
# =============================================================================

def get_best_acc(log_path):
    """Extract final best accuracy from train.log."""
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r'Training complete\. Best accuracy: ([\d.]+)', line)
                if m:
                    return float(m.group(1)) * 100
    except Exception:
        pass
    return None


def get_util_gaps(log_path):
    """Extract util gap over time (list of (epoch, gap))."""
    gaps = []
    if not os.path.exists(log_path):
        return gaps
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Epoch (\d+):.*Util Gap=([\d.]+)', line)
            if m:
                gaps.append((int(m.group(1)), float(m.group(2))))
    return gaps


def get_accuracies(log_path):
    """Extract test accuracy per epoch."""
    accs = []
    if not os.path.exists(log_path):
        return accs
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Epoch (\d+):.*Test Acc=([\d.]+)', line)
            if m:
                accs.append((int(m.group(1)), float(m.group(2)) * 100))
    return accs


def collect_seeds(base_dir, prefix, method, seeds):
    """Collect per-seed best accuracy values."""
    results = []
    for s in seeds:
        p = f"{base_dir}/{prefix}_{method}_seed{s}/train.log"
        acc = get_best_acc(p)
        if acc is not None:
            results.append(acc)
    return results


# =============================================================================
# Statistical tests
# =============================================================================

def paired_test(x, y, name_x="X", name_y="Y"):
    """Run paired Wilcoxon and t-test."""
    x = np.array(x)
    y = np.array(y)
    n = min(len(x), len(y))
    if n < 2:
        return None
    x, y = x[:n], y[:n]

    diff = x - y
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1) if n > 1 else 0

    try:
        t_stat, t_p = stats.ttest_rel(x, y)
    except Exception:
        t_stat, t_p = np.nan, np.nan
    try:
        w_stat, w_p = stats.wilcoxon(x, y)
    except Exception:
        w_stat, w_p = np.nan, np.nan

    cohen_d = mean_diff / std_diff if std_diff > 0 else np.nan

    return {
        "name_x": name_x, "name_y": name_y,
        "n": n, "mean_x": x.mean(), "std_x": x.std(),
        "mean_y": y.mean(), "std_y": y.std(),
        "mean_diff": mean_diff, "cohen_d": cohen_d,
        "t_stat": t_stat, "t_p": t_p,
        "w_stat": w_stat, "w_p": w_p,
    }


# =============================================================================
# Visualizations
# =============================================================================

def plot_operating_conditions(results, output_path):
    """Bar plot: per-dataset baseline vs our method."""
    datasets = list(results.keys())
    labels = []
    baselines = []
    ours = []
    for ds in datasets:
        if "baseline" in results[ds] and "ours" in results[ds]:
            labels.append(ds)
            baselines.append(results[ds]["baseline"][0])
            ours.append(results[ds]["ours"][0])

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, baselines, w, label="Baseline", color="#b0b0b0")
    ax.bar(x + w/2, ours, w, label="Our method (Boost+OGM-GE)", color="#2a7fff")

    for i, (b, o) in enumerate(zip(baselines, ours)):
        diff = o - b
        color = "green" if diff > 0.1 else ("red" if diff < -0.5 else "gray")
        ax.text(i, max(b, o) + 1, f"{diff:+.1f}pp", ha="center", color=color, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best accuracy (%)")
    ax.set_title("Operating Conditions Ablation: Method Performance by Setup")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved: {output_path}")


def plot_util_gap_trajectories(log_paths, labels, output_path):
    """Line plot: probe util gap over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for path, label in zip(log_paths, labels):
        gaps = get_util_gaps(path)
        if not gaps:
            continue
        epochs, vals = zip(*gaps)
        ax.plot(epochs, vals, label=label, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Probe Utilization Gap")
    ax.set_title("Modality Imbalance Detected During Training (Probe Gap)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_trajectories(log_paths, labels, output_path, title="Test Accuracy Over Epochs"):
    """Line plot: test accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for path, label in zip(log_paths, labels):
        accs = get_accuracies(path)
        if not accs:
            continue
        epochs, vals = zip(*accs)
        ax.plot(epochs, vals, label=label, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    out_dir = Path("docs/experiments/figures/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_main = [42, 123, 456, 789, 1024]

    # ========== Data collection ==========
    print("=" * 60)
    print("ABLATION RESULTS SUMMARY (from logs)")
    print("=" * 60)

    collections = {
        "CREMA-D 3f": {
            "baseline": collect_seeds("outputs/sweep_3f", "3f", "baseline", seeds_main),
            "boost_only": collect_seeds("outputs/sweep_3f", "3f", "boost_only", seeds_main),
            "boost_ogm_a075": collect_seeds("outputs/sweep_3f", "3f", "boost_ogm_a075", seeds_main),
        },
        "AVE Pretrained (Table 1)": {
            "baseline": collect_seeds("outputs/sweep_ave", "ave", "baseline", seeds_main),
            "boost_only": collect_seeds("outputs/sweep_ave", "ave", "boost_only", seeds_main),
            "boost_ogm_a075": collect_seeds("outputs/sweep_ave", "ave", "boost_ogm_a075", seeds_main),
        },
        "AVE From-Scratch (Ablation)": {
            "baseline": collect_seeds("outputs/sweep_ave_scratch", "ave_scratch", "baseline", seeds_main),
            "boost_only": collect_seeds("outputs/sweep_ave_scratch", "ave_scratch", "boost_only", seeds_main),
            "boost_ogm_a075": collect_seeds("outputs/sweep_ave_scratch", "ave_scratch", "boost_ogm_a075", seeds_main),
        },
        "Food101 Frozen": {
            "baseline": collect_seeds("outputs/sweep_food101", "food101", "baseline", seeds_main),
            "boost_only": collect_seeds("outputs/sweep_food101", "food101", "boost_only", seeds_main),
            "boost_ogm_a075": collect_seeds("outputs/sweep_food101", "food101", "boost_ogm_a075", seeds_main),
        },
    }

    for ds, methods in collections.items():
        print(f"\n{ds}:")
        for m, vals in methods.items():
            if vals:
                print(f"  {m:22s}: {np.mean(vals):6.2f} ± {np.std(vals):.2f}  (n={len(vals)})  {[f'{v:.2f}' for v in vals]}")
            else:
                print(f"  {m:22s}: no data")

    # ========== Statistical tests ==========
    print("\n" + "=" * 60)
    print("PAIRED STATISTICAL TESTS")
    print("=" * 60)

    tests = [
        ("CREMA-D 3f", "boost_ogm_a075", "baseline"),
        ("CREMA-D 3f", "boost_only", "baseline"),
        ("AVE Pretrained (Table 1)", "boost_only", "baseline"),
        ("AVE From-Scratch (Ablation)", "boost_ogm_a075", "baseline"),
        ("Food101 Frozen", "boost_only", "baseline"),
    ]
    results_rows = []
    for ds, method, reference in tests:
        x = collections[ds].get(method, [])
        y = collections[ds].get(reference, [])
        if len(x) >= 2 and len(y) >= 2:
            r = paired_test(x, y, f"{ds} / {method}", f"{ds} / {reference}")
            print(f"\n{ds}: {method} vs {reference}")
            print(f"  n={r['n']}, mean diff = {r['mean_diff']:+.3f}pp, Cohen d = {r['cohen_d']:.2f}")
            print(f"  t-test p = {r['t_p']:.4f}, Wilcoxon p = {r['w_p']:.4f}")
            results_rows.append(r)
        else:
            print(f"\n{ds}: {method} vs {reference} - insufficient data (n_x={len(x)}, n_y={len(y)})")

    # ========== Visualizations ==========
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Bar plot: operating conditions summary
    plot_data = {}
    for ds, methods in collections.items():
        b = methods.get("baseline", [])
        o = methods.get("boost_ogm_a075", [])
        if b and o:
            plot_data[ds] = {
                "baseline": (np.mean(b), np.std(b), len(b)),
                "ours": (np.mean(o), np.std(o), len(o)),
            }

    if plot_data:
        plot_operating_conditions(plot_data, out_dir / "operating_conditions.png")

    # Util gap trajectories (use seed 42 from key conditions)
    gap_runs = [
        ("outputs/sweep_3f/3f_boost_ogm_a075_seed42/train.log", "CREMA-D 3f (win)"),
        ("outputs/sweep_ave/ave_boost_ogm_a075_seed42/train.log", "AVE Pretrained (small win)"),
        ("outputs/sweep_ave_scratch/ave_scratch_boost_ogm_seed42/train.log", "AVE Scratch (hurts)"),
        ("outputs/sweep_food101/food101_boost_ogm_a075_seed42/train.log", "Food101 Frozen (tie)"),
    ]
    paths = [r[0] for r in gap_runs]
    labels = [r[1] for r in gap_runs]
    plot_util_gap_trajectories(paths, labels, out_dir / "util_gap_trajectories.png")

    # Accuracy trajectories
    plot_accuracy_trajectories(paths, labels, out_dir / "accuracy_trajectories.png",
                                title="Test Accuracy Over Epochs (Boost+OGM-GE, seed 42)")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
