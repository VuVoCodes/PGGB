#!/usr/bin/env python
"""Aggregate sweep results into summary table with mean +/- std.

Usage:
    python scripts/aggregate_sweep.py --sweep-dir outputs/sweep --output outputs/sweep/sweep_summary.md
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np


def parse_train_log(log_path: Path) -> dict:
    """Extract best accuracy, F1, epoch, and final util gap from train.log."""
    text = log_path.read_text()

    best_acc = 0.0
    match = re.search(r"Best accuracy: ([0-9.]+)", text)
    if match:
        best_acc = float(match.group(1))

    # Find the epoch with best test accuracy and its metrics
    best_f1 = 0.0
    best_epoch = 0
    final_util_gap = 0.0

    for line in text.strip().split("\n"):
        m = re.search(
            r"Epoch (\d+):.*Test Acc=([0-9.]+).*Test F1=([0-9.]+).*Util Gap=([0-9.]+)",
            line,
        )
        if m:
            epoch = int(m.group(1))
            acc = float(m.group(2))
            f1 = float(m.group(3))
            util_gap = float(m.group(4))

            if abs(acc - best_acc) < 0.0002:
                best_f1 = f1
                best_epoch = epoch

            final_util_gap = util_gap

    return {
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "final_util_gap": final_util_gap,
    }


def parse_config(config_path: Path) -> dict:
    """Extract hyperparams from saved config.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    asgml = config.get("asgml", {})
    exp = config.get("experiment", {})

    return {
        "mode": exp.get("mode", "unknown"),
        "seed": exp.get("seed", -1),
        "gamma": asgml.get("gamma", 1.0),
        "tau_base": asgml.get("tau_base", 2.0),
        "tau_max": asgml.get("tau_max", 8.0),
        "beta": asgml.get("beta", 0.5),
        "lambda_comp": asgml.get("lambda_comp", 0.1),
        "threshold_delta": asgml.get("threshold_delta", 0.1),
        "signal_source": asgml.get("signal_source", "dual"),
        "soft_mask_scale": asgml.get("soft_mask_scale", 0.1),
        "ogm_ge": "OGM-GE enabled" in (config_path.parent / "train.log").read_text()
        if (config_path.parent / "train.log").exists()
        else False,
    }


def config_key(config: dict) -> str:
    """Generate a hashable key from config (excluding seed)."""
    ogm = "+OGM" if config.get("ogm_ge") else ""
    return (
        f"{config['mode']}{ogm}|"
        f"g={config['gamma']}|"
        f"tb={config['tau_base']}|"
        f"b={config['beta']}|"
        f"td={config['threshold_delta']}|"
        f"ss={config['signal_source']}|"
        f"sms={config['soft_mask_scale']}"
    )


def config_label(config: dict) -> str:
    """Human-readable label showing only non-default params."""
    defaults = {
        "gamma": 1.0, "tau_base": 2.0, "beta": 0.5,
        "threshold_delta": 0.1, "signal_source": "dual",
        "soft_mask_scale": 0.1,
    }
    diffs = []
    for k, default_v in defaults.items():
        if config.get(k) != default_v:
            diffs.append(f"{k}={config[k]}")

    mode_str = config["mode"]
    if config.get("ogm_ge"):
        mode_str += "+OGM-GE"

    if diffs:
        return f"{mode_str} ({', '.join(diffs)})"
    return mode_str


def main():
    parser = argparse.ArgumentParser(description="Aggregate sweep results")
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    grouped = defaultdict(list)
    skipped = []

    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        log_path = run_dir / "train.log"
        config_path = run_dir / "config.yaml"

        if not log_path.exists() or not config_path.exists():
            continue

        if "Training complete" not in log_path.read_text():
            skipped.append(run_dir.name)
            continue

        metrics = parse_train_log(log_path)
        config = parse_config(config_path)

        key = config_key(config)
        grouped[key].append({**metrics, **config, "dir": run_dir.name})

    # Build rows
    rows = []
    for key, results in grouped.items():
        accs = [r["best_acc"] for r in results]
        f1s = [r["best_f1"] for r in results]
        gaps = [r["final_util_gap"] for r in results]

        rows.append({
            "key": key,
            "label": config_label(results[0]),
            "n_seeds": len(results),
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs),
            "mean_f1": np.mean(f1s),
            "std_f1": np.std(f1s),
            "mean_gap": np.mean(gaps),
            "config": results[0],
            "all_accs": accs,
        })

    rows.sort(key=lambda r: r["mean_acc"], reverse=True)

    # Write markdown
    lines = [
        "# ASGML Sweep Results Summary\n",
        f"Total configurations: {len(rows)}",
        f"Skipped (incomplete): {len(skipped)}\n",
        "## Results Table\n",
        "| Rank | Configuration | N | Acc (mean +/- std) | F1 (mean +/- std) | Util Gap | Individual Accs |",
        "|------|--------------|---|-------------------|-------------------|----------|-----------------|",
    ]

    for i, row in enumerate(rows, 1):
        acc_str = f"{row['mean_acc']*100:.2f} +/- {row['std_acc']*100:.2f}"
        f1_str = f"{row['mean_f1']*100:.2f} +/- {row['std_f1']*100:.2f}"
        indiv = ", ".join(f"{a*100:.1f}" for a in row["all_accs"])
        lines.append(
            f"| {i} | {row['label']} | {row['n_seeds']} | "
            f"{acc_str} | {f1_str} | {row['mean_gap']:.4f} | {indiv} |"
        )

    if skipped:
        lines.append(f"\n## Incomplete Runs\n")
        for s in skipped:
            lines.append(f"- {s}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {output_path}")
    print(f"\nTop 5 configurations:")
    for i, row in enumerate(rows[:5], 1):
        print(f"  {i}. {row['label']}: {row['mean_acc']*100:.2f}% +/- {row['std_acc']*100:.2f}% (n={row['n_seeds']})")


if __name__ == "__main__":
    main()
