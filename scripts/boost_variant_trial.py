#!/usr/bin/env python
"""Trial sweep: alternative boost-formula variants for §5 future-work discussion.

This script does NOT modify existing source. It monkey-patches
`ASGMLScheduler.get_continuous_scales` at runtime to one of several variant
forms before invoking the training pipeline. Each variant is run on
CREMA-D 3-frame seed 42 only as a single-seed scoping study.

Variants (all preserve the s_m ∈ [1, s_max] property):
  linear     — current paper formula: s_m = 1 + α * w_m  (reference)
  tanh       — smooth saturation: s_m = 1 + (s_max-1) * tanh(α*w_m / (s_max-1))
  squared    — emphasize large gaps: s_m = 1 + α * w_m^2
  sigmoid    — soft threshold:        s_m = 1 + (s_max-1) * σ(8*(w_m - 0.5))
  adaptive   — curriculum α(t):       α(t) = α_max - (α_max-α_min) * t/T

Usage:
  python scripts/boost_variant_trial.py --variant tanh --seed 42
"""
import argparse, math, sys, os
from pathlib import Path

ROOT = Path("/home/main/AsynchronousFunction")
sys.path.insert(0, str(ROOT))

# Monkey-patch BEFORE importing train.py
from src.losses.asgml import ASGMLScheduler

_ORIG = ASGMLScheduler.get_continuous_scales

def make_variant(variant: str, max_epochs: int = 100):
    """Return a function with same signature as get_continuous_scales."""

    def _scales(self, utilization_scores, alpha=0.5, scale_max=2.0):
        if not utilization_scores:
            return {m: 1.0 for m in self.modalities}
        max_s = max(utilization_scores.values())
        min_s = min(utilization_scores.values())
        gap = max_s - min_s

        # Adaptive alpha schedule (linear decay from 1.5x to 0.5x base)
        eff_alpha = alpha
        if variant == "adaptive":
            t = getattr(self, "current_step", 0)
            # Approx 105 iters/epoch on CREMA-D 3f → 10500 iters in 100 epochs
            T = 10500
            frac = min(t / T, 1.0)
            eff_alpha = alpha * (1.5 - 1.0 * frac)  # 1.5x → 0.5x

        scales = {}
        for m in self.modalities:
            if gap < 1e-8:
                scales[m] = 1.0
                continue
            w = 1.0 - (utilization_scores[m] - min_s) / gap

            if variant == "linear":
                raw = 1.0 + eff_alpha * w
            elif variant == "tanh":
                raw = 1.0 + (scale_max - 1.0) * math.tanh(eff_alpha * w / (scale_max - 1.0))
            elif variant == "squared":
                raw = 1.0 + eff_alpha * (w ** 2)
            elif variant == "sigmoid":
                raw = 1.0 + (scale_max - 1.0) / (1.0 + math.exp(-8.0 * (w - 0.5)))
            elif variant == "adaptive":
                raw = 1.0 + eff_alpha * w
            else:
                raise ValueError(f"unknown variant: {variant}")
            scales[m] = min(raw, scale_max)
        return scales

    return _scales


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=["linear", "tanh", "squared", "sigmoid", "adaptive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="configs/cremad.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--exp-name-prefix", default="trial_boost")
    p.add_argument("--output-dir", default="outputs/sweep_boost_variants")
    args = p.parse_args()

    # Apply variant patch
    ASGMLScheduler.get_continuous_scales = make_variant(args.variant)

    # Build train.py argv and invoke
    exp_name = f"{args.exp_name_prefix}_{args.variant}_seed{args.seed}"
    out_path = Path(args.output_dir) / exp_name
    if (out_path / "train.log").exists():
        with open(out_path / "train.log") as fh:
            if "Training complete" in fh.read():
                print(f"[skip] {exp_name} already complete")
                return

    train_argv = [
        "scripts/train.py",
        "--config", args.config,
        "--mode", "adaptive",
        "--asgml-mode", "continuous",
        "--ogm-ge", "--alpha", "0.8",
        "--continuous-alpha", "0.75",
        "--num-frames", "3", "--fps", "3",
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--exp-name", exp_name,
        "--output-dir", args.output_dir,
    ]

    sys.argv = train_argv
    print(f"[variant={args.variant}] launching {exp_name}")
    # Import and run train.py as if invoked directly.
    import runpy
    runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
