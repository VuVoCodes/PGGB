#!/usr/bin/env python
"""Extract per-modality probe accuracy trajectories from tensorboard events
for {baseline, OGM-GE, Boost+OGM-GE} on CREMA-D 3-frame and produce
appendix figure with 2 panels (audio + video).
"""
from pathlib import Path
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

OUT = Path("/home/main/AsynchronousFunction/Manuscript/figures/probe_trajectories.pdf")

# 5-seed runs for the three conditions
RUNS = {
    "Baseline":      sorted(glob.glob("/home/main/AsynchronousFunction/outputs/sweep_3f/3f_baseline_seed*/tensorboard/events*")),
    "OGM-GE":        sorted(glob.glob("/home/main/AsynchronousFunction/outputs/sweep_3f/3f_ogm_ge_seed*/tensorboard/events*")),
    "Boost+OGM-GE":  sorted(glob.glob("/home/main/AsynchronousFunction/outputs/sweep_3f/3f_boost_ogm_a075_seed*/tensorboard/events*")),
}

COLORS = {"Baseline": "#888888", "OGM-GE": "#4673AA", "Boost+OGM-GE": "#3C8C5A"}
LSTYLE = {"Baseline": ":",       "OGM-GE": "--",      "Boost+OGM-GE": "-"}

def load_scalar(path, tag):
    ea = event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None, None
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])

def aggregate(paths, tag):
    """Mean ± std across seeds. Returns (steps, mean, std). Steps from first run."""
    series = []
    ref_steps = None
    for p in paths:
        steps, vals = load_scalar(p, tag)
        if steps is None or len(vals) == 0: continue
        if ref_steps is None: ref_steps = steps
        # Truncate or pad to match reference length
        n = min(len(ref_steps), len(vals))
        series.append(vals[:n])
    if not series: return None, None, None
    n = min(len(s) for s in series)
    arr = np.stack([s[:n] for s in series])
    return ref_steps[:n], arr.mean(axis=0), arr.std(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)

for ax, modality, ylabel in zip(axes, ["audio", "visual"],
                                 ["Audio (strong) probe acc", "Visual (weak) probe acc"]):
    tag = f"probe/accuracy_{modality}"
    for name, paths in RUNS.items():
        steps, mean, std = aggregate(paths, tag)
        if steps is None:
            print(f"[skip] {name} / {tag}: no data")
            continue
        # Convert step → epoch (probe eval every K=20 iters; ~105 iters/epoch on CREMA-D 3f)
        epochs = steps / 105
        ax.plot(epochs, mean*100, label=name, color=COLORS[name],
                linestyle=LSTYLE[name], linewidth=1.6)
        ax.fill_between(epochs, (mean-std)*100, (mean+std)*100,
                         color=COLORS[name], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.4)

axes[0].legend(loc="lower right", fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}")
