#!/usr/bin/env python
"""Early-training probe trajectory figure (Reviewer tQk1, W2).

Reads the instrumented CREMA-D 3-frame run (PGGB+OGM-GE, alpha=0.75, seed 42),
which logs the EMA-smoothed per-modality probe accuracy and the signed gap at
every probe evaluation (K=20 iterations, 500 events over 100 epochs).
"""
from pathlib import Path
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

RUN = "outputs/probe_traj/probe_traj_seed42/tensorboard"
OUT = Path("Manuscript_NEURIPS_CAMREADY/figures/probe_trajectory_early.pdf")
ITERS_PER_EPOCH = 105
EARLY_EVENTS = 15  # first three epochs

ea = event_accumulator.EventAccumulator(sorted(glob.glob(f"{RUN}/events*"))[0],
                                        size_guidance={"scalars": 0})
ea.Reload()
step = np.array([e.step for e in ea.Scalars("probe/ema_audio")])
audio = np.array([e.value * 100 for e in ea.Scalars("probe/ema_audio")])
visual = np.array([e.value * 100 for e in ea.Scalars("probe/ema_visual")])
gap = np.array([e.value * 100 for e in ea.Scalars("probe/ema_signed_gap")])
epoch = step / ITERS_PER_EPOCH

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

# (a) early window
n = EARLY_EVENTS
ev = np.arange(1, n + 1)
ax1.plot(ev, audio[:n], "o-", color="#4673AA", label=r"audio $\bar{P}_a$", ms=3.5, lw=1.4)
ax1.plot(ev, visual[:n], "s-", color="#C4772F", label=r"visual $\bar{P}_v$", ms=3.5, lw=1.4)
ax1.axvline(1, color="#999999", lw=0.8, ls=":")
ax1.annotate("ordering inverted\n(both at chance)", xy=(1, 1.2), xytext=(2.4, 9.0),
             fontsize=7, color="#666666",
             arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
ax1.set_xlabel("probe evaluation event (every $K{=}20$ iterations)")
ax1.set_ylabel(r"EMA probe accuracy (\%)" if False else "EMA probe accuracy (%)")
ax1.set_title("(a) first three epochs", fontsize=9)
ax1.legend(frameon=False, loc="upper left")
ax1.grid(alpha=0.25, lw=0.5)

# (b) full run
ax2.plot(epoch, audio, color="#4673AA", lw=1.2, label="audio")
ax2.plot(epoch, visual, color="#C4772F", lw=1.2, label="visual")
ax2.fill_between(epoch, visual, audio, color="#4673AA", alpha=0.12,
                 label="utilization gap $\\delta$")
ax2.set_xlabel("epoch")
ax2.set_ylabel("EMA probe accuracy (%)")
ax2.set_title("(b) full run, 500 probe events", fontsize=9)
ax2.legend(frameon=False, loc="lower right")
ax2.grid(alpha=0.25, lw=0.5)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"inversions: {(gap < 0).sum()} of {len(gap)}; first-event gap {gap[0]:+.2f} pp; final gap {gap[-1]:+.2f} pp")
