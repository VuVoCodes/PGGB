#!/usr/bin/env python
"""Walk outputs/ and extract task-appropriate metrics from each completed training run.

Per-task metrics:
  - Classification (CREMA-D, AVE, KS, Twitter15, Sarcasm, MOSI, MOSEI):
        Acc (primary) and F1-macro at the best-Acc epoch.
  - Segmentation (BraTS 2021): Dice (overall) and WT/TC/ET sub-Dice at the
        best-Dice epoch. Acc/F1 are not applicable.
  - Note: MOSI / MOSEI are *sentiment regression* tasks but the paper evaluates
        them via binary positive/negative classification, so Acc/F1 is the
        appropriate metric here. Regression-only metrics (MAE, correlation)
        are not logged in train.log and would require re-running inference.

Output: docs/experiments/saved_models_metrics.md
"""
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/main/AsynchronousFunction/outputs")
REPORT = Path("/home/main/AsynchronousFunction/docs/experiments/saved_models_metrics.md")

# Classification: "Epoch 100: Train Loss=0.83, Train Acc=0.99, Test Acc=0.71, Test F1=0.71, ..."
CLS_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+):.*?Test Acc=([\d.]+),\s*Test F1=([\d.]+)"
)
# Segmentation: "Epoch 100: Train Loss=0.44, Val Dice=0.84 (WT=0.87, TC=0.85, ET=0.81) [78.5s]"
SEG_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+):.*?Val Dice=([\d.]+)\s*\(WT=([\d.]+),\s*TC=([\d.]+),\s*ET=([\d.]+)\)"
)
CLS_COMPLETE_RE = re.compile(r"Training complete\.\s*Best accuracy:\s*([\d.]+)")
SEG_COMPLETE_RE = re.compile(r"Training complete\.\s*Best val dice:\s*([\d.]+)")
SEED_RE = re.compile(r"_seed(\d+)$")


def parse_classification(txt):
    if not CLS_COMPLETE_RE.search(txt):
        return None
    best = {"acc": -1.0, "f1": None, "epoch": None}
    for m in CLS_EPOCH_RE.finditer(txt):
        ep, acc, f1 = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if acc > best["acc"]:
            best = {"acc": acc, "f1": f1, "epoch": ep}
    return best if best["f1"] is not None else None


def parse_segmentation(txt):
    if not SEG_COMPLETE_RE.search(txt):
        return None
    best = {"dice": -1.0, "wt": None, "tc": None, "et": None, "epoch": None}
    for m in SEG_EPOCH_RE.finditer(txt):
        ep, dice, wt, tc, et = (
            int(m.group(1)),
            float(m.group(2)),
            float(m.group(3)),
            float(m.group(4)),
            float(m.group(5)),
        )
        if dice > best["dice"]:
            best = {"dice": dice, "wt": wt, "tc": tc, "et": et, "epoch": ep}
    return best if best["wt"] is not None else None


def task_for(sweep, exp_name):
    """Map (sweep, exp_name) → task type."""
    needle = (sweep + " " + exp_name).lower()
    if "brats" in needle:
        return "segmentation"
    return "classification"  # default for all other sweeps


def parse_log(log_path: Path, task: str):
    if not log_path.is_file():
        return None
    txt = log_path.read_text(errors="replace")
    if task == "segmentation":
        return parse_segmentation(txt)
    return parse_classification(txt)


def parse_exp_name(name: str):
    seed_match = SEED_RE.search(name)
    seed = int(seed_match.group(1)) if seed_match else None
    base = name[: seed_match.start()] if seed_match else name
    return base, seed


def fmt_mean_std(vals, decimals=2):
    if len(vals) >= 2:
        return f"{statistics.mean(vals):.{decimals}f} ± {statistics.stdev(vals):.{decimals}f}"
    if vals:
        return f"{vals[0]:.{decimals}f}"
    return "—"


def main():
    cls_rows, seg_rows = [], []
    for log_path in ROOT.rglob("train.log"):
        rel = log_path.parent.relative_to(ROOT)
        if len(rel.parts) == 0:
            continue
        # Exclude archived / debug subtrees that contaminate aggregates with
        # mixed-config runs (e.g., a stale seed=42 in _archived_mixed/).
        if any(p.startswith("_") for p in rel.parts):
            continue
        sweep = rel.parts[0] if len(rel.parts) > 1 else "(top)"
        exp_name = rel.parts[-1]
        base, seed = parse_exp_name(exp_name)
        task = task_for(sweep, exp_name)
        m = parse_log(log_path, task)
        if m is None:
            continue
        ckpt = (log_path.parent / "best_model.pt").is_file()
        record = {
            "sweep": sweep,
            "exp": exp_name,
            "base": base,
            "seed": seed,
            "epoch": m["epoch"],
            "ckpt": ckpt,
        }
        if task == "segmentation":
            record.update(
                {"dice": m["dice"], "wt": m["wt"], "tc": m["tc"], "et": m["et"]}
            )
            seg_rows.append(record)
        else:
            record.update({"acc": m["acc"], "f1": m["f1"]})
            cls_rows.append(record)

    cls_groups = defaultdict(list)
    for r in cls_rows:
        cls_groups[(r["sweep"], r["base"])].append(r)
    seg_groups = defaultdict(list)
    for r in seg_rows:
        seg_groups[(r["sweep"], r["base"])].append(r)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Saved-Model Metrics: Task-Appropriate Evaluation\n\n")
        f.write(
            "Metrics extracted from `train.log` files at the best-primary-metric epoch. "
            "**Tasks differ in primary metric**:\n"
        )
        f.write(
            "- **Classification** (CREMA-D, AVE, Kinetics-Sounds, Twitter15, Sarcasm, MOSI, MOSEI): Test Accuracy (primary) + F1-macro.\n"
        )
        f.write(
            "- **Segmentation** (BraTS 2021): Mean Dice (primary) + WT/TC/ET sub-Dice.\n\n"
        )
        f.write(
            "**MOSI / MOSEI** are sentiment regression tasks evaluated as binary positive/negative "
            "classification per the paper's protocol (matching MMPareto / OGM-GE / CGGM "
            "comparisons). Regression-specific metrics (MAE, Pearson correlation) are not "
            "logged in `train.log` and would require re-running inference.\n\n"
        )
        f.write(
            "**mAP is not retrievable from logs** — per-class probability archives were not "
            "retained across training runs. Computing mAP requires re-running inference on "
            "saved checkpoints (GPU-bound, deferred until the AVE+Food101 sweep finishes).\n\n"
        )
        f.write(
            f"- Total completed runs surveyed: **{len(cls_rows) + len(seg_rows)}** "
            f"({len(cls_rows)} classification, {len(seg_rows)} segmentation)\n"
        )
        f.write(
            f"- Distinct (sweep, method) groups: **{len(cls_groups) + len(seg_groups)}** "
            f"({len(cls_groups)} classification, {len(seg_groups)} segmentation)\n\n"
        )
        f.write(
            "Aggregates report `mean ± std` across seeds (sample std, n−1) when ≥2 seeds; "
            "otherwise the single-seed value.\n\n"
        )

        # ===== Classification tables =====
        f.write("# Classification Tasks (Acc / F1-macro)\n\n")
        for sweep in sorted({s for (s, _) in cls_groups.keys()}):
            f.write(f"## {sweep}\n\n")
            f.write("| Method | N | Acc (%) | F1-macro (%) | Seeds |\n")
            f.write("|---|---|---|---|---|\n")
            sweep_groups = sorted(
                [(b, gs) for ((s, b), gs) in cls_groups.items() if s == sweep],
                key=lambda kv: kv[0],
            )
            for base, gs in sweep_groups:
                accs = [r["acc"] * 100 for r in gs]
                f1s = [r["f1"] * 100 for r in gs]
                seeds = sorted(r["seed"] for r in gs if r["seed"] is not None)
                f.write(
                    f"| {base} | {len(gs)} | {fmt_mean_std(accs)} | {fmt_mean_std(f1s)} | "
                    f"{','.join(map(str, seeds)) or '-'} |\n"
                )
            f.write("\n")

        # ===== Segmentation tables =====
        if seg_rows:
            f.write("# Segmentation Tasks (Dice + WT / TC / ET)\n\n")
            for sweep in sorted({s for (s, _) in seg_groups.keys()}):
                f.write(f"## {sweep}\n\n")
                f.write(
                    "| Method | N | Mean Dice (%) | WT Dice (%) | TC Dice (%) | ET Dice (%) | Seeds |\n"
                )
                f.write("|---|---|---|---|---|---|---|\n")
                sweep_groups = sorted(
                    [(b, gs) for ((s, b), gs) in seg_groups.items() if s == sweep],
                    key=lambda kv: kv[0],
                )
                for base, gs in sweep_groups:
                    dices = [r["dice"] * 100 for r in gs]
                    wts = [r["wt"] * 100 for r in gs]
                    tcs = [r["tc"] * 100 for r in gs]
                    ets = [r["et"] * 100 for r in gs]
                    seeds = sorted(r["seed"] for r in gs if r["seed"] is not None)
                    f.write(
                        f"| {base} | {len(gs)} | {fmt_mean_std(dices)} | {fmt_mean_std(wts)} | "
                        f"{fmt_mean_std(tcs)} | {fmt_mean_std(ets)} | "
                        f"{','.join(map(str, seeds)) or '-'} |\n"
                    )
                f.write("\n")

        # ===== Per-run indices =====
        f.write("# Per-run Index — Classification\n\n")
        f.write("| Sweep | Experiment | Seed | Best Ep | Acc (%) | F1 (%) | ckpt |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in sorted(cls_rows, key=lambda x: (x["sweep"], x["exp"])):
            f.write(
                f"| {r['sweep']} | {r['exp']} | {r['seed'] or '-'} | {r['epoch']} | "
                f"{r['acc']*100:.2f} | {r['f1']*100:.2f} | {'✓' if r['ckpt'] else '—'} |\n"
            )
        f.write("\n")

        if seg_rows:
            f.write("# Per-run Index — Segmentation\n\n")
            f.write(
                "| Sweep | Experiment | Seed | Best Ep | Dice (%) | WT (%) | TC (%) | ET (%) | ckpt |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(seg_rows, key=lambda x: (x["sweep"], x["exp"])):
                f.write(
                    f"| {r['sweep']} | {r['exp']} | {r['seed'] or '-'} | {r['epoch']} | "
                    f"{r['dice']*100:.2f} | {r['wt']*100:.2f} | {r['tc']*100:.2f} | "
                    f"{r['et']*100:.2f} | {'✓' if r['ckpt'] else '—'} |\n"
                )

    print(
        f"Wrote {REPORT}: {len(cls_rows)} classification + {len(seg_rows)} segmentation runs."
    )


if __name__ == "__main__":
    main()
