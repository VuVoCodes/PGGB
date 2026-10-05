#!/bin/bash
# MSLR (Yao 2022) cross-class comparison sweep — fixed-ratio Keep variant.
# Reproduces App. B.9 Table 10: 7 datasets × 5 seeds.
#
# Per-dataset LR ratios (heuristic Keep variant, no unimodal pre-calibration):
#   CREMA-D:    audio:1.0/visual:2.0  (selected from a 3-config sweep)
#   AVE:        audio:0.5/visual:1.0
#   KS:         audio:0.5/visual:1.0
#   MOSI:       text:0.5/audio:1.0/visual:1.0
#   CH-SIMS:      text:0.5/audio:1.0/vision:1.0
#   Twitter15:  text:0.5/image:1.0
#   Sarcasm:    text:0.5/image:1.0
#
# Usage:
#   bash scripts/sweep_mslr.sh
#
# Optional environment variables:
#   PROJECT_ROOT  : project root (auto-detected from script location if unset)
#   PYTHON        : Python interpreter (defaults to "python")
#   CONDA_ENV     : conda env to activate (optional; skipped if unset)

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
cd "$PROJECT_ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" || true
    conda activate "$CONDA_ENV" || true
fi

OUT=outputs/sweep_mslr
mkdir -p "$OUT"
BEST_RATIO="audio:1.0,visual:2.0"
BEST_NAME="mslr_cremad"

echo "[$(date)] ===== MSLR sweep (CREMA-D best ratio: $BEST_RATIO) ====="

# 1. CREMA-D 5 seeds at best ratio (skip if already complete)
echo "[$(date)] ===== CREMA-D — 5 seeds ====="
for SEED in 42 123 456 789 1024; do
    EXP="${BEST_NAME}_seed${SEED}"
    if [ -f "$OUT/$EXP/train.log" ] && grep -q "Training complete" "$OUT/$EXP/train.log"; then
        echo "  [SKIP] $EXP already done"
        continue
    fi
    if [ -d "$OUT/$EXP" ]; then
        rm -rf "$OUT/$EXP"
    fi
    echo "[$(date)] CREMA-D seed=$SEED ratio=$BEST_RATIO"
    $PYTHON scripts/train.py --config configs/cremad.yaml --mode baseline \
        --fps 3 --num-frames 3 --seed "$SEED" --epochs 100 \
        --output-dir "$OUT" --exp-name "$EXP" \
        --modality-lr "$BEST_RATIO" 2>&1 | tail -3 || echo "  [FAIL] $EXP — continuing"
done

# 2. Per-dataset MSLR sweeps (5 seeds each, heuristic Keep ratios)
declare -A DATASETS
DATASETS[ave]="configs/ave.yaml|audio:0.5,visual:1.0|"
DATASETS[ks]="configs/kinetics_sounds.yaml|audio:0.5,visual:1.0|"
DATASETS[mosi]="configs/mosi.yaml|text:0.5,audio:1.0,visual:1.0|"
DATASETS[chsims]="configs/chsims.yaml|text:0.5,audio:1.0,vision:1.0|"
DATASETS[twitter]="configs/twitter.yaml|text:0.5,image:1.0|"
DATASETS[sarcasm]="configs/sarcasm.yaml|text:0.5,image:1.0|"

ORDER=(mosi twitter sarcasm ave chsims ks)

for DS in "${ORDER[@]}"; do
    SPEC="${DATASETS[$DS]}"
    CONFIG="${SPEC%%|*}"
    REST="${SPEC#*|}"
    LR_RATIO="${REST%%|*}"
    EXTRA_ARGS="${REST#*|}"

    echo ""
    echo "[$(date)] ===== $DS — 5 seeds at ratio $LR_RATIO ====="
    for SEED in 42 123 456 789 1024; do
        EXP="mslr_${DS}_seed${SEED}"
        if [ -f "$OUT/$EXP/train.log" ] && grep -q "Training complete" "$OUT/$EXP/train.log"; then
            echo "  [SKIP] $EXP already done"
            continue
        fi
        if [ -d "$OUT/$EXP" ]; then
            rm -rf "$OUT/$EXP"
        fi
        echo "[$(date)] $DS seed=$SEED ratio=$LR_RATIO"
        $PYTHON scripts/train.py --config "$CONFIG" --mode baseline \
            --seed "$SEED" \
            --output-dir "$OUT" --exp-name "$EXP" \
            --modality-lr "$LR_RATIO" $EXTRA_ARGS 2>&1 | tail -3 || echo "  [FAIL] $EXP — continuing"
    done
done

echo ""
echo "[$(date)] ===== ALL MSLR SWEEPS DONE ====="
echo ""
echo "===== Summary: best test acc per run ====="
for d in $OUT/mslr_*_seed*; do
    LOG="$d/train.log"
    if [ -f "$LOG" ]; then
        BEST=$(grep "Training complete. Best accuracy:" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "INCOMPLETE")
        echo "  $(basename $d): $BEST"
    fi
done
