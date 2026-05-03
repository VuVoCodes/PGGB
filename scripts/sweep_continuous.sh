#!/bin/bash
# ASGML Continuous Mode Sweep v2 - CREMA-D
# Tests boost-weak probe-guided gradient scaling (with probe split-eval fix)
#
# v2 changes from v1:
#   - Boost weak modality (scale > 1.0) instead of throttle dominant (scale < 1.0)
#   - Probe train/eval uses split batches to prevent overfitting
#   - scale_min replaced with scale_max
#
# Usage:
#   bash scripts/sweep_continuous.sh phase1     # Coarse grid (9 runs, seed=42)
#   bash scripts/sweep_continuous.sh phase2     # Top configs x 5 seeds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/cremad.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep"

# Conda activation
eval "$(conda shell.bash hook 2>/dev/null)" || { source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; }
conda activate phd

mkdir -p "$OUTPUT_BASE"

# Run a single experiment
run_experiment() {
    local run_id="$1"
    local mode="$2"
    local seed="$3"
    shift 3
    local extra_args="$*"

    local full_id="${run_id}_seed${seed}"
    local exp_dir="$OUTPUT_BASE/$full_id"

    echo "=========================================="
    echo "Running: $full_id"
    echo "Mode: $mode | Seed: $seed"
    echo "Extra args: $extra_args"
    echo "=========================================="

    # Skip if already completed
    if [ -d "$exp_dir" ] && [ -f "$exp_dir/train.log" ] && grep -q "Training complete" "$exp_dir/train.log" 2>/dev/null; then
        echo "SKIP: Already completed ($full_id)"
        return
    fi

    # Remove incomplete run
    [ -d "$exp_dir" ] && rm -rf "$exp_dir"

    python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --mode "$mode" \
        --seed "$seed" \
        --output-dir "$OUTPUT_BASE" \
        --exp-name "$full_id" \
        --asgml-mode continuous \
        $extra_args

    echo "DONE: $full_id"
    echo ""
}

# ============================================================
# PHASE 1: Coarse Grid (seed=42, ~9 runs)
# Boost-weak mode: weak modality gets scale > 1.0
# ============================================================
run_phase1() {
    local SEED=42

    echo "======================================="
    echo "BOOST-WEAK SWEEP Phase 1 (seed=$SEED)"
    echo "======================================="

    # B1: Default boost (alpha=0.5, scale_max=2.0)
    run_experiment "boost_default" adaptive $SEED

    # B2-B4: Alpha sweep (boost strength)
    run_experiment "boost_a025" adaptive $SEED --continuous-alpha 0.25
    run_experiment "boost_a075" adaptive $SEED --continuous-alpha 0.75
    run_experiment "boost_a100" adaptive $SEED --continuous-alpha 1.0

    # B5-B6: Scale max sweep (cap for weak modality boost)
    run_experiment "boost_sm150" adaptive $SEED --continuous-scale-max 1.5
    run_experiment "boost_sm300" adaptive $SEED --continuous-scale-max 3.0

    # B7: With Gaussian noise injection
    run_experiment "boost_noise" adaptive $SEED --continuous-noise-sigma 0.1

    # B8: Boost + OGM-GE combined (complementary: OGM throttles dominant, we boost weak)
    run_experiment "boost_ogm" adaptive $SEED --ogm-ge --alpha 0.8

    # B9: Boost + OGM-GE + higher alpha
    run_experiment "boost_ogm_a075" adaptive $SEED --ogm-ge --alpha 0.8 --continuous-alpha 0.75

    echo ""
    echo "======================================="
    echo "Boost-weak sweep Phase 1 complete!"
    echo "======================================="
}

# ============================================================
# PHASE 2: Top configs x 5 seeds
# Edit TOP_CONFIGS after reviewing Phase 1 results
# ============================================================
run_phase2() {
    local SEEDS=(42 0 1 2 3)

    echo "======================================="
    echo "BOOST-WEAK SWEEP Phase 2 (5 seeds)"
    echo "======================================="

    # ---- EDITED AFTER PHASE 1 (Feb 12) ----
    # #1: boost+OGM-GE α=0.75 → 62.50%
    # #2: boost+OGM-GE α=0.5  → 61.83%
    # #3: boost only default   → 60.48%
    local -a TOP_CONFIGS=(
        "boost_ogm_a075:--ogm-ge --alpha 0.8 --continuous-alpha 0.75"
        "boost_ogm:--ogm-ge --alpha 0.8"
        "boost_default:"
    )
    # ---- END EDIT ----

    for config_str in "${TOP_CONFIGS[@]}"; do
        local run_id="${config_str%%:*}"
        local extra_args="${config_str#*:}"
        for seed in "${SEEDS[@]}"; do
            run_experiment "p2_${run_id}" adaptive "$seed" $extra_args
        done
    done

    echo "Boost-weak Phase 2 complete!"
}

# ============================================================
# MAIN
# ============================================================
case "${1:-help}" in
    phase1)    run_phase1 ;;
    phase2)    run_phase2 ;;
    *)
        echo "Usage: $0 {phase1|phase2}"
        exit 1
        ;;
esac
