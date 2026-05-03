#!/bin/bash
# ASGML Hyperparameter Sweep - CREMA-D
# 2-Phase sweep: coarse grid (1 seed) -> fine tune (5 seeds)
#
# Usage:
#   bash scripts/sweep_cremad.sh phase1     # Run Phase 1 coarse grid (18 runs, ~6h)
#   bash scripts/sweep_cremad.sh phase2     # Run Phase 2 fine-tuning (edit TOP_CONFIGS first)
#   bash scripts/sweep_cremad.sh baselines  # Run baseline comparisons (5 seeds each)
#   bash scripts/sweep_cremad.sh aggregate  # Parse results and compute mean/std

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
# Args: run_id mode seed [extra_args...]
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

    python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --mode "$mode" \
        --seed "$seed" \
        --output-dir "$OUTPUT_BASE" \
        --exp-name "$full_id" \
        $extra_args

    echo "DONE: $full_id"
    echo ""
}

# ============================================================
# PHASE 1: Coarse Grid Search (single seed=42, ~18 runs)
# ============================================================
run_phase1() {
    local SEED=42

    echo "======================================="
    echo "PHASE 1: Coarse Grid (seed=$SEED)"
    echo "Estimated time: ~6 hours"
    echo "======================================="

    # Group A: gamma sweep (default=1.0)
    run_experiment "p1_gamma025" adaptive $SEED --gamma-asgml 0.25
    run_experiment "p1_gamma050" adaptive $SEED --gamma-asgml 0.5
    # gamma=1.0 is the existing default result (skip)
    run_experiment "p1_gamma150" adaptive $SEED --gamma-asgml 1.5

    # Group B: tau_base sweep (default=2.0)
    run_experiment "p1_taubase10" adaptive $SEED --tau-base 1.0
    run_experiment "p1_taubase15" adaptive $SEED --tau-base 1.5
    run_experiment "p1_taubase25" adaptive $SEED --tau-base 2.5
    run_experiment "p1_taubase30" adaptive $SEED --tau-base 3.0

    # Group C: soft_mask_scale sweep (default=0.1)
    run_experiment "p1_sms000" adaptive $SEED --soft-mask-scale 0.0
    run_experiment "p1_sms005" adaptive $SEED --soft-mask-scale 0.05
    run_experiment "p1_sms020" adaptive $SEED --soft-mask-scale 0.2
    run_experiment "p1_sms030" adaptive $SEED --soft-mask-scale 0.3

    # Group D: beta sweep (default=0.5)
    run_experiment "p1_beta000" adaptive $SEED --beta 0.0
    run_experiment "p1_beta025" adaptive $SEED --beta 0.25
    run_experiment "p1_beta075" adaptive $SEED --beta 0.75
    run_experiment "p1_beta100" adaptive $SEED --beta 1.0

    # Group E: threshold_delta (default=0.1)
    run_experiment "p1_thresh005" adaptive $SEED --threshold-delta 0.05

    # Group F: signal_source (default=dual)
    run_experiment "p1_probe" adaptive $SEED --signal-source probe

    # Group G: Combined best guess
    run_experiment "p1_combo1" adaptive $SEED --gamma-asgml 0.5 --tau-base 1.5

    echo ""
    echo "======================================="
    echo "Phase 1 complete!"
    echo "Run: bash $0 aggregate"
    echo "Then review results and edit TOP_CONFIGS in this script for Phase 2"
    echo "======================================="
}

# ============================================================
# PHASE 2: Fine-tune top configs (5 seeds each)
# Edit TOP_CONFIGS after reviewing Phase 1 results!
# ============================================================
run_phase2() {
    local SEEDS=(42 0 1 2 3)

    echo "======================================="
    echo "PHASE 2: Fine-tune Top Configs (5 seeds)"
    echo "======================================="

    # ---- UPDATED FROM PHASE 1 RESULTS (2026-02-11) ----
    # Top 1: soft_mask_scale=0.0 (61.02%) — hard mask, best overall
    # Top 2: default config (60.75%) — reference adaptive
    # Top 3: beta=1.0 (60.22%) — gradient-only signal
    # Top 4: staleness mode lambda_comp=0.2 (60.22%) — best staleness variant
    local -a TOP_CONFIGS=(
        "sms000:--soft-mask-scale 0.0"
        "default:"
        "beta100:--beta 1.0"
        "stale_lc020:--asgml-mode staleness --lambda-comp 0.2"
    )
    # ---- END EDIT SECTION ----

    for config_str in "${TOP_CONFIGS[@]}"; do
        local run_id="${config_str%%:*}"
        local extra_args="${config_str#*:}"
        for seed in "${SEEDS[@]}"; do
            run_experiment "p2_${run_id}" adaptive "$seed" $extra_args
        done
    done

    echo "Phase 2 complete!"
}

# ============================================================
# BASELINES: 5 seeds each for comparison
# ============================================================
run_baselines() {
    local SEEDS=(42 0 1 2 3)

    echo "======================================="
    echo "BASELINES (5 seeds each)"
    echo "======================================="

    # Baseline (no ASGML)
    for seed in "${SEEDS[@]}"; do
        run_experiment "baseline" baseline "$seed"
    done

    # OGM-GE alone
    for seed in "${SEEDS[@]}"; do
        run_experiment "ogmge" baseline "$seed" --ogm-ge --alpha 0.8
    done

    # Default ASGML adaptive (reference point)
    for seed in "${SEEDS[@]}"; do
        run_experiment "asgml_default" adaptive "$seed"
    done

    echo "Baselines complete!"
}

# ============================================================
# AGGREGATE: Parse all results
# ============================================================
run_aggregate() {
    python "$PROJECT_DIR/scripts/aggregate_sweep.py" \
        --sweep-dir "$OUTPUT_BASE" \
        --output "$OUTPUT_BASE/sweep_summary.md"
}

# ============================================================
# MAIN
# ============================================================
case "${1:-help}" in
    phase1)    run_phase1 ;;
    phase2)    run_phase2 ;;
    baselines) run_baselines ;;
    aggregate) run_aggregate ;;
    all)
        run_baselines
        run_phase1
        echo ""
        echo "Phase 1 + baselines complete. Review results, edit TOP_CONFIGS, then run phase2."
        ;;
    *)
        echo "Usage: $0 {phase1|phase2|baselines|aggregate|all}"
        echo ""
        echo "  phase1     - Coarse grid search (18 runs, seed=42, ~6h)"
        echo "  phase2     - Fine-tune top configs (5 seeds each, edit script first)"
        echo "  baselines  - Baseline + OGM-GE + default ASGML (5 seeds each)"
        echo "  aggregate  - Parse all results into summary table"
        echo "  all        - Run baselines + phase1"
        exit 1
        ;;
esac
