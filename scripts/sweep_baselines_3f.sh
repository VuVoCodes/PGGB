#!/bin/bash
# Baseline Comparison Sweep - CREMA-D (3 frames)
# Runs all methods with 3 frames for fair comparison with MILES and InfoReg papers.
#
# Methods:
#   1. Baseline (joint training) - SGD
#   2. OGM-GE - SGD
#   3. MILES - Adam (paper setting)
#   4. InfoReg - SGD lr=0.002 (paper setting)
#   5. ASGML boost + OGM-GE - SGD (our best method)
#
# Usage:
#   bash scripts/sweep_baselines_3f.sh phase1     # Single seed (seed=42), all methods
#   bash scripts/sweep_baselines_3f.sh phase2     # Top configs x 5 seeds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/cremad.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep_3f"

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

    python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --mode "$mode" \
        --fps 3 --num-frames 3 \
        --seed "$seed" \
        --exp-name "$full_id" \
        --output-dir "$OUTPUT_BASE" \
        $extra_args

    echo "DONE: $full_id"
    echo ""
}

# ========== Phase 1: Single seed exploration ==========
run_phase1() {
    local SEED=42

    echo "============================================================"
    echo "PHASE 1: Single-seed comparison (3 frames), seed=$SEED"
    echo "============================================================"

    # 1. Baseline (joint training) — SGD lr=0.001, 100 epochs
    run_experiment "3f_baseline" "baseline" $SEED \
        --epochs 100

    # 2. OGM-GE — our standard setup with 3 frames
    run_experiment "3f_ogm_ge" "adaptive" $SEED \
        --ogm-ge --alpha 0.8 \
        --asgml-mode continuous --continuous-alpha 0 \
        --epochs 100

    # 3. MILES — Adam optimizer (paper: lr search, τ=0.2, μ=0.5)
    #    Default config uses Adam via miles_optimizer setting
    run_experiment "3f_miles_t02" "miles" $SEED \
        --miles-threshold 0.2 --miles-reduction 0.5 \
        --epochs 100

    run_experiment "3f_miles_t005" "miles" $SEED \
        --miles-threshold 0.05 --miles-reduction 0.5 \
        --epochs 100

    # 4. InfoReg — SGD lr=0.002, 50 epochs (exact paper setting)
    run_experiment "3f_inforeg_paper" "inforeg" $SEED \
        --inforeg-beta 0.9 --inforeg-K 0.04 \
        --lr 0.002 --epochs 50

    # 5. InfoReg — extended to 100 epochs for fair comparison
    run_experiment "3f_inforeg_100ep" "inforeg" $SEED \
        --inforeg-beta 0.9 --inforeg-K 0.04 \
        --lr 0.002 --epochs 100

    # 6. ASGML boost + OGM-GE (our best, α=0.75) — with 3 frames
    run_experiment "3f_boost_ogm_a075" "adaptive" $SEED \
        --ogm-ge --alpha 0.8 \
        --asgml-mode continuous --continuous-alpha 0.75 \
        --epochs 100

    # 7. ASGML boost only (no OGM-GE) — with 3 frames
    run_experiment "3f_boost_only" "adaptive" $SEED \
        --asgml-mode continuous --continuous-alpha 0.5 \
        --epochs 100

    echo ""
    echo "============================================================"
    echo "PHASE 1 COMPLETE"
    echo "============================================================"
    echo ""

    # Print results summary
    echo "Results summary (best test accuracy per run):"
    echo "----------------------------------------------"
    for dir in "$OUTPUT_BASE"/3f_*_seed${SEED}; do
        if [ -d "$dir" ] && [ -f "$dir/train.log" ]; then
            local run_name=$(basename "$dir")
            local best_acc=$(grep "New best model" "$dir/train.log" | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
            echo "  $run_name: $best_acc"
        fi
    done
}

# ========== Phase 2: Multi-seed validation ==========
run_phase2() {
    local -a SEEDS=(42 123 456 789 1024)

    # Top configs from Phase 1 — UPDATE THESE after Phase 1 results
    local -a TOP_CONFIGS=(
        "3f_boost_ogm_a075:adaptive:--ogm-ge --alpha 0.8 --asgml-mode continuous --continuous-alpha 0.75 --epochs 100"
        "3f_ogm_ge:adaptive:--ogm-ge --alpha 0.8 --asgml-mode continuous --continuous-alpha 0 --epochs 100"
        "3f_inforeg_100ep:inforeg:--inforeg-beta 0.9 --inforeg-K 0.04 --lr 0.002 --epochs 100"
        "3f_miles_t02:miles:--miles-threshold 0.2 --miles-reduction 0.5 --epochs 100"
        "3f_baseline:baseline:--epochs 100"
    )

    echo "============================================================"
    echo "PHASE 2: Multi-seed validation (3 frames)"
    echo "Seeds: ${SEEDS[*]}"
    echo "Configs: ${#TOP_CONFIGS[@]}"
    echo "Total runs: $((${#TOP_CONFIGS[@]} * ${#SEEDS[@]}))"
    echo "============================================================"

    for config_str in "${TOP_CONFIGS[@]}"; do
        IFS=':' read -r run_id mode extra_args <<< "$config_str"
        for seed in "${SEEDS[@]}"; do
            run_experiment "$run_id" "$mode" "$seed" $extra_args
        done
    done

    echo ""
    echo "============================================================"
    echo "PHASE 2 COMPLETE"
    echo "============================================================"
    echo ""

    # Print multi-seed summary
    echo "Multi-seed results summary:"
    echo "----------------------------------------------"
    for config_str in "${TOP_CONFIGS[@]}"; do
        IFS=':' read -r run_id mode extra_args <<< "$config_str"
        local accs=""
        for seed in "${SEEDS[@]}"; do
            local dir="$OUTPUT_BASE/${run_id}_seed${seed}"
            if [ -f "$dir/train.log" ]; then
                local acc=$(grep "New best model" "$dir/train.log" | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
                accs="$accs $acc"
            fi
        done
        echo "  $run_id: $accs"
    done
}

# Main
case "${1:-phase1}" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    *) echo "Usage: $0 {phase1|phase2}"; exit 1 ;;
esac
