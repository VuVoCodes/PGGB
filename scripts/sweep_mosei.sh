#!/bin/bash
# CMU-MOSEI Dataset Sweep - 3-Modality Sentiment Analysis (text + audio + vision)
#
# Methods:
#   1. Baseline (joint training) - Adam
#   2. OGM-GE - Adam (N-modality generalization)
#   3. ASGML boost + OGM-GE (α=0.75) - Adam
#   4. ASGML boost only (α=0.5) - Adam
#
# Usage:
#   bash scripts/sweep_mosei.sh phase1     # Single seed (seed=42), all methods
#   bash scripts/sweep_mosei.sh phase2     # All configs x 5 seeds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/mosei.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep_mosei"

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
    echo "PHASE 1: Single-seed MOSEI comparison, seed=$SEED"
    echo "============================================================"

    # 1. Baseline (joint training)
    run_experiment "mosei_baseline" "baseline" $SEED \
        --epochs 100

    # 2. OGM-GE only (N-modality generalized)
    run_experiment "mosei_ogm_ge" "adaptive" $SEED \
        --ogm-ge --alpha 0.8 \
        --asgml-mode continuous --continuous-alpha 0 \
        --epochs 100

    # 3. ASGML boost + OGM-GE (α=0.75) — best from CREMA-D
    run_experiment "mosei_boost_ogm_a075" "adaptive" $SEED \
        --ogm-ge --alpha 0.8 \
        --asgml-mode continuous --continuous-alpha 0.75 \
        --epochs 100

    # 4. ASGML boost only (no OGM-GE)
    run_experiment "mosei_boost_only" "adaptive" $SEED \
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
    for dir in "$OUTPUT_BASE"/mosei_*_seed${SEED}; do
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

    local -a TOP_CONFIGS=(
        "mosei_boost_ogm_a075:adaptive:--ogm-ge --alpha 0.8 --asgml-mode continuous --continuous-alpha 0.75 --epochs 100"
        "mosei_ogm_ge:adaptive:--ogm-ge --alpha 0.8 --asgml-mode continuous --continuous-alpha 0 --epochs 100"
        "mosei_boost_only:adaptive:--asgml-mode continuous --continuous-alpha 0.5 --epochs 100"
        "mosei_baseline:baseline:--epochs 100"
    )

    echo "============================================================"
    echo "PHASE 2: Multi-seed MOSEI validation"
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
