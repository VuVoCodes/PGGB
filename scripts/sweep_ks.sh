#!/bin/bash
# Kinetics-Sounds Sweep — ASGML Experiments
#
# Methods:
#   1. Baseline (no modulation)
#   2. OGM-GE alone (α=0.8)
#   3. ASGML boost only (α=0.5)
#   4. ASGML boost + OGM-GE (α=0.75)
#
# Architecture: ResNet18, late fusion (concat → MLP), 3 frames
# Training: SGD lr=0.001, momentum=0.9, 100 epochs, batch_size=64
#
# Usage:
#   bash scripts/sweep_ks.sh phase1     # Single seed (42)
#   bash scripts/sweep_ks.sh phase2     # 5 seeds

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/kinetics_sounds.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep_ks"

# Use phd python directly to avoid conda run buffering
PYTHON="/home/main/miniconda3/envs/phd/bin/python"

mkdir -p "$OUTPUT_BASE"

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

    $PYTHON "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --mode "$mode" \
        --seed "$seed" \
        --exp-name "$full_id" \
        --output-dir "$OUTPUT_BASE" \
        --num-frames 3 \
        $extra_args

    echo "DONE: $full_id"
    echo ""
}

# ========== Phase 1: Single seed ==========
run_phase1() {
    local SEED=42

    echo "============================================================"
    echo "PHASE 1: Single-seed Kinetics-Sounds sweep, seed=$SEED"
    echo "============================================================"

    # 1. Baseline
    run_experiment "ks_baseline" "baseline" $SEED \
        --epochs 100

    # 2. OGM-GE alone
    run_experiment "ks_ogmge" "baseline" $SEED \
        --ogm-ge --alpha 0.8 \
        --epochs 100

    # 3. ASGML boost only
    run_experiment "ks_boost_only" "adaptive" $SEED \
        --asgml-mode continuous --continuous-alpha 0.5 \
        --epochs 100

    # 4. ASGML boost + OGM-GE
    run_experiment "ks_boost_ogm" "adaptive" $SEED \
        --ogm-ge --alpha 0.8 \
        --asgml-mode continuous --continuous-alpha 0.75 \
        --epochs 100

    echo ""
    echo "============================================================"
    echo "PHASE 1 COMPLETE"
    echo "============================================================"

    # Print results
    echo "Results (best test accuracy):"
    for dir in "$OUTPUT_BASE"/ks_*_seed${SEED}; do
        if [ -d "$dir" ] && [ -f "$dir/train.log" ]; then
            local run_name=$(basename "$dir")
            local best_acc=$(grep "New best model" "$dir/train.log" | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
            echo "  $run_name: $best_acc"
        fi
    done
}

# ========== Phase 2: Multi-seed ==========
run_phase2() {
    local -a SEEDS=(42 123 456 789 1024)

    echo "============================================================"
    echo "PHASE 2: Multi-seed Kinetics-Sounds, seeds: ${SEEDS[*]}"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        # Baseline
        run_experiment "ks_baseline" "baseline" $seed \
            --epochs 100

        # OGM-GE
        run_experiment "ks_ogmge" "baseline" $seed \
            --ogm-ge --alpha 0.8 \
            --epochs 100

        # Boost only
        run_experiment "ks_boost_only" "adaptive" $seed \
            --asgml-mode continuous --continuous-alpha 0.5 \
            --epochs 100

        # Boost + OGM-GE
        run_experiment "ks_boost_ogm" "adaptive" $seed \
            --ogm-ge --alpha 0.8 \
            --asgml-mode continuous --continuous-alpha 0.75 \
            --epochs 100
    done

    echo ""
    echo "============================================================"
    echo "PHASE 2 COMPLETE"
    echo "============================================================"

    # Print multi-seed summary
    echo "Multi-seed results:"
    for method in ks_baseline ks_ogmge ks_boost_only ks_boost_ogm; do
        printf "  %-20s" "$method"
        for seed in "${SEEDS[@]}"; do
            local dir="$OUTPUT_BASE/${method}_seed${seed}"
            if [ -f "$dir/train.log" ]; then
                local acc=$(grep "New best model" "$dir/train.log" | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
                printf "%8s" "$acc"
            else
                printf "%8s" "N/A"
            fi
        done
        echo ""
    done
}

# Main
case "${1:-phase1}" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    *) echo "Usage: $0 {phase1|phase2}"; exit 1 ;;
esac
