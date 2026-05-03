#!/bin/bash
# CGGM Baseline Sweep — Guo et al., NeurIPS 2024
#
# Runs CGGM on all datasets for comparison with ASGML.
# Phase 1: Single seed (42) across all datasets
# Phase 2: 5 seeds on key datasets
#
# Usage:
#   bash scripts/sweep_cggm.sh phase1     # Single seed, all datasets
#   bash scripts/sweep_cggm.sh phase2     # Multi-seed

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep_cggm"
PYTHON="/home/main/miniconda3/envs/phd/bin/python"

mkdir -p "$OUTPUT_BASE"

run_experiment() {
    local run_id="$1"
    local config="$2"
    local mode="$3"
    local seed="$4"
    shift 4
    local extra_args="$*"

    local full_id="${run_id}_seed${seed}"
    local exp_dir="$OUTPUT_BASE/$full_id"

    echo "=========================================="
    echo "Running: $full_id"
    echo "Config: $config | Mode: $mode | Seed: $seed"
    echo "Extra args: $extra_args"
    echo "=========================================="

    if [ -d "$exp_dir" ] && [ -f "$exp_dir/train.log" ] && grep -q "Training complete" "$exp_dir/train.log" 2>/dev/null; then
        echo "SKIP: Already completed ($full_id)"
        return
    fi

    $PYTHON "$TRAIN_SCRIPT" \
        --config "$PROJECT_DIR/configs/$config" \
        --mode "$mode" \
        --seed "$seed" \
        --exp-name "$full_id" \
        --output-dir "$OUTPUT_BASE" \
        $extra_args

    echo "DONE: $full_id"
    echo ""
}

# ========== Phase 1: Single seed, all datasets ==========
run_phase1() {
    local SEED=42

    echo "============================================================"
    echo "PHASE 1: CGGM single-seed sweep across all datasets"
    echo "============================================================"

    # --- CREMA-D (3 frames) ---
    run_experiment "cremad_cggm" "cremad.yaml" "cggm" $SEED \
        --num-frames 3 --fps 3 --epochs 100 \
        --cggm-rou 1.3 --cggm-lamda 0.2

    # --- KS ---
    run_experiment "ks_cggm" "kinetics_sounds.yaml" "cggm" $SEED \
        --num-frames 3 --epochs 100 \
        --cggm-rou 1.3 --cggm-lamda 0.2

    # --- AVE ---
    run_experiment "ave_cggm" "ave.yaml" "cggm" $SEED \
        --epochs 100 \
        --cggm-rou 1.3 --cggm-lamda 0.2

    # --- MOSEI ---
    run_experiment "mosei_cggm" "mosei.yaml" "cggm" $SEED \
        --epochs 100 \
        --cggm-rou 1.3 --cggm-lamda 0.2

    # --- MOSI ---
    run_experiment "mosi_cggm" "mosi.yaml" "cggm" $SEED \
        --epochs 100 \
        --cggm-rou 1.3 --cggm-lamda 0.2

    echo ""
    echo "============================================================"
    echo "PHASE 1 COMPLETE"
    echo "============================================================"

    echo "Results (best test accuracy):"
    for dir in "$OUTPUT_BASE"/*_cggm_seed${SEED}; do
        if [ -d "$dir" ] && [ -f "$dir/train.log" ]; then
            local run_name=$(basename "$dir")
            local best_acc=$(grep "New best model" "$dir/train.log" | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
            echo "  $run_name: $best_acc"
        fi
    done
}

# ========== Phase 2: Multi-seed on key datasets ==========
run_phase2() {
    local -a SEEDS=(42 123 456 789 1024)

    echo "============================================================"
    echo "PHASE 2: CGGM multi-seed, seeds: ${SEEDS[*]}"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        # CREMA-D
        run_experiment "cremad_cggm" "cremad.yaml" "cggm" $seed \
            --num-frames 3 --fps 3 --epochs 100 \
            --cggm-rou 1.3 --cggm-lamda 0.2

        # KS
        run_experiment "ks_cggm" "kinetics_sounds.yaml" "cggm" $seed \
            --num-frames 3 --epochs 100 \
            --cggm-rou 1.3 --cggm-lamda 0.2

        # AVE
        run_experiment "ave_cggm" "ave.yaml" "cggm" $seed \
            --epochs 100 \
            --cggm-rou 1.3 --cggm-lamda 0.2

        # MOSEI
        run_experiment "mosei_cggm" "mosei.yaml" "cggm" $seed \
            --epochs 100 \
            --cggm-rou 1.3 --cggm-lamda 0.2

        # MOSI
        run_experiment "mosi_cggm" "mosi.yaml" "cggm" $seed \
            --epochs 100 \
            --cggm-rou 1.3 --cggm-lamda 0.2
    done

    echo ""
    echo "============================================================"
    echo "PHASE 2 COMPLETE"
    echo "============================================================"
}

case "${1:-phase1}" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    *) echo "Usage: $0 {phase1|phase2}"; exit 1 ;;
esac
