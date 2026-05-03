#!/bin/bash
# CMU-MOSI Sweep — ASGML + baselines
#
# Usage:
#   bash scripts/sweep_mosi.sh phase1     # Single seed (42)
#   bash scripts/sweep_mosi.sh phase2     # 5 seeds

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/mosi.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep_mosi"
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
    echo "=========================================="

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
        $extra_args

    echo "DONE: $full_id"
}

run_phase1() {
    local SEED=42
    echo "============================================================"
    echo "PHASE 1: MOSI single-seed sweep, seed=$SEED"
    echo "============================================================"

    # Baseline
    run_experiment "mosi_baseline" "baseline" $SEED --epochs 100

    # OGM-GE
    run_experiment "mosi_ogmge" "baseline" $SEED --ogm-ge --alpha 0.8 --epochs 100

    # Boost only
    run_experiment "mosi_boost_only" "adaptive" $SEED \
        --asgml-mode continuous --continuous-alpha 0.5 --epochs 100

    # Boost + OGM-GE
    run_experiment "mosi_boost_ogm" "adaptive" $SEED \
        --ogm-ge --alpha 0.8 \
        --asgml-mode continuous --continuous-alpha 0.75 --epochs 100

    # CGGM (already done, skip)
    run_experiment "mosi_cggm" "cggm" $SEED \
        --cggm-rou 1.3 --cggm-lamda 0.2 --epochs 100

    echo "============================================================"
    echo "PHASE 1 COMPLETE"
    echo "============================================================"
    for dir in "$OUTPUT_BASE"/mosi_*_seed${SEED}; do
        [ -d "$dir" ] && name=$(basename "$dir") && acc=$(grep "New best model" "$dir/train.log" 2>/dev/null | tail -1 | grep -oP 'accuracy: \K[0-9.]+') && echo "  $name: $acc"
    done
}

run_phase2() {
    local -a SEEDS=(42 123 456 789 1024)
    echo "============================================================"
    echo "PHASE 2: MOSI multi-seed, seeds: ${SEEDS[*]}"
    echo "============================================================"

    for seed in "${SEEDS[@]}"; do
        run_experiment "mosi_baseline" "baseline" $seed --epochs 100
        run_experiment "mosi_ogmge" "baseline" $seed --ogm-ge --alpha 0.8 --epochs 100
        run_experiment "mosi_boost_only" "adaptive" $seed \
            --asgml-mode continuous --continuous-alpha 0.5 --epochs 100
        run_experiment "mosi_boost_ogm" "adaptive" $seed \
            --ogm-ge --alpha 0.8 \
            --asgml-mode continuous --continuous-alpha 0.75 --epochs 100
    done

    echo "============================================================"
    echo "PHASE 2 COMPLETE"
    echo "============================================================"
}

case "${1:-phase1}" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    *) echo "Usage: $0 {phase1|phase2}"; exit 1 ;;
esac
