#!/bin/bash
# CREMA-D OPM/OGM Sweep — Wei et al. TPAMI 2024 Reproduction
#
# Methods:
#   1. OPM only (feed-forward modulation)
#   2. OPM + OGM combined (feed-forward + back-propagation)
#   3. OGM alone (back-propagation only — already have results, re-run for comparison)
#   4. Baseline (already have results, re-run for comparison)
#
# Architecture: Same as reference code — ConcatFusion Linear(1024→6), no separate classifier
# Training: SGD lr=0.001, momentum=0.9, 100 epochs, batch_size=64
# OPM params: q_base=0.5, λ=0.5, p_exe=0.7, warmup=5
# OGM params: α=0.8
#
# Usage:
#   bash scripts/sweep_cremad_opm.sh phase1     # Single seed (seed=42), all methods
#   bash scripts/sweep_cremad_opm.sh phase2     # Top configs x 5 seeds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/cremad.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep_cremad_opm"

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
        --num-frames 3 --fps 3 \
        $extra_args

    echo "DONE: $full_id"
    echo ""
}

# ========== Phase 1: Single seed exploration ==========
run_phase1() {
    local SEED=42

    echo "============================================================"
    echo "PHASE 1: Single-seed CREMA-D OPM comparison, seed=$SEED"
    echo "============================================================"

    # 1. OPM only (feed-forward modulation)
    run_experiment "cremad_opm" "opm" $SEED \
        --opm-q-base 0.5 --opm-lam 0.5 --opm-p-exe 0.7 --opm-warmup 5 \
        --epochs 100

    # 2. OPM + OGM combined (both feed-forward and back-propagation)
    run_experiment "cremad_opm_ogm" "opm" $SEED \
        --opm-q-base 0.5 --opm-lam 0.5 --opm-p-exe 0.7 --opm-warmup 5 \
        --ogm-ge --alpha 0.8 \
        --epochs 100

    # 3. OGM alone (same arch as OPM for fair comparison — Linear(1024→6))
    # Note: This uses OPM mode but with q_base=0 (no dropping) + OGM-GE
    # Actually, better to use the standard baseline+ogm-ge with the same arch
    # We'll run as OPM with warmup=999 (never activates OPM) + OGM-GE
    run_experiment "cremad_ogm_opmarch" "opm" $SEED \
        --opm-q-base 0.0 --opm-lam 0.0 --opm-p-exe 0.0 --opm-warmup 999 \
        --ogm-ge --alpha 0.8 \
        --epochs 100

    # 4. Baseline (same arch as OPM for fair comparison)
    run_experiment "cremad_baseline_opmarch" "opm" $SEED \
        --opm-q-base 0.0 --opm-lam 0.0 --opm-p-exe 0.0 --opm-warmup 999 \
        --epochs 100

    # 5. ASGML boost + OGM-GE (same arch as OPM for fair comparison)
    python "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --mode adaptive \
        --seed $SEED \
        --exp-name "cremad_boost_ogm_opmarch_seed${SEED}" \
        --output-dir "$OUTPUT_BASE" \
        --num-frames 3 --fps 3 \
        --single-layer-fusion \
        --ogm-ge --alpha 0.8 \
        --asgml-mode continuous --continuous-alpha 0.75 \
        --epochs 100
    echo "DONE: cremad_boost_ogm_opmarch_seed${SEED}"

    echo ""
    echo "============================================================"
    echo "PHASE 1 COMPLETE"
    echo "============================================================"
    echo ""

    # Print results summary
    echo "Results summary (best test accuracy per run):"
    echo "----------------------------------------------"
    for dir in "$OUTPUT_BASE"/cremad_*_seed${SEED}; do
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
        "cremad_opm_ogm:opm:--opm-q-base 0.5 --opm-lam 0.5 --opm-p-exe 0.7 --opm-warmup 5 --ogm-ge --alpha 0.8 --epochs 100"
        "cremad_opm:opm:--opm-q-base 0.5 --opm-lam 0.5 --opm-p-exe 0.7 --opm-warmup 5 --epochs 100"
        "cremad_ogm_opmarch:opm:--opm-q-base 0.0 --opm-lam 0.0 --opm-p-exe 0.0 --opm-warmup 999 --ogm-ge --alpha 0.8 --epochs 100"
        "cremad_baseline_opmarch:opm:--opm-q-base 0.0 --opm-lam 0.0 --opm-p-exe 0.0 --opm-warmup 999 --epochs 100"
    )

    # ASGML boost + OGM-GE needs special handling (uses adaptive mode + --single-layer-fusion)
    echo "--- Running ASGML boost + OGM-GE (OPM arch) across seeds ---"
    for seed in "${SEEDS[@]}"; do
        local full_id="cremad_boost_ogm_opmarch_seed${seed}"
        local exp_dir="$OUTPUT_BASE/$full_id"

        if [ -d "$exp_dir" ] && [ -f "$exp_dir/train.log" ] && grep -q "Training complete" "$exp_dir/train.log" 2>/dev/null; then
            echo "SKIP: Already completed ($full_id)"
            continue
        fi

        echo "Running: $full_id"
        python "$TRAIN_SCRIPT" \
            --config "$CONFIG" \
            --mode adaptive \
            --seed "$seed" \
            --exp-name "$full_id" \
            --output-dir "$OUTPUT_BASE" \
            --num-frames 3 --fps 3 \
            --single-layer-fusion \
            --ogm-ge --alpha 0.8 \
            --asgml-mode continuous --continuous-alpha 0.75 \
            --epochs 100
        echo "DONE: $full_id"
    done

    echo "============================================================"
    echo "PHASE 2: Multi-seed CREMA-D OPM validation"
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
