#!/bin/bash
# ASGML Staleness-mode Adaptive Sweep
# Tests adaptive ASGML with staleness (stale gradients) instead of frequency (gradient skipping)
#
# Usage: bash scripts/sweep_staleness.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/cremad.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/sweep"

eval "$(conda shell.bash hook 2>/dev/null)" || { source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; }
conda activate phd

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

SEED=42

echo "======================================="
echo "STALENESS MODE SWEEP (seed=$SEED)"
echo "======================================="

# S1: Default staleness (same params as best frequency run)
run_experiment "p1_stale_default" adaptive $SEED --asgml-mode staleness

# S2: Staleness + lower lambda_comp (less gradient compensation)
run_experiment "p1_stale_lc005" adaptive $SEED --asgml-mode staleness --lambda-comp 0.05

# S3: Staleness + higher lambda_comp (more compensation)
run_experiment "p1_stale_lc020" adaptive $SEED --asgml-mode staleness --lambda-comp 0.2

# S4: Staleness + lower tau_base (gentler staleness)
run_experiment "p1_stale_tb15" adaptive $SEED --asgml-mode staleness --tau-base 1.5

# S5: Staleness + gamma=0.5 (if frequency sweep shows gamma=0.5 is better)
run_experiment "p1_stale_g050" adaptive $SEED --asgml-mode staleness --gamma-asgml 0.5

# S6: Staleness + combined best guesses
run_experiment "p1_stale_combo" adaptive $SEED --asgml-mode staleness --gamma-asgml 0.5 --tau-base 1.5 --lambda-comp 0.05

echo ""
echo "======================================="
echo "Staleness sweep complete!"
echo "Run: bash scripts/sweep_cremad.sh aggregate"
echo "======================================="
