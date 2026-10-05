#!/bin/bash
# CMU-MOSI alpha-sensitivity sweep (rebuttal, R2-gN93 Q4)
# Boost-only, seed 42, alpha in {0.25, 0.5, 0.75, 1.0, 1.5}
# Mirrors outputs/rebuttal_p1_alpha (CH-SIMS-true) and rebuttal_p1_alpha_chsims.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
CONFIG="$PROJECT_DIR/configs/mosi.yaml"
OUTPUT_BASE="$PROJECT_DIR/outputs/rebuttal_p1_alpha_mosi"
PYTHON="${PYTHON:-python}"
SEED=42

mkdir -p "$OUTPUT_BASE"

for ALPHA in 0.25 0.5 0.75 1.0 1.5; do
    full_id="mosi_alpha${ALPHA}_seed${SEED}"
    exp_dir="$OUTPUT_BASE/$full_id"

    if [ -d "$exp_dir" ] && [ -f "$exp_dir/train.log" ] && grep -q "Training complete" "$exp_dir/train.log" 2>/dev/null; then
        echo "SKIP: Already completed ($full_id)"
        continue
    fi

    echo "RUN: $full_id"
    $PYTHON "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --mode adaptive \
        --seed $SEED \
        --exp-name "$full_id" \
        --output-dir "$OUTPUT_BASE" \
        --asgml-mode continuous --continuous-alpha "$ALPHA" \
        --epochs 100 \
        > "$OUTPUT_BASE/${full_id}.stdout" 2>&1
    echo "DONE: $full_id"
done

echo "SWEEP COMPLETE"
for dir in "$OUTPUT_BASE"/mosi_alpha*_seed${SEED}; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    acc=$(grep "New best model" "$dir/train.log" 2>/dev/null | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
    echo "  $name: $acc"
done
