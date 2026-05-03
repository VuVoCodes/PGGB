#!/bin/bash
set -uo pipefail
PYTHON="/home/main/miniconda3/envs/phd/bin/python"
TRAIN="scripts/train_brats.py"
OUT="outputs/sweep_brats"
mkdir -p "$OUT"

SEEDS=(42 123 456 789 1024)

run() {
    local id="$1"; shift
    local dir="$OUT/$id"
    if [ -d "$dir" ] && grep -q "Training complete" "$dir/train.log" 2>/dev/null; then
        echo "SKIP: $id"; return
    fi
    echo "Running: $id"
    $PYTHON $TRAIN --exp-name "$id" --output-dir "$OUT" "$@"
    echo "DONE: $id"
}

echo "=== BraTS Phase 2: Multi-seed ==="
for seed in "${SEEDS[@]}"; do
    run "brats_baseline_seed${seed}" --mode baseline --seed $seed --epochs 100 --batch-size 12
    run "brats_asgml_seed${seed}" --mode asgml_boost --seed $seed --epochs 100 --batch-size 12 --boost-alpha 0.5
    run "brats_cggm_seed${seed}" --mode cggm --seed $seed --epochs 100 --batch-size 12 --cggm-rou 1.3 --cggm-lamda 0.2
done

echo "=== PHASE 2 COMPLETE ==="
echo "Results:"
for method in brats_baseline brats_asgml brats_cggm; do
    printf "%-20s" "$method"
    for seed in "${SEEDS[@]}"; do
        dir="$OUT/${method}_seed${seed}"
        dice=$(grep "^Test:" "$dir/train.log" 2>/dev/null | grep -oP 'Dice=\K[0-9.]+')
        printf "%8s" "${dice:-—}"
    done
    echo
done
