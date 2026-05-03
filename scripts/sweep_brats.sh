#!/bin/bash
set -uo pipefail
PYTHON="/home/main/miniconda3/envs/phd/bin/python"
TRAIN="scripts/train_brats.py"
OUT="outputs/sweep_brats"
mkdir -p "$OUT"

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

phase1() {
    local S=42
    echo "=== PHASE 1: BraTS single-seed ==="
    run "brats_baseline_seed${S}" --mode baseline --seed $S --epochs 100 --batch-size 12
    run "brats_asgml_seed${S}" --mode asgml_boost --seed $S --epochs 100 --batch-size 12 --boost-alpha 0.5
    run "brats_cggm_seed${S}" --mode cggm --seed $S --epochs 100 --batch-size 12 --cggm-rou 1.3 --cggm-lamda 0.2

    echo "=== RESULTS ==="
    for dir in "$OUT"/brats_*_seed${S}; do
        [ -d "$dir" ] && name=$(basename "$dir") && dice=$(grep "New best model" "$dir/train.log" 2>/dev/null | tail -1 | grep -oP 'dice: \K[0-9.]+') && echo "  $name: $dice"
    done
}

case "${1:-phase1}" in
    phase1) phase1 ;;
    *) echo "Usage: $0 {phase1}"; exit 1 ;;
esac
