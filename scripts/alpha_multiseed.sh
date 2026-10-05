#!/bin/bash
# Multi-seed alpha sensitivity on the sentiment benchmarks (promised to Reviewer gN93, Q4).
# Boost-only, matching the single-seed protocol in rebuttal_p1_alpha_{chsims,mosi}.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# activate your environment here if needed, e.g. conda activate pggb
OUT=outputs/alpha_multiseed; mkdir -p "$OUT"
SEEDS=(42 123 456 789 1024); ALPHAS=(0.25 0.5 0.75 1.0 1.5)
run() {  # dataset_tag config alpha seed
  local tag=$1 cfg=$2 a=$3 s=$4
  local name="${tag}_alpha${a}_seed${s}"
  [ -f "$OUT/$name/train.log" ] && grep -q "Training complete" "$OUT/$name/train.log" 2>/dev/null && { echo "skip $name"; return; }
  echo "[$(date +%H:%M:%S)] start $name"
  python scripts/train.py --config "$cfg" --mode adaptive --seed "$s" \
    --asgml-mode continuous --continuous-alpha "$a" --epochs 100 \
    --exp-name "$name" --output-dir "$OUT" > "$OUT/${name}.stdout" 2>&1
  local acc; acc=$(grep "Training complete" "$OUT/$name/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
  echo "[$(date +%H:%M:%S)] done $name acc=$acc"
}
for a in "${ALPHAS[@]}"; do for s in "${SEEDS[@]}"; do
  run chsims configs/chsims.yaml "$a" "$s"
  run mosi   configs/mosi.yaml  "$a" "$s"
done; done
echo "ALL DONE"
