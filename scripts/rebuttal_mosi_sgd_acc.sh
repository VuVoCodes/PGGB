#!/bin/bash
# MOSI-SGD accuracy control (miLe W3, discussion window).
# Decisive test: does restored actuation under SGD buy an accuracy gain that Adam attenuates?
# 2x2 design, all arms composed (OGM-GE alpha=0.8), continuous-alpha 0.75 vs 0.0:
#   - SGD  a0 / a075  (configs/mosi_sgd.yaml, new)
#   - Adam a0         (configs/mosi.yaml, new; Adam a075 = existing mosi_boost_ogm runs)
# PRE-REGISTRATION: seeds fixed in advance as 42 123 456 789 1024 (the paper's MOSI seeds,
# matched to the existing Adam runs). All runs reported regardless of outcome.
# Mirrors the arm design of scripts/rebuttal_n15.sh and the instrumentation in
# outputs/rebuttal_sgd_control/norms (which was composed, alpha 0.8, CA 0.75 vs 0.0).

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# activate your environment here if needed, e.g. conda activate pggb

OUTDIR="outputs/rebuttal_mosi_sgd_acc"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/mosi_sgd_acc.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

SEEDS=(42 123 456 789 1024)

run_arm() {
  local CONFIG="$1" TAG="$2" CA="$3" SEED="$4"
  local EXPNAME="${TAG}_seed${SEED}"
  local OUTPATH="$OUTDIR/$EXPNAME"
  if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
    log "skip $EXPNAME (already complete)"
    return
  fi
  log "start $EXPNAME"
  python scripts/train.py \
    --config "$CONFIG" \
    --mode adaptive --asgml-mode continuous \
    --ogm-ge --alpha 0.8 \
    --continuous-alpha "$CA" \
    --epochs 100 \
    --seed "$SEED" \
    --exp-name "$EXPNAME" \
    --output-dir "$OUTDIR" \
    > "$OUTDIR/${EXPNAME}.stdout" 2>&1
  local ACC
  ACC=$(grep "New best model" "$OUTPATH/train.log" 2>/dev/null | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
  log "$EXPNAME done, best_acc=$ACC"
}

for SEED in "${SEEDS[@]}"; do
  run_arm configs/mosi_sgd.yaml sgd_a0   0.0  "$SEED"
  run_arm configs/mosi_sgd.yaml sgd_a075 0.75 "$SEED"
  run_arm configs/mosi.yaml     adam_a0  0.0  "$SEED"
done

log "all arms complete"
log "===== SUMMARY (best test accuracy per run) ====="
for dir in "$OUTDIR"/sgd_a0_seed* "$OUTDIR"/sgd_a075_seed* "$OUTDIR"/adam_a0_seed*; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir")
  acc=$(grep "New best model" "$dir/train.log" 2>/dev/null | tail -1 | grep -oP 'accuracy: \K[0-9.]+')
  log "  $name: $acc"
done
