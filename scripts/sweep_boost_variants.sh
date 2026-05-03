#!/bin/bash
# Trial sweep: 4 boost-formula variants on CREMA-D 3-frame, seed 42 only.
# Reference (current paper): "linear" boost+OGM-GE seed 42 = 71.37 (from sweep_3f).
# Goal: scoping study for §5 future-work discussion. NOT a benchmark replacement.

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

OUTDIR="outputs/sweep_boost_variants"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/sweep.log"

# Note: skip "linear" since it equals current default = sweep_3f/3f_boost_ogm_a075_seed42 (71.37)
VARIANTS=(tanh squared sigmoid adaptive)

echo "Start: $(date)" | tee -a "$LOGFILE"
echo "Boost-formula variant trial: 4 variants × 1 seed (42) on CREMA-D 3-frame" | tee -a "$LOGFILE"
echo "Reference: linear (current default) = 71.37 acc seed 42" | tee -a "$LOGFILE"
echo "---" | tee -a "$LOGFILE"

for V in "${VARIANTS[@]}"; do
  EXPNAME="trial_boost_${V}_seed42"
  OUTPATH="$OUTDIR/$EXPNAME"
  if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
    echo "[skip] $EXPNAME already complete" | tee -a "$LOGFILE"
    continue
  fi
  echo "[$(date +%H:%M:%S)] Running $EXPNAME" | tee -a "$LOGFILE"
  python scripts/boost_variant_trial.py \
    --variant "$V" --seed 42 \
    --output-dir "$OUTDIR" \
    > "$OUTDIR/${EXPNAME}.stdout" 2>&1
  ACC=$(grep "Training complete" "$OUTPATH/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
  echo "[$(date +%H:%M:%S)] $EXPNAME done, acc=$ACC" | tee -a "$LOGFILE"
done
echo "End: $(date)" | tee -a "$LOGFILE"

echo ""
echo "=== Summary ===" | tee -a "$LOGFILE"
echo "Reference: linear = 71.37%" | tee -a "$LOGFILE"
for V in "${VARIANTS[@]}"; do
  ACC=$(grep "Training complete" "$OUTDIR/trial_boost_${V}_seed42/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
  if [ -n "$ACC" ]; then
    PCT=$(python -c "print(f'{float('$ACC')*100:.2f}')")
    echo "  $V: $PCT%" | tee -a "$LOGFILE"
  fi
done
