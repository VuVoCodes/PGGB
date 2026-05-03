#!/bin/bash
# T2 sweep: Boost composition with {G-Blend, AGM, MMPareto, CGGM} on CREMA-D 3-frame.
# 4 methods x 5 seeds = 20 runs. Matches Table 1 Boost+OGM-GE setup exactly
# (alpha=0.75, s_max=2.0, K=20, mu=0.3, eval interval, split-batch probe protocol).

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

SEEDS=(42 123 456 789 1024)
METHODS=(gblend agm mmpareto cggm)
OUTDIR="outputs/sweep_boost_compose"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/sweep.log"

echo "Start: $(date)" | tee -a "$LOGFILE"
for METHOD in "${METHODS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    EXPNAME="boost_${METHOD}_seed${SEED}"
    OUTPATH="$OUTDIR/$EXPNAME"
    if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
      echo "[skip] $EXPNAME already complete" | tee -a "$LOGFILE"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Running $EXPNAME" | tee -a "$LOGFILE"
    python scripts/train.py \
      --config configs/cremad.yaml \
      --mode "$METHOD" \
      --boost-compose \
      --continuous-alpha 0.75 \
      --num-frames 3 --fps 3 \
      --seed "$SEED" \
      --exp-name "$EXPNAME" \
      --output-dir "$OUTDIR" \
      > "$OUTDIR/${EXPNAME}.stdout" 2>&1
    ACC=$(grep "Training complete" "$OUTPATH/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
    echo "[$(date +%H:%M:%S)] $EXPNAME done, acc=$ACC" | tee -a "$LOGFILE"
  done
done
echo "End: $(date)" | tee -a "$LOGFILE"
