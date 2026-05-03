#!/bin/bash
# 3-way ablation: Monitor+OGM-GE (α=0, passive monitor) on CREMA-D 3-frame.
#
# Purpose: Isolate the monitor's marginal value from the boost's by running
# the probe monitoring protocol with zero boost actuation (continuous-alpha=0).
# This complements existing OGM-GE (no probes) and Boost+OGM-GE (α=0.75) runs
# already in outputs/sweep_3f/ and outputs/sweep_cremad to form the 3-way:
#
#   OGM-GE alone        (existing: sweep_3f/3f_ogmge_seed{42,123,456,789,1024})
#   Monitor+OGM-GE α=0  (new, this script)
#   Boost+OGM-GE α=0.75 (existing: sweep_3f/3f_boost_ogm_a075_seed{42,123,456,789,1024})
#
# Output: outputs/sweep_3way_ablation/ (separate dir — does NOT override any existing results)

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

SEEDS=(42 123 456 789 1024)
OUTDIR="outputs/sweep_3way_ablation"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/sweep.log"

# Safety: abort if any target name exists OUTSIDE our own output dir
OUTDIR_ABS="/home/main/AsynchronousFunction/$OUTDIR"
for SEED in "${SEEDS[@]}"; do
  EXPNAME="monitor_ogm_noboost_seed${SEED}"
  COLLISIONS=$(find /home/main/AsynchronousFunction/outputs -maxdepth 3 -type d -name "$EXPNAME" 2>/dev/null | grep -Fv "$OUTDIR_ABS/" || true)
  if [ -n "$COLLISIONS" ]; then
    echo "ABORT: Name collision for $EXPNAME outside $OUTDIR_ABS:" >&2
    echo "$COLLISIONS" >&2
    exit 1
  fi
done

echo "Start: $(date)" | tee -a "$LOGFILE"
echo "3-way ablation: Monitor+OGM-GE (continuous-alpha=0, passive monitor)" | tee -a "$LOGFILE"
echo "Config: CREMA-D 3-frame, mode=adaptive, asgml-mode=continuous, --ogm-ge --alpha 0.8, --continuous-alpha 0.0" | tee -a "$LOGFILE"
echo "Seeds: ${SEEDS[*]}" | tee -a "$LOGFILE"
echo "Output: $OUTDIR" | tee -a "$LOGFILE"
echo "---" | tee -a "$LOGFILE"

for SEED in "${SEEDS[@]}"; do
  EXPNAME="monitor_ogm_noboost_seed${SEED}"
  OUTPATH="$OUTDIR/$EXPNAME"
  if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
    echo "[skip] $EXPNAME already complete" | tee -a "$LOGFILE"
    continue
  fi
  echo "[$(date +%H:%M:%S)] Running $EXPNAME" | tee -a "$LOGFILE"
  python scripts/train.py \
    --config configs/cremad.yaml \
    --mode adaptive \
    --asgml-mode continuous \
    --ogm-ge --alpha 0.8 \
    --continuous-alpha 0.0 \
    --num-frames 3 --fps 3 \
    --seed "$SEED" \
    --exp-name "$EXPNAME" \
    --output-dir "$OUTDIR" \
    > "$OUTDIR/${EXPNAME}.stdout" 2>&1
  ACC=$(grep "Training complete" "$OUTPATH/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
  echo "[$(date +%H:%M:%S)] $EXPNAME done, acc=$ACC" | tee -a "$LOGFILE"
done
echo "End: $(date)" | tee -a "$LOGFILE"

echo ""
echo "Aggregating results..."
python -c "
import re, glob
accs = []
for f in sorted(glob.glob('$OUTDIR/monitor_ogm_noboost_seed*/train.log')):
    m = re.search(r'Training complete.*Best accuracy:\s*([\d.]+)', open(f).read())
    if m:
        accs.append(float(m.group(1)))
        print(f'{f.split(\"/\")[-2]}: {float(m.group(1))*100:.2f}%')
if accs:
    import statistics
    print(f'---')
    print(f'N={len(accs)}, mean={statistics.mean(accs)*100:.2f}%, std={statistics.stdev(accs)*100:.2f}pp' if len(accs) > 1 else f'N=1, mean={accs[0]*100:.2f}%')
" | tee -a "$LOGFILE"
