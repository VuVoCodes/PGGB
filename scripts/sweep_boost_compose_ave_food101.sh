#!/bin/bash
# Boost composition sweep on AVE + Food101: tests whether Boost+{non-OGM-GE}
# generalizes beyond CREMA-D. Addresses reviewer "Accept-bar" concern:
#   "non-OGM-GE backbone showing similar gain, OR second high-imbalance
#    dataset replicating +2.31 pp."
#
# Methods: MMPareto, AGM, G-Blend, CGGM (four non-OGM-GE throttlers, same
# composition protocol as Table 8 CREMA-D sweep).
# Datasets: AVE (2 modalities, audio dominant) + Food101 (text+image, frozen)
# Seeds: 42, 123, 456, 789, 1024 (same as existing sweeps)
#
# 4 methods × 2 datasets × 5 seeds = 40 runs
# Estimated runtime: AVE ~22 min/run (7.3h), Food101 ~29 min/run (9.7h). Total ~17h.
#
# Output: outputs/sweep_boost_compose_ave_food101/ (separate — no override risk)

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

SEEDS=(42 123 456 789 1024)
METHODS=(mmpareto agm gblend cggm)
DATASETS=(ave food101)
OUTDIR="outputs/sweep_boost_compose_ave_food101"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/sweep.log"

# Safety: abort if any target name already exists OUTSIDE our output dir
OUTDIR_ABS="/home/main/AsynchronousFunction/$OUTDIR"
for DS in "${DATASETS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      EXPNAME="${DS}_boost_${METHOD}_seed${SEED}"
      COLLISIONS=$(find /home/main/AsynchronousFunction/outputs -maxdepth 3 -type d -name "$EXPNAME" 2>/dev/null | grep -Fv "$OUTDIR_ABS/" || true)
      if [ -n "$COLLISIONS" ]; then
        echo "ABORT: Name collision for $EXPNAME outside $OUTDIR_ABS:" >&2
        echo "$COLLISIONS" >&2
        exit 1
      fi
    done
  done
done

echo "Start: $(date)" | tee -a "$LOGFILE"
echo "Boost composition sweep: 4 methods × 2 datasets × 5 seeds = 40 runs" | tee -a "$LOGFILE"
echo "Methods: ${METHODS[*]}" | tee -a "$LOGFILE"
echo "Datasets: ${DATASETS[*]}" | tee -a "$LOGFILE"
echo "Seeds: ${SEEDS[*]}" | tee -a "$LOGFILE"
echo "---" | tee -a "$LOGFILE"

# Food101 (faster, run first to get initial signal)
for DS in food101 ave; do
  CONFIG="configs/${DS}.yaml"
  EXTRA_CONFIG_ARGS=""
  if [ "$DS" = "ave" ]; then
    EXTRA_CONFIG_ARGS="--epochs 100"
  fi
  for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      EXPNAME="${DS}_boost_${METHOD}_seed${SEED}"
      OUTPATH="$OUTDIR/$EXPNAME"
      if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
        echo "[skip] $EXPNAME already complete" | tee -a "$LOGFILE"
        continue
      fi
      echo "[$(date +%H:%M:%S)] Running $EXPNAME" | tee -a "$LOGFILE"
      python scripts/train.py \
        --config "$CONFIG" \
        --mode "$METHOD" \
        --boost-compose \
        --continuous-alpha 0.75 \
        --seed "$SEED" \
        --exp-name "$EXPNAME" \
        --output-dir "$OUTDIR" \
        $EXTRA_CONFIG_ARGS \
        > "$OUTDIR/${EXPNAME}.stdout" 2>&1
      ACC=$(grep "Training complete" "$OUTPATH/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
      echo "[$(date +%H:%M:%S)] $EXPNAME done, acc=$ACC" | tee -a "$LOGFILE"
    done
  done
done
echo "End: $(date)" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Aggregating results ===" | tee -a "$LOGFILE"
python -c "
import re, glob, statistics
for DS in ['food101', 'ave']:
    print(f'=== {DS.upper()} ===')
    for METHOD in ['mmpareto', 'agm', 'gblend', 'cggm']:
        accs = []
        for f in sorted(glob.glob(f'$OUTDIR/{DS}_boost_{METHOD}_seed*/train.log')):
            m = re.search(r'Training complete.*Best accuracy:\s*([\d.]+)', open(f).read())
            if m: accs.append(float(m.group(1)))
        if len(accs) >= 2:
            print(f'  Boost+{METHOD}: N={len(accs)}, mean={statistics.mean(accs)*100:.2f}%, std={statistics.stdev(accs)*100:.2f}pp')
        elif accs:
            print(f'  Boost+{METHOD}: N={len(accs)}, mean={accs[0]*100:.2f}%')
        else:
            print(f'  Boost+{METHOD}: N=0 (incomplete)')
" | tee -a "$LOGFILE"
