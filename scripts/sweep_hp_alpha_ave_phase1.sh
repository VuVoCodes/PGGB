#!/bin/bash
# Phase 1 HP search: coarse alpha sweep on AVE for non-OGM-GE compositions.
# Goal: identify whether higher alpha pushes Boost+MMPareto/AGM toward +2 pp,
# or whether any alpha rescues Boost+G-Blend (currently regresses at alpha=0.75).
#
# Methods: MMPareto, AGM, G-Blend (skip CGGM — architectural mismatch swamps any alpha effect)
# Alpha values: 0.25, 0.5, 1.0, 1.25, 1.5 (default was 0.75)
# Seed: 42 only (Phase 1 single-seed coarse grid; Phase 2 5-seed confirm conditional)
# 3 methods × 5 alphas × 1 seed = 15 runs
# Estimated: 15 × ~22 min = ~5.5h

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

ALPHAS=(0.25 0.5 1.0 1.25 1.5)
METHODS=(mmpareto agm gblend)
SEED=42
OUTDIR="outputs/sweep_hp_alpha_ave_phase1"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/sweep.log"

echo "Start: $(date)" | tee -a "$LOGFILE"
echo "Phase 1 HP sweep: alpha grid on AVE, single-seed (=$SEED)" | tee -a "$LOGFILE"
echo "Methods: ${METHODS[*]}" | tee -a "$LOGFILE"
echo "Alphas: ${ALPHAS[*]}" | tee -a "$LOGFILE"
echo "---" | tee -a "$LOGFILE"

for METHOD in "${METHODS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    ALPHA_TAG=$(echo "$ALPHA" | tr '.' 'p')
    EXPNAME="ave_boost_${METHOD}_a${ALPHA_TAG}_seed${SEED}"
    OUTPATH="$OUTDIR/$EXPNAME"
    if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
      echo "[skip] $EXPNAME already complete" | tee -a "$LOGFILE"
      continue
    fi
    echo "[$(date +%H:%M:%S)] Running $EXPNAME (alpha=$ALPHA)" | tee -a "$LOGFILE"
    python scripts/train.py \
      --config configs/ave.yaml \
      --mode "$METHOD" \
      --boost-compose \
      --continuous-alpha "$ALPHA" \
      --seed "$SEED" \
      --epochs 100 \
      --exp-name "$EXPNAME" \
      --output-dir "$OUTDIR" \
      > "$OUTDIR/${EXPNAME}.stdout" 2>&1
    ACC=$(grep "Training complete" "$OUTPATH/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
    echo "[$(date +%H:%M:%S)] $EXPNAME done, alpha=$ALPHA acc=$ACC" | tee -a "$LOGFILE"
  done
done
echo "End: $(date)" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Phase 1 results matrix (seed=$SEED only) ===" | tee -a "$LOGFILE"
python -c "
import re, glob
print(f'{\"Method\":<10} | ' + ' | '.join(f'a={a}' for a in ['0.25','0.5','0.75','1.0','1.25','1.5']))
print('-'*80)
# Reference baseline (alpha=0.75 from earlier sweep)
ref = {'mmpareto': 87.04, 'agm': 86.05, 'gblend': 86.91}  # seed 42 from sweep_boost_compose_ave_food101
for METHOD in ['mmpareto', 'agm', 'gblend']:
    row = [f'{METHOD:<10}']
    for ALPHA in ['0.25','0.5','0.75','1.0','1.25','1.5']:
        if ALPHA == '0.75':
            row.append(f'{ref[METHOD]:.2f}*')
            continue
        atag = ALPHA.replace('.','p')
        f = f'$OUTDIR/ave_boost_{METHOD}_a{atag}_seed${SEED}/train.log'
        try:
            txt = open(f).read()
            m = re.search(r'Training complete.*Best accuracy:\s*([\d.]+)', txt)
            row.append(f'{float(m.group(1))*100:.2f}' if m else 'N/A')
        except: row.append('N/A')
    print(' | '.join(row))
print('* alpha=0.75 reference from sweep_boost_compose_ave_food101 seed 42')
" | tee -a "$LOGFILE"
