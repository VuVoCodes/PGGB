#!/bin/bash
# UPMC-Food101 full sweep: 8 methods × 5 seeds = 40 runs
# Expected runtime: ~3-5 hours (MLP encoders on pre-extracted features)
set -e

source activate phd 2>/dev/null || true
cd /home/main/AsynchronousFunction

CONFIG=configs/food101.yaml
OUT=outputs/sweep_food101
PREFIX=food101

mkdir -p $OUT

for seed in 42 123 456 789 1024; do
    echo "=== $PREFIX baseline seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode baseline --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_baseline_seed${seed} 2>&1 | grep "Training complete"

    echo "=== $PREFIX ogm_ge seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode baseline --ogm-ge --alpha 0.8 --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_ogm_ge_seed${seed} 2>&1 | grep "Training complete"

    echo "=== $PREFIX cggm seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode cggm --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_cggm_seed${seed} 2>&1 | grep "Training complete"

    echo "=== $PREFIX mmpareto seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode mmpareto --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_mmpareto_seed${seed} 2>&1 | grep "Training complete"

    echo "=== $PREFIX agm seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode agm --agm-alpha 1.0 --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_agm_seed${seed} 2>&1 | grep "Training complete"

    echo "=== $PREFIX gblend seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode gblend --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_gblend_seed${seed} 2>&1 | grep "Training complete"

    echo "=== $PREFIX boost_only seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode adaptive --asgml-mode continuous --continuous-alpha 0.5 --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_boost_only_seed${seed} 2>&1 | grep "Training complete"

    echo "=== $PREFIX boost_ogm_a075 seed=$seed ==="
    python scripts/train.py --config $CONFIG --mode adaptive --asgml-mode continuous --continuous-alpha 0.75 --ogm-ge --alpha 0.8 --seed $seed \
        --output-dir $OUT --exp-name ${PREFIX}_boost_ogm_a075_seed${seed} 2>&1 | grep "Training complete"
done

echo "ALL FOOD101 DONE"
