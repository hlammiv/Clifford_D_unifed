#!/bin/bash
# launch_hrsa_cascade_parallel.sh — local parallel HRSA cascade sweep.
#
# Sweeps 100 θ × 7 ε = 700 cells across N_WORKERS parallel shards (each
# worker handles 100/N_WORKERS θ values). Each worker invokes
# sweep_hrsa_grid.py with --use-bidir 4 so the full cascade Direct →
# SignExt → bidir K=2..4 → HRSA(max_f=3) runs per cell.
#
# Per-cell RAM: ~1-2 GB during bidir K=4 phase, sub-GB during HRSA.
# 4 workers fits in lucia's 8 GB free comfortably; 6 is tight; 8 OOM-risk.

set -u
ROOT=/home/hlamm/Desktop/efficent_gates/unified
OUT=$ROOT/sweep_hrsa_cascade_2026-05-23
N_THETAS=100
N_WORKERS=${N_WORKERS:-4}
EPS_LIST="0.5,0.4,0.35,0.3,0.2,0.15,0.1"
MAX_F=3
USE_BIDIR=4
TIMEOUT=900
PI2=6.283185307179586

mkdir -p $OUT
cd $ROOT

echo "[cascade-parallel] $N_WORKERS workers × ~$((N_THETAS / N_WORKERS)) θ each"
echo "[cascade-parallel] ε grid: $EPS_LIST  (7 values × 100 θ = 700 cells)"
echo "[cascade-parallel] out_dir=$OUT"

for w in $(seq 0 $((N_WORKERS - 1))); do
    lo=$(( w * N_THETAS / N_WORKERS ))
    hi=$(( (w+1) * N_THETAS / N_WORKERS ))
    [ $hi -le $lo ] && continue
    nth=$((hi - lo))
    th_min=$(awk "BEGIN {printf \"%.10f\", $lo * $PI2 / $N_THETAS}")
    th_max=$(awk "BEGIN {printf \"%.10f\", $hi * $PI2 / $N_THETAS}")
    wdir=$OUT/worker_$(printf "%02d" $w)
    mkdir -p $wdir
    nohup python3 sweep_hrsa_grid.py \
        --n_thetas $nth \
        --theta_min $th_min \
        --theta_max $th_max \
        --max_f $MAX_F \
        --eps $EPS_LIST \
        --timeout $TIMEOUT \
        --use-bidir $USE_BIDIR \
        --out_dir $wdir \
        > $wdir/sweep.log 2>&1 &
    echo "[cascade-parallel] worker $w PID=$!  θ-idx [$lo,$hi) range [$th_min, $th_max)"
done

echo "[cascade-parallel] all $N_WORKERS workers launched at $(date +%H:%M:%S)"
echo "[cascade-parallel] waiting for completion..."
wait
echo "[cascade-parallel] all workers exited at $(date +%H:%M:%S); merging"
cat $OUT/worker_*/summary.csv | awk 'NR==1 || !/^theta_idx,/' > $OUT/summary.csv
total=$(($(wc -l < $OUT/summary.csv) - 1))
pass=$(awk -F, 'NR>1 && $NF~/true/' $OUT/summary.csv | wc -l)
echo "[cascade-parallel] DONE: $total cells, $pass passes"
