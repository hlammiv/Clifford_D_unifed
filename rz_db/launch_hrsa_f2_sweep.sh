#!/bin/bash
# launch_hrsa_f2_sweep.sh — Lenore queue: wait for tier-2, then HRSA(max_f=2) sweep
#
# Purpose: characterize the f=2 envelope of HRSA. With --max-solns 20 (i.e.
# HRSA_bestD) we get the min-D-count solution at f=2 for each (θ, ε) cell.
#
# Three things this gives us:
#   1) Tighter N_D for cells where the existing sweeps used max_f=3 but f=2 suffices
#   2) Clean "f=2 capability envelope" for the paper (cost-of-precision per f-level)
#   3) At what ε does f=2 stop passing — the structural break to f=3
#
# Queues after tier-2 (PID 1191071) by waiting for it to exit, then runs 32
# parallel worker shards on the 100-θ grid × 6 ε grid. Same sharding pattern
# as the tier-1 densification sweep.

set -u
ROOT=/home/hlamm/Desktop/efficent_gates/unified
DATA_ROOT=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data
OUT=$DATA_ROOT/sweep_hrsa_f2_2026-05-23
N_THETAS=100
N_WORKERS=32
EPS_LIST="0.5,0.1,0.05,0.01,0.005,0.001"
MAX_F=2
TIMEOUT=300
mkdir -p $OUT

# --- Wait for tier-2 to free Lenore ---
TIER2_PID=1191071
echo "[f2-launch] waiting for tier-2 PID $TIER2_PID to exit..."
while kill -0 $TIER2_PID 2>/dev/null; do
    sleep 60
done
echo "[f2-launch] tier-2 exited at $(date +%H:%M:%S); launching f=2 sweep"

# --- Shard 100 θ over 32 workers ---
# Each worker covers a slice [theta_min, theta_max) using sweep_hrsa_grid.py.
# Slice boundaries: theta_w = w * 2π / 100 for w in 0..100, slice [a,b] of θ-indices
# maps to theta_min = a*2π/100, theta_max = b*2π/100, n_thetas = b-a.
cd $ROOT
PI2=6.283185307179586

for w in $(seq 0 $((N_WORKERS - 1))); do
    # Compute θ-index range for this worker
    lo=$(( w * N_THETAS / N_WORKERS ))
    hi=$(( (w+1) * N_THETAS / N_WORKERS ))
    [ $hi -le $lo ] && continue  # skip empty slices
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
        --out_dir $wdir \
        > $wdir/sweep.log 2>&1 &
    echo "[f2-launch] worker $w: PID=$! θ-idx [$lo,$hi) range [$th_min, $th_max)"
done

echo "[f2-launch] all 32 workers launched at $(date +%H:%M:%S)"
echo "[f2-launch] out_dir=$OUT"
echo "[f2-launch] waiting for all workers to complete..."
wait
echo "[f2-launch] all workers exited at $(date +%H:%M:%S); merging CSVs"
cat $OUT/worker_*/summary.csv | awk 'NR==1 || !/^theta_idx,/' > $OUT/summary.csv
pass=$(awk -F, 'NR>1 && $NF~/true/' $OUT/summary.csv | wc -l)
total=$(($(wc -l < $OUT/summary.csv) - 1))
echo "[f2-launch] DONE: $total cells, $pass passes. Summary at $OUT/summary.csv"
