#!/bin/bash
# retry_bad_cells_after_chain.sh — wait for tier-2 chain (decompose + f=2) to
# finish, then delete cell_*.json files with N_D=null (the historical OMP=32
# bad rows) and re-run decompose_parallel to re-fill them. Should add ~10 min.

set -u
CHAIN_PID=${CHAIN_PID:-1337208}
TIER2_OUT=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data/sweep_zeta9_tier2_eps1e-4_2026-05-23
ROOT=/home/hlamm/Desktop/efficent_gates/unified

echo "[retry-bad] $(date +%H:%M:%S) waiting for chain PID $CHAIN_PID to exit..."
while kill -0 $CHAIN_PID 2>/dev/null; do
    sleep 60
done
echo "[retry-bad] $(date +%H:%M:%S) chain exited; identifying bad cells"

bad=0
for f in $TIER2_OUT/cell_*.json; do
    val=$(jq -r ".decomposition.N_D" "$f" 2>/dev/null)
    if [ "$val" = "null" ]; then
        rm "$f"
        bad=$((bad+1))
    fi
done
echo "[retry-bad] deleted $bad bad cell files"

echo "[retry-bad] $(date +%H:%M:%S) re-running decompose_parallel"
export OMP_NUM_THREADS=1
python3 $ROOT/decompose_parallel.py \
    --out_dir $TIER2_OUT \
    --n_thetas 10000 \
    --theta_min 0.0 \
    --theta_max 6.283185307179586 \
    --eps 1e-4 \
    --eps_pre 5e-5 \
    --max_f 2 \
    --workers 32 \
    > $TIER2_OUT/decompose_retry.log 2>&1
echo "[retry-bad] $(date +%H:%M:%S) DONE"

# Refresh CSV
ls $TIER2_OUT/cell_*.json | wc -l
