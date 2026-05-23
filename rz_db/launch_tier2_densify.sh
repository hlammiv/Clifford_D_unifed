#!/bin/bash
# launch_tier2_densify.sh — Lenore launcher for SK scaffold tier 2
#
# 10,000 angles uniform in [0, 2π) at ε=10⁻⁴ via zeta9 batched mode.
# ε_leaf budget at tier ε_U=0.01 is ~3×10⁻⁴; this gives slack.
# ETA ~6 hr on Lenore mpi=32 (stage 1 ~30 min, stages 2-4 ~30 min,
# stage 5 batched ~1 sec/cell amortized ~3 hr, decompose ~50 s/cell ~140 min).
#
# Prereq: Lenore code synced to /home/hlamm/Desktop/efficent_gates/unified/
#         (sweep_zeta9_batched.py, zeta9_compile.py, zeta9/zeta9/*.py up to date).
#
# Usage: run this on Lenore:
#   ssh -p 60022 lenore_remote 'bash < /path/to/launch_tier2_densify.sh'
# Or scp and exec locally on Lenore.

set -u
ROOT=/home/hlamm/Desktop/efficent_gates/unified
DATA_ROOT=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data
WORK=$DATA_ROOT/zeta9_workdir_tier2_eps1e-4
OUT=$DATA_ROOT/sweep_zeta9_tier2_eps1e-4_$(date +%Y-%m-%d)
mkdir -p $WORK $OUT

cd $ROOT

# 10K angles uniform in [0, 2π)
ZETA9_NO_OVERSUBSCRIBE=1 nohup python3 sweep_zeta9_batched.py \
    --n_thetas 10000 \
    --theta_min 0.0 \
    --theta_max 6.283185307179586 \
    --max_f 2 \
    --eps 1e-4 \
    --eps_pre 5e-5 \
    --mpi 32 \
    --workdir $WORK \
    --sage_env /home/hlamm/miniforge3/envs/sage \
    --out_dir $OUT \
    > $OUT/sweep.log 2>&1 &

PID=$!
echo "[tier2-launch] PID=$PID"
echo "[tier2-launch] workdir=$WORK"
echo "[tier2-launch] out_dir=$OUT"
echo "[tier2-launch] monitor: tail -f $OUT/sweep.log | grep -E 'stage|cells'"
echo "[tier2-launch] ETA ~6 hr (mostly stage 5 + decompose)."
echo "[tier2-launch] When done, scp $OUT/summary.csv back to lucia and"
echo "[tier2-launch] re-build the R_z DB: python rz_db/build_rz_db.py \\"
echo "[tier2-launch]   --db /tmp/rz_test.sqlite $OUT/summary.csv"
