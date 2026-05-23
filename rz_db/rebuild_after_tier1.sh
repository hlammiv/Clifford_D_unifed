#!/bin/bash
# rebuild_after_tier1.sh — pull tier-1 densification CSV from Lenore,
# rebuild R_z DB, rebuild tier-0 U-net with lazy fallback.
#
# Run AFTER the Lenore HRSA tier-1 densification finishes.
# Prereq: Lenore SSH access + worker_*/summary.csv files merged into one
#         summary.csv on Lenore (the launcher's `Merge with:` instructions).

set -eu
ROOT=/home/hlamm/Desktop/efficent_gates/unified
DENSE_DIR=$ROOT/sweep_hrsa_dense_eps1e-3_2026-05-23

mkdir -p $DENSE_DIR
echo "[step 1/4] pull tier-1 CSV from Lenore"
ssh -p 60022 lenore_remote "
  cd /home/hlamm/Desktop/efficent_gates/unified/sweep_hrsa_dense_eps1e-3_2026-05-23 &&
  cat worker_*/summary.csv | awk 'NR==1 || !/^theta_idx,/' > summary.csv &&
  wc -l summary.csv"
scp -q -P 60022 lenore_remote:/home/hlamm/Desktop/efficent_gates/unified/sweep_hrsa_dense_eps1e-3_2026-05-23/summary.csv \
    $DENSE_DIR/summary.csv

echo "[step 2/4] rebuild R_z DB from scratch"
cd $ROOT
python3 rz_db/build_rz_db.py --db /tmp/rz_test.sqlite --rebuild \
    sweep_zeta9_cal_2026-05-22/summary.csv \
    sweep_hrsa_grid_2026-05-22/summary.csv \
    sweep_hrsa_dense_eps1e-3_2026-05-23/summary.csv \
    /tmp/lenore_f4_eps1e-4.csv \
    /tmp/lenore_f4_eps1e-3.csv 2>&1 | tail -5
python3 -c "from rz_db.rz_lookup import RzLookupDB; print(RzLookupDB('/tmp/rz_test.sqlite').stats())" | python3 -c "import sys,ast,json; s=ast.literal_eval(sys.stdin.read()); print('total:',s['total']); print('tiers:',{e:t['n'] for e,t in s['eps_tiers'].items()})"

echo "[step 3/4] rebuild tier-0 u-net WITH lazy fallback (will fill DB gaps)"
python3 -c "
from u_net.u_net_builder import build_u_net
m = build_u_net(50, 0.5, '/tmp/rz_test.sqlite', '/tmp/tier_05.h5',
                allow_live_fallback=True, log_miss_rate=True)
print('n_persisted:', m['n_persisted'])
print('miss_rate:', m['miss_rate'])
print('build_time_s:', m['build_time_s'])
"

echo "[step 4/4] verify tier-0 covers SU(3)"
python3 -c "
from u_net.u_net_lookup import UNetLookup
from u_net.haar_sampler import haar_su3
import numpy as np
u = UNetLookup('/tmp/tier_05.h5')
print('stats:', u.stats())
# 100-sample coverage
S = haar_su3(100, seed=20260524)
dists = [u.closest(s)['distance'] for s in S]
print(f'coverage: min={min(dists):.3f}, median={np.median(dists):.3f}, max={max(dists):.3f}')
"

echo "Done. Tier-0 net is at /tmp/tier_05.h5."
echo "Lazy fallback may have inserted new R_z rows; commit /tmp/rz_test.sqlite stats:"
python3 -c "from rz_db.rz_lookup import RzLookupDB; print(RzLookupDB('/tmp/rz_test.sqlite').count(), 'rows')"
