#!/bin/bash
# rebuild_after_tier2.sh — analog of rebuild_after_tier1.sh.
# Pull tier-2 (zeta9 ε=10⁻⁴, ~10K θ) summary CSV + per-cell JSONs from
# Lenore, rebuild R_z DB, rebuild U-nets across the scaffold tier ladder.
#
# Run AFTER the Lenore zeta9 batched tier-2 sweep finishes.
# Prereq: launch_tier2_densify.sh already ran and wrote summary.csv +
#         per-cell JSONs to the data partition.

set -eu
ROOT=/home/hlamm/Desktop/efficent_gates/unified
LENORE_OUT=/mnt/993c1724-f80f-4440-a384-daf788d9a041/data/sweep_zeta9_tier2_eps1e-4_2026-05-23
LOCAL_DIR=$ROOT/sweep_zeta9_tier2_eps1e-4_2026-05-23

mkdir -p $LOCAL_DIR
echo "[step 1/4] pull tier-2 CSV + per-cell JSONs from Lenore"
scp -q -P 60022 lenore_remote:$LENORE_OUT/summary.csv $LOCAL_DIR/summary.csv
# Tarball the per-cell JSONs (sweep_zeta9_batched writes them flat alongside summary.csv).
ssh -p 60022 lenore_remote "
  cd $LENORE_OUT && tar -czf /tmp/tier2_jsons.tar.gz zeta9_maxf*.json 2>/dev/null
  ls -la /tmp/tier2_jsons.tar.gz
"
scp -q -P 60022 lenore_remote:/tmp/tier2_jsons.tar.gz /tmp/
tar -xzf /tmp/tier2_jsons.tar.gz -C $LOCAL_DIR
echo "  pulled $(ls $LOCAL_DIR/zeta9_maxf*.json 2>/dev/null | wc -l) per-cell JSONs"

echo "[step 2/4] rebuild R_z DB from scratch (include tier-2)"
cd $ROOT
python3 rz_db/build_rz_db.py --db /tmp/rz_test.sqlite --rebuild \
    sweep_zeta9_cal_2026-05-22/summary.csv \
    sweep_hrsa_grid_2026-05-22/summary.csv \
    sweep_hrsa_dense_eps1e-3_2026-05-23/summary.csv \
    sweep_zeta9_tier2_eps1e-4_2026-05-23/summary.csv \
    /tmp/lenore_f4_eps1e-4.csv \
    /tmp/lenore_f4_eps1e-3.csv 2>&1 | tail -10
python3 -c "
from rz_db.rz_lookup import RzLookupDB
s = RzLookupDB('/tmp/rz_test.sqlite').stats()
print('total:', s['total'])
for e, t in sorted(s['eps_tiers'].items()):
    print(f'  eps={e}: n={t[\"n\"]:>5}  range=[{t[\"theta_min\"]:.4f}, {t[\"theta_max\"]:.4f}]')
"

echo "[step 3/4] rebuild scaffold U-net tiers (adaptive eps_leaf landed)"
python3 -c "
from u_net.u_net_builder import build_u_net
# Decade-spaced tiers per sk_rz_db_lazy_population ladder.
# allow_live_fallback=True: misses at tight eps will trigger live HRSA/zeta9.
for eps_u, n in [(0.5, 200), (0.1, 200), (0.01, 200), (1e-3, 100)]:
    out = f'/tmp/tier_eps{eps_u}.h5'
    m = build_u_net(n, eps_u, '/tmp/rz_test.sqlite', out,
                    allow_live_fallback=True, log_miss_rate=True, seed=20260523)
    print(f'tier eps_U={eps_u}: persisted {m.get(\"n_persisted\")}/{n}, miss_rate={m.get(\"miss_rate\")}')
"

echo "[step 4/4] verify scaffolded SK can load all tiers"
python3 -c "
from u_net.scaffolded_net import ScaffoldedNet
import glob
tiers = sorted(glob.glob('/tmp/tier_eps*.h5'))
s = ScaffoldedNet(tiers)
print(f'ScaffoldedNet loaded {len(s)} tiers: {s.all_tiers()}')
"

echo "Done. Scaffolded SK is now buildable across the full ladder."
echo "Next: pytest hrsa/test_sk_driver_scaffolded.py (smoke) + run E2E demo."
