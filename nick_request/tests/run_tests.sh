#!/usr/bin/env bash
# Unit tests for topk_dump.patch (synthetic exact candidates; no D/ data needed).
# Requires the zeta9 clone on branch topk-dump and the sage conda env (mpi4py, Sage for MatrixC6).
set -e
export PATH=$HOME/miniforge3/envs/sage/bin:$PATH
W=$(mktemp -d); cp "$(dirname "$0")"/*.py "$W"; cd "$W"
mkdir -p origpkg/zeta9
cp /home/hlamm/Desktop/efficent_gates/zeta9/zeta9/{tools.py,__init__.py} origpkg/zeta9/
git -C /home/hlamm/Desktop/efficent_gates/zeta9 show main:zeta9/fit_theta_all_best_one_theta.py > origpkg/zeta9/orig_fit.py
python3 gen_exact.py && python3 make_pool.py
python3 test_patch.py 2>&1 | grep -v Warn
python3 test_dumptopk.py 2>&1 | grep -v Warn
python3 test_regress.py 2>&1 | grep -v Warn
echo "work dir: $W"
