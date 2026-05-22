#!/usr/bin/env python3
# calibrate_direct.py — measure the empirical covering radius δ_k for
# qutrit-Clifford+D directSearch at depth k=0,1,2.
#
# Strategy: run HRSA_tester at a large epsilon (1.0) so directSearch
# logs every k="best=" line without short-circuiting on success.
# For each θ, parse out best_0, best_1, best_2.  δ_k = max over θ of best_k.

import math
import re
import subprocess
import sys
from pathlib import Path

HRSA = Path("/home/hlamm/Desktop/efficent_gates/unified/hrsa/HRSA_tester")
EPS = 1e-9      # tiny ε so directSearch NEVER finds, runs through all k=0,1,2,3
MAX_DIRECT = 3  # sweep through k=0,1,2,3

# Use a fine grid avoiding θ=0 (Clifford trivial; we know best=0 at k=0).
N = 25
ANGLES = [i * 2.0 * math.pi / N for i in range(1, N)]  # 24 angles for the diagnostic

best_re = re.compile(r"k=(\d+): best=([\d.eE+-]+)")

results = []  # list of (theta, [best_0, best_1, best_2, best_3])
for th in ANGLES:
    cmd = [str(HRSA), repr(th), str(EPS), "0",  # max_f=0 so HRSA Phase 3 doesn't run long
           "--max-direct", str(MAX_DIRECT)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    bests = [None, None, None, None]
    for m in best_re.finditer(p.stdout):
        k = int(m.group(1))
        v = float(m.group(2))
        if 0 <= k < 4:
            bests[k] = v
    results.append((th, bests))
    diff32 = (bests[3] - bests[2]) if (bests[3] is not None and bests[2] is not None) else None
    print(f"θ={th:.4f}  best_0={bests[0]}  best_1={bests[1]}  best_2={bests[2]}  best_3={bests[3]}  k3-k2={diff32}", flush=True)

print()
print("# Empirical covering radii δ_k = max over θ of best_k:")
for k in range(4):
    vals = [r[1][k] for r in results if r[1][k] is not None]
    if vals:
        print(f"#   δ_{k} = {max(vals):.6f}   (over {len(vals)} angles)")
    else:
        print(f"#   δ_{k} = (no data)")
print()
print("# k=3 vs k=2 strict improvement count:")
strict = sum(1 for r in results if r[1][3] is not None and r[1][2] is not None and r[1][3] < r[1][2] * 0.9999)
total = sum(1 for r in results if r[1][3] is not None and r[1][2] is not None)
print(f"#   {strict} / {total} angles where best_3 < 0.9999 * best_2")
