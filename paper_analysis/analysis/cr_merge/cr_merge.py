"""cr_merge.py -- T count per R after rotation merging for C+R-shaped chains.

In the normal form of (C+R)_3 words (Kalra et al.), consecutive R = diag(1,1,-1) are
separated by X^delta H D, with D a diagonal Clifford.  We emit each R with the 63-gate
7-T gadget (unified/r_from_d/R_7T_compact.json) on one shared clean ancilla, insert a
random normal-form syllable between gadgets, and merge rotations with
unified/depth_optimality/tmerge.merge, which re-verifies the product against the 9x9 unitary.
Result: exactly 5N+2 T for N gadgets (5.0 per R).  Random H/S words in place of the
normal-form syllable merge further (about 4.2 per R); we use the normal form.
Usage: python3 cr_merge.py [n_R] [trials]"""
import json, sys
from pathlib import Path

def _repo_root(start):
    """Directory holding nick_test/: the repository root, or ../unified next to the paper."""
    for p in [start, *start.parents]:
        if (p / "nick_test").is_dir():
            return p
        if (p / "unified" / "nick_test").is_dir():
            return p / "unified"
    raise FileNotFoundError("cannot locate the repository root (directory containing nick_test/)")

import numpy as np
U = _repo_root(Path(__file__).resolve())
sys.path.insert(0, str(U / "depth_optimality"))
from tdepth import parse          # noqa: E402
from tmerge import merge          # noqa: E402





R7 = json.load(open(U / "r_from_d" / "R_7T_compact.json"))["gates"]
n_R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rng = np.random.default_rng(0)
per = []
for t in range(trials):
    gates = []
    for _ in range(n_R):
        syl = ([("X", 0)] * int(rng.integers(0, 3)) + [("H", 0)]
               + [("S", 0)] * int(rng.integers(0, 3)) + [("Z", 0)] * int(rng.integers(0, 3)))
        gates += [parse(g) for g in syl]                 # normal-form syllable X^d H D
        gates += [parse(g) for g in R7]
    r = merge(gates)
    assert r["merge_err"] < 1e-8, r
    per.append(r["T_merged"] / n_R)
per = np.array(per)
print(f"n_R={n_R} trials={trials}: merged T per R = {per.mean():.3f} +- {per.std(ddof=1)/np.sqrt(len(per)):.3f} "
      f"(min {per.min():.2f}, max {per.max():.2f}); unmerged 7")
