"""Recount Nick's per-rotation R including the trailing-monomial residual R
(mixed entry signs -> one R), which canonical_reducer's classify_monomial_and_d_cost
hard-codes to 0. Uses trailing_info from analyze_candidates.py."""
import sys, json, argparse
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(H.parent / "nick_test"), str(H.parent), str(H.parent / "hrsa")]
import canonical_reducer as cr
from ingest_decompose import parse_fits_file, build_ring
from analyze_candidates import trailing_info
ap = argparse.ArgumentParser(); ap.add_argument("--f", type=int); ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()
f, rows = parse_fits_file(H.parent / "nick_test" / f"fits_f={a.f}.txt")
tot = dict(T3=0, L4=0, Rs=0, Rr=0); n = 0
for _, th, g in rows[: a.n]:
    r = cr.decompose_canonical(build_ring(g, f))
    if not r["success"]: print("fail", th); continue
    for s in r["syllables"]:
        A = [s["a0"], s["a1"], s["a2"]]
        if s["eps"]: tot["Rs"] += 1
        if any(x % 3 for x in A):
            tot["T3" if sum(A) % 3 == 0 else "L4"] += 1
    tot["Rr"] += trailing_info(r["trailing_clifford"])[1]; n += 1
m = {k: v / n for k, v in tot.items()}
print(f"f={a.f} n={n} per circuit: T3 {m['T3']:.1f} L4 {m['L4']:.1f} R_syl {m['Rs']:.2f} R_resid {m['Rr']:.2f} "
      f"-> Tcost {m['T3'] + 8*m['L4'] + 24*(m['Rs'] + m['Rr']):.0f} (without resid {m['T3'] + 8*m['L4'] + 24*m['Rs']:.0f})")
