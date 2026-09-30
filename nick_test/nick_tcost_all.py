"""nick_tcost_all.py — re-decompose Nick's exact R_z approximants with the fixed
canonical_reducer (residual R now charged, 2026-09-30) and record per-matrix
fault-tolerant counts:
  n_T3  level-3 (T-type) syllable diagonals      -> 1 T each
  n_L4  level-4 syllable diagonals               -> 7 T each
  n_R   R syllables + residual R                 -> 7 T each
  Tcost = n_T3 + 7 n_L4 + 7 n_R
(7-T constructions with 1 clean ancilla: unified/r_from_d/verify_R7T.py,
unified/level4/l4_7T_verify.py).  Residual zeta_9 phases of the trailing
monomial: D-pattern cost 1 -> T-type, 2 -> level-4.
Output: nick_tcost_2026-09-30.csv (first 150 rows per f, matching the headline set).
"""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(H.parent), str(H.parent / "hrsa")]


def work(item):
    import numpy as np
    import canonical_reducer as cr
    from ingest_decompose import build_ring, build_complex
    f, th, g = item
    Mc = build_complex(g, f)
    tgt = np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2), 1])
    eps = float(np.linalg.norm(Mc - tgt))
    r = cr.decompose_canonical(build_ring(g, f))
    if not r["success"]:
        return dict(f=f, theta=th, epsilon=eps, ok=0)
    t3 = l4 = rs = 0
    for s in r["syllables"]:
        A = (s["a0"], s["a1"], s["a2"])
        if any(x % 3 for x in A):
            if sum(A) % 3 == 0:
                t3 += 1
            else:
                l4 += 1
        rs += 1 if s["eps"] else 0
    _, md, mr = cr.classify_monomial_and_d_cost(r["trailing_clifford"])
    rd = md - mr
    if rd == 1:
        t3 += 1
    elif rd >= 2:
        l4 += 1
    n_r = rs + mr
    return dict(f=f, theta=th, epsilon=eps, ok=1, N_D=r["D_count"], n_T3=t3, n_L4=l4,
                n_R_syl=rs, n_R_resid=mr, n_R=n_r, Tcost=t3 + 7 * l4 + 7 * n_r)


def main():
    from ingest_decompose import parse_fits_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", default="4,6,8,10,12,14,16")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--per-f", type=int, default=150)
    a = ap.parse_args()
    items = []
    for f in map(int, a.fs.split(",")):
        F, rows = parse_fits_file(H / f"fits_f={f}.txt")
        items += [(F, th, g) for _, th, g in rows[: a.per_f]]
    out = H / "nick_tcost_2026-09-30.csv"
    keys = ["f", "theta", "epsilon", "ok", "N_D", "n_T3", "n_L4", "n_R_syl",
            "n_R_resid", "n_R", "Tcost"]
    with open(out, "w", newline="") as fh, Pool(a.procs) as pool:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(work, items, chunksize=2)):
            w.writerow(row)
            fh.flush()
            if i % 50 == 0:
                print(i, "/", len(items), flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
