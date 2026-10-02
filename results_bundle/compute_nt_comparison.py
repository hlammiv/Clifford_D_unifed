#!/usr/bin/env python3
"""compute_nt_comparison.py — reproduce the numbers in 08_NT_fits_and_qubit_comparison.md.

Inputs:
  ../symmetry_variants/stack_full30_rows.csv(.gz)   per (matrix, ±ζ-conjugation): T_unit, T_merged, T_meas
  ../symmetry_variants/full30_f*_candidates.csv(.gz) per copy: N_D
  ../nick_request/topk_run/fast_all30_candidates.csv  ε per (f, θ)
  ../nick_test/nick_tcost_2026-09-30.csv              900-matrix as-given N_D fit
Published baselines (hard-coded, cited in the .md):
  C+R (Gustafson et al. 2503.20203): N_R = 3.20 + 10.77·log10 (Householder), 2.193 + 8.621·log10 (Exhaustive)
  qubit deterministic (Ross–Selinger 1403.2975): N_T ≈ 3·log2(1/ε)
  qubit RUS (Bocharov–Roetteler–Svore PRL 114, 080502): N_T = 9.2 + 1.15·log2(1/ε)
"""
import collections
import csv
import gzip
import math
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
U = B.parent
L3 = lambda e: math.log(1 / e, 3)  # noqa: E731
LOG3_10 = math.log(10, 3)
CR = {"Householder": (3.20, 10.77), "Exhaustive": (2.193, 8.621)}
R_T = {"unitary": 7.0, "merged": 5.0, "meas": 4.0}


def rd(p):
    p = Path(p)
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    return csv.DictReader(gzip.open(p, "rt") if str(p).endswith(".gz") else open(p))


def fit(pts):
    x, y = zip(*pts)
    b, a = np.polyfit(x, y, 1)
    return a, b


def main():
    key = lambda f, t: (str(f), f"{float(t):.6g}")  # noqa: E731
    eps = {key(r["f"], r["theta"]): float(r["epsilon"])
           for r in rd(U / "nick_request/topk_run/fast_all30_candidates.csv") if r["rank"] == "0"}
    # N_D: 900-matrix as-given
    rows = [r for r in rd(U / "nick_test/nick_tcost_2026-09-30.csv") if r["ok"] == "1"]
    a, b = fit([(L3(float(r["epsilon"])), float(r["N_D"])) for r in rows])
    print(f"N_D (900 matrices, as given, convention A): {a:.2f} + {b:.3f}·log3  ({b * LOG3_10:.2f}·log10)  -> {a + b * L3(1e-10):.1f} @1e-10")
    # N_T per model, as-given and best copy
    by = collections.defaultdict(list)
    for r in rd(U / "symmetry_variants/stack_full30_rows.csv"):
        if r["ok"] == "1":
            by[key(r["f"], r["theta"])].append(r)
    fits = {}
    for m, c in (("unitary", "T_unit"), ("merged", "T_merged"), ("meas", "T_meas")):
        for s in ("asis", "best"):
            pts = []
            for k, v in by.items():
                val = float([r for r in v if r["variant"] == "0"][0][c]) if s == "asis" else min(float(r[c]) for r in v)
                pts.append((L3(eps[k]), val))
            fits[(m, s)] = fit(pts)
            a, b = fits[(m, s)]
            print(f"N_T {m:8s} {s:5s}: {a:5.2f} + {b:.3f}·log3  ({b * LOG3_10:.2f}·log10)  -> {a + b * L3(1e-10):6.1f} @1e-10")
    # C+R in the same units
    for m, t in R_T.items():
        for name, (ic, sl) in CR.items():
            print(f"C+R {name:11s} {m:8s}: N_T = {t * ic:5.2f} + {t * sl / LOG3_10:.2f}·log3  -> {t * (ic + sl * 10):6.0f} @1e-10")
    # qubit comparison
    e = 1e-10
    qdet = 3 * math.log2(1 / e)
    qrus = 9.2 + 1.15 * math.log2(1 / e)
    print(f"\nqubit per R_z @1e-10: deterministic {qdet:.0f}, RUS {qrus:.1f}")
    for m in ("meas", "merged", "unitary"):
        a, b = fits[(m, "best")]
        cd = a + b * L3(e)
        print(f"C+D {m:8s} best: rotation {cd:.0f} (x{cd / qdet:.2f} det, x{cd / qrus:.2f} RUS); "
              f"arbitrary qutrit gate 6x = {6 * cd:.0f} vs 2-qubit 10x: det {10 * qdet:.0f} (x{6 * cd / (10 * qdet):.2f}), "
              f"RUS {10 * qrus:.0f} (x{6 * cd / (10 * qrus):.2f})")
    for m in ("meas", "merged", "unitary"):
        crh = R_T[m] * (CR["Householder"][0] + CR["Householder"][1] * 10)
        print(f"C+R Householder {m:8s}: arbitrary gate 6x = {6 * crh:.0f}: x{6 * crh / (10 * qdet):.2f} det, x{6 * crh / (10 * qrus):.2f} RUS")


if __name__ == "__main__":
    main()
