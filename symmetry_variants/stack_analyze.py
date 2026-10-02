#!/usr/bin/env python3
"""stack_analyze.py — headline table for C+D under all cost models, with and
without symmetry-variant selection, plus C+R in matching units.

Models (T per R_z rotation):
  unitary   : 7-T gadgets as emitted                     C+R: R = 7 T
  merged    : unitary + rotation merging (shared ancilla) C+R: R = 5.0 T (merged R chain, measured)
  meas      : measurement 4-T gadgets (no merging)       C+R: R = 4 T
Usage: python3 stack_analyze.py stack_full30_rows.csv [eps_source.csv]
eps per (f, θ) is taken from nick_request/topk_run/fast_all30_candidates.csv (rank 0).
"""
import collections
import csv
import math
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
CR = {"Householder": (3.20, 10.77), "Exhaustive": (2.193, 8.621)}   # N_R per log10
R_COST = {"unitary": 7.0, "merged": 5.0, "meas": 4.0}
L10_3 = math.log(10, 3)


def main():
    rows_path = sys.argv[1]
    eps_path = sys.argv[2] if len(sys.argv) > 2 else str(H.parent / "nick_request/topk_run/fast_all30_candidates.csv")
    key = lambda f, th: (str(f), f"{float(th):.6g}")  # noqa: E731
    eps = {key(r["f"], r["theta"]): float(r["epsilon"]) for r in csv.DictReader(open(eps_path)) if r["rank"] == "0"}
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(rows_path)):
        if r["ok"] == "1":
            by[key(r["f"], r["theta"])].append(r)
    cols = {"unitary": "T_unit", "merged": "T_merged", "meas": "T_meas"}
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    pts = collections.defaultdict(list)
    for k, v in by.items():
        f = int(k[0])
        x = math.log(1 / eps[k], 3)
        v0 = [r for r in v if r["variant"] == "0"][0]
        for m, c in cols.items():
            a0 = float(v0[c])
            ab = min(float(r[c]) for r in v)
            per[f][m + "_asis"].append(a0)
            per[f][m + "_best"].append(ab)
            pts[m + "_asis"].append((x, a0))
            pts[m + "_best"].append((x, ab))
        per[f]["eps"].append(eps[k])
    print(f"{len(by)} matrices x {len(next(iter(by.values())))} variants\n")
    print(" f  median ε   | unitary as-is → best | merged as-is → best | meas as-is → best")
    for f in sorted(per):
        d = per[f]
        m = lambda q: np.mean(d[q])  # noqa: E731
        print(f"{f:2d}  {np.median(d['eps']):.2e} |  {m('unitary_asis'):6.1f} → {m('unitary_best'):6.1f}   "
              f"|  {m('merged_asis'):6.1f} → {m('merged_best'):6.1f}  |  {m('meas_asis'):6.1f} → {m('meas_best'):6.1f}")
    print("\nFits T = a + b·log3(1/ε); value at 1e-10; C+R in the same gadget model:")
    print(" model    selection | a      b      @1e-10 | C+R Householder @1e-10 (ratio) | C+R Exhaustive @1e-10 (ratio)")
    L = math.log(1e10, 3)
    for m in cols:
        for s in ("asis", "best"):
            xs, ys = zip(*pts[f"{m}_{s}"])
            b, a = np.polyfit(xs, ys, 1)
            v10 = a + b * L
            crh = R_COST[m] * (CR["Householder"][0] + CR["Householder"][1] * 10)
            cre = R_COST[m] * (CR["Exhaustive"][0] + CR["Exhaustive"][1] * 10)
            print(f" {m:8s} {s:9s} | {a:6.2f} {b:6.3f} {v10:7.1f} | {crh:7.0f} ({crh / v10:4.2f}x)               "
                  f"| {cre:6.0f} ({cre / v10:4.2f}x)")


if __name__ == "__main__":
    main()
