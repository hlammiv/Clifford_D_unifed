#!/usr/bin/env python3
"""analyze_variants.py — summarise variant_search.py output.

Per (f, θ): T-cost of the matrix as given (variant 0 = identity, no transpose)
vs the minimum over variant subsets:
  omega  : D with ω-powers only (Clifford diagonals), no transpose
  zeta   : D with ζ-powers, signs +1, no transpose
  conj   : all ±ζ conjugations, no transpose
  all    : conj × {M, Mᵀ}
Also checks that every variant has the same ε as the original (to 1e-12 relative).
Usage: python3 analyze_variants.py CANDIDATES.csv [...]
"""
import ast
import collections
import csv
import math
import sys

import numpy as np


def subset(row, name):
    d = ast.literal_eval(row["d"])
    t = row["transpose"] == "1"
    signs = [s for s, _ in d]
    exps = [a for _, a in d]
    if name == "all":
        return True
    if t:
        return False
    if name == "conj":
        return True
    if name == "zeta":
        return all(s == 1 for s in signs)
    if name == "omega":
        return all(s == 1 for s in signs) and all(a % 3 == 0 for a in exps)
    raise ValueError(name)


def main(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p)) if r["ok"] == "1"]
    by = collections.defaultdict(list)
    for r in rows:
        by[(int(r["f"]), r["theta"])].append(r)
    subsets = ["omega", "zeta", "conj", "all"]
    print(f"{len(rows)} decompositions, {len(by)} (f, θ) matrices")
    bad_eps = 0
    per_f = collections.defaultdict(lambda: collections.defaultdict(list))
    for (f, th), v in by.items():
        e = [float(r["epsilon"]) for r in v]
        if max(e) - min(e) > 1e-12 * max(e) + 1e-15:
            bad_eps += 1
        ident = [r for r in v if r["variant"] == "0"][0]
        T0 = float(ident["Tcost"])
        per_f[f]["T0"].append(T0)
        per_f[f]["eps"].append(e[0])
        per_f[f]["L4_0"].append(float(ident["n_L4"]))
        per_f[f]["R_0"].append(float(ident["n_R"]))
        for s in subsets:
            vs = [r for r in v if subset(r, s)]
            best = min(vs, key=lambda r: float(r["Tcost"]))
            per_f[f][s].append(float(best["Tcost"]))
            if s == "all":
                per_f[f]["L4_best"].append(float(best["n_L4"]))
                per_f[f]["R_best"].append(float(best["n_R"]))
                per_f[f]["T3_best"].append(float(best["n_T3"]))
        per_f[f]["nvar"].append(len(v))
    print(f"matrices whose variants do NOT all share ε: {bad_eps}\n")
    print(" f   n  variants  T as-is | best ω-conj  best ζ-conj  best ±ζ-conj  best all (+ᵀ) | saving(all)   L4 as-is→best  R as-is→best")
    for f in sorted(per_f):
        d = per_f[f]
        m = lambda k: np.mean(d[k])  # noqa: E731
        print(f"{f:2d} {len(d['T0']):3d} {m('nvar'):8.0f}  {m('T0'):7.1f} | {m('omega'):10.1f} {m('zeta'):12.1f} "
              f"{m('conj'):13.1f} {m('all'):15.1f} | {100 * (1 - m('all') / m('T0')):6.1f}%      "
              f"{m('L4_0'):5.1f}→{m('L4_best'):5.1f}   {m('R_0'):4.1f}→{m('R_best'):4.1f}")
    if len(per_f) >= 2:
        for k, lab in (("T0", "as-is"), ("all", "best variant")):
            xs, ys = [], []
            for f, d in per_f.items():
                xs += [math.log(1 / e, 3) for e in d["eps"]]
                ys += d[k]
            b, a = np.polyfit(xs, ys, 1)
            print(f"fit {lab:13s}: T = {a:6.2f} + {b:6.3f}·log3(1/ε)   (n={len(xs)})")


if __name__ == "__main__":
    main(sys.argv[1:])
