#!/usr/bin/env python3
"""variant_search.py — free T-cost reduction from distance-preserving symmetries.

For an exact approximant V of the diagonal target R_z(θ), every
    V' = D V D†            D = diag(±ζ^a, ±ζ^b, ±ζ^c)  (324 classes mod global phase)
and its transpose V'ᵀ is an equally good exact approximant
(‖D V D† − R_z‖ = ‖V − R_z‖ because D commutes with R_z; R_zᵀ = R_z).
They decompose to different words, so we pick the cheapest.

Ring arithmetic is done on Nick's 6-integer coefficient groups (basis ζ^0..ζ^5,
ζ^6 = −ζ^3 − 1), so every variant stays exactly in Z[ζ9, 1/3].
Decomposition uses the fast exact reducer (decomp_speed/fast_decompose.py).

Usage:
  python3 variant_search.py SAMPLE.txt [SAMPLE.txt ...] --n-theta N --procs P --out PREFIX
        [--variants all|conj|conj-omega]   (default all = 324 conj × {M, Mᵀ})
Takes the best-frob candidate (lowest ε) per θ from each top-K sample file.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
U = H.parent
sys.path[:0] = [str(U / "decomp_speed"), str(U / "nick_test"), str(U / "nick_request"), str(U), str(U / "hrsa")]


def zmul(g, k, s):
    """(s·ζ^k)·g for a coefficient list g (basis ζ^0..ζ^5, ζ^9 = 1, ζ^6 = −ζ^3 − 1)."""
    c = [0] * 9
    for i, x in enumerate(g):
        c[(i + k) % 9] += s * x
    for i in (8, 7, 6):                      # reduce ζ^i, i >= 6: ζ^i = −ζ^{i−3} − ζ^{i−6}
        x = c[i]
        if x:
            c[i] = 0
            c[i - 3] -= x
            c[i - 6] -= x
    return c[:6]


def variant_groups(groups, d, transpose):
    """groups: 9 lists (row-major M_ij). d: 3 tuples (sign, exp). Returns groups of D M D† (optionally ᵀ)."""
    out = [None] * 9
    for i in range(3):
        for j in range(3):
            si, ai = d[i]
            sj, aj = d[j]
            out[3 * i + j] = zmul(groups[3 * i + j], (ai - aj) % 9, si * sj)
    if transpose:
        out = [out[3 * (k % 3) + k // 3] for k in range(9)]
    return out


def variant_list(mode):
    """Distinct D mod global phase: fix d_0 = (+1, 0)."""
    signs = (1, -1)
    if mode == "conj-omega":
        exps = (0, 3, 6)
    else:
        exps = range(9)
    ds = [((1, 0), (s1, a1), (s2, a2)) for s1, a1, s2, a2 in itertools.product(signs, exps, signs, exps)]
    ts = (False, True) if mode == "all" else (False,)
    return [(d, t) for d in ds for t in ts]


def work(item):
    import fast_decompose
    fast_decompose.install()
    import nick_tcost_all
    f, th, groups, vid, d, t = item
    g = variant_groups(groups, d, t)
    t0 = time.time()
    rec = nick_tcost_all.work((f, th, g, "d"))
    rec.update(variant=vid, transpose=int(t), d=str(d), wall=round(time.time() - t0, 3))
    return rec


def main():
    from analyze_topk import parse_topk_file, group_by_theta
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--n-theta", type=int, default=30)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--variants", choices=["all", "conj", "conj-omega"], default="all")
    ap.add_argument("--out", default=str(H / "variants"))
    a = ap.parse_args()
    vl = variant_list(a.variants)
    items = []
    for p in a.inputs:
        f, rows = parse_topk_file(p)
        by = group_by_theta(rows, f, colmajor=True)
        ths = sorted(by)
        idx = sorted(set(np.linspace(0, len(ths) - 1, min(a.n_theta, len(ths))).round().astype(int)))
        for i in idx:
            best = by[ths[i]][0]                         # lowest-ε candidate
            for vid, (d, t) in enumerate(vl):
                items.append((f, ths[i], best["groups"], vid, d, t))
        print(f"{Path(p).name}: f={f}, {len(idx)} thetas x {len(vl)} variants", flush=True)
    keys = ["f", "theta", "variant", "transpose", "d", "epsilon", "ok", "N_D", "n_T3", "n_L4",
            "n_R_syl", "n_R_resid", "n_R", "Tcost", "wall"]
    t0 = time.time()
    with open(a.out + "_candidates.csv", "w", newline="") as fh, Pool(a.procs) as pool:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(pool.imap_unordered(work, items, chunksize=8)):
            w.writerow(r)
            fh.flush()
            if i % 500 == 0 or i == len(items) - 1:
                print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    print("wrote", a.out + "_candidates.csv")


if __name__ == "__main__":
    main()
