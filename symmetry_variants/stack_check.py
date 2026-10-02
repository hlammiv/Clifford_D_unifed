#!/usr/bin/env python3
"""stack_check.py — do symmetry-variant selection and rotation merging stack?

For each sampled matrix (best-ε candidate per θ) and each of its 324 ±ζ
conjugations D V D†:
  T_unit   = n_T3 + 7 n_L4 + 7 n_R           (unitary 7-T gadgets, as emitted)
  T_merged = rotation-merged T-count of the shared-ancilla circuit
             (depth_optimality/tmerge.merge_symbolic, which equals the exact numeric
             merge on all 31 test circuits)
  T_meas   = n_T3 + 4 n_L4 + 4 n_R            (measurement gadgets; merging gives nothing)
Writes one row per (matrix, variant).

Usage: python3 stack_check.py SAMPLE.txt [...] --n-theta N --procs P --out PREFIX
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
U = H.parent
sys.path[:0] = [str(H), str(U / "decomp_speed"), str(U / "depth_optimality"), str(U / "compiler"),
                str(U / "nick_test"), str(U / "nick_request"), str(U), str(U / "hrsa")]


def work(item):
    import fast_decompose
    fast_decompose.install()
    import canonical_reducer as cr
    from ingest_decompose import build_ring
    from variant_search import variant_groups
    from emit_variants import emit
    from tmerge import merge_symbolic
    f, th, groups, vid, d = item
    t0 = time.time()
    g = variant_groups(groups, d, False)
    r = cr.decompose_canonical(build_ring(g, f))
    if not r["success"]:
        return dict(f=f, theta=th, variant=vid, ok=0)
    gates, c = emit(r["syllables"], r["trailing_clifford"], "shared", np.random.default_rng(0))
    t3, l4, nr = c["n_T3"], c["n_L4"], c["n_R"]
    return dict(f=f, theta=th, variant=vid, d=str(d), ok=1, n_T3=t3, n_L4=l4, n_R=nr,
                T_unit=t3 + 7 * l4 + 7 * nr, T_merged=merge_symbolic(gates),
                T_meas=t3 + 4 * l4 + 4 * nr, wall=round(time.time() - t0, 3))


def main():
    from analyze_topk import parse_topk_file, group_by_theta
    from variant_search import variant_list
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--n-theta", type=int, default=30)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=str(H / "stack"))
    a = ap.parse_args()
    vl = [d for d, t in variant_list("conj")]
    items = []
    for p in a.inputs:
        f, rows = parse_topk_file(p)
        by = group_by_theta(rows, f, colmajor=True)
        ths = sorted(by)
        idx = sorted(set(np.linspace(0, len(ths) - 1, min(a.n_theta, len(ths))).round().astype(int)))
        for i in idx:
            for vid, d in enumerate(vl):
                items.append((f, ths[i], by[ths[i]][0]["groups"], vid, d))
        print(f"{Path(p).name}: f={f}, {len(idx)} thetas x {len(vl)} variants", flush=True)
    keys = ["f", "theta", "variant", "d", "ok", "n_T3", "n_L4", "n_R", "T_unit", "T_merged", "T_meas", "wall"]
    t0 = time.time()
    with open(a.out + "_rows.csv", "w", newline="") as fh, Pool(a.procs) as pool:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(pool.imap_unordered(work, items, chunksize=8)):
            w.writerow(r)
            fh.flush()
            if i % 2000 == 0 or i == len(items) - 1:
                print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    print("wrote", a.out + "_rows.csv")


if __name__ == "__main__":
    main()
