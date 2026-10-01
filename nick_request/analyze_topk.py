"""analyze_topk.py -- ingest Nick's top-K candidate dumps and measure how much
T-cost we gain by choosing among near-optimal approximants instead of the
single min-Frobenius one.

Input: one or more fits_topk_f=N.txt files (format of fits_f=N.txt plus two
optional trailing columns ", frob , rank"; several rows per theta).  Plain
fits_f=N.txt files also work (one candidate per theta).  The 'vector' format
(header '#f=N norm=... vector', 3 groups of 6 ints = 3^f x) is converted with
zeta9/tools.householder_int_matrix.

Per candidate: exact unitarity over Z[zeta9,1/3] (ingest_decompose.unitarity_exact),
epsilon = raw ||M - R_z(theta)||_F, canonical decomposition with the residual-R fix
(nick_test/nick_tcost_all.work), n_T3, n_L4, n_R, Tcost = n_T3 + 7 n_L4 + 7 n_R.

Outputs (prefix --out):
  <out>_candidates.csv  one row per decomposed candidate
  <out>_summary.csv     per f and selection rule: mean Tcost, mean eps, n_theta
  <out>_fits.csv        Tcost = a*log3(1/eps) + b for each selection rule (all f pooled)
Selection rules (per theta): best_frob (rank 0, = old pipeline), and
min_tcost@K' = cheapest candidate among the K' lowest-frob ones (K' = 1,3,10,30,100,...),
ties broken by lower frob.

Element order: rows are read exactly as ingest_decompose/nick_tcost_all read
fits_f=N.txt (so numbers are comparable with nick_tcost_2026-09-30.csv).  NOTE:
Nick's dump_special_thetas.py writes the 9 groups column-major, so this is M^T;
eps to a diagonal target and exact unitarity are transpose-invariant.  Use
--colmajor to read the true M.

  python3 analyze_topk.py fits_topk_f=4.txt fits_topk_f=8.txt --k 30 --max-theta 40 --procs 8
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
U = H.parent
sys.path[:0] = [str(U / "nick_test"), str(U), str(U / "hrsa"),
                "/home/hlamm/Desktop/efficent_gates/zeta9"]


# ---------------------------------------------------------------- parsing

def parse_topk_file(path):
    """Return (f, rows); rows = list of dict(theta, groups[9][6], frob_file, rank_file, line)."""
    f = None
    vector = False
    rows = []
    with open(path) as fh:
        for ln, raw in enumerate(fh):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line[1:].strip().startswith("f="):
                    f = int(line.split("f=")[1].split()[0].rstrip(","))
                    vector = "vector" in line
                continue
            parts = [p.strip() for p in line.split(",")]
            theta = float(parts[0])
            ng = 3 if vector else 9
            groups = [[int(x) for x in p.split()] for p in parts[1:1 + ng]]
            if len(groups) != ng or any(len(g) != 6 for g in groups):
                raise ValueError(f"{path}:{ln + 1}: expected {ng} groups of 6 ints")
            extra = parts[1 + ng:]
            frob = float(extra[0]) if len(extra) >= 1 and extra[0] else None
            rank = int(extra[1]) if len(extra) >= 2 and extra[1] else None
            if vector:
                from zeta9.tools import householder_int_matrix
                _, N = householder_int_matrix(groups, f, "2/3^f")
                groups = [list(N[i][j]) for i in range(3) for j in range(3)]  # row-major
            rows.append(dict(theta=theta, groups=groups, frob_file=frob, rank_file=rank, line=ln + 1))
    if f is None:
        raise ValueError(f"{path}: no '#f=N' header")
    return f, rows


def group_by_theta(rows, f, colmajor=False):
    """{theta: [cand sorted by eps]}; eps recomputed in float."""
    from ingest_decompose import build_complex
    out = defaultdict(list)
    for r in rows:
        g = r["groups"]
        if colmajor:
            g = [g[3 * (k % 3) + k // 3] for k in range(9)]
            r = dict(r, groups=g)
        th = r["theta"]
        M = build_complex(g, f)
        T = np.diag([np.exp(-0.5j * th), np.exp(0.5j * th), 1.0])
        r["eps"] = float(np.linalg.norm(M - T))
        out[th].append(r)
    for th in out:
        out[th].sort(key=lambda r: (r["eps"], r["line"]))
    return out


# ---------------------------------------------------------------- worker

def work(item):
    from ingest_decompose import build_ring, unitarity_exact
    import nick_tcost_all
    f, th, rank, g, eps_file, select = item
    t0 = time.time()
    exact, resid = unitarity_exact(build_ring(g, f))
    if not exact:
        rec = dict(f=f, theta=th, epsilon=float("nan"), ok=0)
    else:
        rec = nick_tcost_all.work((f, th, g, select))
    rec.update(rank=rank, frob_file=eps_file, unitary_exact=int(exact), wall=round(time.time() - t0, 3))
    return rec


# ---------------------------------------------------------------- analysis

def kprime_list(k):
    ks, v = [], 1
    while v < k:
        ks.append(v)
        v = v * 3 if str(v)[0] == "1" else v * 10 // 3  # 1,3,10,30,100,...
    ks.append(k)
    return sorted(set(ks))


def summarize(recs, k):
    by = defaultdict(list)
    for r in recs:
        if r.get("ok"):
            by[(r["f"], r["theta"])].append(r)
    rules = ["best_frob"] + [f"min_tcost@{kk}" for kk in kprime_list(k)]
    picks = defaultdict(list)  # (f, rule) -> [(eps, Tcost, N_D)]
    for (f, th), L in by.items():
        L.sort(key=lambda r: r["rank"])
        if L[0]["rank"] != 0:
            continue  # best-frob candidate failed; skip theta for a fair comparison
        picks[(f, "best_frob")].append((L[0]["epsilon"], L[0]["Tcost"], L[0]["N_D"]))
        for kk in kprime_list(k):
            sub = [r for r in L if r["rank"] < kk]
            b = min(sub, key=lambda r: (r["Tcost"], r["epsilon"]))
            picks[(f, f"min_tcost@{kk}")].append((b["epsilon"], b["Tcost"], b["N_D"]))
    summ = []
    for (f, rule), P in sorted(picks.items(), key=lambda x: (x[0][0], rules.index(x[0][1]))):
        e, t, nd = (np.array(c, dtype=float) for c in zip(*P))
        summ.append(dict(f=f, rule=rule, n_theta=len(P), mean_Tcost=t.mean(), mean_N_D=nd.mean(),
                         mean_eps=e.mean(), mean_log3_inv_eps=np.mean(np.log(1 / e) / np.log(3))))
    fits = []
    for rule in rules:
        pts = [p for (f, r), P in picks.items() if r == rule for p in P]
        if len(pts) < 3:
            continue
        e, t, _ = (np.array(c, dtype=float) for c in zip(*pts))
        x = np.log(1 / e) / np.log(3)
        if np.ptp(x) < 1e-9:
            continue
        a, b = np.polyfit(x, t, 1)
        fits.append(dict(rule=rule, n=len(pts), slope=a, intercept=b,
                         f_values=" ".join(str(f) for f in sorted({f for (f, r) in picks if r == rule}))))
    return summ, fits


def write_csv(path, rows, keys=None):
    if not rows:
        return
    keys = keys or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--k", type=int, default=100, help="decompose the K lowest-eps candidates per theta")
    ap.add_argument("--max-theta", type=int, default=0, help="evenly subsample at most N thetas per file (0 = all)")
    ap.add_argument("--procs", type=int, default=8, help="worker processes (shared machine: keep <= 8)")
    ap.add_argument("--select", choices=["d", "tcost"], default="d", help="reducer prefix-selection cost")
    ap.add_argument("--colmajor", action="store_true", help="read the 9 groups column-major (true M, not M^T)")
    ap.add_argument("--out", default=str(H / "topk_analysis"))
    a = ap.parse_args(argv)
    procs = min(a.procs, 8)

    items = []
    for p in a.inputs:
        f, rows = parse_topk_file(p)
        byth = group_by_theta(rows, f, a.colmajor)
        ths = sorted(byth)
        if a.max_theta and len(ths) > a.max_theta:
            idx = sorted(set(np.linspace(0, len(ths) - 1, a.max_theta).round().astype(int)))
            ths = [ths[i] for i in idx]
        nc = 0
        for th in ths:
            for rank, r in enumerate(byth[th][: a.k]):
                if r["frob_file"] is not None and abs(r["frob_file"] - r["eps"]) > 1e-9 * max(1.0, r["eps"]) + 1e-12:
                    print(f"WARN {p.name}:{r['line']}: file frob {r['frob_file']:.3e} != recomputed {r['eps']:.3e}")
                items.append((f, th, rank, r["groups"], r["frob_file"], a.select))
                nc += 1
        print(f"{p.name}: f={f}, {len(byth)} thetas, {len(rows)} rows -> {len(ths)} thetas, {nc} candidates queued")

    keys = ["f", "theta", "rank", "epsilon", "frob_file", "unitary_exact", "ok", "N_D", "n_T3", "n_L4",
            "n_R_syl", "n_R_resid", "n_R", "Tcost", "wall"]
    recs = []
    t0 = time.time()
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(work, items, chunksize=1)):
            recs.append(r)
            if i % 50 == 0 or i == len(items) - 1:
                print(f"  {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)
    recs.sort(key=lambda r: (r["f"], r["theta"], r["rank"]))
    write_csv(a.out + "_candidates.csv", recs, keys)
    nbad = sum(1 for r in recs if not r["unitary_exact"])
    nfail = sum(1 for r in recs if r["unitary_exact"] and not r.get("ok"))
    print(f"\n{len(recs)} candidates: {nbad} NOT exactly unitary, {nfail} decomposition failures")

    summ, fits = summarize(recs, a.k)
    write_csv(a.out + "_summary.csv", summ)
    write_csv(a.out + "_fits.csv", fits)
    print(f"\n{'f':>3} {'rule':<16} {'n_th':>5} {'<Tcost>':>9} {'<N_D>':>8} {'<eps>':>10}")
    for s in summ:
        print(f"{s['f']:>3} {s['rule']:<16} {s['n_theta']:>5} {s['mean_Tcost']:>9.2f} "
              f"{s['mean_N_D']:>8.2f} {s['mean_eps']:>10.3e}")
    print("\nTcost = a * log3(1/eps) + b   (all f pooled)")
    for ft in fits:
        print(f"  {ft['rule']:<16} a={ft['slope']:7.3f}  b={ft['intercept']:8.2f}  n={ft['n']}  f={ft['f_values']}")
    print(f"\nwrote {a.out}_candidates.csv, _summary.csv, _fits.csv")


if __name__ == "__main__":
    main()
