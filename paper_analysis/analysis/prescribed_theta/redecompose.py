#!/usr/bin/env python3
"""redecompose.py — re-decompose every stored PRESCRIBED-theta exact R_z
approximant (unified/sweep_*/**.json) with the post-residual-R-fix fast
decomposer and record the same per-class counts as Nick's special-theta set
(unified/nick_test/nick_tcost_2026-09-30.csv).

Counting is delegated verbatim to unified/nick_test/nick_tcost_all.work
(after decomp_speed/fast_decompose.install(), which swaps in the
bit-identical numba decomposer):
  n_T   = n_T3   T-type syllable diagonals (+ residual monomial phase of D-cost 1)
  n_4   = n_L4   level-4 syllable diagonals (+ residual phase of D-cost >= 2)
  n_R   = R syllables + residual R
  N_phi = N_D    (reducer D_count, convention A: R charged; this is the column
                  the headline fit 3.94 + 5.161 log3 is made from)
  ops   = n_T + n_4 + n_R
  tcost_unitary = n_T + 7 n_4 + 7 n_R ;  tcost_meas = n_T + 4 n_4 + 4 n_R

Checks per unitary: exact unitarity over Z[zeta_9, 1/3]
(ingest_decompose.unitarity_exact), and |V - target|_F (plain Frobenius, same
as work()) vs the stored 'achieved_frob'.

Theta: zeta9 backends store full precision in inputs.theta; HRSA JSONs round
inputs.theta to 6 digits, so the full-precision value is taken from
identification.command_line[1].

Denominators: every JSON stores V = (1/3^f) * Z[zeta_9] numerators (same as
Nick's fits files).  HRSA's own "f" (method string "HRSA(f=k)", inputs.max_f)
is the u-denominator; V's f is 2k.  We report sde = sde_3(V) = smallest k with
3^k V integral (all 6 power-basis coefficients divisible by 3 <=> 3 | x, since
the power basis is integral).

Dedup: unitaries are keyed by (sde, reduced numerators).  The same exact V can
appear in several sweeps (redo runs, overlapping grids) and at several target
thetas (e.g. Cliffords at loose eps).  One row per distinct V; the kept
(theta, target_eps) is the occurrence with the smallest achieved eps; n_occ and
n_theta record the multiplicity.

Usage: python3 redecompose.py [--procs 14] [--zeta9-subsample N]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from multiprocessing import Pool
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

UNI = _repo_root(Path(__file__).resolve())
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(UNI / "decomp_speed"), str(UNI / "nick_test"), str(UNI), str(UNI / "hrsa")]

import fast_decompose as fd  # noqa: E402
fd.install()                 # patch cr.decompose_canonical before forking workers
import nick_tcost_all as nta  # noqa: E402
from ingest_decompose import build_ring, unitarity_exact, parse_fits_file  # noqa: E402
import os





W_UNI, W_MEAS = 7, 4


def v3(n: int) -> int:
    n = abs(int(n))
    if n == 0:
        return 10 ** 9
    k = 0
    while n % 3 == 0:
        n //= 3
        k += 1
    return k


def reduce_groups(groups, f):
    """Return (sde, reduced groups) with all numerators divided by the common 3^k."""
    k = min(v3(c) for g in groups for c in g)
    if k >= 10 ** 9:
        raise ValueError("zero matrix")
    k = min(k, f) if f >= 0 else k
    d = 3 ** k
    return f - k, [[c // d for c in g] for g in groups]


def load_json(p):
    try:
        D = json.load(open(p))
    except Exception:
        return None
    if not isinstance(D, dict):
        return None
    u = D.get("unitary")
    if not u or not u.get("V"):
        return None
    ident = D.get("identification", {}) or {}
    backend = ident.get("backend", "?")
    inp = D.get("inputs", {})
    th = float(inp["theta"])
    cl = ident.get("command_line") or []
    if backend == "hrsa" and len(cl) > 2:
        try:
            th_full = float(cl[1])
            if abs(th_full - th) < 1e-5:
                th = th_full
        except ValueError:
            pass
    f = int(u["f"])
    groups = [[int(c) for c in u["V"][i][j]] for i in range(3) for j in range(3)]
    ach = (D.get("achieved") or {})
    return dict(path=str(Path(p).relative_to(UNI)), source=str(Path(p).relative_to(UNI)).split("/")[0],
                backend=backend, method=ach.get("method"), theta=th,
                target_eps=float(inp["epsilon"]) if inp.get("epsilon") is not None else float("nan"),
                stored_eps=ach.get("achieved_frob"), f_stored=f, groups=groups,
                stored_ND=(D.get("decomposition") or {}).get("N_D"))


def job(rec):
    sde, red = rec["sde"], rec["red"]
    out = dict(rec)
    out.pop("red")
    Vr = build_ring(red, sde)
    ok_u, resid = unitarity_exact(Vr)
    out["unitary_exact"] = int(ok_u)
    out["sde_chi"] = sde_chi(Vr)
    t0 = time.time()
    r = nta.work((sde, rec["theta"], red, "d"))
    out["dec_s"] = round(time.time() - t0, 3)
    out["ok"] = r["ok"]
    out["achieved_eps"] = r["epsilon"]
    if r["ok"]:
        nT, n4, nR = r["n_T3"], r["n_L4"], r["n_R"]
        out.update(n_T=nT, n_4=n4, n_R=nR, n_R_syl=r["n_R_syl"], n_R_resid=r["n_R_resid"],
                   N_phi=r["N_D"], N_phi_formula=2 * nT + n4 + nR, ops=nT + n4 + nR,
                   tcost_unitary=r["Tcost"], tcost_meas=nT + W_MEAS * n4 + W_MEAS * nR)
        assert r["Tcost"] == nT + W_UNI * n4 + W_UNI * nR
    return out


def sde_chi(Vr):
    """chi-adic sde of V (chi = 1 - zeta_9): max over entries of cr.sde_chi_full."""
    import canonical_reducer as cr
    return max(cr.sde_chi_full(Vr[i][j]) for i in range(3) for j in range(3))


def nick_sde():
    """sde_3 for every matrix in Nick's fits files, keyed by (f, theta repr)."""
    m = {}
    for p in sorted((UNI / "nick_test").glob("fits_f=*.txt")):
        F, rows = parse_fits_file(p)
        for _, th, g in rows:
            sd, red = reduce_groups(g, F)
            m[(F, th)] = (sd, sde_chi(build_ring(red, sd)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--zeta9-subsample", type=int, default=0,
                    help="keep every N-th zeta9_tier2 cell (0 = all)")
    a = ap.parse_args()

    files = sorted(glob.glob(str(UNI / "sweep_*" / "**" / "*.json"), recursive=True))
    if a.zeta9_subsample:
        files = [p for p in files if "zeta9_tier2" not in p
                 or int(Path(p).stem.split("_")[1]) % a.zeta9_subsample == 0]
    recs, n_null = [], 0
    for p in files:
        r = load_json(p)
        if r is None:
            n_null += 1
            continue
        recs.append(r)
    print(f"{len(files)} json, {n_null} without unitary, {len(recs)} with V", flush=True)

    # stored-eps check (in the input convention, before reduction)
    # dedup by exact reduced V
    distinct = {}
    for r in recs:
        sde, red = reduce_groups(r["groups"], r["f_stored"])
        key = (sde, tuple(tuple(g) for g in red))
        r["sde"], r["red"] = sde, red
        distinct.setdefault(key, []).append(r)
    print(f"{len(distinct)} distinct unitaries", flush=True)

    # eps for every occurrence (cheap, numeric) -> pick best occurrence, check stored
    Z = np.exp(2j * np.pi / 9) ** np.arange(6)
    items, epscheck = [], []
    for key, occ in distinct.items():
        sde, red = key[0], [list(g) for g in key[1]]
        M = np.array([np.dot(g, Z) for g in red]).reshape(3, 3) / 3 ** sde
        for r in occ:
            T = np.diag([np.exp(-1j * r["theta"] / 2), np.exp(1j * r["theta"] / 2), 1])
            r["eps_num"] = float(np.linalg.norm(M - T))
            # metadata consistency: stored achieved_frob must match |V - target(theta)|_F
            # (HRSA prints 6 significant digits -> abs tol 1e-6 + rel 1e-3)
            r["eps_ok"] = int(r["stored_eps"] is None or
                              abs(r["stored_eps"] - r["eps_num"]) <= 1e-6 + 1e-3 * r["eps_num"])
            if r["stored_eps"] is not None:
                epscheck.append((r["path"], r["backend"], r["stored_eps"], r["eps_num"]))
        good = [r for r in occ if r["eps_ok"]]
        best = min(good or occ, key=lambda r: r["eps_num"])
        rec = {k: best[k] for k in ("source", "path", "backend", "method", "theta", "target_eps",
                                    "stored_eps", "f_stored", "sde", "red", "stored_ND", "eps_ok")}
        rec["n_occ"] = len(occ)
        rec["n_theta"] = len({round(r["theta"], 9) for r in occ})
        rec["sources"] = ";".join(sorted({r["source"] for r in occ}))
        items.append(rec)

    # stored eps agreement
    rel = [(abs(s - e) / max(e, 1e-300), b, p) for p, b, s, e in epscheck if e > 0]
    absd = [abs(s - e) for p, b, s, e in epscheck]
    for b in sorted({x[1] for x in rel}):
        rr = [x[0] for x in rel if x[1] == b]
        print(f"stored-vs-recomputed eps [{b}]: n={len(rr)} max rel diff {max(rr):.2e} "
              f"median {np.median(rr):.2e}", flush=True)
    print(f"max abs eps diff (all) {max(absd):.2e}")
    bad = [r for occ in distinct.values() for r in occ if not r["eps_ok"]]
    print(f"{len(bad)} occurrences with stored eps != |V - target(theta)|_F "
          f"(theta metadata inconsistent): {sorted(r['path'] for r in bad)}")
    print(f"distinct unitaries with no consistent occurrence (eps_ok=0, excluded in compare.py): "
          f"{sum(1 for it in items if not it['eps_ok'])}")
    with open(OUT / "eps_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "backend", "stored_eps", "recomputed_eps", "rel_diff"])
        for p, b, s, e in epscheck:
            w.writerow([p, b, s, e, abs(s - e) / e if e > 0 else (0.0 if s == 0 else float("inf"))])

    # decompose (largest sde first for load balance)
    items.sort(key=lambda r: -r["sde"])
    t0 = time.time()
    rows = []
    with Pool(a.procs) as pool:
        for i, row in enumerate(pool.imap_unordered(job, items, chunksize=4)):
            rows.append(row)
            if i % 500 == 0:
                print(f"{i}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    print(f"decomposed {len(rows)} in {time.time() - t0:.0f}s", flush=True)

    rows.sort(key=lambda r: (r["source"], r["theta"], r["achieved_eps"]))
    keys = ["source", "theta", "target_eps", "achieved_eps", "stored_eps", "sde", "sde_chi", "f_stored",
            "n_T", "n_4", "n_R", "n_R_syl", "n_R_resid", "N_phi", "N_phi_formula", "ops",
            "tcost_unitary", "tcost_meas", "stored_ND", "eps_ok", "unitary_exact", "ok", "method",
            "backend", "n_occ", "n_theta", "sources", "path", "dec_s"]
    with open(OUT / "prescribed_counts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("non-unitary:", sum(1 for r in rows if not r["unitary_exact"]),
          " decomposition failures:", sum(1 for r in rows if not r["ok"]))

    # sde of Nick's matrices (for the per-sde comparison)
    ns = nick_sde()
    nick = list(csv.DictReader(open(UNI / "nick_test" / "nick_tcost_2026-09-30.csv")))
    with open(OUT / "nick_sde.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f", "theta", "sde", "sde_chi"])
        miss = 0
        for r in nick:
            k = (int(r["f"]), float(r["theta"]))
            s = ns.get(k)
            miss += s is None
            w.writerow([r["f"], r["theta"], *(s or (None, None))])
    print("nick rows without sde match:", miss)


if __name__ == "__main__":
    main()
