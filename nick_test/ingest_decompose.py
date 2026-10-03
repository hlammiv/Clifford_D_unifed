"""ingest_decompose.py — ingest Nick's `fits_f=N.txt` matrix files, check
unitarity and ring membership, and decompose each matrix into Clifford+D
normal form.

Input file format (one matrix per data line):
    #f=N
    # columns: theta, 3^f*Mij (3x3 6-element-groups)
    <theta> , <6 ints> , <6 ints> , ... , <6 ints>   # 9 groups, row-major M

Each entry M_ij is stored as the 6 coefficients of  3^f · M_ij  in the
Z[ζ_9] integral basis (1, ζ, ζ², ζ³, ζ⁴, ζ⁵), ζ = e^{2πi/9}.  So
    M_ij = (1/3^f) · Σ_{k=0}^{5} c_k ζ^k
and M is an element of M_3(Z[ζ_9, 1/3]).

For each matrix the wrapper reports:
  * numeric unitarity   : ||M M† − I||_max with ζ = e^{2πi/9}
  * exact ring unitarity: whether M M† == I *exactly* in Z[ζ_9,1/3]
                          (the rigorous "lives in the ring AND is unitary"
                          test — decomposition only succeeds when this holds)
  * achieved ε          : global-phase-invariant Frobenius distance to the
                          target diag(e^{-iθ/2}, e^{+iθ/2}, 1)  (a Z-rotation
                          on the {0,1} two-level subspace)
  * normal form         : Kalra canonical syllable decomposition via
                          hrsa/canonical_reducer.decompose_canonical, giving
                          N_D (D-gate count), syllable count, trailing
                          Clifford, and exact reconstruction verification.

CLI:
  python3 ingest_decompose.py --input fits_f=10.txt --out-json results_f10.json
  python3 ingest_decompose.py --input fits_f=10.txt --row 35 -v   # single row
  python3 ingest_decompose.py --input fits_f=10.txt fits_f=12.txt fits_f=14.txt
  python3 ingest_decompose.py --input fits_f=10.txt --no-decompose # checks only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_HRSA = _HERE.parent / "hrsa"
for _p in (str(_HERE.parent), str(_HRSA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import canonical_reducer as cr  # noqa: E402

_ZETA9 = np.exp(2j * np.pi / 9)
_BASIS = np.array([_ZETA9**k for k in range(6)])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_fits_file(path: Path):
    """Yield (row_index, theta, groups) and return the file's f.

    `groups` is a list of 9 lists of 6 ints (row-major M entries)."""
    f = None
    rows = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line[1:].strip().startswith("f="):
                    f = int(line.split("f=")[1].split()[0].rstrip(","))
                continue
            head, rest = line.split(",", 1)
            theta = float(head.strip())
            groups = [[int(x) for x in g.split()] for g in rest.split(",")]
            if len(groups) != 9 or any(len(g) != 6 for g in groups):
                raise ValueError(
                    f"{path}:{len(rows)+1}: expected 9 groups of 6 ints, "
                    f"got {[len(g) for g in groups]}")
            rows.append((len(rows), theta, groups))
    if f is None:
        raise ValueError(f"{path}: no '#f=N' header found")
    return f, rows


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def build_complex(groups, f) -> np.ndarray:
    """Reconstruct the complex 3x3 matrix M with ζ_9 = e^{2πi/9}."""
    M = np.zeros((3, 3), dtype=complex)
    for idx, g in enumerate(groups):
        i, j = divmod(idx, 3)
        M[i, j] = np.dot(g, _BASIS) / (3**f)
    return M


def build_ring(groups, f):
    """Build the 3x3 matrix of _FastZ9Frac (exact ring elements)."""
    V = []
    for i in range(3):
        row = []
        for j in range(3):
            c = tuple(cr._INT(int(x)) for x in groups[3 * i + j])
            row.append(cr._FastZ9Frac(cr._FastZ9(c), f))
        V.append(row)
    return V


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def unitarity_numeric(M: np.ndarray) -> float:
    return float(np.max(np.abs(M @ M.conj().T - np.eye(3))))


def unitarity_exact(V):
    """Return (is_unitary, max_offdiag_or_resid_int).  Computes M M† exactly
    over Z[ζ_9,1/3] and checks it equals the identity matrix."""
    Vd = cr._fast_conjugate_transpose(V)
    P = cr._fast_matmul(V, Vd)
    ok = True
    max_resid = 0
    for i in range(3):
        for j in range(3):
            target = cr._FastZ9Frac.one() if i == j else cr._FastZ9Frac.zero()
            diff = P[i][j] - target
            for c in diff.num.coefs:
                ac = int(abs(c))
                if ac > max_resid:
                    max_resid = ac
                if c:
                    ok = False
    return ok, max_resid


def target_unitary(theta: float) -> np.ndarray:
    """diag(e^{-iθ/2}, e^{+iθ/2}, 1) — Z-rotation on the {0,1} subspace."""
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2), 1.0])


def achieved_eps(M: np.ndarray, theta: float):
    """Distance to the target diag(e^{-iθ/2}, e^{+iθ/2}, 1).

    Returns (frob, opnorm).  `frob` is the RAW Frobenius distance
    ‖M − T‖_F with no global-phase freedom — matching v_validate.py's
    `frobenius()` convention (the metric the N_D-vs-ε plots use).  These
    matrices anchor the global phase via M[2][2]=1, so raw == phase-optimal
    to ~1e-22; opnorm is the (phase-optimal) operator-norm distance, for
    reference."""
    T = target_unitary(theta)
    frob = float(np.linalg.norm(M - T, ord="fro"))
    tr = np.trace(T.conj().T @ M)
    phase = tr / abs(tr) if abs(tr) > 0 else 1.0
    opnorm = float(np.linalg.norm(M - phase * T, ord=2))
    return frob, opnorm


# ---------------------------------------------------------------------------
# Per-matrix processing
# ---------------------------------------------------------------------------

def process_matrix(row_idx, theta, groups, f, *,
                   do_decompose=True, greedy=False, verbose=False):
    M = build_complex(groups, f)
    V = build_ring(groups, f)

    num_err = unitarity_numeric(M)
    exact_ok, exact_resid = unitarity_exact(V)
    frob, opnorm = achieved_eps(M, theta)

    rec = {
        "row": row_idx,
        "theta": theta,
        "f": f,
        "unitary_numeric_err": num_err,
        "unitary_exact": exact_ok,
        "unitary_exact_residual": exact_resid,
        "target": "diag(e^{-i*theta/2}, e^{+i*theta/2}, 1)",
        "achieved_frob": frob,
        "achieved_opnorm": opnorm,
    }

    if do_decompose:
        if not exact_ok:
            rec["decompose"] = {
                "attempted": False,
                "reason": "matrix is not exactly unitary over the ring; "
                          "canonical peeling would not terminate at a monomial",
            }
        else:
            t0 = time.time()
            res = cr.decompose_canonical(
                V, verbose=verbose, greedy_single=greedy)
            ver = None
            if res["success"]:
                ver = cr.verify_decomposition(
                    V, res["syllables"], res["trailing_clifford"])
            rec["decompose"] = {
                "attempted": True,
                "success": res["success"],
                "N_D": res["D_count"],
                "n_syllables": len(res["syllables"]),
                "sde_chi_initial": res["sde_chi_initial"],
                "sde_chi_final": res["sde_chi_final"],
                "syllables": res["syllables"],
                "wall_seconds": time.time() - t0,
                "reconstruction_matches": (ver["matches"] if ver else None),
                "reconstruction_max_residual": (
                    ver["max_coef_residual"] if ver else None),
            }
            if "error" in res:
                rec["decompose"]["error"] = res["error"]
    return rec


def _worker(row, *, f, do_decompose, greedy):
    """Picklable multiprocessing entry: (ridx, theta, groups) -> rec.
    The 4374-entry prefix table is built lazily once per worker process."""
    ridx, theta, groups = row
    return process_matrix(ridx, theta, groups, f,
                          do_decompose=do_decompose, greedy=greedy,
                          verbose=False)


def summarize_row(rec) -> str:
    d = rec.get("decompose", {})
    nd = d.get("N_D", "-")
    ns = d.get("n_syllables", "-")
    vmatch = d.get("reconstruction_matches", None)
    vflag = "" if vmatch is None else ("ok" if vmatch else "MISMATCH")
    if d.get("attempted") and not d.get("success", True):
        nd = "FAIL"
    return (f"row {rec['row']:>3}  θ={rec['theta']:+.5f}  "
            f"unit:{'Y' if rec['unitary_exact'] else 'N'}  "
            f"ε_frob={rec['achieved_frob']:.3e}  "
            f"N_D={nd!s:>4}  syl={ns!s:>4}  {vflag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", "-i", nargs="+", required=True, type=Path,
                    help="One or more fits_f=N.txt files.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Write full per-matrix results to this JSON file.")
    ap.add_argument("--row", type=int, default=None,
                    help="Process only this row index (0-based) of each file.")
    ap.add_argument("--sample", type=int, default=None,
                    help="Evenly subsample at most N rows per file (keeps the "
                         "angle spread; for large files where decomposing every "
                         "row is unnecessary for the plot).")
    ap.add_argument("--no-decompose", action="store_true",
                    help="Only check unitarity/ring membership; skip the "
                         "(slower) canonical decomposition.")
    ap.add_argument("--greedy", action="store_true",
                    help="Allow greedy single-prefix fallback in the peeler "
                         "(only needed for very high f).")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="Parallel worker processes for decomposition "
                         "(rows are independent). Default 1 (serial).")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    do_decompose = not args.no_decompose
    if do_decompose and args.jobs == 1:
        cr.get_prefix_table()  # one-time warm in the main process (~1s)

    all_results = {}
    for path in args.input:
        f, rows = parse_fits_file(path)
        if args.row is not None:
            rows = [r for r in rows if r[0] == args.row]
            if not rows:
                print(f"[ingest] {path.name}: no row {args.row}", file=sys.stderr)
                continue
        elif args.sample is not None and len(rows) > args.sample:
            idx = np.linspace(0, len(rows) - 1, args.sample).round().astype(int)
            idx = sorted(set(int(i) for i in idx))
            rows = [rows[i] for i in idx]
            print(f"[ingest] {path.name}: subsampled to {len(rows)} rows",
                  file=sys.stderr)
        print(f"\n=== {path.name}  (f={f}, {len(rows)} matrices) ===")
        n_unit = n_decomp_ok = n_verify_ok = 0
        nd_vals = []

        if args.jobs > 1 and do_decompose:
            work = partial(_worker, f=f, do_decompose=do_decompose,
                           greedy=args.greedy)
            with Pool(processes=args.jobs) as pool:
                file_recs = []
                for rec in pool.imap(work, rows, chunksize=1):
                    file_recs.append(rec)
                    print(summarize_row(rec), flush=True)
            file_recs.sort(key=lambda r: r["row"])
        else:
            file_recs = []
            for (ridx, theta, groups) in rows:
                rec = process_matrix(ridx, theta, groups, f,
                                     do_decompose=do_decompose,
                                     greedy=args.greedy, verbose=args.verbose)
                file_recs.append(rec)
                print(summarize_row(rec), flush=True)

        for rec in file_recs:
            if rec["unitary_exact"]:
                n_unit += 1
            d = rec.get("decompose", {})
            if d.get("success"):
                n_decomp_ok += 1
                if isinstance(d.get("N_D"), int):
                    nd_vals.append(d["N_D"])
            if d.get("reconstruction_matches"):
                n_verify_ok += 1
        all_results[path.name] = {"f": f, "matrices": file_recs}
        print(f"--- {path.name}: {n_unit}/{len(rows)} exactly unitary; "
              f"{n_decomp_ok}/{len(rows)} decomposed; "
              f"{n_verify_ok}/{len(rows)} reconstruction-verified", end="")
        if nd_vals:
            print(f"; N_D min/med/max = {min(nd_vals)}/"
                  f"{int(np.median(nd_vals))}/{max(nd_vals)}")
        else:
            print()

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\n[ingest] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
