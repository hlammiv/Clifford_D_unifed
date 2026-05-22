#!/usr/bin/env python3
"""hrsa_baseline.py — Characterize HRSA's N_D vs epsilon for fixed theta.

Goal: establish the N_D(eps) curve from HRSA's Householder synthesizer at a
handful of canonical angles, so we know what a recursive (SK-style) compiler
has to beat.

CLI of HRSA_tester (positional):
    ./HRSA_tester theta epsilon max_f [c] [flags]
JSON output is enabled via --json PATH.  Top-level JSON shape (from
/home/hlamm/Desktop/efficent_gates/unified/hrsa/HRSA_test.cpp ~line 463):

    {
      "identification": {...},
      "inputs": {"theta":..., "epsilon":..., "max_f":..., ...},
      "achieved": {"success": bool, "achieved_frob": float|null,
                   "epsilon_passed": bool, "f_level": int|null,
                   "method": "SignExtClifford"|"Direct"|"bidir(K+K)"|"HRSA(f=F)"|"none"},
      "unitary": {... "f": int ...} | null,
      "decomposition": {"N_D": int, "N_D_only":..., "N_R_only":...,
                        "sde_chi_initial":..., "syllables":[...]} | {"N_D":int,...},
      "sanity_checks": {"unitarity_residual": float, ...},
      "performance": {"wall_seconds": float, ...},
      "errors": [...]
    }

We extract N_D from decomposition.N_D, f from achieved.f_level (fallback
unitary.f), phase from achieved.method, walltime from performance.wall_seconds.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HRSA_DIR = Path("/home/hlamm/Desktop/efficent_gates/unified/hrsa")
HRSA_BIN = HRSA_DIR / "HRSA_tester"


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------
def ensure_binary(verbose: bool = True) -> Path:
    """Ensure HRSA_tester is built.  Run `make HRSA_tester` if missing."""
    if HRSA_BIN.exists() and os.access(HRSA_BIN, os.X_OK):
        if verbose:
            print(f"[hrsa_baseline] HRSA_tester present at {HRSA_BIN}")
        return HRSA_BIN
    if verbose:
        print(f"[hrsa_baseline] HRSA_tester not found; running `make HRSA_tester`...")
    rc = subprocess.run(
        ["make", "HRSA_tester"],
        cwd=str(HRSA_DIR),
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"make HRSA_tester failed (rc={rc})")
    if not HRSA_BIN.exists():
        raise RuntimeError(f"build succeeded but {HRSA_BIN} not found")
    return HRSA_BIN


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def run_hrsa(theta: float, epsilon: float, max_f: int = 8,
             timeout_s: float = 300.0, extra_args=None) -> dict:
    """Invoke HRSA_tester once and parse its --json output.

    Returns a dict:
        {'N_D': int|None, 'walltime': float, 'phase': str, 'f': int|None,
         'raw': dict|None, 'error': str|None}
    On timeout / non-zero rc / JSON parse failure, N_D is None and
    'error' carries a short reason.  Walltime is wall-clock of the
    subprocess (so we still report time even if the run failed).
    """
    bin_path = ensure_binary(verbose=False)
    with tempfile.NamedTemporaryFile(
        prefix="hrsa_baseline_", suffix=".json", delete=False,
    ) as tmp:
        json_path = tmp.name

    cmd = [
        str(bin_path),
        f"{theta:.17g}",
        f"{epsilon:.17g}",
        str(int(max_f)),
        "--json", json_path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    err = None
    proc = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(HRSA_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        rc = proc.returncode
        if rc != 0:
            err = f"rc={rc}"
    except subprocess.TimeoutExpired:
        err = f"timeout({timeout_s}s)"
        rc = None
    wall = time.time() - t0

    raw = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if err is None:
                err = f"json_parse: {e}"
        finally:
            try:
                os.unlink(json_path)
            except OSError:
                pass

    n_d = None
    phase = "none"
    f_level = None
    if raw is not None:
        achieved = raw.get("achieved", {}) or {}
        decomp = raw.get("decomposition", {}) or {}
        unitary = raw.get("unitary", {}) or {}
        phase = achieved.get("method", "none") or "none"
        f_level = achieved.get("f_level")
        if f_level is None and isinstance(unitary, dict):
            f_level = unitary.get("f")
        # N_D may live under decomposition.N_D in both shapes.
        nd_raw = decomp.get("N_D")
        if isinstance(nd_raw, (int, float)):
            n_d = int(nd_raw)
        # Prefer the binary's reported wall time when available.
        perf = raw.get("performance", {}) or {}
        if isinstance(perf.get("wall_seconds"), (int, float)):
            wall = float(perf["wall_seconds"])
        # If HRSA didn't report success, force N_D back to None.
        if not achieved.get("success", False):
            n_d = None
            if err is None:
                err = f"no_success(method={phase})"

    return {
        "N_D": n_d,
        "walltime": wall,
        "phase": phase,
        "f": f_level,
        "raw": raw,
        "error": err,
    }


def sweep_epsilon(theta: float, eps_list, max_f: int = 8,
                  timeout_s: float = 300.0, verbose: bool = True,
                  extra_args=None) -> list:
    """Run HRSA across a list of epsilon values for fixed theta.

    Returns a list of dicts (one per eps), each augmented with 'theta'
    and 'epsilon' for easy CSV emission.  Continues on individual failures.
    """
    results = []
    for eps in eps_list:
        if verbose:
            print(f"[hrsa_baseline] theta={theta:.6f}  eps={eps:g}  max_f={max_f}  "
                  f"timeout={timeout_s}s ...", flush=True)
        try:
            res = run_hrsa(theta, eps, max_f=max_f, timeout_s=timeout_s,
                           extra_args=extra_args)
        except Exception as e:
            # Last-ditch guard: even if run_hrsa itself blows up (e.g. build
            # failure), keep sweeping.  Record None.
            res = {"N_D": None, "walltime": 0.0, "phase": "error",
                   "f": None, "raw": None, "error": f"exception: {e}"}
        rec = {
            "theta": theta,
            "epsilon": eps,
            "N_D": res["N_D"],
            "walltime": res["walltime"],
            "phase": res["phase"],
            "f": res["f"],
            "error": res.get("error"),
        }
        if verbose:
            print(f"    -> N_D={rec['N_D']}  phase={rec['phase']}  "
                  f"f={rec['f']}  wall={rec['walltime']:.2f}s"
                  + (f"  err={rec['error']}" if rec.get("error") else ""),
                  flush=True)
        results.append(rec)
    return results


def save_baseline(results, path) -> None:
    """Write CSV with columns: theta, epsilon, N_D, walltime, phase, f."""
    path = str(path)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["theta", "epsilon", "N_D", "walltime", "phase", "f"])
        for r in results:
            w.writerow([
                f"{r['theta']:.17g}",
                f"{r['epsilon']:g}",
                "" if r["N_D"] is None else int(r["N_D"]),
                f"{r['walltime']:.3f}",
                r["phase"],
                "" if r["f"] is None else int(r["f"]),
            ])
    print(f"[hrsa_baseline] wrote {path} ({len(results)} rows)")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def linear_fit(xs, ys):
    """Plain least-squares fit y = a + b*x; ignores (x,y) pairs with y None.
    Returns (a, b, n) where n is the count of pairs used.  None,None,0 if <2.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    n = len(pairs)
    if n < 2:
        return None, None, n
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None, None, n
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b, n


def print_summary_table(sweeps, theta_labels, eps_list):
    """sweeps: dict[label] -> list of result dicts."""
    header_cols = ["epsilon"] + [f"N_D({lbl})" for lbl in theta_labels]
    widths = [max(10, len(h)) for h in header_cols]
    print()
    print("=" * (sum(widths) + 3 * len(widths)))
    print("Summary: N_D by (epsilon, theta)")
    print("=" * (sum(widths) + 3 * len(widths)))
    print("  ".join(h.ljust(w) for h, w in zip(header_cols, widths)))
    print("-" * (sum(widths) + 3 * len(widths)))
    for i, eps in enumerate(eps_list):
        row = [f"{eps:g}"]
        for lbl in theta_labels:
            v = sweeps[lbl][i]["N_D"]
            row.append("-" if v is None else str(v))
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print()


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Baseline N_D vs epsilon sweep for HRSA at canonical thetas."
    )
    ap.add_argument("--out", default="/tmp/hrsa_baseline.csv",
                    help="CSV output path (default /tmp/hrsa_baseline.csv).")
    ap.add_argument("--max-f", type=int, default=8,
                    help="HRSA max_f (V-denominator exponent cap).  Default 8.")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="Per-run timeout in seconds.  Default 300.")
    ap.add_argument("--thetas", type=str,
                    default="pi/2,pi/3,2pi/7",
                    help=("Comma-separated theta list.  Supports pi, /, *, "
                          "and decimals.  Default: pi/2,pi/3,2pi/7"))
    ap.add_argument("--eps-list", type=str,
                    default="0.5,0.3,0.1,0.05,0.03,0.01,0.005,0.001,5e-4,1e-4",
                    help="Comma-separated epsilon list.")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick sanity run: 2 eps values, theta=pi/2 only.")
    args = ap.parse_args()

    def parse_theta(token: str) -> tuple:
        """Return (label, float).  Accepts 'pi/2', '2pi/7', '0.5', etc.
        We insert explicit '*' around 'pi' so '2pi/7' becomes '2*pi/7'."""
        import re
        t = token.strip()
        label = t
        # Insert '*' between a digit/closing-paren and 'pi', and between
        # 'pi' and an opening-paren or digit (the latter is rare but safe).
        x = re.sub(r'(\d|\))\s*pi', r'\1*pi', t)
        x = re.sub(r'pi\s*(\d|\()', r'pi*\1', x)
        x = x.replace("pi", f"({math.pi})")
        try:
            val = float(eval(x, {"__builtins__": {}}, {}))
        except Exception as e:
            raise SystemExit(f"could not parse theta '{token}': {e}")
        return label, val

    thetas = [parse_theta(s) for s in args.thetas.split(",") if s.strip()]
    eps_list = [float(s) for s in args.eps_list.split(",") if s.strip()]

    if args.smoke:
        thetas = thetas[:1]
        eps_list = [0.1, 0.01]

    # 1) Ensure binary
    ensure_binary(verbose=True)

    # 2) Sweep each theta
    sweeps = {}
    all_rows = []
    for label, theta_val in thetas:
        print(f"\n=== theta = {label} ({theta_val:.6f} rad) ===", flush=True)
        rows = sweep_epsilon(theta_val, eps_list,
                             max_f=args.max_f, timeout_s=args.timeout,
                             verbose=True)
        sweeps[label] = rows
        all_rows.extend(rows)

    # 3) CSV
    save_baseline(all_rows, args.out)

    # 4) Summary table
    print_summary_table(sweeps, [lbl for lbl, _ in thetas], eps_list)

    # 5) Linear fit on the canonical pi/2 sweep (if present): N_D vs log10(1/eps).
    fit_label = thetas[0][0] if thetas else None
    target_label = "pi/2" if "pi/2" in sweeps else fit_label
    if target_label in sweeps:
        rows = sweeps[target_label]
        xs = [math.log10(1.0 / r["epsilon"]) for r in rows]
        ys = [r["N_D"] for r in rows]
        a, b, n = linear_fit(xs, ys)
        if a is None:
            print(f"[hrsa_baseline] fit for theta={target_label}: "
                  f"insufficient data ({n} points).")
        else:
            print(f"[hrsa_baseline] fit (theta={target_label}, N_D ~ a + b*log10(1/eps)): "
                  f"a = {a:.3f}, b = {b:.3f}  (n={n})")
            # one-line budget statement at eps=1e-4
            eps_target = 1e-4
            row_1em4 = next((r for r in rows
                             if abs(r["epsilon"] - eps_target) < 1e-12), None)
            if row_1em4 and row_1em4["N_D"] is not None:
                print(f"[hrsa_baseline] N_D budget @ eps=1e-4, theta={target_label}: "
                      f"N_D = {row_1em4['N_D']}  "
                      f"(this is what recursive synthesis must beat for tighter eps).")
            else:
                # Use the fit projection
                proj = a + b * math.log10(1.0 / eps_target)
                print(f"[hrsa_baseline] N_D budget @ eps=1e-4 (projected, theta={target_label}): "
                      f"N_D ~ {proj:.1f}  (HRSA did not return a value at eps=1e-4).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
