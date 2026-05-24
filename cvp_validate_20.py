#!/usr/bin/env python3
"""cvp_validate_20.py — 20-cell validation sweep for the Phase 5 driver.

Picks 10 θ values × 2 ε levels = 20 cells. Runs ``cvp_compile`` on each
and writes per-cell success / N_D / wall to ``cvp_validate_results.csv``.
Where possible, attaches HRSA / zeta9 baseline N_D from existing sweeps
for ratio analysis.

Usage::

    python3 cvp_validate_20.py [--out cvp_validate_results.csv] \\
                              [--max-f-iters 8] \\
                              [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

_UNIFIED_DIR = Path(__file__).resolve().parent
if str(_UNIFIED_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIFIED_DIR))

from cvp_compile import cvp_compile, f_start_recommend
from cvp.diophantine import _NormEqWorker


# Fixed θ list per the prompt.
_THETA_LIST = [
    0.05, 0.5, 1.0, math.pi / 2, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5,
]

# ε levels per the prompt. The Phase 3 ring-unitary construction empirically
# tops out around ε ~ 10⁻³ on lucia (PARI norm-equation hit rate drops to
# ~0% at f >= 9). At ε ∈ {1e-4, 1e-5} we expect mostly failures with the
# current Phase 3; the data documents the ceiling honestly so Phase 6 can
# decide whether to escalate Phase 3 or fall back to SK at tight ε.
_EPS_LIST = [1e-4, 1e-5]


# ---------------------------------------------------------------------------
# HRSA baseline lookup
# ---------------------------------------------------------------------------


def _load_hrsa_baseline(csv_path: Path) -> dict:
    """Load HRSA sweep summary CSV into a dict keyed by (theta_round, eps).

    The HRSA summary uses theta values from the 100-θ grid; we round to
    6 decimal places for matching.
    """
    if not csv_path.exists():
        return {}
    out: dict = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                theta = float(row["theta"])
                eps = float(row["epsilon"])
                nd = int(row["N_D"])
                if nd < 0:
                    continue  # failed cell
                key = (round(theta, 6), eps)
                # Keep best (lowest N_D) per cell if duplicates exist.
                if key not in out or nd < out[key]["N_D"]:
                    out[key] = {
                        "N_D": nd,
                        "method": row.get("method", ""),
                        "f": row.get("f", ""),
                        "wall_s": float(row.get("wall_s", 0) or 0),
                    }
            except (ValueError, KeyError):
                continue
    return out


def _closest_hrsa_match(theta: float, eps: float, hrsa: dict) -> Optional[dict]:
    """Find the HRSA cell with the closest θ at the matching ε.

    HRSA's θ grid is uniform 100-θ in [0, 2π); ours is 10 hand-picked.
    Tolerance: |Δθ| < 0.04 (≈ half the HRSA grid spacing 2π/100=0.063).
    """
    if not hrsa:
        return None
    best_diff = float("inf")
    best = None
    for (t, e), rec in hrsa.items():
        if abs(e - eps) > 1e-12:
            continue
        d = abs(t - theta)
        if d < best_diff:
            best_diff = d
            best = rec
    if best is None or best_diff > 0.04:
        return None
    return {**best, "matched_theta_diff": best_diff}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def _run_one_cell(
    theta: float,
    eps: float,
    *,
    worker,
    max_f_iters: int,
    max_x1_to_try: int,
    max_pairs_per_x1: int,
    max_x3: int,
    verbose: bool,
) -> dict:
    t0 = time.time()
    err: Optional[str] = None
    try:
        rec = cvp_compile(
            theta=theta, eps=eps,
            max_f_iters=max_f_iters,
            max_x1_to_try=max_x1_to_try,
            max_pairs_per_x1=max_pairs_per_x1,
            max_x3=max_x3,
            worker=worker,
            verbose=verbose,
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        rec = None

    wall = time.time() - t0
    if rec is None:
        return {
            "success": False,
            "N_D": -1,
            "wall_s": wall,
            "f_hit": None,
            "achieved_frob": None,
            "attempted_f_levels": [],
            "error": err or "no record",
        }
    return {
        "success": bool(rec["achieved"]["success"]),
        "N_D": int(rec["decomposition"]["N_D"]) if rec["achieved"]["success"] else -1,
        "wall_s": wall,
        "f_hit": (int(rec["achieved"]["f_level"])
                  if rec["achieved"].get("f_level") is not None else None),
        "achieved_frob": (float(rec["achieved"]["achieved_frob"])
                          if rec["achieved"].get("achieved_frob") is not None else None),
        "attempted_f_levels": rec.get("attempted_f_levels", []),
        "error": (rec["errors"][0] if rec.get("errors") else None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str,
                    default=str(_UNIFIED_DIR / "cvp_validate_results.csv"))
    ap.add_argument("--max-f-iters", type=int, default=8,
                    help="f-levels to try beyond f_start.")
    ap.add_argument("--max-x1", type=int, default=8,
                    help="x_1 candidates per f-level (smaller = faster).")
    ap.add_argument("--max-pairs", type=int, default=4,
                    help="Pairs returned per x_1 (small since we stop at first hit).")
    ap.add_argument("--max-x3", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only first N cells (smoke).")
    ap.add_argument("--eps", type=float, default=None,
                    help="Override eps list; run only this ε.")
    ap.add_argument("--eps-list", type=str, default=None,
                    help="Comma-sep list of ε values (overrides --eps).")
    ap.add_argument("--hrsa-csv", type=str,
                    default=str(_UNIFIED_DIR / "sweep_hrsa_grid_2026-05-22" /
                                "summary.csv"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    hrsa_csv = Path(args.hrsa_csv)
    hrsa = _load_hrsa_baseline(hrsa_csv)
    print(f"# HRSA baseline: {len(hrsa)} cells loaded from {hrsa_csv}",
          file=sys.stderr)

    if args.eps_list:
        eps_list = [float(x) for x in args.eps_list.split(",")]
    elif args.eps:
        eps_list = [args.eps]
    else:
        eps_list = list(_EPS_LIST)
    cells = [(t, e) for e in eps_list for t in _THETA_LIST]
    if args.limit:
        cells = cells[:args.limit]

    print(f"# running {len(cells)} cells: |θ|={len(_THETA_LIST)} × |ε|={len(eps_list)}",
          file=sys.stderr)

    rows = []
    header = [
        "theta", "epsilon", "cvp_success", "cvp_N_D", "cvp_wall", "cvp_f_hit",
        "cvp_achieved_frob", "cvp_f_start", "cvp_f_levels_tried",
        "hrsa_N_D", "hrsa_method", "hrsa_f", "hrsa_wall",
        "hrsa_theta_diff", "notes",
    ]
    # Stream rows to disk as we go so a crash leaves partial data.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=header)
        writer.writeheader()
        csvf.flush()

        worker = _NormEqWorker()
        worker.start()
        try:
            for idx, (theta, eps) in enumerate(cells):
                f_start = f_start_recommend(eps)
                t0 = time.time()
                print(
                    f"[cell {idx+1}/{len(cells)}] θ={theta:.4f} ε={eps:.0e} "
                    f"f_start={f_start} ...",
                    file=sys.stderr, flush=True,
                )
                result = _run_one_cell(
                    theta=theta, eps=eps, worker=worker,
                    max_f_iters=args.max_f_iters,
                    max_x1_to_try=args.max_x1,
                    max_pairs_per_x1=args.max_pairs,
                    max_x3=args.max_x3,
                    verbose=args.verbose,
                )
                dt = time.time() - t0

                hrsa_match = _closest_hrsa_match(theta, eps, hrsa)
                notes = []
                if result["error"]:
                    notes.append(f"err:{result['error'][:80]}")
                if hrsa_match is None:
                    notes.append("no_hrsa_baseline")

                f_levels = ",".join(
                    str(d.get("f")) for d in result.get("attempted_f_levels", [])
                )
                row = {
                    "theta": f"{theta:.10f}",
                    "epsilon": f"{eps:.6e}",
                    "cvp_success": int(result["success"]),
                    "cvp_N_D": result["N_D"],
                    "cvp_wall": f"{result['wall_s']:.3f}",
                    "cvp_f_hit": (result["f_hit"]
                                  if result["f_hit"] is not None else ""),
                    "cvp_achieved_frob": (
                        f"{result['achieved_frob']:.6e}"
                        if result["achieved_frob"] is not None else ""
                    ),
                    "cvp_f_start": f_start,
                    "cvp_f_levels_tried": f_levels,
                    "hrsa_N_D": (hrsa_match["N_D"] if hrsa_match else ""),
                    "hrsa_method": (hrsa_match["method"] if hrsa_match else ""),
                    "hrsa_f": (hrsa_match["f"] if hrsa_match else ""),
                    "hrsa_wall": (f"{hrsa_match['wall_s']:.2f}"
                                  if hrsa_match else ""),
                    "hrsa_theta_diff": (
                        f"{hrsa_match['matched_theta_diff']:.4f}"
                        if hrsa_match else ""
                    ),
                    "notes": ";".join(notes),
                }
                writer.writerow(row)
                csvf.flush()
                rows.append(row)

                tag = ("HIT" if result["success"] else "MISS")
                hrsa_tag = (f"HRSA N_D={hrsa_match['N_D']}"
                            if hrsa_match else "no-HRSA")
                print(
                    f"  -> {tag} f_hit={result['f_hit']} "
                    f"N_D={result['N_D']} frob={result['achieved_frob']} "
                    f"wall={dt:.1f}s  ({hrsa_tag})",
                    file=sys.stderr, flush=True,
                )
        finally:
            worker.close()

    # ----- summary -----
    print(f"\n# wrote {out_path}", file=sys.stderr)
    print(file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"SUMMARY ({len(rows)} cells)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    for eps in eps_list:
        cells_at_eps = [r for r in rows if abs(float(r["epsilon"]) - eps) < 1e-12]
        succ = [r for r in cells_at_eps if int(r["cvp_success"])]
        print(f"ε={eps:.0e}: {len(succ)}/{len(cells_at_eps)} succeeded",
              file=sys.stderr)
        if succ:
            walls = sorted(float(r["cvp_wall"]) for r in succ)
            print(f"  wall (s): min={walls[0]:.1f} median={walls[len(walls)//2]:.1f} "
                  f"max={walls[-1]:.1f}", file=sys.stderr)
            ratios = []
            for r in succ:
                if r["hrsa_N_D"]:
                    cvp_nd = int(r["cvp_N_D"])
                    hrsa_nd = int(r["hrsa_N_D"])
                    if hrsa_nd > 0:
                        ratios.append(cvp_nd / hrsa_nd)
            if ratios:
                rs = sorted(ratios)
                print(
                    f"  CVP_N_D / HRSA_N_D (n={len(ratios)}): "
                    f"min={rs[0]:.2f} median={rs[len(rs)//2]:.2f} max={rs[-1]:.2f}",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
