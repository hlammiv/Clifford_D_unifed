"""End-to-end Euler-SK integration test on z-rotation targets.

Runs sk_driver_v2 on R^Z_{(0,1)}(theta) for theta in {pi/4, pi/2, 2*pi/3, pi},
depth 0..3, and prints the depth → error → N_D table.  Provides the headline
numbers for the report.

Compare against the HRSA-alone baseline (~30 N_D at eps=0.05 for these angles).

Usage
-----
  python sk_v2_integration_test.py                # full 4-angle × depth-0..3 sweep
  python sk_v2_integration_test.py --quick        # depth-0 only on 2 angles (smoke)
  python sk_v2_integration_test.py --max-depth 1  # cap recursion at depth 1
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sk_driver_v2 import sk_synthesize_v2
from sk_leaves import _build_RZ, hrsa_cache_stats


def run_sweep(angles, max_depth: int, eps_total: float = 0.05):
    print(f"\nsk_v2_integration_test: {len(angles)} angles × depth 0..{max_depth}")
    print("=" * 100)
    print(f"{'theta':>6}  {'depth':>5}  {'err':>11}  {'N_D':>5}  "
          f"{'tree':>4}  {'base_calls':>10}  {'leaves':>6}  {'t (s)':>7}  "
          f"{'cache (h/m)':>11}  {'note':<24}")
    print("-" * 100)

    rows = []
    for (label, theta) in angles:
        target = _build_RZ(0, 1, theta)
        for d in range(max_depth + 1):
            res = sk_synthesize_v2(target, depth=d, eps_total=eps_total)
            err = res.get("final_error")
            err_s = f"{err:.3e}" if err is not None else "n/a"
            note = (res.get("note") or "").replace("\n", " ")[:24]
            stats = hrsa_cache_stats()
            cache_s = f"{stats['hit']}/{stats['miss']}"
            print(f"{label:>6}  {d:>5}  {err_s:>11}  "
                  f"{(res.get('N_D') if res.get('N_D') is not None else -1):>5}  "
                  f"{res['tree_size']:>4}  {res['base_case_calls']:>10}  "
                  f"{res.get('n_leaves', 0):>6}  {res['wall_seconds']:>7.2f}  "
                  f"{cache_s:>11}  {note:<24}", flush=True)
            rows.append({
                "theta_label": label, "theta": theta, "depth": d,
                "final_error": err, "N_D": res.get("N_D"),
                "wall_seconds": res["wall_seconds"],
                "tree_size": res["tree_size"],
                "base_case_calls": res["base_case_calls"],
                "n_leaves": res.get("n_leaves", 0),
                "note": note,
            })
        print("-" * 100)

    return rows


def summary(rows):
    print("\nSummary (best depth per angle):")
    print(f"  {'theta':>6}  {'best_depth':>10}  {'best_err':>11}  {'N_D':>5}")
    by_angle = {}
    for r in rows:
        if r["final_error"] is None or r["N_D"] is None:
            continue
        prev = by_angle.get(r["theta_label"])
        if prev is None or r["final_error"] < prev["final_error"]:
            by_angle[r["theta_label"]] = r
    for label, r in by_angle.items():
        print(f"  {label:>6}  {r['depth']:>10}  {r['final_error']:>11.3e}  {r['N_D']:>5}")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="depth 0 only on 2 angles, for smoke testing")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--eps-total", type=float, default=0.05)
    p.add_argument("--angles", type=int, default=4,
                   help="how many of {pi/4, pi/2, 2pi/3, pi} to test (default 4)")
    args = p.parse_args()

    all_angles = [
        ("π/4",  math.pi / 4.0),
        ("π/2",  math.pi / 2.0),
        ("2π/3", 2.0 * math.pi / 3.0),
        ("π",    math.pi),
    ]
    if args.quick:
        angles = all_angles[:2]
        rows = run_sweep(angles, max_depth=0, eps_total=args.eps_total)
    else:
        angles = all_angles[: max(1, min(args.angles, 4))]
        rows = run_sweep(angles, max_depth=args.max_depth,
                         eps_total=args.eps_total)
    summary(rows)


if __name__ == "__main__":
    _cli()
