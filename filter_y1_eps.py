#!/usr/bin/env python3
"""Filter a stage-1 Y1 .npy to a tighter eps.

Stage 1 collects all integer triples (m_0, m_1, m_2) with
    |σ_1(Y) - target_t| ≤ epsY_window(eps_pre)
where σ_1(Y) = m_0 + m_1·α_1 + m_2·α_1², α_1 = ζ_9+ζ_9^{-1} = 2cos(2π/9).

A tighter eps (eps_new < eps_pre) produces a strict SUBSET of the rows.
So filtering from cached eps_pre data to eps_new is just one σ_1 check
per row.

Usage:
    python3 filter_y1_eps.py --input Y1_f=4_u=0_eps=0.05.npy --u 0 \
        --f 4 --eps-old 0.05 --eps-new 0.01 \
        --output Y1_f=4_u=0_eps=0.01.npy
"""
from __future__ import annotations
import argparse
import math
import sys
import time
import numpy as np

ALPHA1 = 2.0 * math.cos(2.0 * math.pi / 9.0)  # σ_1 root of t³ - 3t + 1
ALPHA1_SQ = ALPHA1 * ALPHA1
ALPHA2 = 2.0 * math.cos(4.0 * math.pi / 9.0)  # σ_2 root
ALPHA2_SQ = ALPHA2 * ALPHA2
ALPHA4 = 2.0 * math.cos(8.0 * math.pi / 9.0)  # σ_4 root
ALPHA4_SQ = ALPHA4 * ALPHA4


def epsY_window(eps: float, u: float, f: int) -> float:
    base_scale = float(3 ** (2 * f))
    return base_scale * eps * (2.0 * math.sqrt(u) + eps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="input Y1 .npy (eps_old)")
    p.add_argument("--output", required=True, help="output Y1 .npy (eps_new)")
    p.add_argument("--u", type=float, required=True, help="|d_i|^2 target (0 or 1 for Householder)")
    p.add_argument("--f", type=int, required=True, help="internal f (V-denom exponent / 2 in u-denom semantics)")
    p.add_argument("--eps-old", type=float, required=True, help="eps of the input cache (sanity check)")
    p.add_argument("--eps-new", type=float, required=True, help="target tighter eps")
    p.add_argument("--verify", action="store_true",
                   help="also check σ_2, σ_4 ≥ 0 (eps-independent sanity)")
    args = p.parse_args()

    if args.eps_new > args.eps_old:
        sys.exit(f"--eps-new ({args.eps_new}) must be <= --eps-old ({args.eps_old}); "
                 f"this filter is only valid for tightening, not loosening.")

    base_scale = float(3 ** (2 * args.f))
    target_t = base_scale * float(args.u)
    eps_window_old = epsY_window(args.eps_old, args.u, args.f)
    eps_window_new = epsY_window(args.eps_new, args.u, args.f)
    total_scale = 2.0 * base_scale  # for norm=2 (Householder)

    print(f"[filter] input={args.input}")
    print(f"[filter] f={args.f} u={args.u} eps_old={args.eps_old} eps_new={args.eps_new}")
    print(f"[filter] base_scale=3^{2*args.f}={int(base_scale)}  target_t={target_t}")
    print(f"[filter] window_old=±{eps_window_old:.4g}  window_new=±{eps_window_new:.4g}")
    print(f"[filter] expected filter ratio (window_new/window_old)={eps_window_new/eps_window_old:.4f}")

    t0 = time.time()
    arr = np.load(args.input, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.dtype != np.int64:
        sys.exit(f"unexpected input shape/dtype: {arr.shape} {arr.dtype}")
    n_in = arr.shape[0]
    print(f"[filter] input rows: {n_in:,}")

    # σ_1 of each Y = m_0 + m_1*α_1 + m_2*α_1²
    m0 = arr[:, 0].astype(np.float64)
    m1 = arr[:, 1].astype(np.float64)
    m2 = arr[:, 2].astype(np.float64)
    sigma1 = m0 + m1 * ALPHA1 + m2 * ALPHA1_SQ

    abs_err = np.abs(sigma1 - target_t)
    # Tolerance budget: original code uses min(1e-15, 0.1·eps_hi); we'll match.
    tol = min(1e-15, 0.1 * eps_window_new)
    mask = abs_err < eps_window_new + tol

    n_after_sigma1 = int(mask.sum())
    print(f"[filter] after σ_1 filter: {n_after_sigma1:,}  ({100*n_after_sigma1/n_in:.3f}%)")

    if args.verify:
        sigma2 = m0 + m1 * ALPHA2 + m2 * ALPHA2_SQ
        sigma4 = m0 + m1 * ALPHA4 + m2 * ALPHA4_SQ
        ok2 = (sigma2 >= -tol) & (sigma2 <= total_scale + tol)
        ok4 = (sigma4 >= -tol) & (sigma4 <= total_scale + tol)
        n_ok2 = int(ok2.sum())
        n_ok4 = int(ok4.sum())
        print(f"[filter] sanity: σ_2 in [0, total_scale]: {n_ok2:,} rows OK")
        print(f"[filter] sanity: σ_4 in [0, total_scale]: {n_ok4:,} rows OK")
        if n_ok2 != n_in or n_ok4 != n_in:
            print(f"[filter] WARNING: {n_in - n_ok2} σ_2 + {n_in - n_ok4} σ_4 violations in INPUT — "
                  f"input cache may be stale or corrupted.")
        # Don't intersect with σ_2/σ_4 filter (those should already be enforced in
        # the input cache); just report.

    filtered = np.ascontiguousarray(arr[mask], dtype=np.int64)
    np.save(args.output, filtered)
    t1 = time.time()
    print(f"[filter] wrote {filtered.shape[0]:,} rows to {args.output}  ({t1-t0:.2f}s)")


if __name__ == "__main__":
    main()
