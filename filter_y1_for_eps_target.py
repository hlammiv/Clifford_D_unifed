#!/usr/bin/env python3
"""Filter Y1 .npy files for a target ε_target on the V matrix's Frobenius distance.

Different from filter_y1_eps.py: the latter applied stage-1's σ_1 strip filter
at a tighter eps. That doesn't actually prune at f=4 (the strip is non-binding
because the exact-ideal-screen does all the work).

THIS filter exploits a θ-AGNOSTIC LOSSLESS criterion: best_frob(Y_a,Y_b,Y_c)
is determined by σ_1 norms alone (no phases). Triples whose best_frob > ε_target
cannot achieve frob ≤ ε_target for any θ — so we can prune them losslessly.

Per-Y thresholds (conservative budget split: each contribution ≤ ε²/3):

  σ_1(Y_a)               ≤  ε² · 3^{2f} / 12     (4 off-diags from |a_a|·|a_b/c|)
  |1 - σ_1(Y_b)/3^{2f}|  ≤  ε / √3                (V[1,0] off-diag = 1 - |a_b|²/3^{2f})
  |1 - σ_1(Y_c)/3^{2f}|  ≤  ε / √3                (V[0,1] same)

These are necessary conditions: a triple violating any cannot have best_frob ≤ ε.
A triple satisfying all three has best_frob² ≤ ε² + corrections (the
diagonal-magnitude-mismatch and small-z_a quartic terms, both quadratic in ε).

Usage:
    # For Y_a (u=0 small slot):
    python3 filter_y1_for_eps_target.py --input Y1_f=4_u=0_eps=0.05.npy \
        --output Y1_f=4_u=0_eps_target=0.001.npy --u 0 --f 4 --eps-target 0.001

    # For Y_b/Y_c (u=1 large slot):
    python3 filter_y1_for_eps_target.py --input Y1_f=4_u=1_eps=0.05.npy \
        --output Y1_f=4_u=1_eps_target=0.001.npy --u 1 --f 4 --eps-target 0.001
"""
from __future__ import annotations
import argparse
import math
import sys
import time
import numpy as np

ALPHA1 = 2.0 * math.cos(2.0 * math.pi / 9.0)
ALPHA1_SQ = ALPHA1 * ALPHA1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--u", type=float, required=True, help="0 (small slot) or 1 (large slot)")
    p.add_argument("--f", type=int, required=True)
    p.add_argument("--eps-target", type=float, required=True,
                   help="Target Frobenius distance for V (the QUERY ε you want to support)")
    p.add_argument("--budget-frac", type=float, default=1.0/3.0,
                   help="Budget fraction allocated to this Y's contribution. "
                        "Default 1/3 (3-way split with the other two Y's). Smaller "
                        "= tighter (more conservative) threshold.")
    args = p.parse_args()

    base_scale = float(3 ** (2 * args.f))
    eps2 = args.eps_target ** 2
    budget = float(args.budget_frac)

    arr = np.load(args.input, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.dtype != np.int64:
        sys.exit(f"unexpected input shape/dtype: {arr.shape} {arr.dtype}")
    n_in = arr.shape[0]

    m0 = arr[:, 0].astype(np.float64)
    m1 = arr[:, 1].astype(np.float64)
    m2 = arr[:, 2].astype(np.float64)
    sigma1 = m0 + m1 * ALPHA1 + m2 * ALPHA1_SQ

    if args.u == 0.0:
        # Y_a (small slot). |V[0,2]|² + |V[1,2]|² + |V[2,0]|² + |V[2,1]|² ≈ 4·σ_1(Y_a)/3^{2f}.
        # Constraint: 4·σ_1(Y_a)/3^{2f} ≤ budget · ε² → σ_1 ≤ budget·ε²·3^{2f}/4
        threshold = budget * eps2 * base_scale / 4.0
        mask = np.abs(sigma1) <= threshold
        kind = "small_slot (Y_a, u=0)"
        rule = f"|σ_1(Y_a)| ≤ {threshold:.6g}"
    elif args.u == 1.0:
        # Y_b or Y_c (large slot). |V[1,0]|² = (1 - σ_1(Y_b)/3^{2f})².
        # Constraint: (1 - σ_1/3^{2f})² ≤ budget · ε² → |1 - σ_1/3^{2f}| ≤ √budget · ε
        thresh_rel = math.sqrt(budget) * args.eps_target
        target_t = base_scale  # σ_1 should be ≈ 3^{2f}
        rel_dev = np.abs(sigma1 - target_t) / base_scale
        mask = rel_dev <= thresh_rel
        kind = "large_slot (Y_b/Y_c, u=1)"
        rule = f"|σ_1/3^{{2f}} - 1| ≤ {thresh_rel:.6g}  (σ_1 ∈ [{base_scale*(1-thresh_rel):.4f}, {base_scale*(1+thresh_rel):.4f}])"
    else:
        sys.exit(f"--u must be 0 or 1, got {args.u}")

    n_kept = int(mask.sum())
    t0 = time.time()
    filtered = np.ascontiguousarray(arr[mask], dtype=np.int64)
    np.save(args.output, filtered)
    t1 = time.time()

    print(f"[filter_eps_target] kind={kind}")
    print(f"[filter_eps_target] f={args.f}  ε_target={args.eps_target}  budget_frac={budget:.4f}")
    print(f"[filter_eps_target] rule: {rule}")
    print(f"[filter_eps_target] input rows: {n_in:,}")
    print(f"[filter_eps_target] kept:       {n_kept:,}  ({100*n_kept/n_in:.4f}%)")
    print(f"[filter_eps_target] prune:      {n_in/max(n_kept,1):.2f}×")
    print(f"[filter_eps_target] wrote {filtered.shape[0]:,} rows to {args.output}  ({t1-t0:.2f}s)")


if __name__ == "__main__":
    main()
