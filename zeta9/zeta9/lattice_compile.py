"""End-to-end compilation via lattice-round → norm-filter → HRSA decompose.

Pipeline for one (θ, ε, f) cell:

  1. Compute the Householder target u_target = (1/√2)(e^{iθ/2}, -1, 0) for the
     (0,1)-coordinate qutrit rotation R^Z(θ).

  2. Call zeta9.lattice_round to find integer ringZ9 coefficient triples a ∈ Z^18
     whose σ_1-embedding scaled by 1/3^f is close to u_target.

  3. For each candidate, compute the EXACT |u|² in Z[α] (real subfield basis).
     Filter to those satisfying |u|² ≈ 2 within tolerance.

  4. Build V = X_(0,1) · (I - u u†) as a ringZ9chi matrix.

  5. Optionally invoke decompose_tool (or HRSA decompose directly) to count gates.

  6. Return the best-N_D candidate.
"""

from __future__ import annotations
import cmath
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .lattice_round import (
    minkowski_basis_one_entry,
    round_to_zeta9_row,
)


# ---------------------------------------------------------------------------
#  Householder target
# ---------------------------------------------------------------------------

def householder_target_for_R01(theta: float) -> np.ndarray:
    """Return the u-vector for the (0,1)-axis qutrit z-rotation R^Z_{(0,1)}(θ).

    By HRSA convention, R^Z_{(0,1)}(θ) = diag(e^{-iθ/2}, e^{+iθ/2}, 1)
    is approximated as X_(0,1) · (I - u u†) where the Householder vector is
    u = (1/√2)(e^{+iθ/2}, -1, 0).  But for the lattice_round we want the
    *first row of V*, which corresponds to the THIRD entry of u in some
    parameterisations.

    For our purpose: target the "x_1 component" of HRSA's (X_1, X_2, X_3)
    Householder triple.  Per memory `theta_convention.md` and HRSA source:
        x_1 ~ e^{+iθ/2}
        x_2 ~ -1
        x_3 ~ 0
    The full u-vector u in (1/3^f)·Z[ζ_9]^3 satisfies |u|² = 2.
    """
    return np.array([
        cmath.exp(0.5j * theta),
        -1.0 + 0.0j,
        0.0 + 0.0j,
    ])


# ---------------------------------------------------------------------------
#  Norm computation (exact, in Z[α])
# ---------------------------------------------------------------------------

def _z9_norm_squared(a: np.ndarray) -> tuple[int, int, int]:
    """Compute |x|² in Z[α] = Z[ζ_9 + ζ_9⁻¹] for x = sum_k a_k ζ_9^k.

    Returns (m0, m1, m2) such that |x|² = m0 + m1·α + m2·α².
    α = ζ_9 + ζ_9⁻¹.

    Per zeta9's z9_norm_m012 (search_diagonal_matrix... line 194):
      With c = x · conj(x) in ringZ9 basis:
        m2 = -c[4]
        m1 = -c[5]
        m0 =  c[0] + 2·c[4]
    """
    # conj(x) in ringZ9 basis: zeta_9 → zeta_9^{-1} = zeta_9^8 → reduce via Φ_9
    # Equivalent to: bx[k] = a[-k mod 9] coefficient under reduction.
    # Quick implementation: use ringZ9 reduction tables.
    # bx[k] satisfies: conj(x) = sum_k a[k] zeta^{-k} = sum_k a[k] zeta^{9-k}
    # For k in 0..5: 9-k in {9,8,7,6,5,4} → need to reduce 9,8,7,6.
    # zeta^9 = 1; zeta^8 = -zeta^2 - zeta^5; zeta^7 = -zeta - zeta^4; zeta^6 = -1 - zeta^3.

    a = np.asarray(a, dtype=np.int64)
    # conj contribution: a[0]→pos 0 (via zeta^0); a[1]→pos 8 (zeta^8); ...
    cx = np.zeros(6, dtype=np.int64)
    # k=0: zeta^0 = 1 → pos 0 +a[0]
    cx[0] += a[0]
    # k=1: zeta^{-1} = zeta^8 = -zeta^2 - zeta^5 → pos 2: -a[1], pos 5: -a[1]
    cx[2] += -a[1]; cx[5] += -a[1]
    # k=2: zeta^{-2} = zeta^7 = -zeta - zeta^4 → pos 1: -a[2], pos 4: -a[2]
    cx[1] += -a[2]; cx[4] += -a[2]
    # k=3: zeta^{-3} = zeta^6 = -1 - zeta^3 → pos 0: -a[3], pos 3: -a[3]
    cx[0] += -a[3]; cx[3] += -a[3]
    # k=4: zeta^{-4} = zeta^5 → pos 5: +a[4]
    cx[5] += a[4]
    # k=5: zeta^{-5} = zeta^4 → pos 4: +a[5]
    cx[4] += a[5]

    # Now compute c = x * conj(x) in ringZ9 basis (degree-11 product, then reduce).
    # zeta^k for k in 0..10; reduce: 6→-(0+3), 7→-(1+4), 8→-(2+5), 9→0, 10→1
    prod = np.zeros(11, dtype=np.int64)
    for i in range(6):
        for j in range(6):
            prod[i + j] += int(a[i]) * int(cx[j])

    c = prod[:6].copy()
    c[0] -= prod[6]; c[3] -= prod[6]
    c[1] -= prod[7]; c[4] -= prod[7]
    c[2] -= prod[8]; c[5] -= prod[8]
    c[0] += prod[9]      # zeta^9 = 1
    c[1] += prod[10]     # zeta^10 = zeta

    m2 = -int(c[4])
    m1 = -int(c[5])
    m0 = int(c[0]) + 2 * int(c[4])
    return (m0, m1, m2)


def _row_norm_in_alpha(a_18: np.ndarray) -> tuple[int, int, int]:
    """Compute |x_1|² + |x_2|² + |x_3|² in Z[α], with a_18 = 18-int row coef vector."""
    m0 = m1 = m2 = 0
    for j in range(3):
        a_j = a_18[6*j:6*(j+1)]
        n0, n1, n2 = _z9_norm_squared(a_j)
        m0 += n0; m1 += n1; m2 += n2
    return (m0, m1, m2)


# ---------------------------------------------------------------------------
#  Pipeline
# ---------------------------------------------------------------------------

def lattice_compile(theta: float, epsilon: float, f: int,
                    *, max_candidates: int = 32,
                    target_norm: int = 2,
                    verbose: bool = False) -> Optional[dict]:
    """Run the end-to-end lattice-compile pipeline for one cell.

    Args:
      theta, epsilon: target rotation + precision.
      f: V-denominator exponent (V is in (1/3^f) Z[ζ_9]^{3×3}).
      max_candidates: how many lattice candidates to enumerate.
      target_norm: 2 for Householder convention (|u|² = 2 → sum Y_i = 2·3^{2f}).
      verbose: print intermediate stats.

    Returns:
      dict with the best candidate found, or None if no valid candidate.
    """
    u_target = householder_target_for_R01(theta)

    # Phase A: lattice rounding to candidate triples.
    candidates = round_to_zeta9_row(u_target, eps=epsilon, f=f,
                                     max_candidates=max_candidates)
    if verbose:
        print(f"[lattice_compile] phase A: {len(candidates)} candidates within ε={epsilon}")

    if not candidates:
        return None

    # Phase B: filter by norm constraint.  We want |u|² = target_norm exactly,
    # i.e. (m0, m1, m2) = (target_norm · 3^{2f}, 0, 0).
    target_sum = (target_norm * (3 ** (2 * f)), 0, 0)
    norm_passers = []
    for a, residual in candidates:
        m012 = _row_norm_in_alpha(a)
        if m012 == target_sum:
            norm_passers.append((a, residual, m012))
        elif verbose:
            pass  # too many candidates to print individually
    if verbose:
        print(f"[lattice_compile] phase B: {len(norm_passers)} pass norm constraint "
              f"({target_norm}·3^{2*f} = {target_sum[0]})")

    if not norm_passers:
        # Report the closest miss
        norm_dists = []
        for a, residual in candidates[:5]:
            m012 = _row_norm_in_alpha(a)
            norm_dists.append((a, residual, m012))
        return {
            "success": False,
            "reason": "no candidate satisfies the exact norm constraint",
            "expected_norm": target_sum,
            "observed_norms": [(int(r), m) for _, r, m in norm_dists],
            "n_candidates_within_eps": len(candidates),
        }

    best = norm_passers[0]
    return {
        "success": True,
        "a": best[0].tolist(),
        "residual_sigma1": best[1],
        "norm_m012": best[2],
        "n_phase_a": len(candidates),
        "n_phase_b": len(norm_passers),
    }


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reuse the zeta9 test cell that works end-to-end.
    print("=== θ=π/2, ε=0.3, f=2 (zeta9 known-working cell) ===")
    result = lattice_compile(theta=math.pi / 2, epsilon=0.3, f=2,
                              max_candidates=64, verbose=True)
    if result is None:
        print("  No candidates found.")
    elif result["success"]:
        print(f"  SUCCESS")
        print(f"    a = {result['a']}")
        print(f"    residual_σ1 = {result['residual_sigma1']:.4f}")
        print(f"    norm = {result['norm_m012']}")
        print(f"    pipeline: phaseA={result['n_phase_a']} → phaseB={result['n_phase_b']}")
    else:
        print(f"  Failed: {result['reason']}")
        print(f"    expected norm = {result['expected_norm']}")
        print(f"    observed norms (closest 5):")
        for r, m in result["observed_norms"]:
            print(f"      residual={r}, norm={m}")
