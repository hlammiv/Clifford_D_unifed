"""EP-style single-component rounding for qutrit Clifford+D synthesis.

Per Agent B's hybrid proposal (2026-05-12): the Ross-Selinger grid problem in
6D doesn't lift to Z[ζ_9], but it DOES lift to 2D for a single Householder
component.  This module rounds one complex number u_target ∈ C to an integer
coefficient vector a ∈ Z^6 representing an element u_1 = (1/3^f) · Σ a_k ζ_9^k
such that:

  σ_1(u_1) is within ε of u_target  (the principal embedding tracks the target)
  |σ_r(u_1)|² ≤ U_max  for r ∈ {1, 2, 4}  (per-conjugate magnitude bound)

The per-conjugate bound U_max comes from the joint norm constraint
Σ_i |σ_r(u_i)|² = 2·3^{2f} (Householder normalization): a single component is
bounded by the full sum.  We use U_max = (2·3^{2f}) · (1 + δ) as a soft upper
bound; the exact constraint comes downstream when we factor the norm Y_1.

Approach: 6-dim lattice + 6-real linear constraints (2 real per σ_r, three r's)
+ 2-real ε-target on σ_1.  fpylll's enumerate inside a transformed box.

Downstream pipeline:
  1. Get (u_1 candidates) from this module
  2. Y_1 = |u_1|² ∈ Z[α]   (3 ints)
  3. Y_{23} = (2·3^{2f}, 0, 0) - Y_1
  4. Factor Y_{23} = Y_2 + Y_3 via zeta9's select_triples / find_roots
  5. Combine (u_1, u_2, u_3) → Householder V → HRSA decompose
"""

from __future__ import annotations
import math
import cmath
from functools import lru_cache
from typing import Optional

import numpy as np

try:
    from fpylll import IntegerMatrix, LLL, BKZ, Enumeration, EnumerationError
    _HAVE_FPYLLL = True
except ImportError:
    _HAVE_FPYLLL = False


# ---------------------------------------------------------------------------
#  Minkowski embedding of Z[ζ_9]: 6 ints → 6 reals
# ---------------------------------------------------------------------------
# Galois orbits up to complex conjugation: r ∈ {1, 2, 4}.
# Each r gives 2 real coords (Re σ_r, Im σ_r).
# Layout: [Re σ_1, Im σ_1, Re σ_2, Im σ_2, Re σ_4, Im σ_4]

_GALOIS_R = (1, 2, 4)


@lru_cache(maxsize=1)
def minkowski_basis() -> np.ndarray:
    """Return 6x6 real matrix M such that for x = Σ a_k ζ_9^k:
       M @ a = [Re σ_1(x), Im σ_1(x), Re σ_2(x), Im σ_2(x), Re σ_4(x), Im σ_4(x)]
    """
    M = np.zeros((6, 6), dtype=np.float64)
    for r_idx, r in enumerate(_GALOIS_R):
        for k in range(6):
            angle = 2.0 * math.pi * r * k / 9.0
            M[2 * r_idx,     k] = math.cos(angle)
            M[2 * r_idx + 1, k] = math.sin(angle)
    return M


# ---------------------------------------------------------------------------
#  Single-entry rounder
# ---------------------------------------------------------------------------

def round_single_entry(u_target: complex, eps: float, f: int,
                       *, max_candidates: int = 64,
                       conj_bound_factor: float = 1.5) -> list[tuple[np.ndarray, float, float, float]]:
    """Find integer a ∈ Z^6 such that u = (1/3^f) Σ a_k ζ_9^k satisfies
    |σ_1(u) - u_target| < eps  AND  |σ_r(u)|² ≤ conj_bound_factor · 2  for all r.

    Args:
      u_target: complex target for σ_1(u)
      eps: σ_1 distance budget
      f: V-denominator exponent (V-denom = 3^f)
      max_candidates: cap on enumeration output
      conj_bound_factor: relax the joint-norm bound by this factor
                        (default 1.5 → per-conjugate σ_r(u)² ≤ 3)

    Returns:
      List of (a, residual_sigma1, max_sigma_sq, |a|_norm_sq).
      a is 6 int64 cyclotomic coefs; residual_sigma1 is |σ_1(u) - u_target|.
    """
    M = minkowski_basis()  # 6x6 real

    # Scaled target: we work in units of (1/3^f), so the integer-coefficient
    # vector a satisfies M @ a = u_phys * 3^f.
    f_pow = 3**f
    sigma1_target_scaled = np.array([u_target.real, u_target.imag]) * f_pow

    # Continuous "ideal" solution for σ_1 target:
    # M @ a = [t_r, t_i, ?, ?, ?, ?]; only first two rows constrained by σ_1.
    # Pinv of first 2 rows gives min-norm preimage.
    M_sigma1 = M[:2, :]              # (2, 6)
    M_pinv_sigma1 = np.linalg.pinv(M_sigma1)   # (6, 2)
    a_min_norm = M_pinv_sigma1 @ sigma1_target_scaled   # length-6 real

    # Conjugate bounds: |σ_r(u)|² ≤ conj_bound_factor · 2  in physical units
    # ⇒ |M @ a|² in scaled units bounded by (conj_bound · 2) · f_pow²
    conj_bound_sq = conj_bound_factor * 2.0 * (f_pow ** 2)

    candidates = []

    # Brute-force search over perturbations of round(a_min_norm).
    # We round the min-norm solution to integers, then perturb each coord by
    # ±δ for δ ∈ {0, ±1, ±2}.  This is 5^6 = 15625 candidates per call.
    # Filtering should keep it manageable.
    a_center = np.rint(a_min_norm).astype(np.int64)

    # Generate offsets in increasing L∞ distance order so we hit good
    # candidates first and can early-exit.
    import itertools
    radii = [0, 1, 2]
    # Order by descending L∞ but expanding shell-by-shell isn't easy;
    # just enumerate the full cube and sort by L∞.
    offsets = list(itertools.product([-2, -1, 0, 1, 2], repeat=6))
    offsets.sort(key=lambda o: max(abs(x) for x in o))

    for off in offsets:
        if len(candidates) >= max_candidates * 4:
            break
        a = a_center + np.array(off, dtype=np.int64)
        ma = M @ a   # 6-real Minkowski image (scaled by f_pow)
        # σ_1 residual
        d_re = ma[0] - sigma1_target_scaled[0]
        d_im = ma[1] - sigma1_target_scaled[1]
        residual_sq_scaled = d_re*d_re + d_im*d_im
        eps_budget_sq = (eps * f_pow) ** 2
        if residual_sq_scaled > eps_budget_sq * (1 + 1e-8):
            continue
        # Per-conjugate magnitude check
        s1_sq = ma[0]*ma[0] + ma[1]*ma[1]
        s2_sq = ma[2]*ma[2] + ma[3]*ma[3]
        s4_sq = ma[4]*ma[4] + ma[5]*ma[5]
        max_s_sq = max(s1_sq, s2_sq, s4_sq)
        if max_s_sq > conj_bound_sq:
            continue
        # Passes both checks
        residual_phys = math.sqrt(residual_sq_scaled) / f_pow
        max_sigma_phys = math.sqrt(max_s_sq) / f_pow
        a_norm_sq = float(s1_sq + s2_sq + s4_sq) / (f_pow ** 2)  # = 3·|u|² avg
        candidates.append((a, residual_phys, max_sigma_phys, a_norm_sq))

    candidates.sort(key=lambda x: x[1])
    # Dedupe
    seen = set()
    unique = []
    for c in candidates:
        k = tuple(int(x) for x in c[0])
        if k not in seen:
            seen.add(k)
            unique.append(c)
        if len(unique) >= max_candidates:
            break
    return unique


# ---------------------------------------------------------------------------
#  Y-norm computation in Z[α]
# ---------------------------------------------------------------------------

def y_norm(a: np.ndarray) -> tuple[int, int, int]:
    """Compute |x|² in Z[α] for x = Σ a_k ζ_9^k.
    Returns (m0, m1, m2) such that |x|² = m0 + m1·α + m2·α².

    Per zeta9's z9_norm_m012:
      With c = x · conj(x) in ringZ9 basis,
        m2 = -c[4], m1 = -c[5], m0 = c[0] + 2·c[4].
    """
    a = np.asarray(a, dtype=np.int64)
    # Compute conj(x) in ringZ9 basis
    cx = np.zeros(6, dtype=np.int64)
    cx[0] += int(a[0])
    cx[2] += -int(a[1]); cx[5] += -int(a[1])
    cx[1] += -int(a[2]); cx[4] += -int(a[2])
    cx[0] += -int(a[3]); cx[3] += -int(a[3])
    cx[5] += int(a[4])
    cx[4] += int(a[5])

    # c = x * cx, expand to deg-11 then reduce
    prod = np.zeros(11, dtype=np.int64)
    for i in range(6):
        for j in range(6):
            prod[i + j] += int(a[i]) * int(cx[j])
    c = prod[:6].copy()
    c[0] -= prod[6]; c[3] -= prod[6]
    c[1] -= prod[7]; c[4] -= prod[7]
    c[2] -= prod[8]; c[5] -= prod[8]
    c[0] += prod[9]
    c[1] += prod[10]

    m2 = -int(c[4])
    m1 = -int(c[5])
    m0 = int(c[0]) + 2 * int(c[4])
    return (m0, m1, m2)


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------

def _self_test():
    """Sanity checks."""
    M = minkowski_basis()
    assert M.shape == (6, 6)
    # σ_r(1) = 1 for all r → M @ [1,0,0,0,0,0] = [1,0,1,0,1,0]
    assert np.allclose(M @ np.array([1, 0, 0, 0, 0, 0]),
                       [1, 0, 1, 0, 1, 0]), "σ_r(1) check failed"
    # σ_1(ζ_9) = ζ_9, |·|² = 1.  y_norm should give (1, 0, 0).
    y1 = y_norm(np.array([0, 1, 0, 0, 0, 0]))
    assert y1 == (1, 0, 0), f"y_norm of ζ_9 should be (1,0,0), got {y1}"
    # y_norm of (1 + ζ_9): |·|² = (1+ζ)(1+ζ⁻¹) = 2 + ζ + ζ⁻¹ = 2 + α
    y2 = y_norm(np.array([1, 1, 0, 0, 0, 0]))
    assert y2 == (2, 1, 0), f"y_norm of 1+ζ_9 should be (2,1,0), got {y2}"
    print("self_test: PASS")


def _round_demo():
    """Round the x_1 component of a Householder vector for R^Z(θ=π/2).
    HRSA convention: x_1 ~ e^{i θ/2}."""
    theta = math.pi / 2
    u_target = cmath.exp(0.5j * theta)
    print(f"\nRound x_1 ≈ {u_target} at ε=0.05, f=2:")
    cands = round_single_entry(u_target, eps=0.05, f=2, max_candidates=16)
    print(f"  {len(cands)} candidates")
    for i, (a, r, ms, ns) in enumerate(cands[:5]):
        y = y_norm(a)
        print(f"  [{i}] a={a.tolist()}, res_σ1={r:.4f}, max_σ_r={ms:.3f}, "
              f"|u|²_tot/3={ns:.3f}, Y=(m0,m1,m2)={y}")


if __name__ == "__main__":
    _self_test()
    _round_demo()
