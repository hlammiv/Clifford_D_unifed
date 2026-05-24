"""babai.py — Babai-CVP for x_1 candidates on the qutrit Z[ζ_9] lattice.

Phase 2 of the qutrit Babai-CVP plan. Given a target rotation angle ``theta``
and denominator exponent ``f``, returns a list of integer coefficient vectors
``a ∈ Z^6`` (each a length-6 tuple of Python ints) such that the principal
complex embedding ``σ_1(a) = Σ a_k · e^{2πik/9}`` is close to the target
``z = 3^f · e^{iθ/2}`` and the quadratic form ``q(a) ≤ 2·3^{2f}`` (so there is
budget left for ``x_2, x_3`` in the Householder norm equation
``q(x_1)+q(x_2)+q(x_3) = 2·3^{2f}``).

Math
----

``Z[ζ_9]`` has rank 6 over ``Z`` with three pairs of complex embeddings
``σ_1, σ_2, σ_4`` (acting as ``ζ ↦ e^{2πik/9}`` for ``k ∈ {1, 2, 4}``).
The quadratic form ``q`` matches the (normalised) Minkowski norm bit-for-bit::

    q(a) = (1/3) · (|σ_1(a)|² + |σ_2(a)|² + |σ_4(a)|²).

Embedding ``a → (Re σ_1, Im σ_1, Re σ_2, Im σ_2, Re σ_4, Im σ_4)`` (with
unit weights) gives a rank-6 real lattice ``L_M ⊂ R^6`` whose Euclidean
norm² equals ``3·q``.

The algorithm has two stages:

1. **Babai-CVP seed.** Run nearest-plane on two LLL bases:
   (a) the *heavy-weighted* basis (large ``W`` on the principal pair) so
       ``σ_1(a_0) ≈ z`` is enforced tightly,
   (b) the *unweighted* basis so ``q(a_0)`` stays small even when the
       heavy seed overshoots the budget (which happens at small ``f``).
   Both seeds are kept.

2. **Short-vector perturbation.** Enumerate the shortest ``O(n_candidates)``
   lattice vectors in the unweighted lattice (these have small ``q``),
   and add each ``± v`` to every seed. Candidates that pass the
   ``q(a) ≤ q_budget`` and ``|σ_1(a)/3^f − e^{iθ/2}| ≤ ε`` filters are
   sorted by ``|σ_1(a) − z|`` and returned.

The candidate ordering matches HRSA ``entryEnumeration``'s post-sort
criterion (principal-place accuracy).

Public API
----------

``babai_x1(theta, f, n_candidates=100, eps=None) -> list[tuple[int, ...]]``

See its docstring for argument semantics. All return values are 6-tuples of
plain Python ``int`` so arbitrarily large coefficients survive.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from fpylll import (
        LLL,
        Enumeration,
        EnumerationError,
        IntegerMatrix,
    )
    from fpylll.fplll.gso import MatGSO
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "fpylll is required for cvp.babai. Install with `pip install fpylll`."
    ) from exc

from cvp.gram import q_form

__all__ = ["babai_x1", "minkowski_embedding"]


# ---------------------------------------------------------------------------
# Minkowski embedding
# ---------------------------------------------------------------------------

# Principal complex places of Q(ζ_9) acting as ζ → e^{2πi k/9} for k ∈ {1, 2, 4}.
# The remaining three places are the complex conjugates k ∈ {8, 7, 5}; they
# carry no new geometric information for our q-form, which sums the *moduli*
# of the three complex pairs.
_PLACES = (1, 2, 4)


def minkowski_embedding() -> np.ndarray:
    """Return the 6x6 real Minkowski-embedding matrix.

    Row layout::

        [ Re σ_1(ζ^j),  Im σ_1(ζ^j),
          Re σ_2(ζ^j),  Im σ_2(ζ^j),
          Re σ_4(ζ^j),  Im σ_4(ζ^j) ]    (j = 0..5)

    so that for a coefficient vector ``a ∈ R^6``, ``M @ a`` is the stacked
    real/imag of ``(σ_1(a), σ_2(a), σ_4(a))``. Pre-scale by ``1/√3`` and the
    Euclidean norm-squared of the result equals ``q(a)``.
    """
    M = np.zeros((6, 6), dtype=np.float64)
    for row_pair, k in enumerate(_PLACES):
        for j in range(6):
            angle = 2.0 * np.pi * k * j / 9.0
            M[2 * row_pair, j] = math.cos(angle)
            M[2 * row_pair + 1, j] = math.sin(angle)
    return M


_M = minkowski_embedding()


# ---------------------------------------------------------------------------
# Integer scaling for fpylll
# ---------------------------------------------------------------------------

# Base scaling: turns irrational Minkowski entries into reasonable integers.
_FPLLL_SCALE = 10 ** 6

# Heavy weight on the principal place used for the **Babai nearest-plane**
# stage: this pulls ``σ_1(a) → z`` very tightly. Used only to seed the
# search with one ``a_0`` whose principal value is near the target; the
# perturbation stage uses the unweighted lattice (short ζ^j-style vectors).
_PRINCIPAL_WEIGHT = 10 ** 9


def _scaled_basis_and_target(
    theta: float, f: int
) -> Tuple[IntegerMatrix, list, int]:
    """Build the integer-scaled weighted Minkowski basis and target for fpylll.

    Returns ``(B_int, t_int, principal_weight)``.

    ``B_int`` rows are the (heavy-)weighted Minkowski images of the canonical
    ζ-basis. The first 2 ambient coords carry the principal place
    ``σ_1(ζ^j)`` multiplied by ``_PRINCIPAL_WEIGHT``; the last 4 carry the
    auxiliary places ``σ_2, σ_4`` with weight 1. Everything is then scaled
    by ``_FPLLL_SCALE`` and rounded to int.

    ``t_int`` is the target: ``z = 3^f e^{iθ/2}`` placed on the principal
    coords (with the same heavy weight), zeros elsewhere.
    """
    W = _PRINCIPAL_WEIGHT
    weights = np.array([W, W, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    # _M has rows = output coordinates (Re σ_1, Im σ_1, Re σ_2, ...) and
    # columns = input basis vectors ζ^j. The image of ζ^j under the weighted
    # Minkowski embedding is therefore (weights * _M)[:, j] · _FPLLL_SCALE.
    # fpylll wants one basis vector per row of B_int, so we transpose after
    # weighting.
    basis_real = ((_M.T * weights) * _FPLLL_SCALE)  # rows = image of ζ^j
    basis_int = np.round(basis_real).astype(object)  # arbitrary precision
    rows = [[int(x) for x in row] for row in basis_int]

    z_re = (3.0 ** f) * math.cos(theta / 2.0)
    z_im = (3.0 ** f) * math.sin(theta / 2.0)
    t_int = [
        int(round(z_re * W * _FPLLL_SCALE)),
        int(round(z_im * W * _FPLLL_SCALE)),
        0,
        0,
        0,
        0,
    ]

    B_int = IntegerMatrix.from_matrix(rows)
    return B_int, t_int, W


def _unweighted_basis(f: int) -> Tuple[IntegerMatrix, List[int]]:
    """Build the integer-scaled *uniformly weighted* Minkowski basis.

    Rows are the integer-rounded Minkowski images of ``ζ^j`` with unit
    weights on each of the 6 ambient coordinates. Pre-scaled by
    ``_FPLLL_SCALE``. With all-equal weights, the Euclidean norm² of a
    lattice vector equals ``3 · q(v) · _FPLLL_SCALE²`` (from
    ``q(v) = (|σ_1|² + |σ_2|² + |σ_4|²)/3``).

    ``f`` is unused by the basis itself but kept for signature symmetry
    with ``_scaled_basis_and_target``. The empty target list is returned
    as a convenience for callers that build a per-f target separately.
    """
    basis_real = _M.T * _FPLLL_SCALE  # rows = image of ζ^j under unit-weight
    basis_int = np.round(basis_real).astype(object)
    rows = [[int(x) for x in row] for row in basis_int]
    return IntegerMatrix.from_matrix(rows), []


# ---------------------------------------------------------------------------
# Babai enumeration
# ---------------------------------------------------------------------------

def _babai_nearest_plane(
    B_lll: IntegerMatrix, target: Sequence[float]
) -> List[int]:
    """Babai's nearest-plane algorithm on an LLL-reduced basis.

    Returns the integer coefficient vector ``c`` such that the lattice point
    ``Σ c_i b_i`` (with ``b_i`` the rows of ``B_lll``) is close to
    ``target`` in the Euclidean metric. The coefficients are bounded by the
    GSO projections of ``target``; for a properly LLL-reduced basis they stay
    O(1) for reasonable targets (no float blow-up).

    This is the "round each GSO projection" version of Babai's algorithm.
    Approximation factor: ``(2/√3)^n`` ≈ ``1.15^n`` (looser than the
    plain-Babai ``2^{n/2}`` but easier to implement and good enough for
    n = 6).
    """
    n = B_lll.nrows
    d = B_lll.ncols
    B_rows = [
        [float(int(B_lll[i, j])) for j in range(d)] for i in range(n)
    ]
    # Pure-Python Gram-Schmidt orthogonalisation (n=6, tiny — float64 ok).
    B_star: List[List[float]] = []
    mu = [[0.0] * n for _ in range(n)]
    for i in range(n):
        bi_star = list(B_rows[i])
        for j in range(i):
            num = sum(B_rows[i][k] * B_star[j][k] for k in range(d))
            denom = sum(B_star[j][k] * B_star[j][k] for k in range(d))
            mu[i][j] = num / denom if denom else 0.0
            for k in range(d):
                bi_star[k] -= mu[i][j] * B_star[j][k]
        B_star.append(bi_star)

    # Babai nearest-plane: process basis in reverse, round projection onto
    # each GSO direction, subtract.
    residual = [float(t) for t in target]
    coeffs = [0] * n
    for i in range(n - 1, -1, -1):
        denom = sum(B_star[i][k] * B_star[i][k] for k in range(d))
        if denom == 0.0:
            ci = 0
        else:
            num = sum(residual[k] * B_star[i][k] for k in range(d))
            ci = int(round(num / denom))
        coeffs[i] = ci
        if ci != 0:
            for k in range(d):
                residual[k] -= ci * B_rows[i][k]
    return coeffs


def _short_vectors(
    B_lll: IntegerMatrix, max_norm_sq: float, n_vecs: int
) -> List[List[int]]:
    """Enumerate short lattice vectors as combinations of the LLL basis rows.

    Returns up to ``n_vecs`` integer coefficient vectors ``c`` such that
    ``c @ B_lll`` has squared Euclidean norm ≤ ``max_norm_sq``. The first
    returned vector is always the zero vector (the origin is the shortest
    "vector").

    Used by ``babai_x1`` to perturb the Babai-closest lattice point.
    """
    n = B_lll.nrows
    out: List[List[int]] = [[0] * n]

    M = MatGSO(B_lll)
    M.update_gso()

    enum = Enumeration(M, nr_solutions=max(1, int(n_vecs)))
    try:
        results = enum.enumerate(0, n, max_norm_sq, 0)
    except EnumerationError:
        results = []

    for _norm_sq, coeffs in results:
        out.append([int(c) for c in coeffs])
        if len(out) >= n_vecs:
            break
    return out


def _cvp_enumerate(
    B_lll: IntegerMatrix,
    target_amb: Sequence[float],
    max_dist_sq: float,
    n_vecs: int,
) -> List[Tuple[List[int], float]]:
    """Enumerate lattice points near a target via fpylll BDD/CVP enumeration.

    Returns up to ``n_vecs`` ``(c_vec, dist_sq)`` pairs where ``c_vec @ B_lll``
    is a lattice point at squared Euclidean distance ``dist_sq`` from
    ``target_amb`` (and ``dist_sq ≤ max_dist_sq``). Sorted by ascending
    distance.

    fpylll's ``Enumeration.enumerate(..., target=...)`` expects the target
    in *basis coordinates* (i.e. the unique solution ``c_target`` of
    ``c_target @ B_lll = target_amb``). The returned coordinates are also
    in basis coordinates; the corresponding lattice point is
    ``c_vec @ B_lll`` (a lattice point at distance²
    ``||c_vec @ B_lll - target_amb||²``).
    """
    n = B_lll.nrows
    d = B_lll.ncols
    B_arr = np.array(
        [[int(B_lll[i, j]) for j in range(d)] for i in range(n)],
        dtype=np.float64,
    )
    target_amb_arr = np.array([float(t) for t in target_amb], dtype=np.float64)

    # Solve c_target @ B_lll = target_amb  →  c_target = target_amb @ B_lll^{-1}
    # (only valid when B_lll is square invertible; in our use case yes).
    try:
        c_target = np.linalg.solve(B_arr.T, target_amb_arr)
    except np.linalg.LinAlgError:
        return []

    M = MatGSO(B_lll)
    M.update_gso()
    enum = Enumeration(M, nr_solutions=max(1, int(n_vecs)))
    try:
        results = enum.enumerate(
            0, n, max_dist_sq, 0, target=[float(x) for x in c_target]
        )
    except EnumerationError:
        results = []

    out: List[Tuple[List[int], float]] = []
    for dist_sq, coeffs in results:
        out.append(([int(c) for c in coeffs], float(dist_sq)))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def babai_x1(
    theta: float,
    f: int,
    n_candidates: int = 100,
    eps: Optional[float] = None,
) -> List[Tuple[int, ...]]:
    """Return Babai-CVP candidates for ``x_1 ∈ Z[ζ_9]`` near ``3^f e^{iθ/2}``.

    Parameters
    ----------
    theta : float
        Target rotation angle. The principal target for ``x_1`` is
        ``z = 3^f · e^{iθ/2}`` (matching HRSA ``entryEnumeration``'s sign
        convention; see the theta-convention note in the repo memory).
    f : int
        Denominator exponent. The element ``a`` represents ``x_1 / 3^f``.
    n_candidates : int, default 100
        Upper bound on the number of candidates returned.
    eps : float, optional
        Approximation tolerance: returned candidates satisfy
        ``|σ_1(a)/3^f - e^{iθ/2}| ≤ eps``. Defaults to ``8 · 3^{-f}`` (the
        Babai approximation bound for the 6-D LLL-reduced lattice).

    Returns
    -------
    list of tuple[int, ...] of length 6
        Each tuple is the ζ-basis coefficient vector of one candidate ``a``.
        Ordered by ``|σ_1(a) - z|`` ascending (principal-accuracy first).
        Every returned candidate satisfies ``q(a) ≤ 2·3^{2f}`` (the
        Householder budget) and ``|σ_1(a)/3^f - e^{iθ/2}| ≤ eps``.

    Notes
    -----
    The ordering choice (``|σ_1(a) - z|``) mirrors HRSA's sort and reflects
    the only intrinsic 1-D quality measure of an x_1 candidate. Phase 3
    consumes this list and re-ranks by (x_2, x_3) feasibility.

    For typical inputs (``f = 6``, ``n_candidates = 100``) the call runs in
    ~milliseconds, dominated by the fpylll enumeration.

    Algorithm (Selinger 2012 §6.2 pattern, generalised to Z[ζ_9])
    --------------------------------------------------------------
    1. Build two LLL-reduced bases of the rank-6 Minkowski lattice: one
       with a heavy weight on the principal-place coordinates, one with
       unit weights everywhere.
    2. Babai nearest-plane on each → two ``a_0`` seeds. The heavy seed
       gives tight ``σ_1 ≈ z`` (small principal error); the unweighted
       seed gives small ``q`` even when the heavy seed overshoots the
       budget. Both are kept.
    3. Enumerate the shortest ``O(n_candidates)`` lattice vectors in the
       unweighted lattice via fpylll's SVP enumeration. Each has small
       ``q``, so perturbing ``a_0 ± v`` produces a candidate whose
       deviation from the seed is q-bounded.
    4. Apply the ``q(a) ≤ 2·3^{2f}`` and ``|σ_1(a)/3^f - target| ≤ eps``
       filters, sort by ``|σ_1(a) - z|``, return the top n_candidates.
    """
    if f < 0:
        raise ValueError(f"f must be non-negative; got {f}")
    if n_candidates < 1:
        return []

    if eps is None:
        eps = 8.0 * (3.0 ** (-f))

    q_budget = 2 * 3 ** (2 * f)
    z = (3.0 ** f) * complex(math.cos(theta / 2.0), math.sin(theta / 2.0))
    n = 6

    # ----- Build LLL-reduced lattices: heavy (for tight a_0) + unweighted
    # (for short-vector perturbations) -----------------------------------
    B_unw, _t_unw = _unweighted_basis(f)
    U_unw_mat = IntegerMatrix.identity(n)
    LLL.reduction(B_unw, U=U_unw_mat)
    U_unw_z = [
        [int(U_unw_mat[i, j]) for j in range(n)] for i in range(n)
    ]

    B_h, t_h, _W = _scaled_basis_and_target(theta, f)
    U_h_mat = IntegerMatrix.identity(n)
    LLL.reduction(B_h, U=U_h_mat)
    U_h_z = [
        [int(U_h_mat[i, j]) for j in range(n)] for i in range(n)
    ]

    # ----- Stage 1: build seed pool with two Babai variants ------------
    # Heavy-weighted Babai: picks a_0 with tight σ_1 ≈ z (but possibly
    # large q). At large f this is the right seed because the lattice is
    # dense enough that *some* q-budget-valid lattice points sit near a_0.
    # Unweighted Babai: picks a_0 with small q (the "balanced" CVP). At
    # small f where the q-budget shell has few elements, this seed avoids
    # the budget overshoot of the heavy variant.
    coords_h = _babai_nearest_plane(B_h, [float(t) for t in t_h])
    a0_heavy: Tuple[int, ...] = tuple(
        sum(coords_h[i] * U_h_z[i][j] for i in range(n)) for j in range(n)
    )

    z_re_int = int(round(z.real * _FPLLL_SCALE))
    z_im_int = int(round(z.imag * _FPLLL_SCALE))
    t_unw_int = [z_re_int, z_im_int, 0, 0, 0, 0]
    coords_unw = _babai_nearest_plane(B_unw, [float(t) for t in t_unw_int])
    a0_unweighted: Tuple[int, ...] = tuple(
        sum(coords_unw[i] * U_unw_z[i][j] for i in range(n)) for j in range(n)
    )

    seeds: List[Tuple[int, ...]] = [a0_unweighted]
    if a0_heavy != a0_unweighted:
        seeds.append(a0_heavy)

    # ----- Stage 2: enumerate short lattice vectors --------------------
    # Short = small q. In integer-scaled ambient, ||v||² = 3·q(v)·scale².
    # Cap K at q_budget (any v larger than this cannot be added to a seed
    # that already has q ≤ q_budget without exceeding the budget) plus a
    # small absolute floor for tight-budget cases.
    K = max(q_budget, 64)
    short_radius_sq = 3.0 * float(K) * (_FPLLL_SCALE ** 2)
    # n_short controls how many lattice points we visit. At small f the
    # lattice has few q-budget-valid points so a generous floor (8192)
    # nearly exhausts them. At large f the budget shell is huge but each
    # perturbation has high acceptance rate so n_short ≈ 16·n_candidates
    # suffices. The hard cap keeps wall-time bounded.
    n_short = min(max(16 * n_candidates, 8192), 50_000)
    short_coords = _short_vectors(B_unw, short_radius_sq, n_short)

    # ----- Build candidate set + filter ---------------------------------
    omega = [
        complex(math.cos(2 * math.pi * j / 9), math.sin(2 * math.pi * j / 9))
        for j in range(6)
    ]
    target_dist_principal = eps * (3.0 ** f)
    scored: List[Tuple[float, Tuple[int, ...]]] = []
    seen: set[Tuple[int, ...]] = set()

    for c in short_coords:
        delta = tuple(
            sum(c[i] * U_unw_z[i][j] for i in range(n)) for j in range(n)
        )
        for seed in seeds:
            for sign in (1, -1):
                cand = tuple(seed[j] + sign * delta[j] for j in range(n))
                if cand in seen:
                    continue
                seen.add(cand)
                q_a = q_form(cand)
                if q_a > q_budget:
                    continue
                s1 = sum(cand[j] * omega[j] for j in range(6))
                err = abs(s1 - z)
                if err > target_dist_principal:
                    continue
                scored.append((err, cand))

    scored.sort(key=lambda x: x[0])
    return [a for _, a in scored[:n_candidates]]


# ---------------------------------------------------------------------------
# Module entry point — quick smoke test
# ---------------------------------------------------------------------------

def _main():  # pragma: no cover - manual sanity
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Smoke-test cvp.babai.babai_x1 at a single (θ, f)."
    )
    parser.add_argument("--theta", type=float, default=math.pi / 3)
    parser.add_argument("--f", type=int, default=6)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    cands = babai_x1(args.theta, args.f, n_candidates=args.n)
    dt = time.time() - t0

    z = (3.0 ** args.f) * complex(math.cos(args.theta / 2.0), math.sin(args.theta / 2.0))
    print(f"θ={args.theta}, f={args.f}, |cands|={len(cands)}, wall={dt*1e3:.1f} ms")
    print(f"target z = {z}")
    for k, a in enumerate(cands[:10]):
        s1 = sum(
            a[j] * complex(math.cos(2 * math.pi * j / 9), math.sin(2 * math.pi * j / 9))
            for j in range(6)
        )
        print(f"  [{k}] a={a}, q={q_form(a)}, |σ_1(a)-z|={abs(s1-z):.4g}")


if __name__ == "__main__":  # pragma: no cover
    _main()
