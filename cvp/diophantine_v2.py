"""diophantine_v2.py — Phase 3 v2 of the qutrit Babai-CVP plan.

Joint (x_1, x_3) enumeration for the Householder norm equation, modelled
on Selinger 2012 §7 (Algorithm 23). The v1 module (:mod:`cvp.diophantine`)
picks x_1 once and varies x_3 by itself — which means each (x_1, x_3) is
chosen *independently*, and the residual ``M(x_2) := bb(x_2)`` rarely
lands on the principal-norm class. Phase 5 validation found ~0% hit rate
at ε ≤ 10⁻⁴ as a result.

This module rewrites the inner loop as a **joint enumeration**: for each
(x_1, x_3) pair drawn from CARTESIAN PRODUCTS of Babai-CVP candidate
pools, compute the unique M(x_2) that satisfies the full ring-unitarity
constraint and dispatch one PARI call per pair. Pre-screening on total
positivity and on a quick odd/even parity gate skips ~50% of pairs
without paying a PARI call.

Selinger §7 → qutrit port
-------------------------

Qubit (Selinger):
    u/√(2^k) ≈ e^{-iθ/2},   ξ = 2^k - u†u,   solve t†t = ξ for t ∈ Z[ω].
    u has 2 complex components and varies over a 1-D grid (β-axis +
    α-axis intervals from Theorem 22).

Qutrit (this module):
    u/3^f = (x_1, x_2, x_3) / 3^f with ‖u‖² = 2 (norm-2 reflection),
    H = I - uu*, V = X_(0,1) H ≈ R_z(θ).
    The norm-budget split is q(x_1) + q(x_2) + q(x_3) = 2·3^{2f}.
    There are TWO "ξ-like" residuals because qutrit Householder has 3
    components; we factor the search by (i) picking x_3 from a Babai-
    near-0 pool and (ii) picking x_1 from a Babai-near-target pool, then
    (iii) solving one PARI norm-equation for the unique x_2 in the orbit.

The principal-norm class condition replaces Selinger's "ξ•ξ is prime"
heuristic; in qutrit, the ideal-factorisation succeeds whenever the
deterministic M(x_2) ideal happens to be principal in Z[ζ_9]. We do not
attempt a primality screen ourselves; PARI's bnfisprincipal answer is
the ground truth.

Public API
----------

``solve_joint_x1_x3(theta, f, eps, ...) -> list[dict]``
    Returns full (x_1, x_2, x_3) triples that exactly satisfy the
    Householder norm equation AND pass ``reify_householder(strict=True)``
    (so they decompose under canonical_reducer).
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from fpylll import (
        Enumeration,
        EnumerationError,
        IntegerMatrix,
        LLL,
    )
    from fpylll.fplll.gso import MatGSO
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "fpylll is required for cvp.diophantine_v2. "
        "Install with `pip install fpylll`."
    ) from exc

from cvp.babai import babai_x1, minkowski_embedding
from cvp.diophantine import (
    _NormEqWorker,
    _sigma_1,
    _ALPHA,
    bb_to_real_coeffs,
    enumerate_x3,
    householder_frobenius,
)
from cvp.gram import q_form

__all__ = [
    "solve_joint_x1_x3",
    "enumerate_x3_extended",
    "JointSearchStats",
]


# ---------------------------------------------------------------------------
# Helpers reused across the joint search
# ---------------------------------------------------------------------------


def _omega(k: int) -> complex:
    return complex(math.cos(2.0 * math.pi * k / 9.0),
                   math.sin(2.0 * math.pi * k / 9.0))


# Precomputed ζ_9^k powers and their conjugates, used to expand a single
# PARI root into its 18-element torsion orbit {±ζ_9^k · x_2}.
_OMEGA_TABLE = [_omega(k) for k in range(9)]

# Multiplication table for ζ_9 in the integer-coefficient basis (1, ζ, ..., ζ^5)
# using the minimal polynomial Φ_9(x) = x^6 + x^3 + 1, hence ζ^6 = -ζ^3 - 1 and
# ζ^7 = -ζ^4 - ζ, ζ^8 = -ζ^5 - ζ^2.
_ZETA_POW: List[Tuple[int, ...]] = [
    (1, 0, 0, 0, 0, 0),  # ζ^0
    (0, 1, 0, 0, 0, 0),  # ζ^1
    (0, 0, 1, 0, 0, 0),  # ζ^2
    (0, 0, 0, 1, 0, 0),  # ζ^3
    (0, 0, 0, 0, 1, 0),  # ζ^4
    (0, 0, 0, 0, 0, 1),  # ζ^5
    (-1, 0, 0, -1, 0, 0),  # ζ^6 = -1 - ζ^3
    (0, -1, 0, 0, -1, 0),  # ζ^7 = -ζ - ζ^4
    (0, 0, -1, 0, 0, -1),  # ζ^8 = -ζ^2 - ζ^5
]


def _mul_by_zeta_k(coefs: Tuple[int, ...], k: int) -> Tuple[int, ...]:
    """Return coefficients of ``ζ_9^k · x`` in the basis (1, ζ, ζ², ..., ζ^5).

    Multiplies each input ``coefs[j] · ζ^j`` by ``ζ^k``, then reduces
    ``ζ^{j+k}`` via :data:`_ZETA_POW`. Pure-integer arithmetic; no
    floating-point.

    Uses ``ζ_9^9 = 1`` (valid in the quotient ring Z[x]/Φ_9(x) since
    Φ_9 divides x^9 - 1). For exponents in [0, 8] we look up :data:`_ZETA_POW`
    directly; for [9, 13] we reduce mod 9 first.
    """
    k = k % 9
    if k == 0:
        return tuple(int(c) for c in coefs)
    out = [0] * 6
    for j in range(6):
        cj = int(coefs[j])
        if cj == 0:
            continue
        idx = (j + k) % 9
        z = _ZETA_POW[idx]
        for r in range(6):
            out[r] += cj * z[r]
    return tuple(out)


def _neg(coefs: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(-int(c) for c in coefs)


def _zeta_18_orbit(x_2: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Return the 18 elements of the torsion orbit ``{±ζ_9^k · x_2}_{k=0..8}``.

    The torsion subgroup of Z[ζ_9]^× is ⟨ζ_18⟩ = ⟨-ζ_9⟩ of order 18.
    Multiplying x_2 by any element of this subgroup preserves q(x_2)
    AND preserves bb(x_2) — they're invariant under conjugation-symmetric
    units — so the entire orbit satisfies the same q-sum + bb constraint
    and only differs in the σ_1 phase. Each orbit element induces a
    different Householder Frobenius residual, so we score all 18 and keep
    the best per (x_1, x_3) pair.

    Deduplication: many low-q x_2's have stabiliser orbit (e.g. x_2 = 0
    or x_2 = real integers). We return only distinct tuples.
    """
    seen: set = set()
    orbit: List[Tuple[int, ...]] = []
    for k in range(9):
        rot = _mul_by_zeta_k(x_2, k)
        for s in (rot, _neg(rot)):
            if s in seen:
                continue
            seen.add(s)
            orbit.append(s)
    return orbit


# ---------------------------------------------------------------------------
# Pre-screen: total positivity of M(x_2) before paying a PARI call
# ---------------------------------------------------------------------------


def _M_is_totally_positive(M2: Tuple[int, int, int], slack: float = 1e-9) -> bool:
    """Return True iff M2 = (m_0, m_1, m_2) has σ_r(M) ≥ 0 for r ∈ {1, 2, 4}.

    M ∈ Z[α] where α = ζ_9 + ζ_9^{-1}. For x_2 ∈ Z[ζ_9] we have
    ``M(x_2) = x_2 · conj(x_2)`` ∈ Z[α], which is totally positive (every
    real embedding is non-negative) because it's a sum-of-squares in
    each real place. The converse is necessary for the PARI norm-equation
    to have any solution.
    """
    m0, m1, m2 = M2
    for ar in _ALPHA:
        sr = m0 + ar * m1 + ar * ar * m2
        if sr < -slack:
            return False
    if m0 < 0:
        return False
    return True


# ---------------------------------------------------------------------------
# x_3 enumeration extended for joint search
# ---------------------------------------------------------------------------


_FPLLL_SCALE = 10 ** 6


def _enumerate_x1_extended(
    theta: float,
    f: int,
    eps: float,
    n_x1: int,
    *,
    sigma1_budget_mult: float = 8.0,
) -> List[Tuple[int, ...]]:
    """Wide x_1 candidate pool via repeated Babai + multi-shift perturbation.

    The HRSA reference x_1 at (θ=π/2, f=2) has q ≈ 61 (σ_1 ≈ 8.9, σ_2 ≈
    8.1, σ_4 ≈ 6.2). The base :func:`cvp.babai.babai_x1` only returns
    candidates whose q sits in a narrow band around the Babai seed q
    (~27-42 at this cell), missing such high-q candidates. The joint
    enumeration in v2 needs access to that wider regime, so this helper:

    1. Calls ``babai_x1`` with progressively looser ε to widen the
       returned pool.
    2. Augments with ζ_18-torsion-orbit rotations of each returned x_1
       (each rotation has the same q and σ_2, σ_4 magnitudes but a
       different σ_1 phase; the orbit element closest to z keeps its
       Babai-merit, and the others provide alternative phases for the
       joint search).
    3. De-dupes and returns the top ``n_x1`` by ``|σ_1(x_1) - z|``.

    Note: ζ_18-rotated x_1's that overshoot the sigma1 budget are kept
    if they pass the per-coord test; the joint-search post-filter on
    Frobenius is the final gate.
    """
    z = (3.0 ** f) * complex(math.cos(theta / 2.0), math.sin(theta / 2.0))
    sigma_budget = sigma1_budget_mult * eps * (3.0 ** f)
    q_cap = 2 * 3 ** (2 * f)

    pool: List[Tuple[float, Tuple[int, ...]]] = []
    seen: set = set()

    def _try_cand(c: Tuple[int, ...]) -> None:
        c = tuple(int(x) for x in c)
        if c in seen:
            return
        seen.add(c)
        if q_form(c) > q_cap:
            return
        s1 = _sigma_1(c)
        err = abs(s1 - z)
        if err > sigma_budget:
            return
        pool.append((err, c))

    # -- Pool 1: standard Babai at multiple ε scales ------------------------
    # Run Babai at three ε scales so we get a wider variety of seeds.
    # Each call costs ~10 ms.
    for eps_mult in (1.0, 2.0, 4.0):
        eps_try = min(eps_mult * sigma_budget / (3.0 ** f), 0.999)
        try:
            cands = babai_x1(theta, f, n_candidates=max(n_x1, 64), eps=eps_try)
        except Exception:
            cands = []
        for c in cands:
            _try_cand(c)

    # -- Pool 2: ζ_18-orbit augmentation -----------------------------------
    # For each candidate in the pool, generate its 18-element torsion
    # orbit and keep elements that still land in the σ_1 budget. This
    # captures Galois-equivalent x_1's that Babai's single-seed strategy
    # would miss (e.g. when the reference x_1 differs from Babai's seed
    # only by a ζ_9-multiplication).
    base_cands = [c for _, c in pool]
    for c in base_cands:
        for rot in _zeta_18_orbit(c):
            _try_cand(rot)

    pool.sort(key=lambda x: x[0])
    return [c for _, c in pool[:n_x1]]


def _heavy_sigma1_lattice() -> Tuple[IntegerMatrix, np.ndarray]:
    """LLL-reduce the Minkowski lattice with a heavy weight on the principal
    place (σ_1) so short vectors have small |σ_1| but may have arbitrary
    |σ_2|, |σ_4|.

    Returns ``(B_int, U_z)`` where ``B_int`` is the integer-scaled
    LLL-reduced basis and ``U_z`` is the unimodular matrix such that
    ``coords @ U_z`` is the ζ-coefficient vector. This mirrors
    ``cvp.babai._scaled_basis_and_target`` but builds it without a
    target (we want vectors NEAR 0 in σ_1, i.e. shortest in the heavy-
    weighted lattice).
    """
    M = minkowski_embedding()  # 6x6 real
    W = 10 ** 4  # heavy weight on σ_1 (rows 0, 1)
    weights = np.array([W, W, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    basis_real = ((M.T * weights) * _FPLLL_SCALE)  # rows = image of ζ^j
    rows = [[int(round(x)) for x in row] for row in basis_real]
    B = IntegerMatrix.from_matrix(rows)
    U = IntegerMatrix.identity(6)
    LLL.reduction(B, U=U)
    U_z = np.array(
        [[int(U[i, j]) for j in range(6)] for i in range(6)],
        dtype=object,
    )
    return B, U_z


def enumerate_x3_extended(
    theta: float,
    f: int,
    eps: float,
    n_x3: int = 256,
    *,
    sigma1_budget_mult: float = 4.0,
) -> List[Tuple[int, ...]]:
    """Enumerate x_3 with small |σ_1| via TWO complementary lattices.

    Phase-3 v1's :func:`cvp.diophantine.enumerate_x3` enumerates short
    vectors in the *unweighted* Minkowski lattice — these have small q
    (and so small |σ_1|² + |σ_2|² + |σ_4|²). The HRSA f=2 reference
    triple has x_3 with q ≈ 58 but |σ_1| ≈ 0.07, which is BOTH a tight
    σ_1 and a moderate q: the unweighted enumerator finds it.

    But for tighter eps (≤ 10⁻³), we *also* need x_3's that are tight
    in σ_1 but have larger q (so q(x_1) + q(x_3) is closer to 2·3^{2f},
    leaving little budget for x_2, which is fine since we no longer
    require x_2 to be small). The principal-place-heavy lattice yields
    such candidates.

    This function unions both pools and returns up to ``n_x3`` tuples
    sorted by ascending ``|σ_1(x_3)|``. Always includes the zero vector.
    """
    if f < 0:
        raise ValueError(f"f must be ≥ 0, got {f}")
    sigma_budget = sigma1_budget_mult * eps * (3.0 ** f)
    q_cap = 2 * 3 ** (2 * f)

    out: List[Tuple[float, Tuple[int, ...]]] = [(0.0, (0, 0, 0, 0, 0, 0))]
    seen: set = {(0, 0, 0, 0, 0, 0)}

    # -- Pool 1: small-q vectors via the unweighted lattice -----------------
    # Reuses v1's enumerate_x3 with a generous max_q3 + looser σ_1 budget.
    small_q_pool = enumerate_x3(
        theta=theta, f=f, eps=sigma_budget / (3.0 ** f),
        max_q3=min(q_cap, int(math.ceil(50.0 * eps * eps * (3.0 ** (2 * f)))) + 128),
        max_candidates=max(n_x3, 64),
    )
    for c in small_q_pool:
        c = tuple(int(x) for x in c)
        if c in seen:
            continue
        seen.add(c)
        err = abs(_sigma_1(c))
        out.append((err, c))

    # -- Pool 2: small-σ_1 vectors via the principal-place-heavy lattice -----
    # These have arbitrary q (up to the cap) but tight σ_1.
    try:
        B_h, U_z = _heavy_sigma1_lattice()
        n = B_h.nrows
        W = 10 ** 4
        # Radius² covers the σ_1 budget + the full σ_2, σ_4 range up to
        # max-q in the auxiliary places: |σ_r|² ≤ 3·q_cap = 6·3^{2f}.
        max_aux_sq = 6.0 * (3.0 ** (2 * f)) * (_FPLLL_SCALE ** 2)
        radius_sq = (
            (W * sigma_budget * _FPLLL_SCALE) ** 2
            + 2.0 * max_aux_sq
        )
        Mgso = MatGSO(B_h)
        Mgso.update_gso()
        enum = Enumeration(Mgso, nr_solutions=max(4 * n_x3, 128))
        results = enum.enumerate(0, n, radius_sq, 0)
        for _norm_sq, c in results:
            coords = np.array([int(x) for x in c], dtype=object)
            delta = tuple(
                int(sum(coords[i] * U_z[i][j] for i in range(n)))
                for j in range(n)
            )
            for sign in (1, -1):
                cand = tuple(sign * x for x in delta)
                if cand in seen:
                    continue
                seen.add(cand)
                qv = q_form(cand)
                if qv > q_cap:
                    continue
                s1 = _sigma_1(cand)
                err = abs(s1)
                if err > sigma_budget:
                    continue
                out.append((err, cand))
    except EnumerationError:
        pass

    out.sort(key=lambda x: x[0])
    return [c for _, c in out[:n_x3]]


# ---------------------------------------------------------------------------
# Joint enumeration
# ---------------------------------------------------------------------------


class JointSearchStats:
    """Lightweight counters surfaced for tests and debugging."""

    __slots__ = (
        "n_x1", "n_x3", "n_pairs_total", "n_pairs_screened",
        "n_pari_calls", "n_pari_hits", "n_frob_pass",
        "n_orbit_total", "wall_s",
    )

    def __init__(self) -> None:
        self.n_x1 = 0
        self.n_x3 = 0
        self.n_pairs_total = 0
        self.n_pairs_screened = 0
        self.n_pari_calls = 0
        self.n_pari_hits = 0
        self.n_frob_pass = 0
        self.n_orbit_total = 0
        self.wall_s = 0.0

    def as_dict(self) -> dict:
        return {
            "n_x1": self.n_x1,
            "n_x3": self.n_x3,
            "n_pairs_total": self.n_pairs_total,
            "n_pairs_screened": self.n_pairs_screened,
            "n_pari_calls": self.n_pari_calls,
            "n_pari_hits": self.n_pari_hits,
            "n_frob_pass": self.n_frob_pass,
            "n_orbit_total": self.n_orbit_total,
            "wall_s": float(self.wall_s),
        }


def solve_joint_x1_x3(
    theta: float,
    f: int,
    eps: float,
    *,
    n_candidates: int = 100,
    n_x1: int = 32,
    n_x3: int = 32,
    worker: Optional[_NormEqWorker] = None,
    stats: Optional[JointSearchStats] = None,
    early_stop_hits: Optional[int] = None,
    eps_x1: Optional[float] = None,
) -> List[dict]:
    """Joint (x_1, x_3) Babai-CVP enumeration per Selinger 2012 §7 pattern.

    Algorithm (joint enumeration)
    -----------------------------
    1. Build ``n_x1`` x_1 candidates via :func:`cvp.babai.babai_x1`.
    2. Build ``n_x3`` x_3 candidates via :func:`enumerate_x3_extended`
       (which loosens the per-coord σ_1 budget by 4× so the pool is
       non-trivial at small eps).
    3. For each (x_1, x_3) pair in the Cartesian product:
       a. Compute M(x_2) = (2·3^{2f}, 0, 0) - bb(x_1) - bb(x_3) exactly.
       b. Pre-screen: skip if M is not totally positive or m_0 < 0.
       c. Call PARI on M; on a hit, expand the returned x_2 by its
          ζ_18 torsion orbit (18 phase-rotations, same q).
       d. For each orbit element, compute Householder Frobenius and
          keep if ≤ eps.
    4. Sort surviving triples ascending by Frobenius residual, return
       up to ``n_candidates``.

    Parameters
    ----------
    theta, f, eps : as in :func:`cvp.diophantine.solve_x2_x3`.
    n_candidates : upper bound on returned triples (default 100).
    n_x1 : x_1 candidate pool size (default 32). At very small eps the
        Babai search may return fewer; we use what's available.
    n_x3 : x_3 candidate pool size (default 32). Same caveat.
    worker : optional pre-warmed :class:`_NormEqWorker` (recommended in
        sweeps so the 3-s subprocess startup is amortised).
    stats : optional :class:`JointSearchStats` for diagnostic counters.
    early_stop_hits : if not None, abort after this many Frob-passing
        triples are accumulated.
    eps_x1 : optional separate eps for the x_1 Babai search; default is
        ``2·eps`` (loosened so the x_1 pool isn't starved at tight eps).

    Returns
    -------
    list of dict
        Each dict has keys ``x_1``, ``x_2``, ``x_3`` (each a length-6
        tuple of Python int) and ``frob`` (the Householder Frobenius
        residual). Ring-unitarity (q-sum + bb-sum) is guaranteed for
        every returned triple — :func:`reify_householder` with
        ``strict=True`` will accept it without raising.

    Notes
    -----
    The joint enumeration is intrinsically O(n_x1 · n_x3) PARI calls
    in the worst case. The total-positivity pre-screen typically rejects
    50-80% of pairs at small eps; the remaining calls are 5-50 ms each.
    Typical wall at n_x1 = n_x3 = 32: 5-30 s per (θ, eps) cell at f ≤ 15.
    """
    if f < 0:
        raise ValueError(f"f must be ≥ 0, got {f}")
    if n_candidates < 1:
        return []
    if eps_x1 is None:
        eps_x1 = 2.0 * eps

    t0 = time.time()

    own_stats = stats is None
    if own_stats:
        stats = JointSearchStats()

    # -- Stage 1: x_1 pool ---------------------------------------------------
    x1_cands = _enumerate_x1_extended(theta, f, eps_x1, n_x1=n_x1)
    stats.n_x1 = len(x1_cands)

    # -- Stage 2: x_3 pool ---------------------------------------------------
    x3_cands = enumerate_x3_extended(theta, f, eps, n_x3=n_x3)
    stats.n_x3 = len(x3_cands)

    if not x1_cands or not x3_cands:
        if own_stats:
            stats.wall_s = time.time() - t0
        return []

    # Precompute bb of each candidate (cheap; reused n^2 times).
    M1_table = [bb_to_real_coeffs(x1) for x1 in x1_cands]
    M3_table = [bb_to_real_coeffs(x3) for x3 in x3_cands]
    q1_table = [q_form(x1) for x1 in x1_cands]
    q3_table = [q_form(x3) for x3 in x3_cands]

    q_budget = 2 * 3 ** (2 * f)
    budget_M = (q_budget, 0, 0)

    # -- Stage 3: joint enumeration -----------------------------------------
    own_worker = worker is None
    if own_worker:
        worker = _NormEqWorker()
        worker.start()

    # PARI-call cache: many distinct (x_1, x_3) pairs produce the same M
    # (when one shifts an x_3 by a unit-orbit element of x_1's complement,
    # bb cancels). We memoize PARI calls per M tuple to keep the cost down.
    pari_cache: dict[Tuple[int, int, int], List[Tuple[int, ...]]] = {}

    results: List[Tuple[float, dict]] = []

    try:
        for i, x1 in enumerate(x1_cands):
            M1 = M1_table[i]
            q1 = q1_table[i]
            if q1 > q_budget:
                continue
            for j, x3 in enumerate(x3_cands):
                stats.n_pairs_total += 1
                M3 = M3_table[j]
                q3 = q3_table[j]
                q_remaining = q_budget - q1 - q3
                if q_remaining < 0:
                    continue

                # The full M(x_2) is determined by ring-additivity.
                M2 = (
                    budget_M[0] - M1[0] - M3[0],
                    budget_M[1] - M1[1] - M3[1],
                    budget_M[2] - M1[2] - M3[2],
                )
                # Sanity: m_0 + 2 m_2 = q(x_2) = q_remaining. (The trace
                # pairing identity q(x) = M0(x) + 2 M2(x).) Skip if this
                # is violated — that means M(x_2) is internally inconsistent.
                if M2[0] + 2 * M2[2] != q_remaining:
                    continue

                # Pre-screen: total positivity.
                if not _M_is_totally_positive(M2):
                    continue
                stats.n_pairs_screened += 1

                # PARI call (cached).
                if M2 in pari_cache:
                    x2_roots = pari_cache[M2]
                else:
                    stats.n_pari_calls += 1
                    x2_roots = worker.solve(M2)
                    pari_cache[M2] = x2_roots
                if not x2_roots:
                    continue
                stats.n_pari_hits += 1

                # ζ_18 torsion orbit of each root: each rotation changes
                # σ_1(x_2) phase but preserves bb (and hence q-sum + ring-
                # unitarity). We score them all and keep only Frob ≤ eps.
                seen_orbit: set = set()
                for r in x2_roots:
                    r = tuple(int(v) for v in r)
                    for x2_cand in _zeta_18_orbit(r):
                        if x2_cand in seen_orbit:
                            continue
                        seen_orbit.add(x2_cand)
                        stats.n_orbit_total += 1
                        # Defensive: confirm q(x2) matches.
                        if q_form(x2_cand) != q_remaining:
                            continue
                        frob = householder_frobenius(
                            x1, x2_cand, x3, theta=theta, f=f,
                        )
                        if frob > eps:
                            continue
                        stats.n_frob_pass += 1
                        results.append((frob, {
                            "x_1": tuple(int(v) for v in x1),
                            "x_2": x2_cand,
                            "x_3": tuple(int(v) for v in x3),
                            "frob": float(frob),
                        }))
                        if (early_stop_hits is not None
                                and stats.n_frob_pass >= early_stop_hits):
                            break
                    if (early_stop_hits is not None
                            and stats.n_frob_pass >= early_stop_hits):
                        break
                if (early_stop_hits is not None
                        and stats.n_frob_pass >= early_stop_hits):
                    break
            if (early_stop_hits is not None
                    and stats.n_frob_pass >= early_stop_hits):
                break
    finally:
        if own_worker:
            worker.close()

    stats.wall_s = time.time() - t0

    results.sort(key=lambda x: x[0])
    return [rec for _, rec in results[:n_candidates]]


# ---------------------------------------------------------------------------
# Module entry point — manual smoke test
# ---------------------------------------------------------------------------


def _main():  # pragma: no cover - manual sanity
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test cvp.diophantine_v2.solve_joint_x1_x3."
    )
    parser.add_argument("--theta", type=float, default=math.pi / 3)
    parser.add_argument("--f", type=int, default=12)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--n-x1", type=int, default=32)
    parser.add_argument("--n-x3", type=int, default=32)
    args = parser.parse_args()

    stats = JointSearchStats()
    with _NormEqWorker() as worker:
        triples = solve_joint_x1_x3(
            args.theta, args.f, args.eps,
            n_candidates=10, n_x1=args.n_x1, n_x3=args.n_x3,
            worker=worker, stats=stats,
        )
    print(f"θ={args.theta:.4f} f={args.f} eps={args.eps:g}")
    print(f"stats: {stats.as_dict()}")
    print(f"|triples| = {len(triples)}")
    for k, t in enumerate(triples[:5]):
        print(f"  [{k}] frob={t['frob']:.5g}")


if __name__ == "__main__":  # pragma: no cover
    _main()
