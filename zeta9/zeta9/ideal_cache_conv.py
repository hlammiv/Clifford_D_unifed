"""γ → M = γ·γ̄ conversion for K = Q(ζ_9), F = Q(α).

Given γ ∈ O_K in integral basis (c₀, c₁, …, c₅) ↔ Σ c_j ζ_9^j, compute
M = γ·γ̄ ∈ O_F in F's basis (m_0, m_1, m_2) ↔ m_0 + m_1·α + m_2·α²
where α = ζ_9 + ζ_9⁻¹.

Derivation
==========
Let ω = ζ_9. We use the relations:
  Φ_9(x) = x^6 + x^3 + 1 = 0   ⇒   ω^6 = -1 - ω^3
  ω^7 = ω · ω^6 = -ω - ω^4
  ω^8 = ω² · ω^6 = -ω² - ω^5

So the powers ω^j for j = 0..8 reduce to combos of (1, ω, ω², ω³, ω⁴, ω⁵).

For γ = Σ_{j=0}^{5} c_j ω^j, the complex conjugate is
  γ̄ = Σ c_j ω^{-j} = Σ c_j ω^{9-j} (mod Φ_9)
     = c_0 + c_1·ω^8 + c_2·ω^7 + c_3·ω^6 + c_4·ω^5 + c_5·ω^4

Using above reductions:
  γ̄ = c_0 + c_1(-ω² - ω^5) + c_2(-ω - ω^4) + c_3(-1 - ω^3) + c_4·ω^5 + c_5·ω^4
     = (c_0 - c_3) + (-c_2)ω + (-c_1)ω² + (-c_3)ω³ + (-c_2 + c_5)ω⁴ + (-c_1 + c_4)ω⁵

(Quick sanity check: γ̄ should have real same as γ when γ is real, i.e., c_3 = c_5 = c_4 = 0 forces γ̄ = c_0 - c_1·ω² - c_2·ω, hmm — this is only "real" when c_1 = c_2 = 0 too. Real elements of K are those fixed by complex conjugation: solve γ = γ̄ → m_0 + m_1·α + m_2·α² coefficients.)

Then M = γ·γ̄ is computed by polynomial multiplication mod Φ_9, yielding a
length-6 vector (M_0, M_1, ..., M_5) which must reduce to a length-3 vector
in F's basis because M is fixed by complex conjugation.

To convert (M_0,...,M_5) → (m_0, m_1, m_2) ↔ m_0 + m_1·α + m_2·α²:
  α = ω + ω⁻¹ = ω + ω^8 = ω - ω² - ω^5    (using ω^8 = -ω² - ω^5)
  α² = (ω + ω^8)² = ω² + 2·ω·ω^8 + ω^{16} = ω² + 2·ω^9 + ω^{16}
       wait ω^9 = 1; ω^{16} = ω^7 = -ω - ω^4
     = ω² + 2 + (-ω - ω^4)
     = 2 - ω + ω² - ω^4

So in (1, ω, ω², ω³, ω⁴, ω⁵) basis:
  1   = (1, 0, 0, 0, 0, 0)
  α   = (0, 1, -1, 0, 0, -1)
  α²  = (2, -1, 1, 0, -1, 0)

For M = m_0 + m_1·α + m_2·α², we get:
  M_0 = m_0 + 2 m_2
  M_1 = m_1 - m_2
  M_2 = -m_1 + m_2
  M_3 = 0
  M_4 = -m_2
  M_5 = -m_1

These give a 6→3 linear map (with redundancy because real elements are
3-dim in the 6-dim K). Invert: from (M_0, M_1, M_4, M_5) we can recover
(m_0, m_1, m_2):
  m_2 = -M_4
  m_1 = -M_5
  m_0 = M_0 - 2 m_2 = M_0 + 2 M_4

Sanity check via M_1, M_2: should have M_2 = -M_1 (anti-symmetry under conj),
M_3 = 0. We can verify but for performance only compute (M_0, M_4, M_5) of
the product.

Multiplication algorithm
========================
γ·γ̄ = (Σ c_j ω^j) · (Σ c_k' ω^{9-k})   where c_k' = c_k

Let d_j = γ̄'s coefficient at ω^j (from above):
  d_0 = c_0 - c_3
  d_1 = -c_2
  d_2 = -c_1
  d_3 = -c_3
  d_4 = -c_2 + c_5
  d_5 = -c_1 + c_4

Then γ·γ̄ = Σ_{j,k} c_j d_k ω^{j+k}. Modulo Φ_9 (= x^6 + x^3 + 1), reduce
ω^j for j ≥ 6 using:
  ω^6 = -1 - ω^3
  ω^7 = -ω - ω^4
  ω^8 = -ω² - ω^5
  ω^9 = 1
  ω^{10} = ω
  ω^{11} = ω²
  ω^{12} = ω^3
  ω^{13} = ω^4
  ω^{14} = ω^5
  ω^{15} = ω^6 = -1 - ω^3, etc.

Maximum j+k = 5+5 = 10, so we reduce ω^6..ω^{10}. Pre-tabulate the reduction.

We only need to extract M_0, M_4, M_5 (skip the others — but for verification
we'll compute all 6 in the reference Python version, then validate against
Numba batched).
"""
from __future__ import annotations

import numpy as np
import numba as nb


# Reduction table: ω^j for j in [0, 10] expressed in (1, ω, ω², ω³, ω⁴, ω⁵)
# Each row is the coefficient vector of ω^j in the basis.
_OMEGA_REDUCED = np.array([
    [1, 0, 0, 0, 0, 0],   # ω^0 = 1
    [0, 1, 0, 0, 0, 0],   # ω^1 = ω
    [0, 0, 1, 0, 0, 0],   # ω^2
    [0, 0, 0, 1, 0, 0],   # ω^3
    [0, 0, 0, 0, 1, 0],   # ω^4
    [0, 0, 0, 0, 0, 1],   # ω^5
    [-1, 0, 0, -1, 0, 0], # ω^6 = -1 - ω^3
    [0, -1, 0, 0, -1, 0], # ω^7 = -ω - ω^4
    [0, 0, -1, 0, 0, -1], # ω^8 = -ω² - ω^5
    [1, 0, 0, 0, 0, 0],   # ω^9 = 1
    [0, 1, 0, 0, 0, 0],   # ω^10 = ω
], dtype=np.int64)
assert _OMEGA_REDUCED.shape == (11, 6)


def gamma_to_M_coefs_python(gamma_coefs: np.ndarray) -> np.ndarray:
    """Reference Python implementation. Returns (m_0, m_1, m_2) for M = γ·γ̄.

    gamma_coefs: shape (6,) int, coefficients of γ in (1, ω, ..., ω^5) basis.
    Returns: shape (3,) int, coefficients of M in (1, α, α²) basis.
    """
    c = np.asarray(gamma_coefs, dtype=np.int64)
    assert c.shape == (6,)

    # γ̄ coefficients
    d = np.array([
        c[0] - c[3],   # d_0
        -c[2],         # d_1
        -c[1],         # d_2
        -c[3],         # d_3
        -c[2] + c[5],  # d_4
        -c[1] + c[4],  # d_5
    ], dtype=np.int64)

    # Multiply: M_jk = c_j * d_k, then sum over j+k = constant.
    # Resulting polynomial in ω has degree ≤ 10.
    M_full = np.zeros(6, dtype=np.int64)  # final reduced
    for j in range(6):
        for k in range(6):
            exponent = j + k
            M_full += c[j] * d[k] * _OMEGA_REDUCED[exponent]

    # Extract (m_0, m_1, m_2) from (M_0, M_4, M_5):
    # M_0 = m_0 + 2 m_2 (from α² = 2 - ω + ω² - ω⁴), so m_0 = M_0 - 2 m_2.
    # M_4 = -m_2, M_5 = -m_1.
    # Therefore: m_2 = -M_4, m_1 = -M_5, m_0 = M_0 - 2·(-M_4) = M_0 + 2·M_4.
    m_2 = -M_full[4]
    m_1 = -M_full[5]
    m_0 = M_full[0] + 2 * M_full[4]

    # Sanity (in debug): M_1 should equal -M_2, M_3 = 0
    # (M_1 - (-M_2)) and M_3 should both be 0 for the conversion to be exact.
    # Skipping in production for speed.
    return np.array([m_0, m_1, m_2], dtype=np.int64)


# Precompute the reduction coefficient table as a (11, 6) numpy array for Numba.
# (Same data as _OMEGA_REDUCED but in the namespace Numba can see.)
_OMEGA_RED_NB = _OMEGA_REDUCED.copy()


@nb.njit(cache=True, inline="always")
def _gamma_to_M_single(c0, c1, c2, c3, c4, c5):
    """Single-row Numba kernel. Returns (m_0, m_1, m_2)."""
    # γ̄ coefficients
    d0 = c0 - c3
    d1 = -c2
    d2 = -c1
    d3 = -c3
    d4 = -c2 + c5
    d5 = -c1 + c4

    # Accumulate M's coefficients in (1, ω, ω², ω³, ω⁴, ω⁵). Only M_0, M_4, M_5
    # are needed (others recovered or zero).
    M0 = nb.int64(0)
    M4 = nb.int64(0)
    M5 = nb.int64(0)
    # We need to iterate j+k = 0, 4, 5 (mod 9, with reduction)
    # Each (j, k) contributes c_j * d_k * _OMEGA_REDUCED[j+k][target] to M[target].
    # Hardcode the relevant entries.

    # Reduction tables: power_to_basis_at_idx[power][idx] = _OMEGA_REDUCED[power][idx]
    # For idx ∈ {0, 4, 5}:
    # power: 0  1  2  3  4  5  6  7  8  9 10
    # idx 0: 1  0  0  0  0  0 -1  0  0  1  0
    # idx 4: 0  0  0  0  1  0  0 -1  0  0  0
    # idx 5: 0  0  0  0  0  1  0  0 -1  0  0

    # Unroll: for each (j, k) with j+k in {0,1,...,10}, add to M0, M4, M5.
    cs = (c0, c1, c2, c3, c4, c5)
    ds = (d0, d1, d2, d3, d4, d5)
    for j in range(6):
        cj = cs[j]
        for k in range(6):
            exp = j + k
            prod = cj * ds[k]
            # M0: nonzero at exp = 0, 6, 9
            if exp == 0:
                M0 += prod
            elif exp == 6:
                M0 -= prod
            elif exp == 9:
                M0 += prod
            # M4: nonzero at exp = 4, 7
            if exp == 4:
                M4 += prod
            elif exp == 7:
                M4 -= prod
            # M5: nonzero at exp = 5, 8
            if exp == 5:
                M5 += prod
            elif exp == 8:
                M5 -= prod

    m_2 = -M4
    m_1 = -M5
    m_0 = M0 + 2 * M4
    return m_0, m_1, m_2


@nb.njit(cache=True)
def gamma_to_M_batch(gamma_coefs: np.ndarray, out: np.ndarray):
    """Batched Numba kernel: (N, 6) γ coefficients → (N, 3) M coefficients."""
    N = gamma_coefs.shape[0]
    for i in range(N):
        c0 = gamma_coefs[i, 0]; c1 = gamma_coefs[i, 1]; c2 = gamma_coefs[i, 2]
        c3 = gamma_coefs[i, 3]; c4 = gamma_coefs[i, 4]; c5 = gamma_coefs[i, 5]
        m_0, m_1, m_2 = _gamma_to_M_single(c0, c1, c2, c3, c4, c5)
        out[i, 0] = m_0
        out[i, 1] = m_1
        out[i, 2] = m_2


# ---------------------------------------------------------------------------
# Multiplication in O_K = Z[ω] / Φ_9(ω) for σ-fit unit translates.
# γ · u where γ, u ∈ O_K both in (1, ω, ..., ω^5) basis.
# ---------------------------------------------------------------------------

@nb.njit(cache=True, inline="always")
def _mul_in_OK(a0, a1, a2, a3, a4, a5,
               b0, b1, b2, b3, b4, b5):
    """Multiply two elements of O_K. Both in (1, ω, ..., ω^5) basis.

    Reduction table for ω^j (j = 0..10) in (1, ω, ω², ω³, ω⁴, ω⁵):
      ω^0   = ( 1, 0, 0, 0, 0, 0)
      ω^1   = ( 0, 1, 0, 0, 0, 0)
      ω^2   = ( 0, 0, 1, 0, 0, 0)
      ω^3   = ( 0, 0, 0, 1, 0, 0)
      ω^4   = ( 0, 0, 0, 0, 1, 0)
      ω^5   = ( 0, 0, 0, 0, 0, 1)
      ω^6   = (-1, 0, 0,-1, 0, 0)
      ω^7   = ( 0,-1, 0, 0,-1, 0)
      ω^8   = ( 0, 0,-1, 0, 0,-1)
      ω^9   = ( 1, 0, 0, 0, 0, 0)   (ω^9 = 1)
      ω^10  = ( 0, 1, 0, 0, 0, 0)
    """
    a = (a0, a1, a2, a3, a4, a5)
    b = (b0, b1, b2, b3, b4, b5)

    r0 = nb.int64(0); r1 = nb.int64(0); r2 = nb.int64(0)
    r3 = nb.int64(0); r4 = nb.int64(0); r5 = nb.int64(0)

    for j in range(6):
        aj = a[j]
        for k in range(6):
            exp = j + k
            prod = aj * b[k]
            if exp == 0:
                r0 += prod
            elif exp == 1:
                r1 += prod
            elif exp == 2:
                r2 += prod
            elif exp == 3:
                r3 += prod
            elif exp == 4:
                r4 += prod
            elif exp == 5:
                r5 += prod
            elif exp == 6:
                r0 -= prod; r3 -= prod
            elif exp == 7:
                r1 -= prod; r4 -= prod
            elif exp == 8:
                r2 -= prod; r5 -= prod
            elif exp == 9:
                r0 += prod
            elif exp == 10:
                r1 += prod
    return r0, r1, r2, r3, r4, r5


@nb.njit(cache=True)
def mul_in_OK_batch(a_coefs: np.ndarray, b_coefs: np.ndarray, out: np.ndarray):
    """Batched: out[i] = a_coefs[i] · b_coefs[i] in O_K."""
    N = a_coefs.shape[0]
    for i in range(N):
        r = _mul_in_OK(
            a_coefs[i, 0], a_coefs[i, 1], a_coefs[i, 2],
            a_coefs[i, 3], a_coefs[i, 4], a_coefs[i, 5],
            b_coefs[i, 0], b_coefs[i, 1], b_coefs[i, 2],
            b_coefs[i, 3], b_coefs[i, 4], b_coefs[i, 5],
        )
        for k in range(6):
            out[i, k] = r[k]


def mul_in_OK_python(a_coefs: np.ndarray, b_coefs: np.ndarray) -> np.ndarray:
    """Python reference: γ · γ' in O_K for testing."""
    a = np.asarray(a_coefs, dtype=np.int64); b = np.asarray(b_coefs, dtype=np.int64)
    out = np.zeros(6, dtype=np.int64)
    for j in range(6):
        for k in range(6):
            out += a[j] * b[k] * _OMEGA_REDUCED[j + k]
    return out


def compute_unit_power_table(unit_coefs: np.ndarray, unit_inv_coefs: np.ndarray,
                              k_max: int) -> np.ndarray:
    """Precompute u^k for k in [-k_max, k_max].

    unit_coefs: shape (6,) — the unit u in (1, ω, ..., ω^5) basis.
    unit_inv_coefs: shape (6,) — u⁻¹ in same basis (must be computed externally,
        e.g. via PARI nfeltdiv).
    k_max: pre-compute u^k for k ∈ [-k_max, k_max].

    Returns: shape (2·k_max + 1, 6) array where row k_max + k = u^k.
    Row k_max = u^0 = (1, 0, 0, 0, 0, 0).
    """
    n = 2 * k_max + 1
    table = np.zeros((n, 6), dtype=np.int64)
    # u^0 = 1
    table[k_max, 0] = 1
    # u^1 = unit
    table[k_max + 1] = unit_coefs
    # u^{-1} = unit_inv
    table[k_max - 1] = unit_inv_coefs
    # u^k for k = 2, 3, ..., k_max: repeated multiplication
    for k in range(2, k_max + 1):
        prev = table[k_max + k - 1]
        table[k_max + k] = mul_in_OK_python(prev, unit_coefs)
    for k in range(2, k_max + 1):
        prev = table[k_max - (k - 1)]
        table[k_max - k] = mul_in_OK_python(prev, unit_inv_coefs)
    return table


@nb.njit(cache=True)
def sigma_fit_batch(
    gamma_logs: np.ndarray,         # (N, 3) cached log embeddings
    target_log_2L1: nb.float64,
    tol_log_2L1: nb.float64,
    max_log_2L2: nb.float64,
    max_log_2L3: nb.float64,
    U: np.ndarray,                  # (2, 3) fundamental unit log-embeddings
    k_radius: nb.int64,
    out_record_idx: np.ndarray,     # output: cache indices with σ-fit hits (preallocated)
    out_k1: np.ndarray,             # output: k_1 exponent per hit
    out_k2: np.ndarray,             # output: k_2 exponent per hit
) -> int:
    """Vectorized σ-fit: for each cache record γ, find integer (k_1, k_2)
    such that γ · u_1^{k_1} · u_2^{k_2} has σ_K embeddings satisfying:
        2 log|σ_K_1| in [target_log_2L1 ± tol_log_2L1]   (narrow band)
        2 log|σ_K_2| ≤ max_log_2L2                        (upper bound)
        2 log|σ_K_4| ≤ max_log_2L3                        (upper bound)

    Search strategy: for each record, the narrow L_1 band defines a 1D line
    in (k_1, k_2) space; find the rounded center (k1, k2)_center, then search
    a (2·k_radius+1)² integer box around it for points in the band.

    Caller must preallocate out_* arrays large enough for the expected hit
    count. Returns the number of hits written; if it equals out_record_idx.size,
    OVERFLOW occurred (caller should retry with larger output buffer).

    Per-record cost: ~ (2·k_radius+1)² × O(1) = ~50-200 ns at k_radius=3.
    """
    n_hits = 0
    n_cache = gamma_logs.shape[0]
    max_out = out_record_idx.shape[0]

    a00 = U[0, 0]  # log|σ_K_1(u_1)|
    a10 = U[1, 0]  # log|σ_K_1(u_2)|
    denom = a00 * a00 + a10 * a10

    if denom < 1e-30:
        # Both units have ~0 effect on σ_K_1 — degenerate; can't satisfy band
        return 0

    band_lo = target_log_2L1 - tol_log_2L1
    band_hi = target_log_2L1 + tol_log_2L1

    for i in range(n_cache):
        g0 = gamma_logs[i, 0]
        g1 = gamma_logs[i, 1]
        g2 = gamma_logs[i, 2]

        # Solve: 2·(g0 + k1·a00 + k2·a10) = target_log_2L1 (line in k_1, k_2)
        rhs = target_log_2L1 * 0.5 - g0
        k1_center_f = rhs * a00 / denom
        k2_center_f = rhs * a10 / denom
        k1_center = int(round(k1_center_f))
        k2_center = int(round(k2_center_f))

        for dk1 in range(-k_radius, k_radius + 1):
            k1 = k1_center + dk1
            for dk2 in range(-k_radius, k_radius + 1):
                k2 = k2_center + dk2
                s0 = g0 + k1 * U[0, 0] + k2 * U[1, 0]
                two_L1 = 2.0 * s0
                if two_L1 < band_lo or two_L1 > band_hi:
                    continue
                s1 = g1 + k1 * U[0, 1] + k2 * U[1, 1]
                if 2.0 * s1 > max_log_2L2:
                    continue
                s2 = g2 + k1 * U[0, 2] + k2 * U[1, 2]
                if 2.0 * s2 > max_log_2L3:
                    continue

                # Hit
                if n_hits < max_out:
                    out_record_idx[n_hits] = i
                    out_k1[n_hits] = k1
                    out_k2[n_hits] = k2
                n_hits += 1
    return n_hits


@nb.njit(cache=True)
def gamma_times_unit_pow_to_M_batch(
    gamma_coefs: np.ndarray,       # (N, 6) cached γ
    k1_arr: np.ndarray,            # (N,) k_1 exponents (one per hit)
    k2_arr: np.ndarray,            # (N,) k_2 exponents
    u1_power_table: np.ndarray,    # (2*k_max+1, 6) precomputed u_1^k
    u2_power_table: np.ndarray,    # (2*k_max+1, 6) precomputed u_2^k
    k_max: int,                    # offset for power-table indexing
    out_M: np.ndarray,             # (N, 3) output
):
    """For each hit: γ' = γ · u_1^{k1} · u_2^{k2}, then M = γ'·γ̄'.

    All in (1, ω, ..., ω^5) basis for K, then convert to (1, α, α²) for F.
    """
    N = gamma_coefs.shape[0]
    for i in range(N):
        # Load γ
        g0 = gamma_coefs[i, 0]; g1 = gamma_coefs[i, 1]; g2 = gamma_coefs[i, 2]
        g3 = gamma_coefs[i, 3]; g4 = gamma_coefs[i, 4]; g5 = gamma_coefs[i, 5]

        k1 = k1_arr[i]
        k2 = k2_arr[i]
        u1 = u1_power_table[k_max + k1]  # (6,)
        u2 = u2_power_table[k_max + k2]

        # γ' = γ · u_1^k1
        if k1 != 0:
            t0, t1, t2, t3, t4, t5 = _mul_in_OK(
                g0, g1, g2, g3, g4, g5,
                u1[0], u1[1], u1[2], u1[3], u1[4], u1[5],
            )
            g0, g1, g2, g3, g4, g5 = t0, t1, t2, t3, t4, t5

        # γ'' = γ' · u_2^k2
        if k2 != 0:
            t0, t1, t2, t3, t4, t5 = _mul_in_OK(
                g0, g1, g2, g3, g4, g5,
                u2[0], u2[1], u2[2], u2[3], u2[4], u2[5],
            )
            g0, g1, g2, g3, g4, g5 = t0, t1, t2, t3, t4, t5

        # M = γ'' · γ''̄
        m_0, m_1, m_2 = _gamma_to_M_single(g0, g1, g2, g3, g4, g5)
        out_M[i, 0] = m_0
        out_M[i, 1] = m_1
        out_M[i, 2] = m_2


@nb.njit(cache=True)
def gamma_times_unit_pow_to_M_and_gamma_batch(
    gamma_coefs: np.ndarray,       # (N, 6) cached γ
    k1_arr: np.ndarray,            # (N,) k_1 exponents
    k2_arr: np.ndarray,            # (N,) k_2 exponents
    u1_power_table: np.ndarray,    # (2*k_max+1, 6) precomputed u_1^k
    u2_power_table: np.ndarray,    # (2*k_max+1, 6) precomputed u_2^k
    k_max: int,                    # offset for power-table indexing
    out_M: np.ndarray,             # (N, 3) output: M coefs in F basis
    out_gamma: np.ndarray,         # (N, 6) output: γ' = γ·u_1^k1·u_2^k2 in O_K
):
    """For each hit: γ' = γ · u_1^{k1} · u_2^{k2}, then M = γ'·γ̄'.
    Writes BOTH γ' (in O_K standard basis) and M (in F basis).
    """
    N = gamma_coefs.shape[0]
    for i in range(N):
        g0 = gamma_coefs[i, 0]; g1 = gamma_coefs[i, 1]; g2 = gamma_coefs[i, 2]
        g3 = gamma_coefs[i, 3]; g4 = gamma_coefs[i, 4]; g5 = gamma_coefs[i, 5]

        k1 = k1_arr[i]
        k2 = k2_arr[i]
        u1 = u1_power_table[k_max + k1]
        u2 = u2_power_table[k_max + k2]

        if k1 != 0:
            t0, t1, t2, t3, t4, t5 = _mul_in_OK(
                g0, g1, g2, g3, g4, g5,
                u1[0], u1[1], u1[2], u1[3], u1[4], u1[5],
            )
            g0, g1, g2, g3, g4, g5 = t0, t1, t2, t3, t4, t5
        if k2 != 0:
            t0, t1, t2, t3, t4, t5 = _mul_in_OK(
                g0, g1, g2, g3, g4, g5,
                u2[0], u2[1], u2[2], u2[3], u2[4], u2[5],
            )
            g0, g1, g2, g3, g4, g5 = t0, t1, t2, t3, t4, t5

        out_gamma[i, 0] = g0; out_gamma[i, 1] = g1; out_gamma[i, 2] = g2
        out_gamma[i, 3] = g3; out_gamma[i, 4] = g4; out_gamma[i, 5] = g5

        m_0, m_1, m_2 = _gamma_to_M_single(g0, g1, g2, g3, g4, g5)
        out_M[i, 0] = m_0
        out_M[i, 1] = m_1
        out_M[i, 2] = m_2


def validate_gamma_to_M():
    """Quick sanity test comparing Python reference vs Numba batched.
    Should be called from a test, not at import."""
    # Test cases:
    # γ = 1 → M = 1 → (1, 0, 0)
    # γ = ω → M = ω·ω̄ = ω·ω^8 = ω^9 = 1 → (1, 0, 0)
    # γ = 1 + ω → M = (1+ω)(1+ω^8) = 1 + ω + ω^8 + ω^9 = 1 + ω + ω^8 + 1 = 2 + ω + ω^8 = 2 + α
    #   → (2, 1, 0)
    rng = np.random.default_rng(0)
    N = 100
    test_coefs = rng.integers(-10, 11, size=(N, 6), dtype=np.int64)

    py_out = np.empty((N, 3), dtype=np.int64)
    for i in range(N):
        py_out[i] = gamma_to_M_coefs_python(test_coefs[i])

    nb_out = np.empty((N, 3), dtype=np.int64)
    gamma_to_M_batch(test_coefs.astype(np.int64), nb_out)

    assert np.array_equal(py_out, nb_out), "Python ref ≠ Numba batched"
    return True
