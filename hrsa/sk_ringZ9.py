"""Thin adapter module for tracking exact ringZ9 unitaries through SK.

The heavy arithmetic lives in `ep_level.py` (Z9, Z9Frac, matmul,
conjugate_transpose). This module:

  * Converts between the HRSA-JSON int format (3x3 list of 6-int lists +
    a shared denom exponent f) and lists-of-Z9Frac matrices.
  * Loads ringZ9 representations of the 5184/8100 sign-extended Clifford net
    elements (via ep_descent.load_e0_net_with_ring).
  * Provides ringmat_eval (-> complex ndarray) for verification.
  * Provides chain_mul over an arbitrary list of ringZ9 matrices.
  * Exposes a sign-diag(±1) ringZ9 constructor for the SIGN factors in
    Euler decomposition.

Everything here is exact integer arithmetic on Z[ξ, 1/3]. The "f" exponent
combines additively under products (Z9Frac handles it per-entry).
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import cmath
import numpy as np

from ep_level import Z9, Z9Frac, matmul, conjugate_transpose


# ---------------------------------------------------------------------------
#  Type aliases (for readability; no runtime cost)
# ---------------------------------------------------------------------------

RingMat = list[list[Z9Frac]]   # 3x3 over Z[ξ, 1/3]


# ---------------------------------------------------------------------------
#  Constructors
# ---------------------------------------------------------------------------

def ringmat_identity() -> RingMat:
    """The 3x3 identity over Z[ξ, 1/3] with denom_pow3 = 0."""
    one = Z9Frac.one()
    zero = Z9Frac.zero()
    return [
        [one, zero, zero],
        [zero, one, zero],
        [zero, zero, one],
    ]


def ringmat_from_ints(coefs_3x3x6: Sequence[Sequence[Sequence[int]]],
                      f: int) -> RingMat:
    """Build a RingMat from the HRSA / e0_net int format.

    coefs_3x3x6[i][j] is a length-6 sequence of int coefficients on
    {1, ξ, ξ², ξ³, ξ⁴, ξ⁵}; entry value = (sum_k c_k ξ^k) / 3^f.
    """
    out: RingMat = []
    for i in range(3):
        row: list[Z9Frac] = []
        for j in range(3):
            c = coefs_3x3x6[i][j]
            if len(c) != 6:
                raise ValueError(
                    f"ringmat_from_ints: entry ({i},{j}) has {len(c)} coefs; "
                    f"expected 6"
                )
            num = Z9(tuple(int(x) for x in c))
            row.append(Z9Frac(num, int(f)))
        out.append(row)
    return out


def ringmat_to_ints_with_f(M: RingMat) -> tuple[list[list[list[int]]], int]:
    """Flatten a RingMat to (3x3 list of 6-int lists, shared f) by aligning
    each Z9Frac entry to the maximum denom_pow3 across the matrix.

    Useful at the decompose_tool boundary which wants one shared f.
    """
    f_shared = max(M[i][j].denom_pow3 for i in range(3) for j in range(3))
    out: list[list[list[int]]] = []
    for i in range(3):
        row: list[list[int]] = []
        for j in range(3):
            entry = M[i][j]
            scale = 3 ** (f_shared - entry.denom_pow3)
            scaled_num = entry.num * Z9.from_int(scale) if scale != 1 else entry.num
            row.append(list(scaled_num.coefs))
        out.append(row)
    return out, f_shared


def ringmat_eval(M: RingMat) -> np.ndarray:
    """Embed a RingMat into C^{3x3} using ξ = exp(2πi/9) and double precision.

    For full ε=10⁻¹⁰-style verification, replace cmath.exp(...) with mpmath
    at high precision; this routine is intended for sanity checks at ~1e-15
    float roundoff.
    """
    xi = cmath.exp(2j * cmath.pi / 9.0)
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            entry = M[i][j]
            num_val = 0 + 0j
            for k, a in enumerate(entry.num.coefs):
                if a == 0:
                    continue
                num_val += a * (xi ** k)
            out[i, j] = num_val / (3 ** entry.denom_pow3)
    return out


# ---------------------------------------------------------------------------
#  SIGN diagonals (used by Euler decomposition)
# ---------------------------------------------------------------------------

def ringmat_sign_diag(sp: int) -> RingMat:
    """Build diag(±1, ±1, ±1) over Z[ξ] (f=0) from a 3-bit sign pattern sp.

    Bit k of sp set ⇒ entry (k, k) is -1; clear ⇒ +1.
    """
    M = ringmat_identity()
    minus_one = Z9Frac(Z9.from_int(-1), 0)
    for k in range(3):
        if (sp >> k) & 1:
            M[k][k] = minus_one
    return M


# ---------------------------------------------------------------------------
#  Products
# ---------------------------------------------------------------------------

def ringmat_mul(A: RingMat, B: RingMat) -> RingMat:
    """A @ B over Z[ξ, 1/3] (thin wrapper around ep_level.matmul)."""
    return matmul(A, B)


def ringmat_dagger(M: RingMat) -> RingMat:
    """M† via ep_level.conjugate_transpose (ξ ↦ ξ⁸ on each entry, then
    transpose). Preserves denom_pow3 per entry."""
    return conjugate_transpose(M)


def chain_mul(mats: Sequence[RingMat]) -> RingMat:
    """Sequential matmul of a list of RingMat. Empty list ⇒ identity."""
    acc = ringmat_identity()
    for M in mats:
        acc = matmul(acc, M)
    return acc


# ---------------------------------------------------------------------------
#  e0_net (sign-extended Clifford) ringZ9 cache
# ---------------------------------------------------------------------------

_E0_NET_RING_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}


def _get_e0_net_ring(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Cached load of (V_ring, f_per_idx) from the dump file at `path`.

    V_ring is (N, 3, 3, 6) int array of Z[ξ] coefs; f_per_idx is (N,) int
    array of denominator exponents (entry value = sum_k coef[k]·ξ^k / 3^f).
    """
    p = Path(path)
    if p not in _E0_NET_RING_CACHE:
        from ep_descent import load_e0_net_with_ring
        _, _, V_ring, f_per_idx = load_e0_net_with_ring(p)
        _E0_NET_RING_CACHE[p] = (V_ring, f_per_idx)
    return _E0_NET_RING_CACHE[p]


def ringmat_from_net_idx(idx: int, path: str | Path) -> RingMat:
    """Build the RingMat for net element `idx` from the dump at `path`."""
    V, f_arr = _get_e0_net_ring(Path(path))
    return ringmat_from_ints(V[idx].tolist(), f=int(f_arr[idx]))


_CLIFFORD_RING_CACHE: dict[Path, list["RingMat"]] = {}


def get_clifford_ringmat(ci: int, path: str | Path) -> RingMat:
    """Return the ringZ9 representation of the unsigned Clifford with index ci
    (0 ≤ ci < 648). The mapping matches `su3_euler._load_clifford_set()` exactly.

    Built lazily from the sp=0 subset of the net dump on first call.
    """
    p = Path(path)
    if p not in _CLIFFORD_RING_CACHE:
        from ep_descent import load_e0_net_with_ring
        _, meta, V_ring, f_per_idx = load_e0_net_with_ring(p)
        table: list[RingMat] = [None] * 648  # type: ignore
        for idx in range(len(meta)):
            sp, ci_dump = int(meta[idx, 0]), int(meta[idx, 1])
            if sp != 0:
                continue
            if 0 <= ci_dump < 648 and table[ci_dump] is None:
                table[ci_dump] = ringmat_from_ints(
                    V_ring[idx].tolist(), f=int(f_per_idx[idx])
                )
        # Validate every slot filled.
        if any(m is None for m in table):
            missing = [k for k, m in enumerate(table) if m is None]
            raise RuntimeError(
                f"get_clifford_ringmat: missing ci={missing[:5]}... (total {len(missing)})"
            )
        _CLIFFORD_RING_CACHE[p] = table
    return _CLIFFORD_RING_CACHE[p][ci]


# ---------------------------------------------------------------------------
#  Reduction (optional, to keep denom_pow3 small)
# ---------------------------------------------------------------------------

def ringmat_reduce_by_three(M: RingMat) -> RingMat:
    """If every entry's numerator coefficients are divisible by 3, divide them
    out and decrement each entry's denom_pow3 by 1. Iterate until impossible.

    This is the cheap variant of the GCD-by-3 reduction. It does NOT change
    the matrix value; it only keeps the integer coefficients (and the f) small.
    """
    while True:
        # Inspect every coefficient of every entry.
        all_divisible = True
        for i in range(3):
            for j in range(3):
                entry = M[i][j]
                if entry.denom_pow3 == 0:
                    # Can't decrement f below 0 without changing the value.
                    all_divisible = False
                    break
                if any(c % 3 != 0 for c in entry.num.coefs):
                    all_divisible = False
                    break
            if not all_divisible:
                break
        if not all_divisible:
            return M
        # Divide every entry's numerator by 3 and drop denom_pow3 by 1.
        new_M: RingMat = []
        for i in range(3):
            row: list[Z9Frac] = []
            for j in range(3):
                entry = M[i][j]
                new_coefs = tuple(c // 3 for c in entry.num.coefs)
                row.append(Z9Frac(Z9(new_coefs), entry.denom_pow3 - 1))
            new_M.append(row)
        M = new_M
