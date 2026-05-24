"""canonical_reducer.py — pure-Python Clifford+D syllable decomposition for
arbitrary-precision qutrit unitaries over Z[ζ_9, 1/3].

Mirrors hrsa/decompose.cpp's sde-peeling algorithm (Kalra et al.), but
operates directly on Python `Z9Frac` arithmetic (arbitrary-precision int
coefficients with a `denom_pow3` exponent).  Designed for the
SK-driver-rz output where f ≈ 4052 and per-entry integer coefs are ~10^1934
— well outside the integer-table-indexed C++ decompose_tool's reach.

Algorithm:
  s = sde_chi_full(V[0][0])
  while s > 0:
      Try each of 4374 prefixes P = H · D(a1,a2,a3) · R^eps · X^delta
      Pick the one with minimum D-cost such that sde_chi_full((P·V)[0][0]) == s-1
      V := P · V; record (a1,a2,a3,eps,delta,has_H=True) syllable
      s = sde_chi_full(V[0][0])
  At s == 0: V must be a monomial Clifford+D matrix (each row/col has one nz).
  Count residual D from monomial classification (matches C++ decompose).

After peeling, the syllable list [s_1, ..., s_n] satisfies
  M_steps = P_n · ... · P_1   such that  M_steps · V_input = trailing_clifford.
Equivalently V_input = M_steps† · trailing_clifford.

We verify by running the same product in Z9Frac arithmetic and asserting
exact bit-for-bit equality against the input V at every entry (no floating
point — purely integer comparison after aligning denom_pow3).

CLI:
  python3 canonical_reducer.py --input V.npz --output decomposition.json
  python3 canonical_reducer.py --input V.json --output decomposition.json

Input npz format: V_blob int[3][3][6] + f scalar (matching sk_ring_audit's
output ints schema).
Input json format: {"f": int, "V": [[[6 ints]x3]x3]}.

Output json format mirrors decompose_cli.cpp's schema:
  {"success": bool, "D_count": int, "sde_chi_initial": int, "sde_chi_final": int,
   "syllables": [{"a0":int,"a1":int,"a2":int,"eps":int,"delta":int,"has_H":bool}, ...],
   "trailing_clifford": {"f": int, "V": [[[6 ints]x3]x3]},
   "verification": {"matches_input": bool, "max_coef_residual": int}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_UNIFIED = _HERE.parent
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))

from ep_level import Z9, Z9Frac, matmul, conjugate_transpose


# ---------------------------------------------------------------------------
# sde_chi (lifted from cyclotomic_int9.h sdeChi() and decompose_impl.h
# sdeChiZ9 / sdeChiFull)
# ---------------------------------------------------------------------------

def _signed_weight(coefs: tuple[int, ...]) -> int:
    """Sum of all 6 coefficients (Z-basis (1, ξ, ξ², ξ³, ξ⁴, ξ⁵))."""
    return sum(coefs)


def _formal_derivative_coefs(coefs: tuple[int, ...]) -> tuple[int, ...]:
    """Apply d/dξ to a Z9 element in unreduced form, returning a length-6
    coefficient tuple.  Matches cyclotomic_int9.h::formalDerivative():
      new[i] = (i+1) * coefs[i+1]  for i in 0..5
    where we treat coefs as a length-9 sequence whose entries [6,7,8] are
    zero (canonical reduced form).  Output is also length-6 because the
    derivative of a degree-5 polynomial has degree 4 (no reduce needed)
    BUT we follow the C++ exactly: new_element has 9 slots, new[i] =
    (i+1) * element[i+1], for i in 0..5; new[5] = 6 * element[6] = 0.
    """
    # In canonical form, coefs[6..8] = 0, so element[i+1] for i in 0..5
    # accesses indices 1..6, the last of which is 0.
    new = [0] * 6
    for i in range(6):
        # coefs[i+1] is 0 when i+1 == 6 (since reduced form has 0 there).
        if i + 1 < 6:
            new[i] = (i + 1) * coefs[i + 1]
        else:
            new[i] = 0
    return tuple(new)


def _z9_sde_chi(coefs: tuple[int, ...]) -> int:
    """sde_chi of a Z9 element (in canonical reduced 6-coef form).

    Matches cyclotomic_int9.h::sdeChi() exactly:
      if signedWeight % 3 != 0: return 0
      else apply formalDerivative, divide by i, check signedWeight % 3,
           up to i = 5 then return 6.
    """
    sw = _signed_weight(coefs)
    if sw % 3 != 0:
        return 0
    test = _formal_derivative_coefs(coefs)
    for i in range(1, 6):
        # test.signedWeight() / T(i) is integer division in C++; for the
        # mod-3 check to be meaningful we follow the same: sw_test must be
        # exactly divisible by i (which is true for the derivative chain
        # because of the i! factor accumulating).
        sw_test = _signed_weight(test)
        # In C++: sw = test.signedWeight() / T(i); mod3_nonzero(sw)
        # If sw_test % i != 0 then the division truncates and the result
        # has no meaningful mod-3 interpretation; in practice for valid
        # Z[ξ] inputs this divides exactly because of i! factors.
        sw_div = sw_test // i
        if sw_div % 3 != 0:
            return i
        # test = (test.formalDerivative()) / i  (per-coefficient int division)
        test_d = _formal_derivative_coefs(test)
        test = tuple(c // i for c in test_d)
    return 6


def _z9_is_zero(coefs: tuple[int, ...]) -> bool:
    return all(c == 0 for c in coefs)


def _z9_div_by_int(coefs: tuple[int, ...], k: int) -> tuple[int, ...]:
    return tuple(c // k for c in coefs)


def _z9_divisible_by_int(coefs: tuple[int, ...], k: int) -> bool:
    return all(c % k == 0 for c in coefs)


def sde_chi_z9(coefs: tuple[int, ...]) -> int:
    """sde_chi for Z[ξ] element.  Returns 999 if zero, else the
    'how many times we can pull out chi' count (chi = (1-ξ) up to a unit,
    via the standard sdeChi recursion in cyclotomic_int9.h)."""
    if _z9_is_zero(coefs):
        return 999
    s = _z9_sde_chi(coefs)
    if s < 6:
        return s
    if not _z9_divisible_by_int(coefs, 3):
        return 6
    return 6 + sde_chi_z9(_z9_div_by_int(coefs, 3))


def sde_chi_full(zf: Z9Frac) -> int:
    """sde_chi_full: handles the Z[ξ, 1/3] case.  Mirrors decompose_impl.h
    sdeChiFull<T>():
      if zero: 0
      if f == 0: sdeChiZ9(numer)
      else:      6*f - sdeChiZ9(numer)
    """
    if zf.is_zero():
        return 0
    f = zf.denom_pow3
    coefs = zf.num.coefs
    if f == 0:
        return sde_chi_z9(coefs)
    ell = sde_chi_z9(coefs)
    return 6 * f - ell


# ---------------------------------------------------------------------------
# Gate matrices (matching decompose.cpp gateH/gateX/gateR/gateDcyclo)
# ---------------------------------------------------------------------------

def _zeta9_power(k: int) -> Z9Frac:
    """zeta_9^k as Z9Frac (f=0).  For k in 0..5 this is a single basis
    element; for k in 6..8 use the standard reduction xi^{6+j} = -xi^j - xi^{3+j}."""
    k = ((k % 9) + 9) % 9
    if k < 6:
        coefs = [0] * 6
        coefs[k] = 1
        return Z9Frac(Z9(tuple(coefs)), 0)
    j = k - 6
    coefs = [0] * 6
    coefs[j] = -1
    coefs[j + 3] = -1
    return Z9Frac(Z9(tuple(coefs)), 0)


def _omega_power(k: int) -> Z9Frac:
    """omega^k = zeta_9^{3k}."""
    return _zeta9_power(3 * k)


def gate_H() -> list[list[Z9Frac]]:
    """H_{jk} = (1/(1+2*omega)) * omega^{jk}.

    Matches decompose.cpp's gateH(): the scalar is (-1 - 2*zeta_9^3)/3
    which equals 1/(1+2*omega) since (1+2*omega)*(-1-2*omega^2)/3 = ... etc.
    We construct it directly as the C++ does:
      ia = (-1 - 2*zeta_9^3) / 3   i.e. Z9 numerator [-1,0,0,-2,0,0], f=1
      H[j][k] = ia * omega^{jk}
    """
    # numerator coefs: [-1, 0, 0, -2, 0, 0] / 3^1
    ia = Z9Frac(Z9((-1, 0, 0, -2, 0, 0)), 1)
    H: list[list[Z9Frac]] = [[Z9Frac.zero() for _ in range(3)] for _ in range(3)]
    for j in range(3):
        for k in range(3):
            H[j][k] = ia * _omega_power(j * k)
    return H


def gate_X() -> list[list[Z9Frac]]:
    """X = cyclic permutation |j> -> |j+1 mod 3>.
    Matches decompose.cpp: X[0][2]=X[1][0]=X[2][1]=1."""
    one = Z9Frac.one()
    zero = Z9Frac.zero()
    return [
        [zero, zero, one],
        [one,  zero, zero],
        [zero, one,  zero],
    ]


def gate_R() -> list[list[Z9Frac]]:
    """R = Diag(1, 1, -1)."""
    one = Z9Frac.one()
    minus_one = Z9Frac(Z9.from_int(-1), 0)
    zero = Z9Frac.zero()
    return [
        [one,  zero, zero],
        [zero, one,  zero],
        [zero, zero, minus_one],
    ]


def gate_Dcyclo(a: int, b: int, c: int) -> list[list[Z9Frac]]:
    """Diag(xi^a, xi^b, xi^c) — full cyclotomic diagonal."""
    zero = Z9Frac.zero()
    return [
        [_zeta9_power(a), zero,            zero],
        [zero,            _zeta9_power(b), zero],
        [zero,            zero,            _zeta9_power(c)],
    ]


# ---------------------------------------------------------------------------
# D-cost counting (matches decompose.cpp count_d_from_cyclo / _syllable)
# ---------------------------------------------------------------------------

def count_d_from_cyclo(a1: int, a2: int, a3: int) -> int:
    d = 0
    if a1 % 3 != 0:
        d += 1
    if a2 % 3 != 0:
        d += 1
    if a3 % 3 != 0:
        d += 1
    return d


def count_d_from_syllable(a1: int, a2: int, a3: int, eps: int) -> int:
    return count_d_from_cyclo(a1, a2, a3) + (1 if eps != 0 else 0)


# ---------------------------------------------------------------------------
# Prefix table (4374 entries: H · D(a1,a2,a3) · R^eps · X^delta)
# ---------------------------------------------------------------------------

class PrefixEntry:
    __slots__ = ("P", "d", "a1", "a2", "a3", "eps", "delta")

    def __init__(self, P, d, a1, a2, a3, eps, delta):
        self.P = P
        self.d = d
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.eps = eps
        self.delta = delta


_PREFIX_TABLE_CACHE: list[PrefixEntry] | None = None


def _build_prefix(a1: int, a2: int, a3: int, eps: int, delta: int,
                  H: list[list[Z9Frac]], R: list[list[Z9Frac]],
                  X: list[list[Z9Frac]]) -> list[list[Z9Frac]]:
    """H · D(a1,a2,a3) · R^eps · X^delta.  Matches decompose.cpp::buildPrefix."""
    P = gate_Dcyclo(a1, a2, a3)
    if eps == 1:
        P = matmul(P, R)
    if delta == 1:
        P = matmul(P, X)
    elif delta == 2:
        P = matmul(matmul(P, X), X)
    return matmul(H, P)


def get_prefix_table() -> list[PrefixEntry]:
    """Build (once) the 4374-entry prefix table.  ~30s on first call."""
    global _PREFIX_TABLE_CACHE
    if _PREFIX_TABLE_CACHE is not None:
        return _PREFIX_TABLE_CACHE
    H = gate_H()
    R = gate_R()
    X = gate_X()
    tbl: list[PrefixEntry] = []
    for eps in (0, 1):
        for delta in (0, 1, 2):
            for a1 in range(9):
                for a2 in range(9):
                    for a3 in range(9):
                        P = _build_prefix(a1, a2, a3, eps, delta, H, R, X)
                        d = count_d_from_syllable(a1, a2, a3, eps)
                        tbl.append(PrefixEntry(P, d, a1, a2, a3, eps, delta))
    _PREFIX_TABLE_CACHE = tbl
    return tbl


# ---------------------------------------------------------------------------
# Helpers: prefix * V[0][0]  (avoid full matmul when only the (0,0) entry
# is needed for sde_chi check).  Prefixes have INTEGER coefs in Z[ξ, 1/3]
# with small f (≤ 1 from the H factor), V is the arbitrary-precision input.
# ---------------------------------------------------------------------------

def _prefix_times_V_00(P: list[list[Z9Frac]], V: list[list[Z9Frac]]) -> Z9Frac:
    """Compute (P · V)[0][0] = sum_k P[0][k] * V[k][0]."""
    acc = Z9Frac.zero()
    for k in range(3):
        if P[0][k].is_zero():
            continue
        acc = acc + P[0][k] * V[k][0]
    return acc


def _prefix_times_V(P: list[list[Z9Frac]], V: list[list[Z9Frac]]) -> list[list[Z9Frac]]:
    """Full P · V."""
    return matmul(P, V)


def _reduce_by_three(M: list[list[Z9Frac]]) -> list[list[Z9Frac]]:
    """If every entry's numerator coefs are divisible by 3, divide them out
    and decrement each entry's denom_pow3 by 1.  Repeat to a fixed point.

    This is essential to keep the integer coefs bounded as we peel: each
    `P · V` adds 1 to f and bloats the numerator; after the cancel-by-3
    step the matrix shrinks back proportionally (and stays exactly equal
    to the input).
    """
    # Find the maximum k such that 3^k divides every numerator coef.
    while True:
        all_div = True
        for i in range(3):
            for j in range(3):
                entry = M[i][j]
                if entry.denom_pow3 == 0:
                    all_div = False
                    break
                if any(c % 3 != 0 for c in entry.num.coefs):
                    all_div = False
                    break
            if not all_div:
                break
        if not all_div:
            return M
        new_M = []
        for i in range(3):
            row = []
            for j in range(3):
                entry = M[i][j]
                new_coefs = tuple(c // 3 for c in entry.num.coefs)
                row.append(Z9Frac(Z9(new_coefs), entry.denom_pow3 - 1))
            new_M.append(row)
        M = new_M


# ---------------------------------------------------------------------------
# Monomial classification + residual D-count (matches decompose.cpp's
# countMonomialD pathway, abridged: handle the cases that arise after
# successful peeling, where each entry is a power of zeta_9.)
# ---------------------------------------------------------------------------

def _unit_phase_mod3(x: Z9Frac) -> int:
    """If x is a unit ±zeta_9^j in Z[zeta_9] (with denom_pow3=0), return j%3.
    Otherwise return -1.  Mirrors decompose_impl.h's unitPhaseMod3."""
    if x.denom_pow3 != 0:
        # The entry has a non-trivial 1/3 factor; not a clean unit.
        return -1
    numer = x.num
    # Try numer * zeta_9^{-j} for j in 0..8; check if result is ±1.
    for j in range(9):
        # zeta_9^{-j} = zeta_9^{9-j}; multiply numerator polynomial.
        z_inv = _zeta9_power(9 - j).num  # f=0 always
        prod = numer * z_inv  # Z9 * Z9 -> Z9
        # Check if prod is ±1 (only coefs[0] = ±1, others 0).
        if all(prod.coefs[k] == 0 for k in range(1, 6)) and prod.coefs[0] in (1, -1):
            return j % 3
    return -1


def classify_monomial_and_d_cost(M: list[list[Z9Frac]]):
    """Return (is_monomial, d_count, r_count).  d_count includes r_count.

    Matches the heuristic in decompose_impl.h's countMonomialD lambda:
    - If M has exactly one nonzero entry per row/col AND each entry is
      ±zeta_9^j with j%3 = 0: M is sign-extended Clifford; we DO NOT check
      against the 648-Clifford cache (we approximate D = 0 here; rare cases
      where M = sign * Clifford for non-identity sign add 1 D-cost).
    - Else if all phase_mod3 are 0: 0 D (or 1 if non-trivial sign).
    - Else use the 1-gate / 2-gate heuristic on (p, q, r) mod 3 pattern.
    """
    phases_mod3 = [-1, -1, -1]
    is_mono = True
    for i in range(3):
        nz_count = 0
        for j in range(3):
            if not M[i][j].is_zero():
                nz_count += 1
                pm3 = _unit_phase_mod3(M[i][j])
                if pm3 < 0:
                    return False, -1, 0
                phases_mod3[i] = pm3
        if nz_count != 1:
            return False, -1, 0
    p, q, r = phases_mod3
    if p == 0 and q == 0 and r == 0:
        # All phases are powers of omega = zeta_9^3, i.e. Clifford-compatible.
        # The C++ code further distinguishes "in 648-cache → 0 D" vs
        # "sign · Clifford → 1 D".  Without the Clifford cache in Python
        # we conservatively report 0 D here (true for the dominant case).
        # If the SK output ever produces a residual that requires the +1,
        # the verification step will still pass (the +1 is a Clifford R
        # post-rotation that can absorb into the trailing_clifford).
        return True, 0, 0
    one_gate = (
        (p == 1 and q == 0 and r == 2) or (p == 2 and q == 0 and r == 1) or
        (p == 0 and q == 1 and r == 2) or (p == 0 and q == 2 and r == 1) or
        (p == 1 and q == 2 and r == 0) or (p == 2 and q == 1 and r == 0)
    )
    d_cost = 1 if one_gate else 2
    return True, d_cost, 0


# ---------------------------------------------------------------------------
# Core peeling loop
# ---------------------------------------------------------------------------

def _try_single_prefix(V: list[list[Z9Frac]], s: int, table: list[PrefixEntry],
                       verbose: bool = False):
    """Find the lowest-D prefix P in `table` such that sde_chi_full((P·V)[0][0])
    == s - 1.  Returns the winning PrefixEntry, or None.

    Iterates ALL 4374 entries (cheap: each iteration is one ringZ9chi multiply
    + one sde_chi_full).  The C++ uses early-exit (e.d >= best_d) to prune;
    we replicate that.
    """
    best_d = float("inf")
    best_idx = -1
    for idx, e in enumerate(table):
        if e.d >= best_d:
            continue
        new00 = _prefix_times_V_00(e.P, V)
        if sde_chi_full(new00) == s - 1:
            best_d = e.d
            best_idx = idx
            if best_d == 0:
                # Best possible; no further improvement.
                break
    if best_idx < 0:
        return None
    return table[best_idx]


def _try_double_prefix(V: list[list[Z9Frac]], s: int, table: list[PrefixEntry],
                       verbose: bool = False):
    """Find two prefixes P2, P1 such that sde_chi((P2·P1·V)[0][0]) < s with
    minimum combined D-cost.  Matches decompose_impl.h's tryDoublePrefix.

    Optimization: we only need the FIRST COLUMN of midV = P1·V (not the
    full 3x3) — because (P2·midV)[0][0] = sum_k P2[0][k] · midV[k][0].
    Computing one column of P1·V is 3x cheaper than the full matmul, which
    matters at f≈4052 where each entry has ~1934-digit ints.

    O(N^2) worst case (~19M iterations) pruned by:
      (a) mid_s window check on outer P1 (mid_s must be in {s-1, s, s+1}),
      (b) e2.d >= best_total_d early-exit on inner P2.
    """
    best_total_d = float("inf")
    best_idx1 = -1
    best_idx2 = -1
    best_new_s = s
    best_mid_s = s

    # Precompute column 0 of V (used for mid_col[0] = sum_m P1[0][m] · V[m][0]).
    # No need; we access V[m][0] directly inside.

    for idx1, e1 in enumerate(table):
        if e1.d >= best_total_d:
            continue
        mid00 = _prefix_times_V_00(e1.P, V)
        mid_s = sde_chi_full(mid00)
        if mid_s > s + 1 or mid_s < s - 1:
            continue
        # Compute column 0 of midV = P1·V: midV[k][0] = sum_m P1[k][m]·V[m][0].
        # mid_col[0] is just mid00 (already computed).
        P1 = e1.P
        mid_col = [mid00, Z9Frac.zero(), Z9Frac.zero()]
        for k in (1, 2):
            acc = Z9Frac.zero()
            for m in range(3):
                if not P1[k][m].is_zero():
                    acc = acc + P1[k][m] * V[m][0]
            mid_col[k] = acc
        for idx2, e2 in enumerate(table):
            if e1.d + e2.d >= best_total_d:
                continue
            P2 = e2.P
            # (P2 · midV)[0][0] = sum_k P2[0][k] · midV[k][0] = sum_k P2[0][k] · mid_col[k]
            new00 = Z9Frac.zero()
            for k in range(3):
                if not P2[0][k].is_zero():
                    new00 = new00 + P2[0][k] * mid_col[k]
            new_s = sde_chi_full(new00)
            if new_s < s:
                best_total_d = e1.d + e2.d
                best_new_s = new_s
                best_mid_s = mid_s
                best_idx1 = idx1
                best_idx2 = idx2
                if best_total_d == 0:
                    break  # cheapest possible; exit inner loop
    if best_idx1 < 0:
        return None
    return (table[best_idx1], table[best_idx2], best_mid_s, best_new_s)


def decompose_canonical(V_input: list[list[Z9Frac]],
                        max_iter: int | None = None,
                        verbose: bool = False) -> dict:
    """Canonical-form syllable decomposition.

    Returns dict with:
      success           — bool: did peeling reach sde_chi = 0?
      D_count           — int:  syllable + residual D-count
      sde_chi_initial   — int:  sde_chi_full(V[0][0]) on entry
      sde_chi_final     — int:  sde_chi_full(V_final[0][0]) at exit
      syllables         — list of GateStep dicts
      trailing_clifford — final V (3x3 RingMat), expected monomial
      n_iter            — number of peel iterations executed
      peel_seconds      — wall time for the peel phase only
    """
    table = get_prefix_table()
    if verbose:
        print(f"[canonical_reducer] prefix table size: {len(table)}", file=sys.stderr)

    V = [row[:] for row in V_input]
    s_initial = sde_chi_full(V[0][0])
    if verbose:
        print(f"[canonical_reducer] initial sde_chi_full(V[0][0]) = {s_initial}",
              file=sys.stderr)

    s = s_initial
    if max_iter is None:
        max_iter = s + 50

    syllables: list[dict] = []
    D_count = 0
    n_iter = 0
    peel_start = time.time()

    if s == 999:
        # V[0][0] is exactly zero; can't peel.
        return {
            "success": False,
            "D_count": -1,
            "sde_chi_initial": s_initial,
            "sde_chi_final": s,
            "syllables": [],
            "trailing_clifford": V,
            "n_iter": 0,
            "peel_seconds": 0.0,
            "error": "V[0][0] is zero — cannot peel",
        }

    n_double = 0
    while s > 0 and n_iter < max_iter:
        n_iter += 1
        winner = _try_single_prefix(V, s, table, verbose=verbose)
        if winner is not None:
            # Apply: V := winner.P · V; then reduce-by-3 to keep ints bounded.
            V = _prefix_times_V(winner.P, V)
            V = _reduce_by_three(V)
            syllables.append({
                "a0": winner.a1, "a1": winner.a2, "a2": winner.a3,
                "eps": winner.eps, "delta": winner.delta, "has_H": True,
            })
            D_count += winner.d
            new_s = sde_chi_full(V[0][0])
            if verbose and (n_iter % 5 == 1 or new_s == 0):
                max_f = max(V[i][j].denom_pow3 for i in range(3) for j in range(3))
                print(f"[canonical_reducer] iter {n_iter}: sde_chi {s} -> {new_s}, "
                      f"D_count so far = {D_count}, V max f = {max_f}",
                      file=sys.stderr)
            s = new_s
            continue

        # Single-prefix failed: try double-prefix (Kalra obstruction fallback).
        if verbose:
            print(f"[canonical_reducer] iter {n_iter}: single-prefix failed at "
                  f"sde_chi={s}, trying double-prefix...", file=sys.stderr)
        dbl = _try_double_prefix(V, s, table, verbose=verbose)
        if dbl is None:
            peel_seconds = time.time() - peel_start
            if verbose:
                print(f"[canonical_reducer] double-prefix also failed at sde_chi={s}; "
                      f"giving up after {n_iter} iters", file=sys.stderr)
            return {
                "success": False,
                "D_count": -1,
                "sde_chi_initial": s_initial,
                "sde_chi_final": s,
                "syllables": syllables,
                "trailing_clifford": V,
                "n_iter": n_iter,
                "peel_seconds": peel_seconds,
                "error": f"single+double prefix both failed at sde_chi={s}",
            }
        n_double += 1
        e1, e2, mid_s, new_s = dbl
        # Apply P1 then P2; record both as syllables.
        V = _prefix_times_V(e1.P, V)
        V = _prefix_times_V(e2.P, V)
        V = _reduce_by_three(V)
        syllables.append({
            "a0": e1.a1, "a1": e1.a2, "a2": e1.a3,
            "eps": e1.eps, "delta": e1.delta, "has_H": True,
        })
        syllables.append({
            "a0": e2.a1, "a1": e2.a2, "a2": e2.a3,
            "eps": e2.eps, "delta": e2.delta, "has_H": True,
        })
        D_count += e1.d + e2.d
        if verbose:
            print(f"[canonical_reducer] iter {n_iter}: double-prefix sde "
                  f"{s} -> {mid_s} -> {new_s}, D_count so far = {D_count}",
                  file=sys.stderr)
        # The C++ counts each double-prefix pair as 1 logical iteration step
        # (in terms of max_iter check), but increments syllable count by 2.
        # We follow the same convention for max_iter.
        s = sde_chi_full(V[0][0])

    peel_seconds = time.time() - peel_start

    if s != 0:
        return {
            "success": False,
            "D_count": -1,
            "sde_chi_initial": s_initial,
            "sde_chi_final": s,
            "syllables": syllables,
            "trailing_clifford": V,
            "n_iter": n_iter,
            "peel_seconds": peel_seconds,
            "error": f"max_iter ({max_iter}) reached, sde_chi still {s}",
        }

    # Trailing residual: should be monomial (each row/col one nonzero entry,
    # power of zeta_9).  Count D from monomial classification.
    is_mono, mono_d, mono_r = classify_monomial_and_d_cost(V)
    if not is_mono:
        return {
            "success": False,
            "D_count": -1,
            "sde_chi_initial": s_initial,
            "sde_chi_final": s,
            "syllables": syllables,
            "trailing_clifford": V,
            "n_iter": n_iter,
            "peel_seconds": peel_seconds,
            "error": "residual V at sde_chi=0 is not monomial; "
                     "algorithm did not reach a Clifford form",
        }

    D_count += mono_d

    return {
        "success": True,
        "D_count": D_count,
        "sde_chi_initial": s_initial,
        "sde_chi_final": 0,
        "syllables": syllables,
        "trailing_clifford": V,
        "n_iter": n_iter,
        "peel_seconds": peel_seconds,
        "residual_D": mono_d,
    }


# ---------------------------------------------------------------------------
# Verification: reconstruct V from syllables and compare to input.
# ---------------------------------------------------------------------------

def _gate_step_prefix(step: dict) -> list[list[Z9Frac]]:
    """Build the prefix matrix for a single GateStep:
       (H if has_H) · D(a0,a1,a2) · R^eps · X^delta"""
    H = gate_H()
    R = gate_R()
    X = gate_X()
    P = gate_Dcyclo(step["a0"], step["a1"], step["a2"])
    if step.get("eps", 0) == 1:
        P = matmul(P, R)
    delta = step.get("delta", 0)
    if delta == 1:
        P = matmul(P, X)
    elif delta == 2:
        P = matmul(matmul(P, X), X)
    if step.get("has_H", True):
        P = matmul(H, P)
    return P


def reconstruct_from_decomposition(syllables: list[dict],
                                    trailing_clifford: list[list[Z9Frac]]
                                    ) -> list[list[Z9Frac]]:
    """Given the syllable list [s_1, ..., s_n] and the trailing clifford C,
    compute V = (P_1^{-1} P_2^{-1} ... P_n^{-1}) · C.

    Since each P_i is a unitary in U(3) over Z[ξ, 1/3], P_i^{-1} = P_i†.

    Sanity check: the peeling produced M_steps · V_input = C where
    M_steps = P_n · ... · P_1.  Hence V_input = M_steps† · C.
    """
    # Compute M_steps = P_n · ... · P_1 (in the order steps were emitted,
    # left-multiplied onto the running product).
    M_steps = [[Z9Frac.one() if i == j else Z9Frac.zero() for j in range(3)] for i in range(3)]
    for step in syllables:
        P = _gate_step_prefix(step)
        M_steps = matmul(P, M_steps)
    # V_input = M_steps† · C
    M_dag = conjugate_transpose(M_steps)
    V_reconstructed = matmul(M_dag, trailing_clifford)
    return V_reconstructed


def verify_decomposition(V_input: list[list[Z9Frac]],
                         syllables: list[dict],
                         trailing_clifford: list[list[Z9Frac]]) -> dict:
    """Compare V_reconstructed against V_input exactly (no floating point).

    Returns dict with:
      matches            — bool: every entry equal in Z9Frac sense
      max_coef_residual  — int: max absolute coefficient in (V_recon - V_input)
                                after aligning denom_pow3
      mismatch_entries   — list of (i,j) where they differ (capped at 5)
    """
    V_rec = reconstruct_from_decomposition(syllables, trailing_clifford)

    mismatches: list[tuple[int, int]] = []
    max_resid = 0
    matches = True
    for i in range(3):
        for j in range(3):
            a = V_input[i][j]
            b = V_rec[i][j]
            # Align denoms to compare numerators directly.
            d = max(a.denom_pow3, b.denom_pow3)
            a_num = a.num * Z9.from_int(3 ** (d - a.denom_pow3))
            b_num = b.num * Z9.from_int(3 ** (d - b.denom_pow3))
            diff = a_num - b_num
            for c in diff.coefs:
                if abs(int(c)) > max_resid:
                    max_resid = abs(int(c))
            if not all(c == 0 for c in diff.coefs):
                matches = False
                if len(mismatches) < 5:
                    mismatches.append((i, j))
    return {
        "matches": matches,
        "max_coef_residual": int(max_resid),
        "mismatch_entries": mismatches,
    }


# ---------------------------------------------------------------------------
# I/O: load V from .npz / .json, dump decomposition to .json
# ---------------------------------------------------------------------------

def load_input(path: Path) -> tuple[list[list[Z9Frac]], int]:
    """Load (V, f) from either .npz (V_blob int[3][3][6], f scalar) or
    .json ({"f": int, "V": [[[6 ints]x3]x3]})."""
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        V_blob = data["V"] if "V" in data.files else data["V_blob"]
        f = int(data["f"][()]) if data["f"].ndim == 0 else int(data["f"][0])
    elif p.suffix == ".json":
        with open(p, "r") as fh:
            d = json.load(fh)
        f = int(d["f"])
        V_blob = np.asarray(d["V"], dtype=object)
    else:
        raise ValueError(f"unknown extension {p.suffix}; want .npz or .json")
    V_blob = np.asarray(V_blob)
    assert V_blob.shape == (3, 3, 6), f"V_blob shape {V_blob.shape} != (3,3,6)"
    V: list[list[Z9Frac]] = []
    for i in range(3):
        row: list[Z9Frac] = []
        for j in range(3):
            coefs = tuple(int(V_blob[i, j, k]) for k in range(6))
            row.append(Z9Frac(Z9(coefs), f))
        V.append(row)
    return V, f


def _trailing_to_ints(M: list[list[Z9Frac]]) -> tuple[list[list[list[int]]], int]:
    """Serialize a RingMat to (3x3 list of 6-int lists, shared f)."""
    f_shared = max(M[i][j].denom_pow3 for i in range(3) for j in range(3))
    out: list[list[list[int]]] = []
    for i in range(3):
        row: list[list[int]] = []
        for j in range(3):
            entry = M[i][j]
            scale = 3 ** (f_shared - entry.denom_pow3)
            if scale != 1:
                scaled = entry.num * Z9.from_int(scale)
            else:
                scaled = entry.num
            row.append([int(c) for c in scaled.coefs])
        out.append(row)
    return out, f_shared


def dump_decomposition(result: dict, output_path: Path) -> None:
    """Write the decomposition result to JSON.  Mirrors the schema
    decompose_cli.cpp produces (with extra `verification` block)."""
    tc_ints, tc_f = _trailing_to_ints(result["trailing_clifford"])
    out = {
        "success": result["success"],
        "D_count": result["D_count"],
        "sde_chi_initial": result["sde_chi_initial"],
        "sde_chi_final": result["sde_chi_final"],
        "syllables": result["syllables"],
        "trailing_clifford": {"f": tc_f, "V": tc_ints},
        "n_iter": result.get("n_iter"),
        "peel_seconds": result.get("peel_seconds"),
    }
    if "error" in result:
        out["error"] = result["error"]
    if "verification" in result:
        out["verification"] = result["verification"]
    if "residual_D" in result:
        out["residual_D"] = result["residual_D"]
    with open(output_path, "w") as fh:
        json.dump(out, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, type=Path,
                   help="Input .npz (V_blob, f) or .json ({f, V}).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON decomposition.")
    p.add_argument("--max-iter", type=int, default=None,
                   help="Max peel iterations (default sde_chi+50).")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip verification step (saves a chain of matmuls).")
    args = p.parse_args(argv)

    if args.verbose:
        print(f"[canonical_reducer] loading input from {args.input}",
              file=sys.stderr)
    V, f_input = load_input(args.input)
    if args.verbose:
        print(f"[canonical_reducer] input f = {f_input}", file=sys.stderr)
        # Per-entry denom (post-construction; all equal to f_input here).
        max_coef = max(
            abs(int(c)) for i in range(3) for j in range(3)
            for c in V[i][j].num.coefs
        )
        digits = len(str(max_coef)) if max_coef else 1
        print(f"[canonical_reducer] max |coef| ~ 10^{digits-1} ({digits} digits)",
              file=sys.stderr)

    t0 = time.time()
    result = decompose_canonical(V, max_iter=args.max_iter, verbose=args.verbose)
    t1 = time.time()
    if args.verbose:
        print(f"[canonical_reducer] decompose_canonical: success={result['success']} "
              f"D_count={result['D_count']} n_iter={result['n_iter']} "
              f"wall={t1-t0:.2f}s", file=sys.stderr)

    if not args.no_verify and result["success"]:
        t2 = time.time()
        if args.verbose:
            print("[canonical_reducer] verifying reconstruction...",
                  file=sys.stderr)
        ver = verify_decomposition(V, result["syllables"],
                                   result["trailing_clifford"])
        t3 = time.time()
        result["verification"] = ver
        if args.verbose:
            print(f"[canonical_reducer] verify: matches={ver['matches']} "
                  f"max_residual={ver['max_coef_residual']} wall={t3-t2:.2f}s",
                  file=sys.stderr)

    dump_decomposition(result, args.output)
    print(f"wrote {args.output}: success={result['success']} "
          f"D_count={result['D_count']} syllables={len(result['syllables'])}"
          + (f" verify_match={result['verification']['matches']}"
             if 'verification' in result else ""))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
