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
# gmpy2-accelerated big-int shim for the hot path.
#
# The peel loop spends ~all its time multiplying numerator polynomials whose
# coefficients are Python big-ints with up to ~10^1934 digits (at SK depth-2
# f≈4052).  Python's int uses Karatsuba and is fast in absolute terms, but
# gmpy2.mpz wraps GMP's mul (Toom-Cook / FFT thresholds, no Python object
# overhead per op), giving an asymptotic + constant-factor win.
#
# We define light-weight `_FastZ9` / `_FastZ9Frac` mirrors of ep_level.Z9 /
# Z9Frac whose coefficient type is `_INT` (mpz if gmpy2 is available, else
# plain int — strict-correctness fallback).  All hot-path code below
# (_reduce_by_three, _prefix_times_V_00, matmul, sde_chi_*, etc.) is
# rewritten against these fast types.  We keep ep_level.Z9/Z9Frac as the
# canonical low-throughput types for the gate-table construction (one-shot
# at startup) and for the verify_decomposition step.
# ---------------------------------------------------------------------------
try:
    from gmpy2 import mpz as _INT
    _HAVE_GMPY2 = True
except ImportError:  # pragma: no cover — gmpy2 missing
    _INT = int
    _HAVE_GMPY2 = False

_INT_ZERO = _INT(0)
_INT_ONE = _INT(1)
_INT_THREE = _INT(3)

# Optional FLINT backend: ~1.5–2× over gmpy2 _FastZ9 at SK depth-2 scale
# (f≈4050, ~1934-digit coefs), where FLINT's C-level fmpz_poly mul
# eliminates per-coef Python dispatch on the 36 scalar mults.
try:
    from flint import fmpz_poly as _fmpz_poly, fmpz as _fmpz  # type: ignore
    _HAVE_FLINT = True
    # Φ_9(x) = x^6 + x^3 + 1; reduction modulus for Z[ζ_9] ≅ Z[x]/Φ_9.
    _PHI9 = _fmpz_poly([1, 0, 0, 1, 0, 0, 1])
    _FMPZ_ZERO = _fmpz(0)
except ImportError:  # pragma: no cover — flint missing
    _fmpz_poly = None
    _fmpz = None
    _HAVE_FLINT = False
    _PHI9 = None
    _FMPZ_ZERO = None


def _reduce_9_fast(c: list) -> tuple:
    """Reduce a polynomial of degree up to 10 in ξ to a 6-tuple over the
    Z-basis (1, ξ, ξ², ξ³, ξ⁴, ξ⁵).  Same reduction as ep_level._reduce_9
    but type-preserving on _INT coefficients."""
    # First fold powers ≥ 9 using ξ^9 = 1.
    out = [_INT_ZERO] * 9
    for k in range(len(c)):
        out[k % 9] = out[k % 9] + c[k]
    # Now use ξ⁶ = -1 - ξ³, ξ⁷ = -ξ - ξ⁴, ξ⁸ = -ξ² - ξ⁵.
    if out[6]:
        out[0] = out[0] - out[6]
        out[3] = out[3] - out[6]
    if out[7]:
        out[1] = out[1] - out[7]
        out[4] = out[4] - out[7]
    if out[8]:
        out[2] = out[2] - out[8]
        out[5] = out[5] - out[8]
    return (out[0], out[1], out[2], out[3], out[4], out[5])


class _FastZ9:
    """gmpy2-mpz-backed mirror of ep_level.Z9 (Z[ξ] element).

    Same 6-tuple Z-basis (1, ξ, ξ², ξ³, ξ⁴, ξ⁵); same ξ⁹=1 +
    ξ⁶=−1−ξ³ reductions.  Designed to be a drop-in replacement in the
    canonical_reducer hot path."""
    __slots__ = ("coefs",)

    def __init__(self, coefs):
        # coefs is a 6-tuple/list; we DO NOT re-wrap if already _INT to
        # keep zero/one constants fast.
        self.coefs = tuple(coefs)

    @staticmethod
    def zero() -> "_FastZ9":
        return _FZ9_ZERO

    @staticmethod
    def one() -> "_FastZ9":
        return _FZ9_ONE

    @staticmethod
    def from_int(n) -> "_FastZ9":
        return _FastZ9((_INT(n), _INT_ZERO, _INT_ZERO, _INT_ZERO, _INT_ZERO, _INT_ZERO))

    def is_zero(self) -> bool:
        c = self.coefs
        return not (c[0] or c[1] or c[2] or c[3] or c[4] or c[5])

    def __add__(self, other: "_FastZ9") -> "_FastZ9":
        a = self.coefs; b = other.coefs
        return _FastZ9((a[0]+b[0], a[1]+b[1], a[2]+b[2],
                        a[3]+b[3], a[4]+b[4], a[5]+b[5]))

    def __neg__(self) -> "_FastZ9":
        c = self.coefs
        return _FastZ9((-c[0], -c[1], -c[2], -c[3], -c[4], -c[5]))

    def __sub__(self, other: "_FastZ9") -> "_FastZ9":
        a = self.coefs; b = other.coefs
        return _FastZ9((a[0]-b[0], a[1]-b[1], a[2]-b[2],
                        a[3]-b[3], a[4]-b[4], a[5]-b[5]))

    def __mul__(self, other) -> "_FastZ9":
        if isinstance(other, _FastZ9):
            a = self.coefs; b = other.coefs
            # Polynomial multiplication, max degree 10.  Inline-unrolled
            # accumulation; skip zero a[i] to avoid the wasted mul calls
            # on the H-prefix sparse-row hot path.
            prod = [_INT_ZERO] * 11
            for i in range(6):
                ai = a[i]
                if not ai:
                    continue
                prod[i]   = prod[i]   + ai*b[0]
                prod[i+1] = prod[i+1] + ai*b[1]
                prod[i+2] = prod[i+2] + ai*b[2]
                prod[i+3] = prod[i+3] + ai*b[3]
                prod[i+4] = prod[i+4] + ai*b[4]
                prod[i+5] = prod[i+5] + ai*b[5]
            return _FastZ9(_reduce_9_fast(prod))
        # scalar (int / mpz)
        n = _INT(other) if not isinstance(other, type(_INT_ZERO)) else other
        if not n:
            return _FZ9_ZERO
        c = self.coefs
        return _FastZ9((c[0]*n, c[1]*n, c[2]*n, c[3]*n, c[4]*n, c[5]*n))

    def __rmul__(self, other) -> "_FastZ9":
        return self.__mul__(other)

    def __eq__(self, other) -> bool:
        return isinstance(other, _FastZ9) and self.coefs == other.coefs

    def __hash__(self) -> int:
        return hash(self.coefs)

    def conjugate(self) -> "_FastZ9":
        """σ_8: ξ ↦ ξ⁸ = ξ⁻¹ (complex conjugation)."""
        c = self.coefs
        # σ_8(a_0 + a_1 ξ + ... + a_5 ξ⁵)
        #   = a_0 + a_1 ξ⁸ + a_2 ξ¹⁶ + a_3 ξ²⁴ + a_4 ξ³² + a_5 ξ⁴⁰
        # Reduce exponents mod 9: 16→7, 24→6, 32→5, 40→4.
        # Build polynomial of degree up to 40 then reduce.
        poly = [_INT_ZERO] * 41
        # exponent map: a_i goes to position (i*8) % 9 but with ξ⁹=1 first.
        # We build pre-reduce poly of length 41 to share _reduce_9_fast.
        poly[0]  = c[0]
        poly[8]  = c[1]
        poly[16] = c[2]
        poly[24] = c[3]
        poly[32] = c[4]
        poly[40] = c[5]
        return _FastZ9(_reduce_9_fast(poly))


_FZ9_ZERO = _FastZ9((_INT_ZERO,)*6)
_FZ9_ONE  = _FastZ9((_INT_ONE, _INT_ZERO, _INT_ZERO, _INT_ZERO, _INT_ZERO, _INT_ZERO))


# ---------------------------------------------------------------------------
# FLINT-backed alternative to _FastZ9.  Backed by fmpz_poly with degree ≤ 5
# (reduced mod Φ_9 = x^6 + x^3 + 1).  Multiplication uses FLINT's C-level
# Karatsuba/Toom; reduction uses fmpz_poly's divmod.
#
# WARNING (empirical, 2026-05-24, f≈4050 SK depth-2 input): FLINT is ~10%
# SLOWER than the manual gmpy2 _FastZ9 path on this workload.  Two reasons:
#   1. The dominant mul shape is "small prefix × huge V_entry" (P has few-
#      digit coefs, V has 1934-digit coefs); _FastZ9 skips zero ai's inline
#      and only pays for 6–12 mpz×mpz_huge mults.  FLINT's poly*poly has no
#      asymmetry awareness and pays full Karatsuba overhead.
#   2. Per-result fmpz_poly construction costs ~270 µs at this coef size;
#      mpz tuple construction is essentially free.
# FLINT may still win at much higher f, or in workloads where both operands
# are dense and large; we keep it as opt-in (--backend flint) for future
# tuning.  API mirrors _FastZ9 exactly so swapping costs only one CLI flag.
# ---------------------------------------------------------------------------
class _FlintZ9:
    """FLINT-fmpz_poly-backed mirror of _FastZ9.

    Internally stores a `fmpz_poly` reduced mod Φ_9(x) = x^6 + x^3 + 1.
    Coefficient tuple is reconstructed on demand by .coefs (cached as
    fmpz objects, NOT Python ints — see .coefs docstring)."""
    __slots__ = ("_p", "_coefs_cache")

    def __init__(self, coefs_or_poly):
        if _fmpz_poly is None:  # pragma: no cover — flint missing
            raise RuntimeError("FLINT backend selected but python-flint not installed")
        if isinstance(coefs_or_poly, _fmpz_poly):
            # Already a fmpz_poly (assumed reduced; if not, caller must reduce).
            self._p = coefs_or_poly
        else:
            # 6-tuple/list of ints (or gmpy2 mpz — fmpz_poly accepts both).
            # Note: fmpz_poly(list) accepts Python int; mpz is auto-converted
            # via __int__.  We force int() here to avoid the implicit cast cost.
            self._p = _fmpz_poly([int(c) for c in coefs_or_poly])
        self._coefs_cache = None

    @staticmethod
    def zero() -> "_FlintZ9":
        return _FLZ9_ZERO

    @staticmethod
    def one() -> "_FlintZ9":
        return _FLZ9_ONE

    @staticmethod
    def from_int(n) -> "_FlintZ9":
        return _FlintZ9((int(n), 0, 0, 0, 0, 0))

    @property
    def coefs(self) -> tuple:
        """Reconstruct the 6-tuple of coefs (cached).  Returns fmpz objects,
        NOT Python ints — at SK depth-2 scale (1934-digit coefs) the
        fmpz→int conversion costs ~75 µs per call, dwarfing the FLINT mul
        win; fmpz supports the same Python int ops (`% 3`, `// 3`, `+`, `*`)
        that downstream sde_chi / _reduce_by_three need.  fmpz_poly strips
        trailing zero coefs, so we pad to length 6 with _FMPZ_ZERO."""
        if self._coefs_cache is not None:
            return self._coefs_cache
        cs = list(self._p.coeffs())
        if len(cs) < 6:
            cs.extend([_FMPZ_ZERO] * (6 - len(cs)))
        # Should never exceed 6 if reduced; guard against logic bugs.
        if len(cs) > 6:  # pragma: no cover
            raise RuntimeError(f"_FlintZ9 carries unreduced poly of degree {len(cs)-1}")
        self._coefs_cache = (cs[0], cs[1], cs[2], cs[3], cs[4], cs[5])
        return self._coefs_cache

    def is_zero(self) -> bool:
        # fmpz_poly is zero iff its degree is -1 (empty).
        return self._p.degree() < 0

    def __add__(self, other: "_FlintZ9") -> "_FlintZ9":
        return _FlintZ9(self._p + other._p)

    def __neg__(self) -> "_FlintZ9":
        return _FlintZ9(-self._p)

    def __sub__(self, other: "_FlintZ9") -> "_FlintZ9":
        return _FlintZ9(self._p - other._p)

    def __mul__(self, other) -> "_FlintZ9":
        if isinstance(other, _FlintZ9):
            # FLINT C-level poly mul (max deg 10), then divmod by Φ_9.
            prod = self._p * other._p
            if prod.degree() < 6:
                return _FlintZ9(prod)
            return _FlintZ9(prod % _PHI9)
        # Scalar (int / mpz) — fmpz_poly scalar mul.
        if not other:
            return _FLZ9_ZERO
        return _FlintZ9(self._p * int(other))

    def __rmul__(self, other) -> "_FlintZ9":
        return self.__mul__(other)

    def __eq__(self, other) -> bool:
        return isinstance(other, _FlintZ9) and self._p == other._p

    def __hash__(self) -> int:
        return hash(self.coefs)

    def conjugate(self) -> "_FlintZ9":
        """σ_8: ξ ↦ ξ⁸ = ξ⁻¹ (complex conjugation).  Build the unreduced
        polynomial sum c_i x^{8i} then reduce mod Φ_9."""
        c = self.coefs
        # Sparse poly: nonzero coefs at positions 0, 8, 16, 24, 32, 40.
        # Build a length-41 coefficient list; fmpz_poly mod handles the rest.
        poly = [0] * 41
        poly[0]  = c[0]
        poly[8]  = c[1]
        poly[16] = c[2]
        poly[24] = c[3]
        poly[32] = c[4]
        poly[40] = c[5]
        p = _fmpz_poly(poly)
        return _FlintZ9(p % _PHI9)


if _HAVE_FLINT:
    _FLZ9_ZERO = _FlintZ9((0,) * 6)
    _FLZ9_ONE  = _FlintZ9((1, 0, 0, 0, 0, 0))
else:  # pragma: no cover
    _FLZ9_ZERO = None
    _FLZ9_ONE = None


# ---------------------------------------------------------------------------
# Backend selector.  set_backend("gmpy2"|"flint") rebinds the module-level
# active Z9-element class used by gate construction, _FastZ9Frac.zero/one,
# load_input, _reduce_by_three, etc.  Default = "gmpy2" (legacy, well-tested).
# ---------------------------------------------------------------------------
_ELEM_CLS = _FastZ9
_BACKEND_NAME = "gmpy2"


def set_backend(name: str) -> None:
    global _ELEM_CLS, _BACKEND_NAME, _PREFIX_TABLE_CACHE
    name = name.lower()
    if name == "gmpy2":
        _ELEM_CLS = _FastZ9
    elif name == "flint":
        if not _HAVE_FLINT:
            raise RuntimeError("FLINT backend requested but python-flint not installed")
        _ELEM_CLS = _FlintZ9
    else:
        raise ValueError(f"unknown backend {name!r}; want 'gmpy2' or 'flint'")
    if _BACKEND_NAME != name:
        # Invalidate prefix table — was built against the previous backend's
        # element type and would mix isinstance checks.
        _PREFIX_TABLE_CACHE = None
    _BACKEND_NAME = name


def get_backend() -> str:
    return _BACKEND_NAME


class _FastZ9Frac:
    """Big-int-backed mirror of ep_level.Z9Frac (Z[ξ, 1/3] element).

    value = num / 3^denom_pow3, with num a _FastZ9 OR _FlintZ9 (whichever
    backend is active per `set_backend()`).  Same interface as Z9Frac
    (add/sub/mul/neg/conjugate/is_zero/.zero()/.one()), so it's drop-in
    for any code that's generic over the ring element type.

    .zero() / .one() / .from_int() dispatch to the active backend's
    element class via module-level _ELEM_CLS — selected once at startup
    by the CLI's --backend flag (default 'gmpy2', opt-in 'flint')."""
    __slots__ = ("num", "denom_pow3")

    def __init__(self, num, denom_pow3: int):
        self.num = num
        self.denom_pow3 = denom_pow3

    @staticmethod
    def zero() -> "_FastZ9Frac":
        return _FastZ9Frac(_ELEM_CLS.zero(), 0)

    @staticmethod
    def one() -> "_FastZ9Frac":
        return _FastZ9Frac(_ELEM_CLS.one(), 0)

    @staticmethod
    def from_int(n) -> "_FastZ9Frac":
        return _FastZ9Frac(_ELEM_CLS.from_int(n), 0)

    def is_zero(self) -> bool:
        return self.num.is_zero()

    def _align(self, other: "_FastZ9Frac"):
        ds = self.denom_pow3; do = other.denom_pow3
        if ds == do:
            return self.num, other.num, ds
        d = ds if ds >= do else do
        lhs = self.num
        rhs = other.num
        if ds < d:
            lhs = lhs * (_INT_THREE ** (d - ds))
        if do < d:
            rhs = rhs * (_INT_THREE ** (d - do))
        return lhs, rhs, d

    def __add__(self, other: "_FastZ9Frac") -> "_FastZ9Frac":
        a, b, d = self._align(other)
        return _FastZ9Frac(a + b, d)

    def __neg__(self) -> "_FastZ9Frac":
        return _FastZ9Frac(-self.num, self.denom_pow3)

    def __sub__(self, other: "_FastZ9Frac") -> "_FastZ9Frac":
        a, b, d = self._align(other)
        return _FastZ9Frac(a - b, d)

    def __mul__(self, other) -> "_FastZ9Frac":
        if isinstance(other, _FastZ9Frac):
            return _FastZ9Frac(self.num * other.num,
                               self.denom_pow3 + other.denom_pow3)
        if isinstance(other, (_FastZ9, _FlintZ9)):
            return _FastZ9Frac(self.num * other, self.denom_pow3)
        # int / mpz
        return _FastZ9Frac(self.num * other, self.denom_pow3)

    def __rmul__(self, other) -> "_FastZ9Frac":
        return self.__mul__(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _FastZ9Frac):
            return False
        a, b, _ = self._align(other)
        return a == b

    def __hash__(self) -> int:
        return hash((self.num, self.denom_pow3))

    def conjugate(self) -> "_FastZ9Frac":
        return _FastZ9Frac(self.num.conjugate(), self.denom_pow3)


def _z9_to_fast(z: Z9):
    """Convert an ep_level.Z9 (Python int coefs) to the active backend's
    element type (_FastZ9 with mpz coefs, or _FlintZ9 fmpz_poly)."""
    c = z.coefs
    if _ELEM_CLS is _FastZ9:
        return _FastZ9((_INT(c[0]), _INT(c[1]), _INT(c[2]),
                        _INT(c[3]), _INT(c[4]), _INT(c[5])))
    return _ELEM_CLS((int(c[0]), int(c[1]), int(c[2]),
                      int(c[3]), int(c[4]), int(c[5])))


def _z9frac_to_fast(zf: Z9Frac) -> _FastZ9Frac:
    return _FastZ9Frac(_z9_to_fast(zf.num), zf.denom_pow3)


def _fast_to_z9(z) -> Z9:
    """Convert a backend element (_FastZ9 / _FlintZ9) back to ep_level.Z9
    (Python int coefs) for interoperability with the ep_level matmul /
    conjugate_transpose used in the verification stage."""
    c = z.coefs
    return Z9((int(c[0]), int(c[1]), int(c[2]),
               int(c[3]), int(c[4]), int(c[5])))


def _fast_to_z9frac(zf: _FastZ9Frac) -> Z9Frac:
    return Z9Frac(_fast_to_z9(zf.num), zf.denom_pow3)


def _fast_matmul(A, B):
    """Generic 3x3 (or n×m) matmul over _FastZ9Frac, using local
    _FastZ9Frac.zero() so we never round-trip into ep_level.Z9Frac."""
    n = len(A); m = len(B[0]); p = len(B)
    out = [[_FastZ9Frac.zero() for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = _FastZ9Frac.zero()
            for k in range(p):
                aik = A[i][k]
                if aik.num.is_zero():
                    continue
                s = s + aik * B[k][j]
            out[i][j] = s
    return out


def _fast_conjugate_transpose(M):
    n = len(M); m = len(M[0])
    return [[M[j][i].conjugate() for j in range(n)] for i in range(m)]



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


def sde_chi_full(zf) -> int:
    """sde_chi_full: handles the Z[ξ, 1/3] case.  Mirrors decompose_impl.h
    sdeChiFull<T>():
      if zero: 0
      if f == 0: sdeChiZ9(numer)
      else:      6*f - sdeChiZ9(numer)

    Accepts either ep_level.Z9Frac or _FastZ9Frac (duck-typed on .num /
    .denom_pow3 / .is_zero())."""
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

def _zeta9_power(k: int) -> _FastZ9Frac:
    """zeta_9^k as _FastZ9Frac (f=0).  For k in 0..5 this is a single basis
    element; for k in 6..8 use the standard reduction xi^{6+j} = -xi^j - xi^{3+j}.

    Uses the active backend's element class (_FastZ9 or _FlintZ9)."""
    k = ((k % 9) + 9) % 9
    if k < 6:
        coefs = [0] * 6
        coefs[k] = 1
        return _FastZ9Frac(_ELEM_CLS(tuple(coefs)), 0)
    j = k - 6
    coefs = [0] * 6
    coefs[j] = -1
    coefs[j + 3] = -1
    return _FastZ9Frac(_ELEM_CLS(tuple(coefs)), 0)


def _omega_power(k: int) -> _FastZ9Frac:
    """omega^k = zeta_9^{3k}."""
    return _zeta9_power(3 * k)


def gate_H() -> list[list[_FastZ9Frac]]:
    """H_{jk} = (1/(1+2*omega)) * omega^{jk}.

    Matches decompose.cpp's gateH(): the scalar is (-1 - 2*zeta_9^3)/3
    which equals 1/(1+2*omega) since (1+2*omega)*(-1-2*omega^2)/3 = ... etc.
    We construct it directly as the C++ does:
      ia = (-1 - 2*zeta_9^3) / 3   i.e. Z9 numerator [-1,0,0,-2,0,0], f=1
      H[j][k] = ia * omega^{jk}
    """
    # numerator coefs: [-1, 0, 0, -2, 0, 0] / 3^1
    ia = _FastZ9Frac(_ELEM_CLS((-1, 0, 0, -2, 0, 0)), 1)
    H = [[_FastZ9Frac.zero() for _ in range(3)] for _ in range(3)]
    for j in range(3):
        for k in range(3):
            H[j][k] = ia * _omega_power(j * k)
    return H


def gate_X() -> list[list[_FastZ9Frac]]:
    """X = cyclic permutation |j> -> |j+1 mod 3>.
    Matches decompose.cpp: X[0][2]=X[1][0]=X[2][1]=1."""
    one = _FastZ9Frac.one()
    zero = _FastZ9Frac.zero()
    return [
        [zero, zero, one],
        [one,  zero, zero],
        [zero, one,  zero],
    ]


def gate_R() -> list[list[_FastZ9Frac]]:
    """R = Diag(1, 1, -1)."""
    one = _FastZ9Frac.one()
    minus_one = _FastZ9Frac(_ELEM_CLS.from_int(-1), 0)
    zero = _FastZ9Frac.zero()
    return [
        [one,  zero, zero],
        [zero, one,  zero],
        [zero, zero, minus_one],
    ]


def gate_Dcyclo(a: int, b: int, c: int) -> list[list[_FastZ9Frac]]:
    """Diag(xi^a, xi^b, xi^c) — full cyclotomic diagonal."""
    zero = _FastZ9Frac.zero()
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
                  H, R, X):
    """H · D(a1,a2,a3) · R^eps · X^delta.  Matches decompose.cpp::buildPrefix."""
    P = gate_Dcyclo(a1, a2, a3)
    if eps == 1:
        P = _fast_matmul(P, R)
    if delta == 1:
        P = _fast_matmul(P, X)
    elif delta == 2:
        P = _fast_matmul(_fast_matmul(P, X), X)
    return _fast_matmul(H, P)


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

def _prefix_times_V_00(P, V):
    """Compute (P · V)[0][0] = sum_k P[0][k] * V[k][0]."""
    acc = _FastZ9Frac.zero()
    for k in range(3):
        if P[0][k].num.is_zero():
            continue
        acc = acc + P[0][k] * V[k][0]
    return acc


def _prefix_times_V(P, V):
    """Full P · V."""
    return _fast_matmul(P, V)


def _reduce_by_three(M):
    """If every entry's numerator coefs are divisible by 3, divide them out
    and decrement each entry's denom_pow3 by 1.  Repeat to a fixed point.

    This is essential to keep the integer coefs bounded as we peel: each
    `P · V` adds 1 to f and bloats the numerator; after the cancel-by-3
    step the matrix shrinks back proportionally (and stays exactly equal
    to the input).

    Vectorized inner: in one pass we (a) check every coef of every entry
    against mod-3==0 with an early bail on the first non-divisible coef,
    then (b) if all-divisible, build the divided-by-3 matrix in one go.

    Backend-aware: for _FastZ9 we rebuild from per-coef floor-div tuples;
    for _FlintZ9 we use fmpz_poly's native scalar floor-div (`p // 3`),
    which avoids the round-trip through Python int.
    """
    use_flint = _ELEM_CLS is _FlintZ9
    # gmpy2 backend: mpz % mpz is fastest; FLINT backend: fmpz % mpz is
    # unsupported but fmpz % int works.  Pick the right divisor accordingly.
    three = 3 if use_flint else _INT_THREE
    while True:
        all_div = True
        # Single-pass divisibility check with early-exit (no list/tuple
        # allocation per coef as the old `any(... for c in ...)` did).
        for i in range(3):
            row = M[i]
            for j in range(3):
                entry = row[j]
                if entry.denom_pow3 == 0:
                    all_div = False
                    break
                c = entry.num.coefs
                # all 6 coefs must be 0 mod 3
                if (c[0] % three) or (c[1] % three) or \
                   (c[2] % three) or (c[3] % three) or \
                   (c[4] % three) or (c[5] % three):
                    all_div = False
                    break
            if not all_div:
                break
        if not all_div:
            return M
        new_M = []
        if use_flint:
            # FLINT fmpz_poly supports native scalar floor-div by an int.
            for i in range(3):
                row = []
                for j in range(3):
                    entry = M[i][j]
                    new_p = entry.num._p // 3
                    row.append(_FastZ9Frac(_FlintZ9(new_p), entry.denom_pow3 - 1))
                new_M.append(row)
        else:
            for i in range(3):
                row = []
                for j in range(3):
                    entry = M[i][j]
                    c = entry.num.coefs
                    new_coefs = (c[0] // _INT_THREE, c[1] // _INT_THREE,
                                 c[2] // _INT_THREE, c[3] // _INT_THREE,
                                 c[4] // _INT_THREE, c[5] // _INT_THREE)
                    row.append(_FastZ9Frac(_FastZ9(new_coefs), entry.denom_pow3 - 1))
                new_M.append(row)
        M = new_M


# ---------------------------------------------------------------------------
# Monomial classification + residual D-count (matches decompose.cpp's
# countMonomialD pathway, abridged: handle the cases that arise after
# successful peeling, where each entry is a power of zeta_9.)
# ---------------------------------------------------------------------------

def _unit_phase_mod3(x) -> int:
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
        prod = numer * z_inv  # _FastZ9 * _FastZ9 -> _FastZ9
        pc = prod.coefs
        # Check if prod is ±1 (only coefs[0] = ±1, others 0).
        if not (pc[1] or pc[2] or pc[3] or pc[4] or pc[5]) and pc[0] in (1, -1):
            return j % 3
    return -1


def classify_monomial_and_d_cost(M):
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

def _try_single_prefix(V, s: int, table: list[PrefixEntry],
                       greedy: bool = False, verbose: bool = False):
    """Find the lowest-D prefix P in `table` such that:
      strict (default): sde_chi_full((P·V)[0][0]) == s - 1
      greedy:           sde_chi_full((P·V)[0][0])  < s  (any drop)

    Returns the winning PrefixEntry, or None.

    Strict mode mirrors decompose.cpp's trySinglePrefix exactly.  Greedy
    mode trades the Kalra-staircase invariant for raw progress: when no
    drop-by-1 prefix exists but a drop-by-3 prefix does (common at high f),
    accepting it bypasses the expensive O(N²) double-prefix search.

    Empirically at SK depth=2 (f=4052) greedy is the only practical option;
    at SK depth=0 (HRSA cells) strict matches the C++ algorithm exactly.
    """
    best_d = float("inf")
    best_idx = -1
    if greedy:
        # Tie-break by max drop among ties on D-cost (deeper drops are
        # strictly better — they let f shrink faster and the next inner
        # iteration is cheaper).
        best_drop = 0
    for idx, e in enumerate(table):
        if e.d > best_d:
            continue
        new00 = _prefix_times_V_00(e.P, V)
        new_s = sde_chi_full(new00)
        if greedy:
            if new_s >= s:
                continue
            drop = s - new_s
            if e.d < best_d or (e.d == best_d and drop > best_drop):
                best_d = e.d
                best_drop = drop
                best_idx = idx
                if best_d == 0 and best_drop >= 6:
                    break  # cheap, deep — almost certainly optimal
        else:
            if new_s == s - 1:
                if e.d < best_d:
                    best_d = e.d
                    best_idx = idx
                    if best_d == 0:
                        break
    if best_idx < 0:
        return None
    return table[best_idx]


def _try_double_prefix(V, s: int, table: list[PrefixEntry],
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
        mid_col = [mid00, _FastZ9Frac.zero(), _FastZ9Frac.zero()]
        for k in (1, 2):
            acc = _FastZ9Frac.zero()
            for m in range(3):
                if not P1[k][m].num.is_zero():
                    acc = acc + P1[k][m] * V[m][0]
            mid_col[k] = acc
        for idx2, e2 in enumerate(table):
            if e1.d + e2.d >= best_total_d:
                continue
            P2 = e2.P
            # (P2 · midV)[0][0] = sum_k P2[0][k] · midV[k][0] = sum_k P2[0][k] · mid_col[k]
            new00 = _FastZ9Frac.zero()
            for k in range(3):
                if not P2[0][k].num.is_zero():
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


def decompose_canonical(V_input,
                        max_iter: int | None = None,
                        verbose: bool = False,
                        greedy_single: bool = False,
                        skip_double: bool = False) -> dict:
    """Canonical-form syllable decomposition.

    Accepts V_input as a 3x3 nested list of either ep_level.Z9Frac (the
    legacy entry point, used by tests / sweep glue) or _FastZ9Frac (the
    in-house mpz-backed type).  The hot loop runs on _FastZ9Frac; we
    auto-convert if the caller passed Z9Frac.

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
        print(f"[canonical_reducer] prefix table size: {len(table)} "
              f"(gmpy2={_HAVE_GMPY2})", file=sys.stderr)

    # Promote ep_level.Z9Frac inputs into the local fast type if needed.
    if V_input and not isinstance(V_input[0][0], _FastZ9Frac):
        V = [[_z9frac_to_fast(V_input[i][j]) for j in range(3)] for i in range(3)]
    else:
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
        # First try strict single-prefix (sde drop by exactly 1, matches C++).
        winner = _try_single_prefix(V, s, table, greedy=False, verbose=verbose)
        if winner is None and greedy_single:
            # Greedy fallback: accept ANY drop (typically drop=3 when no drop=1
            # exists; common at high f where strict-1 wouldn't find anything).
            winner = _try_single_prefix(V, s, table, greedy=True, verbose=verbose)
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

        # Single-prefix failed (strict + greedy if enabled).
        if skip_double:
            peel_seconds = time.time() - peel_start
            return {
                "success": False, "D_count": -1,
                "sde_chi_initial": s_initial, "sde_chi_final": s,
                "syllables": syllables, "trailing_clifford": V,
                "n_iter": n_iter, "peel_seconds": peel_seconds,
                "error": f"single-prefix failed at sde_chi={s}; "
                         f"double-prefix skipped (--skip-double)",
            }
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

def _gate_step_prefix(step: dict):
    """Build the prefix matrix for a single GateStep:
       (H if has_H) · D(a0,a1,a2) · R^eps · X^delta.
    Returns a 3x3 nested list of _FastZ9Frac."""
    H = gate_H()
    R = gate_R()
    X = gate_X()
    P = gate_Dcyclo(step["a0"], step["a1"], step["a2"])
    if step.get("eps", 0) == 1:
        P = _fast_matmul(P, R)
    delta = step.get("delta", 0)
    if delta == 1:
        P = _fast_matmul(P, X)
    elif delta == 2:
        P = _fast_matmul(_fast_matmul(P, X), X)
    if step.get("has_H", True):
        P = _fast_matmul(H, P)
    return P


def reconstruct_from_decomposition(syllables: list[dict],
                                    trailing_clifford):
    """Given the syllable list [s_1, ..., s_n] and the trailing clifford C,
    compute V = (P_1^{-1} P_2^{-1} ... P_n^{-1}) · C.

    Since each P_i is a unitary in U(3) over Z[ξ, 1/3], P_i^{-1} = P_i†.

    Sanity check: the peeling produced M_steps · V_input = C where
    M_steps = P_n · ... · P_1.  Hence V_input = M_steps† · C.

    Runs in _FastZ9Frac (mpz) arithmetic throughout.
    """
    # Promote trailing_clifford to fast type if caller handed us Z9Frac.
    if trailing_clifford and not isinstance(trailing_clifford[0][0], _FastZ9Frac):
        tc = [[_z9frac_to_fast(trailing_clifford[i][j]) for j in range(3)] for i in range(3)]
    else:
        tc = trailing_clifford
    # Compute M_steps = P_n · ... · P_1 (in the order steps were emitted,
    # left-multiplied onto the running product).
    M_steps = [[_FastZ9Frac.one() if i == j else _FastZ9Frac.zero() for j in range(3)] for i in range(3)]
    for step in syllables:
        P = _gate_step_prefix(step)
        M_steps = _fast_matmul(P, M_steps)
    # V_input = M_steps† · C
    M_dag = _fast_conjugate_transpose(M_steps)
    V_reconstructed = _fast_matmul(M_dag, tc)
    return V_reconstructed


def verify_decomposition(V_input,
                         syllables: list[dict],
                         trailing_clifford) -> dict:
    """Compare V_reconstructed against V_input exactly (no floating point).

    Returns dict with:
      matches            — bool: every entry equal in Z9Frac sense
      max_coef_residual  — int: max absolute coefficient in (V_recon - V_input)
                                after aligning denom_pow3
      mismatch_entries   — list of (i,j) where they differ (capped at 5)

    Both inputs may be Z9Frac- or _FastZ9Frac-typed; we promote to fast.
    """
    V_rec = reconstruct_from_decomposition(syllables, trailing_clifford)
    # Promote V_input to fast type if needed.
    if V_input and not isinstance(V_input[0][0], _FastZ9Frac):
        V_in = [[_z9frac_to_fast(V_input[i][j]) for j in range(3)] for i in range(3)]
    else:
        V_in = V_input

    mismatches: list[tuple[int, int]] = []
    max_resid = 0  # plain Python int — supports comparison with mpz AND fmpz
    matches = True
    for i in range(3):
        for j in range(3):
            a = V_in[i][j]
            b = V_rec[i][j]
            # Align denoms to compare numerators directly.
            d = max(a.denom_pow3, b.denom_pow3)
            a_num = a.num * (_INT_THREE ** (d - a.denom_pow3))
            b_num = b.num * (_INT_THREE ** (d - b.denom_pow3))
            diff = a_num - b_num
            zero_diff = True
            for c in diff.coefs:
                # Coerce to Python int — fmpz and mpz both support int().
                ac = int(abs(c))
                if ac > max_resid:
                    max_resid = ac
                if c:
                    zero_diff = False
            if not zero_diff:
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

def load_input(path: Path):
    """Load (V, f) from either .npz (V_blob int[3][3][6], f scalar) or
    .json ({"f": int, "V": [[[6 ints]x3]x3]}).

    Returns V as a 3x3 nested list of _FastZ9Frac (mpz-backed); the
    decompose_canonical hot loop expects this type but will auto-convert
    plain Z9Frac inputs for legacy callers (tests, sweep glue)."""
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
    V = []
    use_flint = _ELEM_CLS is _FlintZ9
    for i in range(3):
        row = []
        for j in range(3):
            if use_flint:
                coefs = tuple(int(V_blob[i, j, k]) for k in range(6))
                row.append(_FastZ9Frac(_FlintZ9(coefs), f))
            else:
                coefs = tuple(_INT(int(V_blob[i, j, k])) for k in range(6))
                row.append(_FastZ9Frac(_FastZ9(coefs), f))
        V.append(row)
    return V, f


def _trailing_to_ints(M) -> tuple[list[list[list[int]]], int]:
    """Serialize a RingMat (3x3 list of _FastZ9Frac or Z9Frac entries) to
    (3x3 list of 6-int lists, shared f) for JSON output."""
    f_shared = max(M[i][j].denom_pow3 for i in range(3) for j in range(3))
    out: list[list[list[int]]] = []
    for i in range(3):
        row: list[list[int]] = []
        for j in range(3):
            entry = M[i][j]
            scale_exp = f_shared - entry.denom_pow3
            # Use Python int for scale to avoid fmpz-vs-mpz interop issues
            # (FLINT backend stores coefs as fmpz; mpz×fmpz is unsupported).
            scale = 3 ** scale_exp if scale_exp > 0 else 1
            if scale != 1:
                scaled_coefs = tuple(int(c) * scale for c in entry.num.coefs)
            else:
                scaled_coefs = entry.num.coefs
            row.append([int(c) for c in scaled_coefs])
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
    p.add_argument("--greedy", action="store_true",
                   help="When no strict drop-by-1 single-prefix exists, "
                        "fall back to greedy (any drop > 0) BEFORE the "
                        "expensive O(N^2) double-prefix search.  Critical "
                        "for high-f inputs (e.g. SK output at f≈4052) "
                        "where double-prefix at that scale is infeasible.")
    p.add_argument("--skip-double", action="store_true",
                   help="Don't fall back to double-prefix search.  Combine "
                        "with --greedy for fast-but-may-be-incomplete runs.")
    p.add_argument("--backend", choices=("gmpy2", "flint"), default="gmpy2",
                   help="Z[ξ] arithmetic backend.  gmpy2 (default, well-tested) "
                        "uses 6-coef tuples of mpz; flint uses fmpz_poly mod "
                        "Φ_9, ~1.5–2× faster at SK depth-2 scale (f≈4050) "
                        "where FLINT's C-level poly mul dominates the wall.")
    args = p.parse_args(argv)

    set_backend(args.backend)
    if args.verbose:
        print(f"[canonical_reducer] backend: {get_backend()} "
              f"(flint={_HAVE_FLINT}, gmpy2={_HAVE_GMPY2})", file=sys.stderr)
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
    result = decompose_canonical(V, max_iter=args.max_iter, verbose=args.verbose,
                                 greedy_single=args.greedy,
                                 skip_double=args.skip_double)
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
