"""Phase 1 of the Evra-Parzanchevski qutrit Clifford+D pipeline.

Implements the Π-adic valuation, the level function ℓ(g) (EP eq 2.2),
and exact-arithmetic representation of matrices over Z[ξ, 1/3] where
ξ = ζ_9. This is the foundational module — everything in Phases 2-6
depends on the level function for tree navigation.

Mathematical setup (EP §2.1, §2.2):
    ξ      = ζ_9 = exp(2πi/9), primitive 9th root of unity
    E      = Q(ξ),  ring of integers O_E = Z[ξ]
    σ      = ξ + ξ⁻¹ = 2 cos(2π/9),  minimal poly x³ − 3x + 1
    F      = Q(σ),  ring of integers O_F = Z[σ],  totally real
    π      = 1 − ξ ∈ O_E,  with (1 − ξ)⁶ = unit · 3
    Π      = π̄π = (2 − σ) ∈ O_F,  prime above 3, ramified in E
    Γ      = G(O_F[1/Π]) = unitary 3x3 matrices over Z[ξ, 1/3]
    ord_Π  : F → Z ∪ {∞},  the Π-adic valuation
    ℓ(g)   = -2·ord_Π(g)  for g ∈ Γ ⊂ U(3) (EP eq 2.2 specialized to det g = ±ξ^k)

Z-basis of Z[ξ]: we use (1, ξ, ξ², ξ³, ξ⁴, ξ⁵). Since 1 + ξ³ + ξ⁶ = 0,
ξ⁶ = −1 − ξ³, ξ⁷ = −ξ − ξ⁴, ξ⁸ = −ξ² − ξ⁵, every Z-linear combination of
(1, ..., ξ⁸) is uniquely reducible to a Z-linear combination of
(1, ξ, ..., ξ⁵). This is the same reduction used by ringZ9 in cyclotomic_int9.h.

Status: WORKING; unit tests at the bottom pass on Python 3.10+ with numpy only.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Z[ξ] arithmetic (integer-coefficient ninth-cyclotomic ring)
# ---------------------------------------------------------------------------

# We represent α ∈ Z[ξ] as a 6-tuple of integers a = (a_0, ..., a_5) meaning
#     α = a_0 + a_1 ξ + a_2 ξ² + a_3 ξ³ + a_4 ξ⁴ + a_5 ξ⁵.
# Multiplication produces degrees up to 10; we then reduce using
#     ξ⁶ = −1 − ξ³
#     ξ⁷ = −ξ − ξ⁴
#     ξ⁸ = −ξ² − ξ⁵
#     ξ⁹ = 1     (since ξ is a 9th root of unity)
# ξ^k for k ≥ 9: reduce k mod 9 first.

def _reduce_9(c: list[int]) -> tuple[int, int, int, int, int, int]:
    """Reduce a polynomial of degree up to 10 in ξ to a 6-tuple over the
    Z-basis (1, ξ, ξ², ξ³, ξ⁴, ξ⁵)."""
    # First reduce powers ≥ 9 using ξ^9 = 1.
    out = [0] * 9
    for k, v in enumerate(c):
        out[k % 9] += v
    # Now use ξ⁶ = -1 - ξ³, ξ⁷ = -ξ - ξ⁴, ξ⁸ = -ξ² - ξ⁵.
    if out[6]:
        out[0] -= out[6]
        out[3] -= out[6]
        out[6] = 0
    if out[7]:
        out[1] -= out[7]
        out[4] -= out[7]
        out[7] = 0
    if out[8]:
        out[2] -= out[8]
        out[5] -= out[8]
        out[8] = 0
    return tuple(out[:6])


@dataclass(frozen=True)
class Z9:
    """An element of Z[ξ] where ξ = ζ_9."""
    coefs: tuple[int, int, int, int, int, int]

    @staticmethod
    def zero() -> "Z9":
        return Z9((0, 0, 0, 0, 0, 0))

    @staticmethod
    def one() -> "Z9":
        return Z9((1, 0, 0, 0, 0, 0))

    @staticmethod
    def xi(k: int = 1) -> "Z9":
        """Return ξ^k as a Z9 element."""
        k = k % 9
        if k <= 5:
            t = [0] * 6
            t[k] = 1
            return Z9(tuple(t))
        # k in {6, 7, 8}: use reduction
        return _from_poly([0] * k + [1])

    @staticmethod
    def from_int(n: int) -> "Z9":
        return Z9((n, 0, 0, 0, 0, 0))

    def __add__(self, other: "Z9") -> "Z9":
        return Z9(tuple(a + b for a, b in zip(self.coefs, other.coefs)))

    def __neg__(self) -> "Z9":
        return Z9(tuple(-a for a in self.coefs))

    def __sub__(self, other: "Z9") -> "Z9":
        return Z9(tuple(a - b for a, b in zip(self.coefs, other.coefs)))

    def __mul__(self, other) -> "Z9":
        if isinstance(other, int):
            return Z9(tuple(other * a for a in self.coefs))
        if not isinstance(other, Z9):
            return NotImplemented
        # Polynomial multiplication, degrees up to 5+5 = 10.
        a = self.coefs
        b = other.coefs
        prod = [0] * 11
        for i in range(6):
            if a[i] == 0:
                continue
            for j in range(6):
                prod[i + j] += a[i] * b[j]
        return Z9(_reduce_9(prod))

    def __rmul__(self, other) -> "Z9":
        return self.__mul__(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Z9):
            return False
        return self.coefs == other.coefs

    def __hash__(self) -> int:
        return hash(self.coefs)

    def is_zero(self) -> bool:
        return all(c == 0 for c in self.coefs)

    def galois(self, k: int) -> "Z9":
        """Galois automorphism σ_k: ξ ↦ ξ^k, requires gcd(k, 9) = 1.
        For us the relevant ones are k ∈ {1, 2, 4, 5, 7, 8}."""
        if math.gcd(k, 9) != 1:
            raise ValueError(f"σ_{k} is not a Galois automorphism (gcd({k}, 9) ≠ 1)")
        # σ_k(a_0 + a_1 ξ + ... + a_5 ξ⁵) = a_0 + a_1 ξ^k + ... + a_5 ξ^(5k)
        # Build the polynomial and reduce.
        c = [0] * 46  # max degree 5*k <= 45 for k=8
        for i, a in enumerate(self.coefs):
            c[i * k] += a
        return Z9(_reduce_9(c))

    def conjugate(self) -> "Z9":
        """Complex conjugation = σ_{-1} = σ_8."""
        return self.galois(8)

    def to_complex(self) -> complex:
        """Numerical embedding into C using ξ = e^(2πi/9)."""
        xi = cmath.exp(2j * cmath.pi / 9)
        result = 0 + 0j
        for k, a in enumerate(self.coefs):
            result += a * (xi ** k)
        return result

    def field_norm(self) -> int:
        """The absolute norm N_{E/Q}(α) = ∏ σ_k(α) for k ∈ (Z/9Z)*,
        which is always a rational integer for α ∈ O_E."""
        result = self
        for k in (2, 4, 5, 7, 8):
            result = result * self.galois(k)
        # result should now lie in Z (i.e. coefs = (n, 0, 0, 0, 0, 0))
        assert all(c == 0 for c in result.coefs[1:]), (
            f"field_norm yielded non-integer ringZ9 element: {result.coefs}; "
            f"this is a bug — norm of a ring element is always rational."
        )
        return result.coefs[0]

    def __repr__(self) -> str:
        terms = []
        for k, a in enumerate(self.coefs):
            if a == 0:
                continue
            if k == 0:
                terms.append(str(a))
            elif k == 1:
                terms.append(f"{a}ξ")
            else:
                terms.append(f"{a}ξ^{k}")
        return "Z9(" + (" + ".join(terms) if terms else "0") + ")"


def _from_poly(c: Sequence[int]) -> Z9:
    """Helper: convert a list of polynomial coefficients (length ≥ 0) to Z9."""
    return Z9(_reduce_9(list(c)))


# ---------------------------------------------------------------------------
# Z[ξ, 1/3] arithmetic — needed for actual Γ elements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Z9Frac:
    """An element of Z[ξ, 1/3], stored as (numerator, denom_pow3) where
    the value is numerator / 3^denom_pow3.

    We do NOT reduce the gcd; we track denom_pow3 explicitly. Reduction
    can be done via reduce_denom() but is optional for correctness."""
    num: Z9
    denom_pow3: int  # value = num / 3^denom_pow3

    @staticmethod
    def zero() -> "Z9Frac":
        return Z9Frac(Z9.zero(), 0)

    @staticmethod
    def one() -> "Z9Frac":
        return Z9Frac(Z9.one(), 0)

    @staticmethod
    def from_int(n: int) -> "Z9Frac":
        return Z9Frac(Z9.from_int(n), 0)

    @staticmethod
    def xi(k: int = 1) -> "Z9Frac":
        return Z9Frac(Z9.xi(k), 0)

    @staticmethod
    def from_z9(z: Z9, denom_pow3: int = 0) -> "Z9Frac":
        return Z9Frac(z, denom_pow3)

    def _align(self, other: "Z9Frac") -> tuple[Z9, Z9, int]:
        """Return (lhs_num, rhs_num, common_denom_pow3) such that
        self = lhs_num / 3^d and other = rhs_num / 3^d."""
        d = max(self.denom_pow3, other.denom_pow3)
        lhs = self.num * Z9.from_int(3 ** (d - self.denom_pow3))
        rhs = other.num * Z9.from_int(3 ** (d - other.denom_pow3))
        return lhs, rhs, d

    def __add__(self, other: "Z9Frac") -> "Z9Frac":
        a, b, d = self._align(other)
        return Z9Frac(a + b, d)

    def __neg__(self) -> "Z9Frac":
        return Z9Frac(-self.num, self.denom_pow3)

    def __sub__(self, other: "Z9Frac") -> "Z9Frac":
        a, b, d = self._align(other)
        return Z9Frac(a - b, d)

    def __mul__(self, other) -> "Z9Frac":
        if isinstance(other, int):
            return Z9Frac(self.num * other, self.denom_pow3)
        if isinstance(other, Z9):
            return Z9Frac(self.num * other, self.denom_pow3)
        if isinstance(other, Z9Frac):
            return Z9Frac(self.num * other.num, self.denom_pow3 + other.denom_pow3)
        return NotImplemented

    def __rmul__(self, other) -> "Z9Frac":
        return self.__mul__(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Z9Frac):
            return False
        a, b, _ = self._align(other)
        return a == b

    def __hash__(self) -> int:
        # Not perfectly canonical because we don't auto-reduce, but
        # equal Z9Fracs hash equal if they happen to have the same form.
        return hash((self.num, self.denom_pow3))

    def conjugate(self) -> "Z9Frac":
        return Z9Frac(self.num.conjugate(), self.denom_pow3)

    def to_complex(self) -> complex:
        return self.num.to_complex() / (3 ** self.denom_pow3)

    def is_zero(self) -> bool:
        return self.num.is_zero()

    def __repr__(self) -> str:
        if self.denom_pow3 == 0:
            return f"Z9Frac({self.num})"
        return f"Z9Frac({self.num} / 3^{self.denom_pow3})"


# ---------------------------------------------------------------------------
# Π-adic valuation
# ---------------------------------------------------------------------------

# Key facts (EP §2 and standard cyclotomic theory):
#   π = 1 - ξ ∈ O_E.
#   π̄π = 2 - σ = (1-ξ)(1-ξ⁻¹) = N_{E/F}(π).
#   (1 - ξ)^6 = u · 3 for a unit u ∈ O_E^×, so ord_π(3) = 6.
#   Π = π̄π ∈ O_F is the prime above 3; Π is ramified in E with e = 2 (because
#   π and π̄ are both prime factors of Π in O_E? No — the relevant fact is that
#   ord_Π(3) = 3 in F, and ord_π(3) = 6 in E, so the ramification index of
#   π over Π is 2). In short: ord_π = 2·ord_Π for elements of F, and for
#   elements of E (not just F), we use ord_π as the natural valuation.

# Important: EP's level function (eq 2.2) uses ord_π for matrices in U(3)
# over Z[ξ, 1/Π] = Z[ξ, 1/3]:
#       ℓ(g) := ord_p(g g*) − ord_π(g) − ord_π(g*)
# For g g* = I (unitary), this becomes ℓ(g) = -2·ord_π(g) (since ord_π(g*) =
# ord_π(g) when * is complex conjugation acting on entries). The factor of 2
# is also consistent with the tree-distance normalization.

# Concretely: we compute ord_π for elements of Z[ξ] by trial division.
# An element α ∈ Z[ξ] satisfies α = (1-ξ) · β for some β ∈ Z[ξ] iff
# the "augmentation" Σ a_i (evaluating ξ at 1) is 0. Iteratively:
#       while augmentation(α) == 0: α := α / (1-ξ); k += 1
# Each division-by-(1-ξ) can be done explicitly via polynomial division.

# For the Π-adic valuation in F: ord_Π(α) for α ∈ F is ord_π(α) / 2
# (since the ramification index of π over Π is 2; equivalently, F = E ∩ R,
# and α ∈ F means α = ᾱ, and ord_π(α) is necessarily even). We do NOT need
# to compute this distinction explicitly because for ord_π we work directly
# in O_E.


def augmentation_mod3(z: Z9) -> int:
    """Reduction of z mod π = 1-ξ in O_E. Since O_E/π ≅ F_3 (field with 3
    elements, because N(π) = 3), this reduction maps ξ → 1 and lives in F_3.

    Returns Σ a_i mod 3, in the range {0, 1, 2}.
    A nonzero α ∈ Z[ξ] is divisible by π iff augmentation_mod3(α) == 0.
    """
    return sum(z.coefs) % 3


def divide_by_pi(z: Z9) -> Z9:
    """Given α ∈ Z[ξ] with π | α (i.e. augmentation_mod3(α) = 0),
    return β ∈ Z[ξ] such that α = (1 - ξ) · β.

    Method: work in the polynomial ring Z[x]; consider α as a polynomial
    α(x) = a_0 + a_1 x + ... + a_5 x⁵ of degree ≤ 5. We want β(x) such that
    α(x) ≡ (1 - x) β(x) (mod Φ_9(x)) where Φ_9(x) = x⁶ + x³ + 1.

    Direct polynomial division of α(x) by (1 - x) in Q[x] gives a quotient
    of degree ≤ 4 plus a remainder α(1) (a rational integer). When π | α,
    the remainder must be... NOT zero in general (α(1) = 0 only when
    augmentation = 0, but the correct condition is augmentation ≡ 0 mod 3,
    not = 0). So the direct polynomial-division approach is WRONG.

    The correct approach: use that (1 - ξ)(1 + ξ + ... + ξ⁸)/9 = 1/9 · 0 =
    0 → no. Different approach: since N(π) = 3, π divides α iff 3 divides
    α · (conjugate stuff). Specifically:

        π · π̄ · σ_2(π) · σ_2(π̄) · σ_4(π) · σ_4(π̄) = N(π) = 3.

    So if we write α = π · β, then multiplying by all Galois conjugates of
    π except π itself gives:

        α · ∏_{k=2,4,5,7,8} σ_k(π)  =  π · β · ∏_{k=2,4,5,7,8} σ_k(π)
                                     = β · N(π)
                                     = 3 β.

    Hence β = α · ∏_{k=2,4,5,7,8} σ_k(π) / 3.

    The right-hand side is well-defined in Z[ξ] precisely when π | α.
    Implementation: compute P = ∏_{k=2,4,5,7,8} σ_k(π); multiply α · P;
    divide each coef by 3 (must be exact). This is O(1) (no division loop).
    """
    if augmentation_mod3(z) != 0:
        raise ValueError(f"divide_by_pi: π does not divide {z} "
                         f"(augmentation mod 3 = {augmentation_mod3(z)})")
    P = _pi_complement()  # ∏_{k ≠ 1} σ_k(π), precomputed
    prod = z * P  # = 3 · β where β = z / π.
    # Now divide each coef by 3.
    out = []
    for c in prod.coefs:
        if c % 3 != 0:
            raise AssertionError(
                f"divide_by_pi: coef {c} not divisible by 3 in z·P; bug.\n"
                f"z = {z}, P = {P}, z·P = {prod}"
            )
        out.append(c // 3)
    return Z9(tuple(out))


_PI_COMPLEMENT_CACHE: Z9 | None = None


def _pi_complement() -> Z9:
    """Cached value of ∏_{k ∈ {2,4,5,7,8}} σ_k(1-ξ).

    σ_k(1-ξ) = 1 - ξ^k. So this is (1-ξ²)(1-ξ⁴)(1-ξ⁵)(1-ξ⁷)(1-ξ⁸).
    Multiplied by (1-ξ), the product over all primitive-9th-root-of-unity
    units factors of (x⁶+x³+1) evaluated at 1 gives Φ_9(1) = 1+1+1 = 3.
    So this product itself equals 3 / (1-ξ), but stored exactly as a Z[ξ]
    element.
    """
    global _PI_COMPLEMENT_CACHE
    if _PI_COMPLEMENT_CACHE is None:
        pi = Z9.one() - Z9.xi(1)
        P = Z9.one()
        for k in (2, 4, 5, 7, 8):
            P = P * pi.galois(k)
        _PI_COMPLEMENT_CACHE = P
    return _PI_COMPLEMENT_CACHE


def ord_pi(z: Z9 | Z9Frac) -> int:
    """π-adic valuation ord_π(z) where π = 1 - ξ ∈ O_E.

    For z = 0 returns a large sentinel (10**9) which behaves as +∞ for our
    min() purposes."""
    if isinstance(z, Z9Frac):
        # ord_π(num / 3^d) = ord_π(num) - 6·d (since ord_π(3) = 6).
        return ord_pi(z.num) - 6 * z.denom_pow3
    if not isinstance(z, Z9):
        raise TypeError(f"ord_pi: expected Z9 or Z9Frac, got {type(z)}")
    if z.is_zero():
        return 10**9
    k = 0
    cur = z
    # We divide repeatedly while π divides the current value.
    # π | α  iff  augmentation_mod3(α) == 0  (since O_E / π ≅ F_3).
    while augmentation_mod3(cur) == 0:
        cur = divide_by_pi(cur)
        k += 1
        if k > 10000:
            raise RuntimeError(
                f"ord_pi loop exceeded 10000 iterations on input {z}; "
                f"this should not happen for any reasonable input."
            )
    return k


def ord_Pi(z: Z9 | Z9Frac) -> int:
    """Π-adic valuation ord_Π(z) for z ∈ F = Q(σ); requires z ∈ F (i.e. z̄ = z).
    For z ∈ E \\ F, ord_Pi is not generally an integer. We compute it as
    ord_π(z) / 2 because the ramification index of π over Π is 2."""
    op = ord_pi(z)
    if op % 2 != 0 and op < 10**9:
        # Not in F — but ord_Pi only makes sense in F. For matrices, ord_π
        # over E is what's used in EP eq 2.2 anyway, so this is mostly a
        # sanity helper.
        raise ValueError(f"ord_Pi: input has odd ord_π = {op}, so not in F")
    return op // 2


def ord_pi_mat(M: Sequence[Sequence[Z9Frac]]) -> int:
    """min ord_π over all entries of an n×n matrix M over Z[ξ, 1/3]."""
    return min(ord_pi(e) for row in M for e in row)


# ---------------------------------------------------------------------------
# Matrix operations over Z9Frac
# ---------------------------------------------------------------------------

def matmul(A: list[list[Z9Frac]], B: list[list[Z9Frac]]) -> list[list[Z9Frac]]:
    n = len(A); m = len(B[0]); p = len(B)
    out = [[Z9Frac.zero() for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = Z9Frac.zero()
            for k in range(p):
                s = s + A[i][k] * B[k][j]
            out[i][j] = s
    return out


def conjugate_transpose(M: list[list[Z9Frac]]) -> list[list[Z9Frac]]:
    n = len(M); m = len(M[0])
    return [[M[j][i].conjugate() for j in range(n)] for i in range(m)]


def is_unitary(M: list[list[Z9Frac]], tol: float = 1e-9) -> bool:
    """Check M·M* = I via complex-double evaluation (cheap)."""
    n = len(M)
    Mc = np.array([[M[i][j].to_complex() for j in range(n)] for i in range(n)])
    prod = Mc @ Mc.conj().T
    return np.allclose(prod, np.eye(n), atol=tol)


def det3(M: list[list[Z9Frac]]) -> Z9Frac:
    """Exact determinant for 3x3 matrix over Z9Frac."""
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


# ---------------------------------------------------------------------------
# Level function (EP eq 2.2)
# ---------------------------------------------------------------------------

def level(g: list[list[Z9Frac]]) -> int:
    """Level function ℓ(g) per EP eq 2.2:
            ℓ(g) = ord_π(g g*) − ord_π(g) − ord_π(g*)

    For g ∈ U(3), g g* = I has ord_π = 0, so ℓ(g) = −2·ord_π(g).
    Result is always a non-negative even integer for g ∈ Γ.

    Note: EP's distance in T is dist(v_0, g v_0) = ℓ(g) / 2 (Prop 2.2)."""
    g_star = conjugate_transpose(g)
    gg_star = matmul(g, g_star)
    op_g = ord_pi_mat(g)
    op_g_star = ord_pi_mat(g_star)
    op_ggstar = ord_pi_mat(gg_star)
    return op_ggstar - op_g - op_g_star


# ---------------------------------------------------------------------------
# Canonical gates (EP Definition 1.1)
# ---------------------------------------------------------------------------

# H = (1 / sqrt(-3)) · [[1, 1, 1], [1, ξ³, ξ̄³], [1, ξ̄³, ξ³]]
#   where ξ³ = ζ_3 (primitive cube root of unity) and ξ̄³ = ξ⁻³ = ξ⁶.
# Note: sqrt(-3) is the issue — it's not in Z[ξ, 1/3] directly.
# Standard trick: H entries are NOT in Z[ξ, 1/3]; rather, ξ - ξ⁻¹ has
# (ξ - ξ⁻¹)² = ξ² - 2 + ξ⁻² which equals -3 (after some algebra using
# the cube root relation). Actually:
#    Let ω = ξ³. Then ω - ω² = ξ³ - ξ⁶ = ξ³ - (-1 - ξ³) = 1 + 2ξ³.
#    (ω - ω²)² = ω² - 2ω³ + ω⁴ = ω² - 2 + ω. Since 1 + ω + ω² = 0,
#    we get ω² + ω = -1, so (ω - ω²)² = -1 - 2 = -3. ✓
# So sqrt(-3) = ω - ω² = ξ³ - ξ⁶ = 1 + 2ξ³ (using ξ⁶ = -1 - ξ³).
# Therefore 1/sqrt(-3) = (ω² - ω) / 3 = -(1 + 2ξ³)/3 ∈ Z[ξ, 1/3]. ✓

def _omega() -> Z9:
    """ω = ξ³ = ζ_3, primitive cube root of unity."""
    return Z9.xi(3)


def _sqrt_neg3() -> Z9:
    """Returns √(-3) as ω - ω² = ξ³ - ξ⁶ ∈ Z[ξ].
    Numerical check: (ω - ω²)² = -3."""
    return Z9.xi(3) - Z9.xi(6)


def _inv_sqrt_neg3() -> Z9Frac:
    """Returns 1/√(-3) as -(ω - ω²) / 3, an element of Z[ξ, 1/3]."""
    # 1/(ω - ω²): multiply numerator and denominator by (ω - ω²)' (its conj).
    # (ω - ω²)(ω̄ - ω̄²) = ... but easier: (ω - ω²)² = -3, so
    # 1/(ω - ω²) = (ω - ω²)/(-3) = -(ω - ω²)/3.
    return Z9Frac(-(Z9.xi(3) - Z9.xi(6)), 1)


def H_matrix() -> list[list[Z9Frac]]:
    """The qutrit Hadamard H from EP Def 1.1."""
    w = _omega()        # ω = ξ³
    wbar = Z9.xi(6)     # ω̄ = ξ⁶
    s = _inv_sqrt_neg3()  # 1/√(-3) as Z9Frac

    # Rows of H before the 1/√(-3) prefactor:
    #   [1, 1, 1]
    #   [1, ω, ω̄]
    #   [1, ω̄, ω]
    rows_raw = [
        [Z9.one(),  Z9.one(),  Z9.one()],
        [Z9.one(),  w,         wbar],
        [Z9.one(),  wbar,      w],
    ]
    # Multiply by 1/√(-3):
    return [[s * Z9Frac(e, 0) for e in row] for row in rows_raw]


def S_matrix() -> list[list[Z9Frac]]:
    """S = diag(1, ζ_3, 1) = diag(1, ξ³, 1) ∈ Γ."""
    one = Z9Frac.one()
    z = Z9Frac.zero()
    w = Z9Frac(Z9.xi(3), 0)
    return [
        [one, z, z],
        [z, w, z],
        [z, z, one],
    ]


def T_matrix() -> list[list[Z9Frac]]:
    """T = diag(ξ, 1, ξ⁻¹) ∈ Γ (the qubit-T analog from EP Def 1.1)."""
    one = Z9Frac.one()
    z = Z9Frac.zero()
    xi = Z9Frac(Z9.xi(1), 0)
    xi_inv = Z9Frac(Z9.xi(8), 0)  # ξ⁻¹ = ξ⁸
    return [
        [xi, z, z],
        [z, one, z],
        [z, z, xi_inv],
    ]


def D_matrix(a: int, b: int, c: int,
             signs: tuple[int, int, int] = (1, 1, 1)) -> list[list[Z9Frac]]:
    """A D-gate: diag(s_a · ξ^a, s_b · ξ^b, s_c · ξ^c).

    Per EP Definition 1.1, a, b, c ∈ Z/9Z and each sign is ±1. The full
    D set has 6 · 9³ = 4374 elements; in U(3) we have an extra 8/2 = 4
    factor from sign choices that don't all flip overall det (the det
    determines a constraint), so |D| = 1944. But for the level function
    we don't need that constraint."""
    z = Z9Frac.zero()
    e0 = Z9Frac(Z9.xi(a) * signs[0], 0)
    e1 = Z9Frac(Z9.xi(b) * signs[1], 0)
    e2 = Z9Frac(Z9.xi(c) * signs[2], 0)
    return [
        [e0, z, z],
        [z, e1, z],
        [z, z, e2],
    ]


def I_matrix(n: int = 3) -> list[list[Z9Frac]]:
    """Identity matrix as Z9Frac."""
    return [
        [Z9Frac.one() if i == j else Z9Frac.zero() for j in range(n)]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_ring_basics():
    """Z[ξ] basic relations."""
    zero = Z9.zero()
    one = Z9.one()
    xi = Z9.xi(1)

    # ξ⁹ = 1.
    p = one
    for _ in range(9):
        p = p * xi
    assert p == one, f"ξ⁹ should be 1, got {p}"

    # 1 + ξ³ + ξ⁶ = 0.
    s = one + Z9.xi(3) + Z9.xi(6)
    assert s == zero, f"1 + ξ³ + ξ⁶ should be 0, got {s}"

    # 1 + ξ + ξ² + ξ³ + ξ⁴ + ξ⁵ + ξ⁶ + ξ⁷ + ξ⁸ = 0 (sum of all 9th roots).
    s = zero
    for k in range(9):
        s = s + Z9.xi(k)
    assert s == zero, f"sum of all ξ^k should be 0, got {s}"

    # Φ_9(ξ) = ξ⁶ + ξ³ + 1 = 0.
    s = Z9.xi(6) + Z9.xi(3) + one
    assert s == zero, f"Φ_9(ξ) should be 0, got {s}"

    # (ω - ω²)² = -3.
    s = _sqrt_neg3()
    sq = s * s
    expected = Z9.from_int(-3)
    assert sq == expected, f"(ω - ω²)² should be -3, got {sq}"

    print("  PASS  test_ring_basics")


def _test_z9frac_basics():
    one = Z9Frac.one()
    third = Z9Frac(Z9.from_int(1), 1)
    # 3 · (1/3) = 1.
    p = Z9Frac.from_int(3) * third
    assert p == one, f"3 · (1/3) should be 1, got {p}"

    # (1/3)² = 1/9.
    p = third * third
    expected = Z9Frac(Z9.from_int(1), 2)
    assert p == expected, f"(1/3)² should be 1/9, got {p}"

    # 1/√(-3) · √(-3) = 1.
    inv = _inv_sqrt_neg3()
    s = Z9Frac(_sqrt_neg3(), 0)
    p = inv * s
    assert p == one, f"1/√(-3) · √(-3) should be 1, got {p}"

    print("  PASS  test_z9frac_basics")


def _test_galois():
    # σ_8(ξ) = ξ⁸ = ξ⁻¹.
    xi = Z9.xi(1)
    assert xi.galois(8) == Z9.xi(8), "σ_8(ξ) should be ξ⁸"

    # σ_8 ∘ σ_8 = σ_64 mod 9 = σ_1 = id.
    z = Z9.from_int(1) + Z9.xi(1) + Z9.xi(2) + Z9.xi(4)
    assert z.galois(8).galois(8) == z, "σ_8² should be identity"

    # Field norm of ξ is 1 (since ξ is a unit, all conjugates are also units,
    # and the norm is the product of all 9th roots of unity other than ±1...
    # Actually, the norm of ξ_n over Q is ±1 since it's a unit; precisely
    # N(ξ_9) = ∏_{gcd(k,9)=1} ξ^k = product of primitive 9th roots = Φ_9(0) = 1).
    assert Z9.xi(1).field_norm() == 1, "field norm of ξ should be 1"

    # Field norm of 3 is 3⁶ = 729.
    assert Z9.from_int(3).field_norm() == 3**6, "N(3) should be 3⁶"

    # Field norm of 1 - ξ is 9 (the prime 3 ramifies in Q(ξ) with degree 6 / 2 = ...; actually for ζ_9 the relevant fact is N(1-ζ_9) = 3.)
    # Hmm — N_{Q(ξ)/Q}(1 - ξ) = Φ_9(1) = 1⁶ + 1³ + 1 = 3.
    norm_1mxi = (Z9.one() - Z9.xi(1)).field_norm()
    assert norm_1mxi == 3, f"N(1-ξ) should be 3, got {norm_1mxi}"

    print("  PASS  test_galois")


def _test_ord_pi():
    """Verify ord_π on canonical elements."""
    # ord_π(0) = +∞ sentinel.
    assert ord_pi(Z9.zero()) >= 10**8, "ord_π(0) should be ≥ 10⁸"

    # ord_π(1) = 0.
    assert ord_pi(Z9.one()) == 0, "ord_π(1) should be 0"

    # ord_π(ξ) = 0 (unit).
    assert ord_pi(Z9.xi(1)) == 0, "ord_π(ξ) should be 0"

    # ord_π(1 - ξ) = 1.
    pi = Z9.one() - Z9.xi(1)
    assert ord_pi(pi) == 1, f"ord_π(1-ξ) should be 1, got {ord_pi(pi)}"

    # ord_π((1-ξ)²) = 2.
    pi_sq = pi * pi
    assert ord_pi(pi_sq) == 2, f"ord_π((1-ξ)²) should be 2, got {ord_pi(pi_sq)}"

    # ord_π(3) = 6: since (1-ξ)^6 differs from 3 by a unit.
    assert ord_pi(Z9.from_int(3)) == 6, f"ord_π(3) should be 6, got {ord_pi(Z9.from_int(3)) }"

    # ord_π(9) = 12.
    assert ord_pi(Z9.from_int(9)) == 12, f"ord_π(9) should be 12, got {ord_pi(Z9.from_int(9)) }"

    # Valuation additivity: ord_π(α · β) = ord_π(α) + ord_π(β).
    a = Z9.from_int(2) + Z9.xi(1)
    b = Z9.one() - Z9.xi(1) + Z9.xi(3)
    op_a = ord_pi(a)
    op_b = ord_pi(b)
    op_ab = ord_pi(a * b)
    assert op_ab == op_a + op_b, (
        f"ord_π(α·β) = {op_ab} but ord_π(α) + ord_π(β) = {op_a} + {op_b} = {op_a + op_b}"
    )

    # ord_π(α + β) ≥ min(ord_π(α), ord_π(β)).
    a = pi_sq  # ord 2
    b = pi  # ord 1
    op_sum = ord_pi(a + b)
    assert op_sum >= min(op_a, op_b), "valuation strong triangle inequality fails"

    # ord_π in Z9Frac: 1/3 has ord_π(1/3) = -6.
    third = Z9Frac(Z9.from_int(1), 1)
    assert ord_pi(third) == -6, f"ord_π(1/3) should be -6, got {ord_pi(third)}"

    # 1/√(-3) has ord_π = -3 (since ord_π(√(-3)) = ord_π(ω - ω²) = ?).
    # (ω - ω²)² = -3, so ord_π((ω-ω²)²) = ord_π(-3) = 6, so ord_π(ω - ω²) = 3.
    # Then ord_π(1/√(-3)) = -3.
    s = _sqrt_neg3()
    assert ord_pi(s) == 3, f"ord_π(√(-3)) should be 3, got {ord_pi(s)}"
    assert ord_pi(_inv_sqrt_neg3()) == -3, (
        f"ord_π(1/√(-3)) should be -3, got {ord_pi(_inv_sqrt_neg3())}"
    )

    print("  PASS  test_ord_pi")


def _test_level_on_gates():
    """ℓ(I) = 0, ℓ(D-gate) = 0, ℓ(H) = 6."""
    I = I_matrix()
    lev_I = level(I)
    assert lev_I == 0, f"ℓ(I) should be 0, got {lev_I}"

    # S = diag(1, ξ³, 1) has level 0 (it's monomial with entries in ⟨-ξ⟩).
    S = S_matrix()
    lev_S = level(S)
    assert lev_S == 0, f"ℓ(S) should be 0, got {lev_S}"

    # T = diag(ξ, 1, ξ⁻¹), also level 0.
    T = T_matrix()
    lev_T = level(T)
    assert lev_T == 0, f"ℓ(T) should be 0, got {lev_T}"

    # Random D-gate.
    D = D_matrix(2, 5, 7, signs=(-1, 1, -1))
    lev_D = level(D)
    assert lev_D == 0, f"ℓ(D(2,5,7,-1,1,-1)) should be 0, got {lev_D}"

    # H = the qutrit Hadamard. EP claims ℓ(H) = 6 (eq 3.2 etc).
    H = H_matrix()
    # Sanity: H is unitary (numerically).
    assert is_unitary(H), "H should be unitary"
    lev_H = level(H)
    assert lev_H == 6, f"ℓ(H) should be 6, got {lev_H}"

    print("  PASS  test_level_on_gates")


def _test_level_multiplicative_on_unitary():
    """For g, h ∈ Γ ⊂ U(3), we have ℓ(g·h) ≤ ℓ(g) + ℓ(h) but NOT generally
    equality. Distance in T satisfies triangle inequality."""
    H = H_matrix()
    HH = matmul(H, H)
    # ℓ(H·H): could be 0 (if H² stays at v_0) or up to 12.
    # In fact H² is a Clifford (level 0), so ℓ(H²) = 0.
    lev_HH = level(HH)
    assert lev_HH == 0, f"ℓ(H²) should be 0 (H² is in C₀), got {lev_HH}"
    # Triangle: 0 ≤ 6 + 6. ✓

    # ℓ(D · H) — D is level 0, H is level 6, product should be at most 6.
    D = D_matrix(1, 2, 3)
    DH = matmul(D, H)
    lev_DH = level(DH)
    assert lev_DH <= 6, f"ℓ(D·H) should be ≤ 6, got {lev_DH}"
    # Actually it should be exactly 6, because D ∈ C₀ stabilizes v_0.
    assert lev_DH == 6, f"ℓ(D·H) should be exactly 6 (D ∈ C), got {lev_DH}"

    # ℓ(H · D) — same reasoning.
    HD = matmul(H, D)
    lev_HD = level(HD)
    assert lev_HD == 6, f"ℓ(H·D) should be 6, got {lev_HD}"

    print("  PASS  test_level_multiplicative_on_unitary")


def _test_level_at_distance_12():
    """ℓ(H · D · H) ∈ {0, 2, ..., 12}; for D's not in C ∩ HCH⁻¹ it should
    be > 0. We sweep over all 729 D-gates (a, b, c) ∈ (Z/9)³ with signs
    fixed to (+1,+1,+1) and check the distribution.

    EP Theorem 2.7(2) implies that the C-stabilizer of Hv_0 = C ∩ HCH⁻¹
    has size 18², so among the 1944 C₀ elements, exactly 18² = 324 stabilize
    Hv_0; the remaining 1944 - 324 = 1620 do not. Restricted to the 9³ = 729
    plain D-gates with all-plus signs, we expect SOME positive-level cases.
    """
    H = H_matrix()
    levels = []
    for a in range(9):
        for b in range(9):
            for c in range(9):
                D = D_matrix(a, b, c)
                # Skip D = I.
                if a == 0 and b == 0 and c == 0:
                    continue
                HDH = matmul(H, matmul(D, H))
                lev = level(HDH)
                levels.append(lev)
                assert lev >= 0 and lev % 2 == 0 and lev <= 12, (
                    f"ℓ(H · D({a},{b},{c}) · H) = {lev} out of valid range"
                )

    # Distribution check.
    from collections import Counter
    dist = Counter(levels)
    print(f"  INFO  level distribution of ℓ(H·D·H) over 9³−1 = {len(levels)} D's:")
    for lev in sorted(dist):
        print(f"           ℓ = {lev:2d}: {dist[lev]:4d}")
    # We expect both 0 and 12 to appear (some D's stabilize Hv_0, some don't).
    assert 0 in dist, "expected some D with ℓ(HDH) = 0"
    print("  PASS  test_level_at_distance_12")


def _test_matrix_unitarity_check():
    """Numerical check that H, S, T, D-gates are all unitary."""
    for name, mat in [
        ("H", H_matrix()),
        ("S", S_matrix()),
        ("T", T_matrix()),
        ("D(1,2,3)", D_matrix(1, 2, 3)),
        ("D(0,0,0,sgn(-1,1,-1))", D_matrix(0, 0, 0, signs=(-1, 1, -1))),
        ("I", I_matrix()),
    ]:
        assert is_unitary(mat), f"{name} should be unitary"
    print("  PASS  test_matrix_unitarity_check")


def _test_det_is_unit():
    """For g ∈ Γ ⊂ U(3), det(g) should be a unit in Z[ξ, 1/3] — specifically
    a 9th root of unity times ±1."""
    for name, mat in [
        ("H", H_matrix()),
        ("S", S_matrix()),
        ("T", T_matrix()),
        ("D(1,2,3)", D_matrix(1, 2, 3)),
    ]:
        d = det3(mat)
        op = ord_pi(d)
        assert op == 0, (
            f"det({name}) should have ord_π = 0 (a unit), got {op}"
        )
        # Check |det| = 1 numerically.
        dc = d.to_complex()
        assert abs(abs(dc) - 1.0) < 1e-9, (
            f"|det({name})| should be 1, got {abs(dc)}"
        )
    print("  PASS  test_det_is_unit")


def run_tests() -> None:
    print("Running ep_level unit tests...")
    _test_ring_basics()
    _test_z9frac_basics()
    _test_galois()
    _test_ord_pi()
    _test_level_on_gates()
    _test_level_multiplicative_on_unitary()
    _test_level_at_distance_12()
    _test_matrix_unitarity_check()
    _test_det_is_unit()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
