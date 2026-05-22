import math
import cmath
from typing import Optional, Tuple, Dict, Any
import numpy as np


# ============================================================
#  Arithmetic in Z[zeta_9] with basis 1, z, z^2, z^3, z^4, z^5
#  and relation z^6 + z^3 + 1 = 0   (Phi_9(z) = 0)
# ============================================================

DEG = 6


def _reduce_coeffs(coeffs):
    """
    Reduce a polynomial coeff list modulo z^6 + z^3 + 1.
    Input may have length > 6. Output length = 6, integer coefficients.
    """
    c = list(coeffs)
    if len(c) < DEG:
        c += [0] * (DEG - len(c))

    while len(c) > DEG:
        k = len(c) - 1
        v = c.pop()
        if v == 0:
            continue

        # z^k = z^(k-6) * z^6 = - z^(k-3) - z^(k-6)
        if k - 3 >= len(c):
            c += [0] * (k - 3 - len(c) + 1)
        if k - 6 >= len(c):
            c += [0] * (k - 6 - len(c) + 1)

        c[k - 3] -= v
        c[k - 6] -= v

    if len(c) < DEG:
        c += [0] * (DEG - len(c))
    return tuple(int(x) for x in c[:DEG])


def _monomial(exp):
    """
    Return z^exp reduced to the basis.
    exp must be a nonnegative integer.
    """
    coeffs = [0] * (exp + 1)
    coeffs[exp] = 1
    return _reduce_coeffs(coeffs)


# Precompute basis multiplication table
_BASIS_MUL = [[None] * DEG for _ in range(DEG)]
for i in range(DEG):
    for j in range(DEG):
        _BASIS_MUL[i][j] = _monomial(i + j)

# Precompute conjugates of basis elements:
#   conj(z^k) = z^{-k} = z^(9-k), then reduce
_BASIS_CONJ = [None] * DEG
_BASIS_CONJ[0] = (1, 0, 0, 0, 0, 0)
for k in range(1, DEG):
    _BASIS_CONJ[k] = _monomial(9 - k)


def add(x, y):
    return tuple(x[i] + y[i] for i in range(DEG))


def sub(x, y):
    return tuple(x[i] - y[i] for i in range(DEG))


def scale(m, x):
    return tuple(m * x[i] for i in range(DEG))


def conj(x):
    out = [0] * DEG
    for i, xi in enumerate(x):
        if xi == 0:
            continue
        ci = _BASIS_CONJ[i]
        for j in range(DEG):
            out[j] += xi * ci[j]
    return tuple(out)


def mul(x, y):
    out = [0] * DEG
    for i, xi in enumerate(x):
        if xi == 0:
            continue
        for j, yj in enumerate(y):
            if yj == 0:
                continue
            mij = _BASIS_MUL[i][j]
            a = xi * yj
            for k in range(DEG):
                out[k] += a * mij[k]
    return tuple(out)


def norm_to_real_subfield(x):
    """
    Return x * conj(x) in the same 6-coefficient basis.
    For valid x this lies in the real cubic subfield, but we keep the full basis.
    """
    return mul(x, conj(x))


# ============================================================
#  Numerical embeddings (for obstruction checks / diagnostics)
# ============================================================

def embed(x, r):
    """
    Complex embedding sigma_r with zeta_9 -> exp(2*pi*i*r/9),
    where r in {1,2,4}. These represent the three pairs of complex embeddings.
    """
    w = cmath.exp(2j * math.pi * r / 9.0)
    s = 0j
    wk = 1.0 + 0.0j
    for a in x:
        s += a * wk
        wk *= w
    return s


def embedding_targets(M):
    """
    Compute sigma_r(M) for r=1,2,4.
    Since M should be in the real subfield, these should be real.
    """
    vals = []
    for r in (1, 2, 4):
        z = embed(M, r)
        vals.append(z.real)
    return tuple(vals)


def coeffs_to_complex(x_coeffs, f):
    return embed(tuple(int(v) for v in x_coeffs), 1) / (3 ** f)


def vector_coeffs_to_complex(vec_coeffs, f):
    return np.array([coeffs_to_complex(v, f) for v in vec_coeffs], dtype=np.complex128)


# ============================================================
#  Real cubic field basis conversion
#  alpha = zeta + zeta^{-1}
#
#  In the reduced 6-basis:
#    alpha   = z + z^8 = z - z^2 - z^5 = (0,1,-1,0,0,-1)
#    alpha^2 = (2,-1,1,0,-1,0)
#
#  So:
#    m0 + m1*alpha + m2*alpha^2
#      = (m0 + 2*m2,
#         m1 - m2,
#        -m1 + m2,
#         0,
#        -m2,
#        -m1)
# ============================================================

def real6_to_m012(M6):
    """
    Convert an element of the real subfield from the reduced 6-basis
    (1, z, z^2, z^3, z^4, z^5)
    to coefficients (m0,m1,m2) in basis (1, alpha, alpha^2).
    """
    c0, c1, c2, c3, c4, c5 = M6

    if c3 != 0:
        raise ValueError(f"Element is not in the real subfield in this basis: {M6}")
    if c1 + c2 != 0:
        raise ValueError(f"Element is not in the real subfield in this basis: {M6}")
    if c1 - c4 + c5 != 0:
        raise ValueError(f"Element is not in the real subfield in this basis: {M6}")

    m2 = -c4
    m1 = -c5
    m0 = c0 - 2 * m2
    return (m0, m1, m2)


def m012_to_real6(m0, m1, m2):
    return (
        m0 + 2 * m2,
        m1 - m2,
        -m1 + m2,
        0,
        -m2,
        -m1,
    )


def trace_from_real_coeffs(m0, m1, m2):
    """
    Tr_{Q(alpha)/Q}(m0 + m1*alpha + m2*alpha^2) = 3*m0 + 6*m2
    """
    return 3 * m0 + 6 * m2


# ============================================================
#  Exact coefficient formulas for b * conj(b)
#
#  If b = sum n_k zeta^k, then
#     b*conj(b) = Q0 + Q1*alpha + Q2*alpha^2
# ============================================================

def bb_to_real_coeffs(n):
    n0, n1, n2, n3, n4, n5 = n

    A1 = n0 * n1 + n1 * n2 + n2 * n3 + n3 * n4 + n4 * n5
    A2 = n0 * n2 + n1 * n3 + n2 * n4 + n3 * n5
    A8 = n0 * n4 + n1 * n5 + n0 * n5
    E  = n0 * n3 + n1 * n4 + n2 * n5

    Q0 = (n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3 + n4 * n4 + n5 * n5) - E - 2 * A2 + 2 * A8
    Q1 = A1 - A8
    Q2 = A2 - A8
    return (Q0, Q1, Q2)


def verify_b_matches_M012(n, m0, m1, m2):
    return bb_to_real_coeffs(n) == (m0, m1, m2)


# ============================================================
#  Linear solve for (n0, n3) from Q1=m1, Q2=m2
# ============================================================

def _system_coeffs_for_n0_n3(n1, n2, n4, n5, m1, m2):
    a11 = n1 - n4 - n5
    a12 = n2 + n4
    a21 = n2 - n4 - n5
    a22 = n1 + n5

    rhs1 = m1 - (n1 * n2 + n4 * n5 - n1 * n5)
    rhs2 = m2 - (n2 * n4 - n1 * n5)

    return a11, a12, a21, a22, rhs1, rhs2


def _bounded_solutions_of_one_linear_equation(a, b, c, B):
    """
    Yield all integer solutions (x,y) with |x|<=B, |y|<=B of
        a*x + b*y = c
    """
    if a == 0 and b == 0:
        if c != 0:
            return
        for x in range(-B, B + 1):
            for y in range(-B, B + 1):
                yield (x, y)
        return

    if a == 0:
        if c % b != 0:
            return
        y = c // b
        if -B <= y <= B:
            for x in range(-B, B + 1):
                yield (x, y)
        return

    if b == 0:
        if c % a != 0:
            return
        x = c // a
        if -B <= x <= B:
            for y in range(-B, B + 1):
                yield (x, y)
        return

    for x in range(-B, B + 1):
        rem = c - a * x
        if rem % b != 0:
            continue
        y = rem // b
        if -B <= y <= B:
            yield (x, y)


def solve_n0_n3_all(n1, n2, n4, n5, m1, m2, B):
    """
    Exhaustively yield all integer solutions (n0,n3) with |n0|,|n3|<=B
    of the system Q1=m1, Q2=m2 for fixed n1,n2,n4,n5.
    """
    a11, a12, a21, a22, rhs1, rhs2 = _system_coeffs_for_n0_n3(n1, n2, n4, n5, m1, m2)

    det = a11 * a22 - a12 * a21

    if det != 0:
        num0 = rhs1 * a22 - a12 * rhs2
        num3 = a11 * rhs2 - rhs1 * a21
        if num0 % det != 0 or num3 % det != 0:
            return
        n0 = num0 // det
        n3 = num3 // det
        if -B <= n0 <= B and -B <= n3 <= B:
            yield (n0, n3)
        return

    # Rank-deficient case
    if a11 * rhs2 != a21 * rhs1 or a12 * rhs2 != a22 * rhs1:
        return

    if a11 != 0 or a12 != 0:
        yield from _bounded_solutions_of_one_linear_equation(a11, a12, rhs1, B)
    elif a21 != 0 or a22 != 0:
        yield from _bounded_solutions_of_one_linear_equation(a21, a22, rhs2, B)
    else:
        for n0 in range(-B, B + 1):
            for n3 in range(-B, B + 1):
                yield (n0, n3)


# ============================================================
#  Main 4D search for b from M = m0 + m1*alpha + m2*alpha^2
# ============================================================

def find_b_from_M_coeffs_4d(
    m0: int,
    m1: int,
    m2: int,
    *,
    verbose: bool = False
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Find one integer vector n=(n0,...,n5) such that
        b*conj(b) = m0 + m1*alpha + m2*alpha^2.
    Returns one solution or None.
    """
    trM = trace_from_real_coeffs(m0, m1, m2)
    if trM < 0:
        return None

    # From lambda_min(G)=3/2:
    #   sum n_k^2 <= (2/3) Tr(M) = 2*m0 + 4*m2
    B2 = 2 * m0 + 4 * m2
    if B2 < 0:
        return None
    B = int(math.isqrt(B2))

    if verbose:
        print(f"(m0,m1,m2)=({m0},{m1},{m2}), Tr(M)={trM}, B2={B2}, B={B}")

    checked_4d = 0
    checked_linear = 0

    for n1 in range(-B, B + 1):
        for n2 in range(-B, B + 1):
            for n4 in range(-B, B + 1):
                for n5 in range(-B, B + 1):
                    checked_4d += 1

                    for n0, n3 in solve_n0_n3_all(n1, n2, n4, n5, m1, m2, B):
                        checked_linear += 1
                        n = (n0, n1, n2, n3, n4, n5)

                        if sum(x * x for x in n) > B2:
                            continue

                        if verify_b_matches_M012(n, m0, m1, m2):
                            if verbose:
                                print(f"checked 4D tuples     : {checked_4d}")
                                print(f"checked (n0,n3) pairs : {checked_linear}")
                            return n

    if verbose:
        print(f"checked 4D tuples     : {checked_4d}")
        print(f"checked (n0,n3) pairs : {checked_linear}")
    return None


# ============================================================
#  Fixed-layer search: c = b / 3^f
# ============================================================

def find_one_completion_c_via_4d(
    d_num,
    f,
    *,
    positivity_tol: float = 1e-12,
    verbose: bool = False
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Same-layer search only: find c = b / 3^f such that
        c*conj(c) = 1 - d*conj(d)
    """
    a = tuple(int(x) for x in d_num)
    aa = norm_to_real_subfield(a)
    M6 = sub((3 ** (2 * f), 0, 0, 0, 0, 0), aa)

    if verbose:
        print("a       =", a)
        print("a*a_bar =", aa)
        print("M6      =", M6)

    emb = embedding_targets(M6)
    if verbose:
        print("sigma(M) =", emb)

    if emb[0] < -positivity_tol or emb[1] < -positivity_tol or emb[2] < -positivity_tol:
        return None

    if M6 == (0, 0, 0, 0, 0, 0):
        return (0, 0, 0, 0, 0, 0)

    m0, m1, m2 = real6_to_m012(M6)
    if verbose:
        print("(m0,m1,m2) =", (m0, m1, m2))

    return find_b_from_M_coeffs_4d(m0, m1, m2, verbose=verbose)


# ============================================================
#  Deeper-layer search:
#     d = a / 3^f is fixed
#     search c = b / 3^f2 for f2 >= f
#
#  Need:
#     b*conj(b) = 3^(2(f2-f)) * M_f
#  where
#     M_f = 3^(2f) - a*conj(a)
# ============================================================

def search_completion_up_to_depth(
    d_num,
    f,
    f2_max,
    *,
    positivity_tol: float = 1e-12,
    verbose: bool = False,
    embeddings_only: bool = False
) -> Dict[str, Any]:
    """
    If embeddings_only=True:
        do only the cheap necessary test that all 3 real embeddings of
        D = 1 - d*conj(d) are nonnegative.

    Otherwise:
        search for a completion c=b/3^f2 for some f2 in [f, f2_max].

    Returns a dictionary with fields:
      status:
        - "found"
        - "not_a_norm_negative_embedding"
        - "not_found_up_to_max"
      plus diagnostics.

    Important:
      - If status == "not_a_norm_negative_embedding", this is a proof that no
        deeper layer can work.
      - If status == "not_found_up_to_max", this is NOT a proof of impossibility,
        only that no solution was found up to f2_max.
    """
    a = tuple(int(x) for x in d_num)
    aa = norm_to_real_subfield(a)
    M_base = sub((3 ** (2 * f), 0, 0, 0, 0, 0), aa)
    emb_base = embedding_targets(M_base)

    if embeddings_only:
        ok = (
            emb_base[0] >= -positivity_tol and
            emb_base[1] >= -positivity_tol and
            emb_base[2] >= -positivity_tol
        )
        return {
            "status": "embedding_check_only",
            "d_num": a,
            "f": f,
            "M_base": M_base,
            "sigma_M_base": emb_base,
            "D_is_totally_nonnegative": ok,
            "note": "Necessary condition only; not a proof that D is a norm.",
        }

    if f2_max is None:
        raise ValueError("f2_max must be provided unless embeddings_only=True")
    if f2_max < f:
        raise ValueError("f2_max must satisfy f2_max >= f")

    a = tuple(int(x) for x in d_num)
    aa = norm_to_real_subfield(a)
    M_base = sub((3 ** (2 * f), 0, 0, 0, 0, 0), aa)

    emb_base = embedding_targets(M_base)

    if verbose:
        print("a         =", a)
        print("a*a_bar   =", aa)
        print("M_base    =", M_base)
        print("sigma(M_base) =", emb_base)

    # Definite obstruction: if any embedding of D is negative, no deeper layer helps.
    if emb_base[0] < -positivity_tol or emb_base[1] < -positivity_tol or emb_base[2] < -positivity_tol:
        return {
            "status": "not_a_norm_negative_embedding",
            "reason": "A real embedding of D = 1 - d*conj(d) is negative, so D cannot be a norm in any deeper 3-adic layer.",
            "d_num": a,
            "f": f,
            "M_base": M_base,
            "sigma_M_base": emb_base,
        }

    # Trivial case
    if M_base == (0, 0, 0, 0, 0, 0):
        return {
            "status": "found",
            "reason": "Trivial completion with c = 0.",
            "d_num": a,
            "f": f,
            "f2": f,
            "c_num": (0, 0, 0, 0, 0, 0),
            "M_base": M_base,
            "sigma_M_base": emb_base,
        }

    levels_tried = []

    for f2 in range(f, f2_max + 1):
        s = f2 - f
        scale_factor = 3 ** (2 * s)
        M6 = scale(scale_factor, M_base)

        # Embeddings remain nonnegative if base was nonnegative,
        # but recompute in case of debugging.
        emb = embedding_targets(M6)

        if verbose:
            print()
            print(f"Trying deeper layer f2 = {f2}")
            print("scale_factor =", scale_factor)
            print("M6 =", M6)
            print("sigma(M6) =", emb)

        try:
            m0, m1, m2 = real6_to_m012(M6)
        except ValueError as e:
            # This should not happen if M_base came from a*a_bar, but keep it explicit.
            levels_tried.append({
                "f2": f2,
                "status": "conversion_error",
                "error": str(e),
            })
            continue

        c_num = find_b_from_M_coeffs_4d(m0, m1, m2, verbose=verbose)

        levels_tried.append({
            "f2": f2,
            "status": "found" if c_num is not None else "not_found",
            "M6": M6,
            "sigma_M6": emb,
            "m012": (m0, m1, m2),
        })

        if c_num is not None:
            return {
                "status": "found",
                "d_num": a,
                "f": f,
                "f2": f2,
                "c_num": c_num,
                "M_base": M_base,
                "sigma_M_base": emb_base,
                "levels_tried": levels_tried,
            }

    return {
        "status": "not_found_up_to_max",
        "reason": "No completion found up to the requested maximum deeper layer. This is not a proof that D is not a norm.",
        "d_num": a,
        "f": f,
        "f2_max": f2_max,
        "M_base": M_base,
        "sigma_M_base": emb_base,
        "levels_tried": levels_tried,
    }


# ============================================================
#  Verification helpers
# ============================================================

def verify_completion_at_layer(d_num, f, c_num, f2):
    """
    Exact check that
        (c_num / 3^f2) * conj(c_num / 3^f2) = 1 - (d_num / 3^f)*(...)
    i.e.
        c_num*conj(c_num) = 3^(2(f2-f)) * (3^(2f) - a*conj(a)).
    """
    a = tuple(int(x) for x in d_num)
    b = tuple(int(x) for x in c_num)

    aa = norm_to_real_subfield(a)
    bb = norm_to_real_subfield(b)

    M_base = sub((3 ** (2 * f), 0, 0, 0, 0, 0), aa)
    target = scale(3 ** (2 * (f2 - f)), M_base)

    return bb == target


def coeffs_to_string(x, var="z"):
    parts = []
    for k, a in enumerate(x):
        if a == 0:
            continue
        if k == 0:
            parts.append(str(a))
        elif k == 1:
            parts.append(f"{a}*{var}")
        else:
            parts.append(f"{a}*{var}^{k}")
    return "0" if not parts else " + ".join(parts)


def check_D_embeddings_only(d_num, f, *, positivity_tol: float = 1e-12):
    """
    Lightweight necessary test for
        D = 1 - d*conj(d),   d = a / 3^f.

    Returns a dictionary with:
      - D_is_totally_nonnegative: whether all 3 real embeddings are >= 0
      - this is necessary but NOT sufficient for D to be a norm
    """
    a = tuple(int(x) for x in d_num)
    aa = norm_to_real_subfield(a)
    M_base = sub((3 ** (2 * f), 0, 0, 0, 0, 0), aa)

    sigma = embedding_targets(M_base)

    ok = (
        sigma[0] >= -positivity_tol and
        sigma[1] >= -positivity_tol and
        sigma[2] >= -positivity_tol
    )

    return {
        "d_num": a,
        "f": f,
        "M_base": M_base,
        "sigma_M_base": sigma,
        "D_is_totally_nonnegative": ok,
        "note": "This is a necessary condition for D to be a norm, not a sufficient one.",
    }


# ============================================================
#  Example usage
# ============================================================

if __name__ == "__main__":
    # Example:
    f = 2
    d_num = (0, 3, -3, -1, -3, -4)
    f2_max = 5

    result = search_completion_up_to_depth(d_num, f, f2_max, verbose=True)
    print()
    print("RESULT")
    for k, v in result.items():
        print(f"{k}: {v}")

    if result["status"] == "found":
        ok = verify_completion_at_layer(d_num, f, result["c_num"], result["f2"])
        print("verified:", ok)
        print("c =", coeffs_to_string(result["c_num"]), f"/3^{result['f2']}")
