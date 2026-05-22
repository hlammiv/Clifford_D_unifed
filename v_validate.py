#!/usr/bin/env python3
# v_validate.py — independent verifier for compiler outputs.
#
# Reads a JSON file (full unified schema OR a minimal {f, V[3][3][6], theta,
# epsilon} stub) and answers, using EXACT integer arithmetic in Z[ζ_9]:
#
#   - is_unitary:    V·V† == I   (exact, no floating-point comparison)
#   - actual_sde_3:  the smallest k such that V·3^k has integer entries
#   - frobenius:     ‖V − R^Z(θ)‖_F  (target is irrational; this part is FP)
#   - epsilon_pass:  frobenius < epsilon
#
# Usage: ./v_validate.py < input.json
#        ./v_validate.py input.json
#
# Exit codes: 0 = all checks pass; 1 = any check failed (still emits JSON).

import json
import math
import sys
import cmath


# ============ Z[ζ_9] arithmetic in canonical 6-coef basis (1, ζ, ζ², ζ³, ζ⁴, ζ⁵) ============
# Φ_9(ζ) = ζ⁶ + ζ³ + 1 = 0  ⇒  ζ⁶ = -ζ³ - 1
# Multiplication of x = Σ a_i ζ^i (i=0..5) by another y reduces ζ^k for k≥6
# via ζ^6 = -ζ^3 - 1, ζ^7 = ζ * ζ^6 = -ζ^4 - ζ, ..., ζ^10 = ζ.

def ringZ9_zero():
    return [0, 0, 0, 0, 0, 0]


def ringZ9_one():
    return [1, 0, 0, 0, 0, 0]


def ringZ9_neg(a):
    return [-x for x in a]


def ringZ9_add(a, b):
    return [a[i] + b[i] for i in range(6)]


def ringZ9_sub(a, b):
    return [a[i] - b[i] for i in range(6)]


def ringZ9_mul(a, b):
    """Multiply two ringZ9 elements; reduce via ζ^6 = -ζ^3 - 1."""
    # Compute raw product as length-11 polynomial.
    p = [0] * 11
    for i in range(6):
        if a[i] == 0:
            continue
        for j in range(6):
            p[i + j] += a[i] * b[j]
    # Reduce: ζ^k for k in {6, 7, 8, 9, 10}
    #   ζ^6  = -ζ^3 - 1
    #   ζ^7  = ζ*ζ^6  = -ζ^4 - ζ
    #   ζ^8  = ζ²*ζ^6 = -ζ^5 - ζ²
    #   ζ^9  = ζ^3*ζ^6 = -ζ^6 - ζ^3 = ζ^3 + 1 - ζ^3 = 1
    #   ζ^10 = ζ * ζ^9 = ζ
    # Apply highest-degree first.
    # ζ^10 → ζ
    if p[10]:
        p[1] += p[10]
        p[10] = 0
    # ζ^9 → 1
    if p[9]:
        p[0] += p[9]
        p[9] = 0
    # ζ^8 → -ζ^5 - ζ^2
    if p[8]:
        p[5] -= p[8]
        p[2] -= p[8]
        p[8] = 0
    # ζ^7 → -ζ^4 - ζ
    if p[7]:
        p[4] -= p[7]
        p[1] -= p[7]
        p[7] = 0
    # ζ^6 → -ζ^3 - 1
    if p[6]:
        p[3] -= p[6]
        p[0] -= p[6]
        p[6] = 0
    return p[:6]


def ringZ9_conj(a):
    """Complex conjugation in Z[ζ_9]: ζ → ζ^{-1} = ζ^8.
    Re-express ζ^{-i} = ζ^{9-i} (mod 9), then reduce.
    """
    # x̄ = a_0 + a_1·ζ^{-1} + a_2·ζ^{-2} + ... + a_5·ζ^{-5}
    #   = a_0 + a_1·ζ^8 + a_2·ζ^7 + a_3·ζ^6 + a_4·ζ^5 + a_5·ζ^4
    p = [0] * 11
    p[0] = a[0]
    p[8] = a[1]
    p[7] = a[2]
    p[6] = a[3]
    p[5] = a[4]
    p[4] = a[5]
    # Reduce as in mul.
    if p[8]:
        p[5] -= p[8]
        p[2] -= p[8]
        p[8] = 0
    if p[7]:
        p[4] -= p[7]
        p[1] -= p[7]
        p[7] = 0
    if p[6]:
        p[3] -= p[6]
        p[0] -= p[6]
        p[6] = 0
    return p[:6]


def ringZ9_eq(a, b):
    return all(a[i] == b[i] for i in range(6))


def ringZ9_to_complex(a):
    """Evaluate at the principal embedding ζ → e^(2πi/9)."""
    z = cmath.exp(2j * math.pi / 9)
    s = 0
    for i in range(6):
        if a[i]:
            s += a[i] * (z ** i)
    return s


# ============ ringZ9chi: numerator + integer denom-exponent ============

def chi_neg(a, f):
    return ringZ9_neg(a), f


def chi_add(a_f, b_f):
    a, fa = a_f
    b, fb = b_f
    if fa == fb:
        return ringZ9_add(a, b), fa
    if fa < fb:
        # scale a up by 3^(fb - fa)
        d = fb - fa
        scaled = [c * (3 ** d) for c in a]
        return ringZ9_add(scaled, b), fb
    else:
        d = fa - fb
        scaled = [c * (3 ** d) for c in b]
        return ringZ9_add(a, scaled), fa


def chi_mul(a_f, b_f):
    a, fa = a_f
    b, fb = b_f
    return ringZ9_mul(a, b), fa + fb


def chi_conj(a_f):
    a, f = a_f
    return ringZ9_conj(a), f


def chi_eq_one(a_f):
    """Test (a/3^f) == 1 in Q(ζ_9)."""
    a, f = a_f
    if f == 0:
        return ringZ9_eq(a, ringZ9_one())
    # a == 3^f → a[0] = 3^f, others = 0
    target = [3 ** f, 0, 0, 0, 0, 0]
    return ringZ9_eq(a, target)


def chi_eq_zero(a_f):
    a, f = a_f
    return all(x == 0 for x in a)


def chi_to_complex(a_f):
    a, f = a_f
    return ringZ9_to_complex(a) / (3 ** f)


# ============ Validation ============

def actual_sde_3(V_int, f):
    """Find minimum k s.t. V * 3^k has integer entries.
    V_int is the 3x3 numerator at denom 3^f. So V * 3^f = V_int (already integer).
    The "actual sde_3" is the smallest k such that V_int / 3^(f-k) is still in
    Z[ζ_9], i.e., V_int has all entries divisible by 3^(f-k).

    Equivalently: count how many factors of 3 are common to all entries of V_int,
    and subtract from f.
    """
    # Count max k_common s.t. all coefs of all entries are divisible by 3^k_common.
    k_common = f  # upper bound
    for i in range(3):
        for j in range(3):
            for c in range(6):
                v = V_int[i][j][c]
                if v == 0:
                    continue
                # Count factors of 3 in v.
                cnt = 0
                while v % 3 == 0:
                    v //= 3
                    cnt += 1
                if cnt < k_common:
                    k_common = cnt
                if k_common == 0:
                    return f  # can't reduce
    return f - k_common


def is_unitary_exact(V_chi):
    """V is 3x3 of ringZ9chi.  Compute V·V† and compare to I in EXACT arithmetic."""
    Vd = [[chi_conj(V_chi[j][i]) for j in range(3)] for i in range(3)]  # V† = (V_ji)*
    P = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            acc = (ringZ9_zero(), 0)
            for k in range(3):
                acc = chi_add(acc, chi_mul(V_chi[i][k], Vd[k][j]))
            P[i][j] = acc
    # P should equal identity.
    for i in range(3):
        for j in range(3):
            if i == j:
                if not chi_eq_one(P[i][j]):
                    return False, P[i][j]
            else:
                if not chi_eq_zero(P[i][j]):
                    return False, P[i][j]
    return True, None


def frobenius(V_chi, theta):
    """Frobenius distance to target R^Z(θ) = diag(e^{-iθ/2}, e^{+iθ/2}, 1) in floats."""
    target = [
        [cmath.exp(-0.5j * theta), 0, 0],
        [0, cmath.exp( 0.5j * theta), 0],
        [0, 0, 1],
    ]
    s = 0.0
    for i in range(3):
        for j in range(3):
            v = chi_to_complex(V_chi[i][j])
            d = v - target[i][j]
            s += abs(d) ** 2
    return math.sqrt(s)


def main():
    # Load JSON
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fh:
            data = json.load(fh)
    else:
        data = json.load(sys.stdin)

    # Accept either full schema or minimal stub.
    if "unitary" in data and "V" in data["unitary"]:
        f = data["unitary"]["f"]
        V_int = data["unitary"]["V"]
        theta = data["inputs"]["theta"]
        epsilon = data["inputs"]["epsilon"]
        backend = data.get("identification", {}).get("backend", "?")
    else:
        f = data["f"]
        V_int = data["V"]
        theta = data["theta"]
        epsilon = data.get("epsilon", None)
        backend = data.get("backend", "?")

    # Build V as 3x3 of ringZ9chi (each entry: numerator + denom-exp f).
    V_chi = [[(list(V_int[i][j]), f) for j in range(3)] for i in range(3)]

    # Check 1: unitarity (exact).
    is_uni, mismatch = is_unitary_exact(V_chi)

    # Check 2: actual sde_3.
    asde3 = actual_sde_3(V_int, f)

    # Check 3: Frobenius distance.
    frob = frobenius(V_chi, theta)
    eps_pass = (epsilon is not None and frob <= epsilon)

    out = {
        "backend": backend,
        "f_input": f,
        "actual_sde_3": asde3,
        "is_unitary": is_uni,
        "frobenius": frob,
        "epsilon": epsilon,
        "epsilon_pass": bool(eps_pass) if epsilon is not None else None,
        "all_checks_pass": is_uni and (eps_pass if epsilon is not None else True),
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.exit(0 if out["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
