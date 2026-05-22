"""
Fast (Numba-based) replacement for the ideal screen at the heart of stage 1.

The original screen lives in `roots.py`:
  quick_screen_M_fast(m0, m1, m2, ...) -> dict
which calls into Sage/PARI for principal_ideal_factorization. At f=6 that's
~6 ms/row, throttling the build.

This module re-implements the screen as a Numba kernel:

    screen_rows_batch(rows, deg1_table, deg1_starts, deg1_lens, deg1_pmax) -> keep_mask
    screen_rows_python(rows, ...) -> keep_mask  (with Sage fallback for large primes)

The math (F = Q(α), α³ = 3α - 1, K = Q(ζ_9), so K = F(ζ_9), [K:F] = 2):
  M = m0 + m1*α + m2*α²  in O_F
  REJECT iff any prime P of F that is INERT in K/F has odd v_P(M).

Primes of F above rational p (p ≠ 3):
  p mod 9 ∈ {1}             : 3 primes deg 1, ALL split in K/F  → no constraint
  p mod 9 ∈ {4, 7}          : 1 prime  deg 3, splits in K/F     → no constraint
  p mod 9 ∈ {2, 5}          : 1 prime  deg 3, INERT in K/F      → π = p, v_P(M) = v_p(N_F/Q(M))/3
  p mod 9 ∈ {8}             : 3 primes deg 1, ALL inert in K/F  → need π = a+bα+cα² per prime

For p mod 9 == 8 we use the precomputed (p, a, b, c) generators table built by
build_deg1_inert_table.py (Sage script).

For p ≡ 2/5 mod 9: trivial (the rational p generates the unique F-prime; v_P(M)
is just v_p(N(M))/3 since N(P) = p^3).

CAVEAT: if N_F/Q(M) has a rational-prime factor p > BOUND with p mod 9 == 8,
we don't have its generators precomputed. We fall back to Sage for those rows
(rare for typical magnitudes; can rebuild table with larger BOUND if needed).
"""
import os
import math
import time
import numpy as np
import numba as nb
from numba.extending import intrinsic
from numba.core import types as _nb_types
from llvmlite import ir as _llvm_ir


# ---------------------------------------------------------------------------
# umulh intrinsic: high 64 bits of unsigned 64x64 -> 128 multiply.
# Compiles to a single MULQ on x86-64; gives ~30× speedup on _mulmod when
# combined with Granlund-Möller modular reduction below.
# ---------------------------------------------------------------------------

@intrinsic
def _umulh(typingctx, a, b):
    sig = _nb_types.uint64(_nb_types.uint64, _nb_types.uint64)
    def codegen(context, builder, signature, args):
        i64 = _llvm_ir.IntType(64)
        i128 = _llvm_ir.IntType(128)
        a128 = builder.zext(args[0], i128)
        b128 = builder.zext(args[1], i128)
        prod = builder.mul(a128, b128)
        hi = builder.lshr(prod, _llvm_ir.Constant(i128, 64))
        return builder.trunc(hi, i64)
    return sig, codegen


# ---------------------------------------------------------------------------
# Math constants
# ---------------------------------------------------------------------------

# Real conjugates of α = 2 cos(2π/9). Other two roots of x³ - 3x + 1.
_ALPHA1 = 2.0 * math.cos(2.0 * math.pi / 9.0)
_ALPHA2 = 2.0 * math.cos(4.0 * math.pi / 9.0)
_ALPHA4 = 2.0 * math.cos(8.0 * math.pi / 9.0)
_ALPHA1_SQ = _ALPHA1 * _ALPHA1
_ALPHA2_SQ = _ALPHA2 * _ALPHA2
_ALPHA4_SQ = _ALPHA4 * _ALPHA4


# ---------------------------------------------------------------------------
# Load the inert-deg-1 generator table at import time
# ---------------------------------------------------------------------------
_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_deg1_inert_table.npy")


def _load_deg1_table():
    """Returns (table, starts, lens, pmax).

    table  : (N, 4) int64 array of (p, a, b, c) tuples sorted by p, contiguous
             within each p (3 rows per p in F).
    starts : (P+1,) int64; starts[i] = first table row for prime index i
    lens   : (P,) int64;   lens[i]   = number of generators for prime index i (always 3)
    plist  : (P,) int64;   plist[i]  = the prime
    pmax   : largest prime in the table
    """
    if not os.path.exists(_TABLE_PATH):
        return None, None, None, None, 0
    t = np.load(_TABLE_PATH)
    # group by p
    plist = []
    starts = []
    lens = []
    i = 0
    n = t.shape[0]
    while i < n:
        p = int(t[i, 0])
        j = i
        while j < n and int(t[j, 0]) == p:
            j += 1
        starts.append(i)
        lens.append(j - i)
        plist.append(p)
        i = j
    starts.append(n)  # sentinel
    return (
        np.ascontiguousarray(t, dtype=np.int64),
        np.array(starts, dtype=np.int64),
        np.array(lens, dtype=np.int64),
        np.array(plist, dtype=np.int64),
        int(plist[-1]) if plist else 0,
    )


_DEG1_TABLE, _DEG1_STARTS, _DEG1_LENS, _DEG1_PLIST, _DEG1_PMAX = _load_deg1_table()


# ---------------------------------------------------------------------------
# Numba primitives
# ---------------------------------------------------------------------------

@nb.njit(cache=True, inline="always")
def _norm_FQ(m0, m1, m2):
    """N_F/Q(M) for M = m0 + m1*α + m2*α² in F = Q(α), α³ = 3α - 1.

    Derived as det(multiplication-by-M map on F).

    Closed form:
        N = m0³ + 6 m0² m2 + 9 m0 m2² - 3 m0 m1² + 3 m0 m1 m2
            - m1³ + 3 m1 m2² + m2³

    Returns int64. Uses uint64 arithmetic for deterministic 2's-complement
    wraparound on intermediates — final result is correct iff |true N| <
    2^63. Caller must pre-screen via _norm_FQ_estimate_abs.
    """
    u0 = np.uint64(m0)
    u1 = np.uint64(m1)
    u2 = np.uint64(m2)
    SIX = np.uint64(6)
    THREE = np.uint64(3)
    NINE = np.uint64(9)
    s = u0 * u0 * u0
    s = s + SIX * u0 * u0 * u2
    s = s + NINE * u0 * u2 * u2
    s = s - THREE * u0 * u1 * u1
    s = s + THREE * u0 * u1 * u2
    s = s - u1 * u1 * u1
    s = s + THREE * u1 * u2 * u2
    s = s + u2 * u2 * u2
    return np.int64(s)


@nb.njit(cache=True, inline="always")
def _norm_FQ_estimate_abs(m0, m1, m2):
    """Float64 estimate of |N(M)|.

    Used to gate the int64 path: int64 _norm_FQ is correct iff
    |true N| < 2^63; we conservatively fall back to slow path when
    the float estimate exceeds 2^62 (gives ~1 bit of margin against
    float rounding error).
    """
    f0 = float(m0)
    f1 = float(m1)
    f2 = float(m2)
    n = (f0*f0*f0 + 6.0*f0*f0*f2 + 9.0*f0*f2*f2
         - 3.0*f0*f1*f1 + 3.0*f0*f1*f2
         - f1*f1*f1 + 3.0*f1*f2*f2 + f2*f2*f2)
    return n if n >= 0.0 else -n


@nb.njit(cache=True, inline="always")
def _real_embeddings_nonneg(m0, m1, m2):
    """True iff all three real embeddings σ_r(M) ≥ 0.

    σ_r(M) = m0 + m1 α_r + m2 α_r²  for α_r ∈ {α₁, α₂, α₄}.

    M = |a|² requires σ_r(M) ≥ 0 for every real place — quick pre-rejection.
    For nonzero M ∈ O_F, all σ_r(M) are nonzero (since N(M) = ∏ σ_r ≠ 0). So
    a strict `>= 0.0` test is correct for all rows that pass the M ≠ 0 guard.
    Float64 precision (~1e-15 relative) is enough at our magnitudes; the
    worst-case row has |σ_r| ~ 10⁷, so floor precision ~10⁻⁸, well above any
    "real zero" case.
    """
    s1 = m0 + m1*_ALPHA1 + m2*_ALPHA1_SQ
    s2 = m0 + m1*_ALPHA2 + m2*_ALPHA2_SQ
    s4 = m0 + m1*_ALPHA4 + m2*_ALPHA4_SQ
    return s1 >= 0.0 and s2 >= 0.0 and s4 >= 0.0


@nb.njit(cache=True, inline="always")
def _vp_int(n, p):
    """v_p(|n|). Returns 10^9 if n == 0 (a sentinel for "infinite valuation")."""
    if n == 0:
        return 1000000000
    if n < 0:
        n = -n
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


# Small-prime table for trial division (all primes < 1024). Computed at module load.
def _build_small_primes(bound):
    sieve = bytearray(b'\x01' * (bound + 1))
    sieve[:2] = b'\x00\x00'
    for p in range(2, int(math.isqrt(bound)) + 1):
        if sieve[p]:
            for k in range(p*p, bound + 1, p):
                sieve[k] = 0
    return np.array([i for i in range(2, bound + 1) if sieve[i]], dtype=np.int64)


_SMALL_PRIMES = _build_small_primes(1023)


@nb.njit(cache=True)
def _trial_divide(n, small_primes, out_p, out_e):
    """Trial divide |n| by primes in small_primes[].
    Writes (p, e) pairs into out_p, out_e; returns (n_remainder, n_factors_found).
    n must be ≥ 1 on entry.
    """
    if n < 0:
        n = -n
    idx = 0
    for i in range(small_primes.shape[0]):
        p = small_primes[i]
        if p*p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            out_p[idx] = p
            out_e[idx] = e
            idx += 1
            if n == 1:
                break
    return n, idx


# Miller-Rabin primality test (deterministic for n < 3.3 * 10^14 with bases below)
# Bases proven by Pomerance et al.; for our int64 range we use a slightly larger set
_MR_BASES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37], dtype=np.int64)


@nb.njit(cache=True, inline="always")
def _clz_u64(x):
    """Count leading zeros in a positive uint64."""
    n = nb.uint64(0)
    if (x >> nb.uint64(32)) == nb.uint64(0):
        n += nb.uint64(32); x = x << nb.uint64(32)
    if (x >> nb.uint64(48)) == nb.uint64(0):
        n += nb.uint64(16); x = x << nb.uint64(16)
    if (x >> nb.uint64(56)) == nb.uint64(0):
        n += nb.uint64(8); x = x << nb.uint64(8)
    if (x >> nb.uint64(60)) == nb.uint64(0):
        n += nb.uint64(4); x = x << nb.uint64(4)
    if (x >> nb.uint64(62)) == nb.uint64(0):
        n += nb.uint64(2); x = x << nb.uint64(2)
    if (x >> nb.uint64(63)) == nb.uint64(0):
        n += nb.uint64(1)
    return n


@nb.njit(cache=True)
def _div_128_by_64(hi, lo, divisor):
    """Shift-subtract 128/64 division. Returns the 64-bit quotient
    floor((hi << 64 | lo) / divisor). Assumes hi < divisor. Called
    once per modulus to compute the GM inverse — 128 iterations OK."""
    rem = hi
    quot = nb.uint64(0)
    for i in range(63, -1, -1):
        bit = (lo >> nb.uint64(i)) & nb.uint64(1)
        msb = rem >> nb.uint64(63)
        rem = (rem << nb.uint64(1)) | bit
        if msb == nb.uint64(1) or rem >= divisor:
            rem = rem - divisor
            quot = (quot << nb.uint64(1)) | nb.uint64(1)
        else:
            quot = quot << nb.uint64(1)
    return quot


@nb.njit(cache=True)
def _gm_invert(m_norm):
    """Möller-Granlund inverse: v = floor((2^128 - 1) / m_norm) - 2^64.
    m_norm must be normalized (high bit set, 2^63 <= m_norm < 2^64).

    Uses the identity:
        2^128 - 1 = m_norm * 2^64 + ((2^64 - m_norm) * 2^64 - 1)
    so v = floor(((2^64 - m_norm - 1) * 2^64 + (2^64 - 1)) / m_norm).
    Dividend's high word (2^64 - m_norm - 1) < m_norm because m_norm >= 2^63.
    """
    hi = nb.uint64(0) - m_norm - nb.uint64(1)
    lo = nb.uint64(0xFFFFFFFFFFFFFFFF)
    return _div_128_by_64(hi, lo, m_norm)


@nb.njit(cache=True, inline="always")
def _gm_setup(m):
    """Return (m_norm, v, s): m_norm = m << s normalized; v = _gm_invert(m_norm)."""
    mu = nb.uint64(m)
    s = _clz_u64(mu)
    m_norm = mu << s
    v = _gm_invert(m_norm)
    return m_norm, v, s


@nb.njit(cache=True, inline="always")
def _gm_divmod(u_hi, u_lo, d, v):
    """Möller-Granlund 2-by-1 div. Requires d normalized, u_hi < d.
    Returns (q, r): (u_hi << 64 | u_lo) = q * d + r, 0 <= r < d."""
    q0 = v * u_hi
    q1 = _umulh(v, u_hi)
    q0_new = q0 + u_lo
    carry = nb.uint64(1) if q0_new < q0 else nb.uint64(0)
    q1 = q1 + u_hi + carry
    q0 = q0_new
    q1 = q1 + nb.uint64(1)
    r = u_lo - q1 * d
    if r > q0:
        q1 = q1 - nb.uint64(1)
        r = r + d
    if r >= d:
        q1 = q1 + nb.uint64(1)
        r = r - d
    return q1, r


@nb.njit(cache=True, inline="always")
def _mulmod_gm(a, b, m, m_norm, v, s):
    """Compute a*b mod m for 0 <= a, b < m. m_norm = m << s; v from _gm_invert.

    ~30× faster than the bit-by-bit peasant fallback at m ~ 2^57. Each call
    is ~3 ns vs peasant's ~80 ns (peasant scales with log2(m), GM is O(1)).
    """
    au = nb.uint64(a)
    bu = nb.uint64(b)
    lo = au * bu
    hi = _umulh(au, bu)
    s_u = nb.uint64(s)
    if s == 0:
        hi_n = hi
        lo_n = lo
    else:
        hi_n = (hi << s_u) | (lo >> (nb.uint64(64) - s_u))
        lo_n = lo << s_u
    _, r_norm = _gm_divmod(hi_n, lo_n, m_norm, v)
    if s == 0:
        return nb.int64(r_norm)
    return nb.int64(r_norm >> s_u)


# Legacy peasant kept for any caller that doesn't want GM setup. Internal
# rho/MR paths now use _mulmod_gm directly.
@nb.njit(cache=True, inline="always")
def _mulmod(a, b, m):
    if m < (1 << 31):
        return (a * b) % m
    r = 0
    a = a % m
    b = b % m
    while b > 0:
        if b & 1:
            r = r + a
            if r >= m:
                r -= m
        a = a + a
        if a >= m:
            a -= m
        b >>= 1
    return r


@nb.njit(cache=True, inline="always")
def _gcd64(a, b):
    """gcd of two non-negative int64s."""
    while b != 0:
        a, b = b, a % b
    return a


@nb.njit(cache=True)
def _powmod_gm(base, exp, m, m_norm, v, s):
    """Modular exponentiation using _mulmod_gm."""
    if m == 1:
        return 0
    r = nb.int64(1)
    base = base % m
    while exp > 0:
        if exp & 1:
            r = _mulmod_gm(r, base, m, m_norm, v, s)
        base = _mulmod_gm(base, base, m, m_norm, v, s)
        exp >>= 1
    return r


@nb.njit(cache=True)
def _is_prime_mr(n):
    """Miller-Rabin primality test, deterministic for int64 with the
    strong-pseudoprime bases used. Uses _mulmod_gm internally."""
    if n < 2:
        return False
    for i in range(_MR_BASES.shape[0]):
        if n == _MR_BASES[i]:
            return True
        if n % _MR_BASES[i] == 0:
            return False
    m_norm, v, s = _gm_setup(n)
    d = n - 1
    sb = 0
    while (d & 1) == 0:
        d >>= 1
        sb += 1
    for i in range(_MR_BASES.shape[0]):
        a = _MR_BASES[i]
        if a >= n:
            continue
        x = _powmod_gm(a, d, n, m_norm, v, s)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(sb - 1):
            x = _mulmod_gm(x, x, n, m_norm, v, s)
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


@nb.njit(cache=True)
def _pollard_rho(n):
    """Brent's variant of Pollard-rho with batched GCD and Möller-Granlund
    fast modular multiplication. ~30× faster mulmod than the previous
    peasant impl on modulus n in the 2^40 .. 2^62 range typical at f=6.
    Returns a non-trivial factor of n, or 0 if all c attempts failed."""
    if (n & 1) == 0:
        return 2

    m_norm, v, s = _gm_setup(n)
    BATCH = 128
    # MAX_R caps iterations per c value. Sweet spot found empirically:
    #   2^17 = too high (slow tail eats wall on 0.1% pathological rows)
    #   2^12 = too low (legitimate rho on medium semiprimes fails too often)
    #   2^15 = good (caps at ~2ms worst case, lets most rho succeed)
    # Rows that exhaust the cap fall back to FLINT (~20 µs/row) via
    # screen_rows_batch's fallback path.
    MAX_R = 1 << 15

    for c in range(1, 5):
        # Reseed y per c-attempt: pure-c-variation re-walks similar trajectories
        # on pathological N. Hashing c into y decorrelates retries.
        y = nb.int64(2 + 17 * c)
        x = nb.int64(2 + 17 * c)
        ys = nb.int64(2 + 17 * c)
        r = 1
        q = nb.int64(1)
        g = 1

        while g == 1 and r <= MAX_R:
            x = y
            for _ in range(r):
                yy = _mulmod_gm(y, y, n, m_norm, v, s) + c
                if yy >= n:
                    yy -= n
                y = yy

            k = 0
            while k < r and g == 1:
                ys = y
                lim = BATCH if BATCH < (r - k) else (r - k)
                for _ in range(lim):
                    yy = _mulmod_gm(y, y, n, m_norm, v, s) + c
                    if yy >= n:
                        yy -= n
                    y = yy
                    diff = x - y if x >= y else y - x
                    q = _mulmod_gm(q, diff, n, m_norm, v, s)
                g = _gcd64(q, n)
                k += lim
            r *= 2

        if g == n:
            yr = ys
            g = 1
            for _ in range(MAX_R):
                yy = _mulmod_gm(yr, yr, n, m_norm, v, s) + c
                if yy >= n:
                    yy -= n
                yr = yy
                diff = x - yr if x >= yr else yr - x
                if diff != 0:
                    g = _gcd64(diff, n)
                if g > 1:
                    break

        if 1 < g < n:
            return g
    return 0


@nb.njit(cache=True)
def _factor_int(n, out_p, out_e):
    """Full int64 factorization of |n|. Trial divide first, then Pollard-rho.

    Writes (p, e) pairs into out_p[:idx], out_e[:idx]; returns (idx, residual).
    residual is 1 on full success, or > 1 if Pollard-rho failed (caller should
    use Python fallback for that case — rare).
    """
    if n < 0:
        n = -n
    if n <= 1:
        return 0, n

    rem, idx = _trial_divide(n, _SMALL_PRIMES, out_p, out_e)
    if rem == 1:
        return idx, 1

    # Stack of factors to further factor
    work = np.empty(64, dtype=np.int64)
    work[0] = rem
    top = 1
    while top > 0:
        top -= 1
        x = work[top]
        if x == 1:
            continue
        if _is_prime_mr(x):
            # accumulate
            # check if already in out_p[:idx]; if so, increment
            found = -1
            for k in range(idx):
                if out_p[k] == x:
                    found = k
                    break
            if found >= 0:
                out_e[found] += 1
            else:
                out_p[idx] = x
                out_e[idx] = 1
                idx += 1
            continue
        d = _pollard_rho(x)
        if d == 0 or d == x:
            # failed; signal residual
            return idx, x
        work[top] = d
        top += 1
        if top >= work.shape[0]:
            return idx, x  # stack overflow, fallback
        work[top] = x // d
        top += 1
        if top >= work.shape[0]:
            return idx, x

    return idx, 1


# ---------------------------------------------------------------------------
# Per-prime inert-K/F valuation
# ---------------------------------------------------------------------------

@nb.njit(cache=True, inline="always")
def _div_by_pi_inert_deg1(m0, m1, m2, a, b, c, p):
    """Try to divide M = m0 + m1·α + m2·α² by π = a + b·α + c·α² in O_F.

    Returns (divisible, q0, q1, q2): divisible = True iff π | M, in which case
    (q0, q1, q2) is the quotient coefficients in basis (1, α, α²).

    Uses Cramer's rule on the multiplication-by-π matrix M_π over Z:
        M_π = [[a, -c, -b], [b, a+3c, 3b-c], [c, b, a+3c]]
    det(M_π) = ±p = ±N(π). q_i = adj(M_π) · m / det.
    """
    e = a + 3*c
    f = 3*b - c
    # Cofactor entries C_ji (so adj(M_π)_ij = C_ji)
    # C_00 = (a+3c)*(a+3c) - b*(3b-c) = e² - 3b² + b*c
    # C_10 = c*(a+3c) - b² = c*e - b²
    # C_20 = a*b + c²
    # C_01 = -(a*b + c²)
    # C_11 = a*(a+3c) + b*c = a*e + b*c
    # C_21 = -3*a*b + a*c - b²
    # C_02 = b² - c*(a+3c) = b² - c*e
    # C_12 = -(a*b + c²)
    # C_22 = a*(a+3c) + b*c = a*e + b*c
    C00 = e*e - 3*b*b + b*c
    C10 = c*e - b*b
    C20 = a*b + c*c
    C01 = -C20  # -(a*b + c²)
    C11 = a*e + b*c
    C21 = -3*a*b + a*c - b*b
    C02 = b*b - c*e
    C12 = -C20  # -(a*b + c²)
    C22 = a*e + b*c

    # adj · m
    num0 = C00*m0 + C10*m1 + C20*m2
    num1 = C01*m0 + C11*m1 + C21*m2
    num2 = C02*m0 + C12*m1 + C22*m2

    # det = ±p; divisible iff p divides each num_i; quotient = num_i / det
    # det sign:
    det = a*(e*e - b*f) + c*(b*e - c*f) + (-b)*(b*b - c*e)
    # det should be ±p (precondition: π is generator of an inert-deg-1 prime above p)

    # divisibility test
    if (num0 % p) != 0 or (num1 % p) != 0 or (num2 % p) != 0:
        return False, 0, 0, 0

    # divide; sign of det matters
    if det > 0:
        q0 = num0 // p
        q1 = num1 // p
        q2 = num2 // p
    else:
        q0 = -(num0 // p)
        q1 = -(num1 // p)
        q2 = -(num2 // p)
    return True, q0, q1, q2


@nb.njit(cache=True)
def _vP_inert_deg1(m0, m1, m2, a, b, c, p, max_iter):
    """v_π(M) for π = (a, b, c) of norm ±p. Stops at max_iter as safety."""
    v = 0
    for _ in range(max_iter):
        ok, q0, q1, q2 = _div_by_pi_inert_deg1(m0, m1, m2, a, b, c, p)
        if not ok:
            return v
        v += 1
        m0, m1, m2 = q0, q1, q2
    return v


@nb.njit(cache=True, inline="always")
def _binary_search_p(plist, p):
    """Return index i in plist such that plist[i] == p, or -1 if not present.
    plist is sorted ascending."""
    lo = 0
    hi = plist.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if plist[mid] < p:
            lo = mid + 1
        elif plist[mid] > p:
            hi = mid
        else:
            return mid
    return -1


# Return-code conventions for the inner kernel:
#   0 = REJECT (inert-odd obstruction found)
#   1 = PASS   (M passes the screen)
#   2 = ZERO   (M == 0)
#   3 = FALLBACK_REQUIRED (factorization residual unfactored, or p ≡ 8 mod 9 with p > table bound)
#                         caller must run the slow Sage path for this row.


@nb.njit(cache=True)
def _screen_one(m0, m1, m2, deg1_table, deg1_plist, deg1_starts, deg1_pmax,
                check_real_embeddings):
    """Per-row Numba screen.

    Pipeline (early-exits ordered cheapest → most expensive):
      1. Trivial filters (zero, overflow, real-embeddings)
      2. Trial-divide for small prime factors
      3. Scan small factors: REJECT immediately if any inert prime has
         the "wrong parity" exponent (no Pollard-rho needed)
      4. If residual > 1, MR-test it:
           - if PRIME and residual mod 9 == 8: REJECT (no rho)
           - if PRIME and residual mod 9 in {2,5}: FALLBACK (contradiction)
           - if PRIME and residual mod 9 in {1,4,7}: split, skip
           - if COMPOSITE: run Pollard-rho, scan new factors for REJECT
      5. Cramer pass: only for inert-deg-1 primes with EVEN e_p (rare)
    """
    if m0 == 0 and m1 == 0 and m2 == 0:
        return 2

    if _norm_FQ_estimate_abs(m0, m1, m2) > 4.611686018427388e18:
        return 3

    if check_real_embeddings and not _real_embeddings_nonneg(m0, m1, m2):
        return 0

    N = _norm_FQ(m0, m1, m2)
    if N == 0:
        return 2
    abs_N = N if N > 0 else -N

    out_p = np.empty(64, dtype=np.int64)
    out_e = np.empty(64, dtype=np.int64)

    # Step 1: trial-divide
    rem, n_small = _trial_divide(abs_N, _SMALL_PRIMES, out_p, out_e)

    # Step 2: scan small factors for unambiguous REJECT
    # (inert-deg-3 with odd v_P, or inert-deg-1 with odd e_p)
    for k in range(n_small):
        p = out_p[k]
        e_p = out_e[k]
        if p == 3:
            continue
        r = p % 9
        if r == 1 or r == 4 or r == 7:
            continue  # K/F-split
        if r == 2 or r == 5:
            if ((e_p // 3) & 1) == 1:
                return 0
            continue
        # r == 8: inert-deg-1, odd e_p ⇒ REJECT (sum-of-three-v_Pi parity)
        if (e_p & 1) == 1:
            return 0
        # e_p even: defer Cramer to Step 5

    # Step 3: handle residual
    n_total = n_small
    if rem != 1:
        if _is_prime_mr(rem):
            # Residual is a single prime with e_R = 1; classify by mod 9
            rr = rem % 9
            if rr == 8:
                return 0
            if rr == 2 or rr == 5:
                # e_R = 1 with inert-deg-3 violates v_R = e_R/3 ∈ ℤ.
                # Indicates classification error; fall back to Sage.
                return 3
            # rr in {1, 4, 7}: split, no constraint — don't append
        else:
            # Composite residual: factor it with Pollard-rho
            n_avail = 64 - n_small
            if n_avail <= 0:
                return 3  # out of slots (should not happen for N < 2^63)
            sub_p = out_p[n_small:]
            sub_e = out_e[n_small:]
            n_new, residual_check = _factor_int(rem, sub_p, sub_e)
            if residual_check != 1:
                return 3
            n_total = n_small + n_new

            # Scan rho-found factors for unambiguous REJECT
            for k in range(n_small, n_total):
                p = out_p[k]
                e_p = out_e[k]
                if p == 3:
                    continue
                r = p % 9
                if r == 1 or r == 4 or r == 7:
                    continue
                if r == 2 or r == 5:
                    if ((e_p // 3) & 1) == 1:
                        return 0
                    continue
                if (e_p & 1) == 1:
                    return 0
                # e_p even: defer Cramer to Step 5

    # Step 4: Cramer pass for inert-deg-1 with EVEN e_p (sum even, could be
    # (even, even, even) PASS or (odd, odd, even) REJECT)
    for k in range(n_total):
        p = out_p[k]
        if p == 3:
            continue
        if (p % 9) != 8:
            continue
        e_p = out_e[k]
        # e_p is guaranteed even here (odd would have rejected in Step 2 / 3)
        if p > deg1_pmax:
            return 3
        idx = _binary_search_p(deg1_plist, p)
        if idx < 0:
            return 3
        gen_start = deg1_starts[idx]
        gen_end = deg1_starts[idx + 1]
        for j in range(gen_start, gen_end):
            a = deg1_table[j, 1]
            b = deg1_table[j, 2]
            c = deg1_table[j, 3]
            v = _vP_inert_deg1(m0, m1, m2, a, b, c, p, e_p + 1)
            if (v & 1) == 1:
                return 0

    return 1


@nb.njit(cache=True)
def screen_rows_batch_njit(
    rows,
    deg1_table, deg1_plist, deg1_starts, deg1_pmax,
    check_real_embeddings,
    out_status,
):
    """Apply _screen_one to each row of rows[:, 3]; write status code into out_status[i].

    Caller dispatches FALLBACK_REQUIRED (3) rows to the slow Sage path.
    Status: 0=reject, 1=pass, 2=zero, 3=fallback.
    """
    for i in range(rows.shape[0]):
        out_status[i] = _screen_one(
            rows[i, 0], rows[i, 1], rows[i, 2],
            deg1_table, deg1_plist, deg1_starts, deg1_pmax,
            check_real_embeddings,
        )


# ---------------------------------------------------------------------------
# Python wrapper (Sage fallback for FALLBACK_REQUIRED rows + optional caching)
# ---------------------------------------------------------------------------

_table_ready = _DEG1_TABLE is not None
if _table_ready:
    print(f"[roots_fast] loaded inert-deg-1 table: {_DEG1_TABLE.shape[0]} generators, "
          f"covering {_DEG1_PLIST.shape[0]} primes up to p={_DEG1_PMAX}", flush=True)
else:
    print(f"[roots_fast] WARNING: no deg-1 table at {_TABLE_PATH}; "
          f"all p ≡ 8 mod 9 cases will use Sage fallback", flush=True)


# Try to import FLINT for fallback factoring. Used to handle rows where
# Numba's capped Pollard-rho gives up (typically 3-4 prime composites with
# deep cycles). FLINT factors these in ~20 µs vs Numba's 1-50 ms.
try:
    import flint as _flint
    _HAVE_FLINT = True
except ImportError:
    _flint = None
    _HAVE_FLINT = False


def _flint_screen_one(m0, m1, m2, check_real_embeddings=True):
    """Pure-Python screen using FLINT for factoring. Returns True if PASS/ZERO,
    False if REJECT. Used as the default fallback for rows where Numba kernel
    returns status 3 (FALLBACK_REQUIRED)."""
    if m0 == 0 and m1 == 0 and m2 == 0:
        return True

    if check_real_embeddings:
        s1 = float(m0) + float(m1) * _ALPHA1 + float(m2) * _ALPHA1_SQ
        s2 = float(m0) + float(m1) * _ALPHA2 + float(m2) * _ALPHA2_SQ
        s4 = float(m0) + float(m1) * _ALPHA4 + float(m2) * _ALPHA4_SQ
        if s1 < 0 or s2 < 0 or s4 < 0:
            return False

    # Norm via Python bignum (always correct, no overflow risk)
    Nv = (m0**3 + 6*m0*m0*m2 + 9*m0*m2*m2 - 3*m0*m1*m1
          + 3*m0*m1*m2 - m1**3 + 3*m1*m2*m2 + m2**3)
    if Nv == 0:
        return True

    # Factor via FLINT (or fall through to error if not installed)
    if not _HAVE_FLINT:
        # No FLINT available — conservatively keep
        return True
    facs = _flint.fmpz(abs(Nv)).factor()

    # Classify each factor; defer Cramer for inert-deg-1-even
    cramer_list = []
    for p_fmpz, e_fmpz in facs:
        p = int(p_fmpz); e_p = int(e_fmpz)
        if p == 3:
            continue
        r = p % 9
        if r in (1, 4, 7):
            continue
        if r in (2, 5):
            if ((e_p // 3) & 1) == 1:
                return False
            continue
        # r == 8 (inert-deg-1)
        if (e_p & 1) == 1:
            return False
        cramer_list.append((p, e_p))

    # Cramer pass for inert-deg-1 with even e_p (rare)
    for p, e_p in cramer_list:
        if p > _DEG1_PMAX:
            return True  # conservative pass (rare)
        idx = int(np.searchsorted(_DEG1_PLIST, p))
        if idx >= _DEG1_PLIST.shape[0] or int(_DEG1_PLIST[idx]) != p:
            return True
        gs = int(_DEG1_STARTS[idx])
        ge = int(_DEG1_STARTS[idx + 1])
        for j in range(gs, ge):
            a = int(_DEG1_TABLE[j, 1])
            b = int(_DEG1_TABLE[j, 2])
            c = int(_DEG1_TABLE[j, 3])
            v = _vP_inert_deg1(m0, m1, m2, a, b, c, p, e_p + 1)
            if (v & 1) == 1:
                return False
    return True


def screen_rows_batch(rows, check_real_embeddings=True, sage_fallback_fn=None):
    """Screen a batch of rows. rows: (N, 3) int64 array.

    Returns: (keep_mask, n_fallback)
      keep_mask : (N,) bool — True if row passes screen (status 1 or 2)
      n_fallback: int       — number of rows that fell back

    If `sage_fallback_fn` is provided, it's used for FALLBACK rows.
    Otherwise the built-in FLINT-based screen is used (if available).
    If neither is available, fallback rows are conservatively KEPT.
    """
    if rows.shape[0] == 0:
        return np.empty(0, dtype=bool), 0

    rows = np.ascontiguousarray(rows, dtype=np.int64)
    out_status = np.empty(rows.shape[0], dtype=np.int8)

    if _table_ready:
        screen_rows_batch_njit(
            rows,
            _DEG1_TABLE, _DEG1_PLIST, _DEG1_STARTS, _DEG1_PMAX,
            bool(check_real_embeddings),
            out_status,
        )
    else:
        out_status.fill(3)

    keep = (out_status == 1) | (out_status == 2)
    fallback_idx = np.flatnonzero(out_status == 3)
    n_fallback = int(fallback_idx.size)

    if n_fallback > 0:
        # Pick fallback: explicit > FLINT default > conservative keep
        if sage_fallback_fn is not None:
            fb_fn = sage_fallback_fn
        elif _HAVE_FLINT:
            def fb_fn(m0, m1, m2):
                return _flint_screen_one(m0, m1, m2, check_real_embeddings=check_real_embeddings)
        else:
            keep[fallback_idx] = True
            return keep, n_fallback

        for i in fallback_idx:
            m0, m1, m2 = int(rows[i, 0]), int(rows[i, 1]), int(rows[i, 2])
            keep[i] = bool(fb_fn(m0, m1, m2))

    return keep, n_fallback


# ---------------------------------------------------------------------------
# Memoization wrapper (D) — applied at the Python layer above screen_rows_batch.
# Uses gcd-canonicalized key.
# ---------------------------------------------------------------------------

from math import gcd as _math_gcd


class _ScreenLRUCache:
    """A simple capped dict for screen results, keyed on gcd-canonicalized rows.

    Two rows (m0, m1, m2) and (k·m0, k·m1, k·m2) for k > 0 have the same screen
    result IF v_P(k) is even for each inert prime P of F that divides (k). In
    general k can be arbitrary integer scaling, so gcd-canonicalization gives a
    correct cache key for the gcd-FREE M; the screen result for the full M
    depends additionally on the prime factorization of gcd. We sidestep this by
    caching only the gcd-FREE form result and recomputing the gcd factor.

    For simplicity and correctness, cache key = (m0, m1, m2) directly (no
    canonicalization). At our cell sizes (a few million unique rows post-dedup
    per bin), this is a memory hit but bounded.
    """
    def __init__(self, max_entries=2_000_000):
        self.max_entries = int(max_entries)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def store(self, key, val):
        if len(self.cache) >= self.max_entries:
            # Naive eviction: clear half
            keys = list(self.cache.keys())[:len(self.cache) // 2]
            for k in keys:
                del self.cache[k]
        self.cache[key] = val


_DEFAULT_CACHE = _ScreenLRUCache()


def screen_rows_batch_cached(rows, check_real_embeddings=True, sage_fallback_fn=None,
                              cache=None):
    """Same as screen_rows_batch but with a process-wide row-level cache (D).

    Returns (keep_mask, n_fallback, n_cache_hits).
    """
    if cache is None:
        cache = _DEFAULT_CACHE
    if rows.shape[0] == 0:
        return np.empty(0, dtype=bool), 0, 0

    rows = np.ascontiguousarray(rows, dtype=np.int64)
    keep = np.empty(rows.shape[0], dtype=bool)
    needs_screen_idx = []

    for i in range(rows.shape[0]):
        key = (int(rows[i, 0]), int(rows[i, 1]), int(rows[i, 2]))
        cv = cache.lookup(key)
        if cv is not None:
            keep[i] = cv
        else:
            needs_screen_idx.append(i)

    n_cache_hits = rows.shape[0] - len(needs_screen_idx)
    n_fallback = 0

    if needs_screen_idx:
        sub = rows[np.array(needs_screen_idx, dtype=np.int64)]
        sub_keep, n_fallback = screen_rows_batch(
            sub, check_real_embeddings=check_real_embeddings,
            sage_fallback_fn=sage_fallback_fn,
        )
        for j, idx in enumerate(needs_screen_idx):
            keep[idx] = sub_keep[j]
            cache.store(
                (int(rows[idx, 0]), int(rows[idx, 1]), int(rows[idx, 2])),
                bool(sub_keep[j]),
            )

    return keep, n_fallback, n_cache_hits
