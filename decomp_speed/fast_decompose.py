"""fast_decompose.py -- drop-in, bit-identical, much faster replacement for
hrsa/canonical_reducer.decompose_canonical.

What is the same
----------------
The peeling loop, prefix order, selection rule (cr._SELECTION: "d" or
"tcost"), tie-breaking, double-prefix fallback semantics, the apply step
(V := P.V; cr._reduce_by_three) and the final monomial classification are
the original's.  The apply step and classification literally call the
original canonical_reducer functions, so the returned dict (syllables,
D_count, trailing_clifford, ...) is identical.

What is different
-----------------
1. No 4374-entry _FastZ9Frac prefix table (the ~30 s build).  Every prefix
   is P = H . Diag(z^a1, z^a2, z^a3) . R^eps . X^delta, so
     (P.V)[0][0] = ia * sum_j  s_j z^{a_j} V[pi_delta(j)][0],  ia = c/3,
   c = -1 - 2 z^3.  All 4374 candidate numerators are therefore signed sums
   of three rows of a tiny 9x3x6 table {z^a * n_k}, built with integer numpy
   in one shot.  Only the winning P is built (memoised) for the apply step.
2. sde_chi via the chi-adic valuation in closed form.  The original
   sde_chi_z9 recursion computes exactly v_chi(x), chi = 1 - zeta_9:
   with x(1+t) = sum_i b_i t^i (b = Taylor coefficients at zeta = 1),
       v_chi(x) = min_i ( 6 * v_3(b_i) + i ),   i = 0..5
   (the six terms have distinct residues mod 6).  We vectorise this over
   all candidates.  sde_chi_full = 6*f - v for f >= 1, and the candidate
   entries always have f >= 1, so new_s = 6*(f+1) - v(c*y).
3. The double-prefix O(N^2) search is vectorised over the inner prefix and
   reproduces the original's "lexicographically first (idx1, idx2) with
   minimal total cost" result.

Arithmetic is int64 when the column-0 coefficients are small enough
(guard below, holds comfortably through f ~ 24), else exact Python-int
object arrays (same code path, slower but still exact).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_U = Path(__file__).resolve().parent.parent
for _p in (str(_U / "hrsa"), str(_U)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import canonical_reducer as cr  # noqa: E402

# ---------------------------------------------------------------- constants


def _mulmat(coefs):
    """6x6 integer matrix of multiplication by the Z[zeta_9] element `coefs`
    (columns = images of the basis), computed with the original arithmetic."""
    a = cr._FastZ9(tuple(cr._INT(x) for x in coefs))
    M = np.zeros((6, 6), dtype=np.int64)
    for k in range(6):
        e = [0] * 6
        e[k] = 1
        prod = (a * cr._FastZ9(tuple(cr._INT(x) for x in e))).coefs
        M[:, k] = [int(x) for x in prod]
    return M


def _zeta_coefs(a):
    return [int(x) for x in cr._zeta9_power(a).num.coefs]


ZMAT = np.stack([_mulmat(_zeta_coefs(a)) for a in range(9)])   # z^a
CMAT = _mulmat((-1, 0, 0, -2, 0, 0))                               # c = 3*ia
# Taylor transform at zeta = 1:  b_i = sum_k binom(k, i) x_k
import math as _m
TAYLOR = np.array([[_m.comb(k, i) if k >= i else 0 for k in range(6)]
                   for i in range(6)], dtype=np.int64)

_GT = np.stack([TAYLOR @ CMAT @ ZMAT[a] for a in range(9)])          # b = T.c.z^a.(.)

# Prefix enumeration in the original table order: eps, delta, a1, a2, a3.
_E, _DL, _A1, _A2, _A3 = np.meshgrid(np.arange(2), np.arange(3), np.arange(9),
                                     np.arange(9), np.arange(9), indexing="ij")
PFX_EPS = _E.ravel()
PFX_DELTA = _DL.ravel()
PFX_A = np.ascontiguousarray(np.stack([_A1.ravel(), _A2.ravel(), _A3.ravel()], axis=1).astype(np.int64))
NPFX = PFX_EPS.size
# Row j of D.R^eps.X^delta has its single entry at column pi_delta(j):
#   X[0][2] = X[1][0] = X[2][1] = 1  ->  pi_1(j) = j-1, pi_2(j) = j+1 (mod 3)
PFX_SRC = np.ascontiguousarray(np.stack([(np.arange(3) - d) % 3 for d in range(3)])[PFX_DELTA])  # (4374, 3)
PFX_SIGN = np.ones((NPFX, 3), dtype=np.int64)
PFX_SIGN[:, 2] = np.where(PFX_EPS == 1, -1, 1)                       # R = diag(1,1,-1)
PFX_D = ((PFX_A % 3) != 0).sum(axis=1) + PFX_EPS                     # count_d_from_syllable


def _selection_keys():
    sel = cr._SELECTION
    if sel["mode"] == "d":
        return np.ascontiguousarray(PFX_D, dtype=np.int64).copy()
    a = PFX_A
    nontriv = (a % 3 != 0).any(axis=1)
    lvl3 = (a.sum(axis=1) % 3) == 0
    t = np.where(nontriv, np.where(lvl3, 1, sel["w4"]), 0)
    return np.ascontiguousarray(t + np.where(PFX_EPS == 1, sel["wr"], 0), dtype=np.int64)


# ---------------------------------------------------------------- kernels

def _inf(M):
    return int(np.abs(M).sum(axis=-1).max())


# Worst-case growth |Taylor coef| / max|column-0 coef| over both kernels
# (single: 3 terms through T.c.z^a; double: z = 3 terms omega^{jk}.(s z^a n),
# then c, then 3 terms through T.c.z^a).  int64 path iff max|coef| * growth < 2^62.
_GROWTH = max(3 * _inf(_GT),
              3 * _inf(_GT) * _inf(CMAT) * 3 * _inf(ZMAT) * _inf(ZMAT))
_INT64_GUARD = (2 ** 62) // _GROWTH


def _valuation(x):
    """v_chi of each row of x (shape (N, 6)); -1 for the zero element."""
    return _valuation_taylor(x @ TAYLOR.T)


def _valuation_taylor(b):
    """v_chi from Taylor coefficients b (N, 6) at zeta = 1; -1 for zero."""
    N = b.shape[0]
    big = 10 ** 9
    best = np.full(N, big, dtype=np.int64)
    for i in range(6):
        bi = b[:, i]
        nz = bi != 0
        if b.dtype == object:
            v3 = np.array([_v3_int(int(z)) if z else 0 for z in bi], dtype=np.int64)
        else:
            v3 = np.zeros(N, dtype=np.int64)
            r = np.where(nz, bi, 1)
            while True:
                m = (r % 3) == 0
                if not m.any():
                    break
                v3 += m
                r = np.where(m, r // 3, r)
        cand = np.where(nz, 6 * v3 + i, big)
        best = np.minimum(best, cand)
    best[best == big] = -1
    return best


def _v3_int(z):
    n = 0
    while z % 3 == 0:
        z //= 3
        n += 1
    return n


def _column_numerators(V):
    """Column 0 of V as an int array (3, 6) over a common denominator 3^f."""
    f = max(V[k][0].denom_pow3 for k in range(3))
    rows = []
    for k in range(3):
        e = V[k][0]
        sc = 3 ** (f - e.denom_pow3)
        rows.append([int(c) * sc for c in e.num.coefs])
    mx = max(abs(x) for r in rows for x in r)
    # int64 path: no overflow in the kernels, and the 3^(f+3) moduli fit.
    dtype = np.int64 if (mx < _INT64_GUARD and f + 4 <= 39) else object
    return np.array(rows, dtype=dtype), f


def _terms(n):
    """u[p, j] = s_j * z^{a_j} * n[pi(j)]  for every prefix p: (NPFX, 3, 6)."""
    dt = n.dtype
    W = np.matmul(ZMAT.astype(dt), n.T).transpose(0, 2, 1)    # (9, 3, 6): z^a * n_k
    u = W[PFX_A, PFX_SRC]                                      # (NPFX, 3, 6)
    return u * PFX_SIGN[:, :, None].astype(dt)


def _sde_from_num(num, fden):
    """sde_chi_full for entries num/3^fden with fden >= 1 (vector)."""
    v = _valuation(num)
    return np.where(v < 0, 0, 6 * fden - v)


def _single_scan(n, f):
    """new_s for all prefixes (vector, original table order).
    Taylor coefs of c*y, y = sum_j s_j z^{a_j} n_pi(j), via _GT = T.c.z^a."""
    dt = n.dtype
    GW = np.einsum("axy,ky->akx", _GT.astype(dt), n)          # (9, 3, 6)
    b = (GW[PFX_A, PFX_SRC] * PFX_SIGN[:, :, None].astype(dt)).sum(axis=1)
    v = _valuation_taylor(b)
    return np.where(v < 0, 0, 6 * (f + 1) - v)


def _pick_strict(new_s, s, keys):
    valid = new_s == s - 1
    if not valid.any():
        return -1
    kk = np.where(valid, keys, np.iinfo(np.int64).max)
    return int(np.argmin(kk))                                   # first minimal


def _pick_greedy(new_s, s, keys):
    valid = new_s < s
    if not valid.any():
        return -1
    drop = s - new_s
    brk = np.nonzero(valid & (keys == 0) & (drop >= 6))[0]
    if brk.size:
        return int(brk[0])
    big = np.iinfo(np.int64).max
    kmin = np.where(valid, keys, big).min()
    cand = valid & (keys == kmin)
    dmax = np.where(cand, drop, -1).max()
    return int(np.nonzero(cand & (drop == dmax))[0][0])


_ZOM = np.stack([np.stack([ZMAT[(3 * j * k) % 9] for j in range(3)]) for k in range(3)])  # [k][j]: omega^{jk}
_BLOCK = 64


def _double_scan(n, f, s, keys, mid_s, u):
    """Vectorised replica of cr._try_double_prefix; returns (i1, i2, mid_s, new_s).

    Original semantics: result = lexicographically first (idx1, idx2) among
    pairs with minimal k1 + k2, subject to mid_s(idx1) in [s-1, s+1] and
    sde((P2 P1 V)[0][0]) < s.  The inner test sde < s is evaluated as
    v_chi(c*w) >= T, T = 6(f+2) - s + 1, i.e. Taylor coefficient b_i
    divisible by 3^ceil((T-i)/6) for every i (zero element included, as in
    the original, where sde(0) = 0 < s)."""
    dt = n.dtype
    T = 6 * (f + 2) - s + 1
    mods = [3 ** max(0, -(-(T - i) // 6)) for i in range(6)]
    if dt != object:
        mods = np.array(mods, dtype=np.int64)
    else:
        mods = np.array(mods, dtype=object)
    GT = _GT.astype(dt)
    ZOM = _ZOM.astype(dt)
    Cm = CMAT.astype(dt)
    sign = PFX_SIGN[None, :, :, None].astype(dt)
    window = (mid_s >= s - 1) & (mid_s <= s + 1)
    order = np.nonzero(window)[0]
    order = order[np.argsort(keys[order], kind="stable")]
    big = np.iinfo(np.int64).max
    best = (big, -1, -1)                                       # (total, i1, i2)
    for st in range(0, order.size, _BLOCK):
        blk = order[st:st + _BLOCK]
        if keys[blk[0]] > best[0]:
            break
        ub = u[blk]                                            # (B, 3, 6): s_j z^{a_j} n_pi(j)
        z = np.einsum("kjxy,bjy->bkx", ZOM, ub)               # (B, 3, 6): sum_j omega^{jk} u_j
        mid = np.einsum("xy,bky->bkx", Cm, z)                  # numerators of (P1 V)[k][0]
        GW = np.einsum("axy,bky->bakx", GT, mid)                # (B, 9, 3, 6)
        bcoef = (GW[:, PFX_A, PFX_SRC] * sign).sum(axis=2)     # (B, NPFX, 6) Taylor coefs
        valid = ((bcoef % mods) == 0).all(axis=2)              # (B, NPFX)
        kk = np.where(valid, keys[None, :], big)
        i2 = kk.argmin(axis=1)
        k2 = kk[np.arange(blk.size), i2]
        for r in range(blk.size):
            if k2[r] == big:
                continue
            tot = int(keys[blk[r]]) + int(k2[r])
            i1 = int(blk[r])
            if tot < best[0] or (tot == best[0] and i1 < best[1]):
                best = (tot, i1, int(i2[r]))
    if best[1] < 0:
        return None
    return best[1], best[2], int(mid_s[best[1]]), None


# ---------------------------------------------------------------- numba kernels
# Used on the int64 path.  The double-prefix kernel is a literal transcription
# of cr._try_double_prefix's loop (same order, same pruning, same break), with
# the inner sde test done as Taylor-coefficient divisibility (early exit on
# the first non-divisible coefficient).

try:
    import numba as _nb
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False

if _HAVE_NUMBA:
    @_nb.njit(cache=True, nogil=True)
    def _nb_single(GW, A, SRC, SGN, fden, out):
        N = A.shape[0]
        for p in range(N):
            best = 1 << 40
            for x in range(6):
                b = (SGN[p, 0] * GW[A[p, 0], SRC[p, 0], x] + SGN[p, 1] * GW[A[p, 1], SRC[p, 1], x]
                     + SGN[p, 2] * GW[A[p, 2], SRC[p, 2], x])
                if b != 0:
                    v3 = 0
                    while b % 3 == 0:
                        b //= 3
                        v3 += 1
                    c = 6 * v3 + x
                    if c < best:
                        best = c
            out[p] = 0 if best == (1 << 40) else 6 * fden - best

    @_nb.njit(cache=True, nogil=True)
    def _nb_double(GWall, A, SRC, SGN, keys, window, mods):
        N = A.shape[0]
        BIG = 1 << 60
        best = BIG
        bi1 = -1
        bi2 = -1
        for i1 in range(N):
            k1 = keys[i1]
            if k1 >= best:
                continue
            if not window[i1]:
                continue
            for i2 in range(N):
                if k1 + keys[i2] >= best:
                    continue
                ok = True
                for x in range(6):
                    b = (SGN[i2, 0] * GWall[A[i2, 0], i1, SRC[i2, 0], x]
                         + SGN[i2, 1] * GWall[A[i2, 1], i1, SRC[i2, 1], x]
                         + SGN[i2, 2] * GWall[A[i2, 2], i1, SRC[i2, 2], x])
                    if b % mods[x] != 0:
                        ok = False
                        break
                if ok:
                    best = k1 + keys[i2]
                    bi1 = i1
                    bi2 = i2
                    if best == 0:
                        break
        return best, bi1, bi2


def _single_scan_nb(n, f):
    GW = np.ascontiguousarray(np.matmul(_GT, n.T).transpose(0, 2, 1))   # (9, 3, 6)
    out = np.empty(NPFX, dtype=np.int64)
    _nb_single(GW, PFX_A, PFX_SRC, PFX_SIGN, f + 1, out)
    return out


def _double_scan_nb(n, f, s, keys, mid_s):
    u = _terms(n)                                               # (N, 3, 6)
    # z_k = sum_j omega^{jk} u_j ; mid_k = c z_k  (numerators of (P1 V)[k][0])
    mid = np.stack([sum(u[:, j] @ (CMAT @ _ZOM[k, j]).T for j in range(3)) for k in range(3)],
                   axis=1)                                     # (N, 3, 6)
    # GWall[a, i1, k] = T.c.z^a . mid_k(i1)
    GWall = np.ascontiguousarray(
        np.matmul(mid.reshape(1, -1, 6), _GT.transpose(0, 2, 1)).reshape(9, NPFX, 3, 6))
    T = 6 * (f + 2) - s + 1
    mods = np.array([3 ** max(0, -(-(T - i) // 6)) for i in range(6)], dtype=np.int64)
    window = (mid_s >= s - 1) & (mid_s <= s + 1)
    best, i1, i2 = _nb_double(GWall, PFX_A, PFX_SRC, PFX_SIGN, keys, window, mods)
    if i1 < 0:
        return None
    return int(i1), int(i2), int(mid_s[i1]), None


# ---------------------------------------------------------------- driver

_P_CACHE: dict = {}
_HRX = None


def _prefix_matrix(idx):
    global _HRX
    key = (cr._BACKEND_NAME, idx)
    P = _P_CACHE.get(key)
    if P is None:
        if _HRX is None or _HRX[0] != cr._BACKEND_NAME:
            _HRX = (cr._BACKEND_NAME, cr.gate_H(), cr.gate_R(), cr.gate_X())
        a1, a2, a3 = (int(x) for x in PFX_A[idx])
        P = cr._build_prefix(a1, a2, a3, int(PFX_EPS[idx]), int(PFX_DELTA[idx]), *_HRX[1:])
        _P_CACHE[key] = P
    return P


def _syl(idx):
    a1, a2, a3 = (int(x) for x in PFX_A[idx])
    return {"a0": a1, "a1": a2, "a2": a3, "eps": int(PFX_EPS[idx]),
            "delta": int(PFX_DELTA[idx]), "has_H": True}


def decompose_fast(V_input, max_iter=None, verbose=False, greedy_single=False,
                   skip_double=False) -> dict:
    """Same signature and return value as cr.decompose_canonical."""
    keys = _selection_keys()
    if V_input and not isinstance(V_input[0][0], cr._FastZ9Frac):
        V = [[cr._z9frac_to_fast(V_input[i][j]) for j in range(3)] for i in range(3)]
    else:
        V = [row[:] for row in V_input]
    s_initial = cr.sde_chi_full(V[0][0])
    s = s_initial
    if max_iter is None:
        max_iter = s + 50
    syllables, D_count, n_iter = [], 0, 0
    t0 = time.time()

    def fail(err):
        return {"success": False, "D_count": -1, "sde_chi_initial": s_initial,
                "sde_chi_final": s, "syllables": syllables, "trailing_clifford": V,
                "n_iter": n_iter, "peel_seconds": time.time() - t0, "error": err}

    if s == 999:
        return {"success": False, "D_count": -1, "sde_chi_initial": s_initial,
                "sde_chi_final": s, "syllables": [], "trailing_clifford": V,
                "n_iter": 0, "peel_seconds": 0.0, "error": "V[0][0] is zero — cannot peel"}

    while s > 0 and n_iter < max_iter:
        n_iter += 1
        n, f = _column_numerators(V)
        use_nb = _HAVE_NUMBA and n.dtype != object
        new_s = _single_scan_nb(n, f) if use_nb else _single_scan(n, f)
        idx = _pick_strict(new_s, s, keys)
        if idx < 0 and greedy_single:
            idx = _pick_greedy(new_s, s, keys)
        if idx >= 0:
            V = cr._reduce_by_three(cr._prefix_times_V(_prefix_matrix(idx), V))
            syllables.append(_syl(idx))
            D_count += int(PFX_D[idx])
            s = cr.sde_chi_full(V[0][0])
            continue
        if skip_double:
            return fail(f"single-prefix failed at sde_chi={s}; double-prefix skipped (--skip-double)")
        dbl = (_double_scan_nb(n, f, s, keys, new_s) if use_nb
               else _double_scan(n, f, s, keys, new_s, _terms(n)))
        if dbl is None:
            return fail(f"single+double prefix both failed at sde_chi={s}")
        i1, i2, _, _ = dbl
        V = cr._prefix_times_V(_prefix_matrix(i1), V)
        V = cr._prefix_times_V(_prefix_matrix(i2), V)
        V = cr._reduce_by_three(V)
        syllables += [_syl(i1), _syl(i2)]
        D_count += int(PFX_D[i1]) + int(PFX_D[i2])
        s = cr.sde_chi_full(V[0][0])

    peel = time.time() - t0
    if s != 0:
        return fail(f"max_iter ({max_iter}) reached, sde_chi still {s}")
    is_mono, mono_d, _ = cr.classify_monomial_and_d_cost(V)
    if not is_mono:
        return fail("residual V at sde_chi=0 is not monomial; algorithm did not reach a Clifford form")
    return {"success": True, "D_count": D_count + mono_d, "sde_chi_initial": s_initial,
            "sde_chi_final": 0, "syllables": syllables, "trailing_clifford": V,
            "n_iter": n_iter, "peel_seconds": peel, "residual_D": mono_d}


def install():
    """Monkey-patch cr.decompose_canonical -> decompose_fast (for existing callers
    such as nick_tcost_all.work / analyze_topk.work)."""
    cr.decompose_canonical = decompose_fast
