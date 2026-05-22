import math
import os, time, json, shutil
import numpy as np
import numba as nb
from mpi4py import MPI

from .roots import (
    quick_screen_M,
    build_fields,
    quick_screen_status_profiled,
    quick_screen_M_fast,
    quick_screen_status_fast_profiled,
    new_quick_screen_profile,
    _profile_summary,
)

# Optional fast (Numba) screen replacement (~1000× faster than Sage path).
# Falls back row-by-row to Sage for rows that are out of int64 range or have
# a prime factor of N(M) ≡ 8 mod 9 outside the precomputed deg-1 generator table.
try:
    from . import roots_fast as _roots_fast
    _HAVE_FAST_SCREEN = _roots_fast._table_ready
except Exception:
    _roots_fast = None
    _HAVE_FAST_SCREEN = False
# Env override to disable fast screen (e.g. for debugging or strict-compatibility runs)
if os.environ.get("ZETA9_DISABLE_FAST_SCREEN", "") in ("1", "true", "yes"):
    _HAVE_FAST_SCREEN = False

# Streaming-mode helpers reused from stage 2 (Change 1 in streaming refactor plan)
from .select_triples_optimized import (
    ROW_DTYPE,
    _append_rows_raw,
    _bucket_linear_hash_rows,
    _unique_sorted_rows,
)

# Hash coeffs identical to stage 2's bucket partition (same int64 magnitudes,
# same desired distribution). Defining here to avoid an import-time circular
# dep on a private stage-2 constant.
_STREAM_BUCKET_COEFFS = (911382323, 972663749, 9721)


# ============================================================
#  Embeddings for the real cubic field Q(alpha),
#  alpha = zeta9 + zeta9^{-1} = 2 cos(2*pi/9)
# ============================================================

ALPHA1 = 2.0 * math.cos(2.0 * math.pi / 9.0)
ALPHA2 = 2.0 * math.cos(4.0 * math.pi / 9.0)
ALPHA4 = 2.0 * math.cos(8.0 * math.pi / 9.0)

A_EMB = np.array([
    [1.0, ALPHA1, ALPHA1 * ALPHA1],
    [1.0, ALPHA2, ALPHA2 * ALPHA2],
    [1.0, ALPHA4, ALPHA4 * ALPHA4],
], dtype=np.float64)

A_INV = np.linalg.inv(A_EMB)


# ============================================================
#  Pull back embedding box to safe coefficient box
# ============================================================

def coefficient_box_from_embedding_box(s1_lo, s1_hi, s2_lo, s2_hi, s4_lo, s4_hi):
    corners = []
    for s1 in (s1_lo, s1_hi):
        for s2 in (s2_lo, s2_hi):
            for s4 in (s4_lo, s4_hi):
                sig = np.array([s1, s2, s4], dtype=np.float64)
                m = A_INV @ sig
                corners.append(m)

    corners = np.array(corners)
    mins = np.floor(corners.min(axis=0)).astype(np.int64) - 1
    maxs = np.ceil(corners.max(axis=0)).astype(np.int64) + 1

    return (
        (int(mins[0]), int(maxs[0])),
        (int(mins[1]), int(maxs[1])),
        (int(mins[2]), int(maxs[2])),
    )


# ============================================================
#  Exact polygon from the two strip constraints in (n1,n2)
# ============================================================

def strip_polygon_vertices(target_t, epsY, total_scale):
    """
    The existence of some n0 with
        sigma1 in [t-epsY, t+epsY],
        sigma2,sigma4 in [0,total_scale]
    implies the two strip constraints
        L <= a21*n1 + b21*n2 <= U
        L <= a41*n1 + b41*n2 <= U
    with
        L = -epsY - target_t
        U = total_scale + epsY - target_t.

    Their intersection is a parallelogram in the (n1,n2)-plane.
    Return its four vertices as a (4,2) float array.
    """
    alpha1_sq = ALPHA1 * ALPHA1
    alpha2_sq = ALPHA2 * ALPHA2
    alpha4_sq = ALPHA4 * ALPHA4

    a21 = ALPHA2 - ALPHA1
    b21 = alpha2_sq - alpha1_sq
    a41 = ALPHA4 - ALPHA1
    b41 = alpha4_sq - alpha1_sq

    L = -epsY - target_t
    U = total_scale + epsY - target_t

    M = np.array([[a21, b21], [a41, b41]], dtype=np.float64)
    rhs_list = [
        np.array([L, L], dtype=np.float64),
        np.array([L, U], dtype=np.float64),
        np.array([U, L], dtype=np.float64),
        np.array([U, U], dtype=np.float64),
    ]
    verts = np.empty((4, 2), dtype=np.float64)
    for i, rhs in enumerate(rhs_list):
        verts[i, :] = np.linalg.solve(M, rhs)
    return verts


def choose_sweep_from_polygon(vertices):
    """
    Pick the outer-loop direction based on the shorter span of the polygon.
    Returns:
        sweep_axis: 0 means sweep n2 rows, 1 means sweep n1 columns
        n1_lo, n1_hi, n2_lo, n2_hi: integer bounds from the polygon vertices
    """
    n1_min = float(vertices[:, 0].min())
    n1_max = float(vertices[:, 0].max())
    n2_min = float(vertices[:, 1].min())
    n2_max = float(vertices[:, 1].max())

    n1_lo = int(math.floor(n1_min)) - 1
    n1_hi = int(math.ceil(n1_max)) + 1
    n2_lo = int(math.floor(n2_min)) - 1
    n2_hi = int(math.ceil(n2_max)) + 1

    span_n1 = n1_hi - n1_lo + 1
    span_n2 = n2_hi - n2_lo + 1
    sweep_axis = 0 if span_n2 <= span_n1 else 1
    return sweep_axis, n1_lo, n1_hi, n2_lo, n2_hi


# ============================================================
#  Inert primes p == 2 mod 3 up to a limit
# ============================================================

def inert_primes_up_to(B):
    if B < 2:
        return []

    is_prime = np.ones(B + 1, dtype=np.uint8)
    is_prime[:2] = 0
    r = int(math.isqrt(B))
    for p in range(2, r + 1):
        if is_prime[p]:
            is_prime[p * p:B + 1:p] = 0

    out = []
    for p in range(2, B + 1):
        if is_prime[p] and (p % 3 == 2):
            out.append(p)
    return out


# ============================================================
#  Numba helpers
# ============================================================

@nb.njit(cache=True, inline="always")
def _round_nearest_int(x):
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    else:
        return int(math.ceil(x - 0.5))


@nb.njit(cache=True, inline="always")
def _vp_abs(n, p):
    if n == 0:
        return 10**9
    if n < 0:
        n = -n
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


@nb.njit(cache=True, inline="always")
def _passes_inert_parity(n0, n1, n2, inert_primes):
    """Pre-filter rejecting rows where N_F/Q(M) has an unambiguous odd
    inert obstruction at one of the small primes.

    Previously: computed v_p(gcd(n0,n1,n2)) and rejected if odd. That's
    partially correct for inert-deg-3 primes (p ≡ 2,5 mod 9) — but
    incorrect for inert-deg-1 (p ≡ 8 mod 9), and falsely rejected some
    rows that should pass. Now: computes v_p(N(M)) and applies the proper
    screen rule per mod-9 class.

    Uses uint64 wraparound for the closed-form norm (same trick as
    roots_fast._norm_FQ; correct iff |N| < 2^63, which holds for all
    production rows even at f=6 coord-magnitude 1.4M).
    """
    if n0 == 0 and n1 == 0 and n2 == 0:
        return True

    # Closed-form N_F/Q(M) for M = n0 + n1·α + n2·α², α³ = 3α - 1
    u0 = nb.uint64(n0)
    u1 = nb.uint64(n1)
    u2 = nb.uint64(n2)
    SIX = nb.uint64(6)
    THREE = nb.uint64(3)
    NINE = nb.uint64(9)
    sN = u0 * u0 * u0
    sN = sN + SIX * u0 * u0 * u2
    sN = sN + NINE * u0 * u2 * u2
    sN = sN - THREE * u0 * u1 * u1
    sN = sN + THREE * u0 * u1 * u2
    sN = sN - u1 * u1 * u1
    sN = sN + THREE * u1 * u2 * u2
    sN = sN + u2 * u2 * u2
    N = nb.int64(sN)
    if N == 0:
        return True
    if N < 0:
        N = -N

    for i in range(inert_primes.shape[0]):
        p = inert_primes[i]
        # v_p(N)
        e = 0
        nn = N
        while nn % p == 0:
            nn //= p
            e += 1
        r = p % 9
        if r == 2 or r == 5:
            # inert-deg-3: v_P(M) = e // 3 must be even
            if ((e // 3) & 1) == 1:
                return False
        elif r == 8:
            # inert-deg-1: parity of sum of three v_Pi = parity of e
            if (e & 1) == 1:
                return False
        # r in {1, 4, 7}: split, no constraint
    return True


# ============================================================
#  Count local hits
# ============================================================

@nb.njit(cache=True)
def _count_chunk_target(
    total_scale,
    target_t,
    epsY_lo,
    epsY_hi,
    n1_lo, n1_hi,
    n2_lo, n2_hi,
    rank, size,
    alpha1, alpha2, alpha4,
    inert_primes,
    use_inert_parity,
    sweep_axis,
):
    alpha1_sq = alpha1 * alpha1
    alpha2_sq = alpha2 * alpha2
    alpha4_sq = alpha4 * alpha4

    tol = min(1e-15, 0.1 * epsY_hi)

    a21 = alpha2 - alpha1
    b21 = alpha2_sq - alpha1_sq

    a41 = alpha4 - alpha1
    b41 = alpha4_sq - alpha1_sq

    count = 0

    checked_outer = 0
    checked_pairs = 0
    passed_strip_pairs = 0
    near_integer_pairs = 0
    feasible_pairs = 0
    parity_rejects = 0

    if sweep_axis == 0:
        start_outer = n2_lo + rank
        for n2 in range(start_outer, n2_hi + 1, size):
            checked_outer += 1

            lo = n1_lo
            hi = n1_hi

            rhs_lo = -epsY_hi - target_t - b21 * n2
            rhs_hi = total_scale + epsY_hi - target_t - b21 * n2
            if abs(a21) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / a21
                x2 = rhs_hi / a21
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            rhs_lo = -epsY_hi - target_t - b41 * n2
            rhs_hi = total_scale + epsY_hi - target_t - b41 * n2
            if abs(a41) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / a41
                x2 = rhs_hi / a41
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            for n1 in range(lo, hi + 1):
                checked_pairs += 1
                passed_strip_pairs += 1

                c1 = n1 * alpha1 + n2 * alpha1_sq
                c2 = n1 * alpha2 + n2 * alpha2_sq
                c4 = n1 * alpha4 + n2 * alpha4_sq

                x = target_t - c1
                n0 = _round_nearest_int(x)
                s1 = n0 + c1
                abs_err = abs(s1 - target_t)

                if not (abs_err < epsY_hi + tol):
                    continue
                if epsY_lo > 0.0 and abs_err <= epsY_lo + tol:
                    continue

                near_integer_pairs += 1

                s2 = n0 + c2
                if s2 < -tol or s2 > total_scale + tol:
                    continue

                s4 = n0 + c4
                if s4 < -tol or s4 > total_scale + tol:
                    continue

                feasible_pairs += 1

                if use_inert_parity:
                    if not _passes_inert_parity(n0, n1, n2, inert_primes):
                        parity_rejects += 1
                        continue

                count += 1
    else:
        start_outer = n1_lo + rank
        for n1 in range(start_outer, n1_hi + 1, size):
            checked_outer += 1

            lo = n2_lo
            hi = n2_hi

            rhs_lo = -epsY_hi - target_t - a21 * n1
            rhs_hi = total_scale + epsY_hi - target_t - a21 * n1
            if abs(b21) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / b21
                x2 = rhs_hi / b21
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            rhs_lo = -epsY_hi - target_t - a41 * n1
            rhs_hi = total_scale + epsY_hi - target_t - a41 * n1
            if abs(b41) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / b41
                x2 = rhs_hi / b41
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            for n2 in range(lo, hi + 1):
                checked_pairs += 1
                passed_strip_pairs += 1

                c1 = n1 * alpha1 + n2 * alpha1_sq
                c2 = n1 * alpha2 + n2 * alpha2_sq
                c4 = n1 * alpha4 + n2 * alpha4_sq

                x = target_t - c1
                n0 = _round_nearest_int(x)
                s1 = n0 + c1
                abs_err = abs(s1 - target_t)

                if not (abs_err < epsY_hi + tol):
                    continue
                if epsY_lo > 0.0 and abs_err <= epsY_lo + tol:
                    continue

                near_integer_pairs += 1

                s2 = n0 + c2
                if s2 < -tol or s2 > total_scale + tol:
                    continue

                s4 = n0 + c4
                if s4 < -tol or s4 > total_scale + tol:
                    continue

                feasible_pairs += 1

                if use_inert_parity:
                    if not _passes_inert_parity(n0, n1, n2, inert_primes):
                        parity_rejects += 1
                        continue

                count += 1

    return (
        count,
        checked_outer,
        checked_pairs,
        passed_strip_pairs,
        near_integer_pairs,
        feasible_pairs,
        parity_rejects,
    )


@nb.njit(cache=True)
def _fill_chunk_target(
    out_arr,
    total_scale,
    target_t,
    epsY_lo,
    epsY_hi,
    n1_lo, n1_hi,
    n2_lo, n2_hi,
    rank, size,
    alpha1, alpha2, alpha4,
    inert_primes,
    use_inert_parity,
    sweep_axis,
):
    alpha1_sq = alpha1 * alpha1
    alpha2_sq = alpha2 * alpha2
    alpha4_sq = alpha4 * alpha4

    tol = min(1e-15, 0.1 * epsY_hi)

    a21 = alpha2 - alpha1
    b21 = alpha2_sq - alpha1_sq

    a41 = alpha4 - alpha1
    b41 = alpha4_sq - alpha1_sq

    pos = 0

    if sweep_axis == 0:
        start_outer = n2_lo + rank
        for n2 in range(start_outer, n2_hi + 1, size):
            lo = n1_lo
            hi = n1_hi

            rhs_lo = -epsY_hi - target_t - b21 * n2
            rhs_hi = total_scale + epsY_hi - target_t - b21 * n2
            if abs(a21) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / a21
                x2 = rhs_hi / a21
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            rhs_lo = -epsY_hi - target_t - b41 * n2
            rhs_hi = total_scale + epsY_hi - target_t - b41 * n2
            if abs(a41) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / a41
                x2 = rhs_hi / a41
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            for n1 in range(lo, hi + 1):
                c1 = n1 * alpha1 + n2 * alpha1_sq
                c2 = n1 * alpha2 + n2 * alpha2_sq
                c4 = n1 * alpha4 + n2 * alpha4_sq

                x = target_t - c1
                n0 = _round_nearest_int(x)
                s1 = n0 + c1
                abs_err = abs(s1 - target_t)

                if not (abs_err < epsY_hi + tol):
                    continue
                if epsY_lo > 0.0 and abs_err <= epsY_lo + tol:
                    continue
                if epsY_lo > 0.0 and abs_err <= epsY_lo + tol:
                    continue

                s2 = n0 + c2
                if s2 < -tol or s2 > total_scale + tol:
                    continue

                s4 = n0 + c4
                if s4 < -tol or s4 > total_scale + tol:
                    continue

                if use_inert_parity:
                    if not _passes_inert_parity(n0, n1, n2, inert_primes):
                        continue

                out_arr[pos, 0] = n0
                out_arr[pos, 1] = n1
                out_arr[pos, 2] = n2
                pos += 1
    else:
        start_outer = n1_lo + rank
        for n1 in range(start_outer, n1_hi + 1, size):
            lo = n2_lo
            hi = n2_hi

            rhs_lo = -epsY_hi - target_t - a21 * n1
            rhs_hi = total_scale + epsY_hi - target_t - a21 * n1
            if abs(b21) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / b21
                x2 = rhs_hi / b21
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            rhs_lo = -epsY_hi - target_t - a41 * n1
            rhs_hi = total_scale + epsY_hi - target_t - a41 * n1
            if abs(b41) <= tol:
                if not (rhs_lo < 0.0 + tol and 0.0 < rhs_hi + tol):
                    continue
            else:
                x1 = rhs_lo / b41
                x2 = rhs_hi / b41
                if x1 <= x2:
                    band_lo = math.ceil(x1 - tol)
                    band_hi = math.floor(x2 + tol)
                else:
                    band_lo = math.ceil(x2 - tol)
                    band_hi = math.floor(x1 + tol)
                if band_lo > lo:
                    lo = band_lo
                if band_hi < hi:
                    hi = band_hi
                if lo > hi:
                    continue

            for n2 in range(lo, hi + 1):
                c1 = n1 * alpha1 + n2 * alpha1_sq
                c2 = n1 * alpha2 + n2 * alpha2_sq
                c4 = n1 * alpha4 + n2 * alpha4_sq

                x = target_t - c1
                n0 = _round_nearest_int(x)
                s1 = n0 + c1
                abs_err = abs(s1 - target_t)

                if not (abs_err < epsY_hi + tol):
                    continue
                if epsY_lo > 0.0 and abs_err <= epsY_lo + tol:
                    continue
                if epsY_lo > 0.0 and abs_err <= epsY_lo + tol:
                    continue

                s2 = n0 + c2
                if s2 < -tol or s2 > total_scale + tol:
                    continue

                s4 = n0 + c4
                if s4 < -tol or s4 > total_scale + tol:
                    continue

                if use_inert_parity:
                    if not _passes_inert_parity(n0, n1, n2, inert_primes):
                        continue

                out_arr[pos, 0] = n0
                out_arr[pos, 1] = n1
                out_arr[pos, 2] = n2
                pos += 1

    return pos


# ============================================================
#  Buffered MPI transfer helpers for large row arrays
# ============================================================

def _buffered_gather_rows(comm, local_arr, root=0, rows_per_chunk=100000, base_tag=1000):
    """
    Gather 2D int64 arrays with shape (n,3) to root using chunked point-to-point transfers.
    Returns a list of arrays on root and None on non-root ranks.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    arr = np.ascontiguousarray(local_arr, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("local_arr must be 2D")
    nrows = int(arr.shape[0])
    ncols = int(arr.shape[1]) if arr.size else (int(arr.shape[1]) if arr.ndim == 2 else 0)

    meta_tag = base_tag
    data_tag = base_tag + 1
    mpi_dtype = MPI.INT64_T

    if rank == root:
        out = [None] * size
        out[root] = arr.copy()
        for src_rank in range(size):
            if src_rank == root:
                continue
            shape = comm.recv(source=src_rank, tag=meta_tag)
            rows, cols = int(shape[0]), int(shape[1])
            recv_arr = np.empty((rows, cols), dtype=np.int64)
            pos = 0
            while pos < rows:
                stop = min(pos + rows_per_chunk, rows)
                if stop > pos:
                    comm.Recv([recv_arr[pos:stop], mpi_dtype], source=src_rank, tag=data_tag)
                pos = stop
            out[src_rank] = recv_arr
        return out
    else:
        comm.send((nrows, ncols), dest=root, tag=meta_tag)
        pos = 0
        while pos < nrows:
            stop = min(pos + rows_per_chunk, nrows)
            if stop > pos:
                comm.Send([arr[pos:stop], mpi_dtype], dest=root, tag=data_tag)
            pos = stop
        return None


def _buffered_scatter_rows(comm, chunks, root=0, rows_per_chunk=100000, base_tag=2000):
    """
    Scatter a list of 2D int64 arrays with shape (n,3) from root using chunked point-to-point transfers.
    Returns the local array on each rank.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    meta_tag = base_tag
    data_tag = base_tag + 1
    mpi_dtype = MPI.INT64_T

    if rank == root:
        if chunks is None or len(chunks) != size:
            raise ValueError("root must provide chunks with one array per rank")
        local = np.ascontiguousarray(chunks[root], dtype=np.int64)
        for dst_rank in range(size):
            if dst_rank == root:
                continue
            arr = np.ascontiguousarray(chunks[dst_rank], dtype=np.int64)
            rows = int(arr.shape[0])
            cols = int(arr.shape[1]) if arr.size else int(arr.shape[1])
            comm.send((rows, cols), dest=dst_rank, tag=meta_tag)
            pos = 0
            while pos < rows:
                stop = min(pos + rows_per_chunk, rows)
                if stop > pos:
                    comm.Send([arr[pos:stop], mpi_dtype], dest=dst_rank, tag=data_tag)
                pos = stop
        return local
    else:
        rows, cols = comm.recv(source=root, tag=meta_tag)
        local = np.empty((int(rows), int(cols)), dtype=np.int64)
        pos = 0
        while pos < rows:
            stop = min(pos + rows_per_chunk, rows)
            if stop > pos:
                comm.Recv([local[pos:stop], mpi_dtype], source=root, tag=data_tag)
            pos = stop
        return local


# ============================================================
#  Streaming-mode helpers (out_format="streaming")
#
#  Drop-in replacement path for `_collect_targets_single_range_mpi` when the
#  in-memory algorithm would OOM. The existing memory path stays untouched
#  behind out_format="memory" (default). Streaming pipeline:
#
#    Phase A — per-rank: subdivide the polygon outer-axis into segments
#              (sized so per-segment count fits in ~32 MB); for each segment
#              call _count → np.empty(seg_count) → _fill → hash-partition
#              the segment into bucket files at `parts_dir/rank_NNNN/B_bucket_NNNNNN.bin`.
#              Caps per-rank peak memory at one segment's worth of hits.
#
#    Phase B — bucket-parallel: each rank pulls one bucket's fragments from
#              all ranks, runs _unique_sorted_rows (same semantics as
#              np.unique(axis=0)), then runs the exact ideal screen,
#              and appends survivors to `parts_dir/survivors_rank_NNNN.bin`.
#              Distributes the screen across all ranks; no root bottleneck.
#
#    Phase C — root: stream-concat survivor files behind a numpy .npy header
#              into the final --output path. Bit-for-bit equivalent .npy
#              format; downstream stages already re-unique on load, so the
#              bucket-grouped row order is acceptable (verified 2026-05-16).
#
#  Disk: per-rank bucket fragments live in parts_dir (default
#  output_file + ".s1_parts"); cleaned up post-finalize unless --keep_parts.
# ============================================================


def _streaming_parts_dir(output_file, override=None):
    if override:
        return override
    return output_file + ".s1_parts"


def _stream_bucket_file_for(parts_dir, rank, bucket_id):
    rank_dir = os.path.join(parts_dir, f"rank_{rank:04d}")
    return os.path.join(rank_dir, f"B_bucket_{bucket_id:06d}.bin")


def _stream_survivor_file_for(parts_dir, rank):
    return os.path.join(parts_dir, f"survivors_rank_{rank:04d}.bin")


def _stream_partition_segment_to_buckets(parts_dir, rank, seg_buf, n_buckets):
    """Hash seg_buf rows into n_buckets and append each bucket's slice to
    that rank's per-bucket file. Memory: O(seg_buf.shape[0])."""
    if seg_buf.shape[0] == 0:
        return
    bucket_ids = _bucket_linear_hash_rows(seg_buf, n_buckets, _STREAM_BUCKET_COEFFS)
    # Sort by bucket id once, then slice contiguous ranges -> one write per bucket.
    order = np.argsort(bucket_ids, kind="stable")
    sorted_ids = bucket_ids[order]
    sorted_rows = seg_buf[order]
    # Find run boundaries
    if sorted_ids.size > 1:
        diff = sorted_ids[1:] != sorted_ids[:-1]
        boundaries = np.flatnonzero(diff) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sorted_ids.size]))
    else:
        starts = np.array([0], dtype=np.int64)
        ends = np.array([sorted_ids.size], dtype=np.int64)
    for s, e in zip(starts, ends):
        bid = int(sorted_ids[s])
        path = _stream_bucket_file_for(parts_dir, rank, bid)
        _append_rows_raw(path, sorted_rows[s:e])


class _StreamBucketBuffer:
    """Per-rank write-back cache for bucket file appends.

    Replaces the prior per-segment-per-bucket _stream_partition_segment_to_buckets
    pattern (1000s of small appends per segment) with batched in-memory accumulation
    + periodic large writes. Critical for mechanical-disk parts_dir where small
    appends are seek-bound (~107 ms io_wait observed at f=6 u=1 ε=1e-5).

    Memory: holds bucketed rows for one rank up to ~flush_bytes (default 1 GB),
    well within per-rank budget on 125 GB machine.

    Usage:
        buf = _StreamBucketBuffer(parts_dir, rank, n_buckets, flush_bytes=1<<30)
        for each segment:
            buf.add(seg_rows_sorted_by_bid, starts, ends, sorted_ids)
        buf.flush()  # at end of phase A
    """
    def __init__(self, parts_dir, rank, n_buckets, flush_bytes=1 << 30):
        self.parts_dir = parts_dir
        self.rank = rank
        self.n_buckets = n_buckets
        self.flush_bytes = int(flush_bytes)
        # dict bid -> list[np.ndarray of shape (k, 3) int64]
        # NOTE: using list avoids repeated concat until flush
        self._buf = {}
        self._bytes = 0
        self.n_flushes = 0
        self.flush_time = 0.0
        # row size in bytes (3 int64)
        self._row_bytes = 24

    def add_slices(self, sorted_rows, starts, ends, sorted_ids):
        """Append per-bucket slices from a hash-sorted segment. Slices are
        contiguous views into sorted_rows. Total bytes added = sorted_rows.nbytes."""
        for s, e in zip(starts, ends):
            bid = int(sorted_ids[s])
            slc = sorted_rows[s:e]
            self._buf.setdefault(bid, []).append(np.ascontiguousarray(slc))
        self._bytes += int(sorted_rows.nbytes)
        if self._bytes >= self.flush_bytes:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        t0 = time.time()
        for bid, chunks in self._buf.items():
            if len(chunks) == 1:
                merged = chunks[0]
            else:
                merged = np.concatenate(chunks, axis=0)
            path = _stream_bucket_file_for(self.parts_dir, self.rank, bid)
            _append_rows_raw(path, merged)
        self._buf.clear()
        self._bytes = 0
        self.n_flushes += 1
        self.flush_time += (time.time() - t0)


def _stream_partition_segment_to_buffer(buf, seg_buf, n_buckets):
    """Buffered analog of _stream_partition_segment_to_buckets — instead of
    writing each bucket's slice directly to disk, hands the slices to the
    write-back cache `buf`. Auto-flushes when cache exceeds threshold."""
    if seg_buf.shape[0] == 0:
        return
    bucket_ids = _bucket_linear_hash_rows(seg_buf, n_buckets, _STREAM_BUCKET_COEFFS)
    order = np.argsort(bucket_ids, kind="stable")
    sorted_ids = bucket_ids[order]
    sorted_rows = seg_buf[order]
    if sorted_ids.size > 1:
        diff = sorted_ids[1:] != sorted_ids[:-1]
        boundaries = np.flatnonzero(diff) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sorted_ids.size]))
    else:
        starts = np.array([0], dtype=np.int64)
        ends = np.array([sorted_ids.size], dtype=np.int64)
    buf.add_slices(sorted_rows, starts, ends, sorted_ids)


# ============================================================
#  MPI-shuffle mode helpers (out_format="streaming", shuffle_mode="mpi")
#
#  Alternative to disk-based bucket fragments. Per-segment pipeline:
#    1. each rank fills its seg_buf, hashes rows to bucket_id, sorts by bid
#    2. each rank counts how many rows go to each destination rank
#       (destination = bid % size) and sends counts via comm.Alltoall
#    3. each rank packs send buffer in destination order, exchanges data via
#       comm.Alltoallv → each rank receives all rows for its owned buckets
#    4. received rows append to per-owned-bucket accumulator (in memory)
#    5. when total accumulator > screen_threshold_bytes, rank runs
#       _unique_sorted_rows + ideal screen on each owned bucket → survivors
#       appended to local survivor file; accumulator cleared.
#  At end of phase A: final flush of accumulators. Only survivors hit disk
#  (~1000× less than raw hits for u=1 at f≥6). Disk wait collapses to
#  the survivors-write time, not the raw-hits-write time.
# ============================================================


class _StreamShuffleAccumulator:
    """Per-rank per-bucket accumulator for the MPI-shuffle Phase A pipeline.

    Each rank owns buckets {rank, rank+size, rank+2*size, ...}. As rows for
    those buckets arrive via Alltoallv, they accumulate here. When total
    bytes exceed screen_threshold_bytes, every owned bucket is dedupped
    + ideal-screened, and survivors are appended to the rank's survivor file.

    Memory bound: at most screen_threshold_bytes of raw accumulated rows
    PLUS one bucket's worth during dedup+screen. With threshold=512 MB and
    typical bucket sizes <100 MB, per-rank peak ~700 MB. Well under
    Lenore's 4 GB/rank budget.
    """
    def __init__(self, parts_dir, rank, size, n_buckets,
                 screen_threshold_bytes=512 * 1024 * 1024,
                 use_exact_ideal_screen=True,
                 allow_negative_embeddings=False,
                 check_local_p3k=False,
                 profile=False,
                 local_field_data=None,
                 local_qs_profile=None,
                 preserve_survivor_file=False):
        self.parts_dir = parts_dir
        self.rank = rank
        self.size = size
        self.n_buckets = n_buckets
        self.screen_threshold_bytes = int(screen_threshold_bytes)
        self.use_exact_ideal_screen = use_exact_ideal_screen
        self.allow_negative_embeddings = allow_negative_embeddings
        self.check_local_p3k = check_local_p3k
        self.profile = profile
        self.local_field_data = local_field_data
        self.local_qs_profile = local_qs_profile

        self._owned_bids = set(range(rank, n_buckets, size))
        # dict bid -> list[np.ndarray (k, 3) int64]
        self._buf = {}
        self._bytes = 0
        self.local_pre_exact = 0
        self.local_kept_total = 0
        self.local_screen_rejects = 0
        self.n_flushes = 0
        self.screen_time = 0.0
        self.dedup_time = 0.0
        self.io_time = 0.0
        self.survivor_path = _stream_survivor_file_for(parts_dir, rank)
        if not preserve_survivor_file and os.path.exists(self.survivor_path):
            os.remove(self.survivor_path)

    def add_for_bucket(self, bid, rows):
        """Append received rows for a specific owned bucket id."""
        if rows.shape[0] == 0:
            return
        self._buf.setdefault(int(bid), []).append(np.ascontiguousarray(rows))
        self._bytes += int(rows.nbytes)
        if self._bytes >= self.screen_threshold_bytes:
            self.flush()

    def flush(self):
        """Dedup + ideal screen + append survivors for every accumulated bucket.
        Clears the buffer."""
        if not self._buf:
            return
        for bid in list(self._buf.keys()):
            chunks = self._buf[bid]
            if len(chunks) == 1:
                merged = chunks[0]
            else:
                merged = np.concatenate(chunks, axis=0)
            del self._buf[bid]
            t0 = time.time()
            uniq = _unique_sorted_rows(merged)
            self.dedup_time += (time.time() - t0)
            self.local_pre_exact += int(uniq.shape[0])
            if self.use_exact_ideal_screen:
                t0 = time.time()
                # FAST PATH: Numba batched screen (~1000× faster than per-row Sage).
                # Falls back to per-row Sage call for rows beyond table bound or
                # int64 overflow. Disabled by ZETA9_DISABLE_FAST_SCREEN=1.
                # Note: check_local_p3k is currently NOT supported by the fast path
                # (it's a rarely-used extra filter); fall back to Sage in that case.
                if _HAVE_FAST_SCREEN and not self.check_local_p3k and not self.profile:
                    def _sage_fb(m0, m1, m2):
                        if quick_screen_M_fast is not None:
                            r = quick_screen_M_fast(
                                m0, m1, m2,
                                check_real_embeddings=not self.allow_negative_embeddings,
                                check_local_p3k=self.check_local_p3k,
                                field_data=self.local_field_data,
                            )
                        else:
                            r = quick_screen_M(
                                m0, m1, m2,
                                check_real_embeddings=not self.allow_negative_embeddings,
                                check_local_p3k=self.check_local_p3k,
                                field_data=self.local_field_data,
                            )
                        return r["status"] in ("PASSES_IDEAL_SIEVE", "ZERO")
                    keep_mask, n_fb = _roots_fast.screen_rows_batch(
                        uniq,
                        check_real_embeddings=not self.allow_negative_embeddings,
                        sage_fallback_fn=_sage_fb,
                    )
                    self.local_screen_rejects += int(uniq.shape[0] - int(keep_mask.sum()))
                    kept = uniq[keep_mask]
                else:
                    keep_mask = np.zeros(uniq.shape[0], dtype=bool)
                    for irow in range(uniq.shape[0]):
                        row = uniq[irow]
                        if self.profile and self.local_qs_profile is not None and quick_screen_status_fast_profiled is not None:
                            status = quick_screen_status_fast_profiled(
                                int(row[0]), int(row[1]), int(row[2]),
                                check_real_embeddings=not self.allow_negative_embeddings,
                                check_local_p3k=self.check_local_p3k,
                                field_data=self.local_field_data,
                                profile=self.local_qs_profile,
                            )
                        elif self.profile and self.local_qs_profile is not None:
                            status = quick_screen_status_profiled(
                                int(row[0]), int(row[1]), int(row[2]),
                                check_real_embeddings=not self.allow_negative_embeddings,
                                check_local_p3k=self.check_local_p3k,
                                field_data=self.local_field_data,
                                profile=self.local_qs_profile,
                            )
                        else:
                            if quick_screen_M_fast is not None:
                                scr = quick_screen_M_fast(
                                    int(row[0]), int(row[1]), int(row[2]),
                                    check_real_embeddings=not self.allow_negative_embeddings,
                                    check_local_p3k=self.check_local_p3k,
                                    field_data=self.local_field_data,
                                )
                            else:
                                scr = quick_screen_M(
                                    int(row[0]), int(row[1]), int(row[2]),
                                    check_real_embeddings=not self.allow_negative_embeddings,
                                    check_local_p3k=self.check_local_p3k,
                                    field_data=self.local_field_data,
                                )
                            status = scr["status"]
                        if status == "PASSES_IDEAL_SIEVE" or status == "ZERO":
                            keep_mask[irow] = True
                        else:
                            self.local_screen_rejects += 1
                    kept = uniq[keep_mask]
                self.screen_time += (time.time() - t0)
            else:
                kept = uniq
            self.local_kept_total += int(kept.shape[0])
            t0 = time.time()
            _append_rows_raw(self.survivor_path, kept)
            self.io_time += (time.time() - t0)
        self._bytes = 0
        self.n_flushes += 1


def _stream_shuffle_segment(comm, rank, size, seg_buf, n_buckets, accumulator):
    """Hash seg_buf rows by bucket id, group by destination rank
    (dest = bid % size), exchange via MPI Alltoallv. Received rows are
    fed to the accumulator (which buckets them by bid and triggers
    flush + screen as needed)."""
    if seg_buf.shape[0] == 0:
        # Still need to participate in Alltoall to keep ranks in sync.
        sendcounts = np.zeros(size, dtype=np.int32)
    else:
        bucket_ids = _bucket_linear_hash_rows(seg_buf, n_buckets, _STREAM_BUCKET_COEFFS)
        # destination rank for each row = bid % size
        dest_ranks = (bucket_ids % size).astype(np.int32)
        # sort by (dest_rank, bid) so within each dest's slab, buckets are contiguous
        order = np.lexsort((bucket_ids, dest_ranks))
        sorted_dests = dest_ranks[order]
        sorted_bids = bucket_ids[order]
        sorted_rows = seg_buf[order]
        sendcounts = np.bincount(sorted_dests, minlength=size).astype(np.int32)
    # Counts in ELEMENTS of int64 (each row = 3 elements)
    sendcounts_int64 = sendcounts.astype(np.int64) * 3
    senddispls_int64 = np.empty(size, dtype=np.int64)
    senddispls_int64[0] = 0
    senddispls_int64[1:] = np.cumsum(sendcounts_int64[:-1])

    recvcounts_int64 = np.empty(size, dtype=np.int64)
    # MPI Alltoall: counts (in int64-element units, but each count is just an int64)
    comm.Alltoall(sendcounts_int64, recvcounts_int64)

    recvdispls_int64 = np.empty(size, dtype=np.int64)
    recvdispls_int64[0] = 0
    recvdispls_int64[1:] = np.cumsum(recvcounts_int64[:-1])

    total_recv = int(recvcounts_int64.sum())
    recvbuf = np.empty(total_recv, dtype=np.int64)

    if seg_buf.shape[0] == 0:
        sendbuf = np.empty(0, dtype=np.int64)
    else:
        sendbuf = sorted_rows.reshape(-1)  # flatten to int64 row-major

    # mpi4py Alltoallv with int64 counts/displs cast to int — needs python ints
    sendcounts_py = [int(c) for c in sendcounts_int64]
    senddispls_py = [int(c) for c in senddispls_int64]
    recvcounts_py = [int(c) for c in recvcounts_int64]
    recvdispls_py = [int(c) for c in recvdispls_int64]
    comm.Alltoallv(
        [sendbuf, (sendcounts_py, senddispls_py), MPI.INT64_T],
        [recvbuf, (recvcounts_py, recvdispls_py), MPI.INT64_T],
    )

    # Each rank now has its rows. Re-derive bucket ids on the recv side and
    # group by bid. (Could send bids alongside rows to skip, but recomputing
    # the hash on int64 is cheap.)
    if total_recv == 0:
        return
    received = recvbuf.reshape(-1, 3)
    recv_bids = _bucket_linear_hash_rows(received, n_buckets, _STREAM_BUCKET_COEFFS)
    # Group by bid for the accumulator
    order2 = np.argsort(recv_bids, kind="stable")
    sb = recv_bids[order2]
    sr = received[order2]
    if sb.size > 1:
        diff = sb[1:] != sb[:-1]
        boundaries = np.flatnonzero(diff) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sb.size]))
    else:
        starts = np.array([0], dtype=np.int64)
        ends = np.array([sb.size], dtype=np.int64)
    for s, e in zip(starts, ends):
        accumulator.add_for_bucket(int(sb[s]), sr[s:e])


def _stream_seg_manifest_path(parts_dir):
    return os.path.join(parts_dir, "seg_manifest.json")


def _stream_load_seg_manifest(parts_dir, expected_params):
    """Read segment-checkpoint manifest. Returns dict or None if absent/incompatible."""
    p = _stream_seg_manifest_path(parts_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r") as fh:
            m = json.load(fh)
    except Exception:
        return None
    # Validate params match (incompatible params → ignore old manifest)
    for k in ("f", "u", "eps", "norm", "epsY_lo", "epsY_hi", "size", "n_buckets", "n_segments_target"):
        if k in expected_params and m.get(k) != expected_params[k]:
            return None
    return m


def _stream_save_seg_manifest(parts_dir, manifest):
    p = _stream_seg_manifest_path(parts_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def _stream_phase_A_shuffle(
    parts_dir, rank, size, comm,
    total_scale, target_t, epsY_lo, epsY_hi,
    n1_lo, n1_hi, n2_lo, n2_hi,
    alpha1, alpha2, alpha4,
    inert_primes_arr, use_inert_parity, sweep_axis,
    n_buckets, n_segments_target, verbose,
    use_exact_ideal_screen, allow_negative_embeddings,
    check_local_p3k, profile,
    screen_threshold_bytes=512 * 1024 * 1024,
    f_for_manifest=None, u_for_manifest=None, eps_for_manifest=None,
    norm_for_manifest=None,
    checkpoint_every=1,
    resume=False,
):
    """Per-rank streaming fill + per-segment MPI shuffle into per-bucket
    accumulator with periodic dedup+screen+flush.

    Replaces _stream_phase_A + _stream_phase_B disk-based path. Returns
    summary dict including all per-rank aggregates needed by the caller.

    Segment-level checkpointing: every `checkpoint_every` segments, force
    a full accumulator flush + write seg_manifest.json with completed_through.
    If resume=True and a compatible manifest exists, skip already-completed
    segments. Per-rank stats (count, checked_pairs, etc.) for skipped
    segments are recovered from the manifest.
    """
    if use_exact_ideal_screen:
        if quick_screen_M is None:
            raise RuntimeError("use_exact_ideal_screen requested, but zeta9.roots.quick_screen_M is unavailable")
        local_field_data = build_fields() if build_fields is not None else None
    else:
        local_field_data = None
    local_qs_profile = None
    if use_exact_ideal_screen and profile and new_quick_screen_profile is not None and quick_screen_status_profiled is not None:
        local_qs_profile = new_quick_screen_profile()

    if sweep_axis == 0:
        outer_lo, outer_hi = n2_lo, n2_hi
    else:
        outer_lo, outer_hi = n1_lo, n1_hi
    seg_count, seg_size_outer = _stream_compute_segment_outer(
        outer_lo, outer_hi, size, n_segments_target
    )

    # ---- Resume logic (DECIDE BEFORE accumulator init, so we know whether to preserve survivor file) ----
    expected_params = {
        "f": f_for_manifest, "u": u_for_manifest, "eps": eps_for_manifest,
        "norm": norm_for_manifest,
        "epsY_lo": float(epsY_lo), "epsY_hi": float(epsY_hi),
        "size": int(size), "n_buckets": int(n_buckets),
        "n_segments_target": int(n_segments_target),
    }
    resume_from = 0
    resume_stats = None  # per-rank stats array, only on rank 0
    if resume and rank == 0:
        m = _stream_load_seg_manifest(parts_dir, expected_params)
        if m is not None and m.get("completed_through", -1) >= 0:
            resume_from = int(m["completed_through"]) + 1
            resume_stats = m.get("per_rank_stats")
            print(f"phase A (shuffle): RESUMING from segment {resume_from}/{seg_count}", flush=True)
    resume_from = comm.bcast(resume_from, root=0)
    resume_stats = comm.bcast(resume_stats, root=0)

    # Accumulator: preserve survivor file on resume so we keep checkpointed data
    acc = _StreamShuffleAccumulator(
        parts_dir, rank, size, n_buckets,
        screen_threshold_bytes=screen_threshold_bytes,
        use_exact_ideal_screen=use_exact_ideal_screen,
        allow_negative_embeddings=allow_negative_embeddings,
        check_local_p3k=check_local_p3k,
        profile=profile,
        local_field_data=local_field_data,
        local_qs_profile=local_qs_profile,
        preserve_survivor_file=(resume_from > 0),
    )

    t_phase_A0 = time.time()
    local_count = 0
    local_checked_n2 = 0
    local_checked_pairs = 0
    local_passed_strip_pairs = 0
    local_near_integer_pairs = 0
    local_feasible_pairs = 0
    local_parity_rejects = 0
    local_count_time = 0.0
    local_fill_time = 0.0
    local_shuffle_time = 0.0
    local_alloc_time = 0.0
    last_report = time.time()
    # Per-phase timing for last-N-seg breakdown (rank 0)
    _dbg_pre = {"count": 0.0, "alloc": 0.0, "fill": 0.0, "shuf": 0.0,
                "dedup": 0.0, "screen": 0.0, "io": 0.0}
    _dbg_pre["dedup"] = float(acc.dedup_time)
    _dbg_pre["screen"] = float(acc.screen_time)
    _dbg_pre["io"] = float(acc.io_time)
    DBG_TIMING = os.environ.get("ZETA9_PHASE_TIMING", "") in ("1", "true", "yes")

    # Restore aggregates from manifest (per-rank stats)
    if resume_from > 0 and resume_stats is not None and rank < len(resume_stats):
        s = resume_stats[rank]
        local_count = int(s.get("local_count", 0))
        local_checked_n2 = int(s.get("local_checked_n2", 0))
        local_checked_pairs = int(s.get("local_checked_pairs", 0))
        local_passed_strip_pairs = int(s.get("local_passed_strip_pairs", 0))
        local_near_integer_pairs = int(s.get("local_near_integer_pairs", 0))
        local_feasible_pairs = int(s.get("local_feasible_pairs", 0))
        local_parity_rejects = int(s.get("local_parity_rejects", 0))
        # Also restore accumulator aggregates so final report is accurate
        acc.local_pre_exact = int(s.get("acc_local_pre_exact", 0))
        acc.local_kept_total = int(s.get("acc_local_kept_total", 0))
        acc.local_screen_rejects = int(s.get("acc_local_screen_rejects", 0))

    for seg_idx in range(resume_from, seg_count):
        seg_lo = outer_lo + seg_idx * seg_size_outer
        seg_hi = min(outer_lo + (seg_idx + 1) * seg_size_outer - 1, outer_hi)
        if sweep_axis == 0:
            seg_n1_lo, seg_n1_hi = n1_lo, n1_hi
            seg_n2_lo, seg_n2_hi = seg_lo, seg_hi
        else:
            seg_n1_lo, seg_n1_hi = seg_lo, seg_hi
            seg_n2_lo, seg_n2_hi = n2_lo, n2_hi

        t_count0 = time.time()
        seg_local_count, sc_n2, sc_pairs, sc_strip, sc_near, sc_feas, sc_parity = _count_chunk_target(
            total_scale, target_t, epsY_lo, epsY_hi,
            seg_n1_lo, seg_n1_hi, seg_n2_lo, seg_n2_hi,
            rank, size,
            alpha1, alpha2, alpha4,
            inert_primes_arr, use_inert_parity, sweep_axis,
        )
        local_count_time += (time.time() - t_count0)
        local_count += int(seg_local_count)
        local_checked_n2 += int(sc_n2)
        local_checked_pairs += int(sc_pairs)
        local_passed_strip_pairs += int(sc_strip)
        local_near_integer_pairs += int(sc_near)
        local_feasible_pairs += int(sc_feas)
        local_parity_rejects += int(sc_parity)

        if seg_local_count == 0:
            # Still participate in the Alltoallv so other ranks don't block
            seg_buf = np.empty((0, 3), dtype=np.int64)
        else:
            t_alloc0 = time.time()
            seg_buf = np.empty((int(seg_local_count), 3), dtype=np.int64)
            local_alloc_time += (time.time() - t_alloc0)
            t_fill0 = time.time()
            filled = _fill_chunk_target(
                seg_buf, total_scale, target_t, epsY_lo, epsY_hi,
                seg_n1_lo, seg_n1_hi, seg_n2_lo, seg_n2_hi,
                rank, size,
                alpha1, alpha2, alpha4,
                inert_primes_arr, use_inert_parity, sweep_axis,
            )
            local_fill_time += (time.time() - t_fill0)
            if filled != seg_local_count:
                raise RuntimeError(
                    f"Shuffle rank {rank} seg {seg_idx}: filled={filled} != counted={seg_local_count}"
                )

        t_shuf0 = time.time()
        _stream_shuffle_segment(comm, rank, size, seg_buf, n_buckets, acc)
        local_shuffle_time += (time.time() - t_shuf0)
        del seg_buf

        if rank == 0 and verbose and (time.time() - last_report > 10.0):
            last_report = time.time()
            pct = 100.0 * (seg_idx + 1) / seg_count
            if DBG_TIMING:
                dt_count = local_count_time - _dbg_pre["count"]
                dt_alloc = local_alloc_time - _dbg_pre["alloc"]
                dt_fill = local_fill_time - _dbg_pre["fill"]
                dt_shuf = local_shuffle_time - _dbg_pre["shuf"]
                dt_dedup = float(acc.dedup_time) - _dbg_pre["dedup"]
                dt_screen = float(acc.screen_time) - _dbg_pre["screen"]
                dt_io = float(acc.io_time) - _dbg_pre["io"]
                _dbg_pre["count"] = local_count_time
                _dbg_pre["alloc"] = local_alloc_time
                _dbg_pre["fill"] = local_fill_time
                _dbg_pre["shuf"] = local_shuffle_time
                _dbg_pre["dedup"] = float(acc.dedup_time)
                _dbg_pre["screen"] = float(acc.screen_time)
                _dbg_pre["io"] = float(acc.io_time)
                tot = dt_count + dt_alloc + dt_fill + dt_shuf + dt_dedup + dt_screen + dt_io
                if tot > 0:
                    print(f"phase A (shuffle) rank 0: {pct:.1f}% (segs {seg_idx+1}/{seg_count}, hits {local_count}, flushes {acc.n_flushes})  "
                          f"PHASE TIMES (last {tot:.1f}s):  "
                          f"count={dt_count:.1f}({100*dt_count/tot:.0f}%) "
                          f"alloc={dt_alloc:.1f}({100*dt_alloc/tot:.0f}%) "
                          f"fill={dt_fill:.1f}({100*dt_fill/tot:.0f}%) "
                          f"shuf={dt_shuf:.1f}({100*dt_shuf/tot:.0f}%) "
                          f"dedup={dt_dedup:.1f}({100*dt_dedup/tot:.0f}%) "
                          f"screen={dt_screen:.1f}({100*dt_screen/tot:.0f}%) "
                          f"io={dt_io:.1f}({100*dt_io/tot:.0f}%)",
                          flush=True)
                else:
                    print(f"phase A (shuffle) rank 0: {pct:.1f}% (segs {seg_idx+1}/{seg_count}, hits {local_count}, flushes {acc.n_flushes})", flush=True)
            else:
                print(f"phase A (shuffle), rank 0 progress: {pct:.1f}% (segs {seg_idx+1}/{seg_count}, hits {local_count}, accum_bytes {acc._bytes//(1<<20)}MB, flushes {acc.n_flushes}, kept {acc.local_kept_total})", flush=True)

        # ---- Checkpoint at segment boundary ----
        # Order: (1) all ranks force-flush accumulator → survivor file durable on disk
        #        (2) barrier to ensure all flushes done
        #        (3) gather per-rank stats to root
        #        (4) root writes manifest atomically (manifest is the ONLY completion marker)
        #        (5) barrier so all ranks see consistent state
        # If we crash anywhere in 1-3 or before (4) completes, manifest is unchanged
        # (still points to previous checkpoint). The survivor file might have extra
        # rows from segments that crash-resumed will re-do — those duplicates are
        # eliminated by the per-bucket dedup on next flush, and by stage 2's _load_stack.
        if checkpoint_every > 0 and ((seg_idx + 1) % checkpoint_every == 0):
            acc.flush()
            comm.Barrier()
            local_stats = {
                "local_count": int(local_count),
                "local_checked_n2": int(local_checked_n2),
                "local_checked_pairs": int(local_checked_pairs),
                "local_passed_strip_pairs": int(local_passed_strip_pairs),
                "local_near_integer_pairs": int(local_near_integer_pairs),
                "local_feasible_pairs": int(local_feasible_pairs),
                "local_parity_rejects": int(local_parity_rejects),
                "acc_local_pre_exact": int(acc.local_pre_exact),
                "acc_local_kept_total": int(acc.local_kept_total),
                "acc_local_screen_rejects": int(acc.local_screen_rejects),
            }
            all_stats = comm.gather(local_stats, root=0)
            if rank == 0:
                m = dict(expected_params)
                m["completed_through"] = int(seg_idx)
                m["seg_count"] = int(seg_count)
                m["per_rank_stats"] = all_stats
                _stream_save_seg_manifest(parts_dir, m)
            comm.Barrier()

    # Final flush of accumulator (dedup + screen remaining buckets)
    acc.flush()
    # Final manifest after all segments
    if checkpoint_every > 0 and seg_count > 0 and rank == 0:
        m = dict(expected_params)
        m["completed_through"] = int(seg_count - 1)
        m["seg_count"] = int(seg_count)
        m["per_rank_stats"] = None  # post-final not used for resume
        _stream_save_seg_manifest(parts_dir, m)
    comm.Barrier()

    if rank == 0 and verbose:
        print(f"phase A (shuffle), rank 0 progress: 100% (hits {local_count}, total flushes {acc.n_flushes}, kept {acc.local_kept_total}, screen_rej {acc.local_screen_rejects})", flush=True)

    comm.Barrier()
    return {
        "local_count": int(local_count),
        "local_checked_n2": int(local_checked_n2),
        "local_checked_pairs": int(local_checked_pairs),
        "local_passed_strip_pairs": int(local_passed_strip_pairs),
        "local_near_integer_pairs": int(local_near_integer_pairs),
        "local_feasible_pairs": int(local_feasible_pairs),
        "local_parity_rejects": int(local_parity_rejects),
        "local_fill_time": float(local_fill_time),
        "local_shuffle_time": float(local_shuffle_time),
        "local_phase_A_time": float(time.time() - t_phase_A0),
        "seg_count": int(seg_count),
        "seg_size_outer": int(seg_size_outer),
        "local_pre_exact": int(acc.local_pre_exact),
        "local_kept_total": int(acc.local_kept_total),
        "local_screen_rejects": int(acc.local_screen_rejects),
        "local_n_flushes": int(acc.n_flushes),
        "local_dedup_time": float(acc.dedup_time),
        "local_screen_time": float(acc.screen_time),
        "local_io_time": float(acc.io_time),
        "local_qs_profile": local_qs_profile,
    }


def _stream_compute_segment_outer(outer_lo, outer_hi, size, n_segments_target):
    """Return (seg_count, seg_size_outer) such that:
        - the outer-axis range [outer_lo, outer_hi] is split into seg_count
          segments of seg_size_outer outer indices each (last may be shorter)
        - each segment, after rank-striping (stride=size), gives each rank
          roughly seg_size_outer/size outer iterations to process
    Pick seg_size_outer ~ size * 64 by default — covers ~64 outer iterations
    per rank per segment, keeps the per-segment count phase quick, and bounds
    per-segment fill output. n_segments_target is a HINT used as an upper
    bound on segment count when the range is large."""
    n_total = max(1, outer_hi - outer_lo + 1)
    seg_size_outer = max(size * 64, (n_total + n_segments_target - 1) // n_segments_target)
    seg_count = (n_total + seg_size_outer - 1) // seg_size_outer
    return seg_count, seg_size_outer


def _stream_phase_A(
    parts_dir, rank, size, comm,
    total_scale, target_t, epsY_lo, epsY_hi,
    n1_lo, n1_hi, n2_lo, n2_hi,
    alpha1, alpha2, alpha4,
    inert_primes_arr, use_inert_parity, sweep_axis,
    n_buckets, n_segments_target, verbose,
    flush_bytes=1 << 30,
):
    """Per-rank streaming fill + hash-partition. Returns dict with local
    aggregates (count, time, etc) so root can build a profile entry."""
    os.makedirs(os.path.join(parts_dir, f"rank_{rank:04d}"), exist_ok=True)
    buf = _StreamBucketBuffer(parts_dir, rank, n_buckets, flush_bytes=flush_bytes)
    if sweep_axis == 0:
        outer_lo, outer_hi = n2_lo, n2_hi
    else:
        outer_lo, outer_hi = n1_lo, n1_hi
    seg_count, seg_size_outer = _stream_compute_segment_outer(
        outer_lo, outer_hi, size, n_segments_target
    )

    t_phase_A0 = time.time()
    local_count = 0
    local_checked_n2 = 0
    local_checked_pairs = 0
    local_passed_strip_pairs = 0
    local_near_integer_pairs = 0
    local_feasible_pairs = 0
    local_parity_rejects = 0
    local_fill_time = 0.0
    local_partition_time = 0.0
    last_report = time.time()

    for seg_idx in range(seg_count):
        seg_lo = outer_lo + seg_idx * seg_size_outer
        seg_hi = min(outer_lo + (seg_idx + 1) * seg_size_outer - 1, outer_hi)
        if sweep_axis == 0:
            seg_n1_lo, seg_n1_hi = n1_lo, n1_hi
            seg_n2_lo, seg_n2_hi = seg_lo, seg_hi
        else:
            seg_n1_lo, seg_n1_hi = seg_lo, seg_hi
            seg_n2_lo, seg_n2_hi = n2_lo, n2_hi

        seg_local_count, sc_n2, sc_pairs, sc_strip, sc_near, sc_feas, sc_parity = _count_chunk_target(
            total_scale, target_t, epsY_lo, epsY_hi,
            seg_n1_lo, seg_n1_hi, seg_n2_lo, seg_n2_hi,
            rank, size,
            alpha1, alpha2, alpha4,
            inert_primes_arr, use_inert_parity, sweep_axis,
        )
        local_count += int(seg_local_count)
        local_checked_n2 += int(sc_n2)
        local_checked_pairs += int(sc_pairs)
        local_passed_strip_pairs += int(sc_strip)
        local_near_integer_pairs += int(sc_near)
        local_feasible_pairs += int(sc_feas)
        local_parity_rejects += int(sc_parity)

        if seg_local_count == 0:
            if rank == 0 and verbose and (time.time() - last_report > 10.0):
                last_report = time.time()
                pct = 100.0 * (seg_idx + 1) / seg_count
                print(f"phase A, rank 0 segment progress: {pct:.1f}% (empty seg)", flush=True)
            continue

        seg_buf = np.empty((int(seg_local_count), 3), dtype=np.int64)
        t_fill0 = time.time()
        filled = _fill_chunk_target(
            seg_buf, total_scale, target_t, epsY_lo, epsY_hi,
            seg_n1_lo, seg_n1_hi, seg_n2_lo, seg_n2_hi,
            rank, size,
            alpha1, alpha2, alpha4,
            inert_primes_arr, use_inert_parity, sweep_axis,
        )
        local_fill_time += (time.time() - t_fill0)
        if filled != seg_local_count:
            raise RuntimeError(
                f"Streaming rank {rank} seg {seg_idx}: filled={filled} != counted={seg_local_count}"
            )

        t_part0 = time.time()
        _stream_partition_segment_to_buffer(buf, seg_buf, n_buckets)
        local_partition_time += (time.time() - t_part0)
        # Free the segment buffer before next allocation
        del seg_buf

        if rank == 0 and verbose and (time.time() - last_report > 10.0):
            last_report = time.time()
            pct = 100.0 * (seg_idx + 1) / seg_count
            print(f"phase A, rank 0 segment progress: {pct:.1f}% (segs {seg_idx+1}/{seg_count}, hits so far {local_count}, flushes {buf.n_flushes})", flush=True)

    # Final flush of any remaining buffered rows
    t_part0 = time.time()
    buf.flush()
    local_partition_time += (time.time() - t_part0)

    if rank == 0 and verbose:
        print(f"phase A, rank 0 segment progress: 100% (segs {seg_count}/{seg_count}, hits {local_count}, total flushes {buf.n_flushes}, flush_time {buf.flush_time:.1f}s)", flush=True)

    comm.Barrier()
    return {
        "local_count": int(local_count),
        "local_checked_n2": int(local_checked_n2),
        "local_checked_pairs": int(local_checked_pairs),
        "local_passed_strip_pairs": int(local_passed_strip_pairs),
        "local_near_integer_pairs": int(local_near_integer_pairs),
        "local_feasible_pairs": int(local_feasible_pairs),
        "local_parity_rejects": int(local_parity_rejects),
        "local_fill_time": float(local_fill_time),
        "local_partition_time": float(local_partition_time),
        "local_phase_A_time": float(time.time() - t_phase_A0),
        "seg_count": int(seg_count),
        "seg_size_outer": int(seg_size_outer),
    }


def _stream_phase_B(
    parts_dir, rank, size, comm,
    n_buckets,
    use_exact_ideal_screen, allow_negative_embeddings,
    check_local_p3k, profile, verbose,
):
    """Bucket-parallel dedup + ideal screen. Each rank handles bucket ids
    {rank, rank+size, rank+2*size, ...}. Survivors append to per-rank
    survivor file."""
    survivor_path = _stream_survivor_file_for(parts_dir, rank)
    if os.path.exists(survivor_path):
        os.remove(survivor_path)

    if use_exact_ideal_screen:
        if quick_screen_M is None:
            raise RuntimeError("use_exact_ideal_screen requested, but zeta9.roots.quick_screen_M is unavailable")
        local_field_data = build_fields() if build_fields is not None else None
    else:
        local_field_data = None

    local_qs_profile = None
    if use_exact_ideal_screen and profile and new_quick_screen_profile is not None and quick_screen_status_profiled is not None:
        local_qs_profile = new_quick_screen_profile()

    local_pre_exact = 0
    local_kept_total = 0
    local_screen_rejects = 0
    t_phase_B0 = time.time()
    last_report = time.time()
    my_buckets = list(range(rank, n_buckets, size))

    for nth, my_bid in enumerate(my_buckets):
        frags = []
        for r in range(size):
            bp = _stream_bucket_file_for(parts_dir, r, my_bid)
            if os.path.exists(bp) and os.path.getsize(bp) > 0:
                arr = np.fromfile(bp, dtype=ROW_DTYPE)
                if arr.size:
                    frags.append(arr.reshape(-1, 3))
        if not frags:
            continue
        rows = np.concatenate(frags, axis=0) if len(frags) > 1 else frags[0]
        rows = _unique_sorted_rows(rows)
        local_pre_exact += int(rows.shape[0])

        if not use_exact_ideal_screen:
            kept = rows
        else:
            keep_mask = np.zeros(rows.shape[0], dtype=bool)
            for irow in range(rows.shape[0]):
                row = rows[irow]
                if profile and local_qs_profile is not None and quick_screen_status_fast_profiled is not None:
                    status = quick_screen_status_fast_profiled(
                        int(row[0]), int(row[1]), int(row[2]),
                        check_real_embeddings=not allow_negative_embeddings,
                        check_local_p3k=check_local_p3k,
                        field_data=local_field_data,
                        profile=local_qs_profile,
                    )
                elif profile and local_qs_profile is not None:
                    status = quick_screen_status_profiled(
                        int(row[0]), int(row[1]), int(row[2]),
                        check_real_embeddings=not allow_negative_embeddings,
                        check_local_p3k=check_local_p3k,
                        field_data=local_field_data,
                        profile=local_qs_profile,
                    )
                else:
                    if quick_screen_M_fast is not None:
                        scr = quick_screen_M_fast(
                            int(row[0]), int(row[1]), int(row[2]),
                            check_real_embeddings=not allow_negative_embeddings,
                            check_local_p3k=check_local_p3k,
                            field_data=local_field_data,
                        )
                    else:
                        scr = quick_screen_M(
                            int(row[0]), int(row[1]), int(row[2]),
                            check_real_embeddings=not allow_negative_embeddings,
                            check_local_p3k=check_local_p3k,
                            field_data=local_field_data,
                        )
                    status = scr["status"]
                if status == "PASSES_IDEAL_SIEVE" or status == "ZERO":
                    keep_mask[irow] = True
                else:
                    local_screen_rejects += 1
            kept = rows[keep_mask]
        local_kept_total += int(kept.shape[0])
        _append_rows_raw(survivor_path, kept)

        if rank == 0 and verbose and (time.time() - last_report > 10.0):
            last_report = time.time()
            pct = 100.0 * (nth + 1) / max(1, len(my_buckets))
            print(f"phase B, rank 0 bucket progress: {pct:.1f}% (buckets {nth+1}/{len(my_buckets)}, kept {local_kept_total}, rej {local_screen_rejects})", flush=True)

    if rank == 0 and verbose:
        print(f"phase B, rank 0 bucket progress: 100% (kept {local_kept_total}, rej {local_screen_rejects})", flush=True)

    comm.Barrier()
    return {
        "local_pre_exact": int(local_pre_exact),
        "local_kept_total": int(local_kept_total),
        "local_screen_rejects": int(local_screen_rejects),
        "local_phase_B_time": float(time.time() - t_phase_B0),
        "local_qs_profile": local_qs_profile,
    }


def _stream_phase_C_finalize(parts_dir, output_file, size):
    """Root: stream-concat per-rank survivor files behind a .npy header."""
    survivor_paths = [_stream_survivor_file_for(parts_dir, r) for r in range(size)]
    survivor_paths = [p for p in survivor_paths if os.path.exists(p)]
    total_bytes = sum(os.path.getsize(p) for p in survivor_paths)
    bytes_per_row = 3 * 8
    if total_bytes % bytes_per_row != 0:
        raise RuntimeError(
            f"Survivor files total bytes {total_bytes} is not divisible by row size {bytes_per_row}"
        )
    n_rows = total_bytes // bytes_per_row

    save_path = output_file if output_file.endswith(".npy") else output_file + ".npy"
    # np.lib.format.write_array_header_1_0 expects a writeable binary file.
    with open(save_path, "wb") as fh:
        np.lib.format.write_array_header_1_0(fh, {
            "descr": "<i8",
            "fortran_order": False,
            "shape": (int(n_rows), 3),
        })
        copy_buf_size = 8 * 1024 * 1024
        for sp in survivor_paths:
            with open(sp, "rb") as src:
                while True:
                    chunk = src.read(copy_buf_size)
                    if not chunk:
                        break
                    fh.write(chunk)
    return save_path, int(n_rows)


# ============================================================
#  Main MPI wrapper
# ============================================================

def _collect_targets_single_range_mpi(
    f,
    u,
    eps,
    output_file,
    norm=2,
    inert_prime_bound=29,
    use_inert_parity=False,
    use_exact_ideal_screen=False,
    allow_negative_embeddings=False,
    check_local_p3k=False,
    verbose=True,
    profile=False,
    mpi_rows_per_chunk=10000,
    epsY_lo=None,
    epsY_hi=None,
    out_format="memory",
    n_buckets=4096,
    n_segments_target=256,
    streaming_workdir=None,
    keep_parts=False,
    shuffle_mode="disk",
    shuffle_screen_threshold_mb=512,
    checkpoint_every=10,
    resume_streaming=False,
):
    if not (0.0 <= u <= norm):
        raise ValueError("u = |d_i|^2 must lie in [0, norm].")
    if not (eps < 0.5):
        raise ValueError("This code assumes eps < 0.5 so n0 is uniquely determined by rounding.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    wall_t0 = time.time()

    base_scale = float(3 ** (2 * f))                  # for t_i
    total_scale = float(norm * (3 ** (2 * f)))  # for sum_i Y_i
    total_scale_int = int(round(total_scale))

    target_t = base_scale * float(u)

    epsY_total = base_scale * eps * (2 * float(u)**0.5 + eps)
    if epsY_lo is None:
        epsY_lo = 0.0
    if epsY_hi is None:
        epsY_hi = epsY_total

    s1_lo = max(0.0, target_t - epsY_hi)
    s1_hi = min(total_scale, target_t + epsY_hi)

    verts = strip_polygon_vertices(target_t, epsY_hi, total_scale)
    sweep_axis, n1_lo, n1_hi, n2_lo, n2_hi = choose_sweep_from_polygon(verts)

    inert_primes = inert_primes_up_to(inert_prime_bound) if use_inert_parity else []
    inert_primes_arr = np.array(inert_primes, dtype=np.int64)

    if rank == 0 and verbose:
        print(f"f = {f}")
        print(f"base_scale = 3^(2f) = {int(base_scale)}")
        print(f"norm = {norm}")
        print(f"total_scale = norm * 3^(2f) = {total_scale_int}")
        print(f"u = |d_i|^2 = {u}")
        print(f"input vector epsilon = {eps}")
        print(f"internal target t = 3^(2f) * u = {target_t}")
        print(f"internal Y tolerance epsY_total = {epsY_total}")
        print(f"active epsY bin = ({epsY_lo}, {epsY_hi}]")
        print(f"MPI ranks = {size}")
        label_outer = "n2 rows" if sweep_axis == 0 else "n1 columns"
        print(f"search n1 in [{n1_lo}, {n1_hi}] (polygon bounds)")
        print(f"search n2 in [{n2_lo}, {n2_hi}] (polygon bounds)")
        print(f"sweep direction = {label_outer}")
        print(f"use_inert_parity = {use_inert_parity}")
        if use_inert_parity:
            print(f"inert primes up to {inert_prime_bound}: {inert_primes}")
        print(f"use_exact_ideal_screen = {use_exact_ideal_screen}")
        print(f"allow_negative_embeddings = {allow_negative_embeddings}")
        print(f"output file = {output_file}")
        print(f"mpi_rows_per_chunk = {mpi_rows_per_chunk}")
        print(f"out_format = {out_format}")
        if out_format == "streaming":
            print(f"streaming n_buckets = {n_buckets}")
            print(f"streaming n_segments_target = {n_segments_target}")
            print(f"streaming parts_dir = {_streaming_parts_dir(output_file, streaming_workdir)}")
        print(f"[BUCKET START] f={f} u={u} eps={eps} epsY=({epsY_lo:.4g},{epsY_hi:.4g}] -> {os.path.basename(output_file)}", flush=True)

    if out_format not in ("memory", "streaming"):
        raise ValueError(f"out_format must be 'memory' or 'streaming', got {out_format!r}")

    if out_format == "memory":
        t_count0 = time.time()
        (
            local_count,
            local_checked_n2,
            local_checked_pairs,
            local_passed_strip_pairs,
            local_near_integer_pairs,
            local_feasible_pairs,
            local_parity_rejects,
        ) = _count_chunk_target(
            total_scale,
            target_t,
            epsY_lo,
            epsY_hi,
            n1_lo, n1_hi,
            n2_lo, n2_hi,
            rank, size,
            ALPHA1, ALPHA2, ALPHA4,
            inert_primes_arr,
            use_inert_parity,
            sweep_axis,
        )
        t_count1 = time.time()
        local_count_time = t_count1 - t_count0

        total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
        total_checked_n2 = comm.reduce(local_checked_n2, op=MPI.SUM, root=0)
        total_checked_pairs = comm.reduce(local_checked_pairs, op=MPI.SUM, root=0)
        total_passed_strip_pairs = comm.reduce(local_passed_strip_pairs, op=MPI.SUM, root=0)
        total_near_integer_pairs = comm.reduce(local_near_integer_pairs, op=MPI.SUM, root=0)
        total_feasible_pairs = comm.reduce(local_feasible_pairs, op=MPI.SUM, root=0)
        total_parity_rejects = comm.reduce(local_parity_rejects, op=MPI.SUM, root=0)

        t_fill0 = time.time()
        local_hits = np.empty((local_count, 3), dtype=np.int64)
        filled = _fill_chunk_target(
            local_hits,
            total_scale,
            target_t,
            epsY_lo,
            epsY_hi,
            n1_lo, n1_hi,
            n2_lo, n2_hi,
            rank, size,
            ALPHA1, ALPHA2, ALPHA4,
            inert_primes_arr,
            use_inert_parity,
            sweep_axis,
        )
        t_fill1 = time.time()
        local_fill_time = t_fill1 - t_fill0

        if filled != local_count:
            raise RuntimeError(f"Rank {rank}: filled={filled}, counted={local_count}")

        t_gather1_0 = time.time()
        gathered = _buffered_gather_rows(comm, local_hits, root=0, rows_per_chunk=mpi_rows_per_chunk, base_tag=1100)
        t_gather1_1 = time.time()
        local_gather1_time = t_gather1_1 - t_gather1_0

        if rank == 0:
            t_dedup0 = time.time()
            all_hits = np.concatenate(gathered, axis=0) if len(gathered) else np.empty((0, 3), dtype=np.int64)
            if all_hits.size:
                all_hits = np.unique(all_hits, axis=0)
            chunks = [np.ascontiguousarray(x, dtype=np.int64) for x in np.array_split(all_hits, size)]
            pre_exact_count = int(len(all_hits))
            t_dedup1 = time.time()
            dedup_split_time = t_dedup1 - t_dedup0
        else:
            dedup_split_time = 0.0
            chunks = None
            pre_exact_count = None

        t_scatter0 = time.time()
        local_chunk = _buffered_scatter_rows(comm, chunks, root=0, rows_per_chunk=mpi_rows_per_chunk, base_tag=2100)
        t_scatter1 = time.time()
        local_scatter_time = t_scatter1 - t_scatter0

        local_exact_screen_rejects = 0
        local_qs_profile = None
        t_exact0 = time.time()
        local_field_data = None
        if use_exact_ideal_screen:
            if quick_screen_M is None:
                raise RuntimeError("use_exact_ideal_screen requested, but zeta9.roots.quick_screen_M is unavailable")
            if build_fields is not None:
                local_field_data = build_fields()
            if profile and new_quick_screen_profile is not None and quick_screen_status_profiled is not None:
                local_qs_profile = new_quick_screen_profile()

            kept = []
            total = len(local_chunk)
            last_report_time = time.time()
            for irow in range(total):
                row = local_chunk[irow]
                if profile and local_qs_profile is not None and quick_screen_status_fast_profiled is not None:
                    status = quick_screen_status_fast_profiled(
                        int(row[0]), int(row[1]), int(row[2]),
                        check_real_embeddings=not allow_negative_embeddings,
                        check_local_p3k=check_local_p3k,
                        field_data=local_field_data,
                        profile=local_qs_profile,
                    )
                elif profile and local_qs_profile is not None:
                    status = quick_screen_status_profiled(
                        int(row[0]), int(row[1]), int(row[2]),
                        check_real_embeddings=not allow_negative_embeddings,
                        check_local_p3k=check_local_p3k,
                        field_data=local_field_data,
                        profile=local_qs_profile,
                    )
                else:
                    if quick_screen_M_fast is not None:
                        scr = quick_screen_M_fast(
                            int(row[0]), int(row[1]), int(row[2]),
                            check_real_embeddings=not allow_negative_embeddings,
                            check_local_p3k=check_local_p3k,
                            field_data=local_field_data,
                        )
                    else:
                        scr = quick_screen_M(
                            int(row[0]), int(row[1]), int(row[2]),
                            check_real_embeddings=not allow_negative_embeddings,
                            check_local_p3k=check_local_p3k,
                            field_data=local_field_data
                        )
                    status = scr["status"]
                if status == "PASSES_IDEAL_SIEVE" or status == "ZERO":
                    kept.append(row)
                else:
                    local_exact_screen_rejects += 1
                if rank == 0:
                    now = time.time()
                    if now - last_report_time >= 10.0:
                        last_report_time = now
                        pct = 100.0 * (irow + 1) / total if total > 0 else 100.0
                        print(f"phase 1, rank 0 local progress: {pct:.1f}%")
            if rank == 0: print(f"phase 1, rank 0 local progress: 100%")
            if len(kept):
                local_kept = np.array(kept, dtype=np.int64)
            else:
                local_kept = np.empty((0, 3), dtype=np.int64)
        else:
            local_kept = local_chunk
        t_exact1 = time.time()
        local_exact_time = t_exact1 - t_exact0

        t_gather2_0 = time.time()
        screened = _buffered_gather_rows(comm, local_kept, root=0, rows_per_chunk=mpi_rows_per_chunk, base_tag=3100)
        t_gather2_1 = time.time()
        local_gather2_time = t_gather2_1 - t_gather2_0
        exact_screen_rejects = comm.reduce(local_exact_screen_rejects, op=MPI.SUM, root=0)

        local_profile = {
            "count_time": float(local_count_time),
            "fill_time": float(local_fill_time),
            "gather1_time": float(local_gather1_time),
            "scatter_time": float(local_scatter_time),
            "exact_time": float(local_exact_time),
            "gather2_time": float(local_gather2_time),
            "local_count": int(local_count),
            "local_chunk_size": int(len(local_chunk)),
        }
        profile_list = comm.gather(local_profile, root=0)
        qs_profile_list = comm.gather(local_qs_profile, root=0)

        if rank != 0:
            return None

        t_finalize0 = time.time()
        all_hits = np.concatenate(screened, axis=0) if len(screened) else np.empty((0, 3), dtype=np.int64)

        np.save(output_file, all_hits)
        t_finalize1 = time.time()
        finalize_save_time = t_finalize1 - t_finalize0
        n_rows_saved = int(len(all_hits))

    else:
        # ---- out_format == "streaming" ----
        if shuffle_mode not in ("disk", "mpi"):
            raise ValueError(f"shuffle_mode must be 'disk' or 'mpi', got {shuffle_mode!r}")
        parts_dir = _streaming_parts_dir(output_file, streaming_workdir)
        if rank == 0:
            os.makedirs(parts_dir, exist_ok=True)
            # If --resume_streaming was requested AND a valid checkpoint exists,
            # preserve survivor files + the manifest (the segment loop will
            # restart from completed_through+1). Otherwise clean everything.
            preserve = False
            if shuffle_mode == "mpi" and resume_streaming:
                manifest_p = _stream_seg_manifest_path(parts_dir)
                if os.path.exists(manifest_p):
                    preserve = True
            for sub in os.listdir(parts_dir):
                full = os.path.join(parts_dir, sub)
                if sub.startswith("rank_") and os.path.isdir(full):
                    # rank_NNNN bucket-file dirs are from disk-mode only; safe to clean
                    shutil.rmtree(full)
                elif sub.startswith("survivors_rank_") and os.path.isfile(full):
                    if preserve:
                        continue  # keep the checkpoint
                    os.remove(full)
                elif sub == "seg_manifest.json" or sub == "seg_manifest.json.tmp":
                    if preserve:
                        continue
                    try: os.remove(full)
                    except OSError: pass
        comm.Barrier()

        if shuffle_mode == "mpi":
            # Phase A (shuffle): per-rank fill + MPI Alltoallv + per-bucket
            # accumulator with periodic dedup+screen+flush. No Phase B
            # needed (the screen runs inside the accumulator's flush).
            phase_AB = _stream_phase_A_shuffle(
                parts_dir, rank, size, comm,
                total_scale, target_t, epsY_lo, epsY_hi,
                n1_lo, n1_hi, n2_lo, n2_hi,
                ALPHA1, ALPHA2, ALPHA4,
                inert_primes_arr, use_inert_parity, sweep_axis,
                n_buckets, n_segments_target, verbose,
                use_exact_ideal_screen, allow_negative_embeddings,
                check_local_p3k, profile,
                screen_threshold_bytes=int(shuffle_screen_threshold_mb) * 1024 * 1024,
                f_for_manifest=int(f), u_for_manifest=float(u),
                eps_for_manifest=float(eps), norm_for_manifest=int(norm),
                checkpoint_every=int(checkpoint_every),
                resume=bool(resume_streaming),
            )
            local_count = phase_AB["local_count"]
            local_count_time = 0.0
            local_fill_time = phase_AB["local_fill_time"]
            local_partition_time = phase_AB["local_shuffle_time"]
            local_phase_A_time = phase_AB["local_phase_A_time"]
            total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
            total_checked_n2 = comm.reduce(phase_AB["local_checked_n2"], op=MPI.SUM, root=0)
            total_checked_pairs = comm.reduce(phase_AB["local_checked_pairs"], op=MPI.SUM, root=0)
            total_passed_strip_pairs = comm.reduce(phase_AB["local_passed_strip_pairs"], op=MPI.SUM, root=0)
            total_near_integer_pairs = comm.reduce(phase_AB["local_near_integer_pairs"], op=MPI.SUM, root=0)
            total_feasible_pairs = comm.reduce(phase_AB["local_feasible_pairs"], op=MPI.SUM, root=0)
            total_parity_rejects = comm.reduce(phase_AB["local_parity_rejects"], op=MPI.SUM, root=0)
            total_pre_exact = comm.reduce(phase_AB["local_pre_exact"], op=MPI.SUM, root=0)
            exact_screen_rejects = comm.reduce(phase_AB["local_screen_rejects"], op=MPI.SUM, root=0)
            total_kept_b = comm.reduce(phase_AB["local_kept_total"], op=MPI.SUM, root=0)
            local_exact_time = float(phase_AB["local_screen_time"])
            local_qs_profile = phase_AB["local_qs_profile"]
        else:
            # disk mode: original streaming pipeline (Phase A on disk + Phase B bucket-parallel)
            phase_A = _stream_phase_A(
                parts_dir, rank, size, comm,
                total_scale, target_t, epsY_lo, epsY_hi,
                n1_lo, n1_hi, n2_lo, n2_hi,
                ALPHA1, ALPHA2, ALPHA4,
                inert_primes_arr, use_inert_parity, sweep_axis,
                n_buckets, n_segments_target, verbose,
            )
            local_count = phase_A["local_count"]
            local_count_time = 0.0
            local_fill_time = phase_A["local_fill_time"]
            local_partition_time = phase_A["local_partition_time"]
            local_phase_A_time = phase_A["local_phase_A_time"]

            total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
            total_checked_n2 = comm.reduce(phase_A["local_checked_n2"], op=MPI.SUM, root=0)
            total_checked_pairs = comm.reduce(phase_A["local_checked_pairs"], op=MPI.SUM, root=0)
            total_passed_strip_pairs = comm.reduce(phase_A["local_passed_strip_pairs"], op=MPI.SUM, root=0)
            total_near_integer_pairs = comm.reduce(phase_A["local_near_integer_pairs"], op=MPI.SUM, root=0)
            total_feasible_pairs = comm.reduce(phase_A["local_feasible_pairs"], op=MPI.SUM, root=0)
            total_parity_rejects = comm.reduce(phase_A["local_parity_rejects"], op=MPI.SUM, root=0)

            phase_B = _stream_phase_B(
                parts_dir, rank, size, comm,
                n_buckets,
                use_exact_ideal_screen, allow_negative_embeddings,
                check_local_p3k, profile, verbose,
            )
            local_qs_profile = phase_B["local_qs_profile"]
            local_exact_time = phase_B["local_phase_B_time"]
            total_pre_exact = comm.reduce(phase_B["local_pre_exact"], op=MPI.SUM, root=0)
            exact_screen_rejects = comm.reduce(phase_B["local_screen_rejects"], op=MPI.SUM, root=0)
            total_kept_b = comm.reduce(phase_B["local_kept_total"], op=MPI.SUM, root=0)

        if shuffle_mode == "mpi":
            local_chunk_size = int(phase_AB["local_kept_total"])
        else:
            local_chunk_size = int(phase_B["local_kept_total"])
        local_profile = {
            "count_time": 0.0,  # streaming folds count into per-seg fill
            "fill_time": float(local_fill_time),
            "gather1_time": 0.0,
            "scatter_time": 0.0,
            "exact_time": float(local_exact_time),
            "gather2_time": 0.0,
            "local_count": int(local_count),
            "local_chunk_size": local_chunk_size,
            "partition_time": float(local_partition_time),
            "phase_A_time": float(local_phase_A_time),
            "phase_B_time": float(local_exact_time),
        }
        profile_list = comm.gather(local_profile, root=0)
        qs_profile_list = comm.gather(local_qs_profile, root=0)

        if rank != 0:
            return None

        # Phase C: root finalizes .npy from per-rank survivor files
        t_finalize0 = time.time()
        saved_path, n_rows_saved = _stream_phase_C_finalize(parts_dir, output_file, size)
        if saved_path != (output_file if output_file.endswith(".npy") else output_file + ".npy"):
            # Should never happen
            raise RuntimeError(f"Streaming finalize wrote to {saved_path} not derived from {output_file}")
        # Cross-check kept count
        if int(total_kept_b) != int(n_rows_saved):
            raise RuntimeError(
                f"Streaming kept-count mismatch: phase B sum={total_kept_b} vs phase C wrote {n_rows_saved}"
            )
        finalize_save_time = time.time() - t_finalize0

        # Streaming-mode pre_exact_count is the cross-rank sum (each bucket dedupped
        # independently, but buckets are disjoint by hash → sum is the true pre-screen
        # unique count, equivalent to the memory path's np.unique(all_hits, axis=0).size).
        pre_exact_count = int(total_pre_exact)
        dedup_split_time = 0.0  # no centralized dedup/split in streaming

        # Cleanup parts_dir unless --keep_parts
        if not keep_parts:
            try:
                shutil.rmtree(parts_dir)
            except OSError:
                pass

    out = {
        "base_scale": int(base_scale),
        "total_scale": int(total_scale_int),
        "u": float(u),
        "target_t": float(target_t),
        "eps": float(eps),
        "epsY_lo": float(epsY_lo),
        "epsY_hi": float(epsY_hi),
        "epsY_total": float(epsY_total),
        "count": int(n_rows_saved),
        "pre_exact_count": int(pre_exact_count) if pre_exact_count is not None else int(n_rows_saved),
        "saved_file": output_file if output_file.endswith(".npy") else output_file + ".npy",
        "checked_outer": int(total_checked_n2),
        "sweep_axis": int(sweep_axis),
        "checked_pairs": int(total_checked_pairs),
        "passed_strip_pairs": int(total_passed_strip_pairs),
        "near_integer_pairs": int(total_near_integer_pairs),
        "feasible_pairs": int(total_feasible_pairs),
        "parity_rejects": int(total_parity_rejects),
        "inert_primes": inert_primes,
        "exact_screen_rejects": int(exact_screen_rejects),
        "norm": int(norm),
        "count_time_max": max((p["count_time"] for p in profile_list), default=0.0),
        "fill_time_max": max((p["fill_time"] for p in profile_list), default=0.0),
        "exact_time_max": max((p["exact_time"] for p in profile_list), default=0.0),
        "gather1_time_max": max((p["gather1_time"] for p in profile_list), default=0.0),
        "gather2_time_max": max((p["gather2_time"] for p in profile_list), default=0.0),
        "scatter_time_max": max((p["scatter_time"] for p in profile_list), default=0.0),
        "dedup_split_time": float(dedup_split_time),
        "finalize_save_time": float(finalize_save_time),
        "wall_total_time": float(time.time() - wall_t0),
    }

    merged_qs_profile = None
    if profile and qs_profile_list is not None and any(p is not None for p in qs_profile_list):
        merged_qs_profile = new_quick_screen_profile() if new_quick_screen_profile is not None else None
        if merged_qs_profile is not None:
            for p in qs_profile_list:
                if p is None:
                    continue
                merged_qs_profile["calls"] += p.get("calls", 0)
                for key in ("t_build_M", "t_real_embeddings", "t_factorization_F", "t_classify_extension", "t_local_p3", "t_finalize", "t_total"):
                    merged_qs_profile[key] += p.get(key, 0.0)
                for k, v in p.get("status_counts", {}).items():
                    merged_qs_profile["status_counts"][k] = merged_qs_profile["status_counts"].get(k, 0) + v
                ccm = merged_qs_profile["classify_cache_profile"]
                ccp = p.get("classify_cache_profile", {})

                ccm["calls"] += ccp.get("calls", 0)
                ccm["hits"] += ccp.get("hits", 0)
                ccm["misses"] += ccp.get("misses", 0)
                ccm["t_key"] += ccp.get("t_key", 0.0)
                ccm["t_hit_lookup"] += ccp.get("t_hit_lookup", 0.0)
                ccm["t_miss_work"] += ccp.get("t_miss_work", 0.0)



    if rank == 0 and verbose:
        print()
        print(f"checked outer values    : {out['checked_outer']}")
        print(f"checked (n1,n2) pairs   : {out['checked_pairs']}")
        print(f"passed strip pairs      : {out['passed_strip_pairs']}")
        print(f"near-integer pairs      : {out['near_integer_pairs']}")
        print(f"feasible pairs          : {out['feasible_pairs']}")
        print(f"parity rejects          : {out['parity_rejects']}")
        if use_exact_ideal_screen:
            print(f"pre-exact unique triples: {out['pre_exact_count']}")
            print(f"exact ideal rejects     : {out['exact_screen_rejects']}")
        print(f"saved triples           : {out['count']}")
        print(f"saved file              : {out['saved_file']}")
        print(f"[BUCKET DONE]  f={f} u={u} elapsed={out['wall_total_time']:.1f}s kept={out['count']} screen_rej={out['exact_screen_rejects']} -> {os.path.basename(out['saved_file'])}", flush=True)
        if profile:
            def _stats(key):
                vals = [p[key] for p in profile_list]
                return min(vals), max(vals), sum(vals)
            cmin, cmax, csum = _stats("count_time")
            fmin, fmax, fsum = _stats("fill_time")
            emin, emax, esum = _stats("exact_time")
            g1min, g1max, g1sum = _stats("gather1_time")
            smin, smax, ssum = _stats("scatter_time")
            g2min, g2max, g2sum = _stats("gather2_time")
            print()
            print("profiling summary:")
            print(f"  count kernel   min/max/sum : {cmin:.3f} / {cmax:.3f} / {csum:.3f} s")
            print(f"  fill kernel    min/max/sum : {fmin:.3f} / {fmax:.3f} / {fsum:.3f} s")
            print(f"  exact screen   min/max/sum : {emin:.3f} / {emax:.3f} / {esum:.3f} s")
            print(f"  gather1        min/max/sum : {g1min:.3f} / {g1max:.3f} / {g1sum:.3f} s")
            print(f"  scatter        min/max/sum : {smin:.3f} / {smax:.3f} / {ssum:.3f} s")
            print(f"  gather2        min/max/sum : {g2min:.3f} / {g2max:.3f} / {g2sum:.3f} s")
            print(f"  dedup/split on root        : {dedup_split_time:.3f} s")
            print(f"  finalize/save on root      : {finalize_save_time:.3f} s")
            print(f"  total wall time            : {time.time() - wall_t0:.3f} s")
            print("  per-rank sizes:")
            for ir, p in enumerate(profile_list):
                print(f"    rank {ir}: local_count={p['local_count']}, chunk={p['local_chunk_size']}, "
                      f"count={p['count_time']:.3f}s, fill={p['fill_time']:.3f}s, exact={p['exact_time']:.3f}s")
            if merged_qs_profile is not None and _profile_summary is not None:
                print()
                print(_profile_summary(merged_qs_profile))

    return out


# ============================================================
#  Checkpoint / binning helpers
# ============================================================

def _bin_file_name(output_file, idx):
    return f"{output_file}.bin_{idx:04d}.npy"

def _manifest_file_name(output_file):
    return f"{output_file}.manifest.json"

def _default_bin_edges(epsY_total, epsy_bin_width=None, epsy_bins=None):
    if epsY_total <= 0.0:
        return [0.0, 0.0]
    if epsy_bin_width is not None and epsy_bin_width > 0.0:
        edges = [0.0]
        x = 0.0
        while x < epsY_total:
            x = min(epsY_total, x + epsy_bin_width)
            edges.append(float(x))
        if edges[-1] < epsY_total:
            edges.append(float(epsY_total))
        return edges
    nbins = int(epsy_bins) if epsy_bins is not None else 1
    nbins = max(1, nbins)
    return [float(x) for x in np.linspace(0.0, epsY_total, nbins + 1)]

def _load_manifest(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def collect_targets_mpi(
    f,
    u,
    eps,
    output_file,
    norm=2,
    inert_prime_bound=29,
    use_inert_parity=False,
    use_exact_ideal_screen=False,
    allow_negative_embeddings=False,
    check_local_p3k=False,
    verbose=True,
    profile=False,
    mpi_rows_per_chunk=10000,
    eps_bin_width=None,
    epsy_bins=None,
    resume=False,
    out_format="memory",
    n_buckets=4096,
    n_segments_target=256,
    streaming_workdir=None,
    keep_parts=False,
    shuffle_mode="disk",
    shuffle_screen_threshold_mb=512,
    checkpoint_every=10,
    resume_streaming=False,
):
    base_scale = float(3 ** (2 * f))
    epsY_total = base_scale * eps * (2 * float(u)**0.5 + eps)
    epsy_bin_width = base_scale * eps_bin_width * (2 * float(u)**0.5 + eps)

    use_binning = ((epsy_bin_width is not None and epsy_bin_width > 0.0) or (epsy_bins is not None and int(epsy_bins) > 1))
    if not use_binning:
        return _collect_targets_single_range_mpi(
            f=f, u=u, eps=eps, output_file=output_file, norm=norm,
            inert_prime_bound=inert_prime_bound, use_inert_parity=use_inert_parity,
            use_exact_ideal_screen=use_exact_ideal_screen,
            allow_negative_embeddings=allow_negative_embeddings,
            check_local_p3k=check_local_p3k, verbose=verbose, profile=profile,
            mpi_rows_per_chunk=mpi_rows_per_chunk, epsY_lo=0.0, epsY_hi=epsY_total,
            out_format=out_format, n_buckets=n_buckets,
            n_segments_target=n_segments_target,
            streaming_workdir=streaming_workdir, keep_parts=keep_parts,
            shuffle_mode=shuffle_mode,
            shuffle_screen_threshold_mb=shuffle_screen_threshold_mb,
            checkpoint_every=checkpoint_every,
            resume_streaming=resume_streaming,
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    manifest_path = _manifest_file_name(output_file)
    edges = _default_bin_edges(epsY_total, epsy_bin_width=epsy_bin_width, epsy_bins=epsy_bins)

    if rank == 0:
        manifest = _load_manifest(manifest_path) if resume else None
        if manifest is None:
            manifest = {
                "output_file": output_file,
                "f": int(f),
                "u": float(u),
                "eps": float(eps),
                "norm": int(norm),
                "epsY_total": float(epsY_total),
                "epsy_bin_width": None if epsy_bin_width is None else float(epsy_bin_width),
                "epsy_bins": None if epsy_bins is None else int(epsy_bins),
                "bins": [],
            }
        completed = {int(b["index"]): b for b in manifest.get("bins", []) if b.get("completed")}
    else:
        manifest = None
        completed = None

    manifest = comm.bcast(manifest, root=0)
    completed = comm.bcast(completed, root=0)

    results = []
    n_bins = len(edges) - 1
    for idx in range(n_bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        bin_file = _bin_file_name(output_file, idx)

        if idx in completed and os.path.exists(bin_file):
            if rank == 0 and verbose:
                print(f"Skipping completed bin {idx}: ({lo}, {hi}] -> {bin_file}")
                print(f"[BIN {idx+1}/{n_bins} SKIP] (already complete in manifest)", flush=True)
            results.append(completed[idx])
            continue

        bin_t0 = time.time()
        if rank == 0 and verbose:
            print()
            print(f"=== epsY bin {idx}: ({lo}, {hi}] ===")
            print(f"[BIN {idx+1}/{n_bins} START] epsY=({lo:.4g},{hi:.4g}]", flush=True)

        bin_workdir = None
        if streaming_workdir is not None:
            bin_workdir = streaming_workdir.rstrip("/") + f".bin_{idx:04d}"
        out = _collect_targets_single_range_mpi(
            f=f, u=u, eps=eps, output_file=bin_file, norm=norm,
            inert_prime_bound=inert_prime_bound, use_inert_parity=use_inert_parity,
            use_exact_ideal_screen=use_exact_ideal_screen,
            allow_negative_embeddings=allow_negative_embeddings,
            check_local_p3k=check_local_p3k, verbose=verbose, profile=profile,
            mpi_rows_per_chunk=mpi_rows_per_chunk, epsY_lo=lo, epsY_hi=hi,
            out_format=out_format, n_buckets=n_buckets,
            n_segments_target=n_segments_target,
            streaming_workdir=bin_workdir, keep_parts=keep_parts,
            shuffle_mode=shuffle_mode,
            shuffle_screen_threshold_mb=shuffle_screen_threshold_mb,
            checkpoint_every=checkpoint_every,
            resume_streaming=resume_streaming,
        )

        if rank == 0:
            entry = {
                "index": int(idx),
                "epsY_lo": lo,
                "epsY_hi": hi,
                "file": bin_file if bin_file.endswith(".npy") else bin_file + ".npy",
                "count": int(out["count"]),
                "completed": True,
            }
            manifest.setdefault("bins", [])
            manifest["bins"] = [b for b in manifest["bins"] if int(b.get("index", -1)) != idx]
            manifest["bins"].append(entry)
            manifest["bins"].sort(key=lambda x: int(x["index"]))
            _save_manifest(manifest_path, manifest)
            results.append(entry)
            if verbose:
                print(f"[BIN {idx+1}/{n_bins} DONE] elapsed={time.time()-bin_t0:.1f}s kept={int(out['count'])} (cumulative {idx+1}/{n_bins})", flush=True)

    if rank != 0:
        return None

    summary = {
        "mode": "binned",
        "output_file": output_file,
        "manifest_file": manifest_path,
        "epsY_total": float(epsY_total),
        "bin_count": len(edges) - 1,
        "completed_bins": len(manifest.get("bins", [])),
        "total_saved_triples": int(sum(int(b.get("count", 0)) for b in manifest.get("bins", []))),
        "bins": manifest.get("bins", []),
    }
    if verbose:
        print()
        print(f"manifest file           : {manifest_path}")
        print(f"completed bins          : {summary['completed_bins']}/{summary['bin_count']}")
        print(f"total saved triples     : {summary['total_saved_triples']}")
    return summary


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--u", type=float, required=True, help="u = |d_i|^2")
    parser.add_argument("--eps", type=float, required=True, help="final vector epsilon")
    parser.add_argument("--output", type=str, required=True, help="output .npy file")
    parser.add_argument("--norm", type=int, default=2,
                        help="total target norm squared as an integer (default: 2)")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--inert_prime_bound", type=int, default=29)
    parser.add_argument("--check_local_p3k", action="store_true")
    parser.add_argument("--mpi_rows_per_chunk", type=int, default=10000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--eps_bin_width", type=float, default=None)
    parser.add_argument("--epsy_bins", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_format", choices=["memory", "streaming"], default="memory",
                        help="memory: original in-memory pipeline (default). streaming: per-rank disk hash-bucket partition + bucket-parallel screen, for f>=5.")
    parser.add_argument("--n_buckets", type=int, default=4096,
                        help="streaming mode: number of hash buckets for stage-1 partition")
    parser.add_argument("--n_segments_target", type=int, default=256,
                        help="streaming mode: target number of polygon segments per rank")
    parser.add_argument("--streaming_workdir", type=str, default=None,
                        help="streaming mode: override parts_dir location (default: output_file + '.s1_parts')")
    parser.add_argument("--keep_parts", action="store_true",
                        help="streaming mode: do not delete parts_dir after finalize (for debugging)")
    parser.add_argument("--shuffle_mode", choices=["disk", "mpi"], default="disk",
                        help="streaming mode: 'disk' writes per-rank bucket files (current default), 'mpi' uses MPI Alltoallv to shuffle hits to owner ranks and dedup+screen inline (avoids most disk writes; required for f>=6 on mechanical-disk parts_dir)")
    parser.add_argument("--shuffle_screen_threshold_mb", type=int, default=512,
                        help="streaming mpi mode: per-rank accumulator size before triggering dedup+screen+flush (default 512 MB)")
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="streaming mpi mode: write seg_manifest.json every N segments (default 10). "
                             "Smaller = more crash-resilient but more I/O. 0 disables checkpointing.")
    parser.add_argument("--resume_streaming", action="store_true",
                        help="streaming mpi mode: if seg_manifest.json exists in parts_dir, resume from "
                             "completed_through+1 instead of restarting.")
    parser.add_argument("--no_stage1_screen", action="store_true",
                        help="Skip the exact ideal screen in stage 1 (saves time at f>=6 where screen is the bottleneck). "
                             "Must run zeta9.prefilter_y1 on the output before passing to stage 2.")
    parser.add_argument("--epsy_lo", type=float, default=None,
                        help="manual lower epsY bin boundary (overrides --epsy_bins splitting). "
                             "Use with --epsy_hi for single-bin processing.")
    parser.add_argument("--epsy_hi", type=float, default=None,
                        help="manual upper epsY bin boundary (overrides --epsy_bins splitting).")

    args = parser.parse_args()

    # If --epsy_lo/--epsy_hi explicit, take the single-range fast path
    if args.epsy_lo is not None or args.epsy_hi is not None:
        if args.epsy_lo is None or args.epsy_hi is None:
            raise SystemExit("--epsy_lo and --epsy_hi must be provided together")
        # call the single-range MPI entry point directly to skip binning logic
        import sys as _sys
        rank0 = MPI.COMM_WORLD.Get_rank() == 0
        if rank0 and not args.quiet:
            print(f"[manual bin] epsY=({args.epsy_lo}, {args.epsy_hi}]")
        out = _collect_targets_single_range_mpi(
            f=args.f, u=args.u, eps=args.eps, output_file=args.output,
            norm=args.norm,
            inert_prime_bound=args.inert_prime_bound,
            use_inert_parity=True,
            use_exact_ideal_screen=not args.no_stage1_screen,
            allow_negative_embeddings=False,
            check_local_p3k=args.check_local_p3k,
            verbose=not args.quiet,
            profile=args.profile,
            mpi_rows_per_chunk=args.mpi_rows_per_chunk,
            epsY_lo=float(args.epsy_lo),
            epsY_hi=float(args.epsy_hi),
            out_format=args.out_format,
            n_buckets=args.n_buckets,
            n_segments_target=args.n_segments_target,
            streaming_workdir=args.streaming_workdir,
            keep_parts=args.keep_parts,
            shuffle_mode=args.shuffle_mode,
            shuffle_screen_threshold_mb=args.shuffle_screen_threshold_mb,
            checkpoint_every=args.checkpoint_every,
            resume_streaming=args.resume_streaming,
        )
        if rank0 and args.quiet:
            print(out)
        _sys.exit(0)

    result = collect_targets_mpi(
        f=args.f,
        u=args.u,
        eps=args.eps,
        output_file=args.output,
        norm=args.norm,
        inert_prime_bound=args.inert_prime_bound,
        use_inert_parity=True,
        use_exact_ideal_screen=not args.no_stage1_screen,
        allow_negative_embeddings=False,
        check_local_p3k=args.check_local_p3k,
        verbose=not args.quiet,
        mpi_rows_per_chunk=args.mpi_rows_per_chunk,
        profile=args.profile,
        eps_bin_width=args.eps_bin_width,
        epsy_bins=args.epsy_bins,
        resume=args.resume,
        out_format=args.out_format,
        n_buckets=args.n_buckets,
        n_segments_target=args.n_segments_target,
        streaming_workdir=args.streaming_workdir,
        keep_parts=args.keep_parts,
        shuffle_mode=args.shuffle_mode,
        shuffle_screen_threshold_mb=args.shuffle_screen_threshold_mb,
        checkpoint_every=args.checkpoint_every,
        resume_streaming=args.resume_streaming,
    )

    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(result)
