"""Streamed Householder-vector search for a 3x3 diagonal R^Z target over Z[zeta_9, 1/3].

Companion to ``search_diagonal_matrix_two_rows_streamed_mpi.py``.

Mode
----
Consumes "householder triples" produced by ``select_triples_optimized --norm 2``
with input pattern ``(Y1_u=1, Y1_u=1, Y1_u=0)``: each triple
``(Y_0, Y_1, Y_2)`` satisfies ``sum Y_i = (2*3^{2f}, 0, 0)`` and represents the
squared moduli of a candidate Householder vector ``u = (a_0, a_1, a_2)/3^f``
with ``|u|^2 = 2`` and (in this layout) two entries of ``|u_i| ~ 1`` and one
entry of ``|u_i| ~ 0``.

Math
----
Following the zeta9-paper convention used by ``try.py`` and
``zeta9_compile.py:householder_v_from_u``:

    V = X_{(0,1)} . ( I - conj(u) (x) u )       where (conj(u)(x)u)_{ij} = conj(u_i)*u_j

For ``u = (a, b, 0)`` with ``|a| = |b| = 1`` we get

    V_00 = -conj(b)*a,   V_01 = 0,   V_02 = 0
    V_10 = 0,            V_11 = -conj(a)*b, V_12 = 0
    V_22 = 1

Matching ``diag(d_1, d_2, 1)`` requires ``a*conj(b) = -d_1 = -exp(-i*theta/2)``.
The phase constraint is on the *product* of the two large entries; the
individual phases of ``a`` and ``b`` are free.

Algorithm
---------
For each Householder triple ``(Y_0, Y_1, Y_2)``:

  - Identify the small slot (where ``sigma1(Y) ~ 0``) and the two large slots.
  - Enumerate ringZ9 roots of the small slot with ``|a| <= eps`` (sidecar
    descriptor with target ``0+0j``).
  - For each large slot, enumerate **all** ringZ9 roots that satisfy the
    annular radial constraint ``||a|-1| <= eps``.  We bypass the standard
    sidecar phase-shell logic by walking the full root range for the Y and
    filtering directly on ``||z|-1|``.
  - For each (a_b, a_c) pair from the two large slots, check
    ``|a_b * conj(a_c)/3^{2f} + d_1| <= eps``.
  - For each surviving (a_b, a_c, a_a), build the V matrix exactly in ringZ9
    at layer 2f and measure ``||V - diag(d_1, d_2, 1)||_F``.
  - Keep the top-``keep`` results by Frobenius distance.

Output schema is identical to the diagonal stage 5: ``xout.npz`` with keys
``rows_coeffs_layer_f``, ``dist_fro``, ``diag``, ``f`` (= 2*f_user), ``eps``,
``mode``.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Optional

# Ensure the parent of the ``zeta9`` package directory is on sys.path so that
# Numba's cached compiled kernels (which were generated with the package-qualified
# module name ``zeta9.search_diagonal_matrix_two_rows_streamed_mpi``) can be
# loaded when this script is invoked as a path, not as ``-m zeta9.X``.
_THIS = os.path.dirname(os.path.abspath(__file__))           # .../zeta9/zeta9
_PKG_PARENT = os.path.dirname(_THIS)                          # .../zeta9
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import numpy as np
from mpi4py import MPI

# Numba acceleration for the inner pair-filter loop (2026-05-13).
# The Householder stage 5 vectorized broadcast computes a full n_b * n_c
# complex grid plus mask, plus np.nonzero, with one big temporary per triple.
# At f >= 4 the per-triple pair counts grow into the 10k-100k range and the
# Python-side allocation overhead dominates.  A fused @njit kernel processes
# pairs in a tight loop and emits only the passing indices.
try:
    import numba as _nb
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _hh_inner_filter_pairs_nb(
        zb_re, zb_im,        # (n_b,) float64
        zc_re, zc_im,        # (n_c,) float64
        target_re, target_im,  # float64
        eps_plus_pad,        # float64 (the tolerance, NOT its square)
        idx_b_out,           # (cap,) int64  (pre-allocated)
        idx_c_out,           # (cap,) int64  (pre-allocated)
    ):
        """Fused product-phase filter for the Householder stage 5.

        For each (ib, ic) in n_b x n_c:
            prod = z_b[ib] * conj(z_c[ic])
            keep if |prod - target| <= eps_plus_pad

        z_b and z_c are passed as separate real/imag arrays to avoid the
        complex128 ABI inside Numba (works either way; this is a minor
        readability choice).  The check is on the *squared* tolerance
        to avoid a sqrt per pair.

        Returns the number of surviving pairs written into idx_b_out/idx_c_out.
        Caller is responsible for sizing the output arrays (cap >= n_b * n_c).
        """
        n_b = zb_re.shape[0]
        n_c = zc_re.shape[0]
        thr2 = eps_plus_pad * eps_plus_pad
        n_match = np.int64(0)
        cap = idx_b_out.shape[0]
        for ib in range(n_b):
            br = zb_re[ib]
            bi = zb_im[ib]
            for ic in range(n_c):
                # prod = (br + i*bi) * (cr - i*ci)
                cr = zc_re[ic]
                ci = zc_im[ic]
                pr = br * cr + bi * ci
                pi = bi * cr - br * ci
                dr = pr - target_re
                di = pi - target_im
                d2 = dr * dr + di * di
                if d2 <= thr2:
                    if n_match < cap:
                        idx_b_out[n_match] = ib
                        idx_c_out[n_match] = ic
                    n_match += 1
        return n_match


# Reuse infra from the diagonal stage 5 (z9 arithmetic, sidecar helpers).
try:
    from .search_diagonal_matrix_two_rows_streamed_mpi import (  # type: ignore
        _state_paths,
        _read_manifest,
        _read_triple_rows,
        _build_locator,
        _load_chunk_metadata,
        _candidate_desc_from_binned_sidecar,
        coeff_row_to_complex_scaled,
        row_complex,
        z9_mul,
        z9_conj,
        z9_sub,
        sigma1_from_m012,
    )
    from .fit_vectors_mpi_sidecar_binned_v2 import _get_sidecar_batch_cache  # type: ignore
except ImportError:
    from search_diagonal_matrix_two_rows_streamed_mpi import (  # type: ignore
        _state_paths,
        _read_manifest,
        _read_triple_rows,
        _build_locator,
        _load_chunk_metadata,
        _candidate_desc_from_binned_sidecar,
        coeff_row_to_complex_scaled,
        row_complex,
        z9_mul,
        z9_conj,
        z9_sub,
        sigma1_from_m012,
    )
    from fit_vectors_mpi_sidecar_binned_v2 import _get_sidecar_batch_cache  # type: ignore


# ---------------------------------------------------------------------------
# Per-search counters
# ---------------------------------------------------------------------------


@dataclass
class HHCounters:
    chunks_seen: int = 0
    chunks_processed: int = 0
    triples_processed: int = 0
    triples_with_empty_desc: int = 0
    a_large_roots_seen: int = 0
    a_small_roots_seen: int = 0
    a01_pairs_tested: int = 0
    a01_pairs_pass_phase: int = 0
    matrices_found: int = 0

    def add(self, other: "HHCounters") -> "HHCounters":
        for k in self.__dataclass_fields__:
            setattr(self, k, int(getattr(self, k)) + int(getattr(other, k)))
        return self


@dataclass(order=True)
class HeapItem:
    priority: float  # heap is min-heap; priority = -dist so worst is at root.
    result: dict = field(compare=False)


def _v_int_key(result):
    """Compact comparable key for V_int — used for torsion/dedup detection.
    Two V's with identical rows_coeffs_layer_f represent the same unitary
    matrix exactly (Z[ζ_9] basis is Z-linearly independent). Even when the
    integer reps differ but the unitary is the same (rare, requires basis
    relations), this key conservatively keeps them separate, which is fine."""
    return tuple(int(x) for row in result["rows_coeffs_layer_f"]
                            for col in row for x in col)


# ---- ζ_9-torsion orbit canonicalization ----
# Z[ζ_9] basis is {1, ζ_9, ..., ζ_9^5}. Multiplication by ζ_9 sends coefficient
# vector (c0, ..., c5) → M_ζ · (c0, ..., c5) where M_ζ is the companion matrix
# of Φ_9(x) = x^6 + x^3 + 1, i.e., ζ_9^6 = ζ_9^3 - 1.
_M_ZETA = np.array([
    [0, 0, 0, 0, 0, -1],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
], dtype=np.int64)

# Build powers ζ_9^k for k=0..8, plus their negatives → 18 unit operators
_ZETA9_UNITS = [None] * 18
_pow = np.eye(6, dtype=np.int64)
for _k in range(9):
    _ZETA9_UNITS[2 * _k] = _pow.copy()
    _ZETA9_UNITS[2 * _k + 1] = -_pow.copy()
    _pow = _M_ZETA @ _pow  # ζ_9^{k+1}


def _canonical_orbit_indices(roots: np.ndarray) -> np.ndarray:
    """Return indices of roots, keeping one representative per ζ_9-torsion orbit.
    Two roots r1, r2 are torsion-equivalent if r2 = ±ζ_9^k · r1 for some k.

    For HOUSEHOLDER mode: under diagonal torsion η on (a_b, a_c, a_a),
    V is invariant. Canonicalizing a_b alone (one rep per orbit) factors out
    the 18-fold diagonal-torsion redundancy in the (a_b, a_c, a_a) tuple set.

    Input: roots, shape (n, 6), int64.
    Output: indices, shape (n',), where n' ≈ n/18 generically.
    """
    n = roots.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    seen = set()
    keep = []
    for i in range(n):
        r = roots[i]
        # Compute orbit: 18 unit-multiples (under ±ζ_9^k action)
        orbit_key = frozenset(
            tuple(int(x) for x in (U @ r)) for U in _ZETA9_UNITS
        )
        if orbit_key not in seen:
            seen.add(orbit_key)
            keep.append(i)  # First-seen orbit member is the rep
    return np.ascontiguousarray(keep, dtype=np.int64)


def maybe_push_result(heap, result, keep):
    item = HeapItem(priority=-float(result["dist_fro"]), result=result)
    # Dedup: skip if a heap entry has the same V_int (same unitary).
    new_key = _v_int_key(result)
    for existing in heap:
        if _v_int_key(existing.result) == new_key:
            return  # already in heap, skip
    if len(heap) < keep:
        heapq.heappush(heap, item)
    elif item.priority > heap[0].priority:
        heapq.heapreplace(heap, item)


# ---------------------------------------------------------------------------
# Radial-only root enumeration (annulus filter).
#
# We bypass the standard sidecar descriptor (which is built around a complex
# disk target |z - target| <= eps) and instead enumerate *all* roots for the
# given Y, then filter by ||z| - target_abs| <= eps. This is the correct
# pruning for the (a_b, a_c) lookup where the per-coordinate constraint is
# radial only.
# ---------------------------------------------------------------------------


def enumerate_roots_annulus(
    y_tuple, locator, phase_meta, batch_cache, target_abs, eps, f
):
    """Return (roots, zs) for all roots of `y_tuple` with ||z| - target_abs| <= eps.

    `roots` is (n, 6) int64 (numerator coefficients at layer f).
    `zs`    is (n,)   complex128 (z-value at layer f, i.e. /3^f).
    """
    loc = locator.get(y_tuple)
    if loc is None:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)

    sigma1_M = sigma1_from_m012(*y_tuple)
    if sigma1_M < 0:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
    r = math.sqrt(max(0.0, sigma1_M)) / (3 ** f)

    # Quick reject: this Y can never contain any root with ||z| - target_abs| <= eps
    # because sigma1 fixes |z| exactly for all roots of Y.
    if abs(r - target_abs) > eps + 1.0e-15:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)

    batch_idx, row_idx = loc
    bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
    roots_off = bd["roots_off"]
    a = int(roots_off[row_idx])
    b = int(roots_off[row_idx + 1])
    if b <= a:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)

    scale = float(3 ** f)
    z_re = np.asarray(bd["z_re_flat"][a:b], dtype=np.float64)
    z_im = np.asarray(bd["z_im_flat"][a:b], dtype=np.float64)
    zs = (z_re + 1j * z_im) / scale
    # All roots of a given Y share |z| = r (up to numerical noise) so the
    # annulus filter is redundant after the sigma1 prefilter above.  We
    # apply it anyway as a tightness check.
    keep = np.abs(np.abs(zs) - target_abs) <= eps + 1.0e-15
    if not np.any(keep):
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
    idx = np.flatnonzero(keep).astype(np.int64)
    roots_flat = bd["roots_flat"]
    roots = np.ascontiguousarray(roots_flat[a + idx], dtype=np.int64)
    return roots, np.ascontiguousarray(zs[keep], dtype=np.complex128)


def enumerate_roots_small(
    y_tuple, locator, phase_meta, batch_cache, eps, f
):
    """Return (roots, zs) for all roots of `y_tuple` with |z| <= eps.

    Special-cased target = 0+0j.  Includes the synthetic zero root when
    Y = (0, 0, 0) and the sidecar entry is empty.
    """
    if y_tuple == (0, 0, 0):
        # Synthetic zero root.  The rootdb may or may not have an entry for
        # the literal zero element; handle both cases.
        loc = locator.get(y_tuple)
        if loc is None:
            return (
                np.zeros((1, 6), dtype=np.int64),
                np.array([0.0 + 0.0j], dtype=np.complex128),
            )

    return enumerate_roots_annulus(y_tuple, locator, phase_meta, batch_cache, 0.0, eps, f)


# ---------------------------------------------------------------------------
# Householder V construction (exact, ringZ9 at layer 2f).
# ---------------------------------------------------------------------------


def build_householder_V_exact(a0, a1, a2, f):
    """Return V in (3, 3, 6) int64 form, with denominator 3^{2f}.

    V = X_{(0,1)} . ( I - conj(u) (x) u )     where u_i = a_i / 3^f.

    Outer-product convention matches ``try.py:MatrixDinC`` and
    ``zeta9_compile.py:householder_v_from_u``:
        outer[i, j] = conj(u_i) * u_j         (in C)
                    = z9_mul(z9_conj(a_i), a_j) / 3^{2f}     (in ringZ9 at layer 2f)
    Therefore at layer 2f:
        H[i, j] = (3^{2f} if i==j else 0) - z9_mul(z9_conj(a_i), a_j)
        V[i, :] = H[ X_{(0,1)} i, :]    swap rows 0 and 1.
    """
    a = np.stack(
        [np.ascontiguousarray(a0, dtype=np.int64),
         np.ascontiguousarray(a1, dtype=np.int64),
         np.ascontiguousarray(a2, dtype=np.int64)],
        axis=0,
    )
    conj_a = np.stack([z9_conj(a[i]) for i in range(3)], axis=0)
    outer = np.zeros((3, 3, 6), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            outer[i, j] = z9_mul(conj_a[i], a[j])

    one_layer_2f = np.zeros(6, dtype=np.int64)
    one_layer_2f[0] = int(3) ** int(2 * f)
    H = np.zeros((3, 3, 6), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            if i == j:
                H[i, j] = one_layer_2f - outer[i, j]
            else:
                H[i, j] = -outer[i, j]

    V = np.zeros((3, 3, 6), dtype=np.int64)
    V[0] = H[1]
    V[1] = H[0]
    V[2] = H[2]
    return V


def V_to_complex(V_int, denom_pow):
    """Convert (3, 3, 6) int64 V at layer denom_pow into a 3x3 complex matrix."""
    M = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        M[i, :] = row_complex(V_int[i], denom_pow)
    return M


# ---------------------------------------------------------------------------
# Main streamed search
# ---------------------------------------------------------------------------


def search_householder_streamed(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    d1: complex,
    d2: complex,
    d3: complex,
    eps: float,
    output_prefix: str,
    triples_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: Optional[str] = None,
    keep: int = 16,
    verbose: bool = True,
    max_matches: int = 0,
) -> Optional[dict]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = time.monotonic()

    counters = HHCounters()

    # Target for (a_b * conj(a_c)) / 3^{2f} -- matching V_00 = -conj(b)*a = d_1
    # gives a * conj(b) = -d_1 (where the "b" here is the Householder-vector
    # second large entry, i.e. row index 1 in the slot-mapped (a, b, 0) u).
    target_product = -complex(d1)
    diag = np.array([d1, d2, d3], dtype=np.complex128)
    f_v = int(2) * int(f)  # V is at layer 2f.

    if rank == 0:
        tmeta = _read_manifest(triples_json)
        nrows = int(tmeta["rows_written"])
        file_size = os.path.getsize(triples_file)
        expected = nrows * 9 * np.dtype(np.int64).itemsize
        if file_size != expected:
            raise RuntimeError(f"Triple file size mismatch: expected {expected}, got {file_size}")
        paths = _state_paths(rootdb_prefix, n_phase_bins)
        if not os.path.exists(paths["exact_roots_index_meta"]):
            raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")
        if not os.path.exists(paths["phase_sidecar_meta"]):
            raise FileNotFoundError(
                f"Binned phase sidecar not found. Run the sidecar builder first for {rootdb_prefix}"
            )
        phase_meta = _read_manifest(paths["phase_sidecar_meta"])
        chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
        print(
            f"[hh-stage5] streamed Householder search | rows={nrows:,} ranks={size} "
            f"f={f} f_v={f_v} eps={eps:g} "
            f"target_product=({target_product.real:.6f}{target_product.imag:+.6f}j)",
            flush=True,
        )
    else:
        nrows = None
        paths = None
        phase_meta = None
        chunk_meta = None

    nrows = comm.bcast(nrows, root=0)
    paths = comm.bcast(paths, root=0)
    phase_meta = comm.bcast(phase_meta, root=0)
    chunk_meta = comm.bcast(chunk_meta, root=0)

    batch_cache: OrderedDict = OrderedDict()
    heap: list[HeapItem] = []
    eps_plus_pad = eps + 1.0e-15

    n_chunks = (int(nrows) + int(triples_chunk_rows) - 1) // int(triples_chunk_rows)

    last_hb = [time.monotonic()]

    def heartbeat(label: str, force: bool = False):
        now = time.monotonic()
        if not force and now - last_hb[0] < 10.0:
            return
        last_hb[0] = now
        if rank != 0:
            return
        elapsed = now - start_time
        print(
            f"[hh-stage5/{label}] elapsed={elapsed:,.1f}s | "
            f"triples {counters.triples_processed:,}/{nrows:,} | "
            f"a01 pairs {counters.a01_pairs_tested:,} | "
            f"phase pass {counters.a01_pairs_pass_phase:,} | "
            f"matrices {counters.matrices_found:,}",
            flush=True,
        )

    heartbeat("startup", force=True)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * int(triples_chunk_rows)
        take = min(int(triples_chunk_rows), int(nrows) - start)
        counters.chunks_seen += 1
        if chunk_idx % size != rank:
            continue
        counters.chunks_processed += 1

        triples = _read_triple_rows(triples_file, start, take)
        if verbose and rank == 0:
            print(f"[hh-stage5] chunk {chunk_idx+1}/{n_chunks} rows {start:,}..{start + take - 1:,}", flush=True)

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_idx]
            _needed, locator = _load_chunk_metadata(rec["path"])
        else:
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            locator = _build_locator(needed, paths)

        # Per-chunk caches: (Y, target_abs) -> (roots, zs, z_re, z_im).
        # Pre-extracted z.real / z.imag avoid the per-triple ascontiguousarray
        # call inside the pair-filter setup (Round 3, 2026-05-13).
        # LRU-capped (2026-05-22, memory refactor #4): bounds working set so
        # high-Y-cardinality chunks don't accumulate unbounded RSS at f=6.
        # Evicted entries are simply re-hydrated on next access from mmap.
        large_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        small_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        _LRU_CAP = 4096

        def get_large(y_tuple):
            cached = large_cache.get(y_tuple)
            if cached is not None:
                large_cache.move_to_end(y_tuple)
                return cached
            roots, zs = enumerate_roots_annulus(
                y_tuple, locator, phase_meta, batch_cache, 1.0, eps, f
            )
            z_re = np.ascontiguousarray(zs.real, dtype=np.float64)
            z_im = np.ascontiguousarray(zs.imag, dtype=np.float64)
            entry = (roots, zs, z_re, z_im)
            large_cache[y_tuple] = entry
            if len(large_cache) > _LRU_CAP:
                large_cache.popitem(last=False)
            return entry

        def get_small(y_tuple):
            cached = small_cache.get(y_tuple)
            if cached is not None:
                small_cache.move_to_end(y_tuple)
                return cached
            roots, zs = enumerate_roots_small(
                y_tuple, locator, phase_meta, batch_cache, eps, f
            )
            z_re = np.ascontiguousarray(zs.real, dtype=np.float64)
            z_im = np.ascontiguousarray(zs.imag, dtype=np.float64)
            entry = (roots, zs, z_re, z_im)
            small_cache[y_tuple] = entry
            if len(small_cache) > _LRU_CAP:
                small_cache.popitem(last=False)
            return entry

        # Pre-allocated pair-filter output buffers (Round 3, 2026-05-13).
        # Hoisted out of the triple loop to avoid 16 MB of malloc/free per
        # iteration. Initial cap is generous; grows on demand if a triple has
        # n_b * n_c larger than current cap.
        _idx_buf_cap = 1 << 20  # 1 048 576 entries (8 MB per int64 buf)
        idx_b_buf = np.empty(_idx_buf_cap, dtype=np.int64)
        idx_c_buf = np.empty(_idx_buf_cap, dtype=np.int64)

        for tri in triples:
            if max_matches > 0 and counters.matrices_found >= max_matches:
                break
            counters.triples_processed += 1
            Y0 = (int(tri[0]), int(tri[1]), int(tri[2]))
            Y1 = (int(tri[3]), int(tri[4]), int(tri[5]))
            Y2 = (int(tri[6]), int(tri[7]), int(tri[8]))

            # Identify the small slot.  In the smoke-test cell sigma1(Y_2)/3^{2f}
            # is in [0, 0.0025] so slot 2 is the small one.  We still detect
            # defensively in case the file layout changes.
            sig = (float(sigma1_from_m012(*Y0)),
                   float(sigma1_from_m012(*Y1)),
                   float(sigma1_from_m012(*Y2)))
            small_slot = int(np.argmin(sig))

            # X_{(0,1)} reflects the Householder formula across slots 0 and 1
            # of u.  The two non-zero entries of u must therefore live in
            # slots 0 and 1, and the small entry in slot 2.  If the small Y is
            # in slot 0 or 1 we cannot form a diagonal V from this triple -- skip.
            if small_slot != 2:
                heartbeat("main")
                continue

            Yb = Y0  # u_0, |u_0| ~ 1
            Yc = Y1  # u_1, |u_1| ~ 1
            Ya = Y2  # u_2, |u_2| ~ 0

            roots_b, z_b, zb_re, zb_im = get_large(Yb)
            if roots_b.shape[0] == 0:
                counters.triples_with_empty_desc += 1
                heartbeat("main")
                continue
            roots_c, z_c, zc_re, zc_im = get_large(Yc)
            if roots_c.shape[0] == 0:
                counters.triples_with_empty_desc += 1
                heartbeat("main")
                continue
            roots_a, z_a, _za_re, _za_im = get_small(Ya)
            if roots_a.shape[0] == 0:
                counters.triples_with_empty_desc += 1
                heartbeat("main")
                continue

            counters.a_large_roots_seen += roots_b.shape[0] + roots_c.shape[0]
            counters.a_small_roots_seen += roots_a.shape[0]

            n_b = roots_b.shape[0]
            n_c = roots_c.shape[0]

            # Phase filter: (z_b * conj(z_c)) ≈ target_product within eps.
            counters.a01_pairs_tested += n_b * n_c
            if _HAVE_NUMBA:
                # Fused Numba kernel writing passing indices into pre-allocated
                # int64 arrays. Buffers are hoisted out of the triple loop;
                # grow on demand if a triple's n_b * n_c exceeds the current cap.
                cap_needed = n_b * n_c
                if cap_needed > _idx_buf_cap:
                    _idx_buf_cap = max(cap_needed, _idx_buf_cap * 2)
                    idx_b_buf = np.empty(_idx_buf_cap, dtype=np.int64)
                    idx_c_buf = np.empty(_idx_buf_cap, dtype=np.int64)
                n_match = int(_hh_inner_filter_pairs_nb(
                    zb_re, zb_im, zc_re, zc_im,
                    float(target_product.real), float(target_product.imag),
                    float(eps_plus_pad),
                    idx_b_buf, idx_c_buf,
                ))
                if n_match == 0:
                    heartbeat("main")
                    continue
                ib_arr = idx_b_buf[:n_match]
                ic_arr = idx_c_buf[:n_match]
            else:
                # Numpy fallback (correct but allocates the full grid).
                zb_col = z_b[:, None]      # (n_b, 1)
                zc_row = np.conj(z_c)[None, :]  # (1, n_c)
                prod = zb_col * zc_row     # (n_b, n_c)
                mask = np.abs(prod - target_product) <= eps_plus_pad
                n_match = int(mask.sum())
                if n_match == 0:
                    heartbeat("main")
                    continue
                ib_arr, ic_arr = np.nonzero(mask)
            counters.a01_pairs_pass_phase += n_match

            # Materialize V for each (ib, ic, ia) and measure Frobenius.
            for k in range(n_match):
                if max_matches > 0 and counters.matrices_found >= max_matches:
                    break
                ib = int(ib_arr[k])
                ic = int(ic_arr[k])
                a_b = roots_b[ib]
                a_c = roots_c[ic]
                for ia in range(roots_a.shape[0]):
                    a_a = roots_a[ia]
                    # u in original slot order: (a_b, a_c, a_a) since small_slot=2.
                    V_int = build_householder_V_exact(a_b, a_c, a_a, f)
                    Vc = V_to_complex(V_int, f_v)
                    dist_fro = float(np.linalg.norm(Vc - np.diag(diag)))
                    counters.matrices_found += 1
                    result = {
                        "dist_fro": float(dist_fro),
                        "Y0": list(Y0),
                        "Y1": list(Y1),
                        "Y2": list(Y2),
                        "small_slot": int(small_slot),
                        "rows_coeffs_layer_f": V_int.tolist(),
                        "u_complex": [
                            [float(complex(zz).real), float(complex(zz).imag)]
                            for zz in (z_b[ib], z_c[ic], z_a[ia])
                        ],
                    }
                    maybe_push_result(heap, result, keep)
                    heartbeat("main")
            heartbeat("main")
        heartbeat("main", force=(rank == 0))

    local_results = [item.result for item in heap]
    payload = {"rank": rank, "counters": asdict(counters), "results": local_results}
    gathered = comm.gather(payload, root=0)
    if rank != 0:
        return None

    total = HHCounters()
    final_heap: list[HeapItem] = []
    by_rank = {}
    for part in gathered:
        c = HHCounters(**part["counters"])
        total.add(c)
        by_rank[str(part["rank"])] = part["counters"]
        for r in part["results"]:
            maybe_push_result(final_heap, r, keep)

    final_results = [item.result for item in sorted(final_heap, key=lambda x: -x.priority)]
    final_results.sort(key=lambda r: float(r["dist_fro"]))

    out_npz = output_prefix + ".npz"
    if final_results:
        rows_arr = np.asarray([r["rows_coeffs_layer_f"] for r in final_results], dtype=np.int64)
        dist_arr = np.asarray([r["dist_fro"] for r in final_results], dtype=np.float64)
    else:
        rows_arr = np.empty((0, 3, 3, 6), dtype=np.int64)
        dist_arr = np.empty((0,), dtype=np.float64)
    np.savez(
        out_npz,
        rows_coeffs_layer_f=rows_arr,
        dist_fro=dist_arr,
        diag=np.asarray(diag, dtype=np.complex128),
        f=np.asarray([f_v], dtype=np.int64),
        eps=np.asarray([eps], dtype=np.float64),
        mode=np.asarray(["householder"], dtype="U16"),
    )

    summary = {
        "mode": "householder",
        "results_npz": os.path.abspath(out_npz),
        "results_json": os.path.abspath(output_prefix + ".json"),
        "profile_json": os.path.abspath(output_prefix + ".profile.json"),
        "n_results": len(final_results),
        "best_dist_fro": None if not final_results else float(final_results[0]["dist_fro"]),
        "f_u": int(f),
        "f_v": int(f_v),
        "eps": float(eps),
        "diag": [[float(z.real), float(z.imag)] for z in diag],
        "triples_file": os.path.abspath(triples_file),
        "triples_json": os.path.abspath(triples_json),
        "rootdb_prefix": os.path.abspath(rootdb_prefix),
        "mpi_ranks": int(size),
        "n_phase_bins": int(n_phase_bins),
        "counters_sum": asdict(total),
        "top_results": final_results[: min(8, len(final_results))],
    }
    with open(output_prefix + ".json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with open(output_prefix + ".profile.json", "w") as fh:
        json.dump({"by_rank": by_rank, "sum": asdict(total)}, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def search_householder_streamed_batched(
    *,
    queries: list,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    output_dir: str,
    triples_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: Optional[str] = None,
    keep: int = 16,
    verbose: bool = True,
    max_matches: int = 0,
) -> Optional[dict]:
    """Run stage-5 Householder search against MANY (theta, eps) queries in one
    sweep over the triples. V1 constraint: all queries must share the same
    `eps` (so the per-triple root caches can be reused). Only the phase filter
    and the V-vs-diag comparison vary across queries.

    queries: list of {"id": str, "theta": float, "eps": float}.
    output_dir: directory where per-query .json/.npz outputs go, named by id.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = time.monotonic()

    nq = len(queries)
    if nq == 0:
        raise ValueError("queries: at least one query required")

    eps_set = sorted({float(q["eps"]) for q in queries})
    if len(eps_set) != 1:
        raise ValueError(
            f"batched stage-5 V1 requires uniform eps across queries, got {eps_set}"
        )
    eps = float(eps_set[0])
    eps_plus_pad = eps + 1.0e-15

    qids = [str(q.get("id", i)) for i, q in enumerate(queries)]
    thetas = [float(q["theta"]) for q in queries]
    target_real = np.empty(nq, dtype=np.float64)
    target_imag = np.empty(nq, dtype=np.float64)
    diags = np.empty((nq, 3), dtype=np.complex128)
    for i, theta in enumerate(thetas):
        d1 = complex(math.cos(-0.5 * theta), math.sin(-0.5 * theta))
        d2 = complex(math.cos(0.5 * theta), math.sin(0.5 * theta))
        d3 = complex(1.0, 0.0)
        tp = -d1
        target_real[i] = tp.real
        target_imag[i] = tp.imag
        diags[i, 0] = d1
        diags[i, 1] = d2
        diags[i, 2] = d3

    f_v = 2 * int(f)

    if rank == 0:
        tmeta = _read_manifest(triples_json)
        nrows = int(tmeta["rows_written"])
        file_size = os.path.getsize(triples_file)
        expected = nrows * 9 * np.dtype(np.int64).itemsize
        if file_size != expected:
            raise RuntimeError(f"Triple file size mismatch: expected {expected}, got {file_size}")
        paths = _state_paths(rootdb_prefix, n_phase_bins)
        if not os.path.exists(paths["exact_roots_index_meta"]):
            raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")
        if not os.path.exists(paths["phase_sidecar_meta"]):
            raise FileNotFoundError(
                f"Binned phase sidecar not found. Run sidecar builder first for {rootdb_prefix}"
            )
        phase_meta = _read_manifest(paths["phase_sidecar_meta"])
        chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
        os.makedirs(output_dir, exist_ok=True)
        print(
            f"[hh-stage5-batched] streamed Householder search | rows={nrows:,} "
            f"ranks={size} f={f} f_v={f_v} eps={eps:g} queries={nq}",
            flush=True,
        )
    else:
        nrows = None
        paths = None
        phase_meta = None
        chunk_meta = None

    nrows = comm.bcast(nrows, root=0)
    paths = comm.bcast(paths, root=0)
    phase_meta = comm.bcast(phase_meta, root=0)
    chunk_meta = comm.bcast(chunk_meta, root=0)

    batch_cache: OrderedDict = OrderedDict()
    heaps = [[] for _ in range(nq)]
    per_q_counters = [HHCounters() for _ in range(nq)]
    shared_counters = HHCounters()
    n_chunks = (int(nrows) + int(triples_chunk_rows) - 1) // int(triples_chunk_rows)
    last_hb = [time.monotonic()]

    def heartbeat(label: str, force: bool = False):
        now = time.monotonic()
        if not force and now - last_hb[0] < 10.0:
            return
        last_hb[0] = now
        if rank != 0:
            return
        elapsed = now - start_time
        total_matrices = sum(c.matrices_found for c in per_q_counters)
        print(
            f"[hh-stage5-batched/{label}] elapsed={elapsed:,.1f}s | "
            f"triples {shared_counters.triples_processed:,}/{nrows:,} | "
            f"a01 tested {shared_counters.a01_pairs_tested:,} | "
            f"matrices(sum) {total_matrices:,}",
            flush=True,
        )

    heartbeat("startup", force=True)

    _idx_buf_cap = 1 << 20
    idx_b_buf = np.empty(_idx_buf_cap, dtype=np.int64)
    idx_c_buf = np.empty(_idx_buf_cap, dtype=np.int64)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * int(triples_chunk_rows)
        take = min(int(triples_chunk_rows), int(nrows) - start)
        shared_counters.chunks_seen += 1
        if chunk_idx % size != rank:
            continue
        shared_counters.chunks_processed += 1

        triples = _read_triple_rows(triples_file, start, take)
        if verbose and rank == 0:
            print(
                f"[hh-stage5-batched] chunk {chunk_idx+1}/{n_chunks} "
                f"rows {start:,}..{start + take - 1:,}", flush=True,
            )

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_idx]
            _needed, locator = _load_chunk_metadata(rec["path"])
        else:
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            locator = _build_locator(needed, paths)

        # LRU-capped per-chunk caches (memory refactor #4, 2026-05-22). Cap
        # bounds working set; misses re-hydrate from mmap on next access.
        large_cache: OrderedDict = OrderedDict()
        large_canon_cache: OrderedDict = OrderedDict()  # canonicalized version of large roots (for Y_b only)
        small_cache: OrderedDict = OrderedDict()
        _LRU_CAP = 4096

        def get_large(y_tuple):
            cached = large_cache.get(y_tuple)
            if cached is not None:
                large_cache.move_to_end(y_tuple)
                return cached
            roots, zs = enumerate_roots_annulus(
                y_tuple, locator, phase_meta, batch_cache, 1.0, eps, f
            )
            z_re = np.ascontiguousarray(zs.real, dtype=np.float64)
            z_im = np.ascontiguousarray(zs.imag, dtype=np.float64)
            entry = (roots, zs, z_re, z_im)
            large_cache[y_tuple] = entry
            if len(large_cache) > _LRU_CAP:
                large_cache.popitem(last=False)
            return entry

        def get_large_canonical(y_tuple):
            """Like get_large but filtered to one rep per ζ_9-torsion orbit.
            Used for Y_b only — canonicalizing one slot factors out the
            18-fold diagonal-torsion redundancy in (a_b, a_c, a_a) tuples."""
            cached = large_canon_cache.get(y_tuple)
            if cached is not None:
                large_canon_cache.move_to_end(y_tuple)
                return cached
            full_roots, full_zs, _, _ = get_large(y_tuple)
            if full_roots.shape[0] == 0:
                entry = (full_roots, full_zs,
                         np.empty(0, dtype=np.float64),
                         np.empty(0, dtype=np.float64))
            else:
                keep_idx = _canonical_orbit_indices(full_roots)
                cr = np.ascontiguousarray(full_roots[keep_idx], dtype=np.int64)
                cz = np.ascontiguousarray(full_zs[keep_idx], dtype=np.complex128)
                cze = np.ascontiguousarray(cz.real, dtype=np.float64)
                czi = np.ascontiguousarray(cz.imag, dtype=np.float64)
                entry = (cr, cz, cze, czi)
            large_canon_cache[y_tuple] = entry
            if len(large_canon_cache) > _LRU_CAP:
                large_canon_cache.popitem(last=False)
            return entry

        def get_small(y_tuple):
            cached = small_cache.get(y_tuple)
            if cached is not None:
                small_cache.move_to_end(y_tuple)
                return cached
            roots, zs = enumerate_roots_small(
                y_tuple, locator, phase_meta, batch_cache, eps, f
            )
            z_re = np.ascontiguousarray(zs.real, dtype=np.float64)
            z_im = np.ascontiguousarray(zs.imag, dtype=np.float64)
            entry = (roots, zs, z_re, z_im)
            small_cache[y_tuple] = entry
            if len(small_cache) > _LRU_CAP:
                small_cache.popitem(last=False)
            return entry

        # Global early-exit: if EVERY query has hit max_matches, no more useful work.
        if max_matches > 0 and all(
            c.matrices_found >= max_matches for c in per_q_counters
        ):
            break  # break outer chunk loop

        for tri in triples:
            # Per-triple early-exit: same condition (cheap check).
            if max_matches > 0 and all(
                c.matrices_found >= max_matches for c in per_q_counters
            ):
                break
            shared_counters.triples_processed += 1
            Y0 = (int(tri[0]), int(tri[1]), int(tri[2]))
            Y1 = (int(tri[3]), int(tri[4]), int(tri[5]))
            Y2 = (int(tri[6]), int(tri[7]), int(tri[8]))
            sig = (
                float(sigma1_from_m012(*Y0)),
                float(sigma1_from_m012(*Y1)),
                float(sigma1_from_m012(*Y2)),
            )
            small_slot = int(np.argmin(sig))
            if small_slot != 2:
                heartbeat("main")
                continue

            Yb, Yc, Ya = Y0, Y1, Y2
            # Canonicalize Y_b roots to one rep per ζ_9-torsion orbit. This
            # factors the 18-fold diagonal-torsion redundancy in (a_b,a_c,a_a)
            # tuples — every distinct V is hit exactly once.
            roots_b, z_b, zb_re, zb_im = get_large_canonical(Yb)
            if roots_b.shape[0] == 0:
                shared_counters.triples_with_empty_desc += 1
                heartbeat("main")
                continue
            roots_c, z_c, zc_re, zc_im = get_large(Yc)
            if roots_c.shape[0] == 0:
                shared_counters.triples_with_empty_desc += 1
                heartbeat("main")
                continue
            roots_a, z_a, _za_re, _za_im = get_small(Ya)
            if roots_a.shape[0] == 0:
                shared_counters.triples_with_empty_desc += 1
                heartbeat("main")
                continue

            shared_counters.a_large_roots_seen += roots_b.shape[0] + roots_c.shape[0]
            shared_counters.a_small_roots_seen += roots_a.shape[0]
            n_b = roots_b.shape[0]
            n_c = roots_c.shape[0]
            n_a = roots_a.shape[0]
            pair_count = n_b * n_c
            shared_counters.a01_pairs_tested += pair_count
            cap_needed = pair_count
            if cap_needed > _idx_buf_cap:
                _idx_buf_cap = max(cap_needed, _idx_buf_cap * 2)
                idx_b_buf = np.empty(_idx_buf_cap, dtype=np.int64)
                idx_c_buf = np.empty(_idx_buf_cap, dtype=np.int64)

            # V is theta-independent: cache by (ib, ic, ia) within this triple
            # to avoid rebuilding when multiple queries select the same pair.
            v_cache: dict = {}

            for qi in range(nq):
                if max_matches > 0 and per_q_counters[qi].matrices_found >= max_matches:
                    continue
                if _HAVE_NUMBA:
                    n_match = int(_hh_inner_filter_pairs_nb(
                        zb_re, zb_im, zc_re, zc_im,
                        float(target_real[qi]), float(target_imag[qi]),
                        float(eps_plus_pad),
                        idx_b_buf, idx_c_buf,
                    ))
                    if n_match == 0:
                        continue
                    ib_arr = idx_b_buf[:n_match]
                    ic_arr = idx_c_buf[:n_match]
                else:
                    tp_q = complex(target_real[qi], target_imag[qi])
                    zb_col = z_b[:, None]
                    zc_row = np.conj(z_c)[None, :]
                    prod = zb_col * zc_row
                    mask = np.abs(prod - tp_q) <= eps_plus_pad
                    n_match = int(mask.sum())
                    if n_match == 0:
                        continue
                    ib_arr, ic_arr = np.nonzero(mask)

                per_q_counters[qi].a01_pairs_pass_phase += n_match
                diag_q = diags[qi]

                for k in range(n_match):
                    if max_matches > 0 and per_q_counters[qi].matrices_found >= max_matches:
                        break
                    ib = int(ib_arr[k])
                    ic = int(ic_arr[k])
                    for ia in range(n_a):
                        key = (ib, ic, ia)
                        vrec = v_cache.get(key)
                        if vrec is None:
                            V_int = build_householder_V_exact(
                                roots_b[ib], roots_c[ic], roots_a[ia], f
                            )
                            Vc = V_to_complex(V_int, f_v)
                            vrec = (V_int, Vc)
                            v_cache[key] = vrec
                        V_int, Vc = vrec
                        dist_fro = float(np.linalg.norm(Vc - np.diag(diag_q)))
                        per_q_counters[qi].matrices_found += 1
                        result = {
                            "dist_fro": float(dist_fro),
                            "Y0": list(Y0), "Y1": list(Y1), "Y2": list(Y2),
                            "small_slot": int(small_slot),
                            "rows_coeffs_layer_f": V_int.tolist(),
                            "u_complex": [
                                [float(complex(zz).real), float(complex(zz).imag)]
                                for zz in (z_b[ib], z_c[ic], z_a[ia])
                            ],
                        }
                        maybe_push_result(heaps[qi], result, keep)
                heartbeat("main")
            heartbeat("main")
        heartbeat("main", force=(rank == 0))

    # Per-query gather + write.
    overall = {"mode": "householder-batched", "eps": eps, "queries": []}
    for qi in range(nq):
        local_results = [item.result for item in heaps[qi]]
        payload = {
            "rank": rank,
            "counters": asdict(per_q_counters[qi]),
            "results": local_results,
        }
        gathered = comm.gather(payload, root=0)
        if rank != 0:
            continue
        total = HHCounters()
        final_heap: list = []
        by_rank = {}
        for part in gathered:
            c = HHCounters(**part["counters"])
            total.add(c)
            by_rank[str(part["rank"])] = part["counters"]
            for r in part["results"]:
                maybe_push_result(final_heap, r, keep)
        final_results = [item.result for item in sorted(final_heap, key=lambda x: -x.priority)]
        final_results.sort(key=lambda r: float(r["dist_fro"]))

        output_prefix = os.path.join(output_dir, f"q_{qids[qi]}")
        out_npz = output_prefix + ".npz"
        if final_results:
            rows_arr = np.asarray([r["rows_coeffs_layer_f"] for r in final_results], dtype=np.int64)
            dist_arr = np.asarray([r["dist_fro"] for r in final_results], dtype=np.float64)
        else:
            rows_arr = np.empty((0, 3, 3, 6), dtype=np.int64)
            dist_arr = np.empty((0,), dtype=np.float64)
        np.savez(
            out_npz,
            rows_coeffs_layer_f=rows_arr,
            dist_fro=dist_arr,
            diag=np.asarray(diags[qi], dtype=np.complex128),
            f=np.asarray([f_v], dtype=np.int64),
            eps=np.asarray([eps], dtype=np.float64),
            mode=np.asarray(["householder"], dtype="U16"),
        )
        q_summary = {
            "id": qids[qi],
            "theta": thetas[qi],
            "eps": eps,
            "results_npz": os.path.abspath(out_npz),
            "results_json": os.path.abspath(output_prefix + ".json"),
            "n_results": len(final_results),
            "best_dist_fro": None if not final_results else float(final_results[0]["dist_fro"]),
            "f_u": int(f), "f_v": int(f_v),
            "diag": [[float(z.real), float(z.imag)] for z in diags[qi]],
            "counters_sum": asdict(total),
            "top_results": final_results[: min(8, len(final_results))],
        }
        with open(output_prefix + ".json", "w") as fh:
            json.dump(q_summary, fh, indent=2, sort_keys=True)
        with open(output_prefix + ".profile.json", "w") as fh:
            json.dump({"by_rank": by_rank, "sum": asdict(total)}, fh, indent=2, sort_keys=True)
        overall["queries"].append({
            "id": qids[qi], "theta": thetas[qi],
            "n_results": len(final_results),
            "best_dist_fro": q_summary["best_dist_fro"],
            "results_json": q_summary["results_json"],
        })

    if rank != 0:
        return None

    overall["triples_file"] = os.path.abspath(triples_file)
    overall["mpi_ranks"] = int(size)
    overall["shared_counters"] = asdict(shared_counters)
    with open(os.path.join(output_dir, "batch_summary.json"), "w") as fh:
        json.dump(overall, fh, indent=2, sort_keys=True)
    print(json.dumps({"batch_summary": overall["queries"]}, indent=2), flush=True)
    return overall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Streamed Householder-vector search for an R^Z diagonal target over Z[zeta9, 1/3]."
    )
    p.add_argument("--triples_file", required=True)
    p.add_argument("--triples_json", required=True)
    p.add_argument("--rootdb_prefix", required=True)
    p.add_argument("--f", type=int, required=True,
                   help="zeta9 internal --f (= V-denominator exponent / 2 in u-denom semantics).")
    p.add_argument("--eps", type=float, default=None,
                   help="single-query eps; required unless --queries is given.")
    p.add_argument("--output_prefix", default=None,
                   help="single-query output prefix; required unless --queries is given.")
    p.add_argument("--theta", type=float, default=None)
    p.add_argument("--d1", default=None)
    p.add_argument("--d2", default=None)
    p.add_argument("--d3", default="1")
    p.add_argument("--queries", default=None,
                   help="JSON file with [{id, theta, eps}, ...]. Mutually exclusive with "
                        "--theta/--eps. V1 requires uniform eps across queries.")
    p.add_argument("--output_dir", default=None,
                   help="output directory for batched per-query files; "
                        "required with --queries.")
    p.add_argument("--triples_chunk_rows", type=int, default=200000)
    p.add_argument("--n_phase_bins", type=int, default=512)
    p.add_argument("--chunk_meta_json", default=None)
    p.add_argument("--keep", type=int, default=16)
    p.add_argument("--max_matches", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.queries is not None:
        if args.theta is not None or args.d1 is not None or args.d2 is not None:
            raise SystemExit("--queries is mutually exclusive with --theta/--d1/--d2.")
        if args.eps is not None or args.output_prefix is not None:
            raise SystemExit("--queries replaces --eps/--output_prefix; use --output_dir.")
        if args.output_dir is None:
            raise SystemExit("--output_dir is required when using --queries.")
        with open(args.queries, "r") as fh:
            queries = json.load(fh)
        if not isinstance(queries, list) or not queries:
            raise SystemExit(f"--queries: expected non-empty JSON list, got {type(queries)}")
        search_householder_streamed_batched(
            queries=queries,
            triples_file=args.triples_file,
            triples_json=args.triples_json,
            rootdb_prefix=args.rootdb_prefix,
            f=args.f,
            output_dir=args.output_dir,
            triples_chunk_rows=args.triples_chunk_rows,
            n_phase_bins=args.n_phase_bins,
            chunk_meta_json=args.chunk_meta_json,
            keep=args.keep,
            verbose=not args.quiet,
            max_matches=args.max_matches,
        )
        return

    if args.eps is None or args.output_prefix is None:
        raise SystemExit("--eps and --output_prefix are required for single-query mode.")

    if args.theta is not None:
        d1 = complex(math.cos(-0.5 * args.theta), math.sin(-0.5 * args.theta))
        d2 = complex(math.cos(0.5 * args.theta), math.sin(0.5 * args.theta))
        d3 = 1.0 + 0.0j
    else:
        if args.d1 is None or args.d2 is None:
            raise SystemExit("Either provide --theta, or provide both --d1 and --d2.")
        d1 = complex(args.d1)
        d2 = complex(args.d2)
        d3 = complex(args.d3)

    search_householder_streamed(
        triples_file=args.triples_file,
        triples_json=args.triples_json,
        rootdb_prefix=args.rootdb_prefix,
        f=args.f,
        d1=d1, d2=d2, d3=d3,
        eps=args.eps,
        output_prefix=args.output_prefix,
        triples_chunk_rows=args.triples_chunk_rows,
        n_phase_bins=args.n_phase_bins,
        chunk_meta_json=args.chunk_meta_json,
        keep=args.keep,
        verbose=not args.quiet,
        max_matches=args.max_matches,
    )


if __name__ == "__main__":
    main()
