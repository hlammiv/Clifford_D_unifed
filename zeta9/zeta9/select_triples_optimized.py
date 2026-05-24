import argparse
import glob
import json
import math
import os
import struct
import time
import zlib
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

# Memory refactor #A4 (2026-05-22): bucket files are 8× zlib-L3-compressible
# (measured on actual stage-2 output). Disk format is length-prefixed chunks
# preceded by a 4-byte magic. Backward-compatible: files without the magic
# are read as legacy raw int64.
_BUCKET_MAGIC = b"Z9B1"
_BUCKET_COMPRESS_LEVEL = 3

# ============================================================
# Compact "drop column C" final-output format (lossless prune, 2026-05-24)
# ============================================================
# The final TM file conventionally stores 9 int64 per triple: (Y_a, Y_b, Y_c)
# in the original-coordinate order. But the third triple component is fully
# determined by Y_a + Y_b + Y_c = target_sum, so Y_c can be reconstructed on
# read from (Y_a, Y_b) and the constant target_sum. Writing only 6 int64 per
# triple shaves 1/3 off the on-disk size (72 B → 48 B per row, plus a small
# fixed header).
#
# Format (when --drop_col_c is enabled):
#   bytes 0..3   : magic b"Z9TC"
#   bytes 4..7   : little-endian uint32 version (= 1)
#   bytes 8..15  : little-endian int64 target_sum[0]   (== norm * 3^{2f})
#   bytes 16..23 : little-endian int64 target_sum[1]   (== 0 normally)
#   bytes 24..31 : little-endian int64 target_sum[2]   (== 0 normally)
#   bytes 32..   : nrows * 6 int64 (Y_a[0..2], Y_b[0..2]) in native byte order
#
# The 32-byte header makes row-level random access trivial:
#   offset(row i) = 32 + i * 6 * 8 = 32 + i * 48
# and avoids any guessing about the row count via file size.
#
# Legacy format (no header) remains the default for back-compat. Readers
# detect the magic in the first 4 bytes; if absent, fall back to legacy.
_TM_COMPACT_MAGIC = b"Z9TC"
_TM_COMPACT_VERSION = 1
_TM_COMPACT_HEADER_SIZE = 32
_TM_COMPACT_ROW_BYTES = 6 * 8  # 6 int64 per row in the compact format

try:
    import numba as nb  # type: ignore
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# ============================================================
# basic helpers
# ============================================================

ROW_DTYPE = np.int64
STRUCT_DTYPE = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])


def tuple3(x):
    return (int(x[0]), int(x[1]), int(x[2]))


def _resolve_input_specs(specs: List[str]) -> List[str]:
    out = []
    for spec in specs:
        matches = sorted(glob.glob(spec))
        if matches:
            out.extend(matches)
        elif os.path.exists(spec):
            out.append(spec)
    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def _count_rows(paths: List[str]) -> int:
    total = 0
    for p in paths:
        a = np.load(p, mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"{p} does not have shape (n,3)")
        total += int(a.shape[0])
    return total


def _load_stack(paths: List[str], dedup=True) -> np.ndarray:
    arrs = []
    for p in paths:
        a = np.load(p, mmap_mode="r")
        a = np.asarray(a, dtype=ROW_DTYPE)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"{p} does not have shape (n,3)")
        arrs.append(a)
    if not arrs:
        return np.empty((0, 3), dtype=ROW_DTYPE)
    out = np.concatenate(arrs, axis=0)
    if dedup and out.size:
        out = np.unique(out, axis=0)
    return np.ascontiguousarray(out, dtype=ROW_DTYPE)


def _iter_npy_rows(paths: List[str], rows_per_chunk: int):
    for p in paths:
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{p} does not have shape (n,3)")
        n = int(arr.shape[0])
        for i0 in range(0, n, rows_per_chunk):
            i1 = min(i0 + rows_per_chunk, n)
            yield p, i0, i1, np.asarray(arr[i0:i1], dtype=ROW_DTYPE)


def _read_manifest(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


# Profile-driven fix (2026-05-12): completed_buckets was stored as a JSON list
# in the manifest and re-serialized after every phase. With 8192 buckets that's
# 33M list-element encodes per cell (~22s in cProfile output at f=2 ε=0.05 —
# 84% of stage 2 walltime). Replaced with a fixed-size byte bitmap in a
# separate file (8192 bits = 1024 bytes), written atomically per phase.
# Backward compat: older manifests with a "completed_buckets" list are still
# accepted on resume.

def _completed_bitmap_path(manifest_path: str) -> str:
    return manifest_path + ".completed_bits"


def _init_completed_bitmap(manifest_path: str, n_buckets: int, manifest: dict) -> bytearray:
    """Return an in-memory bytearray bitmap of length ceil(n_buckets/8).
    Initializes from the on-disk bitmap if it exists, else from a legacy
    'completed_buckets' list in the manifest (back compat), else zeros."""
    nbytes = (n_buckets + 7) // 8
    path = _completed_bitmap_path(manifest_path)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            data = fh.read()
        buf = bytearray(nbytes)
        # Copy in whatever fits (defensive against file-size drift).
        m = min(len(data), nbytes)
        buf[:m] = data[:m]
        return buf
    buf = bytearray(nbytes)
    for x in manifest.get("completed_buckets", []):  # legacy fallback
        bi = int(x)
        if 0 <= bi < n_buckets:
            buf[bi >> 3] |= (1 << (bi & 7))
    return buf


def _bitmap_set(buf: bytearray, bucket_id: int):
    buf[bucket_id >> 3] |= (1 << (bucket_id & 7))


def _bitmap_to_set(buf: bytearray, n_buckets: int) -> set:
    return {i for i in range(min(n_buckets, len(buf) * 8))
            if buf[i >> 3] & (1 << (i & 7))}


def _save_completed_bitmap(manifest_path: str, buf: bytearray):
    """Atomic write of the bitmap. Fixed-size O(bitmap bytes), no set iteration."""
    path = _completed_bitmap_path(manifest_path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(bytes(buf))
    os.replace(tmp, path)


def _append_rows_raw(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=ROW_DTYPE)
    if arr.size == 0:
        return
    raw = arr.tobytes()
    compressed = zlib.compress(raw, _BUCKET_COMPRESS_LEVEL)
    # First write to a new file gets the 4-byte magic; subsequent appends do not.
    needs_magic = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "ab") as fh:
        if needs_magic:
            fh.write(_BUCKET_MAGIC)
        fh.write(struct.pack("<I", len(compressed)))
        fh.write(compressed)


def _read_bucket_bytes(path: str) -> bytes:
    """Return raw int64-payload bytes for a bucket file.

    Auto-detects new (magic + length-prefixed zlib chunks) vs legacy raw int64.
    Returns empty bytes for missing or empty files.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except (OSError, FileNotFoundError):
        return b""
    if len(data) == 0:
        return b""
    if data[:4] == _BUCKET_MAGIC:
        chunks = []
        pos = 4
        n = len(data)
        while pos < n:
            if pos + 4 > n:
                raise ValueError(f"truncated chunk length in {path} at {pos}")
            (length,) = struct.unpack_from("<I", data, pos)
            pos += 4
            if length == 0 or pos + length > n:
                raise ValueError(f"bad chunk length {length} in {path} at {pos}")
            chunks.append(zlib.decompress(data[pos:pos+length]))
            pos += length
        return b"".join(chunks)
    # Legacy raw int64 format (pre-A4).
    return data


def _read_bucket_rows(path: str) -> np.ndarray:
    """Return rows from a bucket file as a (n,) int64 array (flat). Caller reshapes."""
    payload = _read_bucket_bytes(path)
    if not payload:
        return np.empty((0,), dtype=ROW_DTYPE)
    return np.frombuffer(payload, dtype=ROW_DTYPE)


def _write_tm_compact_header(fh, target_sum: np.ndarray):
    """Write the 32-byte compact-format header to an open file handle."""
    fh.write(_TM_COMPACT_MAGIC)
    fh.write(struct.pack("<I", _TM_COMPACT_VERSION))
    ts = np.asarray(target_sum, dtype=ROW_DTYPE).reshape(-1)
    if ts.shape[0] != 3:
        raise ValueError(f"target_sum must have 3 elements, got {ts.shape[0]}")
    fh.write(ts.tobytes())


def read_tm_compact_header(path: str):
    """Return (version, target_sum np.int64[3]) if `path` is a compact TM file,
    else None. Cheap: only reads 32 bytes."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_TM_COMPACT_HEADER_SIZE)
    except (OSError, FileNotFoundError):
        return None
    if len(head) < _TM_COMPACT_HEADER_SIZE or head[:4] != _TM_COMPACT_MAGIC:
        return None
    (version,) = struct.unpack_from("<I", head, 4)
    ts = np.frombuffer(head[8:32], dtype=ROW_DTYPE).copy()
    return int(version), ts


def read_tm_rows_compact(path: str, start_row: int, nrows: int) -> np.ndarray:
    """Read `nrows` triples from a compact-format TM file starting at `start_row`.

    Returns a (nrows, 9) int64 array, expanding the dropped Y_c column on the
    fly. Caller is responsible for passing valid (start, count) within the file.
    Mirrors the API of the legacy raw reader so it can be drop-in.
    """
    if nrows <= 0:
        return np.empty((0, 9), dtype=ROW_DTYPE)
    info = read_tm_compact_header(path)
    if info is None:
        raise ValueError(f"{path} is not a compact-format TM file")
    _, target_sum = info
    offset = _TM_COMPACT_HEADER_SIZE + int(start_row) * _TM_COMPACT_ROW_BYTES
    with open(path, "rb") as fh:
        fh.seek(offset, os.SEEK_SET)
        raw = np.fromfile(fh, dtype=ROW_DTYPE, count=int(nrows) * 6)
    got = raw.shape[0] // 6
    if got <= 0:
        return np.empty((0, 9), dtype=ROW_DTYPE)
    ab = raw.reshape(got, 6)
    out = np.empty((got, 9), dtype=ROW_DTYPE)
    out[:, 0:6] = ab
    out[:, 6:9] = target_sum[None, :] - ab[:, 0:3] - ab[:, 3:6]
    return out


def tm_compact_row_count(path: str) -> int:
    """Return the number of rows in a compact-format TM file (from file size)."""
    info = read_tm_compact_header(path)
    if info is None:
        raise ValueError(f"{path} is not a compact-format TM file")
    sz = os.path.getsize(path)
    body = sz - _TM_COMPACT_HEADER_SIZE
    if body < 0 or body % _TM_COMPACT_ROW_BYTES != 0:
        raise ValueError(
            f"compact TM file {path}: body size {body} not a multiple of {_TM_COMPACT_ROW_BYTES}"
        )
    return body // _TM_COMPACT_ROW_BYTES


def _buffered_gather_counts(comm, local_value, root=0, tag=9100):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == root:
        vals = [0] * size
        vals[root] = int(local_value)
        for src in range(size):
            if src == root:
                continue
            vals[src] = int(comm.recv(source=src, tag=tag))
        return vals
    comm.send(int(local_value), dest=root, tag=tag)
    return None


# ============================================================
# exact modular bucket hash
# ============================================================


def _bucket_linear_hash_rows(rows: np.ndarray, n_buckets: int, coeffs: Tuple[int, int, int]) -> np.ndarray:
    c0, c1, c2 = coeffs
    x = rows[:, 0].astype(np.int64) * np.int64(c0)
    x += rows[:, 1].astype(np.int64) * np.int64(c1)
    x += rows[:, 2].astype(np.int64) * np.int64(c2)
    return np.mod(x, np.int64(n_buckets)).astype(np.int64)


# ============================================================
# structured/sorted exact-key helpers
# ============================================================


def _as_struct(arr: np.ndarray) -> np.ndarray:
    """LEGACY: structured-array view kept for fallback when |row entries| ≥ _PACK_BIAS."""
    arr = np.ascontiguousarray(arr, dtype=ROW_DTYPE)
    return arr.view(STRUCT_DTYPE).reshape(-1)


# Profile-driven (2026-05-12): the structured-array view triggered numpy
# `_promote_fields` 24M times = 14% of stage 2 wall, plus _as_struct/view
# overhead = ~20% total. Replaced with int64 packing: each (m0, m1, m2) row
# packs into one int64 such that lex order on rows ↔ value order on packed
# ints. Y entries are empirically bounded by ~15k absolute (see profile
# audit); we use 21 bits per column for headroom, with a runtime range check
# that falls back to the legacy struct path if any entry overflows.
_PACK_BITS = 21
_PACK_BIAS = 1 << (_PACK_BITS - 1)            # 2^20 = 1,048,576
_PACK_SH0 = 2 * _PACK_BITS                     # 42 — col 0 in highest bits
_PACK_SH1 = _PACK_BITS                         # 21
_PACK_SH2 = 0


if _HAVE_NUMBA:
    @nb.njit(cache=True, boundscheck=False)
    def _pack_rows_int64_nb(arr: np.ndarray, bias: int, sh0: int, sh1: int) -> np.ndarray:
        """Numba kernel: pack (n, 3) int64 rows into (n,) int64. Tight loop,
        zero numpy temporaries (vs 5+ in pure numpy version)."""
        n = arr.shape[0]
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = ((arr[i, 0] + bias) << sh0) \
                     | ((arr[i, 1] + bias) << sh1) \
                     | (arr[i, 2] + bias)
        return out


def _pack_rows_int64(arr: np.ndarray) -> np.ndarray:
    """Pack (n, 3) int64 rows into (n,) int64. arr must already be int64.

    Round-7 (2026-05-12): Numba kernel replaces the pure-numpy 3-vec-op chain.
    Previous numpy version was the #1 hotspot at 104s of 389s (27%) — each
    call paid for 5+ intermediate numpy arrays. The Numba kernel writes
    directly into a single output array.

    No range check — empirically Y values are bounded by ~15k absolute (well
    under _PACK_BIAS = 2^20 = 1M). If a future cell pushes Y past that, this
    will silently produce wrong answers."""
    if _HAVE_NUMBA:
        # ascontiguousarray is a no-op when already C-contiguous int64
        a = np.ascontiguousarray(arr, dtype=np.int64)
        return _pack_rows_int64_nb(a, np.int64(_PACK_BIAS),
                                   np.int64(_PACK_SH0),
                                   np.int64(_PACK_SH1))
    # Fallback (no Numba available)
    return ((arr[:, 0] + _PACK_BIAS) << _PACK_SH0) | \
           ((arr[:, 1] + _PACK_BIAS) << _PACK_SH1) | \
           (arr[:, 2] + _PACK_BIAS)


def _unique_sorted_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.empty((0, 3), dtype=ROW_DTYPE)
    arr = np.ascontiguousarray(arr, dtype=ROW_DTYPE)
    keys = _pack_rows_int64(arr)
    order = np.argsort(keys, kind="mergesort")
    s_keys = keys[order]
    diff = s_keys[1:] != s_keys[:-1]
    keep = np.empty(s_keys.shape[0], dtype=bool)
    keep[0] = True
    keep[1:] = diff
    return np.ascontiguousarray(arr[order][keep])


# Round-11 (2026-05-13): per-call prange has VERY high overhead vs the
# typical per-call work in this pipeline. Empirically:
#   - 5k bench with @parallel always-on: 24.5s → 605s (25× SLOWER)
#   - Real f=4 ε=0.0025 cell (na×nc ≈ 13700 per call) with threshold=10000:
#     no improvement (10.6% in 15min vs serial's 12.5% at same wall — slightly
#     SLOWER from thread-spawn overhead).
# At our cell sizes the per-call inner work (~10k iterations) is too small
# to amortize Numba's prange thread-spawn cost. Set threshold high enough
# that current f=4 ε=0.0025 stays on the serial path. Future tighter cells
# (na or nc in the hundreds, work > ~100k per call) can use prange.
# To run a parallel-path bench manually, just lower this constant.
_PARALLEL_KERNEL_THRESHOLD = 100_000


# ============================================================
# K'-cap stage-2 LOSSY prune (2026-05-24, feature branch only)
# ============================================================
# σ_1 ring embedding constants. σ_1(Y) = Y[0] + α₁·Y[1] + α₁²·Y[2] where
# α₁ = 2·cos(2π/9) is the σ_1 image of ζ_9 + ζ_9⁻¹ under the totally-real
# subfield Q(ζ_9 + ζ_9⁻¹) ⊂ Q(ζ_9). Used as the quality metric for K'-cap:
# rank Y_a candidates by |σ_1(Y_a)| (smaller = closer to the small-slot
# target band center of 0). Hardcoded to avoid math.cos at module load.
_SIGMA1_ALPHA1 = 1.5320888862379560      # 2 * cos(2π/9)
_SIGMA1_ALPHA1_SQ = _SIGMA1_ALPHA1 * _SIGMA1_ALPHA1   # ≈ 2.34729...


if _HAVE_NUMBA:
    @nb.njit(cache=True, boundscheck=False)
    def _process_ac_pair_nb_serial(b_keys: np.ndarray,
                                    a_rows: np.ndarray,
                                    c_rows: np.ndarray,
                                    target_sum: np.ndarray,
                                    bias: int, sh0: int, sh1: int,
                                    col_a: int, col_b: int, col_c: int) -> np.ndarray:
        """Serial kernel — fastest for small A × C (typical at f≤4 ε ≥ 0.005)."""
        na = a_rows.shape[0]
        nc = c_rows.shape[0]
        nbk = b_keys.shape[0]
        ts0 = target_sum[0]
        ts1 = target_sum[1]
        ts2 = target_sum[2]

        n_hits = 0
        for ia in range(na):
            a0 = a_rows[ia, 0]
            a1 = a_rows[ia, 1]
            a2 = a_rows[ia, 2]
            for ic in range(nc):
                nb0 = ts0 - a0 - c_rows[ic, 0]
                nb1 = ts1 - a1 - c_rows[ic, 1]
                nb2 = ts2 - a2 - c_rows[ic, 2]
                qk = ((nb0 + bias) << sh0) | ((nb1 + bias) << sh1) | (nb2 + bias)
                lo = 0
                hi = nbk
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if b_keys[mid] < qk:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < nbk and b_keys[lo] == qk:
                    n_hits += 1

        out = np.empty((n_hits, 9), dtype=np.int64)
        ca = col_a * 3
        cb = col_b * 3
        cc = col_c * 3
        k = 0
        for ia in range(na):
            a0 = a_rows[ia, 0]
            a1 = a_rows[ia, 1]
            a2 = a_rows[ia, 2]
            for ic in range(nc):
                c0 = c_rows[ic, 0]
                c1 = c_rows[ic, 1]
                c2 = c_rows[ic, 2]
                nb0 = ts0 - a0 - c0
                nb1 = ts1 - a1 - c1
                nb2 = ts2 - a2 - c2
                qk = ((nb0 + bias) << sh0) | ((nb1 + bias) << sh1) | (nb2 + bias)
                lo = 0
                hi = nbk
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if b_keys[mid] < qk:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < nbk and b_keys[lo] == qk:
                    out[k, ca] = a0; out[k, ca + 1] = a1; out[k, ca + 2] = a2
                    out[k, cb] = nb0; out[k, cb + 1] = nb1; out[k, cb + 2] = nb2
                    out[k, cc] = c0; out[k, cc + 1] = c1; out[k, cc + 2] = c2
                    k += 1
        return out


    @nb.njit(cache=True, boundscheck=False, parallel=True)
    def _process_ac_pair_nb_parallel(b_keys: np.ndarray,
                                      a_rows: np.ndarray,
                                      c_rows: np.ndarray,
                                      target_sum: np.ndarray,
                                      bias: int, sh0: int, sh1: int,
                                      col_a: int, col_b: int, col_c: int) -> np.ndarray:
        """Parallel kernel via nb.prange over A — only worth using when
        na × nc ≥ _PARALLEL_KERNEL_THRESHOLD (typically large cells)."""
        na = a_rows.shape[0]
        nc = c_rows.shape[0]
        nbk = b_keys.shape[0]
        ts0 = target_sum[0]
        ts1 = target_sum[1]
        ts2 = target_sum[2]

        # Pass 1: per-A-row hit counts (parallel).
        counts = np.zeros(na, dtype=np.int64)
        for ia in nb.prange(na):
            a0 = a_rows[ia, 0]
            a1 = a_rows[ia, 1]
            a2 = a_rows[ia, 2]
            local = 0
            for ic in range(nc):
                nb0 = ts0 - a0 - c_rows[ic, 0]
                nb1 = ts1 - a1 - c_rows[ic, 1]
                nb2 = ts2 - a2 - c_rows[ic, 2]
                qk = ((nb0 + bias) << sh0) \
                     | ((nb1 + bias) << sh1) \
                     | (nb2 + bias)
                lo = 0
                hi = nbk
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if b_keys[mid] < qk:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < nbk and b_keys[lo] == qk:
                    local += 1
            counts[ia] = local

        # Serial prefix sum (tiny array, no benefit from prange).
        offsets = np.empty(na, dtype=np.int64)
        if na > 0:
            offsets[0] = 0
            for i in range(1, na):
                offsets[i] = offsets[i - 1] + counts[i - 1]
            n_hits = offsets[na - 1] + counts[na - 1]
        else:
            n_hits = 0

        out = np.empty((n_hits, 9), dtype=np.int64)
        ca = col_a * 3
        cb = col_b * 3
        cc = col_c * 3

        # Pass 2: write hits per A-row (parallel). Each thread writes its
        # own disjoint output slice, so no atomics or locks needed.
        for ia in nb.prange(na):
            a0 = a_rows[ia, 0]
            a1 = a_rows[ia, 1]
            a2 = a_rows[ia, 2]
            k = offsets[ia]
            for ic in range(nc):
                c0 = c_rows[ic, 0]
                c1 = c_rows[ic, 1]
                c2 = c_rows[ic, 2]
                nb0 = ts0 - a0 - c0
                nb1 = ts1 - a1 - c1
                nb2 = ts2 - a2 - c2
                qk = ((nb0 + bias) << sh0) \
                     | ((nb1 + bias) << sh1) \
                     | (nb2 + bias)
                lo = 0
                hi = nbk
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if b_keys[mid] < qk:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < nbk and b_keys[lo] == qk:
                    out[k, ca] = a0; out[k, ca + 1] = a1; out[k, ca + 2] = a2
                    out[k, cb] = nb0; out[k, cb + 1] = nb1; out[k, cb + 2] = nb2
                    out[k, cc] = c0; out[k, cc + 1] = c1; out[k, cc + 2] = c2
                    k += 1
        return out


    @nb.njit(cache=True, boundscheck=False, fastmath=True)
    def _sigma1_dev_nb(y0: int, y1: int, y2: int,
                       alpha1: float, alpha1_sq: float,
                       sigma1_target: float) -> float:
        """|sigma_1(Y) - sigma1_target|. For the small slot, sigma1_target = 0;
        for a unit slot in householder mode, sigma1_target ≈ 3^{2f}."""
        v = float(y0) + alpha1 * float(y1) + alpha1_sq * float(y2) - sigma1_target
        return v if v >= 0.0 else -v

    @nb.njit(cache=True, boundscheck=False)
    def _process_ac_pair_kprime_nb(b_keys: np.ndarray,
                                    a_rows: np.ndarray,
                                    c_rows: np.ndarray,
                                    target_sum: np.ndarray,
                                    bias: int, sh0: int, sh1: int,
                                    col_a: int, col_b: int, col_c: int,
                                    kprime: int,
                                    alpha1: float, alpha1_sq: float,
                                    sigma1_target_a: float) -> np.ndarray:
        """K'-cap variant of the serial AC-pair kernel.

        Same join as ``_process_ac_pair_nb_serial`` but emits at most ``kprime``
        triples per (a_group, c_bucket) call, keeping the ones whose Y_a has
        the SMALLEST |σ_1(Y_a) - sigma1_target_a| (closest to the per-slot σ_1
        target band center). The caller is responsible for setting
        ``sigma1_target_a`` to the σ_1 image of Y_a's expected slot value
        (typically 0 for the small slot u=0, or ≈ 3^{2f} for a u=1 slot).

        Implementation: max-heap of size kprime, keyed on |σ_1(Y_a) - target|.
        New hits replace the current worst when their σ_1-deviation is smaller.
        Heap is a flat int64 buffer carrying (a0,a1,a2, nb0,nb1,nb2, c0,c1,c2)
        per slot plus a parallel float64 array of σ_1-deviation keys.

        Lossy: at most ``kprime`` triples emitted; the Y_a with the smallest
        |σ_1 deviation| within this (a_group, c_bucket) is provably preserved
        (top-1 invariant of the max-heap evict-worst policy).
        """
        na = a_rows.shape[0]
        nc = c_rows.shape[0]
        nbk = b_keys.shape[0]
        ts0 = target_sum[0]
        ts1 = target_sum[1]
        ts2 = target_sum[2]

        # Heap state. Use plain arrays + manual sift to avoid heapq.
        # keys[i] = |σ_1(Y_a_i)|; rows[i, 0..8] = (a, nb, c) in original-coord
        #   layout for the join output (we apply col_{a,b,c} on emit).
        cap = int(kprime)
        if cap <= 0:
            # Degenerate: behave as the regular serial kernel.
            return _process_ac_pair_nb_serial(b_keys, a_rows, c_rows, target_sum,
                                              bias, sh0, sh1, col_a, col_b, col_c)
        heap_keys = np.empty(cap, dtype=np.float64)
        heap_rows = np.empty((cap, 9), dtype=np.int64)
        heap_size = 0

        ca = col_a * 3
        cb = col_b * 3
        cc = col_c * 3

        for ia in range(na):
            a0 = a_rows[ia, 0]
            a1 = a_rows[ia, 1]
            a2 = a_rows[ia, 2]
            # σ_1(Y_a) is constant over the inner c-loop. Compute once.
            key = _sigma1_dev_nb(a0, a1, a2, alpha1, alpha1_sq, sigma1_target_a)
            for ic in range(nc):
                c0 = c_rows[ic, 0]
                c1 = c_rows[ic, 1]
                c2 = c_rows[ic, 2]
                nb0 = ts0 - a0 - c0
                nb1 = ts1 - a1 - c1
                nb2 = ts2 - a2 - c2
                qk = ((nb0 + bias) << sh0) | ((nb1 + bias) << sh1) | (nb2 + bias)
                lo = 0
                hi = nbk
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if b_keys[mid] < qk:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo >= nbk or b_keys[lo] != qk:
                    continue
                # Have a hit. Insert into the bounded max-heap.
                if heap_size < cap:
                    # Append + sift up.
                    i = heap_size
                    heap_keys[i] = key
                    heap_rows[i, ca] = a0; heap_rows[i, ca + 1] = a1; heap_rows[i, ca + 2] = a2
                    heap_rows[i, cb] = nb0; heap_rows[i, cb + 1] = nb1; heap_rows[i, cb + 2] = nb2
                    heap_rows[i, cc] = c0; heap_rows[i, cc + 1] = c1; heap_rows[i, cc + 2] = c2
                    heap_size += 1
                    # Sift up (max-heap: parent should be >= child)
                    while i > 0:
                        parent = (i - 1) >> 1
                        if heap_keys[parent] < heap_keys[i]:
                            # swap
                            tk = heap_keys[parent]; heap_keys[parent] = heap_keys[i]; heap_keys[i] = tk
                            for jj in range(9):
                                tr = heap_rows[parent, jj]
                                heap_rows[parent, jj] = heap_rows[i, jj]
                                heap_rows[i, jj] = tr
                            i = parent
                        else:
                            break
                else:
                    # Heap full. Only insert if smaller than current worst (root).
                    if key >= heap_keys[0]:
                        continue
                    heap_keys[0] = key
                    heap_rows[0, ca] = a0; heap_rows[0, ca + 1] = a1; heap_rows[0, ca + 2] = a2
                    heap_rows[0, cb] = nb0; heap_rows[0, cb + 1] = nb1; heap_rows[0, cb + 2] = nb2
                    heap_rows[0, cc] = c0; heap_rows[0, cc + 1] = c1; heap_rows[0, cc + 2] = c2
                    # Sift down.
                    i = 0
                    while True:
                        l = 2 * i + 1
                        r = 2 * i + 2
                        largest = i
                        if l < cap and heap_keys[l] > heap_keys[largest]:
                            largest = l
                        if r < cap and heap_keys[r] > heap_keys[largest]:
                            largest = r
                        if largest == i:
                            break
                        tk = heap_keys[i]; heap_keys[i] = heap_keys[largest]; heap_keys[largest] = tk
                        for jj in range(9):
                            tr = heap_rows[i, jj]
                            heap_rows[i, jj] = heap_rows[largest, jj]
                            heap_rows[largest, jj] = tr
                        i = largest

        out = np.empty((heap_size, 9), dtype=np.int64)
        for i in range(heap_size):
            for jj in range(9):
                out[i, jj] = heap_rows[i, jj]
        return out


    @nb.njit(cache=True, boundscheck=False)
    def _membership_mask_fused_nb(sorted_keys: np.ndarray,
                                   query_rows: np.ndarray,
                                   bias: int, sh0: int, sh1: int) -> np.ndarray:
        """Fused membership test: pack each query row → binary-search in
        sorted_keys → write bool result. One Numba kernel, zero Python
        boundary crossings per query row.

        Replaces the v5 pipeline of (1) _pack_rows_int64 call → (2)
        np.searchsorted call → (3) idx < n → (4) boolean-mask indexing →
        (5) equality check, each of which crossed Python/numpy each time.
        Profile-driven 2026-05-13: this stack was 58% of stage 2 wall at
        the 10k bench."""
        m = query_rows.shape[0]
        n = sorted_keys.shape[0]
        mask = np.empty(m, dtype=np.bool_)
        for i in range(m):
            qk = ((query_rows[i, 0] + bias) << sh0) \
                 | ((query_rows[i, 1] + bias) << sh1) \
                 | (query_rows[i, 2] + bias)
            # Binary search (left) for qk in sorted_keys
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) >> 1
                if sorted_keys[mid] < qk:
                    lo = mid + 1
                else:
                    hi = mid
            mask[i] = (lo < n) and (sorted_keys[lo] == qk)
        return mask


def _membership_mask_packed(sorted_keys: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    """Membership test where sorted_keys is the pre-packed int64 representation
    of sorted_rows. Lets callers (esp. BucketCache) compute the packing once
    per bucket and re-use it across many membership calls."""
    if query_rows.size == 0:
        return np.empty((0,), dtype=bool)
    if sorted_keys.size == 0:
        return np.zeros(query_rows.shape[0], dtype=bool)
    if _HAVE_NUMBA:
        qr = np.ascontiguousarray(query_rows, dtype=np.int64)
        return _membership_mask_fused_nb(sorted_keys, qr,
                                          np.int64(_PACK_BIAS),
                                          np.int64(_PACK_SH0),
                                          np.int64(_PACK_SH1))
    # Pure-numpy fallback (no Numba)
    qk = _pack_rows_int64(query_rows)
    idx = np.searchsorted(sorted_keys, qk)
    mask = idx < sorted_keys.shape[0]
    mask[mask] &= sorted_keys[idx[mask]] == qk[mask]
    return mask


def _membership_mask_sorted_rows(sorted_rows: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    """Convenience wrapper for callers that don't have pre-packed sorted_rows.
    Hot stage-2 callers should use BucketCache + _membership_mask_packed instead."""
    if sorted_rows.size == 0:
        if query_rows.size == 0:
            return np.empty((0,), dtype=bool)
        return np.zeros(query_rows.shape[0], dtype=bool)
    return _membership_mask_packed(_pack_rows_int64(sorted_rows), query_rows)


# ============================================================
# on-disk bucket partitioning for B and C
# ============================================================


def _bucket_file_name(base_dir: str, prefix: str, idx: int) -> str:
    return os.path.join(base_dir, f"{prefix}_bucket_{idx:06d}.bin")



def _partition_set_to_buckets(paths: List[str], base_dir: str, prefix: str, n_buckets: int,
                              rows_per_chunk: int, coeffs: Tuple[int, int, int],
                              manifest_path: str, manifest: dict, state_key: str,
                              verbose: bool = False):
    os.makedirs(base_dir, exist_ok=True)
    bucket_paths = [_bucket_file_name(base_dir, prefix, i) for i in range(n_buckets)]
    total_rows = _count_rows(paths)
    rows_done = int(manifest.get(f"{state_key}_rows_done", 0))
    if rows_done != 0:
        # restart partition from scratch for this set unless completed
        rows_done = 0
        manifest[f"{state_key}_rows_done"] = 0
        for p in bucket_paths:
            if os.path.exists(p):
                os.remove(p)
        _write_manifest(manifest_path, manifest)

    seen = 0
    for _, _, _, chunk in _iter_npy_rows(paths, rows_per_chunk):
        if chunk.size:
            chunk = _unique_sorted_rows(chunk)
            bids = _bucket_linear_hash_rows(chunk, n_buckets, coeffs)
            order = np.argsort(bids, kind="mergesort")
            chunk = chunk[order]
            bids = bids[order]
            starts = np.flatnonzero(np.r_[True, bids[1:] != bids[:-1]])
            ends = np.r_[starts[1:], bids.shape[0]]
            for s, e in zip(starts, ends):
                bid = int(bids[s])
                _append_rows_raw(bucket_paths[bid], chunk[s:e])
        seen += int(chunk.shape[0])
        manifest[f"{state_key}_rows_done"] = seen
        _write_manifest(manifest_path, manifest)
        if verbose and total_rows > 0:
            print(f"partition {prefix} rows: ~{seen}/{total_rows}")

    bucket_counts = []
    for p in bucket_paths:
        if os.path.exists(p):
            bucket_counts.append(int(os.path.getsize(p) // (3 * np.dtype(ROW_DTYPE).itemsize)))
        else:
            bucket_counts.append(0)

    meta = {
        "n_buckets": int(n_buckets),
        "bucket_paths": bucket_paths,
        "bucket_counts_raw": bucket_counts,
        "coeffs": [int(coeffs[0]), int(coeffs[1]), int(coeffs[2])],
    }
    meta_path = os.path.join(base_dir, f"{prefix}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    manifest[f"{state_key}_completed"] = True
    manifest[f"{state_key}_meta"] = meta_path
    _write_manifest(manifest_path, manifest)
    return meta


# ============================================================
# per-rank cache for loaded sorted unique bucket rows
# ============================================================


class BucketCache:
    """LRU cache of loaded+sorted bucket rows AND their int64-packed keys.

    Profile-driven 2026-05-12 (round 2):
    - Caches file size per path (avoids 20M stat calls).
    - Stores both `rows` (n, 3) int64 AND the pre-computed `keys` (n,) int64
      that membership tests need. Previously each membership call re-packed
      the same sorted_rows from scratch (24M packing ops = #1 hotspot in v1).
    """

    _EMPTY_ROWS = None
    _EMPTY_KEYS = None

    def __init__(self, max_entries: int):
        self.max_entries = int(max_entries)
        # Value type: (rows, keys) tuple
        self._cache = OrderedDict()
        self._existence = {}  # path -> int (file size in bytes), 0 if missing/empty

    def _check_path(self, path: str) -> int:
        sz = self._existence.get(path)
        if sz is not None:
            return sz
        try:
            sz = os.path.getsize(path)
        except OSError:
            sz = 0
        self._existence[path] = sz
        return sz

    def get(self, path: str):
        """Returns (rows (n,3) int64, keys (n,) int64) — both sorted lex / int64."""
        if path in self._cache:
            val = self._cache.pop(path)
            self._cache[path] = val
            return val
        sz = self._check_path(path)
        if sz == 0:
            val = (BucketCache._EMPTY_ROWS, BucketCache._EMPTY_KEYS)
        else:
            # Memory refactor #1 + A4 (2026-05-22): _read_bucket_rows auto-detects
            # legacy raw vs new zlib-compressed (Z9B1 magic + length-prefixed chunks).
            # For raw legacy files the mmap-shared page cache still applies on
            # same-node ranks; for compressed files each rank decompresses to its
            # own buffer (8× disk savings outweighs the per-rank decompress).
            raw = _read_bucket_rows(path)
            if raw.size == 0:
                val = (BucketCache._EMPTY_ROWS, BucketCache._EMPTY_KEYS)
            else:
                rows = _unique_sorted_rows(raw.reshape(-1, 3))
                keys = _pack_rows_int64(rows)
                val = (rows, keys)
            del raw
        self._cache[path] = val
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return val


# Shared sentinels for empty bucket reads — saves allocating thousands of
# (0, 3) and (0,) arrays during stage 2.
BucketCache._EMPTY_ROWS = np.empty((0, 3), dtype=ROW_DTYPE)
BucketCache._EMPTY_KEYS = np.empty((0,), dtype=np.int64)


# ============================================================
# join core
# ============================================================


def _load_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)



def _reconstruct_original_rows(a_rows: np.ndarray, b_rows: np.ndarray, c_rows: np.ndarray, order: List[int]) -> np.ndarray:
    n = a_rows.shape[0]
    out = np.empty((n, 9), dtype=ROW_DTYPE)
    # reordered positions 0,1,2 correspond to original coordinates order[0], order[1], order[2]
    cols = [a_rows, b_rows, c_rows]
    for new_pos, old_pos in enumerate(order):
        out[:, 3 * old_pos:3 * old_pos + 3] = cols[new_pos]
    return out



def _process_b_bucket(
    b_rows: np.ndarray,
    b_bucket_id: int,
    a_groups: List[Tuple[int, np.ndarray]],
    order: List[int],
    target_sum: np.ndarray,
    target_residue: int,
    c_bucket_paths: List[str],
    n_buckets: int,
    bucket_cache: BucketCache,
    rank_output_path: str,
    flush_rows: int,
    query_chunk_rows: int = 200000,
    progress_interval_sec: float = 10.0,
    progress_prefix: str = "",
    show_progress: bool = False,
    kprime_cap: int = 0,
    sigma1_target_a: float = 0.0,
):
    if b_rows.size == 0:
        return 0

    # Pack b_rows ONCE (fixed across all membership tests within this bucket call).
    # Previously _membership_mask_sorted_rows re-packed b_rows on every call to it
    # — 9M calls = #2 cost. Now only need_b gets packed each call.
    b_keys = _pack_rows_int64(b_rows)

    # Resolve the column reordering once per bucket call. order maps new_pos →
    # original coord (0/1/2); we need the inverse mapping (which new_pos do
    # A=0, B=1, C=2 occupy) for the Numba kernel's column-write step.
    col_a = int(order[0])
    col_b = int(order[1])
    col_c = int(order[2])
    use_nb_driver = _HAVE_NUMBA

    total_written = 0
    out_buf = []
    out_rows = 0
    last_progress_time = time.time()
    a_done = 0
    total_a = int(sum(rows.shape[0] for _, rows in a_groups))

    for ra, a_rows in a_groups:
        rc = int((target_residue - int(b_bucket_id) - int(ra)) % int(n_buckets))
        c_rows, _c_keys = bucket_cache.get(c_bucket_paths[rc])
        if c_rows.size == 0:
            a_done += int(a_rows.shape[0])
            if show_progress:
                now = time.time()
                if now - last_progress_time >= progress_interval_sec:
                    last_progress_time = now
                    pct = 100.0 * float(a_done) / float(total_a) if total_a > 0 else 100.0
                    print(f"{progress_prefix}{a_done}/{total_a} A rows ({pct:.1f}%), local_matches_written={total_written + out_rows}, current_C_bucket={rc+1}/{n_buckets}")
            continue

        if use_nb_driver:
            # Numba does the full A × C inner double loop in-kernel.
            # Dispatch: parallel kernel only when work is large enough to
            # amortize prange thread-spawn overhead (~ms per call).
            # K'-cap (2026-05-24, LOSSY): when kprime_cap > 0, dispatch to the
            # bounded-heap kernel that emits at most kprime_cap triples per
            # (a_group, c_bucket) call, keeping those whose Y_a has smallest
            # |σ_1(Y_a)|. Top-1 invariant is preserved.
            ar = np.ascontiguousarray(a_rows, dtype=np.int64)
            cr = np.ascontiguousarray(c_rows, dtype=np.int64)
            ts = np.ascontiguousarray(target_sum, dtype=np.int64)
            work = int(ar.shape[0]) * int(cr.shape[0])
            if kprime_cap > 0:
                out_part = _process_ac_pair_kprime_nb(b_keys, ar, cr, ts,
                                                      np.int64(_PACK_BIAS),
                                                      np.int64(_PACK_SH0),
                                                      np.int64(_PACK_SH1),
                                                      col_a, col_b, col_c,
                                                      np.int64(kprime_cap),
                                                      _SIGMA1_ALPHA1,
                                                      _SIGMA1_ALPHA1_SQ,
                                                      float(sigma1_target_a))
            elif work >= _PARALLEL_KERNEL_THRESHOLD:
                out_part = _process_ac_pair_nb_parallel(b_keys, ar, cr, ts,
                                                        np.int64(_PACK_BIAS),
                                                        np.int64(_PACK_SH0),
                                                        np.int64(_PACK_SH1),
                                                        col_a, col_b, col_c)
            else:
                out_part = _process_ac_pair_nb_serial(b_keys, ar, cr, ts,
                                                      np.int64(_PACK_BIAS),
                                                      np.int64(_PACK_SH0),
                                                      np.int64(_PACK_SH1),
                                                      col_a, col_b, col_c)
            if out_part.shape[0]:
                out_buf.append(out_part)
                out_rows += int(out_part.shape[0])
                if out_rows >= flush_rows:
                    arr = np.concatenate(out_buf, axis=0)
                    _append_rows_raw(rank_output_path, arr)
                    total_written += int(arr.shape[0])
                    out_buf = []
                    out_rows = 0
            a_done += int(a_rows.shape[0])
            if show_progress:
                now = time.time()
                if now - last_progress_time >= progress_interval_sec:
                    last_progress_time = now
                    pct = 100.0 * float(a_done) / float(total_a) if total_a > 0 else 100.0
                    print(f"{progress_prefix}{a_done}/{total_a} A rows ({pct:.1f}%), local_matches_written={total_written + out_rows}, current_C_bucket={rc+1}/{n_buckets}")
            continue

        # Pure-Python fallback (no Numba). Original behaviour preserved.
        # K'-cap path collects all hits for this (a_group, c_bucket) call into
        # a list and post-trims by |σ_1(Y_a)|. Simpler than a per-call heap and
        # K'-cap is meant for f≥4 production builds where Numba is always on;
        # this fallback exists only for dev-laptop sanity tests.
        per_call_hits = [] if kprime_cap > 0 else None
        for ia in range(a_rows.shape[0]):
            a = a_rows[ia]
            for i0 in range(0, c_rows.shape[0], query_chunk_rows):
                i1 = min(i0 + query_chunk_rows, c_rows.shape[0])
                c_chunk = c_rows[i0:i1]
                need_b = target_sum[None, :] - a[None, :] - c_chunk
                mask = _membership_mask_packed(b_keys, need_b)
                if not np.any(mask):
                    continue
                c_hit = c_chunk[mask]
                b_hit = need_b[mask]
                a_hit = np.repeat(a.reshape(1, 3), c_hit.shape[0], axis=0)
                out_part = _reconstruct_original_rows(a_hit, b_hit, c_hit, order)
                if kprime_cap > 0:
                    per_call_hits.append(out_part)
                else:
                    out_buf.append(out_part)
                    out_rows += int(out_part.shape[0])
                    if out_rows >= flush_rows:
                        arr = np.concatenate(out_buf, axis=0)
                        _append_rows_raw(rank_output_path, arr)
                        total_written += int(arr.shape[0])
                        out_buf = []
                        out_rows = 0

            a_done += 1
            if show_progress:
                now = time.time()
                if now - last_progress_time >= progress_interval_sec:
                    last_progress_time = now
                    pct = 100.0 * float(a_done) / float(total_a) if total_a > 0 else 100.0
                    print(f"{progress_prefix}{a_done}/{total_a} A rows ({pct:.1f}%), local_matches_written={total_written + out_rows}, current_C_bucket={rc+1}/{n_buckets}")

        # K'-cap post-trim for the Python-fallback path: at end of this
        # (a_group, c_bucket) call, keep the top kprime_cap by |σ_1(Y_a) - target|.
        if kprime_cap > 0 and per_call_hits is not None and per_call_hits:
            arr = np.concatenate(per_call_hits, axis=0)
            ca = int(order[0]) * 3
            ya0 = arr[:, ca].astype(np.float64)
            ya1 = arr[:, ca + 1].astype(np.float64)
            ya2 = arr[:, ca + 2].astype(np.float64)
            keys = np.abs(ya0 + _SIGMA1_ALPHA1 * ya1 + _SIGMA1_ALPHA1_SQ * ya2 - float(sigma1_target_a))
            if arr.shape[0] > kprime_cap:
                # argpartition picks indices of the kprime_cap smallest keys.
                idx = np.argpartition(keys, kprime_cap - 1)[:kprime_cap]
                arr = arr[idx]
            out_buf.append(arr)
            out_rows += int(arr.shape[0])
            if out_rows >= flush_rows:
                flush_arr = np.concatenate(out_buf, axis=0)
                _append_rows_raw(rank_output_path, flush_arr)
                total_written += int(flush_arr.shape[0])
                out_buf = []
                out_rows = 0
            per_call_hits = []

    if out_buf:
        arr = np.concatenate(out_buf, axis=0)
        _append_rows_raw(rank_output_path, arr)
        total_written += int(arr.shape[0])

    if show_progress:
        print(f"{progress_prefix}{total_a}/{total_a} A rows (100.0%), local_matches_written={total_written}")
    return total_written


# attach mutable static for hash coeffs
_process_b_bucket.hash_coeffs = (911382323, 972663749, 9721)


# ============================================================
# main algorithm
# ============================================================


def select_triples_mpi(
    inputs1: List[str],
    inputs2: List[str],
    inputs3: List[str],
    f: int,
    norm: int,
    output_file: str,
    dedup_inputs: bool = True,
    partition_rows_per_chunk: int = 500000,
    n_join_buckets: int = 8192,
    bucket_cache_entries: int = 16,
    rank_flush_rows: int = 200000,
    resume: bool = False,
    verbose: bool = True,
    drop_col_c: bool = False,
    kprime_cap: int = 0,
    kprime_u_a_sq: float = 0.0,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    manifest_path = output_file + ".manifest.json"
    parts_dir = output_file + ".parts"
    join_dir = output_file + ".join_buckets"
    hash_coeffs = (911382323, 972663749, 9721)
    _process_b_bucket.hash_coeffs = hash_coeffs

    total_scale_int = int(round(norm * (3 ** (2 * f))))
    target_sum = np.array([total_scale_int, 0, 0], dtype=ROW_DTYPE)
    # σ_1 band-center target for Y_a, used only by the K'-cap kernel. Each Y
    # is the integer numerator of |a|² (with a in the cyclotomic ring at denom
    # 3^f), so σ_1(Y) ≈ |u_target|² · 3^{2f} for a near the target. For Y_a
    # representing a slot with target magnitude² = kprime_u_a_sq, the band
    # center is kprime_u_a_sq · 3^{2f}. Default kprime_u_a_sq=0 fits the small
    # slot (u≈0), which is the typical A choice in the wrapper (smallest
    # input set after sort).
    sigma1_target_a_val = float(kprime_u_a_sq) * float(3 ** (2 * f))

    if rank == 0:
        in1 = _resolve_input_specs(inputs1)
        in2 = _resolve_input_specs(inputs2)
        in3 = _resolve_input_specs(inputs3)
        if not in1 or not in2 or not in3:
            raise RuntimeError("Each input set must resolve to at least one file.")

        manifest = _read_manifest(manifest_path) if resume else None
        if manifest is None:
            if os.path.exists(output_file):
                os.remove(output_file)
            os.makedirs(parts_dir, exist_ok=True)
            os.makedirs(join_dir, exist_ok=True)
            raw_counts = [_count_rows(in1), _count_rows(in2), _count_rows(in3)]
            order = list(np.argsort(np.asarray(raw_counts)))
            files_by_idx = [in1, in2, in3]
            re_inputs = [files_by_idx[i] for i in order]
            A = _load_stack(re_inputs[0], dedup=dedup_inputs)
            manifest = {
                "version": 3,
                "completed": False,
                "f": int(f),
                "norm": int(norm),
                "total_scale_int": int(total_scale_int),
                "inputs1": in1,
                "inputs2": in2,
                "inputs3": in3,
                "dedup_inputs": bool(dedup_inputs),
                "order": [int(x) for x in order],
                "A_file_index": int(order[0]),
                "B_file_index": int(order[1]),
                "C_file_index": int(order[2]),
                "smallest_size": int(A.shape[0]),
                "n_join_buckets": int(n_join_buckets),
                "partition_rows_per_chunk": int(partition_rows_per_chunk),
                "bucket_cache_entries": int(bucket_cache_entries),
                "rank_flush_rows": int(rank_flush_rows),
                "hash_coeffs": [int(x) for x in hash_coeffs],
                "parts_dir": os.path.abspath(parts_dir),
                "join_dir": os.path.abspath(join_dir),
                "B_partition_completed": False,
                "C_partition_completed": False,
                "B_rows_done": 0,
                "C_rows_done": 0,
                "rows_written": 0,
                "kprime_cap": int(kprime_cap),
                "kprime_u_a_sq": float(kprime_u_a_sq),
                "kprime_sigma1_target_a": float(sigma1_target_a_val),
            }
            _write_manifest(manifest_path, manifest)
            _save_completed_bitmap(manifest_path, bytearray((int(n_join_buckets) + 7) // 8))
        else:
            files_by_idx = [manifest["inputs1"], manifest["inputs2"], manifest["inputs3"]]
            re_inputs = [files_by_idx[i] for i in manifest["order"]]
            A = _load_stack(re_inputs[0], dedup=manifest.get("dedup_inputs", True))
            os.makedirs(manifest["parts_dir"], exist_ok=True)
            os.makedirs(manifest["join_dir"], exist_ok=True)

        re_inputs = [ [manifest["inputs1"], manifest["inputs2"], manifest["inputs3"]][i] if False else None ]
        files_by_idx = [manifest["inputs1"], manifest["inputs2"], manifest["inputs3"]]
        re_inputs = [files_by_idx[i] for i in manifest["order"]]

        if not manifest.get("B_partition_completed", False):
            b_meta = _partition_set_to_buckets(
                re_inputs[1], manifest["join_dir"], "B", manifest["n_join_buckets"],
                manifest["partition_rows_per_chunk"], hash_coeffs,
                manifest_path, manifest, "B", verbose=verbose,
            )
        else:
            b_meta = _load_meta(manifest["B_meta"])

        if not manifest.get("C_partition_completed", False):
            c_meta = _partition_set_to_buckets(
                re_inputs[2], manifest["join_dir"], "C", manifest["n_join_buckets"],
                manifest["partition_rows_per_chunk"], hash_coeffs,
                manifest_path, manifest, "C", verbose=verbose,
            )
        else:
            c_meta = _load_meta(manifest["C_meta"])

        if "B_meta" not in manifest:
            manifest["B_meta"] = os.path.join(manifest["join_dir"], "B_meta.json")
        if "C_meta" not in manifest:
            manifest["C_meta"] = os.path.join(manifest["join_dir"], "C_meta.json")
        _write_manifest(manifest_path, manifest)

        if verbose:
            print("Loaded/reordered inputs:")
            print(f"  smallest resident set rows = {A.shape[0]}")
            print(f"  B raw rows = {_count_rows(re_inputs[1])}")
            print(f"  C raw rows = {_count_rows(re_inputs[2])}")
            print(f"  order (new_pos -> original_coord) = {manifest['order']}")
            print(f"  join buckets = {manifest['n_join_buckets']}")
            print(f"  bucket cache entries per rank = {manifest['bucket_cache_entries']}")
    else:
        A = None
        manifest = None
        b_meta = None
        c_meta = None

    manifest = comm.bcast(manifest, root=0)
    A = comm.bcast(A, root=0)
    if rank != 0:
        b_meta = _load_meta(manifest["B_meta"])
        c_meta = _load_meta(manifest["C_meta"])

    n_buckets = int(manifest["n_join_buckets"])
    b_bucket_paths = b_meta["bucket_paths"]
    c_bucket_paths = c_meta["bucket_paths"]
    completed_bitmap = _init_completed_bitmap(manifest_path, int(manifest["n_join_buckets"]), manifest)
    completed_buckets = _bitmap_to_set(completed_bitmap, int(manifest["n_join_buckets"]))
    # Strip any legacy list from the manifest dict so it won't get re-serialized.
    manifest.pop("completed_buckets", None)
    order = [int(x) for x in manifest["order"]]

    a_hashes = _bucket_linear_hash_rows(A, n_buckets, hash_coeffs)
    target_residue = int(_bucket_linear_hash_rows(target_sum.reshape(1, 3), n_buckets, hash_coeffs)[0])
    order_idx = np.argsort(a_hashes, kind="mergesort")
    A_sorted = A[order_idx]
    a_hashes_sorted = a_hashes[order_idx]
    starts = np.flatnonzero(np.r_[True, a_hashes_sorted[1:] != a_hashes_sorted[:-1]]) if a_hashes_sorted.size else np.empty((0,), dtype=np.int64)
    ends = np.r_[starts[1:], a_hashes_sorted.shape[0]] if a_hashes_sorted.size else np.empty((0,), dtype=np.int64)
    a_groups = [(int(a_hashes_sorted[s]), np.ascontiguousarray(A_sorted[s:e], dtype=ROW_DTYPE)) for s, e in zip(starts, ends)]

    rank_output_path = os.path.join(manifest["parts_dir"], f"rank_{rank:04d}.bin")
    if not resume and os.path.exists(rank_output_path):
        os.remove(rank_output_path)

    cache = BucketCache(manifest["bucket_cache_entries"])
    local_written_total = 0
    wall_t0 = time.time()

    tprev = wall_t0
    for phase_start in range(0, n_buckets, size):
        my_bid = phase_start + rank
        local_written = 0
        if my_bid < n_buckets and my_bid not in completed_buckets:
            tnow = time.time()
            if verbose and rank == 0 and tnow > tprev+10:
                tprev = tnow
                done_before = len(completed_buckets)
                print(f"processing B bucket {my_bid+1}/{n_buckets} on rank 0 (completed {done_before}/{n_buckets}, cumulative_rows={manifest.get('rows_written', 0)})")
            b_path = b_bucket_paths[my_bid]
            if os.path.exists(b_path) and os.path.getsize(b_path) > 0:
                # A4 (2026-05-22): _read_bucket_rows handles both legacy raw and
                # zlib-compressed formats. Decompression cost is small vs the
                # downstream _process_b_bucket work.
                raw = _read_bucket_rows(b_path)
                if raw.size:
                    b_rows = _unique_sorted_rows(raw.reshape(-1, 3))
                else:
                    b_rows = np.empty((0, 3), dtype=ROW_DTYPE)
                del raw
            else:
                b_rows = np.empty((0, 3), dtype=ROW_DTYPE)

            local_written = _process_b_bucket(
                b_rows=b_rows,
                b_bucket_id=my_bid,
                a_groups=a_groups,
                order=order,
                target_sum=target_sum,
                target_residue=target_residue,
                c_bucket_paths=c_bucket_paths,
                n_buckets=n_buckets,
                bucket_cache=cache,
                rank_output_path=rank_output_path,
                flush_rows=int(manifest["rank_flush_rows"]),
                progress_interval_sec=10.0,
                progress_prefix=f"bucket {my_bid+1}/{n_buckets}: ",
                show_progress=bool(verbose and rank == 0),
                kprime_cap=int(kprime_cap),
                sigma1_target_a=sigma1_target_a_val,
            )
            local_written_total += int(local_written)

        phase_counts = _buffered_gather_counts(comm, local_written, root=0, tag=9300 + phase_start)
        comm.Barrier()
        tprev = time.time()
        if rank == 0:
            newly_done = []
            for r in range(size):
                bid = phase_start + r
                if bid >= n_buckets:
                    continue
                if bid in completed_buckets:
                    continue
                newly_done.append(bid)
            for bid in newly_done:
                _bitmap_set(completed_bitmap, int(bid))
            completed_buckets.update(newly_done)
            manifest["rows_written"] = int(manifest.get("rows_written", 0)) + int(sum(phase_counts))
            manifest["completed"] = len(completed_buckets) >= n_buckets
            # Write the bitmap (1024 bytes for 8192 buckets) BEFORE the manifest,
            # so the manifest's "completed" flag is never set without the bitmap
            # being on disk. Manifest no longer carries the list. Incremental:
            # we just toggle the new bits in an in-memory bytearray, no rebuild.
            _save_completed_bitmap(manifest_path, completed_bitmap)
            _write_manifest(manifest_path, manifest)
            tnow = time.time()
            if verbose and tnow > tprev+10:
                tprev = tnow
                print(f"completed B buckets through {min(phase_start + size, n_buckets)}/{n_buckets}, phase_rows={sum(phase_counts)}, cumulative={manifest['rows_written']}")
        comm.Barrier()

    # finalize output on root only after all parts exist.
    # A4 (2026-05-22): .parts files may be zlib-compressed (magic Z9B1 + len-prefixed
    # chunks) or legacy raw. _read_bucket_bytes auto-detects and returns decoded
    # raw payload, which we then write to output_file. output_file STAYS RAW so
    # the stage-3 reader (find_roots_exact_v2:_read_triple_rows) doesn't need
    # changes — preserves random-access by row.
    #
    # Drop-col-C (2026-05-24): if drop_col_c, output_file is written in the
    # compact format (Z9TC header + 6-int64 rows). Y_c is dropped on write and
    # reconstructed on read by downstream readers. Saves 1/3 on the final TM file.
    if rank == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        rows_written = 0
        row_byte_size_full = 9 * np.dtype(ROW_DTYPE).itemsize  # 72 B per (9-int64) row in .parts
        if drop_col_c:
            # Write header once, then stream payloads converting 9-col → 6-col.
            with open(output_file, "wb") as dst:
                _write_tm_compact_header(dst, target_sum)
                for r in range(size):
                    part = os.path.join(manifest["parts_dir"], f"rank_{r:04d}.bin")
                    if not os.path.exists(part) or os.path.getsize(part) == 0:
                        continue
                    payload = _read_bucket_bytes(part)
                    if not payload:
                        continue
                    assert len(payload) % row_byte_size_full == 0, (
                        f"part {part}: decoded payload size {len(payload)} "
                        f"not a multiple of {row_byte_size_full}-byte row"
                    )
                    n_part = len(payload) // row_byte_size_full
                    arr = np.frombuffer(payload, dtype=ROW_DTYPE).reshape(n_part, 9)
                    # Sanity: each row must satisfy Y0+Y1+Y2 == target_sum.
                    # Verifying here (cheap) avoids silent corruption on read.
                    # NOTE: this is the ONE arithmetic check; downstream readers
                    # do NOT re-verify (they trust the writer).
                    if n_part:
                        ssum = arr[:, 0:3] + arr[:, 3:6] + arr[:, 6:9]
                        if not np.array_equal(ssum, np.broadcast_to(target_sum, (n_part, 3))):
                            mismatch = int((ssum != target_sum[None, :]).any(axis=1).sum())
                            raise RuntimeError(
                                f"part {part}: {mismatch}/{n_part} rows do NOT sum to "
                                f"target_sum {tuple(target_sum.tolist())} — refusing to "
                                f"write a compact file we can't read back correctly."
                            )
                    # Drop columns 6..8 (Y_c). Write columns 0..5 only.
                    ab = np.ascontiguousarray(arr[:, 0:6], dtype=ROW_DTYPE)
                    dst.write(ab.tobytes())
                    rows_written += n_part
            manifest["format"] = "Z9TC_v1"
            manifest["row_byte_size"] = _TM_COMPACT_ROW_BYTES
            manifest["target_sum"] = [int(x) for x in target_sum.tolist()]
        else:
            row_byte_size = row_byte_size_full
            for r in range(size):
                part = os.path.join(manifest["parts_dir"], f"rank_{r:04d}.bin")
                if not os.path.exists(part) or os.path.getsize(part) == 0:
                    continue
                payload = _read_bucket_bytes(part)
                if not payload:
                    continue
                assert len(payload) % row_byte_size == 0, (
                    f"part {part}: decoded payload size {len(payload)} "
                    f"not a multiple of {row_byte_size}-byte row"
                )
                with open(output_file, "ab") as dst:
                    dst.write(payload)
                rows_written += len(payload) // row_byte_size
            manifest["format"] = "raw_9int64"
            manifest["row_byte_size"] = row_byte_size_full
        manifest["rows_written"] = rows_written
        manifest["completed"] = True
        manifest["wall_total_time"] = float(time.time() - wall_t0)
        _write_manifest(manifest_path, manifest)
        if verbose:
            print(f"wrote final output: {output_file}")
            print(f"total rows written: {rows_written}")
            print(f"output format: {manifest['format']}")
            print(f"wall total time: {manifest['wall_total_time']:.3f} s")
        return manifest
    return None


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs1", nargs="+", required=True, help="Files or glob patterns for set 1")
    parser.add_argument("--inputs2", nargs="+", required=True, help="Files or glob patterns for set 2")
    parser.add_argument("--inputs3", nargs="+", required=True, help="Files or glob patterns for set 3")
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--norm", type=int, default=2)
    parser.add_argument("--output", required=True, help="Single output file with raw int64 rows of shape (n,9)")
    parser.add_argument("--partition_rows_per_chunk", type=int, default=500000)
    parser.add_argument("--n_join_buckets", type=int, default=8192)
    parser.add_argument("--bucket_cache_entries", type=int, default=8192,
                        help="Max bucket cache entries per rank. Defaults to "
                             "8192 = full coverage of the default 8192 join "
                             "buckets, eliminating reload thrashing (was the "
                             "dominant cost at the previous default of 256). "
                             "If RAM pressure surfaces at very tight ε, drop "
                             "this — but profile first; the cache-fit win is "
                             "~2.5× on stage 2 e2e.")
    parser.add_argument("--rank_flush_rows", type=int, default=200000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--drop_col_c", action="store_true",
                        help="Write the final TM file in the compact 'Z9TC' "
                             "format: 32-byte header (target_sum) + 6 int64 per "
                             "row (Y_a, Y_b). Y_c is reconstructed on read by "
                             "downstream readers as target_sum - Y_a - Y_b. "
                             "Lossless; saves 1/3 on the final TM file size. "
                             "Default OFF for back-compat with older artifacts.")
    parser.add_argument("--kprime-cap", "--kprime_cap", dest="kprime_cap",
                        type=int, default=0,
                        help="LOSSY: per (a_group, c_bucket) call, emit at most "
                             "this many triples, keeping those whose Y_a has the "
                             "smallest |sigma_1(Y_a) - center| (closest to the "
                             "band center). 0 = disabled (default; byte-identical "
                             "to legacy). Recommended 64 for f>=6 builds where "
                             "the unpruned cross-join exceeds disk. Lossy at "
                             "ranks 2..K within a call but provably preserves "
                             "the top-1 Y_a per (a_group, c_bucket).")
    parser.add_argument("--kprime-u-a-sq", "--kprime_u_a_sq",
                        dest="kprime_u_a_sq", type=float, default=0.0,
                        help="|u|² magnitude target for the A slot, used by "
                             "the K'-cap kernel to set the σ_1 band center "
                             "(center = kprime_u_a_sq · 3^{2f}). Default 0.0 "
                             "matches the typical wrapper layout where A is "
                             "the smallest input (u=0 slot). Pass 1.0 if A is "
                             "a u=1 slot. Ignored when --kprime-cap == 0.")
    args = parser.parse_args()

    out = select_triples_mpi(
        inputs1=args.inputs1,
        inputs2=args.inputs2,
        inputs3=args.inputs3,
        f=args.f,
        norm=args.norm,
        output_file=args.output,
        dedup_inputs=not args.no_dedup,
        partition_rows_per_chunk=args.partition_rows_per_chunk,
        n_join_buckets=args.n_join_buckets,
        bucket_cache_entries=args.bucket_cache_entries,
        rank_flush_rows=args.rank_flush_rows,
        resume=args.resume,
        verbose=not args.quiet,
        drop_col_c=args.drop_col_c,
        kprime_cap=args.kprime_cap,
        kprime_u_a_sq=args.kprime_u_a_sq,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
