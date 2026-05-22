import argparse
import glob
import json
import math
import os
import time
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

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
    with open(path, "ab") as fh:
        arr.tofile(fh)


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
            # Memory refactor #1 (2026-05-22): use np.memmap instead of
            # np.fromfile. Same-node ranks then share these read-only pages via
            # the Linux page cache rather than each allocating its own copy.
            # _unique_sorted_rows() materializes a fresh sorted array (no longer
            # tied to the mmap), so the cached entry is regular RAM.
            try:
                raw = np.memmap(path, dtype=ROW_DTYPE, mode='r')
            except (ValueError, OSError):
                # Empty or vanished file → treat as empty
                raw = np.empty((0,), dtype=ROW_DTYPE)
            if raw.size == 0:
                val = (BucketCache._EMPTY_ROWS, BucketCache._EMPTY_KEYS)
            else:
                rows = _unique_sorted_rows(raw.reshape(-1, 3))
                keys = _pack_rows_int64(rows)
                val = (rows, keys)
            # Drop the mmap reference; the materialized rows/keys arrays own
            # their own memory now.
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
            ar = np.ascontiguousarray(a_rows, dtype=np.int64)
            cr = np.ascontiguousarray(c_rows, dtype=np.int64)
            ts = np.ascontiguousarray(target_sum, dtype=np.int64)
            work = int(ar.shape[0]) * int(cr.shape[0])
            if work >= _PARALLEL_KERNEL_THRESHOLD:
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
                # Memory refactor #1: mmap so simultaneous loads across ranks
                # on the same node share via page cache instead of each allocating.
                try:
                    raw = np.memmap(b_path, dtype=ROW_DTYPE, mode='r')
                except (ValueError, OSError):
                    raw = np.empty((0,), dtype=ROW_DTYPE)
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

    # finalize output on root only after all parts exist
    if rank == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        rows_written = 0
        for r in range(size):
            part = os.path.join(manifest["parts_dir"], f"rank_{r:04d}.bin")
            if not os.path.exists(part) or os.path.getsize(part) == 0:
                continue
            with open(part, "rb") as src, open(output_file, "ab") as dst:
                while True:
                    chunk = src.read(8 * 9 * 100000)
                    if not chunk:
                        break
                    dst.write(chunk)
            rows_written += int(os.path.getsize(part) // (9 * np.dtype(ROW_DTYPE).itemsize))
        manifest["rows_written"] = rows_written
        manifest["completed"] = True
        manifest["wall_total_time"] = float(time.time() - wall_t0)
        _write_manifest(manifest_path, manifest)
        if verbose:
            print(f"wrote final output: {output_file}")
            print(f"total rows written: {rows_written}")
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
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
