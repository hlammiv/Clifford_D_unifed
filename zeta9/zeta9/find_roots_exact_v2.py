"""Build an rootdb database for all Y values appearing in a triples file.

This phase is independent of target vectors and eps. It can optionally reuse and
grow a long-lived global rootdb cache via --global_rootdb_prefix.
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Tuple, OrderedDict

import numpy as np
from mpi4py import MPI

from .roots import actual_roots_from_ideal_search


# ---------------------------------------------------------------------------
# Lazy-enumeration pre-filter (audit zeta9_stage3_optimization_audit.md)
# Skip Y's whose θ-agnostic best_frob lower bound exceeds --eps_target.
# Mirrors stage 5's per-Y σ-deviation filter: a Y could be a triple's "large"
# slot iff |sqrt(σ_F_1(Y))/3^f - target_abs_large| ≤ ε, similarly for "small".
# Y passes if it could fill SOME slot. Verified bound (audit table 2026-05-14):
#   ε=1e-4 → 3.2% survival, 31× wall reduction.
# ---------------------------------------------------------------------------
_ALPHA = 2.0 * math.cos(2.0 * math.pi / 9.0)  # α = ζ_9 + ζ_9⁻¹ ≈ 1.532


def _sigma_F_1(m_0, m_1, m_2):
    return float(m_0) + _ALPHA * float(m_1) + _ALPHA * _ALPHA * float(m_2)


def _best_frob_pass(Y, f, eps_target, u_values=(1.0, 1.0, 0.0)):
    """Return True if Y could plausibly contribute to a triple with Frob ≤ ε.

    For each candidate slot target u_i, check if Y's |r|=sqrt(σ_F_1(Y))/3^f
    is within ε of |target_abs|=sqrt(u_i) (mirrors stage 5 line 306). If Y
    matches ANY slot, pass; else skip.

    f is the zeta9-internal V-denominator exponent (= 2 × wrapper max-f).
    """
    s1 = _sigma_F_1(*Y)
    if s1 < 0:
        return False
    scale = 3.0 ** f
    r = math.sqrt(s1) / scale
    for u in u_values:
        target_abs = math.sqrt(max(0.0, u))
        if abs(r - target_abs) <= eps_target + 1.0e-15:
            return True
    return False


def _read_manifest(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _fmt_seconds(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def Y_weight(Y):
    m0, m1, m2 = Y
    return max(1, 2 * m0 + 4 * m2)


def weighted_partition(items, weights, n_parts):
    order = np.argsort([-w for w in weights])
    buckets = [[] for _ in range(n_parts)]
    bucket_w = [0] * n_parts
    for idx in order:
        j = min(range(n_parts), key=lambda k: bucket_w[k])
        buckets[j].append(items[idx])
        bucket_w[j] += weights[idx]
    return buckets, bucket_w


def _split_root_info_batches(root_info, max_items=64, max_roots=2000):
    items = list(root_info.items())
    batches = []
    cur = {}
    cur_items = 0
    cur_roots = 0
    for Y, info in items:
        nroots = len(info.get("roots_all", ()))
        if cur and (cur_items >= max_items or cur_roots + nroots > max_roots):
            batches.append(cur)
            cur = {}
            cur_items = 0
            cur_roots = 0
        cur[Y] = info
        cur_items += 1
        cur_roots += nroots
    if cur:
        batches.append(cur)
    return batches


def gather_root_info_batched(comm, local_root_info, root=0, max_items=64, max_roots=2000):
    rank = comm.Get_rank()
    local_batches = _split_root_info_batches(local_root_info, max_items=max_items, max_roots=max_roots)
    local_n = len(local_batches)
    max_n = comm.allreduce(local_n, op=MPI.MAX)
    merged = {} if rank == root else None
    for i in range(max_n):
        payload = local_batches[i] if i < local_n else {}
        gathered = comm.gather(payload, root=root)
        if rank == root:
            for dct in gathered:
                merged.update(dct)
    return merged


def _read_triple_rows(path: str, start_row: int, nrows: int) -> np.ndarray:
    if nrows <= 0:
        return np.empty((0, 9), dtype=np.int64)
    itemsize = np.dtype(np.int64).itemsize
    offset = start_row * 9 * itemsize
    with open(path, "rb") as fh:
        fh.seek(offset, os.SEEK_SET)
        arr = np.fromfile(fh, dtype=np.int64, count=nrows * 9)
    return arr.reshape((-1, 9))


def _state_paths(prefix: str):
    return {
        "unique_y": prefix + ".unique_y.npy",
        "solved_mask": prefix + ".solved_mask.npy",
        "exact_roots_dir": prefix + ".exact_roots_batches",
        "exact_roots_index_y": prefix + ".exact_roots_index_y.npy",
        "exact_roots_index_file": prefix + ".exact_roots_index_file.npy",
        "exact_roots_index_row": prefix + ".exact_roots_index_row.npy",
        "exact_roots_index_meta": prefix + ".exact_roots_index_meta.json",
    }


def _invalidate_exact_roots_index(paths: dict):
    for key in ("exact_roots_index_y", "exact_roots_index_file", "exact_roots_index_row", "exact_roots_index_meta"):
        path = paths[key]
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _sort_unique_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    arr = np.ascontiguousarray(arr[order], dtype=np.int64)
    if arr.shape[0] <= 1:
        return arr
    keep = np.ones(arr.shape[0], dtype=bool)
    keep[1:] = np.any(arr[1:] != arr[:-1], axis=1)
    return np.ascontiguousarray(arr[keep], dtype=np.int64)


def _merge_two_sorted_unique_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.ascontiguousarray(b, dtype=np.int64)
    if b.size == 0:
        return np.ascontiguousarray(a, dtype=np.int64)
    c = np.concatenate([np.ascontiguousarray(a, dtype=np.int64), np.ascontiguousarray(b, dtype=np.int64)], axis=0)
    return _sort_unique_rows(c)


def _hierarchical_merge_chunk_files(chunk_files, chunk_dir: str, verbose=False):
    files = [str(p) for p in chunk_files]
    round_idx = 0
    while len(files) > 1:
        next_files = []
        for pair_idx in range(0, len(files), 2):
            if pair_idx + 1 >= len(files):
                next_files.append(files[pair_idx])
                continue
            f1 = files[pair_idx]
            f2 = files[pair_idx + 1]
            a = np.load(f1).astype(np.int64, copy=False)
            b = np.load(f2).astype(np.int64, copy=False)
            merged = _merge_two_sorted_unique_arrays(a, b)
            out_path = os.path.join(chunk_dir, f"merge_r{round_idx:03d}_{pair_idx//2:06d}.npy")
            np.save(out_path, merged)
            next_files.append(os.path.abspath(out_path))
            try:
                os.remove(f1)
            except OSError:
                pass
            try:
                os.remove(f2)
            except OSError:
                pass
        files = next_files
        round_idx += 1
        if verbose:
            print(f"hierarchical merge round {round_idx}: {len(files)} file(s) remain")
    if not files:
        return np.empty((0, 3), dtype=np.int64)
    return np.load(files[0]).astype(np.int64, copy=False)


def _discover_unique_y_numpy(triples_file, nrows, discover_chunk_rows, unique_y_path, solved_mask_path, checkpoint_json, ckpt, verbose=False):
    chunk_dir = unique_y_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_files = list(ckpt.get("discover_chunk_files", []))
    start = int(ckpt.get("discover_next_row", 0))
    chunk_idx = int(ckpt.get("discover_chunk_index", 0))
    while start < nrows:
        take = min(int(discover_chunk_rows), nrows - start)
        triples = _read_triple_rows(triples_file, start, take).reshape((-1, 3, 3))
        ys = np.ascontiguousarray(triples.reshape((-1, 3)), dtype=np.int64)
        if ys.size:
            ys = _sort_unique_rows(ys)
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:06d}.npy")
        np.save(chunk_path, ys)
        chunk_files.append(os.path.abspath(chunk_path))
        chunk_idx += 1
        start += take
        ckpt["discover_next_row"] = start
        ckpt["discover_chunk_index"] = chunk_idx
        ckpt["discover_chunk_files"] = chunk_files
        _write_manifest(checkpoint_json, ckpt)
        if verbose:
            print(f"discover triples scanned: {start:,}/{nrows:,} | chunk uniques={ys.shape[0]:,}")

    all_y = _hierarchical_merge_chunk_files(chunk_files, chunk_dir, verbose=verbose)
    all_y = _sort_unique_rows(all_y)
    np.save(unique_y_path, all_y)
    np.save(solved_mask_path, np.zeros(all_y.shape[0], dtype=np.bool_))
    for path in list(chunk_files):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    try:
        if os.path.isdir(chunk_dir):
            leftovers = [name for name in os.listdir(chunk_dir) if name.endswith('.npy')]
            for name in leftovers:
                try:
                    os.remove(os.path.join(chunk_dir, name))
                except OSError:
                    pass
            if not os.listdir(chunk_dir):
                os.rmdir(chunk_dir)
    except OSError:
        pass
    ckpt["unique_y_count"] = int(all_y.shape[0])
    ckpt["solved_y_count"] = int(ckpt.get("solved_y_count", 0))
    ckpt["remaining_unsolved_count"] = int(all_y.shape[0]) - int(ckpt.get("solved_y_count", 0))
    ckpt["discover_chunk_files"] = []
    ckpt["discover_completed"] = True
    _write_manifest(checkpoint_json, ckpt)


def _select_unsolved_batch(unique_y_path, solved_mask_path, solve_batch_size, scan_start):
    Y = np.load(unique_y_path, mmap_mode='r')
    mask = np.load(solved_mask_path, mmap_mode='r')
    n = Y.shape[0]
    batch = []
    idxs = []
    i = max(0, int(scan_start))
    while i < n and len(batch) < solve_batch_size:
        if not bool(mask[i]):
            y = Y[i]
            batch.append((int(y[0]), int(y[1]), int(y[2])))
            idxs.append(i)
        i += 1
    return batch, idxs, i, n


def _mark_solved_batch(solved_mask_path, idxs):
    mask = np.load(solved_mask_path, mmap_mode='r+')
    if idxs:
        mask[np.asarray(idxs, dtype=np.int64)] = True
        mask.flush()


def _pack_roots_flat(root_lists):
    counts = [len(rs) for rs in root_lists]
    off = np.zeros(len(root_lists) + 1, dtype=np.int64)
    if counts:
        off[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    total = int(off[-1])
    flat = np.zeros((total, 6), dtype=np.int64)
    pos = 0
    for rs in root_lists:
        n = len(rs)
        if n:
            flat[pos:pos+n, :] = np.asarray(rs, dtype=np.int64)
            pos += n
    return flat, off


def _unpack_roots_flat(flat: np.ndarray, off: np.ndarray, i: int):
    a = int(off[i])
    b = int(off[i + 1])
    if b <= a:
        return tuple()
    arr = np.asarray(flat[a:b], dtype=np.int64)
    return tuple(tuple(int(x) for x in row) for row in arr)


def _append_exact_roots_batch(roots_dir: str, batch_idx: int, root_info: Dict[Tuple[int, int, int], dict]):
    os.makedirs(roots_dir, exist_ok=True)
    Ys = []
    roots_lists = []
    for Y, info in root_info.items():
        Ys.append(tuple(int(x) for x in Y))
        roots_lists.append(tuple(info.get("roots_all", ())))
    roots_flat, roots_off = _pack_roots_flat(roots_lists)
    out_path = os.path.join(roots_dir, f"roots_batch_{batch_idx:06d}.npz")
    np.savez(out_path, Y=np.asarray(Ys, dtype=np.int64), roots_flat=roots_flat, roots_off=roots_off)
    return out_path


def _structured_view_y(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr.view(dtype=np.dtype([("y0", "<i8"), ("y1", "<i8"), ("y2", "<i8")])).reshape(-1)


def _ensure_exact_roots_index(paths: dict, checkpoint_json: str, ckpt: dict, verbose=False):
    y_path = paths["exact_roots_index_y"]
    f_path = paths["exact_roots_index_file"]
    r_path = paths["exact_roots_index_row"]
    meta_path = paths["exact_roots_index_meta"]
    roots_dir = paths["exact_roots_dir"]
    if os.path.exists(meta_path) and os.path.exists(y_path) and os.path.exists(f_path) and os.path.exists(r_path):
        ckpt["exact_roots_index_built"] = True
        _write_manifest(checkpoint_json, ckpt)
        return
    files = sorted(
        os.path.join(roots_dir, name)
        for name in os.listdir(roots_dir)
        if name.startswith("roots_batch_") and name.endswith(".npz")
    ) if os.path.isdir(roots_dir) else []
    file_ids = []
    row_ids = []
    y_chunks = []
    n_files = len(files)
    for i, path in enumerate(files, 1):
        data = np.load(path, allow_pickle=True)
        Y = np.asarray(data["Y"], dtype=np.int64)
        n = Y.shape[0]
        y_chunks.append(Y)
        file_ids.append(np.full(n, i - 1, dtype=np.int32))
        row_ids.append(np.arange(n, dtype=np.int32))
        if verbose and (i == n_files or i % max(1, n_files // 100) == 0):
            print(f"build rootdbs index: {i}/{n_files}")
    all_y = np.concatenate(y_chunks, axis=0) if y_chunks else np.empty((0, 3), dtype=np.int64)
    all_f = np.concatenate(file_ids, axis=0) if file_ids else np.empty((0,), dtype=np.int32)
    all_r = np.concatenate(row_ids, axis=0) if row_ids else np.empty((0,), dtype=np.int32)
    if all_y.size:
        order = np.lexsort((all_y[:, 2], all_y[:, 1], all_y[:, 0]))
        all_y = np.ascontiguousarray(all_y[order], dtype=np.int64)
        all_f = np.ascontiguousarray(all_f[order], dtype=np.int32)
        all_r = np.ascontiguousarray(all_r[order], dtype=np.int32)
    np.save(y_path, all_y)
    np.save(f_path, all_f)
    np.save(r_path, all_r)
    _write_manifest(meta_path, {"files": files})
    ckpt["exact_roots_index_built"] = True
    _write_manifest(checkpoint_json, ckpt)


def _load_full_roots_rows_for_needed(needed_y: np.ndarray, full_paths: dict, verbose=False, file_cache_size=8):
    if needed_y.size == 0:
        return {}
    meta_path = full_paths["exact_roots_index_meta"]
    if not os.path.exists(meta_path):
        return {}
    index_y = np.load(full_paths["exact_roots_index_y"], mmap_mode='r')
    index_file = np.load(full_paths["exact_roots_index_file"], mmap_mode='r')
    index_row = np.load(full_paths["exact_roots_index_row"], mmap_mode='r')
    meta = _read_manifest(meta_path)
    files = meta["files"]

    key_index = _structured_view_y(index_y)
    key_need = _structured_view_y(np.unique(np.ascontiguousarray(needed_y, dtype=np.int64), axis=0))
    pos = np.searchsorted(key_index, key_need)

    requests = {}
    out = {}
    for ky, p in zip(key_need, pos):
        if p >= key_index.shape[0] or key_index[p] != ky:
            continue
        y = (int(ky["y0"]), int(ky["y1"]), int(ky["y2"]))
        fi = int(index_file[p])
        ri = int(index_row[p])
        requests.setdefault(fi, []).append((ri, y))

    cache = OrderedDict()
    total_files = len(requests)
    done = 0
    last_report = time.time()
    for fi, rows in requests.items():
        path = files[fi]
        if fi in cache:
            data = cache.pop(fi)
            cache[fi] = data
        else:
            data = np.load(path, allow_pickle=True, mmap_mode='r')
            cache[fi] = data
            while len(cache) > file_cache_size:
                cache.popitem(last=False)

        flat = data["roots_flat"]
        off = data["roots_off"]
        for ri, y in rows:
            roots = _unpack_roots_flat(flat, off, ri)
            out[y] = {
                "roots_all": roots,
                "nroots_all": len(roots),
                "screen": "GLOBAL_ROOTDB",
                "status": "GLOBAL_ROOTDB",
                "used_legacy": False,
                "B2": None,
                "B": None,
                "diagnostics": {},
            }

        done += 1
        now = time.time()
        if verbose and total_files > 0 and (done == total_files or now - last_report >= 10.0):
            last_report = now
            print(f"load global rootdb files: {done}/{total_files}")
    return out


def _preload_solved_from_full_cache(unique_y_path: str, solved_mask_path: str, full_paths: dict, checkpoint_json: str, ckpt: dict, verbose=False):
    meta_path = full_paths["exact_roots_index_meta"]
    if not os.path.exists(meta_path):
        return 0, {}
    unique_y = np.load(unique_y_path, mmap_mode='r')
    index_y = np.load(full_paths["exact_roots_index_y"], mmap_mode='r')
    if unique_y.shape[0] == 0 or index_y.shape[0] == 0:
        return 0, {}
    key_unique = _structured_view_y(unique_y)
    key_index = _structured_view_y(index_y)
    pos = np.searchsorted(key_index, key_unique)
    valid = pos < key_index.shape[0]
    mask_existing = np.zeros_like(valid, dtype=bool)
    mask_existing[valid] = (key_index[pos[valid]] == key_unique[valid])
    solved_mask = np.load(solved_mask_path, mmap_mode='r+')
    old = solved_mask.copy()
    solved_mask[mask_existing] = True
    solved_mask.flush()
    added = int(np.count_nonzero(mask_existing & ~old))

    cached_root_info = {}
    if added:
        cached_y = np.ascontiguousarray(unique_y[mask_existing], dtype=np.int64)
        cached_root_info = _load_full_roots_rows_for_needed(cached_y, full_paths, verbose=verbose)
        ckpt["solved_y_count"] = int(ckpt.get("solved_y_count", 0)) + added
        ckpt["remaining_unsolved_count"] = max(0, int(ckpt.get("remaining_unsolved_count", 0)) - added)
        _write_manifest(checkpoint_json, ckpt)
        if verbose:
            print(f"preloaded solved Ys from global rootdb: {added}")
    return added, cached_root_info


def solve_assigned_Ys_exact(Y_list, rank, verbose=False, use_legacy_fallback=True,
                              eps_target=None, lazy_f=None, lazy_u_values=None):
    """Solve each Y in Y_list, returning {Y: root_info dict}.

    Lazy-enum pre-filter: if `eps_target` is set, Y's failing
    _best_frob_pass(Y, lazy_f, eps_target, lazy_u_values) are SKIPPED — they
    don't appear in the returned root_info, so stage 5's locator.get() will
    return None and the triple is dropped naturally. Per audit, 31× stage-3
    wall reduction at ε=1e-4."""
    root_info = {}
    total = len(Y_list)
    batch_start_time = time.time()
    last_report_time = time.time()
    n_skipped = 0
    use_filter = (eps_target is not None and eps_target > 0
                  and lazy_f is not None)
    for i, Y in enumerate(Y_list, 1):
        if use_filter and not _best_frob_pass(Y, lazy_f, eps_target,
                                              lazy_u_values or (1.0, 1.0, 0.0)):
            n_skipped += 1
            continue
        res = actual_roots_from_ideal_search(
            Y,
            orbit=True,
            check_real_embeddings=True,
            verbose=verbose,
            rank=rank,
            use_legacy_fallback=use_legacy_fallback,
        )
        roots = tuple(tuple(int(x) for x in r) for r in res.get("actual_roots", []))
        root_info[Y] = {
            "roots_all": roots,
            "nroots_all": len(roots),
            "screen": res.get("screen"),
            "status": res.get("status"),
            "used_legacy": bool(res.get("used_legacy", False)),
            "B2": res.get("B2"),
            "B": res.get("B"),
            "diagnostics": res.get("diagnostics", {}),
        }
        now = time.time()
        if verbose and rank == 0 and now - last_report_time >= 10.0:
            last_report_time = now
            elapsed = now - batch_start_time
            rate = (i / elapsed) if elapsed > 0 else 0.0
            eta = ((total - i) / rate) if rate > 0 else float("inf")
            print(f"rank {rank} rootdb progress: {i}/{total} ({100.0 * i / total:.1f}%) | rate={rate:.2f} Y/s | ETA={_fmt_seconds(eta)}")
    if use_filter and rank == 0:
        solved = total - n_skipped
        print(f"[lazy-enum] rank {rank}: skipped {n_skipped}/{total} ({100.0*n_skipped/max(1,total):.1f}%) by best_frob bound — solved {solved}", flush=True)
    return root_info


def find_roots_exact_pipeline(
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    checkpoint_json: str,
    discover_chunk_rows: int = 200000,
    solve_batch_size: int = 200000,
    use_legacy_fallback: bool = True,
    verbose: bool = False,
    root_info_batch_items: int = 64,
    root_info_batch_roots: int = 2000,
    resume: bool = False,
    global_rootdb_prefix: str = None,
    eps_target: float = None,
    lazy_f: int = None,
    lazy_u_values: tuple = None,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    paths = _state_paths(rootdb_prefix)
    unique_y_path = paths["unique_y"]
    solved_mask_path = paths["solved_mask"]
    roots_dir = paths["exact_roots_dir"]
    global_cache_paths = _state_paths(global_rootdb_prefix) if global_rootdb_prefix is not None else None

    if rank == 0:
        tmeta = _read_manifest(triples_json)
        nrows = int(tmeta["rows_written"])
        file_size = os.path.getsize(triples_file)
        expected = nrows * 9 * np.dtype(np.int64).itemsize
        if file_size != expected:
            raise RuntimeError(f"Triple file size mismatch: expected {expected}, got {file_size}")
        if resume and os.path.exists(checkpoint_json):
            ckpt = _read_manifest(checkpoint_json)
        else:
            ckpt = {
                "version": 1,
                "triples_file": os.path.abspath(triples_file),
                "triples_json": os.path.abspath(triples_json),
                "rootdb_prefix": os.path.abspath(rootdb_prefix),
                "discover_completed": False,
                "discover_next_row": 0,
                "discover_chunk_index": 0,
                "discover_chunk_files": [],
                "solve_completed": False,
                "solve_scan_start": 0,
                "completed_root_batches": 0,
                "roots_batch_file_index": 0,
                "solved_y_count": 0,
                "remaining_unsolved_count": 0,
                "exact_roots_index_built": False,
            }
            _write_manifest(checkpoint_json, ckpt)
    else:
        nrows = None
        ckpt = None

    nrows = comm.bcast(nrows, root=0)
    ckpt = comm.bcast(ckpt, root=0)

    if rank == 0 and not ckpt["discover_completed"]:
        _discover_unique_y_numpy(triples_file, nrows, discover_chunk_rows, unique_y_path, solved_mask_path, checkpoint_json, ckpt, verbose=verbose)
    comm.Barrier()

    if rank == 0:
        ckpt = _read_manifest(checkpoint_json)
        if "remaining_unsolved_count" not in ckpt and os.path.exists(unique_y_path):
            if "unique_y_count" not in ckpt:
                ckpt["unique_y_count"] = int(np.load(unique_y_path, mmap_mode='r').shape[0])
            ckpt["remaining_unsolved_count"] = int(ckpt["unique_y_count"]) - int(ckpt.get("solved_y_count", 0))
            _write_manifest(checkpoint_json, ckpt)
        if global_cache_paths is not None and os.path.exists(global_cache_paths["exact_roots_dir"]):
            if not os.path.exists(global_cache_paths["exact_roots_index_meta"]):
                _ensure_exact_roots_index(global_cache_paths, checkpoint_json, ckpt, verbose=verbose)
                ckpt = _read_manifest(checkpoint_json)
            _, cached_root_info = _preload_solved_from_full_cache(unique_y_path, solved_mask_path, global_cache_paths, checkpoint_json, ckpt, verbose=verbose)
            ckpt = _read_manifest(checkpoint_json)
            if cached_root_info:
                os.makedirs(paths["exact_roots_dir"], exist_ok=True)
                _append_exact_roots_batch(paths["exact_roots_dir"], int(ckpt.get("roots_batch_file_index", 0)), cached_root_info)
                ckpt["roots_batch_file_index"] = int(ckpt.get("roots_batch_file_index", 0)) + 1
                ckpt["exact_roots_index_built"] = False
                _write_manifest(checkpoint_json, ckpt)
                ckpt = _read_manifest(checkpoint_json)
    ckpt = comm.bcast(ckpt, root=0)

    phase_start = time.time()
    completed_root_batches = int(ckpt.get("completed_root_batches", 0))
    while True:
        if rank == 0:
            remaining_unsolved_before = int(ckpt.get("remaining_unsolved_count", 0))
            batch, idxs, next_scan, _ = _select_unsolved_batch(unique_y_path, solved_mask_path, int(solve_batch_size), int(ckpt.get("solve_scan_start", 0)))
            remaining_batches = int(math.ceil(remaining_unsolved_before / float(solve_batch_size))) if remaining_unsolved_before > 0 else 0
            current_batch_idx = completed_root_batches + 1 if batch else completed_root_batches
            mean_batch_seconds = ((time.time() - phase_start) / completed_root_batches) if completed_root_batches > 0 else None
            if verbose and batch:
                msg = f"rootdb batch {current_batch_idx}/{remaining_batches} | size: {len(batch)} | unsolved remaining={remaining_unsolved_before}"
                if mean_batch_seconds is not None:
                    msg += f" | avg completed batch={_fmt_seconds(mean_batch_seconds)} | elapsed={_fmt_seconds(time.time() - phase_start)}"
                print(msg)
            if not batch:
                buckets = [[] for _ in range(comm.Get_size())]
                batch_payload = {"idxs": [], "next_scan": next_scan, "selected_count": 0, "remaining_unsolved_before": remaining_unsolved_before}
            else:
                weights = [Y_weight(Y) for Y in batch]
                buckets, _ = weighted_partition(batch, weights, comm.Get_size())
                batch_payload = {"idxs": idxs, "next_scan": next_scan, "selected_count": len(idxs), "remaining_unsolved_before": remaining_unsolved_before}
            batch_meta = {"batch_payload": batch_payload}
        else:
            buckets = None
            batch_meta = None
        Ys_local = comm.scatter(buckets, root=0)
        batch_meta = comm.bcast(batch_meta, root=0)
        if int(batch_meta["batch_payload"].get("selected_count", 0)) == 0:
            break
        batch_wall0 = time.time()
        local_root_info = solve_assigned_Ys_exact(
            Ys_local, rank, verbose=verbose, use_legacy_fallback=use_legacy_fallback,
            eps_target=eps_target, lazy_f=lazy_f, lazy_u_values=lazy_u_values)
        merged = gather_root_info_batched(comm, local_root_info, root=0, max_items=root_info_batch_items, max_roots=root_info_batch_roots)
        if rank == 0:
            _mark_solved_batch(solved_mask_path, batch_meta["batch_payload"]["idxs"])
            roots_path = _append_exact_roots_batch(roots_dir, int(ckpt.get("roots_batch_file_index", 0)), merged)
            batch_file_idx = int(ckpt.get("roots_batch_file_index", 0))
            ckpt["roots_batch_file_index"] = batch_file_idx + 1
            if global_cache_paths is not None:
                os.makedirs(global_cache_paths["exact_roots_dir"], exist_ok=True)
                _append_exact_roots_batch(global_cache_paths["exact_roots_dir"], batch_file_idx, merged)
                _invalidate_exact_roots_index(global_cache_paths)
            completed_root_batches += 1
            ckpt["completed_root_batches"] = completed_root_batches
            ckpt["solve_scan_start"] = int(batch_meta["batch_payload"]["next_scan"])
            batch_seconds = time.time() - batch_wall0
            selected_count = int(batch_meta["batch_payload"].get("selected_count", len(batch_meta["batch_payload"].get("idxs", []))))
            ckpt["solved_y_count"] = int(ckpt.get("solved_y_count", 0)) + selected_count
            ckpt["remaining_unsolved_count"] = max(0, int(batch_meta["batch_payload"].get("remaining_unsolved_before", 0)) - selected_count)
            ckpt["exact_roots_index_built"] = False
            if verbose:
                elapsed = time.time() - phase_start
                avg_batch = elapsed / completed_root_batches if completed_root_batches > 0 else batch_seconds
                remaining_unsolved2 = int(ckpt.get("remaining_unsolved_count", 0))
                remaining_batches2 = int(math.ceil(remaining_unsolved2 / float(solve_batch_size))) if remaining_unsolved2 > 0 else 0
                print(
                    f"rootdb batch {completed_root_batches} done | batch time={_fmt_seconds(batch_seconds)} | "
                    f"avg batch={_fmt_seconds(avg_batch)} | solved={ckpt['solved_y_count']} | "
                    f"remaining batches={remaining_batches2} | ETA={_fmt_seconds(remaining_batches2 * avg_batch)} | saved roots={roots_path}"
                )
            _write_manifest(checkpoint_json, ckpt)
        comm.Barrier()

    if rank == 0:
        ckpt = _read_manifest(checkpoint_json)
        ckpt["solve_completed"] = True
        _write_manifest(checkpoint_json, ckpt)
        if not ckpt.get("exact_roots_index_built", False):
            _ensure_exact_roots_index(paths, checkpoint_json, ckpt, verbose=verbose)
            ckpt = _read_manifest(checkpoint_json)
        return ckpt
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--triples_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--checkpoint_json", default=None)
    parser.add_argument("--discover_chunk_rows", type=int, default=200000)
    parser.add_argument("--solve_batch_size", type=int, default=200000)
    parser.add_argument("--no_legacy_fallback", action="store_true")
    parser.add_argument("--root_info_batch_items", type=int, default=64)
    parser.add_argument("--root_info_batch_roots", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--global_rootdb_prefix", default=None)
    # Lazy-enum pre-filter (audit 2026-05-14)
    parser.add_argument("--eps_target", type=float, default=None,
                        help="Frobenius tolerance for lazy-enum filter. If set "
                             "(>0), skip Y's whose best_frob bound exceeds this. "
                             "Per audit: 31× wall reduction at 1e-4, 130× at 5e-5.")
    parser.add_argument("--lazy_f", type=int, default=None,
                        help="zeta9-internal V-denominator exponent (= 2 × wrapper "
                             "max-f). Required if --eps_target is set.")
    parser.add_argument("--lazy_u_values", type=str, default="1.0,1.0,0.0",
                        help="Comma-separated cell u-targets per slot. "
                             "Default Householder (1.0,1.0,0.0). Used by lazy filter.")
    args = parser.parse_args()
    _lazy_u_values = tuple(float(x) for x in args.lazy_u_values.split(",")) if args.lazy_u_values else None

    checkpoint_json = args.checkpoint_json or (args.rootdb_prefix + ".roots.json")
    out = find_roots_exact_pipeline(
        triples_file=args.triples_file,
        triples_json=args.triples_json,
        rootdb_prefix=args.rootdb_prefix,
        checkpoint_json=checkpoint_json,
        discover_chunk_rows=args.discover_chunk_rows,
        solve_batch_size=args.solve_batch_size,
        use_legacy_fallback=not args.no_legacy_fallback,
        verbose=not args.quiet,
        root_info_batch_items=args.root_info_batch_items,
        root_info_batch_roots=args.root_info_batch_roots,
        resume=args.resume,
        global_rootdb_prefix=args.global_rootdb_prefix,
        eps_target=args.eps_target,
        lazy_f=args.lazy_f,
        lazy_u_values=_lazy_u_values,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
