
import argparse
import json
import math
import os
import pickle
import time
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI

try:
    from .tools import embed
except ImportError:
    from tools import embed

from .roots import actual_roots_from_ideal_search

C1 = math.cos(2.0 * math.pi / 9.0)
C2 = math.cos(4.0 * math.pi / 9.0)
C3 = -0.5
C4 = math.cos(8.0 * math.pi / 9.0)
C5 = math.cos(10.0 * math.pi / 9.0)
S1 = math.sin(2.0 * math.pi / 9.0)
S2 = math.sin(4.0 * math.pi / 9.0)
S3 = math.sqrt(3.0) / 2.0
S4 = math.sin(8.0 * math.pi / 9.0)
S5 = math.sin(10.0 * math.pi / 9.0)
ALPHA1 = 2.0 * C1
ALPHA1_SQ = ALPHA1 * ALPHA1


def sigma1_from_m012(m0: int, m1: int, m2: int) -> float:
    return float(m0) + ALPHA1 * float(m1) + ALPHA1_SQ * float(m2)


def passes_target_halfspace(n, Tx: float, Ty: float, sigma1_M: float, R: float) -> bool:
    n0, n1, n2, n3, n4, n5 = n
    lhs = (
        Tx * n0
        + (C1 * Tx + S1 * Ty) * n1
        + (C2 * Tx + S2 * Ty) * n2
        + (C3 * Tx + S3 * Ty) * n3
        + (C4 * Tx + S4 * Ty) * n4
        + (C5 * Tx + S5 * Ty) * n5
    )
    rhs = 0.5 * (sigma1_M + (Tx * Tx + Ty * Ty) - R * R)
    return lhs > rhs


def target_vector_first_case(t: float, norm: int = 2) -> np.ndarray:
    d = np.zeros(3, dtype=np.complex128)
    scale = math.sqrt(norm / 2.0)
    d[0] = scale * np.exp(-0.5j * t)
    d[1] = -scale
    d[2] = 0.0
    return d


def coeffs_to_complex(x_coeffs, f):
    return embed(tuple(int(v) for v in x_coeffs), 1) / (3 ** f)


def vector_coeffs_to_complex(vec_coeffs, f):
    return np.array([coeffs_to_complex(v, f) for v in vec_coeffs], dtype=np.complex128)


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


def _open_phase2_log(rank, enabled=False, log_dir="phase2_logs"):
    if not enabled:
        return None, None
    log_dir = os.path.join(os.getcwd(), log_dir)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"find_roots_rank{rank}.log")
    return open(path, "a", buffering=1), path


def _fmt_seconds(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _diag_reason(diag):
    if isinstance(diag, dict):
        return diag.get("failure_reason") or diag.get("reason")
    return None


def _split_root_info_batches(root_info, max_items=64, max_roots=2000):
    items = list(root_info.items())
    batches = []
    cur = {}
    cur_items = 0
    cur_roots = 0
    for Y, info in items:
        nroots = len(info.get("roots0", ())) + len(info.get("roots1", ())) + len(info.get("roots2", ()))
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


def _read_manifest(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _state_paths(prefix: str):
    return {
        "unique_y": prefix + ".unique_y.npy",
        "solved_mask": prefix + ".solved_mask.npy",
        "roots_dir": prefix + ".roots_batches",
        "roots_index_y": prefix + ".roots_index_y.npy",
        "roots_index_file": prefix + ".roots_index_file.npy",
        "roots_index_row": prefix + ".roots_index_row.npy",
        "roots_index_meta": prefix + ".roots_index_meta.json",
        "full_roots_dir": prefix + ".full_roots_batches",
        "full_roots_index_y": prefix + ".full_roots_index_y.npy",
        "full_roots_index_file": prefix + ".full_roots_index_file.npy",
        "full_roots_index_row": prefix + ".full_roots_index_row.npy",
        "full_roots_index_meta": prefix + ".full_roots_index_meta.json",
        "full_present_mask": prefix + ".full_present_mask.npy",
    }


def _discover_unique_y_numpy(triples_file, nrows, discover_chunk_rows, unique_y_path, solved_mask_path, checkpoint_json, ckpt, verbose=False):
    chunk_files = list(ckpt.get("discover_chunk_files", []))
    start = int(ckpt.get("discover_next_row", 0))
    chunk_idx = int(ckpt.get("discover_chunk_index", 0))
    while start < nrows:
        take = min(int(discover_chunk_rows), nrows - start)
        triples = _read_triple_rows(triples_file, start, take).reshape((-1, 3, 3))
        ys = np.ascontiguousarray(triples.reshape((-1, 3)), dtype=np.int64)
        if ys.size:
            ys = np.unique(ys, axis=0)
        chunk_path = f"{unique_y_path}.chunk_{chunk_idx:06d}.npy"
        np.save(chunk_path, ys)
        chunk_files.append(os.path.abspath(chunk_path))
        chunk_idx += 1
        start += take
        ckpt["discover_next_row"] = start
        ckpt["discover_chunk_index"] = chunk_idx
        ckpt["discover_chunk_files"] = chunk_files
        _write_manifest(checkpoint_json, ckpt)
        if verbose:
            print(f"discover triples scanned: {start}/{nrows} | chunk uniques={ys.shape[0]}")
    arrays = [np.load(path).astype(np.int64, copy=False) for path in chunk_files]
    all_y = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 3), dtype=np.int64)
    if all_y.size:
        all_y = np.unique(all_y, axis=0)
        order = np.lexsort((all_y[:, 2], all_y[:, 1], all_y[:, 0]))
        all_y = np.ascontiguousarray(all_y[order], dtype=np.int64)
    np.save(unique_y_path, all_y)
    np.save(solved_mask_path, np.zeros(all_y.shape[0], dtype=np.bool_))
    for path in chunk_files:
        try:
            os.remove(path)
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


def filter_roots_for_coord(roots, Y, coord, f, d, eps):
    if not roots:
        return tuple()
    Tx = (3 ** f) * float(np.real(d[coord]))
    Ty = (3 ** f) * float(np.imag(d[coord]))
    R = (3 ** f) * float(eps)
    sigma1_M = sigma1_from_m012(*Y)
    return tuple(r for r in roots if passes_target_halfspace(r, Tx, Ty, sigma1_M, R))




def _pack_roots_flat(root_lists):
    """
    Pack a list of per-Y root tuples into:
      flat: int64 array of shape (N, 6)
      off : int64 array of shape (len(root_lists)+1,)
    """
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



def _passes_target_halfspace_mask(arr: np.ndarray, coeffs: np.ndarray, rhs: float) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0,), dtype=bool)
    arrf = np.asarray(arr, dtype=np.float64)
    lhs = arrf @ coeffs
    return lhs > rhs


def _coord_halfspace_params(f: int, d: np.ndarray, eps: float):
    scale = float(3 ** f)
    R = scale * float(eps)
    coeffs = []
    rhs0 = []
    for coord in range(3):
        Tx = scale * float(np.real(d[coord]))
        Ty = scale * float(np.imag(d[coord]))
        coeffs.append(np.array([
            Tx,
            C1 * Tx + S1 * Ty,
            C2 * Tx + S2 * Ty,
            C3 * Tx + S3 * Ty,
            C4 * Tx + S4 * Ty,
            C5 * Tx + S5 * Ty,
        ], dtype=np.float64))
        rhs0.append(0.5 * ((Tx * Tx + Ty * Ty) - R * R))
    return coeffs, rhs0


def _filter_roots_array_for_all_coords(arr: np.ndarray, sigma1_M: float, coeffs_by_coord, rhs0_by_coord):
    if arr.size == 0:
        empty = np.empty((0, 6), dtype=np.int64)
        return empty, empty, empty
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    out = []
    for coord in range(3):
        rhs = rhs0_by_coord[coord] + 0.5 * sigma1_M
        mask = _passes_target_halfspace_mask(arr, coeffs_by_coord[coord], rhs)
        if np.any(mask):
            out.append(np.ascontiguousarray(arr[mask], dtype=np.int64))
        else:
            out.append(np.empty((0, 6), dtype=np.int64))
    return out[0], out[1], out[2]


def _build_full_roots_locator(needed_y: np.ndarray, paths: dict):
    if needed_y.size == 0:
        return {}
    meta_path = paths["full_roots_index_meta"]
    if not os.path.exists(meta_path):
        return {}
    index_y = np.load(paths["full_roots_index_y"], mmap_mode='r')
    index_file = np.load(paths["full_roots_index_file"], mmap_mode='r')
    index_row = np.load(paths["full_roots_index_row"], mmap_mode='r')
    uniq_needed = np.unique(np.ascontiguousarray(needed_y, dtype=np.int64), axis=0)
    key_index = _structured_view_y(index_y)
    key_need = _structured_view_y(uniq_needed)
    pos = np.searchsorted(key_index, key_need)
    locator = {}
    for ky, p in zip(key_need, pos):
        if p >= key_index.shape[0] or key_index[p] != ky:
            continue
        y = (int(ky["y0"]), int(ky["y1"]), int(ky["y2"]))
        locator[y] = (int(index_file[p]), int(index_row[p]))
    return locator


def _get_filtered_roots_from_full_cache(y, locator: dict, paths: dict, coeffs_by_coord, rhs0_by_coord, file_cache: OrderedDict, filtered_cache: OrderedDict, file_cache_size=8, filtered_cache_size=4096):
    cached = filtered_cache.get(y)
    if cached is not None:
        filtered_cache.move_to_end(y)
        return cached
    loc = locator.get(y)
    if loc is None:
        return None
    meta = file_cache.get('_meta')
    if meta is None:
        meta = _read_manifest(paths['full_roots_index_meta'])
        file_cache['_meta'] = meta
    files = meta['files']
    fi, ri = loc
    data = file_cache.get(fi)
    if data is None:
        data = np.load(files[fi], allow_pickle=True, mmap_mode='r')
        file_cache[fi] = data
        while len([k for k in file_cache.keys() if k != '_meta']) > file_cache_size:
            old_key = next(k for k in file_cache.keys() if k != '_meta')
            old = file_cache.pop(old_key)
            try:
                old.close()
            except Exception:
                pass
    else:
        file_cache.move_to_end(fi)

    sigma1_M = sigma1_from_m012(*y)
    if all(k in data for k in ('roots_flat', 'roots_off')):
        off = data['roots_off']
        a = int(off[ri])
        b = int(off[ri + 1])
        arr = np.ascontiguousarray(data['roots_flat'][a:b], dtype=np.int64) if b > a else np.empty((0, 6), dtype=np.int64)
        r0, r1, r2 = _filter_roots_array_for_all_coords(arr, sigma1_M, coeffs_by_coord, rhs0_by_coord)
    elif 'roots_blob' in data:
        roots = tuple(tuple(int(x) for x in r) for r in pickle.loads(bytes(data['roots_blob'][ri])))
        r0 = np.asarray(filter_roots_for_coord(roots, y, 0, 0, np.zeros(3, dtype=np.complex128), 0.0), dtype=np.int64)
        r1 = np.asarray(filter_roots_for_coord(roots, y, 1, 0, np.zeros(3, dtype=np.complex128), 0.0), dtype=np.int64)
        r2 = np.asarray(filter_roots_for_coord(roots, y, 2, 0, np.zeros(3, dtype=np.complex128), 0.0), dtype=np.int64)
    else:
        r0 = np.empty((0, 6), dtype=np.int64)
        r1 = np.empty((0, 6), dtype=np.int64)
        r2 = np.empty((0, 6), dtype=np.int64)
    out = {'roots0': r0, 'roots1': r1, 'roots2': r2}
    filtered_cache[y] = out
    while len(filtered_cache) > filtered_cache_size:
        filtered_cache.popitem(last=False)
    return out


def _load_full_roots_rows_for_needed_filtered(needed_y: np.ndarray, paths: dict, f: int, d: np.ndarray, eps: float, verbose=False, file_cache_size=8):
    locator = _build_full_roots_locator(needed_y, paths)
    coeffs_by_coord, rhs0_by_coord = _coord_halfspace_params(f, d, eps)
    file_cache = OrderedDict()
    filtered_cache = OrderedDict()
    out = {}
    total = len(locator)
    done = 0
    last_report = time.time()
    for y in locator.keys():
        info = _get_filtered_roots_from_full_cache(y, locator, paths, coeffs_by_coord, rhs0_by_coord, file_cache, filtered_cache, file_cache_size=file_cache_size, filtered_cache_size=max(4096, total + 1))
        if info is not None:
            out[y] = info
        done += 1
        now = time.time()
        if verbose and total > 0 and (done == total or now - last_report >= 10.0):
            last_report = now
            print(f"load/filter full-root Ys: {done}/{total}")
    return out


def _append_full_roots_batch(roots_dir: str, batch_idx: int, root_info: Dict[Tuple[int, int, int], dict]):
    os.makedirs(roots_dir, exist_ok=True)
    Ys = []
    roots_lists = []
    for Y, info in root_info.items():
        Ys.append(tuple(int(x) for x in Y))
        roots_lists.append(tuple(info.get("roots_all", ())))
    roots_flat, roots_off = _pack_roots_flat(roots_lists)
    out_path = os.path.join(roots_dir, f"roots_batch_{batch_idx:06d}.npz")
    np.savez(
        out_path,
        Y=np.asarray(Ys, dtype=np.int64),
        roots_flat=roots_flat,
        roots_off=roots_off,
    )
    return out_path


def _ensure_generic_roots_index(roots_dir: str, y_path: str, f_path: str, r_path: str, meta_path: str, checkpoint_json: str, ckpt: dict, verbose=False, ckpt_flag_key=None):
    if os.path.exists(meta_path) and os.path.exists(y_path) and os.path.exists(f_path) and os.path.exists(r_path):
        if ckpt_flag_key is not None:
            ckpt[ckpt_flag_key] = True
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
            print(f"build roots index: {i}/{n_files}")
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
    if ckpt_flag_key is not None:
        ckpt[ckpt_flag_key] = True
        _write_manifest(checkpoint_json, ckpt)


def _ensure_full_roots_index(paths: dict, checkpoint_json: str, ckpt: dict, verbose=False):
    _ensure_generic_roots_index(
        paths["full_roots_dir"],
        paths["full_roots_index_y"],
        paths["full_roots_index_file"],
        paths["full_roots_index_row"],
        paths["full_roots_index_meta"],
        checkpoint_json,
        ckpt,
        verbose=verbose,
        ckpt_flag_key="full_roots_index_built",
    )


def _load_full_roots_rows_for_needed(needed_y: np.ndarray, paths: dict, verbose=False, file_cache_size=8):
    if needed_y.size == 0:
        return {}
    meta_path = paths["full_roots_index_meta"]
    if not os.path.exists(meta_path):
        return {}
    index_y = np.load(paths["full_roots_index_y"], mmap_mode='r')
    index_file = np.load(paths["full_roots_index_file"], mmap_mode='r')
    index_row = np.load(paths["full_roots_index_row"], mmap_mode='r')
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

        has_flat = all(k in data for k in ("roots_flat", "roots_off"))
        for ri, y in rows:
            if has_flat:
                roots = _unpack_roots_flat(data["roots_flat"], data["roots_off"], ri)
            else:
                if "roots_blob" in data:
                    roots = tuple(tuple(int(x) for x in r) for r in pickle.loads(bytes(data["roots_blob"][ri])))
                elif "roots0_blob" in data:
                    roots = tuple(tuple(int(x) for x in r) for r in pickle.loads(bytes(data["roots0_blob"][ri])))
                else:
                    roots = tuple()
            out[y] = roots
        done += 1
        now = time.time()
        if verbose and total_files > 0 and (done == total_files or now - last_report >= 10.0):
            last_report = now
            print(f"load full-root files: {done}/{total_files}")
    return out


def _preload_solved_from_full_cache(unique_y_path: str, solved_mask_path: str, paths: dict, checkpoint_json: str, ckpt: dict, verbose=False):
    if not os.path.exists(paths["full_roots_index_meta"]):
        return 0
    unique_y = np.load(unique_y_path, mmap_mode='r')
    index_y = np.load(paths["full_roots_index_y"], mmap_mode='r')
    if unique_y.shape[0] == 0 or index_y.shape[0] == 0:
        return 0
    key_unique = _structured_view_y(unique_y)
    key_index = _structured_view_y(index_y)
    pos = np.searchsorted(key_index, key_unique)
    mask_existing = (pos < key_index.shape[0]) & (key_index[pos] == key_unique)
    solved_mask = np.load(solved_mask_path, mmap_mode='r+')
    old = solved_mask.copy()
    solved_mask[mask_existing] = True
    solved_mask.flush()
    added = int(np.count_nonzero(mask_existing & ~old))
    if added:
        ckpt["solved_y_count"] = int(ckpt.get("solved_y_count", 0)) + added
        ckpt["remaining_unsolved_count"] = max(0, int(ckpt.get("remaining_unsolved_count", 0)) - added)
        ckpt["preloaded_from_full_cache"] = int(ckpt.get("preloaded_from_full_cache", 0)) + added
        _write_manifest(checkpoint_json, ckpt)
        if verbose:
            print(f"preloaded solved Ys from full cache: {added}")
    return added



def _build_full_present_mask(unique_y_path: str, paths: dict, out_mask_path: str, checkpoint_json: str, ckpt: dict, verbose=False):
    if os.path.exists(out_mask_path):
        ckpt["full_present_mask_built"] = True
        _write_manifest(checkpoint_json, ckpt)
        return
    if not os.path.exists(paths["full_roots_index_meta"]):
        uniq = np.load(unique_y_path, mmap_mode='r')
        mask = np.zeros(uniq.shape[0], dtype=np.bool_)
        np.save(out_mask_path, mask)
        ckpt["full_present_mask_built"] = True
        _write_manifest(checkpoint_json, ckpt)
        return
    unique_y = np.load(unique_y_path, mmap_mode='r')
    index_y = np.load(paths["full_roots_index_y"], mmap_mode='r')
    if unique_y.shape[0] == 0 or index_y.shape[0] == 0:
        mask = np.zeros(unique_y.shape[0], dtype=np.bool_)
        np.save(out_mask_path, mask)
        ckpt["full_present_mask_built"] = True
        _write_manifest(checkpoint_json, ckpt)
        return
    key_unique = _structured_view_y(unique_y)
    key_index = _structured_view_y(index_y)
    pos = np.searchsorted(key_index, key_unique)
    mask_existing = (pos < key_index.shape[0]) & (key_index[pos] == key_unique)
    np.save(out_mask_path, np.asarray(mask_existing, dtype=np.bool_))
    ckpt["full_present_mask_built"] = True
    ckpt["full_present_count"] = int(np.count_nonzero(mask_existing))
    _write_manifest(checkpoint_json, ckpt)
    if verbose:
        print(f"built full-present mask: {int(np.count_nonzero(mask_existing))}/{unique_y.shape[0]} Ys present in full cache")


def _select_unsolved_batch_by_presence(unique_y_path, solved_mask_path, full_present_mask_path, desired_present, batch_size, scan_start):
    Y = np.load(unique_y_path, mmap_mode='r')
    solved = np.load(solved_mask_path, mmap_mode='r')
    present = np.load(full_present_mask_path, mmap_mode='r')
    n = Y.shape[0]
    batch = []
    idxs = []
    i = max(0, int(scan_start))
    desired = bool(desired_present)
    while i < n and len(batch) < batch_size:
        if (not bool(solved[i])) and (bool(present[i]) == desired):
            y = Y[i]
            batch.append((int(y[0]), int(y[1]), int(y[2])))
            idxs.append(i)
        i += 1
    return batch, idxs, i, n


def filter_cached_assigned_Ys(
    Y_list,
    rank,
    f,
    d,
    eps,
    full_paths,
    verbose=False,
    batch_label=None,
    global_elapsed_before_batch=None,
    mean_batch_seconds=None,
    total_batches=None,
):
    root_info = {}
    total = len(Y_list)
    if total == 0:
        return root_info
    arr = np.asarray(Y_list, dtype=np.int64)
    locator = _build_full_roots_locator(arr, full_paths)
    coeffs_by_coord, rhs0_by_coord = _coord_halfspace_params(f, d, eps)
    file_cache = OrderedDict()
    filtered_cache = OrderedDict()
    batch_start_time = time.time()
    last_report_time = time.time()
    for i, Y in enumerate(Y_list, 1):
        info = _get_filtered_roots_from_full_cache(
            Y, locator, full_paths, coeffs_by_coord, rhs0_by_coord,
            file_cache, filtered_cache, file_cache_size=8, filtered_cache_size=max(4096, total + 1)
        )
        if info is None:
            info = {"roots0": tuple(), "roots1": tuple(), "roots2": tuple()}
        info = {
            "roots_all": tuple(),
            "roots0": info["roots0"],
            "roots1": info["roots1"],
            "roots2": info["roots2"],
            "nroots_all": 0,
            "screen": "FULL_CACHE",
            "status": "FULL_CACHE",
            "used_legacy": False,
            "B2": None,
            "B": None,
            "diagnostics": {},
        }
        root_info[Y] = info
        now = time.time()
        if verbose and rank == 0 and now - last_report_time >= 10.0:
            last_report_time = now
            elapsed = now - batch_start_time
            rate = (i / elapsed) if elapsed > 0 else 0.0
            batch_eta = ((total - i) / rate) if rate > 0 else float("inf")
            total_elapsed = (global_elapsed_before_batch or 0.0) + elapsed
            if batch_label is not None and total_batches is not None and mean_batch_seconds is not None:
                remaining_batches = max(0, total_batches - batch_label)
                overall_eta = batch_eta + remaining_batches * mean_batch_seconds
                print(
                    f"rank {rank} cached-filter batch {batch_label}/{total_batches}: {i}/{total} ({100.0 * i / total:.1f}%) | "
                    f"rate={rate:.2f} Y/s | batch ETA={_fmt_seconds(batch_eta)} | "
                    f"elapsed={_fmt_seconds(total_elapsed)} | total ETA={_fmt_seconds(overall_eta)}"
                )
            else:
                print(
                    f"rank {rank} cached-filter progress: {i}/{total} ({100.0 * i / total:.1f}%) | "
                    f"rate={rate:.2f} Y/s | batch ETA={_fmt_seconds(batch_eta)} | elapsed={_fmt_seconds(total_elapsed)}"
                )
    if verbose and rank == 0 and total > 0:
        elapsed = time.time() - batch_start_time
        total_elapsed = (global_elapsed_before_batch or 0.0) + elapsed
        if batch_label is not None and total_batches is not None and mean_batch_seconds is not None:
            remaining_batches = max(0, total_batches - batch_label)
            overall_eta = remaining_batches * mean_batch_seconds
            print(
                f"rank {rank} cached-filter batch {batch_label}/{total_batches}: {total}/{total} (100%) | "
                f"rate={(total / elapsed) if elapsed > 0 else 0.0:.2f} Y/s | batch time={_fmt_seconds(elapsed)} | "
                f"elapsed={_fmt_seconds(total_elapsed)} | total ETA={_fmt_seconds(overall_eta)}"
            )
        else:
            print(f"rank {rank} cached-filter progress: {total}/{total} (100%)")
    return root_info


def _append_roots_batch(roots_dir: str, batch_idx: int, root_info: Dict[Tuple[int, int, int], dict]):
    os.makedirs(roots_dir, exist_ok=True)
    Ys = []
    roots0_lists = []
    roots1_lists = []
    roots2_lists = []
    for Y, info in root_info.items():
        Ys.append(tuple(int(x) for x in Y))
        roots0_lists.append(tuple(info.get("roots0", ())))
        roots1_lists.append(tuple(info.get("roots1", ())))
        roots2_lists.append(tuple(info.get("roots2", ())))
    roots0_flat, roots0_off = _pack_roots_flat(roots0_lists)
    roots1_flat, roots1_off = _pack_roots_flat(roots1_lists)
    roots2_flat, roots2_off = _pack_roots_flat(roots2_lists)
    out_path = os.path.join(roots_dir, f"roots_batch_{batch_idx:06d}.npz")
    np.savez(
        out_path,
        Y=np.asarray(Ys, dtype=np.int64),
        roots0_flat=roots0_flat,
        roots0_off=roots0_off,
        roots1_flat=roots1_flat,
        roots1_off=roots1_off,
        roots2_flat=roots2_flat,
        roots2_off=roots2_off,
    )
    return out_path



def _structured_view_y(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr.view(dtype=np.dtype([("y0", "<i8"), ("y1", "<i8"), ("y2", "<i8")])).reshape(-1)


def _ensure_roots_index(paths: dict, checkpoint_json: str, ckpt: dict, verbose=False):
    _ensure_generic_roots_index(
        paths["roots_dir"],
        paths["roots_index_y"],
        paths["roots_index_file"],
        paths["roots_index_row"],
        paths["roots_index_meta"],
        checkpoint_json,
        ckpt,
        verbose=verbose,
        ckpt_flag_key="roots_index_built",
    )



def _load_roots_rows_for_needed(needed_y: np.ndarray, paths: dict, f: int, d: np.ndarray, eps: float, verbose=False, file_cache_size=8):
    if needed_y.size == 0:
        return {}
    index_y = np.load(paths["roots_index_y"], mmap_mode='r')
    index_file = np.load(paths["roots_index_file"], mmap_mode='r')
    index_row = np.load(paths["roots_index_row"], mmap_mode='r')
    meta = _read_manifest(paths["roots_index_meta"])
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

        has_flat = all(k in data for k in ("roots0_flat", "roots0_off", "roots1_flat", "roots1_off", "roots2_flat", "roots2_off"))
        has_pre = all(k in data for k in ("roots0_blob", "roots1_blob", "roots2_blob"))

        if has_flat:
            r0_flat = data["roots0_flat"]
            r0_off = data["roots0_off"]
            r1_flat = data["roots1_flat"]
            r1_off = data["roots1_off"]
            r2_flat = data["roots2_flat"]
            r2_off = data["roots2_off"]

        for ri, y in rows:
            if has_flat:
                r0 = _unpack_roots_flat(r0_flat, r0_off, ri)
                r1 = _unpack_roots_flat(r1_flat, r1_off, ri)
                r2 = _unpack_roots_flat(r2_flat, r2_off, ri)
            elif has_pre:
                r0 = tuple(tuple(int(x) for x in r) for r in pickle.loads(bytes(data["roots0_blob"][ri])))
                r1 = tuple(tuple(int(x) for x in r) for r in pickle.loads(bytes(data["roots1_blob"][ri])))
                r2 = tuple(tuple(int(x) for x in r) for r in pickle.loads(bytes(data["roots2_blob"][ri])))
            else:
                roots = tuple(tuple(int(x) for x in r) for r in pickle.loads(bytes(data["roots_blob"][ri])))
                r0 = filter_roots_for_coord(roots, y, 0, f, d, eps)
                r1 = filter_roots_for_coord(roots, y, 1, f, d, eps)
                r2 = filter_roots_for_coord(roots, y, 2, f, d, eps)
            out[y] = {"roots0": r0, "roots1": r1, "roots2": r2}

        done += 1
        now = time.time()
        if verbose and total_files > 0 and (done == total_files or now - last_report >= 10.0):
            last_report = now
            print(f"load root files: {done}/{total_files}")
    return out



def solve_assigned_Ys(
    Y_list,
    rank,
    f,
    d,
    eps,
    verbose=False,
    use_legacy_fallback=True,
    log_dir="phase2_logs",
    batch_label=None,
    global_elapsed_before_batch=None,
    mean_batch_seconds=None,
    total_batches=None,
    log_roots=False,
):
    root_info = {}
    total = len(Y_list)
    log_fh, _ = _open_phase2_log(rank, enabled=log_roots, log_dir=log_dir)
    batch_start_time = time.time()
    try:
        last_report_time = time.time()
        for i, Y in enumerate(Y_list, 1):
            if log_fh is not None:
                print(f"rank {rank} roots start {i}/{total}: Y={Y}", file=log_fh)
            res = actual_roots_from_ideal_search(
                Y,
                orbit=True,
                check_real_embeddings=True,
                verbose=verbose,
                rank=rank,
                use_legacy_fallback=use_legacy_fallback,
            )
            roots = tuple(tuple(int(x) for x in r) for r in res.get("actual_roots", []))
            r0 = filter_roots_for_coord(roots, Y, 0, f, d, eps)
            r1 = filter_roots_for_coord(roots, Y, 1, f, d, eps)
            r2 = filter_roots_for_coord(roots, Y, 2, f, d, eps)
            info = {
                "roots_all": roots,
                "roots0": r0,
                "roots1": r1,
                "roots2": r2,
                "nroots_all": len(roots),
                "screen": res.get("screen"),
                "status": res.get("status"),
                "used_legacy": bool(res.get("used_legacy", False)),
                "B2": res.get("B2"),
                "B": res.get("B"),
                "diagnostics": res.get("diagnostics", {}),
            }
            root_info[Y] = info
            if log_fh is not None:
                diag_reason = _diag_reason(info["diagnostics"])
                print(
                    f"rank {rank} roots done  {i}/{total}: Y={Y}, screen={info['screen']}, status={info['status']}, "
                    f"used_legacy={info['used_legacy']}, B={info['B']}, roots={len(roots)}, "
                    f"keep=({len(r0)},{len(r1)},{len(r2)}), diag={diag_reason}",
                    file=log_fh,
                )
            now = time.time()
            if verbose and rank == 0 and now - last_report_time >= 10.0:
                last_report_time = now
                elapsed = now - batch_start_time
                rate = (i / elapsed) if elapsed > 0 else 0.0
                batch_eta = ((total - i) / rate) if rate > 0 else float("inf")
                total_elapsed = (global_elapsed_before_batch or 0.0) + elapsed
                if batch_label is not None and total_batches is not None and mean_batch_seconds is not None:
                    remaining_batches = max(0, total_batches - batch_label)
                    overall_eta = batch_eta + remaining_batches * mean_batch_seconds
                    print(
                        f"rank {rank} root batch {batch_label}/{total_batches}: {i}/{total} ({100.0 * i / total:.1f}%) | "
                        f"rate={rate:.2f} Y/s | batch ETA={_fmt_seconds(batch_eta)} | "
                        f"elapsed={_fmt_seconds(total_elapsed)} | total ETA={_fmt_seconds(overall_eta)}"
                    )
                else:
                    print(
                        f"rank {rank} root progress: {i}/{total} ({100.0 * i / total:.1f}%) | "
                        f"rate={rate:.2f} Y/s | batch ETA={_fmt_seconds(batch_eta)} | elapsed={_fmt_seconds(total_elapsed)}"
                    )
        if verbose and rank == 0 and total > 0:
            elapsed = time.time() - batch_start_time
            total_elapsed = (global_elapsed_before_batch or 0.0) + elapsed
            if batch_label is not None and total_batches is not None and mean_batch_seconds is not None:
                remaining_batches = max(0, total_batches - batch_label)
                overall_eta = remaining_batches * mean_batch_seconds
                print(
                    f"rank {rank} root batch {batch_label}/{total_batches}: {total}/{total} (100%) | "
                    f"rate={(total / elapsed) if elapsed > 0 else 0.0:.2f} Y/s | batch time={_fmt_seconds(elapsed)} | "
                    f"elapsed={_fmt_seconds(total_elapsed)} | total ETA={_fmt_seconds(overall_eta)}"
                )
            else:
                print(f"rank {rank} root progress: {total}/{total} (100%)")
    finally:
        if log_fh is not None:
            log_fh.close()
    return root_info


def _append_candidates_npz_chunk(prefix: str, chunk_idx: int, candidates):
    n = len(candidates)
    dists = np.zeros(n, dtype=np.float64)
    x = np.zeros((n, 3), dtype=np.complex128)
    u_coeffs = np.zeros((n, 3, 6), dtype=np.int64)
    Ys = np.zeros((n, 3, 3), dtype=np.int64)
    for k, item in enumerate(candidates):
        dists[k] = item["dist"]
        x[k] = item["x"]
        u_coeffs[k] = item["u_coeffs"]
        Ys[k] = item["Y"]
    out_path = f"{prefix}.candidates_{chunk_idx:06d}.npz"
    np.savez(out_path, dist=dists, x=x, u_coeffs=u_coeffs, Y=Ys)
    return out_path, n


def find_roots_pipeline(
    triples_file: str,
    triples_json: str,
    f: int,
    t: float,
    eps: float,
    norm: int = 2,
    state_prefix: str = None,
    full_root_prefix: str = None,
    checkpoint_json: str = None,
    candidate_prefix: str = None,
    discover_chunk_rows: int = 200000,
    solve_batch_size: int = 200000,
    finalize_chunk_rows: int = 200000,
    max_roots_per_Y=None,
    use_legacy_fallback=True,
    verbose=False,
    root_info_batch_items=64,
    root_info_batch_roots=2000,
    resume=False,
    log_roots=False,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if candidate_prefix is None:
        candidate_prefix = triples_file
    if checkpoint_json is None:
        checkpoint_json = candidate_prefix + ".roots.json"
    if state_prefix is None:
        state_prefix = candidate_prefix + ".state"

    paths = _state_paths(state_prefix)
    unique_y_path = paths["unique_y"]
    solved_mask_path = paths["solved_mask"]
    roots_dir = paths["roots_dir"]

    full_paths = _state_paths(full_root_prefix) if full_root_prefix is not None else None

    d = target_vector_first_case(t, norm=norm)

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
                "version": 3,
                "triples_file": os.path.abspath(triples_file),
                "triples_json": os.path.abspath(triples_json),
                "state_prefix": os.path.abspath(state_prefix),
                "full_root_prefix": None if full_root_prefix is None else os.path.abspath(full_root_prefix),
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
                "roots_index_built": False,
                "full_roots_index_built": False,
                "full_present_mask_built": False,
                "full_roots_batch_file_index": 0,
                "cached_filter_completed": False,
                "cached_scan_start": 0,
                "completed_cached_batches": 0,
                "completed_missing_batches": 0,
                "finalize_completed": False,
                "finalize_next_row": 0,
                "candidate_chunk_index": 0,
                "candidates_written": 0,
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
        if "remaining_unsolved_count" not in ckpt:
            if os.path.exists(unique_y_path):
                if "unique_y_count" not in ckpt:
                    ckpt["unique_y_count"] = int(np.load(unique_y_path, mmap_mode='r').shape[0])
                ckpt["remaining_unsolved_count"] = int(ckpt["unique_y_count"]) - int(ckpt.get("solved_y_count", 0))
                _write_manifest(checkpoint_json, ckpt)
        if full_paths is not None and os.path.exists(full_paths["full_roots_dir"]):
            if not ckpt.get("full_roots_index_built", False):
                _ensure_full_roots_index(full_paths, checkpoint_json, ckpt, verbose=verbose)
                ckpt = _read_manifest(checkpoint_json)
            if not ckpt.get("full_present_mask_built", False):
                _build_full_present_mask(unique_y_path, full_paths, paths["full_present_mask"], checkpoint_json, ckpt, verbose=verbose)
                ckpt = _read_manifest(checkpoint_json)
    ckpt = comm.bcast(ckpt, root=0)

    phase_loop_start = time.time()
    completed_root_batches = int(ckpt.get("completed_root_batches", 0))

    if full_paths is not None and os.path.exists(full_paths["full_roots_dir"]):
        while True:
            if rank == 0:
                if not os.path.exists(paths["full_present_mask"]):
                    if not ckpt.get("full_roots_index_built", False):
                        _ensure_full_roots_index(full_paths, checkpoint_json, ckpt, verbose=verbose)
                        ckpt = _read_manifest(checkpoint_json)
                    _build_full_present_mask(unique_y_path, full_paths, paths["full_present_mask"], checkpoint_json, ckpt, verbose=verbose)
                    ckpt = _read_manifest(checkpoint_json)
                present_mask = np.load(paths["full_present_mask"], mmap_mode='r')
                remaining_cached_before = int(np.count_nonzero((~np.load(solved_mask_path, mmap_mode='r')) & present_mask))
                batch, idxs, next_scan, _ = _select_unsolved_batch_by_presence(
                    unique_y_path, solved_mask_path, paths["full_present_mask"], True,
                    int(solve_batch_size), int(ckpt.get("cached_scan_start", 0))
                )
                total_batches = int(math.ceil(remaining_cached_before / float(solve_batch_size))) if remaining_cached_before > 0 else 0
                current_batch_idx = int(ckpt.get("completed_cached_batches", 0)) + 1 if batch else int(ckpt.get("completed_cached_batches", 0))
                mean_batch_seconds = ((time.time() - phase_loop_start) / completed_root_batches) if completed_root_batches > 0 else None
                if verbose and batch:
                    msg = f"cached-filter batch {current_batch_idx}/{max(total_batches, current_batch_idx)} | size: {len(batch)} | cached remaining={remaining_cached_before}"
                    if mean_batch_seconds is not None:
                        msg += f" | avg completed batch={_fmt_seconds(mean_batch_seconds)} | elapsed={_fmt_seconds(time.time() - phase_loop_start)}"
                    print(msg)
                if not batch:
                    buckets = [[] for _ in range(comm.Get_size())]
                    batch_payload = {"idxs": [], "next_scan": next_scan, "selected_count": 0, "remaining_before": remaining_cached_before}
                else:
                    weights = [Y_weight(Y) for Y in batch]
                    buckets, _ = weighted_partition(batch, weights, comm.Get_size())
                    batch_payload = {"idxs": idxs, "next_scan": next_scan, "selected_count": len(idxs), "remaining_before": remaining_cached_before}
                batch_meta = {
                    "current_batch_idx": current_batch_idx,
                    "total_batches": total_batches,
                    "mean_batch_seconds": mean_batch_seconds,
                    "global_elapsed_before_batch": time.time() - phase_loop_start,
                    "batch_payload": batch_payload,
                }
            else:
                buckets = None
                batch_meta = None
            Ys_local = comm.scatter(buckets, root=0)
            batch_meta = comm.bcast(batch_meta, root=0)
            if int(batch_meta["batch_payload"].get("selected_count", 0)) == 0:
                break
            batch_wall0 = time.time()
            local_root_info = filter_cached_assigned_Ys(
                Ys_local, rank, f=f, d=d, eps=eps, full_paths=full_paths, verbose=verbose,
                batch_label=batch_meta.get("current_batch_idx"),
                global_elapsed_before_batch=batch_meta.get("global_elapsed_before_batch"),
                mean_batch_seconds=batch_meta.get("mean_batch_seconds"),
                total_batches=batch_meta.get("total_batches"),
            )
            merged = gather_root_info_batched(
                comm, local_root_info, root=0,
                max_items=root_info_batch_items, max_roots=root_info_batch_roots,
            )
            if rank == 0:
                _mark_solved_batch(solved_mask_path, batch_meta["batch_payload"]["idxs"])
                roots_path = _append_roots_batch(roots_dir, int(ckpt.get("roots_batch_file_index", 0)), merged)
                ckpt["roots_batch_file_index"] = int(ckpt.get("roots_batch_file_index", 0)) + 1
                completed_root_batches += 1
                ckpt["completed_root_batches"] = completed_root_batches
                ckpt["completed_cached_batches"] = int(ckpt.get("completed_cached_batches", 0)) + 1
                ckpt["cached_scan_start"] = int(batch_meta["batch_payload"]["next_scan"])
                batch_seconds = time.time() - batch_wall0
                selected_count = int(batch_meta["batch_payload"].get("selected_count", len(batch_meta["batch_payload"].get("idxs", []))))
                ckpt["solved_y_count"] = int(ckpt.get("solved_y_count", 0)) + selected_count
                ckpt["remaining_unsolved_count"] = max(0, int(ckpt.get("remaining_unsolved_count", 0)) - selected_count)
                ckpt["roots_index_built"] = False
                if verbose:
                    elapsed = time.time() - phase_loop_start
                    avg_batch = elapsed / completed_root_batches if completed_root_batches > 0 else batch_seconds
                    print(
                        f"cached-filter batch {ckpt['completed_cached_batches']} done | batch time={_fmt_seconds(batch_seconds)} | "
                        f"avg batch={_fmt_seconds(avg_batch)} | solved={ckpt['solved_y_count']} | saved roots={roots_path}"
                    )
                _write_manifest(checkpoint_json, ckpt)
            comm.Barrier()
        if rank == 0:
            ckpt = _read_manifest(checkpoint_json)
            ckpt["cached_filter_completed"] = True
            _write_manifest(checkpoint_json, ckpt)
        ckpt = comm.bcast(_read_manifest(checkpoint_json) if rank == 0 else None, root=0)

    while True:
        if rank == 0:
            remaining_unsolved_before = int(ckpt.get("remaining_unsolved_count", 0))
            if full_paths is not None and os.path.exists(full_paths["full_roots_dir"]):
                if not os.path.exists(paths["full_present_mask"]):
                    if not ckpt.get("full_roots_index_built", False):
                        _ensure_full_roots_index(full_paths, checkpoint_json, ckpt, verbose=verbose)
                        ckpt = _read_manifest(checkpoint_json)
                    _build_full_present_mask(unique_y_path, full_paths, paths["full_present_mask"], checkpoint_json, ckpt, verbose=verbose)
                    ckpt = _read_manifest(checkpoint_json)
                batch, idxs, next_scan, _ = _select_unsolved_batch_by_presence(
                    unique_y_path, solved_mask_path, paths["full_present_mask"], False,
                    int(solve_batch_size), int(ckpt.get("solve_scan_start", 0))
                )
            else:
                batch, idxs, next_scan, _ = _select_unsolved_batch(unique_y_path, solved_mask_path, int(solve_batch_size), int(ckpt.get("solve_scan_start", 0)))
            total_batches = int(math.ceil(remaining_unsolved_before / float(solve_batch_size))) if remaining_unsolved_before > 0 else 0
            current_batch_idx = completed_root_batches + 1 if batch else completed_root_batches
            mean_batch_seconds = ((time.time() - phase_loop_start) / completed_root_batches) if completed_root_batches > 0 else None
            if verbose and batch:
                msg = f"root-solve batch {current_batch_idx}/{max(total_batches, current_batch_idx)} | size: {len(batch)} | unsolved remaining={remaining_unsolved_before}"
                if mean_batch_seconds is not None:
                    msg += f" | avg completed batch={_fmt_seconds(mean_batch_seconds)} | elapsed={_fmt_seconds(time.time() - phase_loop_start)}"
                print(msg)
            if not batch:
                buckets = [[] for _ in range(comm.Get_size())]
                batch_payload = {"idxs": [], "next_scan": next_scan, "selected_count": 0, "remaining_unsolved_before": remaining_unsolved_before}
            else:
                weights = [Y_weight(Y) for Y in batch]
                buckets, _ = weighted_partition(batch, weights, comm.Get_size())
                batch_payload = {"idxs": idxs, "next_scan": next_scan, "selected_count": len(idxs), "remaining_unsolved_before": remaining_unsolved_before}
            batch_meta = {
                "current_batch_idx": current_batch_idx,
                "total_batches": total_batches,
                "mean_batch_seconds": mean_batch_seconds,
                "global_elapsed_before_batch": time.time() - phase_loop_start,
                "batch_payload": batch_payload,
            }
        else:
            buckets = None
            batch_meta = None
        Ys_local = comm.scatter(buckets, root=0)
        batch_meta = comm.bcast(batch_meta, root=0)
        if int(batch_meta["batch_payload"].get("selected_count", 0)) == 0:
            break
        batch_wall0 = time.time()
        local_root_info = solve_assigned_Ys(
            Ys_local,
            rank,
            f=f,
            d=d,
            eps=eps,
            verbose=verbose,
            use_legacy_fallback=use_legacy_fallback,
            batch_label=batch_meta.get("current_batch_idx"),
            global_elapsed_before_batch=batch_meta.get("global_elapsed_before_batch"),
            mean_batch_seconds=batch_meta.get("mean_batch_seconds"),
            total_batches=batch_meta.get("total_batches"),
            log_roots=log_roots,
        )
        merged = gather_root_info_batched(
            comm,
            local_root_info,
            root=0,
            max_items=root_info_batch_items,
            max_roots=root_info_batch_roots,
        )
        if rank == 0:
            _mark_solved_batch(solved_mask_path, batch_meta["batch_payload"]["idxs"])
            roots_path = _append_roots_batch(roots_dir, int(ckpt.get("roots_batch_file_index", 0)), merged)
            ckpt["roots_batch_file_index"] = int(ckpt.get("roots_batch_file_index", 0)) + 1
            if full_paths is not None:
                full_roots_path = _append_full_roots_batch(full_paths["full_roots_dir"], int(ckpt.get("full_roots_batch_file_index", 0)), merged)
                ckpt["full_roots_batch_file_index"] = int(ckpt.get("full_roots_batch_file_index", 0)) + 1
                ckpt["full_roots_index_built"] = False
                ckpt["full_present_mask_built"] = False
            else:
                full_roots_path = None
            completed_root_batches += 1
            ckpt["completed_root_batches"] = completed_root_batches
            ckpt["completed_missing_batches"] = int(ckpt.get("completed_missing_batches", 0)) + 1
            ckpt["solve_scan_start"] = int(batch_meta["batch_payload"]["next_scan"])
            batch_seconds = time.time() - batch_wall0
            ckpt["last_root_batch_seconds"] = batch_seconds
            selected_count = int(batch_meta["batch_payload"].get("selected_count", len(batch_meta["batch_payload"].get("idxs", []))))
            ckpt["solved_y_count"] = int(ckpt.get("solved_y_count", 0)) + selected_count
            ckpt["remaining_unsolved_count"] = max(0, int(batch_meta["batch_payload"].get("remaining_unsolved_before", 0)) - selected_count)
            ckpt["roots_index_built"] = False
            if verbose:
                elapsed = time.time() - phase_loop_start
                avg_batch = elapsed / completed_root_batches if completed_root_batches > 0 else batch_seconds
                remaining_unsolved2 = int(ckpt.get("remaining_unsolved_count", 0))
                remaining_batches2 = int(math.ceil(remaining_unsolved2 / float(solve_batch_size))) if remaining_unsolved2 > 0 else 0
                print(
                    f"root-solve batch {completed_root_batches} done | batch time={_fmt_seconds(batch_seconds)} | "
                    f"avg batch={_fmt_seconds(avg_batch)} | solved={ckpt['solved_y_count']} | "
                    f"remaining batches={remaining_batches2} | ETA={_fmt_seconds(remaining_batches2 * avg_batch)} | saved roots={roots_path}" + (f" | saved full roots={full_roots_path}" if full_roots_path is not None else "")
                )
            _write_manifest(checkpoint_json, ckpt)
        comm.Barrier()

    if rank == 0:
        ckpt = _read_manifest(checkpoint_json)
        ckpt["solve_completed"] = True
        _write_manifest(checkpoint_json, ckpt)
        if verbose:
            print("building/loading compact roots index...")
        if not ckpt.get("roots_index_built", False):
            _ensure_roots_index(paths, checkpoint_json, ckpt, verbose=verbose)
            ckpt = _read_manifest(checkpoint_json)
        if verbose:
            print("starting finalization over triples...")
    else:
        d = None

    if rank == 0 and not ckpt["finalize_completed"]:
        start = int(ckpt["finalize_next_row"])
        last_report_time = time.time()
        while start < nrows:
            take = min(int(finalize_chunk_rows), nrows - start)
            if verbose:
                print(f"finalize chunk: rows {start}..{start + take - 1} of {nrows}")
            triples = _read_triple_rows(triples_file, start, take)
            if verbose:
                print(f"triples found: {len(triples)}")
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            if verbose:
                print(f"unique Ys needed: {len(needed)}")
            roots_cache = _load_roots_rows_for_needed(needed, paths, f, d, eps, verbose=False)
            exact_candidates = []
            if verbose:
                print(f"starting final matching...")
            for (itri,tri) in enumerate(triples):
                Y1_t = (int(tri[0]), int(tri[1]), int(tri[2]))
                Y2_t = (int(tri[3]), int(tri[4]), int(tri[5]))
                Y3_t = (int(tri[6]), int(tri[7]), int(tri[8]))
                r1info = roots_cache.get(Y1_t)
                r2info = roots_cache.get(Y2_t)
                r3info = roots_cache.get(Y3_t)
                roots1 = () if r1info is None else r1info["roots0"]
                roots2 = () if r2info is None else r2info["roots1"]
                roots3 = () if r3info is None else r3info["roots2"]
                if len(roots1) == 0 or len(roots2) == 0 or len(roots3) == 0:
                    continue
                if max_roots_per_Y is not None:
                    roots1 = roots1[:max_roots_per_Y]
                    roots2 = roots2[:max_roots_per_Y]
                    roots3 = roots3[:max_roots_per_Y]
                for u1 in roots1:
                    for u2 in roots2:
                        for u3 in roots3:
                            vec_coeffs = (u1, u2, u3)
                            x = vector_coeffs_to_complex(vec_coeffs, f)
                            dist = float(np.linalg.norm(x - d))
                            if dist <= eps:
                                exact_candidates.append({
                                    "dist": dist,
                                    "x": x,
                                    "u_coeffs": np.array(vec_coeffs, dtype=np.int64),
                                    "Y": np.array([Y1_t, Y2_t, Y3_t], dtype=np.int64),
                                })
                now = time.time()
                if verbose and (now - last_report_time >= 10.0 or start >= nrows):
                    last_report_time = now
                    print(f"checking roots: {itri}/{len(triples)}")
            if exact_candidates:
                exact_candidates.sort(key=lambda z: z["dist"])
                out_path, nnew = _append_candidates_npz_chunk(candidate_prefix, ckpt["candidate_chunk_index"], exact_candidates)
                ckpt["candidate_chunk_index"] += 1
                ckpt["candidates_written"] += nnew
                if verbose:
                    print(f"wrote {nnew} candidates to {out_path}")
            start += take
            ckpt["finalize_next_row"] = start
            ckpt["finalize_completed"] = start >= nrows
            _write_manifest(checkpoint_json, ckpt)
            now = time.time()
            if verbose and (now - last_report_time >= 10.0 or start >= nrows):
                last_report_time = now
                print(f"finalize triples: {start}/{nrows}")
        return ckpt
    if rank == 0:
        return ckpt
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--triples_json", required=True)
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--t", type=float, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--norm", type=int, default=2)
    parser.add_argument("--state_prefix", default=None)
    parser.add_argument("--full_root_prefix", default=None)
    parser.add_argument("--checkpoint_json", default=None)
    parser.add_argument("--candidate_prefix", default=None)
    parser.add_argument("--discover_chunk_rows", type=int, default=200000)
    parser.add_argument("--solve_batch_size", type=int, default=200000)
    parser.add_argument("--finalize_chunk_rows", type=int, default=200000)
    parser.add_argument("--max_roots_per_Y", type=int, default=None)
    parser.add_argument("--no_legacy_fallback", action="store_true")
    parser.add_argument("--root_info_batch_items", type=int, default=64)
    parser.add_argument("--root_info_batch_roots", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_roots", action="store_true")
    args = parser.parse_args()

    out = find_roots_pipeline(
        triples_file=args.triples_file,
        triples_json=args.triples_json,
        f=args.f,
        t=args.t,
        eps=args.eps,
        norm=args.norm,
        state_prefix=args.state_prefix,
        full_root_prefix=args.full_root_prefix,
        checkpoint_json=args.checkpoint_json,
        candidate_prefix=args.candidate_prefix,
        discover_chunk_rows=args.discover_chunk_rows,
        solve_batch_size=args.solve_batch_size,
        finalize_chunk_rows=args.finalize_chunk_rows,
        max_roots_per_Y=args.max_roots_per_Y,
        use_legacy_fallback=not args.no_legacy_fallback,
        verbose=not args.quiet,
        root_info_batch_items=args.root_info_batch_items,
        root_info_batch_roots=args.root_info_batch_roots,
        resume=args.resume,
        log_roots=args.log_roots,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
