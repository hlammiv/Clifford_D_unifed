"""Fit one or many target vectors using a precomputed rootdb database.

Requires a prebuilt binned phase sidecar from build_phase_sidecar_binned.py.

MPI-parallel version: targets are partitioned across ranks, while each rank
streams the same triple chunks and uses the same rootdb for its local targets.

This version keeps the original rootdb format unchanged, but adds an on-disk
BINNED phase sidecar:
- roots remain indexed by Y -> (file,row) through the existing rootdb index
- a separate sidecar stores, for each row, roots sorted by phase together with
  embedded complex values
- per-row bin offsets let fitting read only a small number of contiguous ranges
  for each target/component
- final filtering still uses the exact |z-d_i| <= eps condition, so the solver
  remains exact
"""

import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
from mpi4py import MPI

try:
    from .tools import embed
except ImportError:
    from tools import embed

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
TWOPI = 2.0 * math.pi


def _read_manifest(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def sigma1_from_m012(m0: int, m1: int, m2: int) -> float:
    return float(m0) + ALPHA1 * float(m1) + ALPHA1_SQ * float(m2)


def coeffs_to_complex_noscale(x_coeffs):
    return embed(tuple(int(v) for v in x_coeffs), 1)


def _read_triple_rows(path: str, start_row: int, nrows: int) -> np.ndarray:
    if nrows <= 0:
        return np.empty((0, 9), dtype=np.int64)
    # Auto-detect compact format (Z9TC header) written by select_triples_optimized
    # with --drop_col_c. The reader reconstructs Y_c from the stored Y_a, Y_b
    # and the target_sum embedded in the file header. See
    # select_triples_optimized.read_tm_rows_compact for the canonical impl.
    try:
        from .select_triples_optimized import read_tm_compact_header, read_tm_rows_compact
    except ImportError:
        from select_triples_optimized import read_tm_compact_header, read_tm_rows_compact  # type: ignore
    if read_tm_compact_header(path) is not None:
        return read_tm_rows_compact(path, start_row, nrows)
    itemsize = np.dtype(np.int64).itemsize
    offset = start_row * 9 * itemsize
    with open(path, "rb") as fh:
        fh.seek(offset, os.SEEK_SET)
        arr = np.fromfile(fh, dtype=np.int64, count=nrows * 9)
    return arr.reshape((-1, 9))


def _state_paths(prefix: str, n_phase_bins: int):
    suffix = f".phase_sidecar_binned_bins={int(n_phase_bins)}"
    return {
        "exact_roots_dir": prefix + ".exact_roots_batches",
        "exact_roots_index_y": prefix + ".exact_roots_index_y.npy",
        "exact_roots_index_file": prefix + ".exact_roots_index_file.npy",
        "exact_roots_index_row": prefix + ".exact_roots_index_row.npy",
        "exact_roots_index_meta": prefix + ".exact_roots_index_meta.json",
        "phase_sidecar_dir": prefix + suffix,
        "phase_sidecar_meta": prefix + suffix + ".meta.json",
    }


def _structured_view_y(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr.view(dtype=np.dtype([("y0", "<i8"), ("y1", "<i8"), ("y2", "<i8")])).reshape(-1)


def _partition_target_indices(n_targets: int, size: int, rank: int):
    base = n_targets // size
    rem = n_targets % size
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    end = start + count
    return np.arange(start, end, dtype=np.int64)


def _build_locator(needed_y: np.ndarray, paths: dict):
    if needed_y.size == 0:
        return {}
    index_y = np.load(paths["exact_roots_index_y"], mmap_mode='r')
    index_file = np.load(paths["exact_roots_index_file"], mmap_mode='r')
    index_row = np.load(paths["exact_roots_index_row"], mmap_mode='r')
    key_index = _structured_view_y(index_y)
    key_need = _structured_view_y(np.unique(np.ascontiguousarray(needed_y, dtype=np.int64), axis=0))
    pos = np.searchsorted(key_index, key_need)
    locator = {}
    for ky, p in zip(key_need, pos):
        if p >= key_index.shape[0] or key_index[p] != ky:
            continue
        locator[(int(ky["y0"]), int(ky["y1"]), int(ky["y2"]))] = (int(index_file[p]), int(index_row[p]))
    return locator


def _load_chunk_metadata(path: str):
    data = np.load(path, allow_pickle=False)
    needed_y = np.ascontiguousarray(data["needed_y"], dtype=np.int64)
    file_idx = np.asarray(data["file_idx"], dtype=np.int32)
    row_idx = np.asarray(data["row_idx"], dtype=np.int32)
    locator = {}
    for i in range(needed_y.shape[0]):
        fi = int(file_idx[i]); ri = int(row_idx[i])
        if fi < 0 or ri < 0:
            continue
        y = needed_y[i]
        locator[(int(y[0]), int(y[1]), int(y[2]))] = (fi, ri)
    return needed_y, locator


def _sidecar_batch_paths(sidecar_dir: str, batch_idx: int):
    stem = os.path.join(sidecar_dir, f"phase_batch_{batch_idx:06d}")
    return {
        "roots_flat": stem + ".roots_flat.npy",
        "phases_flat": stem + ".phases_flat.npy",
        "z_re_flat": stem + ".z_re_flat.npy",
        "z_im_flat": stem + ".z_im_flat.npy",
        "roots_off": stem + ".roots_off.npy",
        "bin_off": stem + ".bin_off.npy",
    }


def _ensure_binned_phase_sidecar(paths: dict, n_phase_bins: int = 512, verbose: bool = False):
    meta_path = paths["phase_sidecar_meta"]
    if os.path.exists(meta_path):
        meta = _read_manifest(meta_path)
        ok = int(meta.get("n_phase_bins", -1)) == int(n_phase_bins)
        if ok:
            for rec in meta.get("batches", []):
                for key in ("roots_flat", "phases_flat", "z_re_flat", "z_im_flat", "roots_off", "bin_off"):
                    if not os.path.exists(rec[key]):
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            return meta

    root_meta = _read_manifest(paths["exact_roots_index_meta"])
    root_files = root_meta["files"]
    sidecar_dir = paths["phase_sidecar_dir"]
    os.makedirs(sidecar_dir, exist_ok=True)

    batches_meta = []
    for batch_idx, root_path in enumerate(root_files):
        if verbose:
            print(f"build binned phase sidecar batch {batch_idx+1}/{len(root_files)}")
        data = np.load(root_path, allow_pickle=True, mmap_mode='r')
        Y = np.asarray(data["Y"], dtype=np.int64)
        roots_flat = np.asarray(data["roots_flat"], dtype=np.int64)
        roots_off = np.asarray(data["roots_off"], dtype=np.int64)

        out_roots = np.empty_like(roots_flat)
        out_phases = np.empty((roots_flat.shape[0],), dtype=np.float64)
        out_z_re = np.empty((roots_flat.shape[0],), dtype=np.float64)
        out_z_im = np.empty((roots_flat.shape[0],), dtype=np.float64)
        bin_off = np.zeros((Y.shape[0], int(n_phase_bins) + 1), dtype=np.int64)

        for row in range(Y.shape[0]):
            a = int(roots_off[row]); b = int(roots_off[row + 1])
            if b <= a:
                continue
            arr = np.ascontiguousarray(roots_flat[a:b], dtype=np.int64)
            z = np.array([coeffs_to_complex_noscale(r) for r in arr], dtype=np.complex128)
            phases = np.mod(np.angle(z), TWOPI)
            order = np.argsort(phases, kind="mergesort")
            arr = arr[order]
            z = z[order]
            phases = phases[order]

            out_roots[a:b, :] = arr
            out_phases[a:b] = phases
            out_z_re[a:b] = np.real(z)
            out_z_im[a:b] = np.imag(z)

            local_bins = np.floor(phases / TWOPI * n_phase_bins).astype(np.int64)
            local_bins = np.clip(local_bins, 0, n_phase_bins - 1)
            counts = np.bincount(local_bins, minlength=n_phase_bins)
            bin_off[row, 1:] = np.cumsum(counts, dtype=np.int64)

        bp = _sidecar_batch_paths(sidecar_dir, batch_idx)
        np.save(bp["roots_flat"], out_roots)
        np.save(bp["phases_flat"], out_phases)
        np.save(bp["z_re_flat"], out_z_re)
        np.save(bp["z_im_flat"], out_z_im)
        np.save(bp["roots_off"], roots_off)
        np.save(bp["bin_off"], bin_off)

        batches_meta.append({
            "batch_idx": int(batch_idx),
            "root_file": os.path.abspath(root_path),
            **bp,
        })

    meta = {
        "version": 1,
        "n_phase_bins": int(n_phase_bins),
        "source_root_files": [os.path.abspath(p) for p in root_files],
        "batches": batches_meta,
    }
    _write_manifest(meta_path, meta)
    return meta


def _get_sidecar_batch_cache(batch_idx: int, phase_meta: dict, batch_cache: OrderedDict, max_entries: int = 4):
    cached = batch_cache.get(batch_idx)
    if cached is not None:
        batch_cache.move_to_end(batch_idx)
        return cached

    rec = phase_meta["batches"][batch_idx]
    cached = {
        "roots_flat": np.load(rec["roots_flat"], mmap_mode="r"),
        "phases_flat": np.load(rec["phases_flat"], mmap_mode="r"),
        "z_re_flat": np.load(rec["z_re_flat"], mmap_mode="r"),
        "z_im_flat": np.load(rec["z_im_flat"], mmap_mode="r"),
        "roots_off": np.load(rec["roots_off"], mmap_mode="r"),
        "bin_off": np.load(rec["bin_off"], mmap_mode="r"),
    }
    batch_cache[batch_idx] = cached
    while len(batch_cache) > max_entries:
        batch_cache.popitem(last=False)
    return cached


def _intervals_from_exact_phase(theta: float, delta: float):
    lo = theta - delta
    hi = theta + delta
    if lo < 0.0:
        return [(0.0, hi), (lo + TWOPI, TWOPI)]
    if hi >= TWOPI:
        return [(lo, TWOPI), (0.0, hi - TWOPI)]
    return [(lo, hi)]


def _bin_ranges_for_interval(bin_off_row: np.ndarray, n_phase_bins: int, lo: float, hi: float):
    # Inclusive coverage of bins intersecting [lo, hi].
    if hi < lo:
        return []
    b0 = int(math.floor(lo / TWOPI * n_phase_bins))
    b1 = int(math.floor(hi / TWOPI * n_phase_bins))
    b0 = max(0, min(n_phase_bins - 1, b0))
    b1 = max(0, min(n_phase_bins - 1, b1))
    if b1 < b0:
        return []
    start = int(bin_off_row[b0])
    stop = int(bin_off_row[b1 + 1])
    if stop <= start:
        return []
    return [(start, stop)]


def _candidate_desc_from_binned_sidecar(
    y, locator: dict, phase_meta: dict, batch_cache: OrderedDict, target: complex, eps: float, f: int, profile: dict | None = None, disable_y_pruning: bool = False
):
    loc = locator.get(y)
    if loc is None:
        return None

    sigma1_M = sigma1_from_m012(*y)
    if sigma1_M < 0:
        return None
    r = math.sqrt(max(0.0, sigma1_M)) / (3 ** f)
    batch_idx, row_idx = loc
    bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)

    off = bd["roots_off"]
    a = int(off[row_idx]); b = int(off[row_idx + 1])
    if b <= a:
        return None

    if disable_y_pruning:
        if profile is not None:
            profile["candidate_rows_loaded"] += int(b - a)
        return (batch_idx, [(a, b)], target)

    rho = abs(target)
    n_phase_bins = int(phase_meta["n_phase_bins"])

    if rho <= 0.0:
        if r > eps:
            return None
        if profile is not None:
            profile["candidate_rows_loaded"] += int(b - a)
        return (batch_idx, [(a, b)], target)

    if abs(r - rho) > eps:
        return None

    if r + rho <= eps:
        if profile is not None:
            profile["candidate_rows_loaded"] += int(b - a)
        return (batch_idx, [(a, b)], target)

    denom = 2.0 * r * rho
    if denom <= 0.0:
        return None

    c = (r * r + rho * rho - eps * eps) / denom
    if c > 1.0:
        return None
    if c <= -1.0:
        if profile is not None:
            profile["candidate_rows_loaded"] += int(b - a)
        return (batch_idx, [(a, b)], target)

    delta = math.acos(c)
    theta = float(np.mod(np.angle(target), TWOPI))
    intervals = _intervals_from_exact_phase(theta, delta)
    bin_off_row = np.asarray(bd["bin_off"][row_idx], dtype=np.int64)

    ranges = []
    for lo, hi in intervals:
        for s, t in _bin_ranges_for_interval(bin_off_row, n_phase_bins, lo, hi):
            ranges.append((a + s, a + t))

    if not ranges:
        return None

    merged = []
    for s, t in sorted(ranges):
        if not merged or s > merged[-1][1]:
            merged.append([s, t])
        else:
            merged[-1][1] = max(merged[-1][1], t)
    merged = [(int(s), int(t)) for s, t in merged if t > s]
    if not merged:
        return None

    if profile is not None:
        profile["candidate_rows_loaded"] += int(sum(t - s for s, t in merged))
    return (batch_idx, merged, target)


def _iter_desc_z_and_indices(desc, phase_meta: dict, batch_cache: OrderedDict, f: int, eps: float):
    batch_idx, ranges, target = desc
    bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
    scale = float(3 ** f)
    for s, t in ranges:
        z = (np.asarray(bd["z_re_flat"][s:t], dtype=np.float64) + 1j * np.asarray(bd["z_im_flat"][s:t], dtype=np.float64)) / scale
        mask = np.abs(z - target) <= (eps + 1e-15)
        if np.any(mask):
            idx_local = np.flatnonzero(mask).astype(np.int64)
            yield s, idx_local, np.ascontiguousarray(z[mask], dtype=np.complex128)


def _load_coeff_row(batch_idx: int, abs_index: int, phase_meta: dict, batch_cache: OrderedDict):
    bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
    return np.asarray(bd["roots_flat"][abs_index], dtype=np.int64)


def fit_vectors(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    targets_npy: str,
    eps: float,
    output_prefix: str,
    triples_chunk_rows: int = 200000,
    targets_chunk_size: int = 16,
    n_phase_bins: int = 512,
    chunk_meta_json: str | None = None,
    disable_y_pruning: bool = False,
    profile_enabled: bool = True,
    verbose: bool = False,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rank_profile = {
        "sidecar_ensure": 0.0,
        "triple_read": 0.0,
        "needed_unique": 0.0,
        "locator_build": 0.0,
        "candidate_load": 0.0,
        "combination_loop": 0.0,
        "gather": 0.0,
        "chunks": 0,
        "target_batches": 0,
        "needed_y": 0,
        "candidate_rows_loaded": 0,
        "candidate_cache_entries": 0,
        "triples_processed": 0,
        "combinations_tested": 0,
        "solutions_improved": 0,
        "batch_cache_hits": 0,
        "batch_cache_misses": 0,
        "disable_y_pruning": bool(disable_y_pruning),
    }

    if rank == 0:
        tmeta = _read_manifest(triples_json)
        nrows = int(tmeta["rows_written"])
        file_size = os.path.getsize(triples_file)
        expected = nrows * 9 * np.dtype(np.int64).itemsize
        if file_size != expected:
            raise RuntimeError(f"Triple file size mismatch: expected {expected}, got {file_size}")
        targets = np.load(targets_npy)
        targets = np.asarray(targets, dtype=np.complex128)
        if targets.ndim == 1:
            targets = targets.reshape(1, 3)
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("targets_npy must have shape (3,) or (n_targets, 3)")
        paths = _state_paths(rootdb_prefix, n_phase_bins)
        if not os.path.exists(paths["exact_roots_index_meta"]):
            raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")
        t0 = os.times()[4]
        if not os.path.exists(paths["phase_sidecar_meta"]):
            raise FileNotFoundError(
                f"Binned phase sidecar not found. Run build_phase_sidecar.py first for {rootdb_prefix}"
            )
        phase_meta = _read_manifest(paths["phase_sidecar_meta"])
        chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
        rank_profile["sidecar_ensure"] += os.times()[4] - t0
    else:
        nrows = None
        targets = None
        paths = None
        phase_meta = None
        chunk_meta = None

    nrows = comm.bcast(nrows, root=0)
    targets = comm.bcast(targets, root=0)
    paths = comm.bcast(paths, root=0)
    phase_meta = comm.bcast(phase_meta, root=0)
    chunk_meta = comm.bcast(chunk_meta, root=0)
    comm.Barrier()

    n_targets = targets.shape[0]
    local_idx = _partition_target_indices(n_targets, size, rank)
    local_targets = targets[local_idx]

    local_best_dist = np.full(local_idx.shape[0], np.inf, dtype=np.float64)
    local_best_u_coeffs = np.zeros((local_idx.shape[0], 3, 6), dtype=np.int64)
    local_best_Y = np.zeros((local_idx.shape[0], 3, 3), dtype=np.int64)
    local_solved = np.zeros(local_idx.shape[0], dtype=bool)

    batch_cache = OrderedDict()
    start = 0

    chunk_iter = 0
    while start < nrows:
        take = min(int(triples_chunk_rows), nrows - start)
        t0 = os.times()[4]
        triples = _read_triple_rows(triples_file, start, take)
        rank_profile["triple_read"] += os.times()[4] - t0
        rank_profile["chunks"] += 1

        if verbose and rank == 0:
            print(f"fit chunk rows {start:,}..{start + take - 1:,} of {nrows:,}")

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_iter]
            t0 = os.times()[4]
            needed, locator = _load_chunk_metadata(rec["path"])
            rank_profile["needed_unique"] += os.times()[4] - t0
            rank_profile["needed_y"] += int(needed.shape[0])
        else:
            t0 = os.times()[4]
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            rank_profile["needed_unique"] += os.times()[4] - t0
            rank_profile["needed_y"] += int(needed.shape[0])

            t0 = os.times()[4]
            locator = _build_locator(needed, paths)
            rank_profile["locator_build"] += os.times()[4] - t0

        for t0_idx in range(0, local_targets.shape[0], int(targets_chunk_size)):
            t1_idx = min(local_targets.shape[0], t0_idx + int(targets_chunk_size))
            targ = local_targets[t0_idx:t1_idx]
            rank_profile["target_batches"] += 1

            t0 = os.times()[4]
            desc_cache = {}
            for y in locator.keys():
                for j in range(t1_idx - t0_idx):
                    for coord in range(3):
                        desc = _candidate_desc_from_binned_sidecar(
                            y, locator, phase_meta, batch_cache, targ[j, coord], eps, f, profile=rank_profile, disable_y_pruning=disable_y_pruning
                        )
                        desc_cache[(j, y, coord)] = desc
            rank_profile["candidate_load"] += os.times()[4] - t0
            rank_profile["candidate_cache_entries"] += int(len(desc_cache))

            t0 = os.times()[4]
            for tri in triples:
                Y1_t = (int(tri[0]), int(tri[1]), int(tri[2]))
                Y2_t = (int(tri[3]), int(tri[4]), int(tri[5]))
                Y3_t = (int(tri[6]), int(tri[7]), int(tri[8]))
                if Y1_t not in locator or Y2_t not in locator or Y3_t not in locator:
                    continue
                rank_profile["triples_processed"] += 1

                for j in range(t1_idx - t0_idx):
                    local_j = t0_idx + j
                    desc1 = desc_cache[(j, Y1_t, 0)]
                    desc2 = desc_cache[(j, Y2_t, 1)]
                    desc3 = desc_cache[(j, Y3_t, 2)]
                    if desc1 is None or desc2 is None or desc3 is None:
                        continue

                    d0, d1, d2 = targ[j, 0], targ[j, 1], targ[j, 2]
                    cur_best = local_best_dist[local_j]
                    cur_best_sq = cur_best * cur_best

                    for base1, idx1_local, z1 in _iter_desc_z_and_indices(desc1, phase_meta, batch_cache, f, eps):
                        if z1.size == 0:
                            continue
                        for a_idx, za in enumerate(z1):
                            da = abs(za - d0) ** 2
                            if da >= cur_best_sq:
                                continue
                            abs_a = int(base1 + idx1_local[a_idx])

                            for base2, idx2_local, z2 in _iter_desc_z_and_indices(desc2, phase_meta, batch_cache, f, eps):
                                if z2.size == 0:
                                    continue
                                for b_idx, zb in enumerate(z2):
                                    dab = da + abs(zb - d1) ** 2
                                    if dab >= cur_best_sq:
                                        continue
                                    abs_b = int(base2 + idx2_local[b_idx])

                                    for base3, idx3_local, z3 in _iter_desc_z_and_indices(desc3, phase_meta, batch_cache, f, eps):
                                        if z3.size == 0:
                                            continue
                                        for c_idx, zc in enumerate(z3):
                                            rank_profile["combinations_tested"] += 1
                                            dist = math.sqrt(dab + abs(zc - d2) ** 2)
                                            if dist < cur_best:
                                                cur_best = dist
                                                cur_best_sq = dist * dist
                                                local_best_dist[local_j] = dist
                                                local_best_u_coeffs[local_j, 0, :] = _load_coeff_row(desc1[0], abs_a, phase_meta, batch_cache)
                                                local_best_u_coeffs[local_j, 1, :] = _load_coeff_row(desc2[0], abs_b, phase_meta, batch_cache)
                                                abs_c = int(base3 + idx3_local[c_idx])
                                                local_best_u_coeffs[local_j, 2, :] = _load_coeff_row(desc3[0], abs_c, phase_meta, batch_cache)
                                                local_best_Y[local_j, 0, :] = np.asarray(Y1_t, dtype=np.int64)
                                                local_best_Y[local_j, 1, :] = np.asarray(Y2_t, dtype=np.int64)
                                                local_best_Y[local_j, 2, :] = np.asarray(Y3_t, dtype=np.int64)
                                                local_solved[local_j] = True
                                                rank_profile["solutions_improved"] += 1
            rank_profile["combination_loop"] += os.times()[4] - t0

            if verbose and rank == 0:
                print(f"targets batch {t0_idx}:{t1_idx} done | solved so far in batch={int(np.count_nonzero(local_solved[t0_idx:t1_idx]))}/{t1_idx-t0_idx}")

        start += take
        chunk_iter += 1

    t0 = os.times()[4]
    payload = {
        "idx": local_idx,
        "best_dist": local_best_dist,
        "solved": local_solved,
        "best_u_coeffs": local_best_u_coeffs,
        "best_Y": local_best_Y,
        "profile": rank_profile,
    }
    gathered = comm.gather(payload, root=0)
    rank_profile["gather"] += os.times()[4] - t0

    if rank == 0:
        best_dist = np.full(n_targets, np.inf, dtype=np.float64)
        best_u_coeffs = np.zeros((n_targets, 3, 6), dtype=np.int64)
        best_Y = np.zeros((n_targets, 3, 3), dtype=np.int64)
        solved = np.zeros(n_targets, dtype=bool)

        profile_by_rank = {}
        profile_sum = {}
        for part_i, part in enumerate(gathered):
            idx = np.asarray(part["idx"], dtype=np.int64)
            best_dist[idx] = np.asarray(part["best_dist"], dtype=np.float64)
            solved[idx] = np.asarray(part["solved"], dtype=bool)
            best_u_coeffs[idx, :, :] = np.asarray(part["best_u_coeffs"], dtype=np.int64)
            best_Y[idx, :, :] = np.asarray(part["best_Y"], dtype=np.int64)

            rp = part["profile"]
            profile_by_rank[str(part_i)] = rp
            for k, v in rp.items():
                if isinstance(v, (int, float)):
                    profile_sum[k] = profile_sum.get(k, 0) + v

        out_npz = output_prefix + ".npz"
        np.savez(
            out_npz,
            targets=targets,
            best_dist=np.where(np.isfinite(best_dist), best_dist, np.nan),
            solved=solved,
            best_u_coeffs=best_u_coeffs,
            best_Y=best_Y,
        )
        summary = {
            "n_targets": int(n_targets),
            "n_solved": int(np.count_nonzero(solved)),
            "results_npz": os.path.abspath(out_npz),
            "rootdb_prefix": os.path.abspath(rootdb_prefix),
            "mpi_ranks": int(size),
            "exact_phase_filter": True,
            "phase_sidecar": os.path.abspath(paths["phase_sidecar_dir"]),
            "n_phase_bins": int(phase_meta["n_phase_bins"]),
            "chunk_meta_json": None if chunk_meta_json is None else os.path.abspath(chunk_meta_json),
            "disable_y_pruning": bool(disable_y_pruning),
        }
        _write_manifest(output_prefix + ".json", summary)
        if profile_enabled:
            _write_manifest(output_prefix + ".profile.json", {"by_rank": profile_by_rank, "sum": profile_sum})
        return summary
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--triples_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--targets_npy", required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--triples_chunk_rows", type=int, default=200000)
    parser.add_argument("--targets_chunk_size", type=int, default=16)
    parser.add_argument("--n_phase_bins", type=int, default=512)
    parser.add_argument("--chunk_meta_json", default=None)
    parser.add_argument("--disable_y_pruning", action="store_true", help="Disable Y-level sidecar pruning and scan all roots in each selected Y row.")
    parser.add_argument("--no_profile", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = fit_vectors(
        triples_file=args.triples_file,
        triples_json=args.triples_json,
        rootdb_prefix=args.rootdb_prefix,
        f=args.f,
        targets_npy=args.targets_npy,
        eps=args.eps,
        output_prefix=args.output_prefix,
        triples_chunk_rows=args.triples_chunk_rows,
        targets_chunk_size=args.targets_chunk_size,
        n_phase_bins=args.n_phase_bins,
        chunk_meta_json=args.chunk_meta_json,
        disable_y_pruning=args.disable_y_pruning,
        profile_enabled=not args.no_profile,
        verbose=not args.quiet,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
