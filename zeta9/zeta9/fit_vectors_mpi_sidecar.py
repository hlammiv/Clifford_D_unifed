"""Fit one or many target vectors using a precomputed rootdb database.

MPI-parallel version: targets are partitioned across ranks, while each rank
streams the same triple chunks and uses the same rootdb for its local targets.

This version keeps the original rootdb format unchanged, but adds an on-disk
phase sidecar:
- roots remain indexed by Y -> (file,row) through the existing rootdb index
- a separate sidecar stores, for each row, roots sorted by phase together with
  embedded complex values
- fitting computes the exact admissible phase interval implied by |z-d_i|<=eps
  and reads only the relevant contiguous sidecar segments for each target

The solver remains exact.
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


def coeffs_to_complex(x_coeffs, f):
    return embed(tuple(int(v) for v in x_coeffs), 1) / (3 ** f)


def coeffs_to_complex_noscale(x_coeffs):
    return embed(tuple(int(v) for v in x_coeffs), 1)


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
        "exact_roots_dir": prefix + ".exact_roots_batches",
        "exact_roots_index_y": prefix + ".exact_roots_index_y.npy",
        "exact_roots_index_file": prefix + ".exact_roots_index_file.npy",
        "exact_roots_index_row": prefix + ".exact_roots_index_row.npy",
        "exact_roots_index_meta": prefix + ".exact_roots_index_meta.json",
        "phase_sidecar_dir": prefix + ".phase_sidecar",
        "phase_sidecar_meta": prefix + ".phase_sidecar.meta.json",
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


def _unpack_roots_flat(flat: np.ndarray, off: np.ndarray, i: int):
    a = int(off[i])
    b = int(off[i + 1])
    if b <= a:
        return np.empty((0, 6), dtype=np.int64)
    return np.ascontiguousarray(flat[a:b], dtype=np.int64)


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


def _sidecar_batch_paths(sidecar_dir: str, batch_idx: int):
    stem = os.path.join(sidecar_dir, f"phase_batch_{batch_idx:06d}")
    return {
        "roots_flat": stem + ".roots_flat.npy",
        "phases_flat": stem + ".phases_flat.npy",
        "z_re_flat": stem + ".z_re_flat.npy",
        "z_im_flat": stem + ".z_im_flat.npy",
        "roots_off": stem + ".roots_off.npy",
    }


def _ensure_phase_sidecar(paths: dict, verbose=False):
    meta_path = paths["phase_sidecar_meta"]
    if os.path.exists(meta_path):
        meta = _read_manifest(meta_path)
        ok = True
        for rec in meta.get("batches", []):
            for key in ("roots_flat", "phases_flat", "z_re_flat", "z_im_flat", "roots_off"):
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
            print(f"build phase sidecar batch {batch_idx+1}/{len(root_files)}")
        data = np.load(root_path, allow_pickle=True, mmap_mode='r')
        Y = np.asarray(data["Y"], dtype=np.int64)
        roots_flat = np.asarray(data["roots_flat"], dtype=np.int64)
        roots_off = np.asarray(data["roots_off"], dtype=np.int64)

        out_roots = np.empty_like(roots_flat)
        out_phases = np.empty((roots_flat.shape[0],), dtype=np.float64)
        out_z_re = np.empty((roots_flat.shape[0],), dtype=np.float64)
        out_z_im = np.empty((roots_flat.shape[0],), dtype=np.float64)

        for row in range(Y.shape[0]):
            a = int(roots_off[row])
            b = int(roots_off[row + 1])
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

        bp = _sidecar_batch_paths(sidecar_dir, batch_idx)
        np.save(bp["roots_flat"], out_roots)
        np.save(bp["phases_flat"], out_phases)
        np.save(bp["z_re_flat"], out_z_re)
        np.save(bp["z_im_flat"], out_z_im)
        np.save(bp["roots_off"], roots_off)

        rec = {
            "batch_idx": int(batch_idx),
            "root_file": os.path.abspath(root_path),
            **bp,
        }
        batches_meta.append(rec)

    meta = {
        "version": 1,
        "rootdb_prefix": os.path.abspath(paths["exact_roots_dir"]).removesuffix(".exact_roots_batches"),
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
    }
    batch_cache[batch_idx] = cached
    while len(batch_cache) > max_entries:
        batch_cache.popitem(last=False)
    return cached


def _phase_slice_ranges(phase_sorted: np.ndarray, theta: float, delta: float):
    lo = theta - delta
    hi = theta + delta
    ranges = []
    if lo < 0.0:
        l1 = int(np.searchsorted(phase_sorted, 0.0, side="left"))
        r1 = int(np.searchsorted(phase_sorted, hi, side="right"))
        l2 = int(np.searchsorted(phase_sorted, lo + TWOPI, side="left"))
        r2 = int(np.searchsorted(phase_sorted, TWOPI, side="right"))
        if r1 > l1:
            ranges.append((l1, r1))
        if r2 > l2:
            ranges.append((l2, r2))
    elif hi >= TWOPI:
        l1 = int(np.searchsorted(phase_sorted, lo, side="left"))
        r1 = int(np.searchsorted(phase_sorted, TWOPI, side="right"))
        l2 = int(np.searchsorted(phase_sorted, 0.0, side="left"))
        r2 = int(np.searchsorted(phase_sorted, hi - TWOPI, side="right"))
        if r1 > l1:
            ranges.append((l1, r1))
        if r2 > l2:
            ranges.append((l2, r2))
    else:
        l = int(np.searchsorted(phase_sorted, lo, side="left"))
        r = int(np.searchsorted(phase_sorted, hi, side="right"))
        if r > l:
            ranges.append((l, r))
    return ranges


def _load_candidates_from_sidecar(
    y, locator: dict, phase_meta: dict, batch_cache: OrderedDict, target: complex, eps: float, f: int
):
    loc = locator.get(y)
    if loc is None:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)

    sigma1_M = sigma1_from_m012(*y)
    if sigma1_M < 0:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
    r = math.sqrt(max(0.0, sigma1_M)) / (3 ** f)

    rho = abs(target)
    if rho <= 0.0:
        if r > eps:
            return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
        batch_idx, row_idx = loc
        bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
        off = bd["roots_off"]
        a = int(off[row_idx]); b = int(off[row_idx+1])
        coeffs = np.ascontiguousarray(bd["roots_flat"][a:b], dtype=np.int64)
        z = (np.asarray(bd["z_re_flat"][a:b], dtype=np.float64) + 1j*np.asarray(bd["z_im_flat"][a:b], dtype=np.float64)) / (3 ** f)
        if z.size == 0:
            return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
        mask = np.abs(z - target) <= (eps + 1e-15)
        return np.ascontiguousarray(coeffs[mask], dtype=np.int64), np.ascontiguousarray(z[mask], dtype=np.complex128)

    if abs(r - rho) > eps:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
    if r + rho <= eps:
        batch_idx, row_idx = loc
        bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
        off = bd["roots_off"]
        a = int(off[row_idx]); b = int(off[row_idx+1])
        coeffs = np.ascontiguousarray(bd["roots_flat"][a:b], dtype=np.int64)
        z = (np.asarray(bd["z_re_flat"][a:b], dtype=np.float64) + 1j*np.asarray(bd["z_im_flat"][a:b], dtype=np.float64)) / (3 ** f)
        mask = np.abs(z - target) <= (eps + 1e-15)
        return np.ascontiguousarray(coeffs[mask], dtype=np.int64), np.ascontiguousarray(z[mask], dtype=np.complex128)

    denom = 2.0 * r * rho
    if denom <= 0.0:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
    c = (r * r + rho * rho - eps * eps) / denom
    if c > 1.0:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)
    if c <= -1.0:
        batch_idx, row_idx = loc
        bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
        off = bd["roots_off"]
        a = int(off[row_idx]); b = int(off[row_idx+1])
        coeffs = np.ascontiguousarray(bd["roots_flat"][a:b], dtype=np.int64)
        z = (np.asarray(bd["z_re_flat"][a:b], dtype=np.float64) + 1j*np.asarray(bd["z_im_flat"][a:b], dtype=np.float64)) / (3 ** f)
        mask = np.abs(z - target) <= (eps + 1e-15)
        return np.ascontiguousarray(coeffs[mask], dtype=np.int64), np.ascontiguousarray(z[mask], dtype=np.complex128)

    delta = math.acos(c)
    theta = float(np.mod(np.angle(target), TWOPI))

    batch_idx, row_idx = loc
    bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
    off = bd["roots_off"]
    a = int(off[row_idx]); b = int(off[row_idx+1])
    if b <= a:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)

    phase_row = np.asarray(bd["phases_flat"][a:b], dtype=np.float64)
    ranges = _phase_slice_ranges(phase_row, theta, delta)
    if not ranges:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)

    coeff_parts = []
    z_parts = []
    for l, r_ in ranges:
        coeff_slice = np.ascontiguousarray(bd["roots_flat"][a + l:a + r_], dtype=np.int64)
        z_slice = (np.asarray(bd["z_re_flat"][a + l:a + r_], dtype=np.float64) + 1j*np.asarray(bd["z_im_flat"][a + l:a + r_], dtype=np.float64)) / (3 ** f)
        coeff_parts.append(coeff_slice)
        z_parts.append(np.ascontiguousarray(z_slice, dtype=np.complex128))

    coeffs = np.concatenate(coeff_parts, axis=0) if coeff_parts else np.empty((0, 6), dtype=np.int64)
    z = np.concatenate(z_parts, axis=0) if z_parts else np.empty((0,), dtype=np.complex128)
    if z.size == 0:
        return np.empty((0, 6), dtype=np.int64), np.empty((0,), dtype=np.complex128)

    mask = np.abs(z - target) <= (eps + 1e-15)
    return np.ascontiguousarray(coeffs[mask], dtype=np.int64), np.ascontiguousarray(z[mask], dtype=np.complex128)


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
    max_roots_per_Y: int | None = None,
    verbose: bool = False,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
        paths = _state_paths(rootdb_prefix)
        if not os.path.exists(paths["exact_roots_index_meta"]):
            raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")
        phase_meta = _ensure_phase_sidecar(paths, verbose=verbose)
    else:
        nrows = None
        targets = None
        paths = None
        phase_meta = None

    nrows = comm.bcast(nrows, root=0)
    targets = comm.bcast(targets, root=0)
    paths = comm.bcast(paths, root=0)
    phase_meta = comm.bcast(phase_meta, root=0)
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
    while start < nrows:
        take = min(int(triples_chunk_rows), nrows - start)
        if verbose and rank == 0:
            print(f"fit chunk rows {start}..{start + take - 1} of {nrows}")
        triples = _read_triple_rows(triples_file, start, take)
        needed = np.unique(triples.reshape((-1, 3)), axis=0)
        locator = _build_locator(needed, paths)

        for t0 in range(0, local_targets.shape[0], int(targets_chunk_size)):
            t1 = min(local_targets.shape[0], t0 + int(targets_chunk_size))
            targ = local_targets[t0:t1]
            candidate_cache = {}

            for y in locator.keys():
                for j in range(t1 - t0):
                    for coord in range(3):
                        coeffs, z = _load_candidates_from_sidecar(
                            y, locator, phase_meta, batch_cache, targ[j, coord], eps, f
                        )
                        if max_roots_per_Y is not None and coeffs.shape[0] > max_roots_per_Y:
                            coeffs = coeffs[:max_roots_per_Y, :]
                            z = z[:max_roots_per_Y]
                        candidate_cache[(j, y, coord)] = (coeffs, z)

            for tri in triples:
                Y1_t = (int(tri[0]), int(tri[1]), int(tri[2]))
                Y2_t = (int(tri[3]), int(tri[4]), int(tri[5]))
                Y3_t = (int(tri[6]), int(tri[7]), int(tri[8]))
                if Y1_t not in locator or Y2_t not in locator or Y3_t not in locator:
                    continue
                for j in range(t1 - t0):
                    local_j = t0 + j
                    coeff1, z1_all = candidate_cache[(j, Y1_t, 0)]
                    coeff2, z2_all = candidate_cache[(j, Y2_t, 1)]
                    coeff3, z3_all = candidate_cache[(j, Y3_t, 2)]
                    if coeff1.shape[0] == 0 or coeff2.shape[0] == 0 or coeff3.shape[0] == 0:
                        continue
                    d0, d1, d2 = targ[j, 0], targ[j, 1], targ[j, 2]
                    cur_best = local_best_dist[local_j]
                    cur_best_sq = cur_best * cur_best
                    for a in range(z1_all.shape[0]):
                        za = z1_all[a]
                        da = abs(za - d0) ** 2
                        if da >= cur_best_sq:
                            continue
                        for b in range(z2_all.shape[0]):
                            zb = z2_all[b]
                            dab = da + abs(zb - d1) ** 2
                            if dab >= cur_best_sq:
                                continue
                            for c in range(z3_all.shape[0]):
                                zc = z3_all[c]
                                dist = math.sqrt(dab + abs(zc - d2) ** 2)
                                if dist < cur_best:
                                    cur_best = dist
                                    cur_best_sq = dist * dist
                                    local_best_dist[local_j] = dist
                                    local_best_u_coeffs[local_j, 0, :] = coeff1[a]
                                    local_best_u_coeffs[local_j, 1, :] = coeff2[b]
                                    local_best_u_coeffs[local_j, 2, :] = coeff3[c]
                                    local_best_Y[local_j, 0, :] = np.asarray(Y1_t, dtype=np.int64)
                                    local_best_Y[local_j, 1, :] = np.asarray(Y2_t, dtype=np.int64)
                                    local_best_Y[local_j, 2, :] = np.asarray(Y3_t, dtype=np.int64)
                                    local_solved[local_j] = True
            if verbose and rank == 0:
                print(f"targets batch {t0}:{t1} done | solved so far in batch={int(np.count_nonzero(local_solved[t0:t1]))}/{t1-t0}")

        start += take

    payload = {
        "idx": local_idx,
        "best_dist": local_best_dist,
        "solved": local_solved,
        "best_u_coeffs": local_best_u_coeffs,
        "best_Y": local_best_Y,
    }
    gathered = comm.gather(payload, root=0)

    if rank == 0:
        best_dist = np.full(n_targets, np.inf, dtype=np.float64)
        best_u_coeffs = np.zeros((n_targets, 3, 6), dtype=np.int64)
        best_Y = np.zeros((n_targets, 3, 3), dtype=np.int64)
        solved = np.zeros(n_targets, dtype=bool)

        for part in gathered:
            idx = np.asarray(part["idx"], dtype=np.int64)
            best_dist[idx] = np.asarray(part["best_dist"], dtype=np.float64)
            solved[idx] = np.asarray(part["solved"], dtype=bool)
            best_u_coeffs[idx, :, :] = np.asarray(part["best_u_coeffs"], dtype=np.int64)
            best_Y[idx, :, :] = np.asarray(part["best_Y"], dtype=np.int64)

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
        }
        _write_manifest(output_prefix + ".json", summary)
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
    parser.add_argument("--max_roots_per_Y", type=int, default=None)
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
        max_roots_per_Y=args.max_roots_per_Y,
        verbose=not args.quiet,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
