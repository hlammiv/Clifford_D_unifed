"""Fit one or many target vectors using a precomputed rootdb database.

This phase uses roots from --rootdb_prefix and evaluates geometric fits
for general complex target vectors supplied in --targets_npy.
"""

import argparse
import json
import math
import os
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np

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
    }


def _structured_view_y(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr.view(dtype=np.dtype([("y0", "<i8"), ("y1", "<i8"), ("y2", "<i8")])).reshape(-1)


def _passes_target_halfspace_mask(arr: np.ndarray, coeffs: np.ndarray, rhs: float) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0,), dtype=bool)
    arrf = np.asarray(arr, dtype=np.float64)
    lhs = arrf @ coeffs
    return lhs > rhs


def _coord_halfspace_params_for_targets(f: int, targets: np.ndarray, eps: float):
    scale = float(3 ** f)
    R = scale * float(eps)
    n_targets = targets.shape[0]
    coeffs = np.zeros((n_targets, 3, 6), dtype=np.float64)
    rhs0 = np.zeros((n_targets, 3), dtype=np.float64)
    for j in range(n_targets):
        for coord in range(3):
            Tx = scale * float(np.real(targets[j, coord]))
            Ty = scale * float(np.imag(targets[j, coord]))
            coeffs[j, coord, :] = np.array([
                Tx,
                C1 * Tx + S1 * Ty,
                C2 * Tx + S2 * Ty,
                C3 * Tx + S3 * Ty,
                C4 * Tx + S4 * Ty,
                C5 * Tx + S5 * Ty,
            ], dtype=np.float64)
            rhs0[j, coord] = 0.5 * ((Tx * Tx + Ty * Ty) - R * R)
    return coeffs, rhs0


def _unpack_roots_flat(flat: np.ndarray, off: np.ndarray, i: int):
    a = int(off[i])
    b = int(off[i + 1])
    if b <= a:
        return np.empty((0, 6), dtype=np.int64)
    return np.ascontiguousarray(flat[a:b], dtype=np.int64)


def _load_exact_roots_rows_for_needed(needed_y: np.ndarray, paths: dict, verbose=False, file_cache_size=8):
    if needed_y.size == 0:
        return {}
    index_y = np.load(paths["exact_roots_index_y"], mmap_mode='r')
    index_file = np.load(paths["exact_roots_index_file"], mmap_mode='r')
    index_row = np.load(paths["exact_roots_index_row"], mmap_mode='r')
    meta = _read_manifest(paths["exact_roots_index_meta"])
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
    last_report = 0.0
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
            out[y] = _unpack_roots_flat(flat, off, ri)
        done += 1
        if verbose:
            now = done / max(1, total_files)
            if done == total_files or now - last_report >= 0.1:
                last_report = now
                print(f"load rootdb files: {done}/{total_files}")
    return out


def fit_vectors(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    targets: np.ndarray,
    eps: float,
    output_root: str,
    triples_chunk_rows: int = 200000,
    targets_chunk_size: int = 16,
    max_roots_per_Y: int | None = None,
    verbose: bool = False,
):
    tmeta = _read_manifest(triples_json)
    nrows = int(tmeta["rows_written"])
    file_size = os.path.getsize(triples_file)
    expected = nrows * 9 * np.dtype(np.int64).itemsize
    if file_size != expected:
        raise RuntimeError(f"Triple file size mismatch: expected {expected}, got {file_size}")

    targets = np.asarray(targets, dtype=np.complex128)
    if targets.ndim == 1:
        targets = targets.reshape(1, 3)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_npy must have shape (3,) or (n_targets, 3)")

    paths = _state_paths(rootdb_prefix)
    if not os.path.exists(paths["exact_roots_index_meta"]):
        raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")

    n_targets = targets.shape[0]
    best_dist = np.full(n_targets, np.inf, dtype=np.float64)
    best_u_coeffs = np.zeros((n_targets, 3, 6), dtype=np.int64)
    best_Y = np.zeros((n_targets, 3, 3), dtype=np.int64)
    solved = np.zeros(n_targets, dtype=bool)

    start = 0
    while start < nrows:
        take = min(int(triples_chunk_rows), nrows - start)
        if verbose:
            print(f"fit chunk rows {start}..{start + take - 1} of {nrows}")
        triples = _read_triple_rows(triples_file, start, take)
        needed = np.unique(triples.reshape((-1, 3)), axis=0)
        exact_cache = _load_exact_roots_rows_for_needed(needed, paths, verbose=False)

        root_complex = {}
        for y, coeff_arr in exact_cache.items():
            if max_roots_per_Y is not None and coeff_arr.shape[0] > max_roots_per_Y:
                coeff_arr = coeff_arr[:max_roots_per_Y, :]
                exact_cache[y] = coeff_arr
            root_complex[y] = np.array([coeffs_to_complex(row, f) for row in coeff_arr], dtype=np.complex128)

        for t0 in range(0, n_targets, int(targets_chunk_size)):
            t1 = min(n_targets, t0 + int(targets_chunk_size))
            targ = targets[t0:t1]
            coeffs_by_t, rhs0_by_t = _coord_halfspace_params_for_targets(f, targ, eps)
            filter_idx = {}
            for y, coeff_arr in exact_cache.items():
                sigma1_M = sigma1_from_m012(*y)
                for j in range(t1 - t0):
                    for coord in range(3):
                        rhs = rhs0_by_t[j, coord] + 0.5 * sigma1_M
                        mask = _passes_target_halfspace_mask(coeff_arr, coeffs_by_t[j, coord], rhs)
                        filter_idx[(j, y, coord)] = np.flatnonzero(mask)

            for tri in triples:
                Y1_t = (int(tri[0]), int(tri[1]), int(tri[2]))
                Y2_t = (int(tri[3]), int(tri[4]), int(tri[5]))
                Y3_t = (int(tri[6]), int(tri[7]), int(tri[8]))
                coeff1 = exact_cache.get(Y1_t)
                coeff2 = exact_cache.get(Y2_t)
                coeff3 = exact_cache.get(Y3_t)
                if coeff1 is None or coeff2 is None or coeff3 is None:
                    continue
                z1_all = root_complex[Y1_t]
                z2_all = root_complex[Y2_t]
                z3_all = root_complex[Y3_t]
                for j in range(t1 - t0):
                    global_j = t0 + j
                    idx1 = filter_idx[(j, Y1_t, 0)]
                    idx2 = filter_idx[(j, Y2_t, 1)]
                    idx3 = filter_idx[(j, Y3_t, 2)]
                    if idx1.size == 0 or idx2.size == 0 or idx3.size == 0:
                        continue
                    d0, d1, d2 = targ[j, 0], targ[j, 1], targ[j, 2]
                    cur_best = best_dist[global_j]
                    for a in idx1:
                        za = z1_all[a]
                        da = abs(za - d0) ** 2
                        if da >= cur_best * cur_best:
                            continue
                        for b in idx2:
                            zb = z2_all[b]
                            dab = da + abs(zb - d1) ** 2
                            if dab >= cur_best * cur_best:
                                continue
                            for c in idx3:
                                zc = z3_all[c]
                                dist = math.sqrt(dab + abs(zc - d2) ** 2)
                                if dist < cur_best:
                                    cur_best = dist
                                    best_dist[global_j] = dist
                                    best_u_coeffs[global_j, 0, :] = coeff1[a]
                                    best_u_coeffs[global_j, 1, :] = coeff2[b]
                                    best_u_coeffs[global_j, 2, :] = coeff3[c]
                                    best_Y[global_j, 0, :] = np.asarray(Y1_t, dtype=np.int64)
                                    best_Y[global_j, 1, :] = np.asarray(Y2_t, dtype=np.int64)
                                    best_Y[global_j, 2, :] = np.asarray(Y3_t, dtype=np.int64)
                                    solved[global_j] = True
            if verbose:
                nsol = int(np.count_nonzero(solved[t0:t1]))
                print(f"targets batch {t0}:{t1} done | solved so far in batch={nsol}/{t1-t0}")

        start += take

    out_npz = output_root + ".npz"
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
    }
    _write_manifest(output_root + ".json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--triples_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--targets_npy", required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--output_root", required=True)
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
        output_root=args.output_root,
        triples_chunk_rows=args.triples_chunk_rows,
        targets_chunk_size=args.targets_chunk_size,
        max_roots_per_Y=args.max_roots_per_Y,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(out)
