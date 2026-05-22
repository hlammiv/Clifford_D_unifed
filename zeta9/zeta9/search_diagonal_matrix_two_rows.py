from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
from mpi4py import MPI

# Prefer the latest fit_vectors helper module if present; fall back to the non-v2 name.
try:
    from fit_vectors_mpi_sidecar_binned_v2 import (
        _build_locator,
        _candidate_desc_from_binned_sidecar,
        _iter_desc_z_and_indices,
        _load_chunk_metadata,
        _load_coeff_row,
        _read_manifest,
        _read_triple_rows,
        _state_paths,
        coeffs_to_complex_noscale,
        sigma1_from_m012,
    )
except ImportError:
    from fit_vectors_mpi_sidecar_binned import (
        _build_locator,
        _candidate_desc_from_binned_sidecar,
        _iter_desc_z_and_indices,
        _load_chunk_metadata,
        _load_coeff_row,
        _read_manifest,
        _read_triple_rows,
        _state_paths,
        coeffs_to_complex_noscale,
        sigma1_from_m012,
    )


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _vector_dist(z0: complex, z1: complex, z2: complex, d0: complex, d1: complex, d2: complex) -> float:
    return math.sqrt(abs(z0 - d0) ** 2 + abs(z1 - d1) ** 2 + abs(z2 - d2) ** 2)


def _frobenius_dist_from_rows(r1, r2, r3, d1: complex, d2: complex, d3: complex) -> float:
    a, b, c = r1
    u, v, w = r2
    x, y, z = r3
    return math.sqrt(
        abs(a - d1) ** 2 + abs(b) ** 2 + abs(c) ** 2
        + abs(u) ** 2 + abs(v - d2) ** 2 + abs(w) ** 2
        + abs(x) ** 2 + abs(y) ** 2 + abs(z - d3) ** 2
    )


def _candidate_key_from_coeffs(coeffs_3x6: np.ndarray):
    return tuple(int(v) for v in coeffs_3x6.reshape(-1))


def _coeffs6_to_complex_scaled(coeffs6: np.ndarray, f: int) -> complex:
    return coeffs_to_complex_noscale(tuple(int(v) for v in coeffs6)) / (3 ** f)


def _save_row1_candidates(output_prefix: str, row1_cands: list[dict]):
    npz_path = output_prefix + ".row1_candidates.npz"
    json_path = output_prefix + ".row1_candidates.json"
    n = len(row1_cands)
    if n == 0:
        np.savez(
            npz_path,
            dist=np.empty((0,), dtype=np.float64),
            z=np.empty((0, 3), dtype=np.complex128),
            u_coeffs=np.empty((0, 3, 6), dtype=np.int64),
            Y=np.empty((0, 3, 3), dtype=np.int64),
        )
    else:
        np.savez(
            npz_path,
            dist=np.asarray([r["dist"] for r in row1_cands], dtype=np.float64),
            z=np.asarray([r["z"] for r in row1_cands], dtype=np.complex128),
            u_coeffs=np.asarray([r["u_coeffs"] for r in row1_cands], dtype=np.int64),
            Y=np.asarray([r["Y"] for r in row1_cands], dtype=np.int64),
        )
    _write_manifest(json_path, {"n_candidates": int(n), "npz": os.path.abspath(npz_path)})


def _load_row1_candidates(output_prefix: str) -> Optional[list[dict]]:
    npz_path = output_prefix + ".row1_candidates.npz"
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=False)
    dist = np.asarray(data["dist"], dtype=np.float64)
    z = np.asarray(data["z"], dtype=np.complex128)
    u_coeffs = np.asarray(data["u_coeffs"], dtype=np.int64)
    Y = np.asarray(data["Y"], dtype=np.int64)
    out = []
    for i in range(dist.shape[0]):
        out.append({
            "dist": float(dist[i]),
            "z": np.asarray(z[i], dtype=np.complex128),
            "u_coeffs": np.asarray(u_coeffs[i], dtype=np.int64),
            "Y": np.asarray(Y[i], dtype=np.int64),
        })
    return out


def _save_results(output_prefix: str, results: list[dict]):
    npz_path = output_prefix + ".results.npz"
    json_path = output_prefix + ".results.json"
    n = len(results)
    if n == 0:
        np.savez(
            npz_path,
            frobenius_dist=np.empty((0,), dtype=np.float64),
            row1_index=np.empty((0,), dtype=np.int64),
            row1_dist=np.empty((0,), dtype=np.float64),
            row2_dist=np.empty((0,), dtype=np.float64),
            row1_u_coeffs=np.empty((0, 3, 6), dtype=np.int64),
            row2_u_coeffs=np.empty((0, 3, 6), dtype=np.int64),
            row3_u_coeffs=np.empty((0, 3, 6), dtype=np.int64),
            row1_Y=np.empty((0, 3, 3), dtype=np.int64),
            row2_Y=np.empty((0, 3, 3), dtype=np.int64),
        )
    else:
        np.savez(
            npz_path,
            frobenius_dist=np.asarray([r["frobenius_dist"] for r in results], dtype=np.float64),
            row1_index=np.asarray([r["row1_index"] for r in results], dtype=np.int64),
            row1_dist=np.asarray([r["row1_dist"] for r in results], dtype=np.float64),
            row2_dist=np.asarray([r["row2_dist"] for r in results], dtype=np.float64),
            row1_u_coeffs=np.asarray([r["row1_u_coeffs"] for r in results], dtype=np.int64),
            row2_u_coeffs=np.asarray([r["row2_u_coeffs"] for r in results], dtype=np.int64),
            row3_u_coeffs=np.asarray([r["row3_u_coeffs"] for r in results], dtype=np.int64),
            row1_Y=np.asarray([r["row1_Y"] for r in results], dtype=np.int64),
            row2_Y=np.asarray([r["row2_Y"] for r in results], dtype=np.int64),
        )
    _write_manifest(json_path, {"n_results": int(n), "npz": os.path.abspath(npz_path)})


def _result_key(rec: dict):
    return (
        tuple(int(v) for v in rec["row1_u_coeffs"].reshape(-1)),
        tuple(int(v) for v in rec["row2_u_coeffs"].reshape(-1)),
    )


def _merge_top_results(results: list[dict], limit: int) -> list[dict]:
    by_key = {}
    for rec in results:
        key = _result_key(rec)
        prev = by_key.get(key)
        if prev is None or (rec["frobenius_dist"], rec["row1_dist"], rec["row2_dist"]) < (
            prev["frobenius_dist"], prev["row1_dist"], prev["row2_dist"]
        ):
            by_key[key] = rec
    out = list(by_key.values())
    out.sort(key=lambda r: (r["frobenius_dist"], r["row1_dist"], r["row2_dist"], r["row1_index"]))
    return out[: int(limit)]


def _coerce_target_diag_value(v):
    if isinstance(v, str):
        try:
            from sage.all import CC, I, pi, e, exp, sage_eval, sqrt, sin, cos
            val = sage_eval(v, locals={
                "CC": CC,
                "I": I,
                "pi": pi,
                "e": e,
                "exp": exp,
                "sqrt": sqrt,
                "sin": sin,
                "cos": cos,
            })
            return complex(CC(val))
        except Exception:
            return complex(v)
    if np.isscalar(v):
        if np.iscomplexobj(v):
            return complex(v)
        return complex(float(v))
    return complex(v)


def _load_target_diag_npy(path: str):
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.shape != (3,):
        raise ValueError("target_diag_npy must have shape (3,)")
    return tuple(_coerce_target_diag_value(v) for v in arr.tolist())


def enumerate_vector_fits(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    target_vec: Sequence[complex],
    eps: float,
    max_candidates: Optional[int] = None,
    triples_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: str | None = None,
    deduplicate: bool = True,
    disable_y_pruning: bool = False,
    verbose: bool = False,
):
    target_vec = np.asarray(target_vec, dtype=np.complex128)
    if target_vec.shape != (3,):
        raise ValueError("target_vec must have shape (3,)")

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
            f"Binned phase sidecar not found. Run build_phase_sidecar.py first for {rootdb_prefix}"
        )

    phase_meta = _read_manifest(paths["phase_sidecar_meta"])
    chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
    batch_cache = OrderedDict()

    out = []
    best_by_key = {}

    start = 0
    chunk_iter = 0
    while start < nrows:
        take = min(int(triples_chunk_rows), nrows - start)
        triples = _read_triple_rows(triples_file, start, take)
        if verbose:
            print(f"row1 enumerate chunk rows {start:,}..{start + take - 1:,} of {nrows:,}")

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_iter]
            _, locator = _load_chunk_metadata(rec["path"])
        else:
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            locator = _build_locator(needed, paths)

        desc_cache = {}
        for y in locator.keys():
            for coord in range(3):
                desc_cache[(y, coord)] = _candidate_desc_from_binned_sidecar(
                    y, locator, phase_meta, batch_cache, target_vec[coord], eps, f, profile=None,
                    disable_y_pruning=disable_y_pruning,
                )

        d0, d1, d2 = target_vec[0], target_vec[1], target_vec[2]
        for tri in triples:
            Y1_t = (int(tri[0]), int(tri[1]), int(tri[2]))
            Y2_t = (int(tri[3]), int(tri[4]), int(tri[5]))
            Y3_t = (int(tri[6]), int(tri[7]), int(tri[8]))
            if Y1_t not in locator or Y2_t not in locator or Y3_t not in locator:
                continue

            desc1 = desc_cache[(Y1_t, 0)]
            desc2 = desc_cache[(Y2_t, 1)]
            desc3 = desc_cache[(Y3_t, 2)]
            if desc1 is None or desc2 is None or desc3 is None:
                continue

            for base1, idx1_local, z1 in _iter_desc_z_and_indices(desc1, phase_meta, batch_cache, f, eps):
                for a_idx, za in enumerate(z1):
                    abs_a = int(base1 + idx1_local[a_idx])
                    coeff_a = _load_coeff_row(desc1[0], abs_a, phase_meta, batch_cache)
                    for base2, idx2_local, z2 in _iter_desc_z_and_indices(desc2, phase_meta, batch_cache, f, eps):
                        for b_idx, zb in enumerate(z2):
                            abs_b = int(base2 + idx2_local[b_idx])
                            coeff_b = _load_coeff_row(desc2[0], abs_b, phase_meta, batch_cache)
                            for base3, idx3_local, z3 in _iter_desc_z_and_indices(desc3, phase_meta, batch_cache, f, eps):
                                for c_idx, zc in enumerate(z3):
                                    abs_c = int(base3 + idx3_local[c_idx])
                                    coeff_c = _load_coeff_row(desc3[0], abs_c, phase_meta, batch_cache)
                                    coeffs = np.stack([coeff_a, coeff_b, coeff_c], axis=0)
                                    dist = _vector_dist(za, zb, zc, d0, d1, d2)
                                    rec_out = {
                                        "dist": float(dist),
                                        "z": np.asarray([za, zb, zc], dtype=np.complex128),
                                        "u_coeffs": np.asarray(coeffs, dtype=np.int64),
                                        "Y": np.asarray([Y1_t, Y2_t, Y3_t], dtype=np.int64),
                                    }
                                    if deduplicate:
                                        key = _candidate_key_from_coeffs(rec_out["u_coeffs"])
                                        prev = best_by_key.get(key)
                                        if prev is None or dist < prev["dist"]:
                                            best_by_key[key] = rec_out
                                    else:
                                        out.append(rec_out)

        start += take
        chunk_iter += 1

    if deduplicate:
        out = list(best_by_key.values())
    out.sort(key=lambda r: r["dist"])
    if max_candidates is not None:
        out = out[: int(max_candidates)]
    return out


def _descending_partition(start: int, stop: int, size: int, rank: int) -> list[int]:
    if stop <= start:
        return []
    seq = list(range(stop - 1, start - 1, -1))
    return seq[rank::size]


def estimate_row1_candidate_counts(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    target_vec: Sequence[complex],
    eps: float,
    triples_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: str | None = None,
    disable_y_pruning: bool = False,
    verbose: bool = False,
):
    """
    Estimate/count possible row1 candidates for target_vec=(d1,0,0).

    Returns counts with multiplicity over triples:
      - unconstrained_root_product_count = sum r(Y1) r(Y2) r(Y3)
      - filtered_root_product_count      = sum r_eps(Y1) r_eps(Y2) r_eps(Y3)

    These are before vector deduplication.
    """
    target_vec = np.asarray(target_vec, dtype=np.complex128)
    if target_vec.shape != (3,):
        raise ValueError("target_vec must have shape (3,)")

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
            f"Binned phase sidecar not found. Run build_phase_sidecar.py first for {rootdb_prefix}"
        )

    phase_meta = _read_manifest(paths["phase_sidecar_meta"])
    chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
    batch_cache = OrderedDict()

    total_triples = 0
    triples_with_all_rootdb_rows = 0
    triples_with_all_filtered_components = 0
    unconstrained_root_product_count = 0
    filtered_root_product_count = 0
    unique_y_seen = set()

    root_count_cache = {}
    filtered_count_cache = {}

    def get_root_count(y, locator):
        key = tuple(int(v) for v in y)
        val = root_count_cache.get(key)
        if val is not None:
            return val
        loc = locator.get(key)
        if loc is None:
            root_count_cache[key] = 0
            return 0
        batch_idx, row_idx = loc
        off = batch_cache.get((batch_idx, 'counts'))
        if off is None:
            rec = phase_meta["batches"][batch_idx]
            off = np.load(rec["roots_off"], mmap_mode="r")
            batch_cache[(batch_idx, 'counts')] = off
        cnt = int(off[row_idx + 1] - off[row_idx])
        root_count_cache[key] = cnt
        return cnt

    def get_filtered_count(y, coord, locator):
        key = (tuple(int(v) for v in y), int(coord))
        val = filtered_count_cache.get(key)
        if val is not None:
            return val
        desc = _candidate_desc_from_binned_sidecar(
            key[0], locator, phase_meta, batch_cache, target_vec[coord], eps, f, profile=None,
            disable_y_pruning=disable_y_pruning,
        )
        if desc is None:
            filtered_count_cache[key] = 0
            return 0
        cnt = 0
        for _base, idx_local, _z in _iter_desc_z_and_indices(desc, phase_meta, batch_cache, f, eps):
            cnt += int(idx_local.size)
        filtered_count_cache[key] = cnt
        return cnt

    start = 0
    chunk_iter = 0
    while start < nrows:
        take = min(int(triples_chunk_rows), nrows - start)
        triples = _read_triple_rows(triples_file, start, take)
        if verbose:
            print(f"row1 estimate chunk rows {start:,}..{start + take - 1:,} of {nrows:,}")

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_iter]
            _, locator = _load_chunk_metadata(rec["path"])
        else:
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            locator = _build_locator(needed, paths)

        for tri in triples:
            total_triples += 1
            Y1_t = (int(tri[0]), int(tri[1]), int(tri[2]))
            Y2_t = (int(tri[3]), int(tri[4]), int(tri[5]))
            Y3_t = (int(tri[6]), int(tri[7]), int(tri[8]))
            unique_y_seen.add(Y1_t); unique_y_seen.add(Y2_t); unique_y_seen.add(Y3_t)

            r1 = get_root_count(Y1_t, locator)
            r2 = get_root_count(Y2_t, locator)
            r3 = get_root_count(Y3_t, locator)
            if r1 > 0 and r2 > 0 and r3 > 0:
                triples_with_all_rootdb_rows += 1
                unconstrained_root_product_count += int(r1) * int(r2) * int(r3)

            c1 = get_filtered_count(Y1_t, 0, locator)
            c2 = get_filtered_count(Y2_t, 1, locator)
            c3 = get_filtered_count(Y3_t, 2, locator)
            if c1 > 0 and c2 > 0 and c3 > 0:
                triples_with_all_filtered_components += 1
                filtered_root_product_count += int(c1) * int(c2) * int(c3)

        start += take
        chunk_iter += 1

    return {
        "total_triples": int(total_triples),
        "triples_with_all_rootdb_rows": int(triples_with_all_rootdb_rows),
        "triples_with_all_filtered_components": int(triples_with_all_filtered_components),
        "unique_y_seen": int(len(unique_y_seen)),
        "unconstrained_root_product_count": int(unconstrained_root_product_count),
        "filtered_root_product_count": int(filtered_root_product_count),
        "note": "Counts are with multiplicity over triples and before vector deduplication.",
    }


def _coeffs6_to_sage_elem(coeffs6, K):
    z = K.gen()
    out = K(0)
    for k, ck in enumerate(coeffs6):
        out += K(int(ck)) * (z ** k)
    return out


def _k_to_coeffs6_int(x, K):
    v = list(K(x).vector())
    if len(v) != 6:
        raise ValueError(f"Expected 6 coordinates, got {len(v)}")
    out = []
    for c in v:
        try:
            out.append(int(c))
        except Exception:
            if hasattr(c, "is_integer") and c.is_integer():
                out.append(int(c))
            else:
                raise ValueError(f"Non-integral coefficient encountered: {c}")
    return np.asarray(out, dtype=np.int64)


def _ring_abs_sq_complex(x, f: int) -> float:
    z = coeffs_to_complex_noscale(tuple(int(v) for v in x)) / (3 ** f)
    return float(abs(z) ** 2)


def _row2_pair_count_desc(desc_v, desc_w, phase_meta: dict, batch_cache: OrderedDict, f: int, eps: float) -> int:
    total_v = 0
    total_w = 0
    for _base, idx_local, _z in _iter_desc_z_and_indices(desc_v, phase_meta, batch_cache, f, eps):
        total_v += int(idx_local.size)
    for _base, idx_local, _z in _iter_desc_z_and_indices(desc_w, phase_meta, batch_cache, f, eps):
        total_w += int(idx_local.size)
    return total_v * total_w


def search_diagonal_matrix_two_rows(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    target_diag_npy: str,
    eps: float,
    output_prefix: str,
    max_row1_candidates: Optional[int] = None,
    max_matrix_results: int = 100,
    triples_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: str | None = None,
    disable_y_pruning: bool = False,
    reuse_row1_cache: bool = True,
    verbose: bool = False,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    target_diag = _load_target_diag_npy(target_diag_npy)
    d1, d2, d3 = target_diag
    row1_target = np.asarray([complex(d1), 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    row1_estimate = None

    try:
        from sage.all import CyclotomicField, ZZ
        K = CyclotomicField(9)
    except Exception as exc:
        raise RuntimeError("This script expects to run under Sage.") from exc

    if rank == 0:
        row1_estimate = estimate_row1_candidate_counts(
            triples_file=triples_file,
            triples_json=triples_json,
            rootdb_prefix=rootdb_prefix,
            f=f,
            target_vec=row1_target,
            eps=eps,
            triples_chunk_rows=triples_chunk_rows,
            n_phase_bins=n_phase_bins,
            chunk_meta_json=chunk_meta_json,
            disable_y_pruning=disable_y_pruning,
            verbose=verbose,
        )
        if verbose:
            print(json.dumps({
                "row1_candidate_estimate": row1_estimate,
            }, indent=2, sort_keys=True))

        row1_cands = None
        cached = _load_row1_candidates(output_prefix) if reuse_row1_cache else None
        if cached is not None and verbose:
            print(f"loaded row1 candidates from cache: {len(cached)}")
        need_rebuild = cached is None or (max_row1_candidates is not None and len(cached) < int(max_row1_candidates))
        if need_rebuild:
            if cached is not None and verbose:
                print(
                    f"cached row1 candidates ({len(cached)}) are fewer than requested max_row1_candidates="
                    f"{int(max_row1_candidates)}; rebuilding cache"
                )
            row1_cands = enumerate_vector_fits(
                triples_file=triples_file,
                triples_json=triples_json,
                rootdb_prefix=rootdb_prefix,
                f=f,
                target_vec=row1_target,
                eps=eps,
                max_candidates=max_row1_candidates,
                triples_chunk_rows=triples_chunk_rows,
                n_phase_bins=n_phase_bins,
                chunk_meta_json=chunk_meta_json,
                deduplicate=True,
                disable_y_pruning=disable_y_pruning,
                verbose=verbose,
            )
            _save_row1_candidates(output_prefix, row1_cands)
        else:
            row1_cands = cached if max_row1_candidates is None else cached[: int(max_row1_candidates)]
    else:
        row1_cands = None

    row1_cands = comm.bcast(row1_cands, root=0)
    n_row1 = len(row1_cands)
    local_row1_indices = _descending_partition(0, n_row1, size, rank)

    if rank == 0 and verbose:
        print(f"row1 candidates available = {n_row1}")
        print(f"searching over row1 candidates across {size} MPI ranks")

    # Load global triple/rootdb metadata on each rank.
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
        raise FileNotFoundError(f"Binned phase sidecar not found. Run build_phase_sidecar.py first for {rootdb_prefix}")
    phase_meta = _read_manifest(paths["phase_sidecar_meta"])
    chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
    batch_cache = OrderedDict()

    row1_coeffs = [np.asarray(rec["u_coeffs"], dtype=np.int64) for rec in row1_cands]
    row1_dists = [float(rec["dist"]) for rec in row1_cands]
    row1_Y = [np.asarray(rec["Y"], dtype=np.int64) for rec in row1_cands]

    total_row1 = len(local_row1_indices)
    row1_done = 0
    current_row1_idx = -1
    current_row1_dist = float("nan")
    pair_checks_done = 0
    orth_divisible = 0
    norm_match = 0
    same_layer_pass = 0
    local_results = []
    last_progress = time.time()

    start_row = 0
    chunk_iter = 0
    while start_row < nrows:
        take = min(int(triples_chunk_rows), nrows - start_row)
        triples = _read_triple_rows(triples_file, start_row, take)
        if rank == 0 and verbose:
            print(f"row2 search chunk rows {start_row:,}..{start_row + take - 1:,} of {nrows:,}")

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_iter]
            _, locator = _load_chunk_metadata(rec["path"])
        else:
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            locator = _build_locator(needed, paths)

        # Build row2 descriptors that depend only on the row2 targets (0,d2,0):
        # V near d2, W near 0. U is derived, so no descriptor for coord 0.
        desc_v_cache = {}
        desc_w_cache = {}
        pair_count_cache = {}
        for y in locator.keys():
            desc_v_cache[y] = _candidate_desc_from_binned_sidecar(
                y, locator, phase_meta, batch_cache, complex(d2), eps, f, profile=None,
                disable_y_pruning=disable_y_pruning,
            )
            desc_w_cache[y] = _candidate_desc_from_binned_sidecar(
                y, locator, phase_meta, batch_cache, 0.0 + 0.0j, eps, f, profile=None,
                disable_y_pruning=disable_y_pruning,
            )

        for local_pos, row1_idx in enumerate(local_row1_indices):
            row1_idx = int(row1_idx)
            coeffs1 = row1_coeffs[row1_idx]
            A = _coeffs6_to_sage_elem(coeffs1[0], K)
            B = _coeffs6_to_sage_elem(coeffs1[1], K)
            C = _coeffs6_to_sage_elem(coeffs1[2], K)

            current_row1_idx = row1_idx
            current_row1_dist = row1_dists[row1_idx]

            a = _coeffs6_to_complex_scaled(coeffs1[0], f)
            b = _coeffs6_to_complex_scaled(coeffs1[1], f)
            c = _coeffs6_to_complex_scaled(coeffs1[2], f)

            for tri in triples:
                Z1_t = (int(tri[0]), int(tri[1]), int(tri[2]))
                Z2_t = (int(tri[3]), int(tri[4]), int(tri[5]))
                Z3_t = (int(tri[6]), int(tri[7]), int(tri[8]))
                if Z1_t not in locator or Z2_t not in locator or Z3_t not in locator:
                    continue

                desc_v = desc_v_cache[Z2_t]
                desc_w = desc_w_cache[Z3_t]
                if desc_v is None or desc_w is None:
                    continue

                key_pair = (Z2_t, Z3_t)
                n_pair = pair_count_cache.get(key_pair)
                if n_pair is None:
                    n_pair = _row2_pair_count_desc(desc_v, desc_w, phase_meta, batch_cache, f, eps)
                    pair_count_cache[key_pair] = n_pair
                pair_checks_done += int(n_pair)

                # Enumerate candidate V near d2 and W near 0. U is forced by orthogonality.
                for base_v, idx_v_local, z_v in _iter_desc_z_and_indices(desc_v, phase_meta, batch_cache, f, eps):
                    for iv, zv in enumerate(z_v):
                        abs_v = int(base_v + idx_v_local[iv])
                        coeff_v = _load_coeff_row(desc_v[0], abs_v, phase_meta, batch_cache)
                        V = _coeffs6_to_sage_elem(coeff_v, K)

                        for base_w, idx_w_local, z_w in _iter_desc_z_and_indices(desc_w, phase_meta, batch_cache, f, eps):
                            for iw, zw in enumerate(z_w):
                                abs_w = int(base_w + idx_w_local[iw])
                                coeff_w = _load_coeff_row(desc_w[0], abs_w, phase_meta, batch_cache)
                                W = _coeffs6_to_sage_elem(coeff_w, K)

                                T = B * V.conjugate() + C * W.conjugate()
                                q = T / A if A != 0 else None
                                if q is None or not q.is_integral():
                                    continue
                                orth_divisible += 1

                                U = -q.conjugate()
                                U_norm = U * U.conjugate()
                                z1_expected = K(ZZ(Z1_t[0]) + ZZ(Z1_t[1]) * (K.gen() + K.gen() ** (-1)) + ZZ(Z1_t[2]) * (K.gen() + K.gen() ** (-1)) ** 2)
                                if U_norm != z1_expected:
                                    continue
                                # exact component target filter for u near 0
                                u_complex = complex(coeffs_to_complex_noscale(tuple(_k_to_coeffs6_int(U, K))) / (3 ** f))
                                if abs(u_complex) > (eps + 1e-15):
                                    continue
                                norm_match += 1

                                N1 = B * W - C * V
                                N2 = C * U - A * W
                                N3 = A * V - B * U
                                p3f = 3 ** int(f)
                                if not ((N1 / p3f).is_integral() and (N2 / p3f).is_integral() and (N3 / p3f).is_integral()):
                                    continue
                                same_layer_pass += 1

                                coeff_u = _k_to_coeffs6_int(U, K)
                                coeff_r3_1 = _k_to_coeffs6_int(N1.conjugate() / p3f, K)
                                coeff_r3_2 = _k_to_coeffs6_int(N2.conjugate() / p3f, K)
                                coeff_r3_3 = _k_to_coeffs6_int(N3.conjugate() / p3f, K)
                                row2_coeffs = np.stack([coeff_u, coeff_v, coeff_w], axis=0)
                                row3_coeffs = np.stack([coeff_r3_1, coeff_r3_2, coeff_r3_3], axis=0)

                                u = _coeffs6_to_complex_scaled(coeff_u, f)
                                v = _coeffs6_to_complex_scaled(coeff_v, f)
                                w = _coeffs6_to_complex_scaled(coeff_w, f)
                                r3 = (
                                    _coeffs6_to_complex_scaled(coeff_r3_1, f),
                                    _coeffs6_to_complex_scaled(coeff_r3_2, f),
                                    _coeffs6_to_complex_scaled(coeff_r3_3, f),
                                )
                                row2_dist = _vector_dist(u, v, w, 0.0 + 0.0j, complex(d2), 0.0 + 0.0j)
                                frob = _frobenius_dist_from_rows((a, b, c), (u, v, w), r3, complex(d1), complex(d2), complex(d3))

                                local_results.append({
                                    "frobenius_dist": float(frob),
                                    "row1_index": int(row1_idx),
                                    "row1_dist": float(row1_dists[row1_idx]),
                                    "row2_dist": float(row2_dist),
                                    "row1_u_coeffs": np.asarray(coeffs1, dtype=np.int64),
                                    "row2_u_coeffs": np.asarray(row2_coeffs, dtype=np.int64),
                                    "row3_u_coeffs": np.asarray(row3_coeffs, dtype=np.int64),
                                    "row1_Y": np.asarray(row1_Y[row1_idx], dtype=np.int64),
                                    "row2_Y": np.asarray([Z1_t, Z2_t, Z3_t], dtype=np.int64),
                                })

            row1_done += 1
            current_row1_idx = -1
            current_row1_dist = float("nan")

            payload = {
                "row1_done": int(row1_done),
                "pair_checks_done": int(pair_checks_done),
                "orth_divisible": int(orth_divisible),
                "norm_match": int(norm_match),
                "same_layer_pass": int(same_layer_pass),
                "current_row1_idx": int(current_row1_idx),
                "current_row1_dist": float(current_row1_dist),
                "results": int(len(local_results)),
            }
            all_payload = comm.gather(payload, root=0)
            if rank == 0 and verbose:
                now = time.time()
                if (now - last_progress) >= 10.0 or row1_done == total_row1:
                    total_done = sum(p["row1_done"] for p in all_payload)
                    pair_done = sum(p["pair_checks_done"] for p in all_payload)
                    orth_ok = sum(p["orth_divisible"] for p in all_payload)
                    norm_ok = sum(p["norm_match"] for p in all_payload)
                    layer_ok = sum(p["same_layer_pass"] for p in all_payload)
                    frontier_idx = -1
                    frontier_dist = float("nan")
                    active = [p for p in all_payload if p["current_row1_idx"] >= 0]
                    if active:
                        frontier = min(active, key=lambda p: p["current_row1_idx"])
                        frontier_idx = int(frontier["current_row1_idx"])
                        frontier_dist = float(frontier["current_row1_dist"])
                    remaining = max(0, n_row1 - total_done)
                    print(
                        f"progress: row1 done {total_done}/{n_row1} | row2 pair checks done {pair_done} | "
                        f"orth ok {orth_ok} | norm ok {norm_ok} | same-layer ok {layer_ok} | "
                        f"results {sum(p['results'] for p in all_payload)} | "
                        f"frontier row1 idx={frontier_idx}, dist={frontier_dist:.12g} | remaining row1 ~{remaining}",
                        flush=True,
                    )
                    last_progress = now
        start_row += take
        chunk_iter += 1

    gathered = comm.gather(local_results, root=0)
    if rank == 0:
        merged = []
        for part in gathered:
            merged.extend(part)
        merged = _merge_top_results(merged, max_matrix_results)
        _save_results(output_prefix, merged)
        summary = {
            "row1_candidate_estimate": row1_estimate,
            "n_row1_candidates": int(n_row1),
            "n_matrix_results": int(len(merged)),
            "results_npz": os.path.abspath(output_prefix + ".results.npz"),
            "results_json": os.path.abspath(output_prefix + ".results.json"),
            "row1_candidates_npz": os.path.abspath(output_prefix + ".row1_candidates.npz"),
            "row1_candidates_json": os.path.abspath(output_prefix + ".row1_candidates.json"),
            "summary_json": os.path.abspath(output_prefix + ".summary.json"),
            "target_diag_npy": os.path.abspath(target_diag_npy),
            "mpi_ranks": int(size),
            "disable_y_pruning": bool(disable_y_pruning),
        }
        _write_manifest(output_prefix + ".summary.json", summary)
        return summary
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--triples_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--target_diag_npy", required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--max_row1_candidates", type=int, default=None)
    parser.add_argument("--max_matrix_results", type=int, default=100)
    parser.add_argument("--triples_chunk_rows", type=int, default=200000)
    parser.add_argument("--n_phase_bins", type=int, default=512)
    parser.add_argument("--chunk_meta_json", default=None)
    parser.add_argument("--disable_y_pruning", action="store_true")
    parser.add_argument("--no_reuse_row1_cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = search_diagonal_matrix_two_rows(
        triples_file=args.triples_file,
        triples_json=args.triples_json,
        rootdb_prefix=args.rootdb_prefix,
        f=args.f,
        target_diag_npy=args.target_diag_npy,
        eps=args.eps,
        output_prefix=args.output_prefix,
        max_row1_candidates=args.max_row1_candidates,
        max_matrix_results=args.max_matrix_results,
        triples_chunk_rows=args.triples_chunk_rows,
        n_phase_bins=args.n_phase_bins,
        chunk_meta_json=args.chunk_meta_json,
        disable_y_pruning=args.disable_y_pruning,
        reuse_row1_cache=not args.no_reuse_row1_cache,
        verbose=not args.quiet,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and not args.quiet:
        print(json.dumps(out, indent=2, sort_keys=True))
