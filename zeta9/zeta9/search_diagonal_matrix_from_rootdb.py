"""Search diagonal-matrix completions using vector fits from fit_vectors.

This module is intended to live alongside fit_vectors_mpi_sidecar_binned.py and
reuse its rootdb / sidecar / chunk-streaming helpers.

Workflow
--------
1. Enumerate *all* vector fits x=(x1,x2,x3) for the diagonal-target vector
      (d1, 0, 0)
   with the per-component tolerance |x_i - u_i| <= eps.
2. Sort those vector fits by Euclidean vector distance.
3. For a diagonal target matrix diag(d1,d2,d3), try pairs
      (first column = candidate n, first row = candidate k),  k <= n
   only once per unordered pair.
4. For each pair with matching V11, complete the matrix using the built-in
   admissible_unitary_completions_zeta9().
5. Rank successful matrices by Frobenius norm to the target diagonal matrix.

Extras
------
* MPI parallelization is used for the matrix-pairing stage, partitioned over the
  first-column index n.
* The pairing search is resumable/extendable: after searching columns [0,n1), a
  later run can search [n1,n2) without redoing [0,n1).

Notes
-----
* This code expects to run under Sage.
* It imports internal helpers from fit_vectors_mpi_sidecar_binned.py.
"""

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

from fit_vectors_mpi_sidecar_binned import (
    _build_locator,
    _candidate_desc_from_binned_sidecar,
    _iter_desc_z_and_indices,
    _load_chunk_metadata,
    _load_coeff_row,
    _read_manifest,
    _read_triple_rows,
    _state_paths,
)


# ----------------------------
# Basic helpers
# ----------------------------

def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _vector_dist(z0: complex, z1: complex, z2: complex, d0: complex, d1: complex, d2: complex) -> float:
    return math.sqrt(abs(z0 - d0) ** 2 + abs(z1 - d1) ** 2 + abs(z2 - d2) ** 2)



def _frobenius_dist_diag_from_entries(a, b, c, d, x, y, g, z, w, d1, d2, d3) -> float:
    return math.sqrt(
        abs(a - d1) ** 2 + abs(b) ** 2 + abs(c) ** 2
        + abs(d) ** 2 + abs(x - d2) ** 2 + abs(y) ** 2
        + abs(g) ** 2 + abs(z) ** 2 + abs(w - d3) ** 2
    )



def _coeffs6_to_sage_elem(coeffs6, K):
    z = K.gen()
    out = K(0)
    for k, ck in enumerate(coeffs6):
        out += K(int(ck)) * (z ** k)
    return out



def _candidate_key_from_coeffs(coeffs_3x6: np.ndarray):
    return tuple(int(v) for v in coeffs_3x6.reshape(-1))



def _matrix_key_from_coeffs(row_u_coeffs: np.ndarray, col_u_coeffs: np.ndarray, t_index: int):
    return (
        tuple(int(v) for v in row_u_coeffs.reshape(-1)),
        tuple(int(v) for v in col_u_coeffs.reshape(-1)),
        int(t_index),
    )



def _result_sort_key(rec: dict):
    return (
        float(rec["frobenius_dist"]),
        float(rec["row_vector_dist"]),
        float(rec["col_vector_dist"]),
        int(rec["row_candidate_index"]),
        int(rec["col_candidate_index"]),
        int(rec["t_index"]),
    )



def _merge_top_results(existing: list[dict], new_items: list[dict], limit: int) -> list[dict]:
    by_key = {}
    for rec in existing:
        by_key[_matrix_key_from_coeffs(rec["row_u_coeffs"], rec["col_u_coeffs"], rec["t_index"])] = rec
    for rec in new_items:
        key = _matrix_key_from_coeffs(rec["row_u_coeffs"], rec["col_u_coeffs"], rec["t_index"])
        prev = by_key.get(key)
        if prev is None or _result_sort_key(rec) < _result_sort_key(prev):
            by_key[key] = rec
    out = list(by_key.values())
    out.sort(key=_result_sort_key)
    return out[: int(limit)]


# ----------------------------
# Persistence helpers
# ----------------------------

def _vector_cache_paths(output_prefix: str):
    return {
        "npz": output_prefix + ".vector_candidates.npz",
        "json": output_prefix + ".vector_candidates.json",
    }



def _search_state_paths(output_prefix: str):
    return {
        "npz": output_prefix + ".search_state.npz",
        "json": output_prefix + ".search_state.json",
    }



def _contiguous_prefix_upto(processed_cols: np.ndarray) -> int:
    """
    Given a sorted unique array of processed first-column indices, return the
    smallest n such that all columns [0, n) have been processed.
    """
    processed_cols = np.asarray(processed_cols, dtype=np.int64)
    if processed_cols.size == 0:
        return 0
    upto = 0
    for v in processed_cols:
        iv = int(v)
        if iv != upto:
            break
        upto += 1
    return upto



def _save_vector_candidates(output_prefix: str, vec_cands: list[dict]):
    paths = _vector_cache_paths(output_prefix)
    n = len(vec_cands)
    if n == 0:
        np.savez(paths["npz"], dist=np.empty((0,), dtype=np.float64), z=np.empty((0, 3), dtype=np.complex128),
                 u_coeffs=np.empty((0, 3, 6), dtype=np.int64), Y=np.empty((0, 3, 3), dtype=np.int64))
    else:
        np.savez(
            paths["npz"],
            dist=np.asarray([r["dist"] for r in vec_cands], dtype=np.float64),
            z=np.asarray([r["z"] for r in vec_cands], dtype=np.complex128),
            u_coeffs=np.asarray([r["u_coeffs"] for r in vec_cands], dtype=np.int64),
            Y=np.asarray([r["Y"] for r in vec_cands], dtype=np.int64),
        )
    _write_manifest(paths["json"], {"n_candidates": int(n), "npz": os.path.abspath(paths["npz"])})



def _load_vector_candidates(output_prefix: str) -> Optional[list[dict]]:
    paths = _vector_cache_paths(output_prefix)
    if not os.path.exists(paths["npz"]):
        return None
    data = np.load(paths["npz"], allow_pickle=False)
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



def _save_search_state(
    output_prefix: str,
    results: list[dict],
    searched_upto: int,
    n_candidates_total: int,
    processed_cols: Optional[Sequence[int]] = None,
):
    paths = _search_state_paths(output_prefix)
    n = len(results)
    processed_cols_arr = np.unique(np.asarray(processed_cols if processed_cols is not None else [], dtype=np.int64))
    if n == 0:
        np.savez(
            paths["npz"],
            frobenius_dist=np.empty((0,), dtype=np.float64),
            t_index=np.empty((0,), dtype=np.int64),
            row_candidate_index=np.empty((0,), dtype=np.int64),
            col_candidate_index=np.empty((0,), dtype=np.int64),
            row_vector_dist=np.empty((0,), dtype=np.float64),
            col_vector_dist=np.empty((0,), dtype=np.float64),
            row_u_coeffs=np.empty((0, 3, 6), dtype=np.int64),
            col_u_coeffs=np.empty((0, 3, 6), dtype=np.int64),
            row_Y=np.empty((0, 3, 3), dtype=np.int64),
            col_Y=np.empty((0, 3, 3), dtype=np.int64),
            processed_cols=processed_cols_arr,
        )
    else:
        np.savez(
            paths["npz"],
            frobenius_dist=np.asarray([r["frobenius_dist"] for r in results], dtype=np.float64),
            t_index=np.asarray([r["t_index"] for r in results], dtype=np.int64),
            row_candidate_index=np.asarray([r["row_candidate_index"] for r in results], dtype=np.int64),
            col_candidate_index=np.asarray([r["col_candidate_index"] for r in results], dtype=np.int64),
            row_vector_dist=np.asarray([r["row_vector_dist"] for r in results], dtype=np.float64),
            col_vector_dist=np.asarray([r["col_vector_dist"] for r in results], dtype=np.float64),
            row_u_coeffs=np.asarray([r["row_u_coeffs"] for r in results], dtype=np.int64),
            col_u_coeffs=np.asarray([r["col_u_coeffs"] for r in results], dtype=np.int64),
            row_Y=np.asarray([r["row_Y"] for r in results], dtype=np.int64),
            col_Y=np.asarray([r["col_Y"] for r in results], dtype=np.int64),
            processed_cols=processed_cols_arr,
        )
    _write_manifest(paths["json"], {
        "searched_upto": int(searched_upto),
        "n_candidates_total": int(n_candidates_total),
        "n_results": int(n),
        "n_processed_cols": int(processed_cols_arr.size),
        "processed_cols_preview": [int(v) for v in processed_cols_arr[:32]],
        "processed_cols_tail_preview": [int(v) for v in processed_cols_arr[-32:]],
        "npz": os.path.abspath(paths["npz"]),
    })



def _load_search_state(output_prefix: str) -> tuple[list[dict], int, int, np.ndarray] | tuple[None, None, None, None]:
    paths = _search_state_paths(output_prefix)
    if not os.path.exists(paths["npz"]) or not os.path.exists(paths["json"]):
        return None, None, None, None
    meta = _read_manifest(paths["json"])
    data = np.load(paths["npz"], allow_pickle=False)
    n = int(np.asarray(data["frobenius_dist"]).shape[0])
    out = []
    for i in range(n):
        out.append({
            "frobenius_dist": float(data["frobenius_dist"][i]),
            "t_index": int(data["t_index"][i]),
            "row_candidate_index": int(data["row_candidate_index"][i]),
            "col_candidate_index": int(data["col_candidate_index"][i]),
            "row_vector_dist": float(data["row_vector_dist"][i]),
            "col_vector_dist": float(data["col_vector_dist"][i]),
            "row_u_coeffs": np.asarray(data["row_u_coeffs"][i], dtype=np.int64),
            "col_u_coeffs": np.asarray(data["col_u_coeffs"][i], dtype=np.int64),
            "row_Y": np.asarray(data["row_Y"][i], dtype=np.int64),
            "col_Y": np.asarray(data["col_Y"][i], dtype=np.int64),
        })
    if "processed_cols" in data.files:
        processed_cols = np.unique(np.asarray(data["processed_cols"], dtype=np.int64))
    else:
        processed_cols = np.arange(int(meta.get("searched_upto", 0)), dtype=np.int64)
    return out, int(meta.get("searched_upto", 0)), int(meta.get("n_candidates_total", 0)), processed_cols


# ----------------------------
# Unitary completion routine
# ----------------------------

def admissible_unitary_completions_zeta9(A, B, C, D, G, f, check_normalization=True, verbose=False):
    try:
        from sage.all import CyclotomicField, Matrix, identity_matrix, ZZ
    except Exception as exc:
        raise RuntimeError("admissible_unitary_completions_zeta9 expects to run under Sage.") from exc

    K = CyclotomicField(9)
    zeta = K.gen()
    R = K.ring_of_integers()

    A = K(A)
    B = K(B)
    C = K(C)
    D = K(D)
    G = K(G)

    def bar(x):
        return x.conjugate()

    threef = ZZ(3) ** ZZ(f)
    M = B * bar(B) + C * bar(C)

    if check_normalization:
        lhs_row = A * bar(A) + B * bar(B) + C * bar(C)
        lhs_col = A * bar(A) + D * bar(D) + G * bar(G)
        target = ZZ(3) ** (2 * ZZ(f))
        if lhs_row != target:
            raise ValueError(f"Top row is not normalized: got {lhs_row}, expected {target}")
        if lhs_col != target:
            raise ValueError(f"First column is not normalized: got {lhs_col}, expected {target}")
        if M != D * bar(D) + G * bar(G):
            raise ValueError("Inconsistent M: B*Bbar+C*Cbar != D*Dbar+G*Gbar")

    def divisible_by_M(x):
        return (x / M).is_integral()

    mu18 = []
    seen = set()
    for k in range(9):
        for eps_sign in [1, -1]:
            t = K(eps_sign) * (zeta ** k)
            key = tuple(t.vector())
            if key not in seen:
                seen.add(key)
                mu18.append(t)

    solutions = []
    for t_index, t in enumerate(mu18):
        s = threef * t
        N1 = -bar(A) * D * B + s * G * bar(C)
        N2 = -bar(A) * D * C - s * G * bar(B)
        N3 = -bar(A) * G * B - s * D * bar(C)
        N4 = -bar(A) * G * C + s * D * bar(B)

        ok1 = divisible_by_M(N1)
        ok2 = divisible_by_M(N2)
        ok3 = divisible_by_M(N3)
        ok4 = divisible_by_M(N4)

        if verbose:
            print(f"t = {t}")
            print(f"  divisibility = {(ok1, ok2, ok3, ok4)}")

        if ok1 and ok2 and ok3 and ok4:
            X = N1 / M
            Y = N2 / M
            Z = N3 / M
            W = N4 / M
            if not (X.is_integral() and Y.is_integral() and Z.is_integral() and W.is_integral()):
                continue

            x = X / threef
            y = Y / threef
            z = Z / threef
            w = W / threef

            a = A / threef
            b = B / threef
            c = C / threef
            d = D / threef
            g = G / threef
            alpha2 = 1 - a * bar(a)

            x2 = (-bar(a) * d * b + t * g * bar(c)) / alpha2
            y2 = (-bar(a) * d * c - t * g * bar(b)) / alpha2
            z2 = (-bar(a) * g * b - t * d * bar(c)) / alpha2
            w2 = (-bar(a) * g * c + t * d * bar(b)) / alpha2
            if x != x2 or y != y2 or z != z2 or w != w2:
                raise ValueError("Formula consistency check failed")

            V = Matrix(K, [[a, b, c], [d, x, y], [g, z, w]])
            if V * V.conjugate_transpose() != identity_matrix(K, 3):
                raise ValueError("Computed completion is not unitary")

            solutions.append({
                "t": t,
                "t_index": int(t_index),
                "s": s,
                "x": x,
                "y": y,
                "z": z,
                "w": w,
                "X": X,
                "Y": Y,
                "Z": Z,
                "W": W,
                "numerators": (N1, N2, N3, N4),
                "matrix": V,
            })

    return {"field": K, "ring": R, "zeta": zeta, "M": M, "solutions": solutions}


# ----------------------------
# Enumerate vector fits
# ----------------------------

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
        raise FileNotFoundError(f"Binned phase sidecar not found. Run build_phase_sidecar.py first for {rootdb_prefix}")

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
            print(f"enumerate chunk rows {start:,}..{start + take - 1:,} of {nrows:,}")

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
                    y, locator, phase_meta, batch_cache, target_vec[coord], eps, f, profile=None
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
                if z1.size == 0:
                    continue
                for a_idx, za in enumerate(z1):
                    abs_a = int(base1 + idx1_local[a_idx])
                    coeff_a = _load_coeff_row(desc1[0], abs_a, phase_meta, batch_cache)
                    for base2, idx2_local, z2 in _iter_desc_z_and_indices(desc2, phase_meta, batch_cache, f, eps):
                        if z2.size == 0:
                            continue
                        for b_idx, zb in enumerate(z2):
                            abs_b = int(base2 + idx2_local[b_idx])
                            coeff_b = _load_coeff_row(desc2[0], abs_b, phase_meta, batch_cache)
                            for base3, idx3_local, z3 in _iter_desc_z_and_indices(desc3, phase_meta, batch_cache, f, eps):
                                if z3.size == 0:
                                    continue
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


# ----------------------------
# MPI helpers
# ----------------------------

def _descending_partition(col_start: int, col_stop: int, size: int, rank: int) -> list[int]:
    if col_stop <= col_start:
        return []
    seq = list(range(col_stop - 1, col_start - 1, -1))
    return seq[rank::size]


def estimate_vector_candidate_counts(
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
    verbose: bool = False,
):
    """
    Estimate/count the number of possible first-column vectors.

    Returns counts with multiplicity over triples:
      - unconstrained_root_product_count = sum r(Y1) r(Y2) r(Y3)
      - filtered_root_product_count      = sum r_eps(Y1) r_eps(Y2) r_eps(Y3)

    Here r(Y) is the total number of exact roots for Y in the rootdb, and
    r_eps(Y_i) counts roots that satisfy the exact per-component filter for the
    given target component. These are counts before any vector deduplication.

    For choosing max_vector_candidates, the most relevant quantity is the exact
    number of unique filtered vectors after deduplication. That exact count is
    obtained separately by enumerate_vector_fits(..., deduplicate=True,
    max_candidates=None), because it requires hashing the actual coefficient
    triples rather than only summing multiplicities over triples.
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
        raise FileNotFoundError(f"Binned phase sidecar not found. Run build_phase_sidecar.py first for {rootdb_prefix}")

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
        bd = batch_cache.get((batch_idx, 'counts'))
        if bd is None:
            rec = phase_meta["batches"][batch_idx]
            off = np.load(rec["roots_off"], mmap_mode="r")
            batch_cache[(batch_idx, 'counts')] = off
        else:
            off = bd
        cnt = int(off[row_idx + 1] - off[row_idx])
        root_count_cache[key] = cnt
        return cnt

    def get_filtered_count(y, coord, locator):
        key = (tuple(int(v) for v in y), int(coord))
        val = filtered_count_cache.get(key)
        if val is not None:
            return val
        desc = _candidate_desc_from_binned_sidecar(
            key[0], locator, phase_meta, batch_cache, target_vec[coord], eps, f, profile=None
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
            print(f"estimate chunk rows {start:,}..{start + take - 1:,} of {nrows:,}")

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


# ----------------------------
# Matrix search driver
# ----------------------------

def search_diagonal_matrix_from_rootdb(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    target_diag,
    eps: float,
    max_vector_candidates: Optional[int] = None,
    max_matrix_results: int = 100,
    triples_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: str | None = None,
    deduplicate_vectors: bool = True,
    output_prefix: Optional[str] = None,
    reuse_vector_cache: bool = True,
    resume: bool = True,
    col_start: Optional[int] = None,
    col_stop: Optional[int] = None,
    progress_every_cols: int = 10,
    verbose: bool = False,
):
    """Search diagonal-matrix completions from rootdb-enumerated vector fits.

    Parameters of interest
    ----------------------
    max_vector_candidates:
        Truncate the sorted vector-candidate list to this prefix length before pairing.
    output_prefix:
        If provided, save vector candidates and search state to disk.
    resume:
        If True and a prior search_state exists, continue from searched_upto automatically.
    col_start, col_stop:
        Search the first-column range [col_start, col_stop). If omitted, defaults to
        [loaded searched_upto or 0, max_vector_candidates or n_candidates].

    MPI behavior
    ------------
    Vector enumeration is done on rank 0 and broadcast to all ranks.
    Matrix pairing over the first-column index n is distributed across ranks, using
    a largest-to-smallest interleaving to balance work better.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(target_diag) != 3:
        raise ValueError("target_diag must have length 3")

    try:
        from sage.all import CyclotomicField
        K = CyclotomicField(9)
    except Exception as exc:
        raise RuntimeError("This routine expects to run under Sage.") from exc

    d1, d2, d3 = target_diag
    target_vec = np.asarray([complex(d1), 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    loaded_results = []
    loaded_upto = 0
    loaded_n_candidates_total = 0
    loaded_processed_cols = np.empty((0,), dtype=np.int64)

    if rank == 0:
        vec_cands = None
        cached_vec_cands = None
        if output_prefix and reuse_vector_cache:
            cached_vec_cands = _load_vector_candidates(output_prefix)
            if cached_vec_cands is not None and verbose:
                print(f"loaded vector candidates from cache: {len(cached_vec_cands)}")

        need_reenumeration = False
        if cached_vec_cands is None:
            need_reenumeration = True
        elif max_vector_candidates is None:
            # Caller wants all available candidates. A truncated cache is not enough.
            need_reenumeration = False
            vec_cands = cached_vec_cands
        elif len(cached_vec_cands) < int(max_vector_candidates):
            # Existing cache is too small for the newly requested search horizon.
            need_reenumeration = True
        else:
            vec_cands = cached_vec_cands[: int(max_vector_candidates)]

        if need_reenumeration:
            if verbose and cached_vec_cands is not None and max_vector_candidates is not None:
                print(
                    f"cached vector candidates ({len(cached_vec_cands)}) are fewer than requested "
                    f"max_vector_candidates={int(max_vector_candidates)}; rebuilding cache"
                )
            vec_cands = enumerate_vector_fits(
                triples_file=triples_file,
                triples_json=triples_json,
                rootdb_prefix=rootdb_prefix,
                f=f,
                target_vec=target_vec,
                eps=eps,
                max_candidates=max_vector_candidates,
                triples_chunk_rows=triples_chunk_rows,
                n_phase_bins=n_phase_bins,
                chunk_meta_json=chunk_meta_json,
                deduplicate=deduplicate_vectors,
                verbose=verbose,
            )
            if output_prefix:
                _save_vector_candidates(output_prefix, vec_cands)

        if output_prefix and resume:
            prev_results, prev_upto, prev_n_total, prev_processed_cols = _load_search_state(output_prefix)
            if prev_results is not None:
                loaded_results = prev_results
                loaded_upto = int(prev_upto)
                loaded_n_candidates_total = int(prev_n_total)
                loaded_processed_cols = np.unique(np.asarray(prev_processed_cols, dtype=np.int64))
                if verbose:
                    print(
                        f"loaded prior search state: searched_upto={loaded_upto}, "
                        f"processed_cols={loaded_processed_cols.size}, results={len(loaded_results)}"
                    )
    else:
        vec_cands = None

    vec_cands = comm.bcast(vec_cands, root=0)
    loaded_results = comm.bcast(loaded_results if rank == 0 else None, root=0)
    loaded_upto = comm.bcast(loaded_upto if rank == 0 else None, root=0)
    loaded_n_candidates_total = comm.bcast(loaded_n_candidates_total if rank == 0 else None, root=0)
    loaded_processed_cols = comm.bcast(loaded_processed_cols if rank == 0 else None, root=0)

    n_candidates = len(vec_cands)
    if max_vector_candidates is None:
        effective_n = n_candidates
    else:
        effective_n = min(int(max_vector_candidates), n_candidates)
        vec_cands = vec_cands[:effective_n]
        n_candidates = effective_n

    if col_start is None:
        col_start = int(_contiguous_prefix_upto(loaded_processed_cols)) if resume else 0
    else:
        col_start = int(col_start)
    if col_stop is None:
        col_stop = n_candidates
    else:
        col_stop = min(int(col_stop), n_candidates)

    if col_start < 0 or col_start > col_stop or col_stop > n_candidates:
        raise ValueError(f"Invalid column range: [{col_start}, {col_stop}) for n_candidates={n_candidates}")

    local_cols = _descending_partition(col_start, col_stop, size, rank)
    if verbose and rank == 0:
        print(f"vector candidates available = {n_candidates}")
        print(f"searching column range [{col_start}, {col_stop}) over {size} MPI ranks")

    total_pair_checks = int(sum((n + 1) for n in range(col_start, col_stop)))

    def coeffs3x6_to_layer_vec(coeffs3x6: np.ndarray):
        return tuple(_coeffs6_to_sage_elem(coeffs3x6[i], K) / (3 ** f) for i in range(3))

    local_results = []
    local_pair_checks_done = 0
    local_processed_cols_done = 0
    local_current_n = -1
    local_current_dist = float("nan")
    max_local_steps = comm.allreduce(len(local_cols), op=MPI.MAX)
    last_progress_print = time.time()

    for step in range(max_local_steps):
        if step < len(local_cols):
            n = int(local_cols[step])
            col_rec = vec_cands[n]
            col_vec = coeffs3x6_to_layer_vec(col_rec["u_coeffs"])
            local_current_n = n
            local_current_dist = float(col_rec["dist"])

            for k in range(n + 1):
                row_rec = vec_cands[k]
                row_vec = coeffs3x6_to_layer_vec(row_rec["u_coeffs"])
                if row_vec[0] != col_vec[0]:
                    continue

                a = row_vec[0]
                A = _coeffs6_to_sage_elem(row_rec["u_coeffs"][0], K)
                B = _coeffs6_to_sage_elem(row_rec["u_coeffs"][1], K)
                C = _coeffs6_to_sage_elem(row_rec["u_coeffs"][2], K)
                D = _coeffs6_to_sage_elem(col_rec["u_coeffs"][1], K)
                G = _coeffs6_to_sage_elem(col_rec["u_coeffs"][2], K)

                comp = admissible_unitary_completions_zeta9(
                    A=A, B=B, C=C, D=D, G=G, f=f,
                    check_normalization=True,
                    verbose=False,
                )

                b = row_vec[1]
                c = row_vec[2]
                d = col_vec[1]
                g = col_vec[2]

                for sol in comp["solutions"]:
                    x = sol["x"]
                    y = sol["y"]
                    z = sol["z"]
                    w = sol["w"]
                    frob = _frobenius_dist_diag_from_entries(a, b, c, d, x, y, g, z, w, d1, d2, d3)
                    local_results.append({
                        "frobenius_dist": float(frob),
                        "t_index": int(sol["t_index"]),
                        "row_candidate_index": int(k),
                        "col_candidate_index": int(n),
                        "row_vector_dist": float(row_rec["dist"]),
                        "col_vector_dist": float(col_rec["dist"]),
                        "row_u_coeffs": np.asarray(row_rec["u_coeffs"], dtype=np.int64),
                        "col_u_coeffs": np.asarray(col_rec["u_coeffs"], dtype=np.int64),
                        "row_Y": np.asarray(row_rec["Y"], dtype=np.int64),
                        "col_Y": np.asarray(col_rec["Y"], dtype=np.int64),
                    })

            local_pair_checks_done += int(n + 1)
            local_processed_cols_done += 1
            local_current_n = -1
            local_current_dist = float("nan")

        progress_payload = {
            "pairs_done": int(local_pair_checks_done),
            "cols_done": int(local_processed_cols_done),
            "current_n": int(local_current_n),
            "current_dist": float(local_current_dist),
            "results": int(len(local_results)),
        }
        progress_all = comm.gather(progress_payload, root=0)
        if rank == 0 and verbose:
            now = time.time()
            if (now - last_progress_print) >= 10.0 or step == max_local_steps - 1:
                pairs_done = int(sum(p["pairs_done"] for p in progress_all))
                cols_done = int(sum(p["cols_done"] for p in progress_all))
                active = [p for p in progress_all if p["current_n"] >= 0]
                if active:
                    frontier = min(active, key=lambda p: p["current_n"])
                    frontier_n = int(frontier["current_n"])
                    frontier_dist = float(frontier["current_dist"])
                elif cols_done > 0:
                    done_cols = []
                    for r in range(size):
                        done_cols.extend(int(v) for v in _descending_partition(col_start, col_stop, size, r)[: progress_all[r]["cols_done"]])
                    frontier_n = min(done_cols) if done_cols else -1
                    frontier_dist = float(vec_cands[frontier_n]["dist"]) if frontier_n >= 0 else float("nan")
                else:
                    frontier_n = max(col_start, col_stop - 1) if col_start < col_stop else -1
                    frontier_dist = float(vec_cands[frontier_n]["dist"]) if frontier_n >= 0 else float("nan")
                pairs_left = max(0, total_pair_checks - pairs_done)
                print(
                    f"progress: cols done {cols_done}/{col_stop - col_start} | "
                    f"pair checks done {pairs_done}/{total_pair_checks}, remaining ~{pairs_left} | "
                    f"results {sum(p['results'] for p in progress_all)} | "
                    f"frontier col n={frontier_n}, dist={frontier_dist:.12g}",
                    flush=True,
                )
                last_progress_print = now

    gathered = comm.gather({"results": local_results, "processed_cols": np.asarray(local_cols, dtype=np.int64)}, root=0)

    if rank == 0:
        merged = list(loaded_results)
        processed_cols_merged = np.unique(np.asarray(loaded_processed_cols, dtype=np.int64))
        for part in gathered:
            merged = _merge_top_results(merged, part["results"], max_matrix_results)
            processed_cols_merged = np.unique(
                np.concatenate([processed_cols_merged, np.asarray(part["processed_cols"], dtype=np.int64)])
            )

        searched_upto_new = _contiguous_prefix_upto(processed_cols_merged)
        if output_prefix:
            _save_search_state(output_prefix, merged, searched_upto_new, n_candidates, processed_cols_merged)

        return {
            "vector_candidates": vec_cands,
            "matrix_results": merged,
            "searched_range": (int(col_start), int(col_stop)),
            "searched_upto": int(searched_upto_new),
            "n_candidates": int(n_candidates),
            "n_processed_cols": int(processed_cols_merged.size),
            "processed_cols": processed_cols_merged,
            "used_cached_search_state": bool(resume and loaded_processed_cols.size > 0),
        }
    return None


def _coerce_target_diag_value(v):
    """
    Coerce a target-diagonal entry from CLI/JSON/NPY input to a plain Python
    complex number.

    The target matrix is not assumed to be cyclotomic; it is just a complex
    unitary diagonal target. Only the approximation candidates and completed
    matrices live in Q(zeta9).

    Accepted forms:
      - Python/NumPy real or complex numeric values
      - strings representing numerical expressions, evaluated in Sage's complex
        field, e.g. "0.9510565162951535-0.3090169943749474j",
        "exp(-I*pi/10)", "CC(exp(-I*pi/10))"
    """
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


def _load_target_diag_from_args(args):
    try:
        from sage.all import CC  # noqa: F401
    except Exception as exc:
        raise RuntimeError("This script expects to run under Sage.") from exc

    if args.target_diag_npy is not None:
        arr = np.load(args.target_diag_npy, allow_pickle=True)
        arr = np.asarray(arr)
        if arr.shape != (3,):
            raise ValueError("target_diag_npy must have shape (3,)")
        return tuple(_coerce_target_diag_value(v) for v in arr.tolist())

    raise ValueError("Provide target diagonal via --target_diag_npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--triples_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--target_diag_npy", required=True,
                        help="NumPy file of shape (3,) for the target diagonal")
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--max_vector_candidates", type=int, default=None)
    parser.add_argument("--max_matrix_results", type=int, default=100)
    parser.add_argument("--triples_chunk_rows", type=int, default=200000)
    parser.add_argument("--n_phase_bins", type=int, default=512)
    parser.add_argument("--chunk_meta_json", default=None)
    parser.add_argument("--col_start", type=int, default=None)
    parser.add_argument("--col_stop", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--no_reuse_vector_cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--estimate_only", action="store_true",
                        help="Only estimate counts of possible first-column vectors; do not run matrix pairing")
    args = parser.parse_args()

    target_diag = _load_target_diag_from_args(args)

    if args.estimate_only:
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            target_vec = np.asarray([complex(target_diag[0]), 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
            estimate = estimate_vector_candidate_counts(
                triples_file=args.triples_file,
                triples_json=args.triples_json,
                rootdb_prefix=args.rootdb_prefix,
                f=args.f,
                target_vec=target_vec,
                eps=args.eps,
                triples_chunk_rows=args.triples_chunk_rows,
                n_phase_bins=args.n_phase_bins,
                chunk_meta_json=args.chunk_meta_json,
                verbose=not args.quiet,
            )

            if not args.quiet:
                print("counting exact unique filtered vectors after deduplication...")
            exact_vec_cands = enumerate_vector_fits(
                triples_file=args.triples_file,
                triples_json=args.triples_json,
                rootdb_prefix=args.rootdb_prefix,
                f=args.f,
                target_vec=target_vec,
                eps=args.eps,
                max_candidates=None,
                triples_chunk_rows=args.triples_chunk_rows,
                n_phase_bins=args.n_phase_bins,
                chunk_meta_json=args.chunk_meta_json,
                deduplicate=True,
                verbose=not args.quiet,
            )
            exact_unique_filtered_vector_count = int(len(exact_vec_cands))
            last_vector_dist = None
            if exact_unique_filtered_vector_count > 0:
                last_vector_dist = float(exact_vec_cands[-1]["dist"])

            summary = {
                **estimate,
                "estimate_only": True,
                "target_diag_npy": os.path.abspath(args.target_diag_npy),
                "exact_unique_filtered_vector_count": exact_unique_filtered_vector_count,
                "last_unique_vector_dist": last_vector_dist,
                "summary_json": os.path.abspath(args.output_prefix + ".summary.json"),
            }
            _write_manifest(args.output_prefix + ".summary.json", summary)
            if args.output_prefix:
                _save_vector_candidates(args.output_prefix, exact_vec_cands)
            if not args.quiet:
                print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        out = search_diagonal_matrix_from_rootdb(
            triples_file=args.triples_file,
            triples_json=args.triples_json,
            rootdb_prefix=args.rootdb_prefix,
            f=args.f,
            target_diag=target_diag,
            eps=args.eps,
            max_vector_candidates=args.max_vector_candidates,
            max_matrix_results=args.max_matrix_results,
            triples_chunk_rows=args.triples_chunk_rows,
            n_phase_bins=args.n_phase_bins,
            chunk_meta_json=args.chunk_meta_json,
            deduplicate_vectors=True,
            output_prefix=args.output_prefix,
            reuse_vector_cache=not args.no_reuse_vector_cache,
            resume=not args.no_resume,
            col_start=args.col_start,
            col_stop=args.col_stop,
            verbose=not args.quiet,
        )
        if MPI.COMM_WORLD.Get_rank() == 0:
            if args.output_prefix:
                summary = {
                    "searched_range": list(out["searched_range"]),
                    "searched_upto": int(out["searched_upto"]),
                    "n_candidates": int(out["n_candidates"]),
                    "n_matrix_results": int(len(out["matrix_results"])),
                    "used_cached_search_state": bool(out["used_cached_search_state"]),
                    "vector_cache_npz": os.path.abspath(_vector_cache_paths(args.output_prefix)["npz"]),
                    "vector_cache_json": os.path.abspath(_vector_cache_paths(args.output_prefix)["json"]),
                    "search_state_npz": os.path.abspath(_search_state_paths(args.output_prefix)["npz"]),
                    "search_state_json": os.path.abspath(_search_state_paths(args.output_prefix)["json"]),
                }
                _write_manifest(args.output_prefix + ".summary.json", summary)
            if not args.quiet:
                print(json.dumps({
                    "searched_range": list(out["searched_range"]),
                    "searched_upto": int(out["searched_upto"]),
                    "n_candidates": int(out["n_candidates"]),
                    "n_matrix_results": int(len(out["matrix_results"])),
                    "used_cached_search_state": bool(out["used_cached_search_state"]),
                    "summary_json": os.path.abspath(args.output_prefix + ".summary.json"),
                }, indent=2, sort_keys=True))
