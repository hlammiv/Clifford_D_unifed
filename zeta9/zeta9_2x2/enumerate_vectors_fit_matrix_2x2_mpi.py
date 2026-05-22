
import argparse
import json
import math
import os
from collections import OrderedDict, defaultdict
from typing import List, Optional, Tuple

import numpy as np
from mpi4py import MPI

try:
    from sage.all import CC, CyclotomicField
except Exception as exc:
    raise ImportError("This script requires SageMath. Run it with sage -python.") from exc

try:
    from .tools import embed
except ImportError:
    from tools import embed

C1 = math.cos(2.0 * math.pi / 9.0)
ALPHA1 = 2.0 * C1
ALPHA1_SQ = ALPHA1 * ALPHA1
TWOPI = 2.0 * math.pi
MU18_EXP = list(range(9)) + list(range(9))
MU18_SGN = [1] * 9 + [-1] * 9


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


def coeffs_to_ring_element(coeffs, zeta):
    coeffs = np.asarray(coeffs, dtype=np.int64).reshape(-1)
    return sum(int(coeffs[i]) * (zeta ** i) for i in range(coeffs.shape[0]))


def ring_element_to_coeffs6(x, zeta):
    K = zeta.parent()
    vec = K(x).vector()
    out = np.zeros((6,), dtype=np.int64)
    n = min(6, len(vec))
    for i in range(n):
        out[i] = int(vec[i])
    return out


def _read_double_rows(path: str, start_row: int, nrows: int) -> np.ndarray:
    if nrows <= 0:
        return np.empty((0, 6), dtype=np.int64)
    itemsize = np.dtype(np.int64).itemsize
    offset = start_row * 6 * itemsize
    with open(path, "rb") as fh:
        fh.seek(offset, os.SEEK_SET)
        arr = np.fromfile(fh, dtype=np.int64, count=nrows * 6)
    return arr.reshape((-1, 6))


def _state_paths(prefix: str, n_phase_bins: int):
    suffix = f".phase_sidecar_binned_bins={int(n_phase_bins)}"
    return {
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


def _build_locator(needed_y: np.ndarray, paths: dict):
    if needed_y.size == 0:
        return {}
    index_y = np.load(paths["exact_roots_index_y"], mmap_mode="r")
    index_file = np.load(paths["exact_roots_index_file"], mmap_mode="r")
    index_row = np.load(paths["exact_roots_index_row"], mmap_mode="r")
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
        fi = int(file_idx[i])
        ri = int(row_idx[i])
        if fi < 0 or ri < 0:
            continue
        y = needed_y[i]
        locator[(int(y[0]), int(y[1]), int(y[2]))] = (fi, ri)
    return needed_y, locator


def _get_sidecar_batch_cache(batch_idx: int, phase_meta: dict, batch_cache: OrderedDict, max_entries: int = 4):
    cached = batch_cache.get(batch_idx)
    if cached is not None:
        batch_cache.move_to_end(batch_idx)
        return cached
    rec = phase_meta["batches"][batch_idx]
    cached = {
        "roots_flat": np.load(rec["roots_flat"], mmap_mode="r"),
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


def _candidate_desc_from_binned_sidecar(y, locator, phase_meta, batch_cache, target, eps, f, profile=None):
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
    a = int(off[row_idx])
    b = int(off[row_idx + 1])
    if b <= a:
        return None

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


def _iter_desc_z_and_indices(desc, phase_meta, batch_cache, f, eps):
    batch_idx, ranges, target = desc
    bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
    scale = float(3 ** f)
    for s, t in ranges:
        z = (np.asarray(bd["z_re_flat"][s:t], dtype=np.float64) + 1j * np.asarray(bd["z_im_flat"][s:t], dtype=np.float64)) / scale
        mask = np.abs(z - target) <= (eps + 1e-15)
        if np.any(mask):
            idx_local = np.flatnonzero(mask).astype(np.int64)
            yield s, idx_local, np.ascontiguousarray(z[mask], dtype=np.complex128)


def _load_coeff_row(batch_idx, abs_index, phase_meta, batch_cache):
    bd = _get_sidecar_batch_cache(batch_idx, phase_meta, batch_cache)
    return np.asarray(bd["roots_flat"][abs_index], dtype=np.int64)


def _complex_from_K(x) -> complex:
    return complex(CC(x))


def _matrix_frobenius_to_diag(U: np.ndarray, diag_target: np.ndarray) -> float:
    return float(np.linalg.norm(U - np.diag(np.asarray(diag_target, dtype=np.complex128))))


def _best_norm1_root_of_unity(target, zeta):
    K = zeta.parent()
    best = None
    best_dist = math.inf
    best_idx = -1
    for t_idx, (sgn, expk) in enumerate(zip(MU18_SGN, MU18_EXP)):
        u = K(sgn) * (zeta ** expk)
        dist = abs(_complex_from_K(u) - complex(target))
        if dist < best_dist:
            best_dist = dist
            best = u
            best_idx = int(t_idx)
    return best, best_idx


def _complete_2x2_exact(row_coeffs: np.ndarray, col_coeffs: np.ndarray, f: int, target_diag: np.ndarray, zeta):
    K = zeta.parent()
    threef = int(3 ** f)

    X = coeffs_to_ring_element(row_coeffs[0], zeta)
    X2 = coeffs_to_ring_element(col_coeffs[0], zeta)
    if X != X2:
        return []

    Y = coeffs_to_ring_element(row_coeffs[1], zeta)
    Z = coeffs_to_ring_element(col_coeffs[1], zeta)

    # both row and column must be normalized unit vectors, so |Y| = |Z|
    Ny = Y * Y.conjugate()
    Nz = Z * Z.conjugate()
    if Ny != Nz:
        return []

    x = X / threef
    y = Y / threef
    z = Z / threef

    out = []

    # degenerate case: |x|=1, hence y=z=0 and w is arbitrary norm-1 element
    if Ny == 0:
        if Y != 0 or Z != 0:
            return []
        for t_idx, (sgn, expk) in enumerate(zip(MU18_SGN, MU18_EXP)):
            wK = K(sgn) * (zeta ** expk)
            U = np.array([
                [_complex_from_K(x), 0.0 + 0.0j],
                [0.0 + 0.0j, _complex_from_K(wK)],
            ], dtype=np.complex128)
            out.append({
                "t_idx": int(t_idx),
                "dist": _matrix_frobenius_to_diag(U, target_diag),
                "matrix": U,
                "w_coeffs": ring_element_to_coeffs6(wK, zeta),
            })
        return out

    # generic case: w = -conj(x) * y / conj(z)
    W = -(X.conjugate() * Y) / Z.conjugate()
    if not W.is_integral():
        return []
    w = W / threef

    U = np.array([
        [_complex_from_K(x), _complex_from_K(y)],
        [_complex_from_K(z), _complex_from_K(w)],
    ], dtype=np.complex128)
    out.append({
        "t_idx": -1,
        "dist": _matrix_frobenius_to_diag(U, target_diag),
        "matrix": U,
        "w_coeffs": ring_element_to_coeffs6(W, zeta),
    })
    return out


def _flush_vector_batch(path: str, recs: List[dict]):
    if not recs:
        return None
    np.savez(
        path,
        Y=np.asarray([r["Y"] for r in recs], dtype=np.int64),
        x_coeffs=np.asarray([r["x_coeffs"] for r in recs], dtype=np.int64),
        err=np.asarray([r["err"] for r in recs], dtype=np.float64),
        x_complex=np.asarray([r["x_complex"] for r in recs], dtype=np.complex128),
    )
    return os.path.abspath(path)


def _load_unique_vectors_from_batches(batch_entries: List[dict]):
    unique = {}
    for rec in batch_entries:
        arr = np.load(rec["path"], allow_pickle=False)
        coeffs = np.asarray(arr["x_coeffs"], dtype=np.int64)
        Y = np.asarray(arr["Y"], dtype=np.int64)
        err = np.asarray(arr["err"], dtype=np.float64)
        x_complex = np.asarray(arr["x_complex"], dtype=np.complex128)
        for i in range(coeffs.shape[0]):
            key = tuple(coeffs[i].reshape(-1).tolist())
            if key in unique:
                continue
            unique[key] = {
                "x_coeffs": coeffs[i],
                "Y": Y[i],
                "err": err[i],
                "x_complex": x_complex[i],
            }
    return list(unique.values())


def _group_vectors_by_first_entry(vectors: List[dict]):
    groups = defaultdict(list)
    for rec in vectors:
        x_key = tuple(np.asarray(rec["x_coeffs"][0], dtype=np.int64).tolist())
        groups[x_key].append(rec)
    return groups


def fit_diagonal_matrix_2x2(
    *,
    doubles_file: str,
    doubles_json: str,
    rootdb_prefix: str,
    f: int,
    diag_target_npy: str,
    eps: float,
    output_prefix: str,
    doubles_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: Optional[str] = None,
    max_vectors_total: Optional[int] = None,
    profile_enabled: bool = True,
    verbose: bool = False,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    profile = {
        "double_read": 0.0,
        "needed_unique": 0.0,
        "locator_build": 0.0,
        "candidate_load": 0.0,
        "enumeration_loop": 0.0,
        "pairing": 0.0,
        "gather": 0.0,
        "chunks_seen": 0,
        "chunks_processed": 0,
        "needed_y": 0,
        "candidate_rows_loaded": 0,
        "doubles_processed": 0,
        "doubles_skipped_empty": 0,
        "vector_combinations_tested": 0,
        "vectors_emitted": 0,
        "matrices_tested": 0,
        "best_matrix_improved": 0,
    }

    if rank == 0:
        tmeta = _read_manifest(doubles_json)
        nrows = int(tmeta["rows_written"])
        expected = nrows * 6 * np.dtype(np.int64).itemsize
        got = os.path.getsize(doubles_file)
        if expected != got:
            raise RuntimeError(f"Double file size mismatch: expected {expected}, got {got}")
        diag_target = np.asarray(np.load(diag_target_npy), dtype=np.complex128).reshape(-1)
        if diag_target.shape != (2,):
            raise ValueError("diag_target_npy must contain shape (2,)")
        vector_target = np.asarray([diag_target[0], 0.0 + 0.0j], dtype=np.complex128)
        paths = _state_paths(rootdb_prefix, n_phase_bins)
        if not os.path.exists(paths["exact_roots_index_meta"]):
            raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")
        if not os.path.exists(paths["phase_sidecar_meta"]):
            raise FileNotFoundError(f"Binned phase sidecar not found for {rootdb_prefix}")
        phase_meta = _read_manifest(paths["phase_sidecar_meta"])
        chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
    else:
        nrows = None
        diag_target = None
        vector_target = None
        paths = None
        phase_meta = None
        chunk_meta = None

    nrows = comm.bcast(nrows, root=0)
    diag_target = comm.bcast(diag_target, root=0)
    vector_target = comm.bcast(vector_target, root=0)
    paths = comm.bcast(paths, root=0)
    phase_meta = comm.bcast(phase_meta, root=0)
    chunk_meta = comm.bcast(chunk_meta, root=0)
    comm.Barrier()

    batch_cache = OrderedDict()
    vector_batch_records: List[dict] = []
    vector_manifest_entries: List[dict] = []
    vector_batch_counter = 0
    dedup_keys = set()
    vector_batches_dir = output_prefix + ".vector_batches"
    os.makedirs(vector_batches_dir, exist_ok=True)

    def flush_vector_records():
        nonlocal vector_batch_counter, vector_batch_records
        if not vector_batch_records:
            return
        path = os.path.join(vector_batches_dir, f"vectors.rank{rank:03d}.batch{vector_batch_counter:06d}.npz")
        out = _flush_vector_batch(path, vector_batch_records)
        vector_manifest_entries.append({"path": out, "n_records": len(vector_batch_records)})
        vector_batch_counter += 1
        vector_batch_records = []

    n_chunks = (nrows + int(doubles_chunk_rows) - 1) // int(doubles_chunk_rows)
    for chunk_iter in range(n_chunks):
        start = chunk_iter * int(doubles_chunk_rows)
        take = min(int(doubles_chunk_rows), nrows - start)
        profile["chunks_seen"] += 1
        if chunk_iter % size != rank:
            continue
        profile["chunks_processed"] += 1

        t0 = os.times()[4]
        doubles = _read_double_rows(doubles_file, start, take)
        profile["double_read"] += os.times()[4] - t0

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_iter]
            t0 = os.times()[4]
            needed, locator = _load_chunk_metadata(rec["path"])
            profile["needed_unique"] += os.times()[4] - t0
        else:
            t0 = os.times()[4]
            needed = np.unique(doubles.reshape((-1, 3)), axis=0)
            profile["needed_unique"] += os.times()[4] - t0
            t0 = os.times()[4]
            locator = _build_locator(needed, paths)
            profile["locator_build"] += os.times()[4] - t0
        profile["needed_y"] += int(needed.shape[0])

        t0 = os.times()[4]
        desc_cache = {}
        for y in locator.keys():
            for coord in range(2):
                desc_cache[(y, coord)] = _candidate_desc_from_binned_sidecar(
                    y, locator, phase_meta, batch_cache, vector_target[coord], eps, f, profile=profile
                )
        profile["candidate_load"] += os.times()[4] - t0

        t0 = os.times()[4]
        for dbl in doubles:
            if max_vectors_total is not None and len(dedup_keys) >= int(max_vectors_total):
                break
            Y1_t = (int(dbl[0]), int(dbl[1]), int(dbl[2]))
            Y2_t = (int(dbl[3]), int(dbl[4]), int(dbl[5]))
            if Y1_t not in locator or Y2_t not in locator:
                continue
            profile["doubles_processed"] += 1

            desc1 = desc_cache[(Y1_t, 0)]
            desc2 = desc_cache[(Y2_t, 1)]
            if desc1 is None or desc2 is None:
                profile["doubles_skipped_empty"] += 1
                continue

            for base1, idx1_local, z1 in _iter_desc_z_and_indices(desc1, phase_meta, batch_cache, f, eps):
                for ia, za in enumerate(z1):
                    abs_x = int(base1 + idx1_local[ia])
                    coeff_x = _load_coeff_row(desc1[0], abs_x, phase_meta, batch_cache)
                    err_x = abs(za - vector_target[0])

                    for base2, idx2_local, z2 in _iter_desc_z_and_indices(desc2, phase_meta, batch_cache, f, eps):
                        for ib, zy in enumerate(z2):
                            profile["vector_combinations_tested"] += 1
                            abs_y = int(base2 + idx2_local[ib])
                            coeff_y = _load_coeff_row(desc2[0], abs_y, phase_meta, batch_cache)
                            coeffs = np.stack([coeff_x, coeff_y], axis=0)
                            key = tuple(coeffs.reshape(-1).tolist())
                            if key in dedup_keys:
                                continue
                            dedup_keys.add(key)
                            vector_batch_records.append({
                                "Y": np.asarray([Y1_t, Y2_t], dtype=np.int64),
                                "x_coeffs": coeffs,
                                "err": np.asarray([err_x, abs(zy - vector_target[1])], dtype=np.float64),
                                "x_complex": np.asarray([za, zy], dtype=np.complex128),
                            })
                            profile["vectors_emitted"] += 1
                            if len(vector_batch_records) >= 2000:
                                flush_vector_records()
                            if max_vectors_total is not None and len(dedup_keys) >= int(max_vectors_total):
                                break
                        if max_vectors_total is not None and len(dedup_keys) >= int(max_vectors_total):
                            break
                    if max_vectors_total is not None and len(dedup_keys) >= int(max_vectors_total):
                        break
        profile["enumeration_loop"] += os.times()[4] - t0
        if verbose:
            print(
                f"rank {rank}: processed chunk {chunk_iter+1}/{n_chunks} "
                f"| unique vectors so far={len(dedup_keys):,} "
                f"| emitted so far={profile['vectors_emitted']:,}",
                flush=True,
            )

    flush_vector_records()
    if verbose and rank == 0:
        print(
            f"rank {rank}: enumeration finished "
            f"| total unique vectors={len(dedup_keys):,} "
            f"| batches written={len(vector_manifest_entries):,}",
            flush=True,
        )

    t0 = os.times()[4]
    payload = {"vector_manifest_entries": vector_manifest_entries, "profile": profile}
    gathered = comm.gather(payload, root=0)
    profile["gather"] += os.times()[4] - t0

    if rank != 0:
        return None

    all_vector_batches = []
    profile_by_rank = {}
    profile_sum = {}
    for part_i, part in enumerate(gathered):
        all_vector_batches.extend(part["vector_manifest_entries"])
        rp = part["profile"]
        profile_by_rank[str(part_i)] = rp
        for k, v in rp.items():
            if isinstance(v, (int, float)):
                profile_sum[k] = profile_sum.get(k, 0) + v

    K = CyclotomicField(9)
    zeta = K.gen()

    if verbose:
        print(f"rank 0: loading and deduplicating {len(all_vector_batches):,} vector batch files", flush=True)
    vectors = _load_unique_vectors_from_batches(all_vector_batches)
    if verbose:
        print(f"rank 0: loaded {len(vectors):,} unique vectors", flush=True)

    component_unique_counts = {"x": 0, "y": 0}
    bucket_size_stats = {"min": 0, "median": 0.0, "mean": 0.0, "max": 0, "top_bucket_sizes": [], "top_bucket_pair_counts": []}
    if vectors:
        unique_x = set()
        unique_y = set()
        for rec in vectors:
            coeffs = np.asarray(rec["x_coeffs"], dtype=np.int64)
            unique_x.add(tuple(coeffs[0].tolist()))
            unique_y.add(tuple(coeffs[1].tolist()))
        component_unique_counts = {"x": int(len(unique_x)), "y": int(len(unique_y))}
        if verbose:
            print(
                f"rank 0: component unique counts | x={component_unique_counts['x']:,} | y={component_unique_counts['y']:,}",
                flush=True,
            )

    grouped = _group_vectors_by_first_entry(vectors)
    if verbose:
        print(f"rank 0: grouped vectors into {len(grouped):,} shared-x buckets", flush=True)

    if grouped:
        bucket_sizes = np.asarray(sorted((len(bucket) for bucket in grouped.values()), reverse=True), dtype=np.int64)
        bucket_pair_counts = bucket_sizes.astype(object) * bucket_sizes.astype(object)
        bucket_size_stats = {
            "min": int(bucket_sizes.min()),
            "median": float(np.median(bucket_sizes)),
            "mean": float(bucket_sizes.mean()),
            "max": int(bucket_sizes.max()),
            "top_bucket_sizes": [int(x) for x in bucket_sizes[:10].tolist()],
            "top_bucket_pair_counts": [int(x) for x in bucket_pair_counts[:10].tolist()],
        }
        if verbose:
            print(
                f"rank 0: bucket size stats | min={bucket_size_stats['min']:,} "
                f"| median={bucket_size_stats['median']:,.1f} "
                f"| mean={bucket_size_stats['mean']:,.1f} "
                f"| max={bucket_size_stats['max']:,}",
                flush=True,
            )
            print(
                f"rank 0: top bucket sizes={bucket_size_stats['top_bucket_sizes']} "
                f"| top bucket pair counts={bucket_size_stats['top_bucket_pair_counts']}",
                flush=True,
            )

    best_matrix_dist = math.inf
    best_matrix = np.zeros((2, 2), dtype=np.complex128)
    best_matrix_coeffs = np.zeros((2, 2, 6), dtype=np.int64)
    best_row_coeffs = np.zeros((2, 6), dtype=np.int64)
    best_col_coeffs = np.zeros((2, 6), dtype=np.int64)
    best_row_Y = np.zeros((2, 3), dtype=np.int64)
    best_col_Y = np.zeros((2, 3), dtype=np.int64)
    best_t_idx = -1

    t0 = os.times()[4]
    pair_buckets_done = 0
    pair_buckets_total = len(grouped)
    total_pair_count = sum(len(bucket) * len(bucket) for bucket in grouped.values())
    total_pairs_done = 0
    total_pairs_last_report = 0
    last_pair_report_time = t0
    pair_report_interval_sec = 10.0

    if verbose:
        print(
            f"rank 0: starting pairing over {pair_buckets_total:,} shared-x buckets "
            f"| total row/col pairs={total_pair_count:,}",
            flush=True,
        )

    for bucket in grouped.values():
        pair_buckets_done += 1
        bucket_size = len(bucket)
        bucket_pair_count = bucket_size * bucket_size
        bucket_pairs_done = 0
        bucket_start_time = os.times()[4]
        if verbose:
            print(
                f"rank 0: starting bucket {pair_buckets_done}/{pair_buckets_total} "
                f"| bucket size={bucket_size:,} | bucket pairs={bucket_pair_count:,}",
                flush=True,
            )
        for row_idx, row_rec in enumerate(bucket, start=1):
            for col_idx, col_rec in enumerate(bucket, start=1):
                bucket_pairs_done += 1
                total_pairs_done += 1
                completions = _complete_2x2_exact(row_rec["x_coeffs"], col_rec["x_coeffs"], f, diag_target, zeta)
                profile_sum["matrices_tested"] = profile_sum.get("matrices_tested", 0) + len(completions)
                for comp in completions:
                    if comp["dist"] < best_matrix_dist:
                        best_matrix_dist = comp["dist"]
                        best_matrix[:, :] = comp["matrix"]

                        best_row_coeffs[:, :] = row_rec["x_coeffs"]
                        best_col_coeffs[:, :] = col_rec["x_coeffs"]
                        best_row_Y[:, :] = row_rec["Y"]
                        best_col_Y[:, :] = col_rec["Y"]
                        best_t_idx = int(comp["t_idx"])

                        # exact matrix coeffs
                        best_matrix_coeffs[:, :, :] = 0
                        best_matrix_coeffs[0, 0, :] = row_rec["x_coeffs"][0]
                        best_matrix_coeffs[0, 1, :] = row_rec["x_coeffs"][1]
                        best_matrix_coeffs[1, 0, :] = col_rec["x_coeffs"][1]
                        best_matrix_coeffs[1, 1, :] = np.asarray(comp["w_coeffs"], dtype=np.int64)

                        profile_sum["best_matrix_improved"] = profile_sum.get("best_matrix_improved", 0) + 1
                        if verbose:
                            print(
                                f"rank 0: improved best matrix | bucket {pair_buckets_done}/{pair_buckets_total} "
                                f"| row {row_idx}/{bucket_size} | col {col_idx}/{bucket_size} "
                                f"| dist={best_matrix_dist:.12g}",
                                flush=True,
                            )
                if verbose:
                    now = os.times()[4]
                    if now - last_pair_report_time >= pair_report_interval_sec:
                        delta_pairs = total_pairs_done - total_pairs_last_report
                        delta_t = max(now - last_pair_report_time, 1e-12)
                        rate = delta_pairs / delta_t
                        bucket_pct = (100.0 * bucket_pairs_done / bucket_pair_count) if bucket_pair_count else 100.0
                        total_pct = (100.0 * total_pairs_done / total_pair_count) if total_pair_count else 100.0
                        print(
                            f"rank 0: pairing progress | bucket {pair_buckets_done}/{pair_buckets_total} "
                            f"| row {row_idx}/{bucket_size} | col {col_idx}/{bucket_size} "
                            f"| bucket {bucket_pairs_done:,}/{bucket_pair_count:,} ({bucket_pct:.2f}%) "
                            f"| total pairs {total_pairs_done:,}/{total_pair_count:,} ({total_pct:.2f}%) "
                            f"| matrices tested={profile_sum.get('matrices_tested', 0):,} "
                            f"| rate={rate:,.1f} pairs/s",
                            flush=True,
                        )
                        last_pair_report_time = now
                        total_pairs_last_report = total_pairs_done
            if verbose and bucket_size <= 2000:
                row_report_every = max(1, bucket_size // 10)
            else:
                row_report_every = max(1, min(1000, bucket_size // 20))
            if verbose and (row_idx == bucket_size or row_idx % row_report_every == 0):
                bucket_elapsed = os.times()[4] - bucket_start_time
                print(
                    f"rank 0: bucket {pair_buckets_done}/{pair_buckets_total} row progress "
                    f"| rows {row_idx}/{bucket_size} "
                    f"| pairs {bucket_pairs_done:,}/{bucket_pair_count:,} "
                    f"| elapsed={bucket_elapsed:.1f}s",
                    flush=True,
                )

        if verbose:
            bucket_elapsed = os.times()[4] - bucket_start_time
            rate = bucket_pair_count / max(bucket_elapsed, 1e-12)
            print(
                f"rank 0: finished bucket {pair_buckets_done}/{pair_buckets_total} "
                f"| bucket size={bucket_size:,} | bucket pairs={bucket_pair_count:,} "
                f"| elapsed={bucket_elapsed:.1f}s | rate={rate:,.1f} pairs/s "
                f"| matrices tested={profile_sum.get('matrices_tested', 0):,}",
                flush=True,
            )

    profile_sum["pairing"] = profile_sum.get("pairing", 0.0) + (os.times()[4] - t0)
    if verbose:
        print(
            f"rank 0: pairing finished | total pairs={total_pairs_done:,}/{total_pair_count:,} "
            f"| matrices tested={profile_sum.get('matrices_tested', 0):,} "
            f"| best found={math.isfinite(best_matrix_dist)}",
            flush=True,
        )

    best_npz = output_prefix + ".best_matrix.npz"
    np.savez(
        best_npz,
        diag_target=diag_target,
        vector_target=vector_target,
        best_found=np.asarray([math.isfinite(best_matrix_dist)], dtype=bool),
        best_matrix_dist=np.asarray([best_matrix_dist if math.isfinite(best_matrix_dist) else np.nan], dtype=np.float64),
        best_matrix=best_matrix[np.newaxis, :, :],
        best_matrix_coeffs=best_matrix_coeffs[np.newaxis, :, :, :],
        best_row_coeffs=best_row_coeffs[np.newaxis, :, :],
        best_col_coeffs=best_col_coeffs[np.newaxis, :, :],
        best_row_Y=best_row_Y[np.newaxis, :, :],
        best_col_Y=best_col_Y[np.newaxis, :, :],
        best_t_idx=np.asarray([best_t_idx], dtype=np.int64),
        n_unique_vectors=np.asarray([len(vectors)], dtype=np.int64),
        n_shared_x_buckets=np.asarray([len(grouped)], dtype=np.int64),
        component_unique_counts=np.asarray([[component_unique_counts["x"], component_unique_counts["y"]]], dtype=np.int64),
    )

    summary = {
        "best_matrix_npz": os.path.abspath(best_npz),
        "diag_target": [[float(np.real(v)), float(np.imag(v))] for v in diag_target],
        "vector_target": [[float(np.real(v)), float(np.imag(v))] for v in vector_target],
        "n_unique_vectors": int(len(vectors)),
        "n_shared_x_buckets": int(len(grouped)),
        "component_unique_counts": component_unique_counts,
        "bucket_size_stats": bucket_size_stats,
        "n_vector_batches": int(len(all_vector_batches)),
        "best_found": bool(math.isfinite(best_matrix_dist)),
        "best_matrix_dist": None if not math.isfinite(best_matrix_dist) else float(best_matrix_dist),
        "mpi_ranks": int(size),
        "rootdb_prefix": os.path.abspath(rootdb_prefix),
        "phase_sidecar": os.path.abspath(paths["phase_sidecar_dir"]),
        "n_phase_bins": int(phase_meta["n_phase_bins"]),
        "vector_batches": all_vector_batches,
    }
    _write_manifest(output_prefix + ".json", summary)
    if profile_enabled:
        _write_manifest(output_prefix + ".profile.json", {"by_rank": profile_by_rank, "sum": profile_sum})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doubles_file", required=True)
    parser.add_argument("--doubles_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--diag_target_npy", required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--doubles_chunk_rows", type=int, default=200000)
    parser.add_argument("--n_phase_bins", type=int, default=512)
    parser.add_argument("--chunk_meta_json", default=None)
    parser.add_argument("--max_vectors_total", type=int, default=None)
    parser.add_argument("--no_profile", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = fit_diagonal_matrix_2x2(
        doubles_file=args.doubles_file,
        doubles_json=args.doubles_json,
        rootdb_prefix=args.rootdb_prefix,
        f=args.f,
        diag_target_npy=args.diag_target_npy,
        eps=args.eps,
        output_prefix=args.output_prefix,
        doubles_chunk_rows=args.doubles_chunk_rows,
        n_phase_bins=args.n_phase_bins,
        chunk_meta_json=args.chunk_meta_json,
        max_vectors_total=args.max_vectors_total,
        profile_enabled=not args.no_profile,
        verbose=not args.quiet,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
