"""Build a binned phase sidecar for a rootdb.

This is a one-time preprocessing step for fit_vectors_mpi_sidecar_binned_fit.py.
It leaves the original rootdb untouched and writes:
  <rootdb_prefix>.phase_sidecar_binned/
  <rootdb_prefix>.phase_sidecar_binned.meta.json
"""
"""Fit one or many target vectors using a precomputed rootdb database.

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
        "phase_sidecar_dir": prefix + ".phase_sidecar_binned",
        "phase_sidecar_meta": prefix + ".phase_sidecar_binned.meta.json",
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
    y, locator: dict, phase_meta: dict, batch_cache: OrderedDict, target: complex, eps: float, f: int, profile: dict | None = None
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




def build_phase_sidecar(*, rootdb_prefix: str, n_phase_bins: int = 512, verbose: bool = False):
    paths = _state_paths(rootdb_prefix)
    if not os.path.exists(paths["exact_roots_index_meta"]):
        raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")
    meta = _ensure_binned_phase_sidecar(paths, n_phase_bins=n_phase_bins, verbose=verbose)
    out = {
        "rootdb_prefix": os.path.abspath(rootdb_prefix),
        "phase_sidecar": os.path.abspath(paths["phase_sidecar_dir"]),
        "phase_sidecar_meta": os.path.abspath(paths["phase_sidecar_meta"]),
        "n_phase_bins": int(meta["n_phase_bins"]),
        "nbatches": int(len(meta.get("batches", []))),
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--n_phase_bins", type=int, default=512)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = build_phase_sidecar(
        rootdb_prefix=args.rootdb_prefix,
        n_phase_bins=args.n_phase_bins,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(out)
    else:
        print(json.dumps(out, indent=2, sort_keys=True))
