"""Build a binned phase sidecar for a rootdb.

MPI-parallel version: root batch files are distributed across ranks. Each rank
builds a disjoint subset of sidecar batch files, and rank 0 writes the final
manifest once all ranks finish.

This leaves the original rootdb untouched and writes:
  <rootdb_prefix>.phase_sidecar_binned_bins=<N>/
  <rootdb_prefix>.phase_sidecar_binned_bins=<N>.meta.json
"""

import argparse
import json
import math
import os
from typing import List, Dict

import numpy as np
from mpi4py import MPI

try:
    from .tools import embed
except ImportError:
    from tools import embed

TWOPI = 2.0 * math.pi


def _read_manifest(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def coeffs_to_complex_noscale(x_coeffs):
    return embed(tuple(int(v) for v in x_coeffs), 1)


def _state_paths(prefix: str, n_phase_bins: int):
    suffix = f".phase_sidecar_binned_bins={int(n_phase_bins)}"
    return {
        "exact_roots_dir": prefix + ".exact_roots_batches",
        "exact_roots_index_meta": prefix + ".exact_roots_index_meta.json",
        "phase_sidecar_dir": prefix + suffix,
        "phase_sidecar_meta": prefix + suffix + ".meta.json",
    }


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


def _existing_sidecar_ok(meta_path: str, n_phase_bins: int) -> dict | None:
    if not os.path.exists(meta_path):
        return None
    meta = _read_manifest(meta_path)
    if int(meta.get("n_phase_bins", -1)) != int(n_phase_bins):
        return None
    for rec in meta.get("batches", []):
        for key in ("roots_flat", "phases_flat", "z_re_flat", "z_im_flat", "roots_off", "bin_off"):
            if not os.path.exists(rec[key]):
                return None
    return meta


def _build_one_batch(root_path: str, sidecar_dir: str, batch_idx: int, n_phase_bins: int) -> dict:
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

    return {
        "batch_idx": int(batch_idx),
        "root_file": os.path.abspath(root_path),
        **bp,
    }


def build_phase_sidecar_mpi(*, rootdb_prefix: str, n_phase_bins: int = 512, force: bool = False, verbose: bool = False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    paths = _state_paths(rootdb_prefix, n_phase_bins)

    if rank == 0:
        if not os.path.exists(paths["exact_roots_index_meta"]):
            raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")

        if not force:
            meta = _existing_sidecar_ok(paths["phase_sidecar_meta"], n_phase_bins)
            if meta is not None:
                out = {
                    "rootdb_prefix": os.path.abspath(rootdb_prefix),
                    "phase_sidecar": os.path.abspath(paths["phase_sidecar_dir"]),
                    "phase_sidecar_meta": os.path.abspath(paths["phase_sidecar_meta"]),
                    "n_phase_bins": int(meta["n_phase_bins"]),
                    "nbatches": int(len(meta.get("batches", []))),
                    "mpi_ranks": int(size),
                    "rebuilt": False,
                }
                payload = {"done": True, "out": out, "root_files": None}
                if verbose:
                    print("existing binned phase sidecar is valid; nothing to do")
            else:
                root_meta = _read_manifest(paths["exact_roots_index_meta"])
                root_files = root_meta["files"]
                os.makedirs(paths["phase_sidecar_dir"], exist_ok=True)
                payload = {"done": False, "out": None, "root_files": root_files}
        else:
            root_meta = _read_manifest(paths["exact_roots_index_meta"])
            root_files = root_meta["files"]
            os.makedirs(paths["phase_sidecar_dir"], exist_ok=True)
            payload = {"done": False, "out": None, "root_files": root_files}
    else:
        payload = None

    payload = comm.bcast(payload, root=0)
    if payload["done"]:
        return payload["out"]

    root_files: List[str] = payload["root_files"]
    local_recs: List[Dict] = []
    n_batches = len(root_files)

    for batch_idx, root_path in enumerate(root_files):
        if batch_idx % size != rank:
            continue
        if verbose:
            print(f"build binned phase sidecar batch {batch_idx + 1}/{n_batches} on rank {rank}")
        local_recs.append(
            _build_one_batch(
                root_path=root_path,
                sidecar_dir=paths["phase_sidecar_dir"],
                batch_idx=batch_idx,
                n_phase_bins=n_phase_bins,
            )
        )

    gathered = comm.gather(local_recs, root=0)

    if rank == 0:
        batches_meta = []
        for recs in gathered:
            batches_meta.extend(recs)
        batches_meta.sort(key=lambda r: r["batch_idx"])

        meta = {
            "version": 1,
            "n_phase_bins": int(n_phase_bins),
            "source_root_files": [os.path.abspath(p) for p in root_files],
            "batches": batches_meta,
        }
        _write_manifest(paths["phase_sidecar_meta"], meta)

        out = {
            "rootdb_prefix": os.path.abspath(rootdb_prefix),
            "phase_sidecar": os.path.abspath(paths["phase_sidecar_dir"]),
            "phase_sidecar_meta": os.path.abspath(paths["phase_sidecar_meta"]),
            "n_phase_bins": int(meta["n_phase_bins"]),
            "nbatches": int(len(meta.get("batches", []))),
            "mpi_ranks": int(size),
            "rebuilt": True,
        }
        return out
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--n_phase_bins", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = build_phase_sidecar_mpi(
        rootdb_prefix=args.rootdb_prefix,
        n_phase_bins=args.n_phase_bins,
        force=args.force,
        verbose=not args.quiet,
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        if args.quiet:
            print(out)
        else:
            print(json.dumps(out, indent=2, sort_keys=True))
