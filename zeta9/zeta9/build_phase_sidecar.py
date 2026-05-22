"""Build the phase sidecar for a rootdb without running any fitting.

This is a one-time preprocessing step. The fitter can then reuse the sidecar
across many target scans without paying the build cost again.
"""

import argparse
import json
import math
import os
import time

import numpy as np

try:
    from .tools import embed
except ImportError:
    from tools import embed

C1 = math.cos(2.0 * math.pi / 9.0)
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


def coeffs_to_complex_noscale(x_coeffs):
    return embed(tuple(int(v) for v in x_coeffs), 1)


def _state_paths(prefix: str):
    return {
        "exact_roots_dir": prefix + ".exact_roots_batches",
        "exact_roots_index_meta": prefix + ".exact_roots_index_meta.json",
        "phase_sidecar_dir": prefix + ".phase_sidecar",
        "phase_sidecar_meta": prefix + ".phase_sidecar.meta.json",
    }


def _sidecar_batch_paths(sidecar_dir: str, batch_idx: int):
    stem = os.path.join(sidecar_dir, f"phase_batch_{batch_idx:06d}")
    return {
        "roots_flat": stem + ".roots_flat.npy",
        "phases_flat": stem + ".phases_flat.npy",
        "z_re_flat": stem + ".z_re_flat.npy",
        "z_im_flat": stem + ".z_im_flat.npy",
        "roots_off": stem + ".roots_off.npy",
    }


def build_phase_sidecar(rootdb_prefix: str, force: bool = False, verbose: bool = False):
    paths = _state_paths(rootdb_prefix)
    if not os.path.exists(paths["exact_roots_index_meta"]):
        raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")

    meta_path = paths["phase_sidecar_meta"]
    if (not force) and os.path.exists(meta_path):
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
            return {
                "status": "already_exists",
                "phase_sidecar_meta": os.path.abspath(meta_path),
                "phase_sidecar_dir": os.path.abspath(paths["phase_sidecar_dir"]),
                "batches": len(meta.get("batches", [])),
            }

    t0 = time.time()
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

        batches_meta.append({
            "batch_idx": int(batch_idx),
            "root_file": os.path.abspath(root_path),
            **bp,
        })

    meta = {
        "version": 1,
        "rootdb_prefix": os.path.abspath(rootdb_prefix),
        "source_root_files": [os.path.abspath(p) for p in root_files],
        "batches": batches_meta,
        "elapsed_seconds": float(time.time() - t0),
    }
    _write_manifest(meta_path, meta)
    return {
        "status": "built",
        "phase_sidecar_meta": os.path.abspath(meta_path),
        "phase_sidecar_dir": os.path.abspath(paths["phase_sidecar_dir"]),
        "batches": len(batches_meta),
        "elapsed_seconds": meta["elapsed_seconds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = build_phase_sidecar(
        rootdb_prefix=args.rootdb_prefix,
        force=args.force,
        verbose=not args.quiet,
    )
    print(out)
