"""Build reusable per-chunk metadata for a doubles file.

For each doubles chunk, this stores:
- the unique Y values needed by that chunk
- the corresponding rootdb (file,row) locator entries

This lets fit runs skip both:
- np.unique(doubles.reshape((-1,3)), axis=0)
- rebuilding the Y -> (file,row) locator
"""

import argparse
import json
import os
import numpy as np

from mpi4py import MPI


STRUCT_DTYPE = np.dtype([("y0", "<i8"), ("y1", "<i8"), ("y2", "<i8")])


def _read_manifest(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_double_rows(path: str, start_row: int, nrows: int) -> np.ndarray:
    if nrows <= 0:
        return np.empty((0, 6), dtype=np.int64)
    itemsize = np.dtype(np.int64).itemsize
    offset = start_row * 6 * itemsize
    with open(path, "rb") as fh:
        fh.seek(offset, os.SEEK_SET)
        arr = np.fromfile(fh, dtype=np.int64, count=nrows * 6)
    return arr.reshape((-1, 6))


def _structured_view_y(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr.view(dtype=STRUCT_DTYPE).reshape(-1)


def _state_paths(prefix: str):
    return {
        "exact_roots_index_y": prefix + ".exact_roots_index_y.npy",
        "exact_roots_index_file": prefix + ".exact_roots_index_file.npy",
        "exact_roots_index_row": prefix + ".exact_roots_index_row.npy",
        "exact_roots_index_meta": prefix + ".exact_roots_index_meta.json",
    }


def build_double_chunk_metadata(*, doubles_file: str, doubles_json: str, rootdb_prefix: str, output_prefix: str, doubles_chunk_rows: int = 200000, verbose: bool = False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        return None

    tmeta = _read_manifest(doubles_json)
    nrows = int(tmeta["rows_written"])
    file_size = os.path.getsize(doubles_file)
    expected = nrows * 6 * np.dtype(np.int64).itemsize
    if file_size != expected:
        raise RuntimeError(f"Double file size mismatch: expected {expected}, got {file_size}")

    paths = _state_paths(rootdb_prefix)
    if not os.path.exists(paths["exact_roots_index_meta"]):
        raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")

    index_y = np.load(paths["exact_roots_index_y"], mmap_mode='r')
    index_file = np.load(paths["exact_roots_index_file"], mmap_mode='r')
    index_row = np.load(paths["exact_roots_index_row"], mmap_mode='r')
    key_index = _structured_view_y(index_y)

    meta_dir = output_prefix + ".chunks"
    os.makedirs(meta_dir, exist_ok=True)

    chunk_records = []
    total_chunks = (int(nrows) + int(doubles_chunk_rows) - 1) // int(doubles_chunk_rows)
    start = 0
    chunk_id = 0
    while start < nrows:
        take = min(int(doubles_chunk_rows), nrows - start)
        doubles = _read_double_rows(doubles_file, start, take)
        needed = np.unique(doubles.reshape((-1, 3)), axis=0)
        key_need = _structured_view_y(needed)
        pos = np.searchsorted(key_index, key_need)

        file_idx = np.full((needed.shape[0],), -1, dtype=np.int32)
        row_idx = np.full((needed.shape[0],), -1, dtype=np.int32)
        valid = pos < key_index.shape[0]
        valid2 = np.zeros_like(valid, dtype=bool)
        valid2[valid] = (key_index[pos[valid]] == key_need[valid])
        if np.any(valid2):
            p = pos[valid2]
            file_idx[valid2] = np.asarray(index_file[p], dtype=np.int32)
            row_idx[valid2] = np.asarray(index_row[p], dtype=np.int32)

        out_path = os.path.join(meta_dir, f"chunk_{chunk_id:06d}.npz")
        np.savez(out_path, needed_y=np.ascontiguousarray(needed, dtype=np.int64), file_idx=file_idx, row_idx=row_idx)

        chunk_records.append({
            "chunk_id": int(chunk_id),
            "start_row": int(start),
            "nrows": int(take),
            "path": os.path.abspath(out_path),
            "needed_y_count": int(needed.shape[0]),
        })
        if verbose:
            print(f"metadata chunk {chunk_id+1}/{total_chunks}: rows {start}..{start+take-1}, needed_y={needed.shape[0]}")
        start += take
        chunk_id += 1

    manifest = {
        "version": 1,
        "doubles_file": os.path.abspath(doubles_file),
        "doubles_json": os.path.abspath(doubles_json),
        "rootdb_prefix": os.path.abspath(rootdb_prefix),
        "doubles_chunk_rows": int(doubles_chunk_rows),
        "nrows": int(nrows),
        "chunks": chunk_records,
    }
    _write_manifest(output_prefix + ".json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doubles_file", required=True)
    parser.add_argument("--doubles_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--doubles_chunk_rows", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = build_double_chunk_metadata(
        doubles_file=args.doubles_file,
        doubles_json=args.doubles_json,
        rootdb_prefix=args.rootdb_prefix,
        output_prefix=args.output_prefix,
        doubles_chunk_rows=args.doubles_chunk_rows,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(out)
