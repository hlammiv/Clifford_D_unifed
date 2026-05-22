"""Build reusable per-chunk metadata for a triples file.

For each triples chunk, this stores:
- the unique Y values needed by that chunk
- the corresponding rootdb (file,row) locator entries

This lets fit runs skip both:
- np.unique(triples.reshape((-1,3)), axis=0)
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


def _read_triple_rows(path: str, start_row: int, nrows: int) -> np.ndarray:
    if nrows <= 0:
        return np.empty((0, 9), dtype=np.int64)
    itemsize = np.dtype(np.int64).itemsize
    offset = start_row * 9 * itemsize
    with open(path, "rb") as fh:
        fh.seek(offset, os.SEEK_SET)
        arr = np.fromfile(fh, dtype=np.int64, count=nrows * 9)
    return arr.reshape((-1, 9))


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


def build_triple_chunk_metadata(*, triples_file: str, triples_json: str, rootdb_prefix: str, output_prefix: str, triples_chunk_rows: int = 200000, verbose: bool = False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tmeta = _read_manifest(triples_json)
    nrows = int(tmeta["rows_written"])
    file_size = os.path.getsize(triples_file)
    expected = nrows * 9 * np.dtype(np.int64).itemsize
    if file_size != expected:
        raise RuntimeError(f"Triple file size mismatch: expected {expected}, got {file_size}")

    paths = _state_paths(rootdb_prefix)
    if not os.path.exists(paths["exact_roots_index_meta"]):
        raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")

    index_y = np.load(paths["exact_roots_index_y"], mmap_mode='r')
    index_file = np.load(paths["exact_roots_index_file"], mmap_mode='r')
    index_row = np.load(paths["exact_roots_index_row"], mmap_mode='r')
    key_index = _structured_view_y(index_y)

    meta_dir = output_prefix + ".chunks"
    if rank == 0:
        os.makedirs(meta_dir, exist_ok=True)
    comm.Barrier()

    total_chunks = (int(nrows) + int(triples_chunk_rows) - 1) // int(triples_chunk_rows)
    # Distribute chunk_id by modulo so all ranks share work.
    local_records = []
    for chunk_id in range(total_chunks):
        if chunk_id % size != rank:
            continue
        start = chunk_id * int(triples_chunk_rows)
        take = min(int(triples_chunk_rows), nrows - start)
        triples = _read_triple_rows(triples_file, start, take)
        needed = np.unique(triples.reshape((-1, 3)), axis=0)
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

        local_records.append({
            "chunk_id": int(chunk_id),
            "start_row": int(start),
            "nrows": int(take),
            "path": os.path.abspath(out_path),
            "needed_y_count": int(needed.shape[0]),
        })
        if verbose:
            print(f"[rank {rank}] metadata chunk {chunk_id+1}/{total_chunks}: rows {start}..{start+take-1}, needed_y={needed.shape[0]}", flush=True)

    # Gather all per-rank records to rank 0.
    all_records = comm.gather(local_records, root=0)
    if rank != 0:
        return None

    chunk_records = sorted((r for lst in all_records for r in lst),
                           key=lambda r: r["chunk_id"])
    manifest = {
        "version": 1,
        "triples_file": os.path.abspath(triples_file),
        "triples_json": os.path.abspath(triples_json),
        "rootdb_prefix": os.path.abspath(rootdb_prefix),
        "triples_chunk_rows": int(triples_chunk_rows),
        "nrows": int(nrows),
        "chunks": chunk_records,
    }
    _write_manifest(output_prefix + ".json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--triples_json", required=True)
    parser.add_argument("--rootdb_prefix", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--triples_chunk_rows", type=int, default=200000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = build_triple_chunk_metadata(
        triples_file=args.triples_file,
        triples_json=args.triples_json,
        rootdb_prefix=args.rootdb_prefix,
        output_prefix=args.output_prefix,
        triples_chunk_rows=args.triples_chunk_rows,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(out)
