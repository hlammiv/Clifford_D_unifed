
import argparse
import json
import math
import os
import time
import glob
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from typing import List

import numpy as np
from mpi4py import MPI


def tuple3(x):
    return (int(x[0]), int(x[1]), int(x[2]))


def reorder_by_size(Y_lists):
    sizes = [(Y_lists[i].shape[0], i) for i in range(3)]
    sizes.sort()
    order = [idx for _, idx in sizes]
    reordered = [Y_lists[i] for i in order]
    return reordered, order


def _resolve_input_specs(specs: List[str]) -> List[str]:
    out = []
    for spec in specs:
        matches = sorted(glob.glob(spec))
        if matches:
            out.extend(matches)
        elif os.path.exists(spec):
            out.append(spec)
    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def _load_stack(paths: List[str], dedup=True) -> np.ndarray:
    arrs = []
    for p in paths:
        a = np.load(p, mmap_mode='r')
        a = np.asarray(a, dtype=np.int64)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"{p} does not have shape (n,3)")
        arrs.append(a)
    if not arrs:
        return np.empty((0, 3), dtype=np.int64)
    out = np.concatenate(arrs, axis=0)
    if dedup and out.size:
        out = np.unique(out, axis=0)
    return out


def _iter_npy_rows(paths: List[str], rows_per_chunk: int):
    for p in paths:
        arr = np.load(p, mmap_mode='r')
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{p} does not have shape (n,3)")
        n = int(arr.shape[0])
        for i0 in range(0, n, rows_per_chunk):
            i1 = min(i0 + rows_per_chunk, n)
            yield p, i0, i1, np.asarray(arr[i0:i1], dtype=np.int64)


def _count_rows(paths: List[str]) -> int:
    total = 0
    for p in paths:
        a = np.load(p, mmap_mode='r')
        total += int(a.shape[0])
    return total


def _buffered_gather_rows(comm, local_arr, root=0, rows_per_chunk=100000, base_tag=4000):
    rank = comm.Get_rank()
    size = comm.Get_size()
    arr = np.ascontiguousarray(local_arr, dtype=np.int64)
    nrows = int(arr.shape[0])
    ncols = int(arr.shape[1]) if arr.ndim == 2 else 0
    meta_tag = base_tag
    data_tag = base_tag + 1
    mpi_dtype = MPI.INT64_T
    if rank == root:
        out = [None] * size
        out[root] = arr.copy()
        for src_rank in range(size):
            if src_rank == root:
                continue
            rows, cols = comm.recv(source=src_rank, tag=meta_tag)
            recv_arr = np.empty((int(rows), int(cols)), dtype=np.int64)
            pos = 0
            while pos < rows:
                stop = min(pos + rows_per_chunk, rows)
                if stop > pos:
                    comm.Recv([recv_arr[pos:stop], mpi_dtype], source=src_rank, tag=data_tag)
                pos = stop
            out[src_rank] = recv_arr
        return out
    else:
        comm.send((nrows, ncols), dest=root, tag=meta_tag)
        pos = 0
        while pos < nrows:
            stop = min(pos + rows_per_chunk, nrows)
            if stop > pos:
                comm.Send([arr[pos:stop], mpi_dtype], dest=root, tag=data_tag)
            pos = stop
        return None


def _append_rows(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    if arr.size == 0:
        return
    with open(path, "ab") as fh:
        arr.tofile(fh)


def _read_manifest(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _bucket_id(row: np.ndarray, n_buckets: int) -> int:
    # stable integer hash on 3 signed int64 entries
    x0 = int(row[0]) * 1315423911
    x1 = int(row[1]) * 2654435761
    x2 = int(row[2]) * 97531
    return ((x0 ^ x1 ^ x2) & 0x7fffffffffffffff) % n_buckets


def _bucket_id_tuple(t, n_buckets: int) -> int:
    x0 = int(t[0]) * 1315423911
    x1 = int(t[1]) * 2654435761
    x2 = int(t[2]) * 97531
    return ((x0 ^ x1 ^ x2) & 0x7fffffffffffffff) % n_buckets


def _partition_c_files(paths_c: List[str], partition_dir: str, n_buckets: int, chunk_rows: int, manifest_path: str, manifest: dict, verbose=False):
    os.makedirs(partition_dir, exist_ok=True)
    bucket_paths = [os.path.join(partition_dir, f"c_bucket_{i:06d}.bin") for i in range(n_buckets)]
    counts = [0] * n_buckets
    total_rows = _count_rows(paths_c)
    seen = 0
    for p, i0, i1, chunk in _iter_npy_rows(paths_c, chunk_rows):
        # optional within-chunk dedup only
        if chunk.size:
            chunk = np.unique(chunk, axis=0)
        groups = defaultdict(list)
        for row in chunk:
            groups[_bucket_id(row, n_buckets)].append(row)
        for bid, rows in groups.items():
            arr = np.asarray(rows, dtype=np.int64)
            with open(bucket_paths[bid], "ab") as fh:
                arr.tofile(fh)
            counts[bid] += int(arr.shape[0])
        seen += int(chunk.shape[0])
        manifest["c_partition_rows_done"] = seen
        _write_manifest(manifest_path, manifest)
        if verbose and total_rows > 0:
            print(f"partition C rows: ~{seen}/{total_rows}")
    meta = {
        "n_buckets": int(n_buckets),
        "bucket_paths": bucket_paths,
        "bucket_counts": counts,
    }
    with open(os.path.join(partition_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    manifest["c_partition_completed"] = True
    manifest["c_partition_meta"] = os.path.join(partition_dir, "meta.json")
    _write_manifest(manifest_path, manifest)
    return meta


def _load_bucket_map(bucket_path: str):
    if not os.path.exists(bucket_path) or os.path.getsize(bucket_path) == 0:
        return {}
    arr = np.fromfile(bucket_path, dtype=np.int64)
    if arr.size == 0:
        return {}
    arr = arr.reshape((-1, 3))
    # dedup within bucket
    arr = np.unique(arr, axis=0)
    m = {}
    for row in arr:
        t = tuple3(row)
        m.setdefault(t, []).append(row)
    return m


def _split_rows_for_scatter(arr: np.ndarray, size: int):
    return [np.ascontiguousarray(x, dtype=np.int64) for x in np.array_split(arr, size, axis=0)]


def select_triples_mpi(
    inputs1: List[str],
    inputs2: List[str],
    inputs3: List[str],
    f: int,
    norm: int,
    output_file: str,
    dedup_inputs: bool = True,
    b_chunk_rows: int = 20000,
    c_partition_chunk_rows: int = 200000,
    n_c_buckets: int = 1024,
    file_cache_buckets: int = 8,
    mpi_rows_per_chunk: int = 100000,
    resume: bool = False,
    verbose: bool = True,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    manifest_path = output_file + ".json"
    total_scale_int = int(round(norm * (3 ** (2 * f))))

    if rank == 0:
        in1 = _resolve_input_specs(inputs1)
        in2 = _resolve_input_specs(inputs2)
        in3 = _resolve_input_specs(inputs3)
        if not in1 or not in2 or not in3:
            raise RuntimeError("Each input set must resolve to at least one file.")
        m = _read_manifest(manifest_path) if resume else None
        if m is None:
            if os.path.exists(output_file) and not resume:
                os.remove(output_file)
            # Determine smallest set by raw row counts, load only that one
            c0 = _count_rows(in1)
            c1 = _count_rows(in2)
            c2 = _count_rows(in3)
            counts = [(c0, 0), (c1, 1), (c2, 2)]
            counts.sort()
            order = [idx for _, idx in counts]
            smallest_idx = order[0]
            files_by_idx = [in1, in2, in3]
            A = _load_stack(files_by_idx[smallest_idx], dedup=dedup_inputs)
            # reorder file lists only, stream B and C later
            re_inputs = [files_by_idx[i] for i in order]
            # total batches over reordered B
            B_total_rows = _count_rows(re_inputs[1])
            total_batches = (B_total_rows + b_chunk_rows - 1) // b_chunk_rows if B_total_rows else 0
            partition_dir = output_file + ".c_partitions"
            manifest = {
                "version": 2,
                "completed": False,
                "f": int(f),
                "norm": int(norm),
                "total_scale_int": int(total_scale_int),
                "inputs1": in1,
                "inputs2": in2,
                "inputs3": in3,
                "dedup_inputs": bool(dedup_inputs),
                "order": [int(x) for x in order],
                "smallest_size": int(A.shape[0]),
                "b_chunk_rows": int(b_chunk_rows),
                "c_partition_chunk_rows": int(c_partition_chunk_rows),
                "n_c_buckets": int(n_c_buckets),
                "file_cache_buckets": int(file_cache_buckets),
                "mpi_rows_per_chunk": int(mpi_rows_per_chunk),
                "next_batch_idx": 0,
                "total_batches": int(total_batches),
                "rows_written": 0,
                "partition_dir": os.path.abspath(partition_dir),
                "c_partition_completed": False,
                "c_partition_rows_done": 0,
                "A_file_index": int(order[0]),
                "B_file_index": int(order[1]),
                "C_file_index": int(order[2]),
            }
            _write_manifest(manifest_path, manifest)
            # build/rebuild partitions of C
            c_meta = _partition_c_files(re_inputs[2], partition_dir, n_c_buckets, c_partition_chunk_rows, manifest_path, manifest, verbose=verbose)
        else:
            manifest = m
            order = manifest["order"]
            files_by_idx = [manifest["inputs1"], manifest["inputs2"], manifest["inputs3"]]
            re_inputs = [files_by_idx[i] for i in order]
            A = _load_stack(re_inputs[0], dedup=manifest.get("dedup_inputs", True))
            if not manifest.get("c_partition_completed", False):
                c_meta = _partition_c_files(re_inputs[2], manifest["partition_dir"], manifest["n_c_buckets"], manifest["c_partition_chunk_rows"], manifest_path, manifest, verbose=verbose)
            else:
                with open(manifest["c_partition_meta"], "r") as fh:
                    c_meta = json.load(fh)
            if manifest.get("completed"):
                if verbose:
                    print(f"select_triples already completed: {output_file}")
                return manifest
        if verbose:
            print("Loaded input bins:")
            print(f"  smallest resident set rows={A.shape[0]}")
            print(f"  stream B raw rows={_count_rows(re_inputs[1])}")
            print(f"  partitioned C raw rows={_count_rows(re_inputs[2])}")
            print(f"order (new_pos -> original_coord) = {manifest['order']}")
            print(f"B chunk rows = {manifest['b_chunk_rows']}")
            print(f"C partition buckets = {manifest['n_c_buckets']}")
    else:
        A = None
        manifest = None
        re_inputs = None
        c_meta = None

    manifest = comm.bcast(manifest, root=0)
    A = comm.bcast(A, root=0)
    if rank != 0:
        files_by_idx = [manifest["inputs1"], manifest["inputs2"], manifest["inputs3"]]
        re_inputs = [files_by_idx[i] for i in manifest["order"]]
        with open(manifest["c_partition_meta"], "r") as fh:
            c_meta = json.load(fh)

    bucket_paths = c_meta["bucket_paths"]
    n_buckets = int(c_meta["n_buckets"])
    total_batches = int(manifest["total_batches"])
    start_batch = int(manifest["next_batch_idx"])

    # Build B batch iterator offsets without loading whole B
    B_paths = re_inputs[1]
    # precompute file row ranges
    b_files = []
    cum = 0
    for p in B_paths:
        n = int(np.load(p, mmap_mode='r').shape[0])
        b_files.append((p, cum, cum + n))
        cum += n

    bucket_cache = OrderedDict() if rank == 0 else None

    def read_b_global_slice(g0, g1):
        parts = []
        for p, s0, s1 in b_files:
            if g1 <= s0 or g0 >= s1:
                continue
            l0 = max(0, g0 - s0)
            l1 = min(s1 - s0, g1 - s0)
            arr = np.load(p, mmap_mode='r')
            parts.append(np.asarray(arr[l0:l1], dtype=np.int64))
        if not parts:
            return np.empty((0, 3), dtype=np.int64)
        out = np.concatenate(parts, axis=0)
        if manifest.get("dedup_inputs", True) and out.size:
            out = np.unique(out, axis=0)
        return out

    target_sum = np.array([total_scale_int, 0, 0], dtype=np.int64)

    for batch_idx in range(start_batch, total_batches):
        if rank == 0:
            g0 = batch_idx * manifest["b_chunk_rows"]
            g1 = min((batch_idx + 1) * manifest["b_chunk_rows"], cum)
            B_batch = read_b_global_slice(g0, g1)
            B_chunks = _split_rows_for_scatter(B_batch, size)
            if verbose:
                print(f"select_triples batch {batch_idx+1}/{total_batches}: B rows {g0}:{g1} deduped={B_batch.shape[0]}")
        else:
            B_chunks = None
        B_local = comm.scatter(B_chunks, root=0)

        # Each rank computes needed queries grouped by C bucket
        local_queries = defaultdict(list)
        local_total = int(B_local.shape[0])
        last_report = time.time()
        for ib in range(local_total):
            b = B_local[ib]
            for a in A:
                need = target_sum - a - b
                bid = _bucket_id(need, n_buckets)
                # store orig rows in reordered positions
                local_queries[bid].append((tuple3(a), tuple3(b), tuple3(need)))
            if verbose and rank == 0:
                now = time.time()
                if now - last_report >= 10.0:
                    last_report = now
                    pct = 100.0 * (ib + 1) / local_total if local_total > 0 else 100.0
                    print(f"phase 1 batch query build, rank 0 local progress: {pct:.1f}%")

        local_matches_all = []
        # Process buckets in sorted order for bounded memory
        for bid in sorted(local_queries.keys()):
            queries = local_queries[bid]
            if rank == 0:
                if bid in bucket_cache:
                    cmap = bucket_cache.pop(bid)
                    bucket_cache[bid] = cmap
                else:
                    cmap = _load_bucket_map(bucket_paths[bid])
                    bucket_cache[bid] = cmap
                    while len(bucket_cache) > int(manifest["file_cache_buckets"]):
                        bucket_cache.popitem(last=False)
            else:
                cmap = None
            # broadcast only small map? still maybe big. Better each rank load bucket locally.
            if rank != 0:
                cmap = _load_bucket_map(bucket_paths[bid])
            matches = []
            for a_t, b_t, need_t in queries:
                c_rows = cmap.get(need_t)
                if c_rows is None:
                    continue
                for c in c_rows:
                    Y_reordered = [a_t, b_t, tuple3(c)]
                    Y_orig = [None, None, None]
                    for new_pos, old_pos in enumerate(manifest["order"]):
                        Y_orig[old_pos] = Y_reordered[new_pos]
                    matches.append((
                        Y_orig[0][0], Y_orig[0][1], Y_orig[0][2],
                        Y_orig[1][0], Y_orig[1][1], Y_orig[1][2],
                        Y_orig[2][0], Y_orig[2][1], Y_orig[2][2],
                    ))
            if matches:
                local_matches_all.append(np.asarray(matches, dtype=np.int64))
        local_matches = np.concatenate(local_matches_all, axis=0) if local_matches_all else np.empty((0, 9), dtype=np.int64)

        gathered = _buffered_gather_rows(comm, local_matches, root=0, rows_per_chunk=mpi_rows_per_chunk, base_tag=5000 + 10 * batch_idx)
        if rank == 0:
            batch_rows = np.concatenate(gathered, axis=0) if gathered else np.empty((0, 9), dtype=np.int64)
            _append_rows(output_file, batch_rows)
            manifest["rows_written"] += int(batch_rows.shape[0])
            manifest["next_batch_idx"] = batch_idx + 1
            manifest["completed"] = manifest["next_batch_idx"] >= total_batches
            _write_manifest(manifest_path, manifest)
            if verbose:
                print(f"  wrote {batch_rows.shape[0]} triples, cumulative={manifest['rows_written']}")
        comm.Barrier()

    if rank == 0:
        manifest["completed"] = True
        _write_manifest(manifest_path, manifest)
        return manifest
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs1", nargs="+", required=True, help="Files or glob patterns for set 1")
    parser.add_argument("--inputs2", nargs="+", required=True, help="Files or glob patterns for set 2")
    parser.add_argument("--inputs3", nargs="+", required=True, help="Files or glob patterns for set 3")
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--norm", type=int, default=2)
    parser.add_argument("--output", required=True, help="Single output file with raw int64 rows of shape (n,9)")
    parser.add_argument("--b_chunk_rows", type=int, default=20000, help="Rows of reordered B processed at once")
    parser.add_argument("--c_partition_chunk_rows", type=int, default=200000, help="Rows of reordered C used while building on-disk partitions")
    parser.add_argument("--n_c_buckets", type=int, default=1024, help="Number of on-disk hash buckets for reordered C")
    parser.add_argument("--file_cache_buckets", type=int, default=8, help="Number of C buckets cached in memory during matching")
    parser.add_argument("--mpi_rows_per_chunk", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = select_triples_mpi(
        inputs1=args.inputs1,
        inputs2=args.inputs2,
        inputs3=args.inputs3,
        f=args.f,
        norm=args.norm,
        output_file=args.output,
        dedup_inputs=not args.no_dedup,
        b_chunk_rows=args.b_chunk_rows,
        c_partition_chunk_rows=args.c_partition_chunk_rows,
        n_c_buckets=args.n_c_buckets,
        file_cache_buckets=args.file_cache_buckets,
        mpi_rows_per_chunk=args.mpi_rows_per_chunk,
        resume=args.resume,
        verbose=not args.quiet,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
