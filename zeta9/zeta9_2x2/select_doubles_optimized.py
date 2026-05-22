import argparse
import glob
import json
import os
import time
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

ROW_DTYPE = np.int64
STRUCT_DTYPE = np.dtype([('x', '<i8'), ('y', '<i8'), ('z', '<i8')])


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


def _count_rows(paths: List[str]) -> int:
    total = 0
    for p in paths:
        a = np.load(p, mmap_mode='r')
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f'{p} does not have shape (n,3)')
        total += int(a.shape[0])
    return total


def _load_stack(paths: List[str], dedup=True) -> np.ndarray:
    arrs = []
    for p in paths:
        a = np.load(p, mmap_mode='r')
        a = np.asarray(a, dtype=ROW_DTYPE)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f'{p} does not have shape (n,3)')
        arrs.append(a)
    if not arrs:
        return np.empty((0, 3), dtype=ROW_DTYPE)
    out = np.concatenate(arrs, axis=0)
    if dedup and out.size:
        out = np.unique(out, axis=0)
    return np.ascontiguousarray(out, dtype=ROW_DTYPE)


def _iter_npy_rows(paths: List[str], rows_per_chunk: int):
    for p in paths:
        arr = np.load(p, mmap_mode='r')
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f'{p} does not have shape (n,3)')
        n = int(arr.shape[0])
        for i0 in range(0, n, rows_per_chunk):
            i1 = min(i0 + rows_per_chunk, n)
            yield p, i0, i1, np.asarray(arr[i0:i1], dtype=ROW_DTYPE)


def _read_manifest(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _write_manifest(path: str, data: dict):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _append_rows_raw(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=ROW_DTYPE)
    if arr.size == 0:
        return
    with open(path, 'ab') as fh:
        arr.tofile(fh)


def _buffered_gather_counts(comm, local_value, root=0, tag=9100):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == root:
        vals = [0] * size
        vals[root] = int(local_value)
        for src in range(size):
            if src == root:
                continue
            vals[src] = int(comm.recv(source=src, tag=tag))
        return vals
    comm.send(int(local_value), dest=root, tag=tag)
    return None


def _bucket_linear_hash_rows(rows: np.ndarray, n_buckets: int, coeffs: Tuple[int, int, int]) -> np.ndarray:
    c0, c1, c2 = coeffs
    x = rows[:, 0].astype(np.int64) * np.int64(c0)
    x += rows[:, 1].astype(np.int64) * np.int64(c1)
    x += rows[:, 2].astype(np.int64) * np.int64(c2)
    return np.mod(x, np.int64(n_buckets)).astype(np.int64)


def _as_struct(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=ROW_DTYPE)
    return arr.view(STRUCT_DTYPE).reshape(-1)


def _unique_sorted_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.empty((0, 3), dtype=ROW_DTYPE)
    s = np.sort(_as_struct(arr), kind='mergesort')
    keep = np.empty(s.shape[0], dtype=bool)
    keep[0] = True
    keep[1:] = s[1:] != s[:-1]
    out = s[keep].view(ROW_DTYPE).reshape(-1, 3)
    return np.ascontiguousarray(out, dtype=ROW_DTYPE)


def _membership_mask_sorted_rows(sorted_rows: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    if query_rows.size == 0:
        return np.empty((0,), dtype=bool)
    if sorted_rows.size == 0:
        return np.zeros(query_rows.shape[0], dtype=bool)
    base = _as_struct(sorted_rows)
    q = _as_struct(query_rows)
    idx = np.searchsorted(base, q)
    mask = idx < base.shape[0]
    mask[mask] &= base[idx[mask]] == q[mask]
    return mask


def _bucket_file_name(base_dir: str, prefix: str, idx: int) -> str:
    return os.path.join(base_dir, f'{prefix}_bucket_{idx:06d}.bin')


def _partition_set_to_buckets(paths, base_dir, prefix, n_buckets, rows_per_chunk, coeffs, manifest_path, manifest, state_key, verbose=False):
    os.makedirs(base_dir, exist_ok=True)
    bucket_paths = [_bucket_file_name(base_dir, prefix, i) for i in range(n_buckets)]
    total_rows = _count_rows(paths)
    rows_done = int(manifest.get(f'{state_key}_rows_done', 0))
    if rows_done != 0:
        manifest[f'{state_key}_rows_done'] = 0
        for p in bucket_paths:
            if os.path.exists(p):
                os.remove(p)
        _write_manifest(manifest_path, manifest)
    seen = 0
    for _, _, _, chunk in _iter_npy_rows(paths, rows_per_chunk):
        if chunk.size:
            chunk = _unique_sorted_rows(chunk)
            bids = _bucket_linear_hash_rows(chunk, n_buckets, coeffs)
            order = np.argsort(bids, kind='mergesort')
            chunk = chunk[order]
            bids = bids[order]
            starts = np.flatnonzero(np.r_[True, bids[1:] != bids[:-1]])
            ends = np.r_[starts[1:], bids.shape[0]]
            for s, e in zip(starts, ends):
                _append_rows_raw(bucket_paths[int(bids[s])], chunk[s:e])
        seen += int(chunk.shape[0])
        manifest[f'{state_key}_rows_done'] = seen
        _write_manifest(manifest_path, manifest)
        if verbose and total_rows > 0:
            print(f'partition {prefix} rows: ~{seen}/{total_rows}')
    bucket_counts = []
    for p in bucket_paths:
        bucket_counts.append(int(os.path.getsize(p) // (3 * np.dtype(ROW_DTYPE).itemsize)) if os.path.exists(p) else 0)
    meta = {
        'n_buckets': int(n_buckets),
        'bucket_paths': bucket_paths,
        'bucket_counts_raw': bucket_counts,
        'coeffs': [int(coeffs[0]), int(coeffs[1]), int(coeffs[2])],
    }
    meta_path = os.path.join(base_dir, f'{prefix}_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    manifest[f'{state_key}_completed'] = True
    manifest[f'{state_key}_meta'] = meta_path
    _write_manifest(manifest_path, manifest)
    return meta


class BucketCache:
    def __init__(self, max_entries: int):
        self.max_entries = int(max_entries)
        self._cache = OrderedDict()

    def get(self, path: str):
        if path in self._cache:
            val = self._cache.pop(path)
            self._cache[path] = val
            return val
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            arr = np.empty((0, 3), dtype=ROW_DTYPE)
        else:
            raw = np.fromfile(path, dtype=ROW_DTYPE)
            arr = np.empty((0, 3), dtype=ROW_DTYPE) if raw.size == 0 else _unique_sorted_rows(raw.reshape(-1, 3))
        self._cache[path] = arr
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return arr


def _load_meta(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _reconstruct_original_rows(a_rows: np.ndarray, b_rows: np.ndarray, order: List[int]) -> np.ndarray:
    n = a_rows.shape[0]
    out = np.empty((n, 6), dtype=ROW_DTYPE)
    cols = [a_rows, b_rows]
    for new_pos, old_pos in enumerate(order):
        out[:, 3 * old_pos:3 * old_pos + 3] = cols[new_pos]
    return out


def _process_a_groups_against_buckets(a_groups, order, target_sum, target_residue, b_bucket_paths, n_buckets, bucket_cache, rank_output_path, flush_rows, progress_interval_sec=10.0, progress_prefix='', show_progress=False):
    total_written = 0
    out_buf = []
    out_rows = 0
    last_progress_time = time.time()
    a_done = 0
    total_a = int(sum(rows.shape[0] for _, rows in a_groups))
    for ra, a_rows in a_groups:
        rb = int((target_residue - int(ra)) % int(n_buckets))
        b_rows = bucket_cache.get(b_bucket_paths[rb])
        if b_rows.size == 0:
            a_done += int(a_rows.shape[0])
            continue
        for ia in range(a_rows.shape[0]):
            a = a_rows[ia]
            need_b = target_sum - a
            if _membership_mask_sorted_rows(b_rows, need_b.reshape(1, 3))[0]:
                out_part = _reconstruct_original_rows(a.reshape(1, 3), need_b.reshape(1, 3), order)
                out_buf.append(out_part)
                out_rows += 1
                if out_rows >= flush_rows:
                    arr = np.concatenate(out_buf, axis=0)
                    _append_rows_raw(rank_output_path, arr)
                    total_written += int(arr.shape[0])
                    out_buf = []
                    out_rows = 0
            a_done += 1
            if show_progress:
                now = time.time()
                if now - last_progress_time >= progress_interval_sec:
                    last_progress_time = now
                    pct = 100.0 * float(a_done) / float(total_a) if total_a > 0 else 100.0
                    print(f'{progress_prefix}{a_done}/{total_a} A rows ({pct:.1f}%), local_matches_written={total_written + out_rows}, current_B_bucket={rb+1}/{n_buckets}')
    if out_buf:
        arr = np.concatenate(out_buf, axis=0)
        _append_rows_raw(rank_output_path, arr)
        total_written += int(arr.shape[0])
    if show_progress:
        print(f'{progress_prefix}{total_a}/{total_a} A rows (100.0%), local_matches_written={total_written}')
    return total_written


def select_doubles_mpi(inputs1, inputs2, f, norm, output_file, dedup_inputs=True, partition_rows_per_chunk=500000, n_join_buckets=8192, bucket_cache_entries=16, rank_flush_rows=200000, resume=False, verbose=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    manifest_path = output_file + '.manifest.json'
    parts_dir = output_file + '.parts'
    join_dir = output_file + '.join_buckets'
    hash_coeffs = (911382323, 972663749, 9721)

    total_scale_int = int(round(norm * (3 ** (2 * f))))
    target_sum = np.array([total_scale_int, 0, 0], dtype=ROW_DTYPE)

    if rank == 0:
        in1 = _resolve_input_specs(inputs1)
        in2 = _resolve_input_specs(inputs2)
        if not in1 or not in2:
            raise RuntimeError('Each input set must resolve to at least one file.')
        manifest = _read_manifest(manifest_path) if resume else None
        if manifest is None:
            if os.path.exists(output_file):
                os.remove(output_file)
            os.makedirs(parts_dir, exist_ok=True)
            os.makedirs(join_dir, exist_ok=True)
            raw_counts = [_count_rows(in1), _count_rows(in2)]
            order = list(np.argsort(np.asarray(raw_counts)))
            files_by_idx = [in1, in2]
            re_inputs = [files_by_idx[i] for i in order]
            A = _load_stack(re_inputs[0], dedup=dedup_inputs)
            manifest = {
                'version': 1, 'completed': False, 'f': int(f), 'norm': int(norm),
                'total_scale_int': int(total_scale_int),
                'inputs1': in1, 'inputs2': in2, 'dedup_inputs': bool(dedup_inputs),
                'order': [int(x) for x in order], 'A_file_index': int(order[0]), 'B_file_index': int(order[1]),
                'smallest_size': int(A.shape[0]), 'n_join_buckets': int(n_join_buckets),
                'partition_rows_per_chunk': int(partition_rows_per_chunk),
                'bucket_cache_entries': int(bucket_cache_entries), 'rank_flush_rows': int(rank_flush_rows),
                'hash_coeffs': [int(x) for x in hash_coeffs],
                'parts_dir': os.path.abspath(parts_dir), 'join_dir': os.path.abspath(join_dir),
                'B_partition_completed': False, 'B_rows_done': 0, 'completed_buckets': [], 'rows_written': 0,
            }
            _write_manifest(manifest_path, manifest)
        else:
            files_by_idx = [manifest['inputs1'], manifest['inputs2']]
            re_inputs = [files_by_idx[i] for i in manifest['order']]
            A = _load_stack(re_inputs[0], dedup=manifest.get('dedup_inputs', True))
            os.makedirs(manifest['parts_dir'], exist_ok=True)
            os.makedirs(manifest['join_dir'], exist_ok=True)

        files_by_idx = [manifest['inputs1'], manifest['inputs2']]
        re_inputs = [files_by_idx[i] for i in manifest['order']]
        if not manifest.get('B_partition_completed', False):
            b_meta = _partition_set_to_buckets(re_inputs[1], manifest['join_dir'], 'B', manifest['n_join_buckets'],
                                               manifest['partition_rows_per_chunk'], hash_coeffs,
                                               manifest_path, manifest, 'B', verbose=verbose)
        else:
            b_meta = _load_meta(manifest['B_meta'])
        if 'B_meta' not in manifest:
            manifest['B_meta'] = os.path.join(manifest['join_dir'], 'B_meta.json')
        _write_manifest(manifest_path, manifest)
        if verbose:
            print('Loaded/reordered inputs:')
            print(f'  smallest resident set rows = {A.shape[0]}')
            print(f'  B raw rows = {_count_rows(re_inputs[1])}')
            print(f'  order (new_pos -> original_coord) = {manifest["order"]}')
            print(f'  join buckets = {manifest["n_join_buckets"]}')
            print(f'  bucket cache entries per rank = {manifest["bucket_cache_entries"]}')
    else:
        A = None
        manifest = None
        b_meta = None

    manifest = comm.bcast(manifest, root=0)
    A = comm.bcast(A, root=0)
    if rank != 0:
        b_meta = _load_meta(manifest['B_meta'])

    n_buckets = int(manifest['n_join_buckets'])
    b_bucket_paths = b_meta['bucket_paths']
    completed_buckets = set(int(x) for x in manifest.get('completed_buckets', []))
    order = [int(x) for x in manifest['order']]

    a_hashes = _bucket_linear_hash_rows(A, n_buckets, hash_coeffs)
    target_residue = int(_bucket_linear_hash_rows(target_sum.reshape(1, 3), n_buckets, hash_coeffs)[0])
    order_idx = np.argsort(a_hashes, kind='mergesort')
    A_sorted = A[order_idx]
    a_hashes_sorted = a_hashes[order_idx]
    starts = np.flatnonzero(np.r_[True, a_hashes_sorted[1:] != a_hashes_sorted[:-1]]) if a_hashes_sorted.size else np.empty((0,), dtype=np.int64)
    ends = np.r_[starts[1:], a_hashes_sorted.shape[0]] if a_hashes_sorted.size else np.empty((0,), dtype=np.int64)
    a_groups = [(int(a_hashes_sorted[s]), np.ascontiguousarray(A_sorted[s:e], dtype=ROW_DTYPE)) for s, e in zip(starts, ends)]

    rank_output_path = os.path.join(manifest['parts_dir'], f'rank_{rank:04d}.bin')
    if not resume and os.path.exists(rank_output_path):
        os.remove(rank_output_path)

    cache = BucketCache(manifest['bucket_cache_entries'])
    wall_t0 = time.time()
    tprev = wall_t0

    for phase_start in range(0, n_buckets, size):
        my_bid = phase_start + rank
        local_written = 0
        if my_bid < n_buckets and my_bid not in completed_buckets:
            tnow = time.time()
            if verbose and rank == 0 and tnow > tprev + 10:
                tprev = tnow
                done_before = len(completed_buckets)
                print(f'processing B bucket {my_bid+1}/{n_buckets} on rank 0 (completed {done_before}/{n_buckets}, cumulative_rows={manifest.get("rows_written", 0)})')
            local_written = _process_a_groups_against_buckets(
                a_groups, order, target_sum, target_residue, b_bucket_paths, n_buckets, cache, rank_output_path,
                int(manifest['rank_flush_rows']), progress_interval_sec=10.0,
                progress_prefix=f'bucket {my_bid+1}/{n_buckets}: ', show_progress=bool(verbose and rank == 0)
            )
        phase_counts = _buffered_gather_counts(comm, local_written, root=0, tag=9300 + phase_start)
        comm.Barrier()
        if rank == 0:
            newly_done = []
            for r in range(size):
                bid = phase_start + r
                if bid >= n_buckets or bid in completed_buckets:
                    continue
                newly_done.append(bid)
            completed_buckets.update(newly_done)
            manifest['completed_buckets'] = sorted(int(x) for x in completed_buckets)
            manifest['rows_written'] = int(manifest.get('rows_written', 0)) + int(sum(phase_counts))
            manifest['completed'] = len(completed_buckets) >= n_buckets
            _write_manifest(manifest_path, manifest)
        comm.Barrier()

    if rank == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        rows_written = 0
        for r in range(size):
            part = os.path.join(manifest['parts_dir'], f'rank_{r:04d}.bin')
            if not os.path.exists(part) or os.path.getsize(part) == 0:
                continue
            with open(part, 'rb') as src, open(output_file, 'ab') as dst:
                while True:
                    chunk = src.read(8 * 6 * 100000)
                    if not chunk:
                        break
                    dst.write(chunk)
            rows_written += int(os.path.getsize(part) // (6 * np.dtype(ROW_DTYPE).itemsize))
        manifest['rows_written'] = rows_written
        manifest['completed'] = True
        manifest['wall_total_time'] = float(time.time() - wall_t0)
        _write_manifest(manifest_path, manifest)
        if verbose:
            print(f'wrote final output: {output_file}')
            print(f'total rows written: {rows_written}')
            print(f'wall total time: {manifest["wall_total_time"]:.3f} s')
        return manifest
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs1', nargs='+', required=True, help='Files or glob patterns for set 1')
    parser.add_argument('--inputs2', nargs='+', required=True, help='Files or glob patterns for set 2')
    parser.add_argument('--f', type=int, required=True)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--output', required=True, help='Single output file with raw int64 rows of shape (n,6)')
    parser.add_argument('--partition_rows_per_chunk', type=int, default=500000)
    parser.add_argument('--n_join_buckets', type=int, default=8192)
    parser.add_argument('--bucket_cache_entries', type=int, default=16)
    parser.add_argument('--rank_flush_rows', type=int, default=200000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no_dedup', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    out = select_doubles_mpi(
        inputs1=args.inputs1,
        inputs2=args.inputs2,
        f=args.f,
        norm=args.norm,
        output_file=args.output,
        dedup_inputs=not args.no_dedup,
        partition_rows_per_chunk=args.partition_rows_per_chunk,
        n_join_buckets=args.n_join_buckets,
        bucket_cache_entries=args.bucket_cache_entries,
        rank_flush_rows=args.rank_flush_rows,
        resume=args.resume,
        verbose=not args.quiet,
    )
    if MPI.COMM_WORLD.Get_rank() == 0 and args.quiet:
        print(out)
