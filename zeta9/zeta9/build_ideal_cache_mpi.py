"""CLI: MPI-parallel Stage B for the ideal cache.

Stage B = DFS-enumerate all composite ideals as products of prime ideals from
the prime table (already built by build_prime_table_mpi.py). Partition by
FIRST prime index assigned to each rank (round-robin to balance small-vs-
large prime subtree sizes).

The DFS algorithm uses the constraint "non-decreasing prime indices" to
guarantee no duplicates. Partitioning by FIRST prime gives a clean,
non-overlapping split since every composite ideal has a unique smallest
prime in its factorization.

Each rank:
  - Loads the prime table (small, ~tens of MB)
  - Iterates first_prime_idx ∈ {its assigned indices}
  - DFS from each such root, with subsequent primes ≥ first_prime_idx
  - Writes records to a per-rank raw binary file

Rank 0:
  - Also emits the identity ideal (γ = 1) ONCE
  - After all ranks finish: gathers file sizes, writes header + concatenates

Usage:
    mpirun -n 32 python -m zeta9.build_ideal_cache_mpi \\
        --prime_table primes.npz \\
        --unit_logs cache.bin.units.npz \\
        --norm_bound 1000000000 \\
        --output cache.bin

Output is identical to build_ideal_cache.py's stage B (same record format,
header, identity-first-ish), just parallelized.
"""
import argparse
import os
import struct
import time

import numpy as np
from mpi4py import MPI

from .ideal_cache import (
    CACHE_MAGIC, CACHE_VERSION, CACHE_HEADER_FMT, CACHE_HEADER_SIZE,
    IDEAL_RECORD_DTYPE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime_table", required=True,
                        help="Path to primes.npz from build_prime_table_mpi")
    parser.add_argument("--unit_logs", required=True,
                        help="Path to units.npz (not used here but checked exists)")
    parser.add_argument("--norm_bound", type=int, required=True)
    parser.add_argument("--output", required=True, help="Output cache file path")
    parser.add_argument("--workdir", default=None,
                        help="Per-rank temp file dir (default: output + '.mpi_parts')")
    parser.add_argument("--pari_stack_gb", type=int, default=4,
                        help="PARI stack size in GiB per rank")
    parser.add_argument("--keep_parts", action="store_true",
                        help="Don't delete per-rank temp files after concat")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.workdir is None:
        args.workdir = args.output + ".mpi_parts"

    norm_bound = int(args.norm_bound)

    if rank == 0:
        print(f"=== Stage B MPI (mpi={size}) ===", flush=True)
        print(f"  norm_bound: {norm_bound:,}", flush=True)
        print(f"  prime_table: {args.prime_table}", flush=True)
        print(f"  output: {args.output}", flush=True)
        print(f"  workdir: {args.workdir}", flush=True)
        os.makedirs(args.workdir, exist_ok=True)
        # Clean stale per-rank files
        for fn in os.listdir(args.workdir):
            if fn.startswith("rank_"):
                try: os.remove(os.path.join(args.workdir, fn))
                except OSError: pass
    comm.Barrier()

    # ---- Load prime table (all ranks, mmap'd so OS page cache shares it) ----
    # Without mmap, each rank gets its own copy in RAM → 32 × 3 GB = 96 GB.
    # With mmap, all ranks share one read-only OS page-cache copy.
    data = np.load(args.prime_table, mmap_mode="r")
    p_arr = data["p"]              # mmap'd
    norm_arr = data["norm"]        # mmap'd
    coefs_arr = data["coefs"]      # mmap'd; (N, 6) int32 in STANDARD basis
    logs_arr = data["logs"]        # mmap'd
    n_primes = norm_arr.shape[0]

    if rank == 0:
        print(f"  loaded {n_primes:,} prime ideals (mmap)", flush=True)

    # NOTE: we do NOT sort by norm here — sorting requires materializing arrays
    # in RAM (defeats mmap). The MPI Stage A writer already sorted the table,
    # so the on-disk order is sorted-by-norm.

    # Use our Numba O_K multiplication directly — no PARI in the hot loop.
    # This saves ~100s of GB of PARI stack across ranks AND is faster per
    # multiplication (no Python boundary crossing).
    from .ideal_cache_conv import _mul_in_OK

    # ---- Round-robin work distribution. No broadcast needed — each rank
    # computes its own work on-the-fly via modulo partitioning.
    #
    # Identity: rank 0
    # Length-1 (singletons π_i): rank (i % size)
    # Length-≥2 (chains starting (i_1, i_2, ...)): rank (i_2 % size)
    #
    # The heaviest individual pair (0, 0) contributes one rank's load ~1.5×
    # average; we accept this minor imbalance.
    max_i1_norm = int(norm_bound ** 0.5)  # i_1 must have N ≤ √B for any pair
    if rank == 0:
        # Count how many i_1's are viable (early-exit horizon)
        n_viable_i1 = int(np.sum(norm_arr <= max_i1_norm))
        print(f"  partition: length-1 by (i % {size}), "
              f"length-≥2 by (i_2 % {size})", flush=True)
        print(f"  max_i1_norm: {max_i1_norm}  viable i_1 count: {n_viable_i1:,}",
              flush=True)

    # ---- Run DFS, write per-rank raw file ----
    rank_path = os.path.join(args.workdir, f"rank_{rank:04d}.bin")
    t0 = time.perf_counter()

    BUF_SIZE = 65536
    buf = np.empty(BUF_SIZE, dtype=IDEAL_RECORD_DTYPE)
    buf_idx = 0
    n_local = 0

    with open(rank_path, "wb") as fh:
        # Rank 0 emits the identity ideal (γ = 1)
        if rank == 0:
            buf[0]["coefs"] = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
            buf[0]["logsigma"] = np.zeros(3, dtype=np.float64)
            buf[0]["norm"] = 1
            buf_idx = 1
            n_local = 1

        # Length-1 singletons: emit π_i for i % size == rank
        for i in range(rank, n_primes, size):
            n_i = int(norm_arr[i])
            if n_i > norm_bound:
                continue
            buf[buf_idx]["coefs"] = coefs_arr[i]
            buf[buf_idx]["logsigma"] = logs_arr[i]
            buf[buf_idx]["norm"] = n_i
            buf_idx += 1
            n_local += 1
            if buf_idx == BUF_SIZE:
                buf[:buf_idx].tofile(fh)
                buf_idx = 0

        # Length-≥2: for each i_1 with N_i1 ≤ √B, iterate i_2 ≥ i_1 in this
        # rank's mod class. From each (i_1, i_2) pair, DFS the full subtree
        # extending with primes ≥ i_2.
        for i1 in range(n_primes):
            n_i1 = int(norm_arr[i1])
            if n_i1 > max_i1_norm:
                break  # no pairs possible from here on (norm_arr sorted)
            g1_std = coefs_arr[i1]
            g1_log = logs_arr[i1]

            # First i_2 ≥ i_1 in this rank's mod class
            start_i2 = i1 + ((rank - i1) % size)
            for i2 in range(start_i2, n_primes, size):
                n_i2 = int(norm_arr[i2])
                new_norm = n_i1 * n_i2
                if new_norm > norm_bound:
                    break
                # γ_2 = π_{i_1} × π_{i_2}
                p2_std = coefs_arr[i2]
                r0, r1, r2, r3, r4, r5 = _mul_in_OK(
                    int(g1_std[0]), int(g1_std[1]), int(g1_std[2]),
                    int(g1_std[3]), int(g1_std[4]), int(g1_std[5]),
                    int(p2_std[0]), int(p2_std[1]), int(p2_std[2]),
                    int(p2_std[3]), int(p2_std[4]), int(p2_std[5]),
                )
                gamma2_std = np.array([r0, r1, r2, r3, r4, r5], dtype=np.int64)
                gamma2_log = g1_log + logs_arr[i2]

                # DFS from (i_1, i_2) extending with primes ≥ i_2
                stack = [(i2, gamma2_std, gamma2_log, new_norm)]
                while stack:
                    cur_idx, gamma_std, gamma_log, gamma_norm = stack.pop()
                    buf[buf_idx]["coefs"] = gamma_std.astype(np.int32)
                    buf[buf_idx]["logsigma"] = gamma_log
                    buf[buf_idx]["norm"] = gamma_norm
                    buf_idx += 1
                    n_local += 1
                    if buf_idx == BUF_SIZE:
                        buf[:buf_idx].tofile(fh)
                        buf_idx = 0
                    for j in range(cur_idx, n_primes):
                        nj = int(norm_arr[j])
                        next_norm = gamma_norm * nj
                        if next_norm > norm_bound:
                            break
                        pj_std = coefs_arr[j]
                        e0, e1, e2, e3, e4, e5 = _mul_in_OK(
                            int(gamma_std[0]), int(gamma_std[1]), int(gamma_std[2]),
                            int(gamma_std[3]), int(gamma_std[4]), int(gamma_std[5]),
                            int(pj_std[0]), int(pj_std[1]), int(pj_std[2]),
                            int(pj_std[3]), int(pj_std[4]), int(pj_std[5]),
                        )
                        next_std = np.array([e0, e1, e2, e3, e4, e5], dtype=np.int64)
                        next_log = gamma_log + logs_arr[j]
                        stack.append((j, next_std, next_log, next_norm))

        # Flush remainder
        if buf_idx > 0:
            buf[:buf_idx].tofile(fh)

    t_local = time.perf_counter() - t0
    file_size = os.path.getsize(rank_path)

    # ---- Gather per-rank counts to rank 0 ----
    all_counts = comm.gather(n_local, root=0)
    all_sizes = comm.gather(file_size, root=0)
    all_times = comm.gather(t_local, root=0)

    if rank == 0:
        n_total = sum(all_counts)
        bytes_total = sum(all_sizes)
        print(f"\n[stage B done]", flush=True)
        print(f"  total ideals: {n_total:,}", flush=True)
        print(f"  total bytes: {bytes_total:,} ({bytes_total / 1e9:.2f} GB)", flush=True)
        print(f"  rank walltimes: min {min(all_times):.1f}s  max {max(all_times):.1f}s  "
              f"avg {sum(all_times) / len(all_times):.1f}s", flush=True)
        print(f"  load imbalance: max/min = {max(all_times) / max(0.1, min(all_times)):.2f}",
              flush=True)

        # ---- Concatenate per-rank files into final cache ----
        print(f"\n[concat] writing {args.output}", flush=True)
        t0 = time.perf_counter()
        with open(args.output, "wb") as out:
            # Write header with correct n_ideals
            header = struct.pack(
                CACHE_HEADER_FMT,
                CACHE_MAGIC,
                CACHE_VERSION,
                n_total,
                norm_bound,
                norm_bound,  # prime_bound — matches what build_ideal_cache uses
            )
            out.write(header)
            # Append each per-rank file
            BUF_BYTES = 8 * 1024 * 1024  # 8 MB chunks
            for r in range(size):
                p = os.path.join(args.workdir, f"rank_{r:04d}.bin")
                if not os.path.exists(p):
                    continue
                with open(p, "rb") as fh:
                    while True:
                        chunk = fh.read(BUF_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
        t_concat = time.perf_counter() - t0
        print(f"  concat done in {t_concat:.1f}s", flush=True)
        print(f"  final size: {os.path.getsize(args.output) / 1e9:.2f} GB", flush=True)

        # ---- Cleanup ----
        if not args.keep_parts:
            for r in range(size):
                p = os.path.join(args.workdir, f"rank_{r:04d}.bin")
                try: os.remove(p)
                except OSError: pass
            try: os.rmdir(args.workdir)
            except OSError: pass


if __name__ == "__main__":
    main()
