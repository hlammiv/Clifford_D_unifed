"""CLI: build the prime-ideal table for K = Q(ζ_9) using MPI.

Splits the rational prime range [2, prime_bound] across ranks; each rank
computes the prime decomposition + generator + log-embeddings for its slice.
Rank 0 gathers and writes one .npz.

This is the parallelized version of IdealCacheBuilder.build_prime_table()
from ideal_cache.py. Saves ~3 hours on the cache build by parallelizing the
embarrassingly parallel prime-table phase.

Usage:
    mpirun -n 32 python -m zeta9.build_prime_table_mpi \\
        --prime_bound 800000000 \\
        --norm_bound 800000000 \\
        --output /mnt/.../primes_K=zeta9_B=8e8.npz

After this completes, pass the output to build_ideal_cache.py via
--prime_table_path together with --resume.

Note on correctness:
- Each prime p is processed independently (no shared state).
- bnfinit(K) is called per-rank (some duplicated startup ~1s).
- Generators are returned in K's integral basis; consistent across ranks.
"""
import argparse
import os
import time

import numpy as np
from mpi4py import MPI

from .ideal_cache import K_DEFINING_POLY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime_bound", type=int, required=True)
    parser.add_argument("--norm_bound", type=int, required=True,
                        help="Skip prime ideals with N(P) > norm_bound")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--pari_stack_gb", type=int, default=4)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"=== prime table build (mpi={size}) ===", flush=True)
        print(f"  prime_bound: {args.prime_bound:,}", flush=True)
        print(f"  norm_bound: {args.norm_bound:,}", flush=True)
        print(f"  output: {args.output}", flush=True)

    # Get rational primes — generate once, scatter to ranks
    if rank == 0:
        try:
            from sympy import primerange
            all_primes = list(primerange(2, args.prime_bound + 1))
        except ImportError:
            # Fallback: use PARI per-rank later. Use simple sieve here.
            n = args.prime_bound + 1
            sieve = np.ones(n, dtype=bool)
            sieve[:2] = False
            for i in range(2, int(np.sqrt(n)) + 1):
                if sieve[i]:
                    sieve[i*i::i] = False
            all_primes = np.nonzero(sieve)[0].tolist()
        print(f"  total rational primes: {len(all_primes):,}", flush=True)
    else:
        all_primes = None
    all_primes = comm.bcast(all_primes, root=0)

    # Divide primes among ranks
    n_primes = len(all_primes)
    per_rank = (n_primes + size - 1) // size
    lo = rank * per_rank
    hi = min(lo + per_rank, n_primes)
    my_primes = all_primes[lo:hi]

    if rank == 0:
        print(f"  primes/rank ≈ {per_rank}", flush=True)
        print(f"\n[mpi] each rank starts cypari2 + bnfinit...", flush=True)

    t0 = time.perf_counter()

    import cypari2
    pari = cypari2.Pari()
    pari.allocatemem(args.pari_stack_gb * (1 << 30))
    K = pari.bnfinit(K_DEFINING_POLY)

    t_setup = time.perf_counter() - t0
    if rank == 0:
        print(f"  bnfinit done in {t_setup:.1f}s/rank", flush=True)

    # Per-rank prime ideal extraction
    local_p = []
    local_norm = []
    local_coefs = []
    local_logs = []

    zeta = np.exp(2j * np.pi / 9)
    embed_powers = np.empty((3, 6), dtype=np.complex128)
    for idx, k in enumerate([1, 2, 4]):
        sz = zeta ** k
        for j in range(6):
            embed_powers[idx, j] = sz ** j

    import math
    t1 = time.perf_counter()
    progress = 0
    for p in my_primes:
        progress += 1
        if rank == 0 and progress % 5000 == 0:
            elapsed = time.perf_counter() - t1
            rate = progress / elapsed
            eta = (len(my_primes) - progress) / rate
            print(f"    [rank 0] {progress}/{len(my_primes)} primes  "
                  f"rate {rate:.0f}/s  eta {eta:.0f}s",
                  flush=True)

        prime_ideals = pari.idealprimedec(K, p)
        for P in prime_ideals:
            f_P = int(P[3])
            N_P = p ** f_P
            if N_P > args.norm_bound:
                continue
            _cl, gen = pari.bnfisprincipal(K, P)
            # PARI returns gen in its OWN integral basis [1, ω³, ω, ω⁴, ω², ω⁵].
            # Permute to standard (1, ω, ω², ω³, ω⁴, ω⁵).
            # standard[j] = pari[PARI_INV_PERM[j]], PARI_INV_PERM = [0,2,4,1,3,5]
            gen_coefs = np.array(
                [int(gen[idx_p]) for idx_p in (0, 2, 4, 1, 3, 5)],
                dtype=np.int32,
            )

            # Log embeddings — computed in standard basis (ω-power)
            val = embed_powers @ gen_coefs.astype(np.complex128)
            log_sig = np.log(np.abs(val))

            local_p.append(p)
            local_norm.append(N_P)
            local_coefs.append(gen_coefs)
            local_logs.append(log_sig)

    t_compute = time.perf_counter() - t1
    n_local = len(local_p)
    if rank == 0:
        print(f"  rank 0: {n_local} prime ideals in {t_compute:.1f}s "
              f"({1e3 * t_compute / max(1, n_local):.2f} ms/ideal)", flush=True)

    # Gather to rank 0
    local_p = np.array(local_p, dtype=np.int64)
    local_norm = np.array(local_norm, dtype=np.int64)
    local_coefs = np.stack(local_coefs) if local_coefs else np.empty((0, 6), dtype=np.int32)
    local_logs = np.stack(local_logs) if local_logs else np.empty((0, 3), dtype=np.float64)

    all_p = comm.gather(local_p, root=0)
    all_norm = comm.gather(local_norm, root=0)
    all_coefs = comm.gather(local_coefs, root=0)
    all_logs = comm.gather(local_logs, root=0)

    if rank == 0:
        merged_p = np.concatenate(all_p, axis=0)
        merged_norm = np.concatenate(all_norm, axis=0)
        merged_coefs = np.concatenate(all_coefs, axis=0)
        merged_logs = np.concatenate(all_logs, axis=0)

        # Sort by norm (helpful for DFS in IdealCacheBuilder)
        order = np.argsort(merged_norm, kind="stable")
        merged_p = merged_p[order]
        merged_norm = merged_norm[order]
        merged_coefs = merged_coefs[order]
        merged_logs = merged_logs[order]

        np.savez(args.output,
                 p=merged_p, norm=merged_norm,
                 coefs=merged_coefs, logs=merged_logs)
        total = len(merged_p)
        print(f"\n[done] {total:,} prime ideals → {args.output}", flush=True)
        print(f"  file size: {os.path.getsize(args.output) / 1e6:.1f} MB", flush=True)
        print(f"  total wall: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
