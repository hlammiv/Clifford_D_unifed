"""CLI: build the one-time ideal cache for K = Q(ζ_9).

Three stages (Stage A is the slow one — see also the MPI variant):
  0. Compute fundamental-unit data (coefs + inv_coefs + log-embeddings). Once.
  A. Enumerate rational primes p ≤ prime_bound, factor (p) in O_K, get a
     generator of each prime ideal of K. Save prime table.
  B. DFS-enumerate composite ideals up to norm_bound using incremental
     generator multiplication. Save ideal cache.

Stages 0 and A can be done in parallel via build_prime_table_mpi.py:
    mpirun -n 32 python -m zeta9.build_prime_table_mpi \\
        --prime_bound 800000000 --norm_bound 800000000 \\
        --output primes.npz
Then for stage B:
    python -m zeta9.build_ideal_cache \\
        --prime_table_path primes.npz --resume \\
        --norm_bound 800000000 --output cache.bin

Usage (single-process, all stages):
    python -m zeta9.build_ideal_cache \\
        --norm_bound 800000000 \\
        --output /mnt/.../zeta9_D/ideal_cache_K=zeta9_B=8e8.bin

For f=6 we need norm_bound ≈ 7.7×10⁸ (B = 6×10¹⁷ for M = γγ̄, so γ-norm
bound is √B). Cache size ≈ 14 GB.
"""
import argparse
import os
import time

from .ideal_cache import (
    IdealCacheBuilder,
    compute_fundamental_unit_data,
    save_fund_unit_data,
    CACHE_HEADER_SIZE,
    IDEAL_RECORD_DTYPE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_bound", type=int, required=True,
                        help="Max N_K/Q(γ) to enumerate")
    parser.add_argument("--prime_bound", type=int, default=None,
                        help="Max rational prime p to consider (default: norm_bound)")
    parser.add_argument("--output", required=True,
                        help="Output cache file path (binary)")
    parser.add_argument("--prime_table_path", default=None,
                        help="Path to save/load prime table (.npz). "
                             "Default: derive from --output")
    parser.add_argument("--unit_logs_path", default=None,
                        help="Path to save fundamental-unit log-embeddings. "
                             "Default: derive from --output")
    parser.add_argument("--pari_stack_gb", type=int, default=8,
                        help="PARI stack size in GiB (default 8)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing prime table (skip Stage A)")
    args = parser.parse_args()

    if args.prime_table_path is None:
        args.prime_table_path = args.output + ".primes.npz"
    if args.unit_logs_path is None:
        # New format: .npz with coefs + inv_coefs + logs (not just logs)
        args.unit_logs_path = args.output + ".units.npz"

    print(f"=== ideal cache build ===", flush=True)
    print(f"  K = Q(ζ_9), defining poly: x^6 + x^3 + 1", flush=True)
    print(f"  norm_bound: {args.norm_bound:,}", flush=True)
    print(f"  prime_bound: {args.prime_bound or args.norm_bound:,}", flush=True)
    print(f"  output: {args.output}", flush=True)
    print(f"  prime_table: {args.prime_table_path}", flush=True)
    print(f"  unit_logs: {args.unit_logs_path}", flush=True)
    print(f"  PARI stack: {args.pari_stack_gb} GiB", flush=True)

    builder = IdealCacheBuilder(
        norm_bound=args.norm_bound,
        prime_bound=args.prime_bound,
        pari_stack_bytes=args.pari_stack_gb * (1 << 30),
    )

    # Stage 0: fundamental unit data (coefs, inv_coefs, logs)
    if not os.path.exists(args.unit_logs_path):
        print(f"\n[stage 0] computing fundamental unit data (coefs + inv + logs)...",
              flush=True)
        t0 = time.perf_counter()
        unit_data = compute_fundamental_unit_data(
            pari_stack_bytes=args.pari_stack_gb * (1 << 30),
        )
        t1 = time.perf_counter()
        print(f"  unit coefs:\n{unit_data['coefs']}", flush=True)
        print(f"  unit inv_coefs:\n{unit_data['inv_coefs']}", flush=True)
        print(f"  unit logs:\n{unit_data['logs']}", flush=True)
        print(f"  ({t1 - t0:.1f}s)", flush=True)
        save_fund_unit_data(unit_data, args.unit_logs_path)
    else:
        print(f"\n[stage 0] reusing existing unit data: {args.unit_logs_path}",
              flush=True)

    # Stage A: prime table
    if args.resume and os.path.exists(args.prime_table_path):
        print(f"\n[stage A] resuming from existing prime table", flush=True)
        t0 = time.perf_counter()
        builder.load_prime_table(args.prime_table_path)
        t1 = time.perf_counter()
        print(f"  loaded {len(builder._primes):,} prime ideals in {t1 - t0:.1f}s", flush=True)
    else:
        print(f"\n[stage A] building prime table", flush=True)
        t0 = time.perf_counter()
        builder.build_prime_table(save_path=args.prime_table_path)
        t1 = time.perf_counter()
        n_primes = len(builder._primes)
        print(f"  {n_primes:,} prime ideals enumerated in {t1 - t0:.1f}s "
              f"({(t1 - t0) / n_primes * 1e3:.2f} ms/prime)", flush=True)

    # Stage B: composite ideal DFS
    print(f"\n[stage B] DFS-enumerating composite ideals → {args.output}", flush=True)
    t0 = time.perf_counter()
    n_ideals = builder.build_ideal_cache(args.output)
    t1 = time.perf_counter()
    cache_size = os.path.getsize(args.output)
    print(f"  {n_ideals:,} ideals → {cache_size / 1e9:.2f} GB "
          f"in {t1 - t0:.1f}s ({(t1 - t0) / n_ideals * 1e6:.1f} µs/ideal)", flush=True)

    print(f"\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
