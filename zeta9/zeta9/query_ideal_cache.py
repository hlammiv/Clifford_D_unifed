"""CLI: query the ideal cache for one cell (f, u, ε, θ slice) and produce
a Y1_*.npy file in the same format as collect_targets.

Pipeline
========
1. Load the cache file (mmap) and the fundamental-unit logs.
2. For each cached γ:
   a. Use SigmaFitter to find unit translates u ∈ <u_1, u_2> such that
      M = γ·u·(γ·u)̄ satisfies the σ-strip constraints for this cell.
   b. For each fitting u, compute γ' = γ·u, then M = γ'·γ'̄ via Numba kernel.
   c. Apply the epsy bin filter (final precision check on σ_1).
   d. Output (m_0, m_1, m_2) to the running array.
3. Sort / dedup output array.
4. Write to .npy file in collect_targets's format.

Output format
=============
Same as collect_targets: (N, 3) int64 array, rows = (m_0, m_1, m_2).
The screen is automatically passed (since M = γγ̄ is a K-norm by construction
and class number 1 makes the screen exact).

Important caveats
=================
- The cache must have norm_bound ≥ max N_K/Q(γ) for the cell. For f=6, the
  smallest expected γ-norm is σ_F_1(M)/2 ≈ 3^12/2 ≈ 265K; largest is √(B_max)
  ≈ 7.7×10⁸. Cache should be built with norm_bound ≥ 8×10⁸.
- The σ-fit is a 2D lattice search per cached γ. For most γ's, no unit
  translate satisfies the narrow σ_1 band → 0 hits. Typical hit rate ~10⁻⁴.
- The cache lacks γ multiplied by torsion units (18 roots of unity in K).
  But torsion units have |σ_K_r(ζ)| = 1, so they don't affect log embeddings;
  γ·ζ → M is the same M. So we only enumerate γ up to torsion, no loss.
- Output should equal lattice-first output as a SET — both produce all M's
  satisfying screen + σ-strip + epsy bin. Validation via set comparison.

Usage
=====
    python -m zeta9.query_ideal_cache \\
        --cache /mnt/.../ideal_cache_K=zeta9_B=8e8.bin \\
        --unit_logs /mnt/.../ideal_cache_K=zeta9_B=8e8.bin.units.npy \\
        --f 6 --norm 2 --u 1 --eps 1e-6 \\
        --epsy_lo 0.0 --epsy_hi 0.053 \\
        --output /mnt/.../Y1_f=6_u=1_eps=1e-06_bin_0000_idealfirst
"""
import argparse
import math
import os
import time

import numpy as np

from .ideal_cache import IdealCacheReader, SigmaFitter, load_fund_unit_data
from .ideal_cache_conv import (
    gamma_to_M_batch,
    compute_unit_power_table,
    gamma_times_unit_pow_to_M_batch,
    gamma_times_unit_pow_to_M_and_gamma_batch,
    sigma_fit_batch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Ideal cache file path")
    parser.add_argument("--unit_logs", required=True,
                        help="Fundamental-unit log-embeddings .npy file")
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--norm", type=int, required=True,
                        help="2 for Householder mode, 1 for diagonal")
    parser.add_argument("--u", type=float, required=True,
                        help="|d_i|² target (typically 1)")
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--epsy_lo", type=float, required=True)
    parser.add_argument("--epsy_hi", type=float, required=True)
    parser.add_argument("--output", required=True,
                        help="Output Y1 .npy path (without .npy suffix)")
    parser.add_argument("--k_radius", type=int, default=3,
                        help="Unit-orbit search radius (Babai box)")
    parser.add_argument("--batch_size", type=int, default=10_000,
                        help="Cache scan batch size for memory bound")
    parser.add_argument("--mpi", action="store_true",
                        help="Distribute cache scan across MPI ranks")
    parser.add_argument("--output_gamma", action="store_true",
                        help="Also save γ' (rotated cached γ in O_K standard "
                             "basis) to <output>.gamma.npy — one row per "
                             "unique M, picked from the first σ-fit hit. "
                             "Needed for downstream V reconstruction / HRSA.")
    args = parser.parse_args()

    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    # Load cache + fundamental units
    if rank == 0:
        print(f"=== ideal-cache query ===", flush=True)
        print(f"  f={args.f} norm={args.norm} u={args.u} eps={args.eps}", flush=True)
        print(f"  epsy bin: ({args.epsy_lo}, {args.epsy_hi}]", flush=True)
        print(f"  cache: {args.cache}", flush=True)
        print(f"  mpi ranks: {size}", flush=True)

    cache = IdealCacheReader(args.cache)
    unit_data = load_fund_unit_data(args.unit_logs)
    unit_logs = unit_data["logs"]
    unit_coefs = unit_data.get("coefs")
    unit_inv_coefs = unit_data.get("inv_coefs")

    if rank == 0:
        print(f"  cache: {cache.n_ideals:,} ideals (norm_bound {cache.norm_bound:,})", flush=True)
        print(f"  unit_logs shape: {unit_logs.shape}", flush=True)
        if unit_coefs is None:
            print(f"  WARNING: unit coefs not in file ({args.unit_logs}) — falling back",
                  flush=True)
            print(f"  to (k_1, k_2) = (0, 0) only (will miss most σ-fit hits)",
                  flush=True)

    # Make unit_logs contiguous float64 for Numba kernel
    unit_logs = np.ascontiguousarray(unit_logs, dtype=np.float64)

    # Precompute u^k tables for k ∈ [-k_radius, k_radius]
    if unit_coefs is not None and unit_inv_coefs is not None:
        u1_power_table = compute_unit_power_table(
            unit_coefs[0], unit_inv_coefs[0], args.k_radius,
        )
        u2_power_table = compute_unit_power_table(
            unit_coefs[1], unit_inv_coefs[1], args.k_radius,
        )
        if rank == 0:
            print(f"  precomputed u^k tables for k ∈ [-{args.k_radius}, {args.k_radius}]",
                  flush=True)
    else:
        u1_power_table = None
        u2_power_table = None

    # Compute cell parameters (matching collect_targets internals)
    base_scale = 3 ** (2 * args.f)
    target_t = base_scale * args.u  # σ_F_1 target value (in F's "scaled" units)
    total_scale = args.norm * base_scale

    # Internal "Y" value = σ_F_1(M) - target_t (a small displacement). The bin
    # is on |Y| ∈ (epsy_lo, epsy_hi].
    # → 2 * log|σ_K_1(γ)| ≈ log(target_t)
    # with tolerance derived from epsy bin.

    # σ_F_1 must be in [target_t + epsy_lo, target_t + epsy_hi]
    # In log: 2 L_1 ∈ [log(target_t + epsy_lo), log(target_t + epsy_hi)]
    # The "center" target is log(target_t); tolerance is roughly epsy/target_t in linear,
    # log(1 + epsy/target_t) ≈ epsy/target_t in log.
    if args.epsy_lo <= 0:
        log_target_2L1_lo = math.log(target_t - args.epsy_hi)
    else:
        log_target_2L1_lo = math.log(target_t + args.epsy_lo)
    log_target_2L1_hi = math.log(target_t + args.epsy_hi)
    target_log_2L1 = 0.5 * (log_target_2L1_lo + log_target_2L1_hi)
    tol_log_2L1 = 0.5 * (log_target_2L1_hi - log_target_2L1_lo)

    # σ_F_2, σ_F_3 ∈ [0, total_scale]
    max_log_2L2 = math.log(total_scale)
    max_log_2L3 = math.log(total_scale)

    if rank == 0:
        print(f"  target log(2 L_1): {target_log_2L1:.4f} ± {tol_log_2L1:.4f}", flush=True)
        print(f"  max log(2 L_2) = log(2 L_3) = {max_log_2L2:.4f}", flush=True)

    # Distribute cache among ranks: each rank handles a slice
    n_total = cache.n_ideals
    per_rank = (n_total + size - 1) // size
    lo = rank * per_rank
    hi = min(lo + per_rank, n_total)

    if rank == 0:
        print(f"\n[scan] cache slice per rank ≈ {per_rank:,}", flush=True)

    t0 = time.perf_counter()

    out_counts = {"checked": 0, "fitted": 0, "epsy_pass": 0}

    records = cache.records[lo:hi]
    n_recs = records.shape[0]
    coefs_arr = records["coefs"]
    log_arr = np.ascontiguousarray(records["logsigma"])  # Numba needs contiguous

    have_unit_coefs = u1_power_table is not None

    # ----- σ-fit (vectorized Numba kernel) -----
    # Preallocate output: for each cache record, up to (2·k_radius+1)² hits.
    # Per-record max hits is small (~50 at k_radius=3), but ACTUAL hits are
    # typically 0-2. Allocate conservatively and detect overflow.
    max_hits = n_recs * ((2 * args.k_radius + 1) ** 2)
    # Cap memory: 50M slots × 16 bytes = 800 MB. Most builds need way less.
    max_hits = min(max_hits, 50_000_000)
    out_record_idx = np.empty(max_hits, dtype=np.int64)
    out_k1 = np.empty(max_hits, dtype=np.int64)
    out_k2 = np.empty(max_hits, dtype=np.int64)

    if rank == 0:
        print(f"  σ-fit allocation: {max_hits:,} hit slots ({max_hits * 24 // (1024*1024)} MB)",
              flush=True)

    t_sfit_0 = time.perf_counter()
    n_hits = sigma_fit_batch(
        log_arr,
        target_log_2L1, tol_log_2L1, max_log_2L2, max_log_2L3,
        unit_logs, args.k_radius,
        out_record_idx, out_k1, out_k2,
    )
    t_sfit = time.perf_counter() - t_sfit_0

    if n_hits > max_hits:
        # Overflow — would need to retry. For now, fail loud.
        raise RuntimeError(f"σ-fit overflow: {n_hits} hits > {max_hits} slots. "
                           f"Retry with larger output buffer.")

    out_counts["checked"] = n_recs
    out_counts["fitted"] = n_hits

    if rank == 0:
        print(f"  σ-fit: {n_recs:,} records → {n_hits:,} hits in {t_sfit:.2f}s "
              f"({1e6*t_sfit/n_recs:.1f} µs/record)", flush=True)

    # ----- γ · u^{k_1} · u^{k_2} → M conversion (vectorized) -----
    if n_hits == 0:
        local_M = np.empty((0, 3), dtype=np.int64)
        local_gamma = np.empty((0, 6), dtype=np.int64)
    elif not have_unit_coefs:
        # Legacy fallback: only (0, 0) hits valid
        mask = (out_k1[:n_hits] == 0) & (out_k2[:n_hits] == 0)
        if rank == 0:
            print(f"  WARNING: no unit coefs — restricting to (0,0) hits "
                  f"({int(mask.sum())} of {n_hits})", flush=True)
        n_legacy = int(mask.sum())
        if n_legacy == 0:
            local_M = np.empty((0, 3), dtype=np.int64)
            local_gamma = np.empty((0, 6), dtype=np.int64)
        else:
            legacy_idx = out_record_idx[:n_hits][mask]
            gathered = coefs_arr[legacy_idx].astype(np.int64, copy=False)
            local_M = np.empty((n_legacy, 3), dtype=np.int64)
            gamma_to_M_batch(gathered, local_M)
            local_gamma = gathered  # (k_1, k_2) = (0, 0) → γ' == γ
    else:
        # Full path: gather γ for each hit, then batched γ·u^k → M (+γ')
        gather_idx = out_record_idx[:n_hits]
        hits_gamma = coefs_arr[gather_idx].astype(np.int64, copy=False)
        hits_k1 = out_k1[:n_hits]
        hits_k2 = out_k2[:n_hits]
        local_M = np.empty((n_hits, 3), dtype=np.int64)
        local_gamma = np.empty((n_hits, 6), dtype=np.int64)

        t_conv_0 = time.perf_counter()
        gamma_times_unit_pow_to_M_and_gamma_batch(
            hits_gamma, hits_k1, hits_k2,
            u1_power_table, u2_power_table, args.k_radius,
            local_M, local_gamma,
        )
        t_conv = time.perf_counter() - t_conv_0
        if rank == 0:
            print(f"  γ→(M,γ') conversion: {n_hits:,} rows in {t_conv:.2f}s "
                  f"({1e6*t_conv/max(1,n_hits):.1f} µs/row)", flush=True)

    # local_M was already set in the conversion paths above.
    # Now apply final epsy bin filter: |σ_F_1(M) - target_t| ∈ (epsy_lo, epsy_hi]
    # using floating point with α₁ embedding.
    if local_M.shape[0] > 0:
        alpha1 = 2.0 * math.cos(2.0 * math.pi / 9.0)
        alpha1_sq = alpha1 * alpha1
        s1 = (local_M[:, 0].astype(np.float64)
              + local_M[:, 1].astype(np.float64) * alpha1
              + local_M[:, 2].astype(np.float64) * alpha1_sq)
        y = np.abs(s1 - target_t)
        mask = (y > args.epsy_lo) & (y <= args.epsy_hi)
        local_M = local_M[mask]
        local_gamma = local_gamma[mask]
        out_counts["epsy_pass"] = int(mask.sum())

    t_local = time.perf_counter() - t0

    # MPI gather to rank 0
    if comm is not None:
        all_M = comm.gather(local_M, root=0)
        all_gamma = comm.gather(local_gamma, root=0)
        all_counts = comm.gather(out_counts, root=0)
    else:
        all_M = [local_M]
        all_gamma = [local_gamma]
        all_counts = [out_counts]

    if rank == 0:
        full_M = np.concatenate(all_M, axis=0)
        full_gamma = np.concatenate(all_gamma, axis=0)
        # Dedup on M, keep first γ' per unique M
        full_M, first_idx = np.unique(full_M, axis=0, return_index=True)
        full_gamma = full_gamma[first_idx]

        out_path = args.output if args.output.endswith(".npy") else args.output + ".npy"
        np.save(out_path, full_M)
        if args.output_gamma:
            gamma_path = out_path[:-4] + ".gamma.npy"
            np.save(gamma_path, full_gamma)

        total_checked = sum(c["checked"] for c in all_counts)
        total_fitted = sum(c["fitted"] for c in all_counts)
        total_epsy = sum(c["epsy_pass"] for c in all_counts)

        print(f"\n[done] {t_local:.1f}s (rank 0)", flush=True)
        print(f"  checked: {total_checked:,}", flush=True)
        print(f"  fitted (σ-fit hits): {total_fitted:,}", flush=True)
        print(f"  passed epsy bin: {total_epsy:,}", flush=True)
        print(f"  final unique M rows: {full_M.shape[0]:,}", flush=True)
        print(f"  output: {out_path}", flush=True)
        if args.output_gamma:
            print(f"  output_gamma: {gamma_path}", flush=True)


if __name__ == "__main__":
    main()
