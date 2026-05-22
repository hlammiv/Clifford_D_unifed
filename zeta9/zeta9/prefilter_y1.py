"""MPI pre-filter for unscreened Y1*.npy files produced by
`collect_targets --no_stage1_screen`.

Runs the same exact ideal screen that stage 1 would have run, but as a
clean parallel pass between stage 1 and stage 2. Each rank reads a slice
of the input .npy, calls `quick_screen_M_fast` per row, and writes its
survivors to a per-rank temp file. Rank 0 then concatenates and writes
the final .npy.

Required when stage 1 was run with --no_stage1_screen, otherwise stage 2
will be fed unscreened rows and cross-join can explode.

Algorithm:
    input  = Y1_raw.npy    # all (n0, n1, n2) rows from stage 1, NOT screened
    output = Y1.npy        # screened survivors (same format as the
                             original screened stage-1 output)

Usage:
    mpirun -n 32 python3 -m zeta9.prefilter_y1 \\
        --input Y1_raw.npy --output Y1.npy
"""
import os
import sys
import time
import argparse
import numpy as np
from mpi4py import MPI

from .roots import (
    build_fields,
    quick_screen_M,
    quick_screen_M_fast,
)
from .select_triples_optimized import ROW_DTYPE, _append_rows_raw

# Optional: import the fast (Numba) screen replacement.
# If roots_fast is unavailable (e.g., table not built), fall back to Sage.
try:
    from . import roots_fast
    _HAVE_FAST_SCREEN = roots_fast._table_ready
except Exception as _e:
    roots_fast = None
    _HAVE_FAST_SCREEN = False


def _screen_rows(rows, field_data, allow_negative_embeddings, check_local_p3k):
    """Run the exact ideal screen on each row. Returns boolean keep_mask."""
    n = rows.shape[0]
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        r = rows[i]
        if quick_screen_M_fast is not None:
            scr = quick_screen_M_fast(
                int(r[0]), int(r[1]), int(r[2]),
                check_real_embeddings=not allow_negative_embeddings,
                check_local_p3k=check_local_p3k,
                field_data=field_data,
            )
        else:
            scr = quick_screen_M(
                int(r[0]), int(r[1]), int(r[2]),
                check_real_embeddings=not allow_negative_embeddings,
                check_local_p3k=check_local_p3k,
                field_data=field_data,
            )
        status = scr["status"]
        if status == "PASSES_IDEAL_SIEVE" or status == "ZERO":
            keep[i] = True
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="unscreened Y1*.npy from collect_targets --no_stage1_screen")
    parser.add_argument("--output", required=True, help="screened Y1*.npy (overwrites if exists)")
    parser.add_argument("--check_local_p3k", action="store_true")
    parser.add_argument("--allow_negative_embeddings", action="store_true")
    parser.add_argument("--workdir", default=None,
                        help="per-rank temp dir (default: output + '.prefilter_parts')")
    parser.add_argument("--report_every", type=int, default=50000,
                        help="rank-0 progress every N rows (default 50k)")
    parser.add_argument("--use_fast_screen", action="store_true",
                        help="use the Numba roots_fast screen (1000× faster); falls back to Sage for rows outside the deg-1 table bound or for huge magnitudes")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    workdir = args.workdir or (args.output + ".prefilter_parts")
    if rank == 0:
        os.makedirs(workdir, exist_ok=True)
        # cleanup leftovers
        for fn in os.listdir(workdir):
            if fn.startswith("survivors_rank_"):
                try:
                    os.remove(os.path.join(workdir, fn))
                except OSError:
                    pass
    comm.Barrier()

    t0 = time.time()

    # Load metadata to determine row count
    if rank == 0:
        full = np.load(args.input, mmap_mode="r")
        n_total = int(full.shape[0])
        del full
        print(f"[prefilter] input={args.input}  total_rows={n_total}", flush=True)
        print(f"[prefilter] ranks={size}  rows/rank≈{n_total // size}", flush=True)
    else:
        n_total = None
    n_total = comm.bcast(n_total, root=0)

    # Each rank reads its slice (mmap is fine; we don't need it all in memory)
    per_rank = (n_total + size - 1) // size
    lo = rank * per_rank
    hi = min(lo + per_rank, n_total)

    if lo >= hi:
        local_rows = np.empty((0, 3), dtype=np.int64)
    else:
        full = np.load(args.input, mmap_mode="r")
        local_rows = np.ascontiguousarray(full[lo:hi], dtype=np.int64)
        del full
    local_n = local_rows.shape[0]

    use_fast = bool(args.use_fast_screen and _HAVE_FAST_SCREEN)
    if rank == 0:
        print(f"[prefilter] screen mode: {'FAST (roots_fast Numba)' if use_fast else 'SAGE'}", flush=True)

    field_data = build_fields() if build_fields is not None else None

    # Sage fallback closure for fast-screen FALLBACK_REQUIRED rows
    def _sage_fb(m0, m1, m2):
        if quick_screen_M_fast is not None:
            r = quick_screen_M_fast(
                m0, m1, m2,
                check_real_embeddings=not args.allow_negative_embeddings,
                check_local_p3k=args.check_local_p3k,
                field_data=field_data,
            )
        else:
            r = quick_screen_M(
                m0, m1, m2,
                check_real_embeddings=not args.allow_negative_embeddings,
                check_local_p3k=args.check_local_p3k,
                field_data=field_data,
            )
        return r["status"] in ("PASSES_IDEAL_SIEVE", "ZERO")

    # Screen in mini-chunks so we can report progress + bound peak memory
    chunk_size = 50000 if use_fast else 5000
    survivors = []
    n_done = 0
    n_kept = 0
    n_fallback_total = 0
    last_report = time.time()
    while n_done < local_n:
        end = min(n_done + chunk_size, local_n)
        sub = local_rows[n_done:end]
        if use_fast:
            keep, n_fb = roots_fast.screen_rows_batch(
                sub,
                check_real_embeddings=not args.allow_negative_embeddings,
                sage_fallback_fn=_sage_fb,
            )
            n_fallback_total += int(n_fb)
        else:
            keep = _screen_rows(
                sub, field_data,
                allow_negative_embeddings=args.allow_negative_embeddings,
                check_local_p3k=args.check_local_p3k,
            )
        kept = sub[keep]
        if kept.size:
            survivors.append(np.ascontiguousarray(kept))
            n_kept += kept.shape[0]
        n_done = end
        if rank == 0 and (n_done % args.report_every == 0 or n_done == local_n) \
                and (time.time() - last_report > 5.0):
            last_report = time.time()
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (local_n - n_done) / rate if rate > 0 else 0
            fb_note = f" fallback={n_fallback_total}" if use_fast else ""
            print(f"[prefilter] rank 0: {n_done}/{local_n} ({100*n_done/local_n:.1f}%) "
                  f"kept={n_kept} rate={rate:.0f} rows/sec eta={eta:.0f}s{fb_note}",
                  flush=True)

    # Concat and write per-rank survivor file
    if survivors:
        out = np.concatenate(survivors, axis=0)
    else:
        out = np.empty((0, 3), dtype=np.int64)
    survivor_path = os.path.join(workdir, f"survivors_rank_{rank:04d}.bin")
    _append_rows_raw(survivor_path, out)

    local_walltime = time.time() - t0
    walltimes = comm.gather(local_walltime, root=0)
    total_kept_arr = comm.gather(int(n_kept), root=0)
    total_done_arr = comm.gather(int(local_n), root=0)

    comm.Barrier()

    if rank != 0:
        return

    # Rank 0: concat all per-rank survivor files into the final .npy
    total_bytes = 0
    paths = []
    for r in range(size):
        p = os.path.join(workdir, f"survivors_rank_{r:04d}.bin")
        if os.path.exists(p):
            paths.append(p)
            total_bytes += os.path.getsize(p)
    bytes_per_row = 3 * 8
    if total_bytes % bytes_per_row != 0:
        raise RuntimeError(f"survivor bytes {total_bytes} not divisible by {bytes_per_row}")
    n_rows = total_bytes // bytes_per_row

    save_path = args.output if args.output.endswith(".npy") else args.output + ".npy"
    with open(save_path, "wb") as fh:
        np.lib.format.write_array_header_1_0(fh, {
            "descr": "<i8",
            "fortran_order": False,
            "shape": (int(n_rows), 3),
        })
        buf_sz = 8 * 1024 * 1024
        for p in paths:
            with open(p, "rb") as src:
                while True:
                    chunk = src.read(buf_sz)
                    if not chunk:
                        break
                    fh.write(chunk)

    # Cleanup per-rank files
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(workdir)
    except OSError:
        pass

    elapsed = time.time() - t0
    total_kept = sum(total_kept_arr)
    total_done = sum(total_done_arr)
    print(f"[prefilter] done in {elapsed:.1f}s", flush=True)
    print(f"[prefilter] screened {total_done} rows, kept {total_kept} ({100*total_kept/max(1,total_done):.2f}%) -> {save_path}", flush=True)
    print(f"[prefilter] per-rank walltimes: min={min(walltimes):.1f}s max={max(walltimes):.1f}s avg={sum(walltimes)/len(walltimes):.1f}s", flush=True)


if __name__ == "__main__":
    main()
