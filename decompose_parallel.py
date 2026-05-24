#!/usr/bin/env python3
"""decompose_parallel.py — parallel decompose driver over existing stage-5 outputs.

Sweep_zeta9_batched.py's per-cell decompose runs SEQUENTIALLY, taking
~1 cell/minute. For 10000 cells that's ~7 days. This driver replays just
the post-stage-5 part: reads existing q_*.npz files, runs decompose in
a multiprocessing pool, writes summary.csv.

Skips cells that already have cell_NNNN.json (idempotent / resumable).

Usage:
  ./decompose_parallel.py \\
      --out_dir /mnt/.../sweep_zeta9_tier2_eps1e-4_2026-05-23 \\
      --n_thetas 10000 --eps 1e-4 --max_f 2 \\
      --workers 16
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Reuse sweep_zeta9_batched's helpers
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sweep_zeta9_batched import (  # type: ignore
    CSV_FIELDS, batched_query_npz_path, extract_best_v, emit_schema_json,
)


def process_cell(arg):
    """Process one cell: extract best V from npz, decompose, write cell JSON, return row."""
    qid, theta, eps, eps_pre, f_u, out_dir, mpi = arg
    out_dir = Path(out_dir)
    json_path = out_dir / f"cell_{qid}.json"
    npz_path = batched_query_npz_path(out_dir, qid)

    V_int, v_f, frob = None, None, None
    npz_path = Path(npz_path) if not isinstance(npz_path, Path) else npz_path
    if npz_path.exists():
        try:
            V_int, v_f, frob = extract_best_v(str(npz_path), f_u)
        except Exception:
            pass
    success = V_int is not None and frob is not None and frob < eps
    method = "zeta9-householder" if success else "none"

    if not json_path.exists():
        try:
            emit_schema_json(
                str(json_path),
                theta=theta, epsilon=eps, max_f=f_u,
                V_int=V_int, v_f=v_f, achieved_frob=frob, success=success,
                wall_seconds=0.0, mpi_ranks=mpi,
                command_line=["decompose_parallel"],
                decompose=True,
            )
        except Exception as exc:
            return {"qid": qid, "theta": theta, "error": repr(exc)}

    N_D = None
    try:
        cell_doc = json.load(open(json_path))
        N_D = (cell_doc.get("decomposition") or {}).get("N_D")
    except Exception:
        pass

    return {
        "max_f": f_u, "theta_idx": int(qid), "theta": theta,
        "eps_target": eps, "eps_pre": eps_pre,
        "success": success, "achieved_frob": frob, "f_level": v_f,
        "N_D": N_D, "method": method,
        "wall_total": 0.0,
        "wall_stage5_batched_share": 0.0,
        "timed_out": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_thetas", type=int, default=10000)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=2 * math.pi)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--eps_pre", type=float, default=5e-5)
    p.add_argument("--max_f", type=int, default=2)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--mpi", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "summary.csv"
    step = (args.theta_max - args.theta_min) / args.n_thetas
    thetas = [args.theta_min + (i + 0.5) * step for i in range(args.n_thetas)]
    qids = [f"{i:04d}" for i in range(args.n_thetas)]

    work = [(qids[i], thetas[i], args.eps, args.eps_pre, args.max_f,
             str(out_dir), args.mpi) for i in range(args.n_thetas)]

    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
    fh.flush()

    print(f"[decompose-parallel] {args.n_thetas} cells, {args.workers} workers", flush=True)
    t0 = time.time()
    n_done = 0
    n_err = 0
    with mp.Pool(args.workers) as pool:
        for row in pool.imap_unordered(process_cell, work):
            if row.get("error"):
                n_err += 1
            else:
                writer.writerow(row)
                fh.flush()
            n_done += 1
            if n_done % 100 == 0 or n_done == args.n_thetas:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (args.n_thetas - n_done) / max(rate, 1e-9)
                print(f"  [{n_done}/{args.n_thetas}] rate={rate:.2f}/s "
                      f"err={n_err} elapsed={elapsed/60:.1f}min "
                      f"eta={eta/60:.1f}min",
                      flush=True)
    fh.close()
    print(f"[decompose-parallel] done in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
