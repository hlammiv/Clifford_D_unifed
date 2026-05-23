#!/usr/bin/env python3
"""sweep_zeta9_batched.py — batched θ-sweep using stage-5 --queries mode.

Per Round-2 refactor C1 (2026-05-22): the wrapper currently invokes stage 5
once per (θ, ε) cell, paying the mpirun + Numba JIT + sidecar mmap warmup +
root-enumeration startup ~N times for an N-cell sweep. Stage 5 already exposes
a batched mode (search_householder_streamed_batched) that consumes a list of
queries sharing one ε. This driver wires it up.

Cells are grouped by (max_f, eps_pre, eps_target, mode). Each group does:
  - ensure_precompute() ONCE
  - run_stage5_batched()  ONCE for all θ in the group
  - extract_best_v() per query; persist V + decompose to per-cell JSON + CSV

For a 100-θ × single-(max_f, ε) sweep the typical saving is 2-5× total wall.

Usage examples:

  # min-frob calibration (all cells share eps=0.5):
  ./sweep_zeta9_batched.py --n_thetas 100 --max_f 2 --eps 0.5 \\
      --mpi 24 --out_dir /mnt/.../sweep_calib_f4

  # fixed-eps row at ε=10⁻³ (max-f=2 = internal f=4):
  ./sweep_zeta9_batched.py --n_thetas 100 --max_f 2 --eps 0.001 \\
      --mpi 24 --out_dir /mnt/.../sweep_eps1e-3_f4

  # multiple max-f rows (separate runs per max-f, since precompute is per-(max-f, eps_pre)):
  for mf in 0 1 2; do ./sweep_zeta9_batched.py --max_f $mf --eps 0.5 ...; done
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

UNIFIED = Path(__file__).resolve().parent
sys.path.insert(0, str(UNIFIED))

# Reuse the wrapper's helpers
from zeta9_compile import (
    artifact_paths,
    batched_query_npz_path,
    DEFAULT_SAGE_ENV,
    emit_schema_json,
    ensure_precompute,
    extract_best_v,
    run_hrsa_decompose,
    run_stage5_batched,
    ZETA9_DIR,
)

CSV_FIELDS = [
    "max_f", "theta_idx", "theta", "eps_target", "eps_pre",
    "success", "achieved_frob", "f_level", "N_D", "method",
    "wall_total", "wall_stage5_batched_share", "timed_out",
]


def main():
    p = argparse.ArgumentParser(description="zeta9 batched θ-sweep (Round 2 C1)")
    p.add_argument("--n_thetas", type=int, default=100)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=math.pi / 2)
    p.add_argument("--max_f", type=int, required=True,
                   help="u-denom cap (wrapper sense). V-denom = 2*max_f.")
    p.add_argument("--eps", type=float, required=True,
                   help="ε for all queries (uniform per batched run, V1 constraint).")
    p.add_argument("--eps_pre", type=float, default=None,
                   help="Stage-1 σ-strip floor. Default = eps/2.")
    p.add_argument("--mpi", type=int, default=8)
    p.add_argument("--mode", choices=["householder", "diagonal"], default="householder")
    p.add_argument("--workdir", default=None,
                   help="zeta9 D/ workdir. Default = ZETA9_DIR.")
    p.add_argument("--sage_env", default=DEFAULT_SAGE_ENV)
    p.add_argument("--out_dir", required=True,
                   help="Output dir for per-cell JSON, queries.json, q_*.npz, summary.csv.")
    p.add_argument("--no_decompose", action="store_true",
                   help="Skip HRSA decomposition (N_D will be None). Use for cheap "
                        "min-frob calibration where N_D isn't the headline.")
    p.add_argument("--check_local_p3k", action="store_true",
                   help="Force-enable p3k pre-screen. Auto-on at eps ≤ 1e-4 normally.")
    args = p.parse_args()

    eps_pre = args.eps_pre if args.eps_pre is not None else (args.eps / 2.0)
    workdir = Path(args.workdir).resolve() if args.workdir else ZETA9_DIR
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    f_u = args.max_f
    f_v = 2 * f_u

    # Auto p3k at tight eps (matches single-cell wrapper behavior).
    check_local_p3k = args.check_local_p3k or (args.eps <= 1.0e-4)

    # Uniform θ grid (matches sweep_zeta9_calibration's convention).
    n = args.n_thetas
    step = (args.theta_max - args.theta_min) / n
    thetas = [args.theta_min + (i + 0.5) * step for i in range(n)]
    qids = [f"{i:04d}" for i in range(n)]

    print(f"[batched-sweep] max_f={f_u} (f_v={f_v}), mode={args.mode}, "
          f"eps={args.eps} eps_pre={eps_pre}, p3k={check_local_p3k}")
    print(f"[batched-sweep] {n} thetas in ({args.theta_min:.4f}, {args.theta_max:.4f}), "
          f"mpi={args.mpi}, workdir={workdir}")
    print(f"[batched-sweep] out_dir={out_dir}")

    t_overall = time.time()

    # --- precompute (stages 1-4) -----
    _lazy_u = (1.0, 1.0, 0.0) if args.mode == "householder" else None
    times14 = ensure_precompute(
        workdir, f_v, eps_pre, args.mpi, args.sage_env, dry_run=False,
        check_local_p3k=check_local_p3k, mode=args.mode,
        eps_target=args.eps, lazy_u_values=_lazy_u,
    )
    print(f"[batched-sweep] precompute walls: {times14}")

    # --- batched stage 5 -----
    t_stage5 = time.time()
    run_stage5_batched(
        workdir, f_v, eps_pre, args.eps, thetas, args.mpi, args.sage_env,
        output_dir=out_dir, dry_run=False,
        check_local_p3k=check_local_p3k, mode=args.mode,
        query_ids=qids,
    )
    stage5_wall = time.time() - t_stage5
    per_cell_share = stage5_wall / max(n, 1)
    print(f"[batched-sweep] stage 5 batched done in {stage5_wall:.1f}s "
          f"({per_cell_share:.2f}s/cell amortized)")

    # --- extract per-query V + decompose + per-cell JSON + CSV row -----
    csv_path = out_dir / "summary.csv"
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()

    n_success = 0
    n_failed = 0
    for i, (theta, qid) in enumerate(zip(thetas, qids)):
        cell_t0 = time.time()
        npz_path = batched_query_npz_path(out_dir, qid)
        V_int, v_f, frob = (None, None, None)
        if os.path.exists(npz_path):
            V_int, v_f, frob = extract_best_v(npz_path, f_u)
        success = V_int is not None and frob is not None and frob < args.eps
        method = "zeta9-householder" if success else "none"

        # per-cell schema JSON. emit_schema_json optionally decomposes V to
        # populate decomposition block. Single decompose per cell.
        json_path = out_dir / f"cell_{qid}.json"
        emit_schema_json(
            str(json_path),
            theta=theta, epsilon=args.eps, max_f=f_u,
            V_int=V_int, v_f=v_f, achieved_frob=frob, success=success,
            wall_seconds=per_cell_share, mpi_ranks=args.mpi,
            command_line=sys.argv,
            decompose=(not args.no_decompose),
        )

        # Read back N_D from the emitted JSON (avoids double decompose).
        N_D = None
        if not args.no_decompose:
            try:
                cell_doc = json.load(open(json_path))
                N_D = (cell_doc.get("decomposition") or {}).get("N_D")
            except Exception:
                pass
        cell_wall = time.time() - cell_t0
        writer.writerow({
            "max_f": f_u, "theta_idx": i, "theta": theta,
            "eps_target": args.eps, "eps_pre": eps_pre,
            "success": success, "achieved_frob": frob, "f_level": v_f,
            "N_D": N_D, "method": method,
            "wall_total": per_cell_share + cell_wall,
            "wall_stage5_batched_share": per_cell_share,
            "timed_out": False,
        })
        fh.flush()
        if success:
            n_success += 1
        else:
            n_failed += 1

    fh.close()
    total_wall = time.time() - t_overall
    print(f"\n[batched-sweep] {n_success}/{n} success, {n_failed} fail, "
          f"wall={total_wall:.1f}s ({total_wall/n:.2f}s/cell)")
    print(f"[batched-sweep] summary: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())
