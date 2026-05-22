#!/usr/bin/env python3
# sweep_zeta9_calibration.py — min-frob calibration sweep.
#
# For each (theta, max_f), invokes zeta9_compile.py with a loose epsilon and
# records the achieved Frobenius distance. Goal: see what each f-cap actually
# reaches as a function of theta, independent of any target-eps cutoff.
#
# Stages 1-4 of zeta9 are cached per (f_v, eps_pre, mode) inside the workdir,
# so running 100 thetas at one max_f does the precompute once and then runs
# stage 5 (per-theta search) 100 times.
#
# Output:
#   <out_dir>/zeta9_maxf{f}_t{i:03d}.json   per-cell schema JSON
#   <out_dir>/zeta9_maxf{f}_t{i:03d}.log    per-cell stdout/stderr
#   <out_dir>/summary.csv                    aggregator (appended; safe to resume)

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
ZETA9_COMPILE = UNIFIED / "zeta9_compile.py"
DEFAULT_WORKDIR = UNIFIED / "zeta9"

CSV_FIELDS = [
    "max_f", "theta_idx", "theta", "eps_target", "eps_pre",
    "success", "achieved_frob", "f_level", "N_D", "method",
    "wall", "timed_out",
]


def run_cell(theta_idx, theta, max_f, eps, eps_pre, mpi, workdir, out_dir,
             mode, timeout):
    json_path = out_dir / f"zeta9_maxf{max_f}_t{theta_idx:03d}.json"
    log_path = out_dir / f"zeta9_maxf{max_f}_t{theta_idx:03d}.log"

    cmd = [
        sys.executable, str(ZETA9_COMPILE),
        "--theta", repr(theta),
        "--epsilon", str(eps),
        "--eps-pre", str(eps_pre),
        "--max-f", str(max_f),
        "--mpi", str(mpi),
        "--workdir", str(workdir),
        "--mode", mode,
        "--json", str(json_path),
    ]

    t0 = time.time()
    timed_out = False
    try:
        with open(log_path, "w") as logf:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = time.time() - t0

    row = {
        "max_f": max_f, "theta_idx": theta_idx, "theta": theta,
        "eps_target": eps, "eps_pre": eps_pre,
        "success": False, "achieved_frob": None, "f_level": None,
        "N_D": None, "method": None,
        "wall": wall, "timed_out": timed_out,
    }

    if json_path.exists():
        try:
            d = json.loads(json_path.read_text())
            ach = d.get("achieved") or {}
            row["success"] = bool(ach.get("success", False))
            row["achieved_frob"] = ach.get("achieved_frob")
            row["f_level"] = ach.get("f_level")
            row["method"] = ach.get("method")
            row["N_D"] = (d.get("decomposition") or {}).get("N_D")
        except Exception as e:
            row["method"] = f"parse_error: {e}"
    return row


def load_done_set(csv_path):
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            try:
                done.add((int(r["max_f"]), int(r["theta_idx"])))
            except (KeyError, ValueError):
                continue
    return done


def main():
    p = argparse.ArgumentParser(description="zeta9 min-frob calibration sweep")
    p.add_argument("--n_thetas", type=int, default=100,
                   help="Number of theta values uniform in (theta_min, theta_max)")
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=math.pi / 2)
    p.add_argument("--max_f_min", type=int, default=0)
    p.add_argument("--max_f_max", type=int, default=2,
                   help="Inclusive. Default 2 keeps it local-machine-friendly.")
    p.add_argument("--eps", type=float, default=0.5,
                   help="Loose target eps; success=frob<eps. Default 0.5 so any "
                        "reasonable hit counts.")
    p.add_argument("--eps_pre", type=float, default=None,
                   help="Stage-1 eps floor. Default = eps/2 (wrapper default).")
    p.add_argument("--mpi", type=int, default=4)
    p.add_argument("--mode", choices=["householder", "diagonal"], default="householder")
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    p.add_argument("--out_dir", required=True,
                   help="Where per-cell JSON/log + summary.csv land.")
    p.add_argument("--timeout", type=int, default=3600,
                   help="Per-cell timeout in seconds.")
    p.add_argument("--resume", action="store_true",
                   help="Skip (max_f, theta_idx) pairs already in summary.csv.")
    p.add_argument("--max_f_order", default="ascending",
                   choices=["ascending", "descending"],
                   help="Order to iterate max_f. Ascending builds cheap caches first.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"

    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        sys.exit(f"workdir not found: {workdir}")

    # Uniform theta grid, excluding both endpoints (0 is Clifford-trivial; pi/2
    # is often a Clifford boundary). Use np.linspace endpoints=False semantics.
    n = args.n_thetas
    step = (args.theta_max - args.theta_min) / n
    thetas = [args.theta_min + (i + 0.5) * step for i in range(n)]

    max_fs = list(range(args.max_f_min, args.max_f_max + 1))
    if args.max_f_order == "descending":
        max_fs = max_fs[::-1]

    eps_pre = args.eps_pre if args.eps_pre is not None else (args.eps / 2.0)

    done = load_done_set(csv_path) if args.resume else set()
    n_total = len(max_fs) * n
    n_skip = len([1 for f in max_fs for i in range(n) if (f, i) in done])
    print(f"[sweep] grid: {n} thetas in ({args.theta_min:.4f}, {args.theta_max:.4f}), "
          f"max_f in {max_fs}, eps={args.eps} eps_pre={eps_pre}, mpi={args.mpi}, "
          f"mode={args.mode}")
    print(f"[sweep] workdir={workdir}")
    print(f"[sweep] out_dir={out_dir}")
    print(f"[sweep] cells total={n_total} resume_skip={n_skip} todo={n_total - n_skip}")

    write_header = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        fh.flush()

    sweep_t0 = time.time()
    n_done = 0
    n_todo = n_total - n_skip
    for max_f in max_fs:
        print(f"\n[sweep] === max_f = {max_f} (V-denom = {2*max_f}) ===")
        for i, theta in enumerate(thetas):
            if (max_f, i) in done:
                continue
            t0 = time.time()
            row = run_cell(i, theta, max_f, args.eps, eps_pre, args.mpi,
                           workdir, out_dir, args.mode, args.timeout)
            writer.writerow(row)
            fh.flush()
            n_done += 1
            dt = time.time() - t0
            elapsed = time.time() - sweep_t0
            eta = elapsed / max(n_done, 1) * (n_todo - n_done)
            frob = row["achieved_frob"]
            frob_s = f"{frob:.3e}" if isinstance(frob, (float, int)) else "—"
            print(f"  [maxf={max_f} t={i:03d}/{n} theta={theta:.4f}] "
                  f"success={row['success']} frob={frob_s} "
                  f"N_D={row['N_D']} wall={dt:.1f}s "
                  f"[done {n_done}/{n_todo}, eta {eta/60:.1f}m]")

    fh.close()
    print(f"\n[sweep] complete. csv={csv_path}")


if __name__ == "__main__":
    sys.exit(main())
