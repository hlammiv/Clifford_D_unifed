#!/usr/bin/env python3
"""sweep_hrsa_grid.py — HRSA sweep on a uniform θ grid (matches zeta9 batched sweep grid).

Same θ grid as sweep_zeta9_batched.py default (100 uniform in (0, π/2)) so
the two CSVs overlay cleanly for the N_D vs ε paper plot.

Outputs one CSV row per (θ, ε) cell with HRSA's achieved method, N_D, frob,
wall, validate status. CSV columns match sweep_hrsa.py for compatibility.

Usage:
  ./sweep_hrsa_grid.py --n_thetas 100 --max_f 3 \\
      --eps 0.5,0.1,0.01,0.001 --out_dir /tmp/hrsa_grid
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
HRSA_TESTER = UNIFIED / "hrsa" / "HRSA_tester"
V_VALIDATE = UNIFIED / "v_validate.py"

CSV_FIELDS = [
    "theta_idx", "theta", "epsilon", "method", "N_D",
    "N_D_only", "N_R_only", "N_T", "f", "achieved_frob",
    "validate_frob", "wall_s", "all_checks_pass",
]


def run_one(theta, eps, max_f, out_dir, timeout, use_bidir=0):
    label = f"t{theta:.6f}_eps{eps:g}".replace(".", "_")
    json_path = out_dir / f"hrsa_{label}.json"
    log_path = out_dir / f"hrsa_{label}.log"
    # --max-solns 20: invoke HRSA_bestD which collects up to 20 valid candidates
    # at the first successful f-level, decomposes each, and returns the one with
    # min D-count. See memory `hrsa_first_hit_bias.md`.
    # --use-bidir N: iterate bidir K=2..N greedily before falling to HRSA — fills
    # the N_D=2-10 gap left by Direct/SignExtClifford. See memory
    # `hrsa_first_hit_bias.md` + the cascade_unified_2026-05-23 plan.
    cmd = [str(HRSA_TESTER), repr(theta), str(eps), str(max_f),
           "--max-solns", "20",
           "--json", str(json_path)]
    if use_bidir > 0:
        cmd += ["--use-bidir", str(use_bidir)]
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
        "theta_idx": None, "theta": theta, "epsilon": eps,
        "method": "TIMEOUT" if timed_out else None,
        "N_D": "", "N_D_only": "", "N_R_only": "", "N_T": "",
        "f": "", "achieved_frob": "", "validate_frob": "",
        "wall_s": f"{wall:.3f}", "all_checks_pass": "false",
    }
    if json_path.exists():
        try:
            d = json.loads(json_path.read_text())
            ach = d.get("achieved", {}) or {}
            row["method"] = ach.get("method") or "none"
            row["achieved_frob"] = ach.get("achieved_frob") or ""
            row["f"] = ach.get("f_level") or ""
            decomp = d.get("decomposition", {}) or {}
            row["N_D"] = decomp.get("N_D") if decomp.get("N_D") is not None else ""
            row["N_D_only"] = decomp.get("N_D_only") if decomp.get("N_D_only") is not None else row["N_D"]
            row["N_R_only"] = decomp.get("N_R_only") if decomp.get("N_R_only") is not None else 0
            row["N_T"] = decomp.get("N_total") if decomp.get("N_total") is not None else ""
        except Exception as e:
            row["method"] = f"parse_error:{e}"
        try:
            vp = subprocess.run([str(V_VALIDATE), str(json_path)],
                                capture_output=True, text=True, timeout=30)
            if vp.stdout:
                vd = json.loads(vp.stdout)
                row["validate_frob"] = vd.get("actual_frob", "")
                row["all_checks_pass"] = "true" if vd.get("all_checks_pass") else "false"
        except Exception:
            pass
    return row


def main():
    p = argparse.ArgumentParser(description="HRSA grid sweep on uniform θ")
    p.add_argument("--n_thetas", type=int, default=100)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=math.pi / 2)
    p.add_argument("--max_f", type=int, default=3,
                   help="HRSA u-denom cap.")
    p.add_argument("--eps", default="0.5,0.1,0.01,0.001",
                   help="Comma-separated ε values.")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-cell timeout in seconds. Default 600 (10 min).")
    p.add_argument("--use-bidir", type=int, default=0,
                   help="Pass --use-bidir N to HRSA_tester (try bidir K=2..N before HRSA). "
                        "Default 0 = bidir disabled.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resume", action="store_true",
                   help="Skip (theta_idx, epsilon) pairs already in summary.csv.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"

    step = (args.theta_max - args.theta_min) / args.n_thetas
    thetas = [args.theta_min + (i + 0.5) * step for i in range(args.n_thetas)]
    eps_list = [float(e) for e in args.eps.split(",")]

    done = set()
    if args.resume and csv_path.exists():
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                try:
                    done.add((int(r["theta_idx"]), float(r["epsilon"])))
                except (ValueError, KeyError):
                    pass

    n_total = len(thetas) * len(eps_list)
    n_skip = len([1 for i in range(len(thetas)) for e in eps_list if (i, e) in done])
    print(f"[hrsa-grid] {len(thetas)} thetas × {len(eps_list)} eps = {n_total} cells "
          f"({n_skip} cached, {n_total - n_skip} todo), out_dir={out_dir}")

    write_header = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        fh.flush()

    t_sweep = time.time()
    n_done = 0
    n_todo = n_total - n_skip
    # 2026-05-22: ε-outer, θ-inner. Each ε row completes for all 100 θ before
    # the next (harder) ε row starts. This way the easier rows finish quickly
    # for the paper plot, and the tight-ε row (which has many TIMEOUTs) doesn't
    # block per-θ progress on the easier rows.
    for eps in eps_list:
        for i, theta in enumerate(thetas):
            if (i, eps) in done:
                continue
            t0 = time.time()
            row = run_one(theta, eps, args.max_f, out_dir, args.timeout,
                          use_bidir=args.use_bidir)
            row["theta_idx"] = i
            writer.writerow(row)
            fh.flush()
            n_done += 1
            elapsed = time.time() - t_sweep
            eta = elapsed / max(n_done, 1) * (n_todo - n_done)
            print(f"  [t={i:03d} eps={eps:g}] method={row['method']} "
                  f"N_D={row['N_D']} frob={row['achieved_frob']} "
                  f"wall={time.time()-t0:.1f}s [done {n_done}/{n_todo}, "
                  f"eta {eta/60:.1f}m]")
    fh.close()
    print(f"\n[hrsa-grid] complete in {(time.time()-t_sweep)/60:.1f}m")
    print(f"[hrsa-grid] csv={csv_path}")


if __name__ == "__main__":
    sys.exit(main())
