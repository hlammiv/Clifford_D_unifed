#!/usr/bin/env python3
"""run_hrsa_fast_baseline.py — Regenerate the HRSA baseline using --fast.

Companion to hrsa_baseline.py.  Produces /tmp/hrsa_baseline_fast.csv with
the same (theta, epsilon) grid as /tmp/hrsa_baseline.csv but with HRSA_tester
invoked under --fast (max_solns=1 + no-lookahead).

It also (optionally) emits a side-by-side comparison Markdown at
/tmp/hrsa_fast_comparison.md.

The grid (30 cells):
    theta ∈ {π/2, π/3, ≈0.89759790102565518 (~2π/7)}
    eps   ∈ {0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001, 5e-4, 1e-4}

Per-cell timeout defaults to 600s (double the prior 300s baseline timeout)
to give --fast a chance at the tight-ε cells.

Usage:
    python3 run_hrsa_fast_baseline.py \
        --out /tmp/hrsa_baseline_fast.csv \
        --compare /tmp/hrsa_baseline.csv \
        --markdown /tmp/hrsa_fast_comparison.md

CSV schema (matches hrsa_baseline.csv):
    theta, epsilon, N_D, walltime, phase, f
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HRSA_DIR = Path("/home/hlamm/Desktop/efficent_gates/unified/hrsa")
HRSA_BIN = HRSA_DIR / "HRSA_tester"

# The same 30 (theta, epsilon) cells as /tmp/hrsa_baseline.csv.
THETAS = [
    ("pi/2", math.pi / 2.0),
    ("pi/3", math.pi / 3.0),
    ("2pi/7", 0.89759790102565518),  # matches the value in the baseline CSV
]
EPS_LIST = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001, 5e-4, 1e-4]


def run_hrsa_fast(theta: float, epsilon: float, max_f: int,
                  timeout_s: float) -> dict:
    """Invoke HRSA_tester --fast and parse JSON.

    Returns:
        {'N_D': int|None, 'walltime': float, 'phase': str, 'f': int|None,
         'error': str|None}
    """
    with tempfile.NamedTemporaryFile(prefix="hrsa_fast_", suffix=".json",
                                     delete=False) as tmp:
        json_path = tmp.name

    cmd = [
        str(HRSA_BIN),
        f"{theta:.17g}",
        f"{epsilon:.17g}",
        str(int(max_f)),
        "--fast",
        "--json", json_path,
    ]

    t0 = time.time()
    err = None
    try:
        subprocess.run(
            cmd,
            cwd=str(HRSA_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        err = f"timeout({timeout_s}s)"
    wall = time.time() - t0

    raw = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            if err is None:
                err = f"json_parse: {e}"
        finally:
            try:
                os.unlink(json_path)
            except OSError:
                pass

    n_d = None
    phase = "none"
    f_level = None
    if raw is not None:
        achieved = raw.get("achieved", {}) or {}
        decomp = raw.get("decomposition", {}) or {}
        unitary = raw.get("unitary", {}) or {}
        phase = achieved.get("method", "none") or "none"
        f_level = achieved.get("f_level")
        if f_level is None and isinstance(unitary, dict):
            f_level = unitary.get("f")
        nd_raw = decomp.get("N_D")
        if isinstance(nd_raw, (int, float)):
            n_d = int(nd_raw)
        perf = raw.get("performance", {}) or {}
        if isinstance(perf.get("wall_seconds"), (int, float)):
            wall = float(perf["wall_seconds"])
        if not achieved.get("success", False):
            n_d = None
            if err is None:
                err = f"no_success(method={phase})"

    return {
        "N_D": n_d,
        "walltime": wall,
        "phase": phase,
        "f": f_level,
        "error": err,
    }


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["theta", "epsilon", "N_D", "walltime", "phase", "f"])
        for r in rows:
            w.writerow([
                f"{r['theta']:.17g}",
                f"{r['epsilon']:g}",
                "" if r["N_D"] is None else int(r["N_D"]),
                f"{r['walltime']:.3f}",
                r["phase"],
                "" if r["f"] is None else int(r["f"]),
            ])
    print(f"[run_hrsa_fast_baseline] wrote {path} ({len(rows)} rows)")


def read_baseline_csv(path: str) -> dict:
    """Returns a dict keyed by (theta_str, eps_str) -> row dict."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r") as fh:
        r = csv.DictReader(fh)
        for row in r:
            # Skip blank rows from trailing newline-only data
            if not row.get("theta") or not row.get("epsilon"):
                continue
            key = (row["theta"], row["epsilon"])
            out[key] = row
    return out


def emit_markdown(default_rows: dict, fast_rows: list, md_path: str) -> None:
    """Write a side-by-side comparison Markdown."""
    lines = []
    lines.append("# HRSA --fast vs default baseline\n")
    lines.append("Per-cell comparison: default mode (from `/tmp/hrsa_baseline.csv`) "
                 "vs `--fast` (this run).\n")
    lines.append("Default-mode timeouts in the original baseline were 300s; "
                 "the `--fast` sweep uses 600s.\n")
    lines.append("")
    lines.append("| θ | ε | N_D (default) | N_D (fast) | walltime default (s) | walltime fast (s) | wall ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    # We need to look up default rows by exact theta string used in baseline CSV.
    # The baseline uses 17g format; we use the same in our CSV.
    nd_savings = 0
    nd_cost = 0
    nd_cost_total = 0
    nd_savings_total = 0
    walltime_default_sum = 0.0
    walltime_fast_sum = 0.0
    rescued_cells = []
    for r in fast_rows:
        theta_s = f"{r['theta']:.17g}"
        eps_s = f"{r['epsilon']:g}"
        key = (theta_s, eps_s)
        d = default_rows.get(key)
        nd_d = "" if (d is None or not d.get("N_D")) else d["N_D"]
        wall_d = "" if d is None else d.get("walltime", "")
        nd_f = "" if r["N_D"] is None else str(int(r["N_D"]))
        wall_f = f"{r['walltime']:.3f}"
        if nd_d != "" and nd_f != "":
            diff = int(nd_f) - int(nd_d)
            if diff < 0:
                nd_savings += 1
                nd_savings_total += -diff
            elif diff > 0:
                nd_cost += 1
                nd_cost_total += diff
        if d is not None and nd_d == "" and nd_f != "":
            rescued_cells.append((theta_s, eps_s, r['N_D'], r['walltime']))
        try:
            wd = float(wall_d)
            wf = float(wall_f)
            walltime_default_sum += wd
            walltime_fast_sum += wf
            ratio = wf / wd if wd > 0 else float("nan")
            ratio_s = f"{ratio:.2f}"
        except (ValueError, TypeError):
            ratio_s = "-"
        lines.append(f"| {theta_s} | {eps_s} | {nd_d or '-'} | {nd_f or '-'} | "
                     f"{wall_d or '-'} | {wall_f} | {ratio_s} |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cells with **higher** N_D under `--fast`: **{nd_cost}** "
                 f"(total extra D-gates: {nd_cost_total})")
    lines.append(f"- Cells with **lower** N_D under `--fast`: **{nd_savings}** "
                 f"(total saved D-gates: {nd_savings_total})")
    lines.append(f"- Aggregate walltime — default: **{walltime_default_sum:.1f}s**, "
                 f"`--fast`: **{walltime_fast_sum:.1f}s** "
                 f"(over cells where both produced a number)")
    if walltime_default_sum > 0:
        speedup = walltime_default_sum / walltime_fast_sum if walltime_fast_sum > 0 else float("nan")
        lines.append(f"- Aggregate speedup: **{speedup:.2f}×**")
    if rescued_cells:
        lines.append("")
        lines.append("### Cells `--fast` pushed to feasibility (default timed out)")
        lines.append("")
        for ts, es, nd, wt in rescued_cells:
            lines.append(f"- θ={ts}, ε={es}: N_D={nd}, walltime={wt:.1f}s")
    else:
        lines.append("")
        lines.append("### Cells `--fast` rescued from default-mode timeout")
        lines.append("")
        lines.append("- None — every cell that timed out at 300s default also failed "
                     "to complete under `--fast` at 600s.")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[run_hrsa_fast_baseline] wrote {md_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/hrsa_baseline_fast.csv")
    ap.add_argument("--compare", default="/tmp/hrsa_baseline.csv")
    ap.add_argument("--markdown", default="/tmp/hrsa_fast_comparison.md")
    ap.add_argument("--max-f", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="Flush CSV every N cells (default 1).")
    args = ap.parse_args()

    if not HRSA_BIN.exists():
        print(f"[ERROR] HRSA_tester not found at {HRSA_BIN}", file=sys.stderr)
        return 2

    rows = []
    t_start = time.time()
    for theta_label, theta_val in THETAS:
        print(f"\n=== theta = {theta_label} ({theta_val:.6f} rad) ===",
              flush=True)
        for eps in EPS_LIST:
            print(f"[fast] theta={theta_val:.6f} eps={eps:g} max_f={args.max_f} "
                  f"timeout={args.timeout}s ... ", end="", flush=True)
            t0 = time.time()
            res = run_hrsa_fast(theta_val, eps,
                                max_f=args.max_f, timeout_s=args.timeout)
            tdone = time.time() - t0
            rec = {
                "theta": theta_val,
                "epsilon": eps,
                "N_D": res["N_D"],
                "walltime": res["walltime"],
                "phase": res["phase"],
                "f": res["f"],
                "error": res.get("error"),
            }
            rows.append(rec)
            extra = (f"  err={rec['error']}" if rec.get("error") else "")
            print(f"N_D={rec['N_D']}  phase={rec['phase']}  f={rec['f']}  "
                  f"wall={rec['walltime']:.2f}s{extra}", flush=True)
            if len(rows) % max(1, args.checkpoint_every) == 0:
                write_csv(args.out, rows)
    total = time.time() - t_start
    print(f"\n[run_hrsa_fast_baseline] DONE in {total:.1f}s ({total/60.0:.1f} min)")

    write_csv(args.out, rows)

    default_rows = read_baseline_csv(args.compare)
    if default_rows:
        emit_markdown(default_rows, rows, args.markdown)
    else:
        print(f"[run_hrsa_fast_baseline] WARN: no baseline at {args.compare}; "
              f"skipping markdown comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
