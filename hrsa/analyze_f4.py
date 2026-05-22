#!/usr/bin/env python3
"""f=4 datagen post-process: collect JSONs, build CSV, plot N_D vs log(eps).

Stdlib + numpy + matplotlib only (no pandas).

Inputs (auto-detected, override with --local-dir, --lenore-dir):
  /tmp/canddump_f4_local/*.json
  rsync'd from lenore:.../canddump_f4_lenore/*.json
Outputs (in --out-dir, default /tmp/f4_analysis):
  sweep_f4.csv      # tidy per-cell record
  nd_vs_logeps.png  # N_D vs log10(eps) per theta
  nd_vs_v4.png      # delta vs v4 baseline per cell
"""
import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V4_BASELINE_CSVS = [
    "/home/hlamm/Desktop/efficent_gates/unified/hrsa_sweep_v4_2026-05-10.csv",
    "/home/hlamm/Desktop/efficent_gates/unified/hrsa_sweep_v4_timeouts.csv",
]

def load_json_records(dirs):
    rows = []
    for d in dirs:
        for p in sorted(glob.glob(str(Path(d) / "*.json"))):
            try:
                with open(p) as f:
                    j = json.load(f)
            except Exception as e:
                print(f"  skip {p}: {e}"); continue
            inp = j.get("inputs", {}) or {}
            ach = j.get("achieved", {}) or {}
            dec = j.get("decomposition", {}) or {}
            perf = j.get("performance", {}) or {}
            rows.append({
                "theta": inp.get("theta"),
                "epsilon": inp.get("epsilon"),
                "max_f": inp.get("max_f"),
                "max_solns": inp.get("max_solns"),
                "method": ach.get("method"),
                "success": ach.get("success"),
                "achieved_frob": ach.get("achieved_frob"),
                "f_level": ach.get("f_level"),
                "N_D": dec.get("N_D"),
                "N_D_only": dec.get("N_D_only"),
                "N_R_only": dec.get("N_R_only"),
                "N_T_combined": dec.get("N_T_combined"),
                "wall_s": perf.get("wall_seconds"),
                "source": Path(p).parent.name,
                "path": p,
            })
    return rows

def load_v4_baseline():
    """Merge all v4 baseline sources.  Later sources (e.g. timeout reruns) win
    on overlap since they have actual N_D vs TIMEOUT entries."""
    out = {}
    for path in V4_BASELINE_CSVS:
        if not Path(path).exists(): continue
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    key = (float(r["theta"]), float(r["epsilon"]))
                    nd_str = r.get("N_D", "").strip()
                    nd = int(nd_str) if nd_str else None
                    # Only overwrite if we have a real N_D or no existing entry
                    if nd is not None or key not in out:
                        out[key] = {"method_v4": r.get("method", ""), "N_D_v4": nd}
                except (KeyError, ValueError):
                    continue
    return out

def write_csv(rows, path):
    if not rows: return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def plot_nd_vs_logeps(rows, out_path):
    by_theta = defaultdict(list)
    for r in rows:
        if r["N_D"] is None or not r["success"]: continue
        if r["theta"] is None or r["epsilon"] is None: continue
        by_theta[r["theta"]].append((float(r["epsilon"]), int(r["N_D"])))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for theta, pts in sorted(by_theta.items()):
        pts.sort()
        eps = np.array([p[0] for p in pts])
        nd  = np.array([p[1] for p in pts])
        ax.plot(np.log10(eps), nd, "o-", label=f"θ={theta:.4f}", alpha=0.85)
    ax.set_xlabel("log₁₀(ε)")
    ax.set_ylabel("N_D (D-gate count)")
    ax.set_title("HRSA N_D vs log₁₀(ε) — f≤4 dataset, K_3=1 + lookahead")
    ax.grid(alpha=0.3)
    if by_theta:
        ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_vs_v4_baseline(rows, v4, out_path):
    if not v4:
        print("  no v4 baseline available — skipping comparison plot"); return
    overlap = []
    for r in rows:
        if r["N_D"] is None or not r["success"]: continue
        key = (r["theta"], r["epsilon"])
        if key not in v4: continue
        b = v4[key]
        if b["N_D_v4"] is None: continue
        overlap.append({
            "label": f"θ={r['theta']:.3f} ε={r['epsilon']}",
            "delta": r["N_D"] - b["N_D_v4"],
            "N_D": r["N_D"], "N_D_v4": b["N_D_v4"],
        })
    if not overlap:
        print("  no overlap with v4 baseline — skipping"); return
    overlap.sort(key=lambda x: x["delta"])
    deltas = np.array([x["delta"] for x in overlap])
    labels = [x["label"] for x in overlap]
    colors = ["tab:red" if d > 0 else "tab:green" for d in deltas]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(overlap))))
    pos = np.arange(len(overlap))
    ax.barh(pos, deltas, color=colors)
    ax.set_yticks(pos); ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("ΔN_D (new − v4 baseline)")
    ax.set_title(f"N_D change vs v4 baseline (negative = improvement)\n"
                 f"n={len(overlap)} mean Δ={deltas.mean():+.2f} median Δ={np.median(deltas):+.1f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  vs-v4: n={len(overlap)} mean Δ={deltas.mean():+.2f} min={deltas.min()} max={deltas.max()}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir",  default="/tmp/canddump_f4_local")
    p.add_argument("--lenore-dir", default="/tmp/canddump_f4_lenore_mirror")
    p.add_argument("--pull-lenore", action="store_true")
    p.add_argument("--out-dir", default="/tmp/f4_analysis")
    args = p.parse_args()

    if args.pull_lenore:
        import subprocess
        Path(args.lenore_dir).mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["rsync", "-av", "--include=*.json", "--exclude=*",
             "lenore:/home/hlamm/Desktop/efficent_gates/unified/canddump_f4_lenore/",
             args.lenore_dir + "/"], check=False)
        if rc.returncode != 0:
            print(f"  rsync exited {rc.returncode} (continuing)")

    rows = load_json_records([args.local_dir, args.lenore_dir])
    if not rows:
        print("No JSON records found")
        return 1
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_dir) / "sweep_f4.csv"
    write_csv(rows, out_csv)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    # Compact preview
    for r in sorted(rows, key=lambda x: (x["theta"] or 0, x["epsilon"] or 0)):
        nd = r["N_D"] if r["N_D"] is not None else "—"
        wall = f"{r['wall_s']:.0f}s" if r.get("wall_s") else "—"
        method = r["method"] or "—"
        print(f"  θ={r['theta']} ε={r['epsilon']} {method:18s} N_D={nd}  wall={wall}  ({r['source']})")

    plot_nd_vs_logeps(rows, Path(args.out_dir) / "nd_vs_logeps.png")
    plot_vs_v4_baseline(rows, load_v4_baseline(), Path(args.out_dir) / "nd_vs_v4.png")
    print(f"\nPlots: {args.out_dir}/nd_vs_logeps.png, nd_vs_v4.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
