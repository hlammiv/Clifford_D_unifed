#!/usr/bin/env python3
"""plot_nd_vs_eps_v2.py — N_D vs ε plots, two panels (target ε + achieved frob).

Two panels:
  Left  — N_D vs TARGET ε  (the spec we asked the backend to meet)
  Right — N_D vs ACHIEVED Frobenius distance (what the backend actually found)

Colored by METHOD (not angle). One scatter per (method, ε) cell.

Takes any number of CSVs as positional args. Recognizes:
  - HRSA sweep CSVs (columns: theta, epsilon, method, N_D, achieved_frob, all_checks_pass)
  - zeta9 batched sweep CSVs (columns: max_f, theta, eps_target, achieved_frob, N_D, method, success)
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """Return list of dicts with normalized keys: theta, eps_target, method, N_D, achieved_frob, success.

    For zeta9 CSVs, the recorded method is "zeta9-householder" but the *max-f*
    knob varies per cell — we splice that into the label as "zeta9(max-f=N)"
    so the legend distinguishes different zeta9 capability levels.
    """
    path = Path(path)
    out = []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return out
    # Detect schema
    if "epsilon" in rows[0]:  # HRSA
        for r in rows:
            try:
                if r.get("all_checks_pass") != "true":
                    continue
                out.append({
                    "theta": float(r["theta"]),
                    "eps_target": float(r["epsilon"]),
                    "method": r["method"],
                    "N_D": int(r["N_D"]),
                    "achieved_frob": float(r["achieved_frob"]),
                    "source": path.name,
                })
            except (ValueError, TypeError, KeyError):
                continue
    elif "eps_target" in rows[0]:  # zeta9 batched sweep
        for r in rows:
            try:
                if r.get("success") not in ("True", "true", "1"):
                    continue
                if not r.get("N_D"):
                    continue
                # Relabel "zeta9-householder" → "zeta9(max-f=N)" so the legend
                # distinguishes V-denom levels (max-f=2 today, max-f=3 later).
                base = r["method"]
                mf = r.get("max_f", "")
                if "zeta9" in base and mf:
                    method = f"zeta9(max-f={mf})"
                else:
                    method = base
                out.append({
                    "theta": float(r["theta"]),
                    "eps_target": float(r["eps_target"]),
                    "method": method,
                    "N_D": int(r["N_D"]),
                    "achieved_frob": float(r["achieved_frob"]),
                    "source": path.name,
                })
            except (ValueError, TypeError, KeyError):
                continue
    else:
        sys.stderr.write(f"warning: unknown schema in {path}\n")
    return out


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(
            "usage: plot_nd_vs_eps_v2.py OUT.png CSV [CSV ...]\n"
        )
        sys.exit(1)
    out_png = Path(sys.argv[1])
    csv_paths = [Path(p) for p in sys.argv[2:]]

    pts = []
    for p in csv_paths:
        pts.extend(load_csv(p))
    print(f"loaded {len(pts)} successful cells across {len(csv_paths)} CSVs")
    if not pts:
        sys.exit("no successful cells found")

    # Method cascade order: HRSA_tester tries Direct → SignExtClifford →
    # bidir(k+k) → HRSA(f=k), then zeta9 picks up the long tail at tight ε.
    # Legend reads top-to-bottom in this order; colors progress along the
    # cascade so close-cascade methods have similar hues (viridis).
    method_order = [
        "Direct",
        "SignExtClifford",
        "bidir(0+0)", "bidir(1+1)", "bidir(2+2)", "bidir(3+3)", "bidir(4+4)", "bidir(5+5)",
        "HRSA(f=0)", "HRSA(f=1)", "HRSA(f=2)", "HRSA(f=3)", "HRSA(f=4)",
        "zeta9(max-f=0)", "zeta9(max-f=1)", "zeta9(max-f=2)",
        "zeta9(max-f=3)", "zeta9(max-f=4)",
        "zeta9-householder",  # fallback if max_f wasn't in CSV
    ]
    methods_in_data = sorted(set(p["method"] for p in pts),
                              key=lambda m: (method_order.index(m) if m in method_order else 99, m))
    # viridis traverses dark→light along the cascade. Use ALL known methods
    # for the colormap index space (not just the ones in data) so colors are
    # consistent across replots as more methods appear.
    cmap = plt.get_cmap("viridis", len(method_order))
    method_color = {m: cmap(method_order.index(m)) if m in method_order else (0.5, 0.5, 0.5)
                    for m in methods_in_data}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Group points by method for plotting
    by_method = defaultdict(list)
    for p in pts:
        by_method[p["method"]].append(p)

    for m in methods_in_data:
        ps = by_method[m]
        eps_t = np.array([p["eps_target"] for p in ps])
        frob = np.array([p["achieved_frob"] for p in ps])
        nd = np.array([p["N_D"] for p in ps])
        ax1.scatter(eps_t, nd, c=[method_color[m]], s=25, alpha=0.6, label=f"{m} (n={len(ps)})", edgecolors="none")
        ax2.scatter(frob, nd, c=[method_color[m]], s=25, alpha=0.6, label=f"{m} (n={len(ps)})", edgecolors="none")

    for ax, xlabel, title in [
        (ax1, r"$\varepsilon$ (target Frobenius)", "N_D vs target ε"),
        (ax2, r"achieved Frobenius distance", "N_D vs achieved frob"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$N_D$ (D-gate count)")
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)

    fig.suptitle(f"N_D vs ε — colored by method, all sources stacked", y=1.00)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    sys.exit(main())
