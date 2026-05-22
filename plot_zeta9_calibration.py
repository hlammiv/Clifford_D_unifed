#!/usr/bin/env python3
# plot_zeta9_calibration.py — visualize min-frob calibration.
# Two panels: (left) achieved frob vs theta; (right) N_D vs theta.
# One curve per max_f.

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/home/hlamm/Desktop/efficent_gates/unified/sweep_zeta9_cal_2026-05-22/summary.csv")
OUT_PNG = Path(sys.argv[2] if len(sys.argv) > 2 else
               "/home/hlamm/Desktop/efficent_gates/unified/zeta9_calibration_2026-05-22.png")

rows = []
with open(CSV_PATH) as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

by_f = {}
for r in rows:
    if r["success"] != "True":
        continue
    f = int(r["max_f"])
    by_f.setdefault(f, []).append((
        float(r["theta"]),
        float(r["achieved_frob"]) if r["achieved_frob"] else None,
        int(r["N_D"]) if r["N_D"] else None,
    ))

colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for f in sorted(by_f):
    data = sorted(by_f[f])
    thetas = np.array([d[0] for d in data])
    frobs = np.array([d[1] for d in data])
    nds = np.array([d[2] for d in data])
    label = f"max-f={f} (V-denom={2*f})"
    ax1.plot(thetas, frobs, "o-", color=colors[f], markersize=3, lw=1, label=label)
    ax2.plot(thetas, nds, "o-", color=colors[f], markersize=3, lw=1, label=label)

# Reach-floor reference lines: 3^(-V-denom)
for f in sorted(by_f):
    floor = 3.0 ** (-2 * f)
    if floor > 1e-5:
        ax1.axhline(floor, color=colors[f], ls="--", alpha=0.3,
                    label=f"  3^-{2*f} = {floor:.3g}")

ax1.set_xlabel(r"$\theta$ (rad)")
ax1.set_ylabel("achieved Frobenius distance")
ax1.set_yscale("log")
ax1.set_title("zeta9 min-frob calibration (loose ε=0.5)")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(True, alpha=0.3)

ax2.set_xlabel(r"$\theta$ (rad)")
ax2.set_ylabel("$N_D$ (D-gate count)")
ax2.set_title("Decomposition cost")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)

fig.suptitle(f"zeta9 calibration sweep — {CSV_PATH.name}", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"cells plotted: {sum(len(v) for v in by_f.values())}")
print(f"per max_f: {{f: len for f, len in [(f, len(by_f[f])) for f in sorted(by_f)]}}")
