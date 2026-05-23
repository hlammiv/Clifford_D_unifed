#!/usr/bin/env python3
"""plot_nd_vs_eps.py — N_D vs ε across HRSA + zeta9 backends.

Replaces the lost-to-/tmp version. Layout matches hrsa_v3_ND_vs_eps_2026-05-09:
  X: ε (log scale, descending)
  Y: N_D (D-gate count)
  Color: θ
  Marker: method (Direct, HRSA(f=N), zeta9, etc.)
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# ---- inputs ----
HRSA_CSV   = Path(sys.argv[1] if len(sys.argv) > 1 else
                  "/home/hlamm/Desktop/efficent_gates/unified/hrsa_sweep_v4_2026-05-10.csv")
ZETA9_CSV  = Path(sys.argv[2] if len(sys.argv) > 2 else
                  "/tmp/lenore_f4_eps1e-4.csv")
OUT_PNG    = Path(sys.argv[3] if len(sys.argv) > 3 else
                  "/home/hlamm/Desktop/efficent_gates/unified/nd_vs_eps_2026-05-22.png")

# ---- load HRSA ----
hrsa_pts = []  # (theta, eps, N_D, method)
with open(HRSA_CSV) as fh:
    for r in csv.DictReader(fh):
        if r["all_checks_pass"] != "true":
            continue
        try:
            theta = float(r["theta"])
            eps = float(r["epsilon"])
            nd = int(r["N_D"])
        except (ValueError, TypeError):
            continue
        hrsa_pts.append((theta, eps, nd, r["method"]))

# ---- load zeta9 ----
zeta9_pts = []
with open(ZETA9_CSV) as fh:
    for r in csv.DictReader(fh):
        if r.get("success") not in ("True", "true", "1"):
            continue
        try:
            theta = float(r["theta"])
            eps = float(r["eps_target"])
            nd = int(r["N_D"]) if r["N_D"] else None
        except (ValueError, TypeError):
            continue
        if nd is None:
            continue
        zeta9_pts.append((theta, eps, nd, "zeta9(f=4)"))

print(f"HRSA points: {len(hrsa_pts)}")
print(f"zeta9 points: {len(zeta9_pts)}")

# ---- marker per method ----
method_marker = {
    "Direct":          ("o", "Direct"),
    "bidir(0+0)":      ("v", "bidir(f=0)"),
    "bidir(4+4)":      ("^", "bidir(4+4)"),
    "SignExtClifford": ("P", "SignExtCliff"),
    "HRSA(f=0)":       ("s", "HRSA(f=0)"),
    "HRSA(f=1)":       ("D", "HRSA(f=1)"),
    "HRSA(f=2)":       ("p", "HRSA(f=2)"),
    "HRSA(f=3)":       ("H", "HRSA(f=3)"),
    "HRSA(f=4)":       ("*", "HRSA(f=4)"),
    "zeta9(f=4)":      ("x", "zeta9(max-f=2)"),
}

# ---- color per theta (use viridis for HRSA's 8 thetas; zeta9 thetas overflow) ----
hrsa_thetas = sorted(set(p[0] for p in hrsa_pts))
theta_color = {}
cmap = cm.get_cmap("viridis", len(hrsa_thetas))
for i, t in enumerate(hrsa_thetas):
    theta_color[t] = cmap(i)

def label_theta(t):
    # Label as fraction of π if simple
    frac = t / math.pi
    for denom in (12, 9, 6, 4, 3, 2):
        num = round(frac * denom)
        if abs(frac - num / denom) < 1e-4 and 0 < num < 2 * denom:
            return f"{num}π/{denom}" if num != 1 else f"π/{denom}"
    return f"{t:.3f}"

# ---- plot ----
fig, ax = plt.subplots(figsize=(10, 6.5))

# HRSA: scatter colored by θ, marker by method
for theta, eps, nd, method in hrsa_pts:
    mk, _ = method_marker.get(method, ("o", method))
    color = theta_color.get(theta, "gray")
    ax.scatter(eps, nd, marker=mk, c=[color], s=80, alpha=0.85, edgecolors="black", linewidths=0.5)

# zeta9: distinct marker, semi-transparent, color by closest HRSA theta if avail
# else use a faint gray
for theta, eps, nd, method in zeta9_pts:
    # Find closest HRSA theta for color matching
    closest = min(hrsa_thetas, key=lambda t: abs(t - theta)) if hrsa_thetas else None
    color = theta_color.get(closest, "gray") if closest and abs(closest - theta) < 0.05 else (0.5, 0.5, 0.5)
    ax.scatter(eps, nd, marker="x", c=[color], s=40, alpha=0.5, linewidths=1.0)

# --- legend: methods (markers) ---
from matplotlib.lines import Line2D
method_handles = []
for method, (mk, label) in method_marker.items():
    if not any(p[3] == method for p in hrsa_pts + zeta9_pts):
        continue
    method_handles.append(Line2D([0], [0], marker=mk, color="w",
                                  markerfacecolor="black", markeredgecolor="black",
                                  markersize=9, label=label))
leg1 = ax.legend(handles=method_handles, title="Method", loc="upper right",
                 fontsize=8, framealpha=0.95)
ax.add_artist(leg1)

# --- legend: thetas (colors) ---
theta_handles = []
for t, c in theta_color.items():
    theta_handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=c, markeredgecolor=c,
                                  markersize=8, label=f"θ={label_theta(t)}"))
ax.legend(handles=theta_handles, title="θ (HRSA grid)", loc="lower left",
          fontsize=8, framealpha=0.95, ncol=2)

ax.set_xscale("log")
ax.set_xlabel(r"$\varepsilon$ (target Frobenius distance)")
ax.set_ylabel(r"$N_D$ (D-gate count)")
ax.set_title(
    f"N_D vs ε — HRSA + zeta9 (max-f=2 at ε=10⁻⁴ via batched sweep, 2026-05-22)"
)
ax.invert_xaxis()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"  HRSA: {len(hrsa_pts)} points, {len(hrsa_thetas)} angles")
print(f"  zeta9: {len(zeta9_pts)} points (mostly off-HRSA-grid thetas)")
