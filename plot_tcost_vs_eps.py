#!/usr/bin/env python3
"""plot_tcost_vs_eps.py — headline figure: C+D vs C+R, per-phase count and T-cost.

Left panel  — per-phase non-Clifford count vs precision.  C+D N_D (per ζ₉ phase,
              corrected for residual R) against Gustafson's C+R N_R fits: the tie.
Right panel — fault-tolerant T-cost per rotation on a qutrit T-factory device
              (T-type 1 T, level-4 7 T, R 7 T, one clean ancilla) against C+R
              with each R built from 7 T.  An R-state factory is worse still
              (c_R/c_T ≥ 16 even optimistically; unified/factory_model/).

Data: nick_test/nick_tcost_2026-09-30.csv (900 exact approximants, f = 4…16).
Usage: python3 plot_tcost_vs_eps.py [out.png]
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "nick_test" / "nick_tcost_2026-09-30.csv"

# Validated default categorical order (dataviz reference palette, light mode).
C_CD, C_CRH, C_CRE, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# Gustafson et al. arXiv:2503.20203 empirical C+R fits (per log10), and the SU(3) floor.
CRH = (3.20, 10.77)     # Householder: N_R = 3.20 + 10.77 log10(1/ε)
CRE = (2.193, 8.621)    # Exhaustive
FLOOR = (-2.16, 10.27)  # information-theoretic lower bound (not a fit)
R_T = 7                 # T per R (7-T construction, one clean ancilla)


def load():
    rows = [r for r in csv.DictReader(open(CSV)) if r["ok"] == "1"]
    x = np.array([math.log10(1 / float(r["epsilon"])) for r in rows])
    return rows, x


def per_f(rows, x, key):
    out = []
    for f in sorted({int(r["f"]) for r in rows}):
        idx = [i for i, r in enumerate(rows) if int(r["f"]) == f]
        y = np.array([float(rows[i][key]) for i in idx])
        out.append((x[idx].mean(), y.mean(), y.std(), f))
    return out


def style(ax, ylabel):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(r"precision  $\log_{10}(1/\varepsilon)$", color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_xlim(0, 11)


def line(ax, xs, a, b, color, ls, label, end_label, lw=2.0, dy=0):
    ys = a + b * xs
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, label=label, zorder=2)
    ax.annotate(end_label, (xs[-1], ys[-1]), xytext=(4, dy), textcoords="offset points",
                va="center", fontsize=8.5, color=INK)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tcost_vs_eps_2026-09-30.png"
    rows, x = load()
    xs = np.linspace(0.5, 10.5, 50)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURF)

    # ---- left: per-phase counts (the tie)
    style(a1, "non-Clifford gates per rotation")
    y = np.array([float(r["N_D"]) for r in rows])
    a1.scatter(x, y, s=9, color=C_CD, alpha=0.18, lw=0, zorder=1)
    b, a = np.polyfit(x, y, 1)
    line(a1, xs, a, b, C_CD, "-", f"C+D, per ζ₉ phase ($N_D$): {b:.2f}·log₁₀ + {a:.1f}", "C+D  $N_D$", dy=9)
    line(a1, xs, *CRH, C_CRH, "--", "C+R Householder ($N_R$): 10.77·log₁₀ + 3.2", "C+R Householder", dy=-9)
    line(a1, xs, *CRE, C_CRE, "--", "C+R Exhaustive ($N_R$): 8.62·log₁₀ + 2.2", "C+R Exhaustive")
    line(a1, xs, *FLOOR, C_REF, ":", "SU(3) covering bound (not a fit)", "", lw=1.5)
    for mx, my, sd, f in per_f(rows, x, "N_D"):
        a1.errorbar(mx, my, yerr=sd, fmt="o", ms=5, color=C_CD, mec=SURF, mew=1.5,
                    elinewidth=1.2, capsize=0, zorder=3)
    a1.set_ylim(0, 150)
    a1.set_title("Per-phase count: C+D ties C+R", loc="left", fontsize=11, color=INK)
    a1.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)

    # ---- right: T-cost
    style(a2, "T gates per rotation (qutrit T-factory device)")
    y = np.array([float(r["Tcost"]) for r in rows])
    a2.scatter(x, y, s=9, color=C_CD, alpha=0.18, lw=0, zorder=1)
    b, a = np.polyfit(x, y, 1)
    line(a2, xs, a, b, C_CD, "-", f"C+D: {b:.1f}·log₁₀ + {a:.0f}  (T + 7·level-4 + 7·R)", "C+D")
    line(a2, xs, R_T * CRH[0], R_T * CRH[1], C_CRH, "--",
         "C+R Householder, R = 7 T", "C+R Householder")
    line(a2, xs, R_T * CRE[0], R_T * CRE[1], C_CRE, "--",
         "C+R Exhaustive, R = 7 T", "C+R Exhaustive")
    for mx, my, sd, f in per_f(rows, x, "Tcost"):
        a2.errorbar(mx, my, yerr=sd, fmt="o", ms=5, color=C_CD, mec=SURF, mew=1.5,
                    elinewidth=1.2, capsize=0, zorder=3)
    e10 = 10.0
    cd10, crh10 = a + b * e10, R_T * (CRH[0] + CRH[1] * e10)
    a2.annotate(f"{crh10 / cd10:.1f}× at ε = 10⁻¹⁰", xy=(e10, cd10), xytext=(e10 - 1.2, cd10 + 110),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a2.text(0.3, 0.97 * 820, "R from an R-state factory: ≥16× a T each\n(optimistic; 10⁴–10⁸× as published)",
            fontsize=8, color=INK2, va="top")
    a2.set_ylim(0, 820)
    a2.set_title("Fault-tolerant cost: C+D ≈ 3.4× cheaper", loc="left", fontsize=11, color=INK)
    a2.legend(loc="center left", fontsize=8, frameon=False, labelcolor=INK)

    for ax in (a1, a2):
        ax.set_xlim(0, 12.6)
    fig.text(0.01, -0.02, "C+D: 900 exact approximants (points; dots with bars = per-f mean ± sd), "
             "canonical synthesis with residual-R fix. C+R: Gustafson et al., arXiv:2503.20203 fits.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
