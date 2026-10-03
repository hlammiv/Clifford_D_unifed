#!/usr/bin/env python3
"""plot_figures.py — every data figure in the paper, from the canonical CSVs (read-only).

  fig/headline.pdf     Fig. 2: (a) per-phase count, (b) T gates per rotation, C+D vs C+R
  fig/composition.pdf  Fig. 3: where the gates go at eps = 1e-10 (per-phase units and T gates by class)
  fig/qubit.pdf        Fig. 4: magic states per arbitrary single-qutrit gate vs two-qubit emulation
  fig/angles.pdf       App. E: error at the special angles vs error inherited by a random angle

One visual scheme for all figures:
  gate set   C+D blue, C+R orange, qubit baselines and analytic bounds gray
  model      unitary gadgets: solid line, circle, solid fill;
             measurement gadgets: dashed line, square, hatched fill (always listed in that order)
  class      T-type yellow, level-4 green, R magenta (composition figure only)
  data       filled markers are measured means at special angles; lines are fits

Usage: python3 scripts/plot_figures.py
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402

FIG = mn.ROOT / "fig"
FIG.mkdir(exist_ok=True)

# ---- the scheme (reference palette of the dataviz skill, light mode) ----
CD, CR = "#2a78d6", "#eb6834"
REF, REF2 = "#8a8a8a", "#bdbdbd"
CLASS = {"T": "#eda100", "L": "#1baf7a", "R": "#e87ba4"}
MODEL = {"unitary": dict(ls="-", marker="o", label="unitary"),
         "meas": dict(ls="--", marker="s", label="measurement")}
INK = "#222222"
COL_W, PAGE_W = 3.4, 7.0
L10_3 = math.log(10, 3)
CDL = r"$(\mathbf{C}+\mathbf{D})_3$"
CRL = r"$(\mathbf{C}+\mathbf{R})_3$"
TQ = r"$\mathbf{T}$"   # qutrit T gate

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "lines.linewidth": 1.4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42, "hatch.linewidth": 0.6,
    "axes.axisbelow": True,
})


def numbers():
    txt = (mn.ROOT / "numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", txt, re.M))


def per_level_means(pts):
    """pts: (log3(1/eps), y) pairs -> per-denominator-level means (log10 x, y)."""
    by = collections.defaultdict(list)
    for x, y in pts:
        by[round(x / 3)].append((x, y))      # levels sit ~3 units of log3 apart
    return np.array([(np.median([p[0] for p in v]) / L10_3, np.mean([p[1] for p in v]))
                     for _, v in sorted(by.items())])


def headline(rows, pts):
    delta, _ = mn.eps_penalty()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE_W, 2.6))
    xx = np.linspace(0, 11, 50)
    x3 = xx * L10_3

    # (a) per-phase count
    xs3 = [mn.L3(r["epsilon"]) for r in rows]
    ys = [r["N_D"] for r in rows]
    f = mn.jackfit(xs3, ys)
    m = per_level_means(zip(xs3, ys))
    a1.plot(xx, f["a"] + f["b"] * (x3 + delta), color=CD, label=CDL)
    a1.plot(xx, f["a"] + f["b"] * x3, ":", color=CD, lw=1.0)
    a1.plot(m[:, 0], m[:, 1], "o", color=CD, ms=4, mec="white", mew=0.6)
    a1.plot(xx, mn.CR["Householder"][0] + mn.CR["Householder"][1] * xx, "-", color=CR, label=CRL + ", Householder")
    a1.plot(xx, mn.CR["Exhaustive"][0] + mn.CR["Exhaustive"][1] * xx, "-.", color=CR, label=CRL + ", exhaustive")
    a1.plot(xx, -2.16 + 4.90 * x3, ":", color=REF, lw=1.0, label="covering bound")
    a1.set_xlabel(r"$\log_{10}(1/\varepsilon)$")
    a1.set_ylabel(r"per-phase count $N_\varphi$")
    a1.set_xlim(0, 11); a1.set_ylim(0, 140)
    a1.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0))
    a1.text(0.0, 1.02, "(a)", transform=a1.transAxes)

    # (b) T gates per rotation
    for who, color in ((CDL, CD), (CRL, CR)):
        for model in ("unitary", "meas"):
            st = MODEL[model]
            if who == CDL:
                p = np.array(pts[(model, "best")])
                g = mn.jackfit(p[:, 0], p[:, 1])
                y = g["a"] + g["b"] * (x3 + delta)
                mm = per_level_means(map(tuple, p))
                a2.plot(mm[:, 0], mm[:, 1], st["marker"], color=color, ms=3.6, mec="white", mew=0.6)
            else:
                a, b = mn.CR["Householder"]
                y = mn.R_COST[model] * (a + b * xx)
            a2.plot(xx, y, st["ls"], color=color, label=f"{who}, {st['label']}")
    a2.set_xlabel(r"$\log_{10}(1/\varepsilon)$")
    a2.set_ylabel(TQ + r" gates per $R_z(\theta)$")
    a2.set_xlim(0, 11); a2.set_ylim(0, 800)
    a2.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0))
    a2.text(0.0, 1.02, "(b)", transform=a2.transAxes)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "headline.pdf")


def composition():
    """Class content of one rotation at a prescribed angle and eps = 1e-10."""
    c = json.loads((mn.ROOT / "tables" / "composition.json").read_text())
    k = c["classes"]["unitary"]            # the same copy is cheapest in both models (w4 = wR)
    nT, n4, nR = k["n_T3"], k["n_L4"], k["n_R"]
    NR = c["CR_NR"]["Householder"]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.6),
                             gridspec_kw=dict(height_ratios=[2, 4], hspace=0.8))

    def bars(ax, rows, xmax, xlabel):
        for y, (_, segs) in enumerate(rows):
            left = 0.0
            for val, col in segs:
                ax.barh(y, val, left=left, color=col, height=0.62, edgecolor="white", linewidth=1.0)
                left += val
            ax.text(left + xmax * 0.012, y, f"{left:.0f}", va="center", fontsize=6.5, color=INK)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)

    bars(axes[0], [(CDL, [(2 * nT, CLASS["T"]), (n4, CLASS["L"]), (nR, CLASS["R"])]),
                   (CRL, [(NR, CLASS["R"])])], 140, r"per-phase count $N_\varphi$")
    rows = []
    for model in ("unitary", "meas"):
        w = c["R_COST"][model]
        lab = MODEL[model]["label"]
        rows.append((f"{CDL}, {lab}", [(nT, CLASS["T"]), (w * n4, CLASS["L"]), (w * nR, CLASS["R"])]))
        rows.append((f"{CRL}, {lab}", [(w * NR, CLASS["R"])]))
    bars(axes[1], rows, 880, TQ + " gates")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS[x]) for x in "TLR"]
    axes[0].legend(handles, [r"$T$-type", "level-4", r"$\mathbf{R}$"], loc="lower right", ncol=3,
                   bbox_to_anchor=(1.0, 1.0), handlelength=1.0, columnspacing=0.8)
    fig.savefig(FIG / "composition.pdf")


def qubit():
    """Magic states per arbitrary single-qutrit gate (6 rotations) vs two-qubit emulation (10 R_z)."""
    n = numbers()
    num = lambda key: float(n[key].replace("{,}", ""))  # noqa: E731
    rows = [(f"qutrit {CDL}, unitary", num("Qutritunitary"), dict(color=CD, edgecolor=CD)),
            (f"qutrit {CDL}, measurement", num("Qutritmeas"), dict(color="white", edgecolor=CD, hatch="////")),
            ("two qubits, deterministic", num("QubitDet"), dict(color=REF, edgecolor=REF)),
            ("two qubits, RUS", num("QubitRUS"), dict(color=REF2, edgecolor=REF2))]
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    for y, (lab, v, kw) in enumerate(rows):
        ax.barh(y, v, height=0.62, linewidth=0.8, **kw)
        ax.text(v + 20, y, f"{v:,.0f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"magic states per arbitrary gate at $\varepsilon=10^{-10}$")
    ax.set_xlim(0, 1600)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "qubit.pdf")


def angles():
    """Error at the special angles vs the error a random target inherits from its nearest one."""
    d = mn.ROOT / "analysis" / "special_angles"
    rng = np.random.default_rng(0)
    fs, e_spec, e_near, cnt = [], [], [], []
    for f in (4, 6, 8, 10, 12, 14, 16):
        z = np.load(d / f"ang_f{f}.npz")
        th, eps = z["th"], z["eps"]
        keep = np.concatenate([[True], np.diff(th) > 1e-12])
        th, eps = th[keep], eps[keep]
        T = rng.uniform(0, np.pi, 20000)
        idx = np.searchsorted(th, T)
        best = np.full(T.shape, np.inf)
        for k in (idx - 1, idx):
            k = np.clip(k, 0, len(th) - 1)
            best = np.minimum(best, eps[k] + 2 * np.sqrt(2) * np.abs(np.sin(np.abs(T - th[k]) / 4)))
        fs.append(f); e_spec.append(np.median(eps)); e_near.append(np.median(best)); cnt.append(len(th))
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.semilogy(fs, e_spec, "o-", color=CD, ms=4, mec="white", mew=0.6, label="at the special angles")
    ax.semilogy(fs, e_near, "s--", color=REF, ms=4, mec="white", mew=0.6,
                label=r"random $\theta$, nearest special angle")
    for f, y, k in zip(fs, e_near, cnt):
        ax.annotate(f"{k:,}", (f, y), textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=5.5, color=REF)
    ax.set_xlabel(r"denominator exponent $f$")
    ax.set_ylabel(r"median $\varepsilon$")
    ax.legend(loc="lower left")
    fig.savefig(FIG / "angles.pdf")


def main():
    rows = mn.load_nick()
    pts, _, _ = mn.load_stack()
    headline(rows, pts)
    composition()
    qubit()
    angles()
    print("wrote", *sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
