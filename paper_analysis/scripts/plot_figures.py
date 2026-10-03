#!/usr/bin/env python3
"""plot_figures.py — paper figures from the canonical CSVs (read-only).

  fig/headline.pdf   F3: (a) N_phi vs log10(1/eps) with C+R; (b) T-count in three gadget models
  fig/classes.pdf    F4: per-class counts n_T, n_4, n_R vs log3(1/eps)
  fig/theta.pdf      F5: T-count vs theta at each f (no angle dependence)
  fig/angles.pdf     F9: special-angle eps vs eps a prescribed theta inherits
  fig/qubit.pdf      F7: T gates per arbitrary gate, qutrit vs two-qubit emulation
  fig/prescribed.pdf App.: prescribed- vs special-angle counts

Usage: python3 scripts/plot_figures.py
"""
from __future__ import annotations

import collections
import math
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
# Reference categorical palette (dataviz skill, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#222222", "#8a8a8a"
COL_W, PAGE_W = 3.4, 7.0
L10_3 = math.log(10, 3)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "lines.linewidth": 1.4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def per_f(rows, key):
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["f"])].append(r)
    out = []
    for f in sorted(by):
        rr = by[f]
        x = np.median([math.log10(1 / r["epsilon"]) for r in rr])
        y = [r[key] for r in rr]
        out.append((x, np.mean(y), np.std(y, ddof=1) / math.sqrt(len(y)), np.std(y, ddof=1)))
    return np.array(out)


def headline(rows, pts):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE_W, 2.6))
    xx = np.linspace(0, 11, 50)               # log10(1/eps)
    x3 = xx * L10_3                            # log3(1/eps)

    # (a) per-phase count
    xs = [math.log10(1 / r["epsilon"]) for r in rows]
    a1.scatter(xs, [r["N_D"] for r in rows], s=3, color=C[0], alpha=0.15, lw=0, rasterized=True)
    pf = per_f(rows, "N_D")
    a1.errorbar(pf[:, 0], pf[:, 1], yerr=pf[:, 3], fmt="o", ms=4, color=C[0], mec="white",
                mew=0.6, elinewidth=0.8, capsize=0, label=r"$(\mathbf{C}+\mathbf{D})_3$, special $\theta$")
    f = mn.jackfit([x * L10_3 for x in xs], [r["N_D"] for r in rows])
    dlt, _ = mn.eps_penalty()
    a1.plot(xx, f["a"] + f["b"] * (x3 + dlt), color=C[0], label=r"$(\mathbf{C}+\mathbf{D})_3$, prescribed $\theta$")
    a1.plot(xx, f["a"] + f["b"] * x3, ":", color=C[0], lw=0.9)
    for (name, (a, b)), col, ls in zip(mn.CR.items(), (C[1], C[3]), ("--", "-.")):
        a1.plot(xx, a + b * xx, ls, color=col, label=rf"$(\mathbf{{C}}+\mathbf{{R}})_3$ {name}")
    a1.plot(xx, -2.16 + 4.90 * x3, ":", color=MUTED, lw=1.0, label="SU(3) covering bound")
    a1.set_xlabel(r"$\log_{10}(1/\varepsilon)$")
    a1.set_ylabel(r"non-Clifford count $N_\varphi$ ($N_R$ for C+R)")
    a1.set_xlim(0, 11); a1.set_ylim(0, 130)
    a1.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0))
    a1.text(0.03, 0.94, "(a)", transform=a1.transAxes, ha="right")

    # (b) T-count in three gadget models, best copy, vs C+R Householder in the same model.
    # Solid lines are the conservative prescribed-angle values (special fit shifted by
    # the fixed-angle penalty); dotted lines are the special-angle fits.
    delta, _ = mn.eps_penalty()
    models = [("unitary", r"unitary (7 $\mathbf{T}$)", C[0]), ("merged", "unitary + merging", C[2]),
              ("meas", r"measurement (4 $\mathbf{T}$)", C[4])]
    for m, lab, col in models:
        p = np.array(pts[(m, "best")])
        a2.scatter(p[:, 0] / L10_3, p[:, 1], s=3, color=col, alpha=0.18, lw=0, rasterized=True)
        g = mn.jackfit(p[:, 0], p[:, 1])
        a2.plot(xx, g["a"] + g["b"] * (x3 + delta), color=col, label=f"C+D, {lab}")
        a2.plot(xx, g["a"] + g["b"] * x3, ":", color=col, lw=0.9)
        a, b = mn.CR["Householder"]
        a2.plot(xx, mn.R_COST[m] * (a + b * xx), "--", color=col, lw=1.1)
    a2.plot([], [], ":", color=MUTED, label="C+D at special angles (lower bound)")
    a2.plot([], [], "--", color=MUTED, label="C+R Householder, same model")
    a2.set_xlabel(r"$\log_{10}(1/\varepsilon)$")
    a2.set_ylabel(r"$\mathbf{T}$ gates per $R_z(\theta)$")
    a2.set_xlim(0, 11); a2.set_ylim(0, 800)
    a2.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0))
    a2.text(0.03, 0.94, "(b)", transform=a2.transAxes, ha="right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "headline.pdf", dpi=300)


def classes(rows):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for (key, lab), col, mk in zip([("n_T3", r"$T$-type, $n_T$"), ("n_L4", r"level-4, $n_4$"),
                                    ("n_R", r"$\mathbf{R}$, $n_R$")], C, ("o", "s", "^")):
        pf = per_f(rows, key)
        x3 = pf[:, 0] * L10_3
        ax.errorbar(x3, pf[:, 1], yerr=pf[:, 3], fmt=mk, ms=4, color=col, mec="white", mew=0.6,
                    elinewidth=0.8, label=lab)
        g = mn.jackfit([mn.L3(r["epsilon"]) for r in rows], [r[key] for r in rows])
        xx = np.linspace(0, 23, 20)
        ax.plot(xx, g["a"] + g["b"] * xx, color=col, lw=1.0)
    ax.set_xlabel(r"$\log_3(1/\varepsilon)$")
    ax.set_ylabel("gates per approximant")
    ax.set_xlim(0, 23); ax.set_ylim(0, 55)
    ax.legend(loc="upper left")
    fig.savefig(FIG / "classes.pdf")


def theta():
    """Uses the n=210 symmetry-variant sample (30 theta per f spread over [0, pi]);
    the 900-matrix sample covers only theta < 0.6 at f = 4, 6, 8, 16."""
    import csv
    by = collections.defaultdict(list)
    for r in csv.DictReader(mn._open(mn.STACK_CSV)):
        if r["ok"] == "1" and r["variant"] == "0":
            by[int(r["f"])].append((float(r["theta"]), float(r["T_unit"])))
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    cmap = plt.get_cmap("Blues")
    fs = sorted(by)
    for i, f in enumerate(fs):
        p = np.array(sorted(by[f]))
        col = cmap(0.35 + 0.65 * i / (len(fs) - 1))
        ax.plot(p[:, 0], p[:, 1], "o", ms=3, color=col, mec="white", mew=0.3)
        ax.text(3.22, p[:, 1].mean(), f"$f={f}$", fontsize=6, va="center", color=INK)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\mathbf{T}$ gates (unitary gadgets)")
    ax.set_xlim(0, math.pi + 0.45)
    fig.savefig(FIG / "theta.pdf", dpi=300)


def qubit():
    """F7: T gates per arbitrary single-qutrit gate (6 rotations, prescribed angle,
    eps = 1e-10) against two-qubit emulation (10 qubit R_z)."""
    import re
    nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", (mn.ROOT / "numbers.tex").read_text(), re.M))
    num = lambda k: float(nums[k].replace("{,}", ""))  # noqa: E731
    rows = [("qutrit C+D, measurement", num("Qutritmeas"), C[4]),
            ("qutrit C+D, unitary + merging", num("Qutritmerged"), C[2]),
            ("qutrit C+D, unitary", num("Qutritunitary"), C[0]),
            ("two qubits, deterministic", num("QubitDet"), MUTED),
            ("two qubits, RUS", num("QubitRUS"), "#b5b5b5")]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62, edgecolor="white", linewidth=1.0)
    for yi, (lab, v, _) in zip(y, rows):
        ax.text(v + 20, yi, f"{v:,.0f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel(r"$T$ gates per arbitrary gate at $\varepsilon=10^{-10}$")
    ax.set_xlim(0, 1600)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "qubit.pdf")


def prescribed(rows):
    """App. figure: prescribed-angle vs special-angle counts (analysis/prescribed_theta)."""
    import csv
    pr = [r for r in csv.DictReader(open(mn.ROOT / "analysis" / "prescribed_theta" / "prescribed_counts.csv"))
          if r["ok"] == "1" and r["eps_ok"] == "1" and int(r["sde"]) >= 1 and float(r["achieved_eps"]) <= 0.3]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.9), sharex=True)
    for ax, (kp, ks, lab) in zip(axes, [("N_phi", "N_D", r"$N_\varphi$"),
                                        ("tcost_unitary", "Tcost", r"$\mathbf{T}$ gates (unitary)")]):
        # thin the 10k-point tier for legibility (plotting only; means use all points)
        show = [r for r in pr if rng.random() < (0.05 if r["backend"] == "zeta9" else 1.0)]
        ax.scatter([math.log10(1 / float(r["achieved_eps"])) for r in show], [float(r[kp]) for r in show],
                   s=3, color=C[1], alpha=0.25, lw=0, rasterized=True)
        ax.scatter([math.log10(1 / r["epsilon"]) for r in rows], [r[ks] for r in rows],
                   s=3, color=C[0], alpha=0.2, lw=0, rasterized=True)
        for data, col, mk, ekey, vkey, lbl in ((pr, C[1], "s", "achieved_eps", kp, r"prescribed $\theta$"),
                                                (rows, C[0], "o", "epsilon", ks, r"special $\theta$")):
            by = collections.defaultdict(list)
            for r in data:
                key = int(r["sde"]) if "sde" in r else int(r["f"])
                by[key].append((math.log10(1 / float(r[ekey])), float(r[vkey])))
            pts = np.array([np.mean(v, axis=0) for k, v in sorted(by.items()) if len(v) >= 5])
            ax.plot(pts[:, 0], pts[:, 1], mk, ms=4.5, mfc="white", mec=col, mew=1.0, label=lbl)
        ax.set_ylabel(lab)
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel(r"$\log_{10}(1/\varepsilon)$")
    axes[1].set_xlim(0, 11.3)
    fig.savefig(FIG / "prescribed.pdf", dpi=300)


def main():
    rows = mn.load_nick()
    pts, _, _ = mn.load_stack()
    headline(rows, pts)
    classes(rows)
    theta()
    angles()
    qubit()
    prescribed(rows)
    print("wrote", *(p.name for p in FIG.glob("*.pdf")))



def angles():
    """F9: special-angle density and the error a prescribed theta would inherit
    from the nearest special angle (analysis/special_angles/*.npz)."""
    d = mn.ROOT / "analysis" / "special_angles"
    rng = np.random.default_rng(0)
    fs, e_spec, e_near, n = [], [], [], []
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
        fs.append(f); e_spec.append(np.median(eps)); e_near.append(np.median(best)); n.append(len(th))
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.semilogy(fs, e_spec, "o-", color=C[0], label=r"at the special angles")
    ax.semilogy(fs, e_near, "s--", color=C[1], label=r"random $\theta$ via nearest special angle")
    for f, y, k in zip(fs, e_near, n):
        ax.annotate(f"{k:,}", (f, y), textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=5.5, color=MUTED)
    ax.set_xlabel(r"denominator exponent $f$")
    ax.set_ylabel(r"median $\varepsilon$")
    ax.legend(loc="lower left")
    fig.savefig(FIG / "angles.pdf")


if __name__ == "__main__":
    main()
