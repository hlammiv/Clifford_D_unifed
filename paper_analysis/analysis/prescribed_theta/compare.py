#!/usr/bin/env python3
"""compare.py — prescribed-theta (prescribed_counts.csv, from redecompose.py) vs
special-theta (Nick's 900, unified/nick_test/nick_tcost_2026-09-30.csv) gate counts.

Writes compare_results.json, compare_tables.md and prescribed_vs_special.pdf.

Fits y = a + b*log3(1/eps) by OLS; errors by delete-block jackknife (G = 50
random blocks; for n <= 200 delete-one).  Cost columns:
  N_phi = N_D (convention A, R charged; = headline 3.94 + 5.161 log3)
  T-cost unitary  n_T + 7 n_4 + 7 n_R ;  T-cost meas  n_T + 4 n_4 + 4 n_R
Prescribed set used in fits: eps_ok = 1 (theta metadata consistent), sde >= 1
(sde 0 = exact zeta_9 diagonal monomials, eps = 0 or a lucky hit; not an
approximation) and eps <= 0.3.
"Balanced" variant: the 9,900 zeta9_tier2 cells (all at eps ~ 3e-5, sde 8)
thinned to every 100th so that one sweep does not carry the fit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import os


def _repo_root(start):
    """Directory holding nick_test/: the repository root, or ../unified next to the paper."""
    for p in [start, *start.parents]:
        if (p / "nick_test").is_dir():
            return p
        if (p / "unified" / "nick_test").is_dir():
            return p / "unified"
    raise FileNotFoundError("cannot locate the repository root (directory containing nick_test/)")



H = Path(__file__).resolve().parent
NICK = _repo_root(Path(__file__).resolve()) / "nick_test" / "nick_tcost_2026-09-30.csv"
L3 = lambda e: np.log(1 / np.asarray(e, float)) / np.log(3)  # noqa: E731
QTY = [("N_phi", "N_phi"), ("tcost_unitary", "T-cost (w4=wR=7)"), ("tcost_meas", "T-cost (w4=wR=4)"),
       ("ops", "ops"), ("n_T", "n_T"), ("n_4", "n_4"), ("n_R", "n_R")]
HEADLINE = {"N_phi": (3.94, 5.161), "tcost_unitary": (16.61, 10.017), "tcost_meas": (9.74, 6.584)}


def load():
    p = pd.read_csv(H / "prescribed_counts.csv")
    p = p[(p.ok == 1) & (p.unitary_exact == 1)]
    p["family"] = np.where(p.backend == "zeta9", "zeta9", "hrsa")
    n = pd.read_csv(NICK)
    n = n[n.ok == 1].merge(pd.read_csv(H / "nick_sde.csv"), on=["f", "theta"])
    n = n.rename(columns={"epsilon": "achieved_eps", "N_D": "N_phi", "n_T3": "n_T", "n_L4": "n_4",
                          "Tcost": "tcost_unitary"})
    n["tcost_meas"] = n.n_T + 4 * n.n_4 + 4 * n.n_R
    n["ops"] = n.n_T + n.n_4 + n.n_R
    return p, n


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def fit_jk(x, y, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = ols(x, y)
    n = len(x)
    if n <= 200:
        blocks = [np.array([i]) for i in range(n)]
    else:
        idx = np.random.default_rng(seed).permutation(n)
        blocks = np.array_split(idx, 50)
    G = len(blocks)
    est = []
    for blk in blocks:
        m = np.ones(n, bool)
        m[blk] = False
        est.append(ols(x[m], y[m]))
    est = np.array(est)
    err = np.sqrt((G - 1) / G * ((est - est.mean(0)) ** 2).sum(0))
    yhat = a + b * x
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(n=n, a=a, da=err[0], b=b, db=err[1], r2=r2,
                x_range=[float(x.min()), float(x.max())])


def fmt(f):
    return f"{f['a']:.2f} ± {f['da']:.2f} | {f['b']:.3f} ± {f['db']:.3f} | {f['r2']:.3f} | {f['n']}"


def main():
    p_all, n = load()
    res = {}
    out = []
    P = lambda *s: out.append(" ".join(str(x) for x in s))  # noqa: E731

    p = p_all[(p_all.eps_ok == 1) & (p_all.sde >= 1) & (p_all.achieved_eps <= 0.3)
              & (p_all.achieved_eps > 0)].copy()
    tier2 = p.source.str.startswith("sweep_zeta9_tier2")
    cell = p.path.str.extract(r"cell_(\d+)")[0].astype(float)
    pb = p[~tier2 | (cell % 100 == 0)]
    n_ov = n[n.achieved_eps >= 1e-4]
    lo, hi = max(p.achieved_eps.min(), n.achieved_eps.min()), min(p.achieved_eps.max(), n.achieved_eps.max())
    n_int = n[(n.achieved_eps >= lo) & (n.achieved_eps <= hi)]
    p_int = p[(p.achieved_eps >= lo) & (p.achieved_eps <= hi)]
    pb_int = pb[(pb.achieved_eps >= lo) & (pb.achieved_eps <= hi)]
    sets = {
        "prescribed (all, eps<=0.3, sde>=1)": p,
        "prescribed balanced (tier2 thinned 1/100)": pb,
        f"prescribed in common range [{lo:.1e},{hi:.1e}]": p_int,
        f"prescribed balanced in common range": pb_int,
        "special: all 900 (headline)": n,
        "special: eps >= 1e-4 (f=4 + part of f=6)": n_ov,
        f"special in common range [{lo:.1e},{hi:.1e}]": n_int,
    }
    P("# Prescribed-θ vs special-θ gate counts\n")
    P(f"Prescribed rows: {len(p_all)} distinct exact unitaries; used in fits: {len(p)} "
      f"(balanced: {len(pb)}).  Special rows: {len(n)}.")
    P(f"Common achieved-ε range of the two sets: [{lo:.2e}, {hi:.2e}] "
      f"(special max ε = {n.achieved_eps.max():.2e}; Nick has nothing at ε > 7.5e-3).\n")
    res["fits"] = {}
    for q, lab in QTY:
        P(f"\n## Fit {lab} = a + b·log3(1/ε)  (jackknife errors)\n")
        P("| set | a | b (per log3) | R² | n |")
        P("|---|---|---|---|---|")
        res["fits"][q] = {}
        for name, d in sets.items():
            if len(d) < 3 or d.achieved_eps.nunique() < 2:
                continue
            f = fit_jk(L3(d.achieved_eps), d[q])
            res["fits"][q][name] = f
            P(f"| {name} | {fmt(f)} |")

    # residuals of prescribed data w.r.t. the headline special fits
    P("\n## Prescribed data vs the headline special-θ fits (mean residual = data − fit)\n")
    P("| sde | family | n | median ε | N_φ − (3.94+5.161 L3) | T-cost7 − (16.61+10.017 L3) | T-cost4 − (9.74+6.584 L3) |")
    P("|---|---|---|---|---|---|---|")
    res["headline_residuals"] = []
    for (s, fam), d in p.groupby(["sde", "family"]):
        x = L3(d.achieved_eps)
        r = {}
        for q in HEADLINE:
            a, b = HEADLINE[q]
            rr = d[q] - (a + b * x)
            r[q] = (float(rr.mean()), float(rr.std(ddof=1) / math.sqrt(len(rr))) if len(rr) > 1 else float("nan"))
        res["headline_residuals"].append(dict(sde=int(s), family=fam, n=len(d),
                                              med_eps=float(d.achieved_eps.median()), **r))
        P(f"| {s} | {fam} | {len(d)} | {d.achieved_eps.median():.2e} | "
          + " | ".join(f"{r[q][0]:+.1f} ± {r[q][1]:.1f}" for q in HEADLINE) + " |")

    # per-sde comparison (the key test): include all sde, all eps (eps_ok only)
    pe = p_all[(p_all.eps_ok == 1)]
    P("\n## Counts per sde level (mean ± s.e.m.; sd in brackets)\n")
    P("| sde | set | n | median ε | ε range (5–95%) | N_φ | T-cost7 | T-cost4 | n_T | n_4 | n_R | ops |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    res["per_sde"] = []

    def row(s, name, d):
        e = d.achieved_eps
        cells = []
        rec = dict(sde=int(s), set=name, n=len(d), med_eps=float(e.median()),
                   eps_q05=float(e.quantile(.05)), eps_q95=float(e.quantile(.95)))
        for q in ["N_phi", "tcost_unitary", "tcost_meas", "n_T", "n_4", "n_R", "ops"]:
            m, sd = d[q].mean(), d[q].std(ddof=1) if len(d) > 1 else float("nan")
            se = sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            rec[q] = (float(m), float(se), float(sd))
            cells.append(f"{m:.1f} ± {se:.1f} [{sd:.1f}]" if q in ("N_phi", "tcost_unitary", "tcost_meas")
                         else f"{m:.2f}")
        res["per_sde"].append(rec)
        P(f"| {s} | {name} | {len(d)} | {e.median():.2e} | {e.quantile(.05):.1e}–{e.quantile(.95):.1e} | "
          + " | ".join(cells) + " |")

    for s in sorted(set(pe.sde) | set(n.sde)):
        dn = n[n.sde == s]
        if len(dn):
            row(s, "special", dn)
        dp = pe[pe.sde == s]
        if len(dp):
            row(s, "prescribed (all)", dp)
            for fam, df in dp.groupby("family"):
                if dp.family.nunique() > 1:
                    row(s, f"prescribed {fam}", df)

    # per-sde split by the prescribed target eps (loose targets -> atypically cheap words?)
    P("\n## Prescribed per (sde, target ε) vs special at the same sde (N_φ, T-cost7 means)\n")
    P("| sde | target ε | n | median ε | N_φ | ΔN_φ vs special | T-cost7 | ΔT-cost7 vs special |")
    P("|---|---|---|---|---|---|---|---|")
    res["per_sde_target"] = []
    for (s_, t), d in pe[pe.sde.isin([4, 6, 8])].groupby(["sde", "target_eps"]):
        dn = n[n.sde == s_]
        dN = d.N_phi.mean() - dn.N_phi.mean()
        dT = d.tcost_unitary.mean() - dn.tcost_unitary.mean()
        seN = math.sqrt(d.N_phi.var(ddof=1) / len(d) + dn.N_phi.var(ddof=1) / len(dn)) if len(d) > 1 else float("nan")
        seT = math.sqrt(d.tcost_unitary.var(ddof=1) / len(d) + dn.tcost_unitary.var(ddof=1) / len(dn)) if len(d) > 1 else float("nan")
        res["per_sde_target"].append(dict(sde=int(s_), target_eps=float(t), n=len(d), N_phi=float(d.N_phi.mean()),
                                          dN=float(dN), dN_se=seN, tcost=float(d.tcost_unitary.mean()),
                                          dT=float(dT), dT_se=seT))
        P(f"| {s_} | {t:g} | {len(d)} | {d.achieved_eps.median():.1e} | {d.N_phi.mean():.1f} | {dN:+.1f} ± {seN:.1f} | "
          f"{d.tcost_unitary.mean():.1f} | {dT:+.1f} ± {seT:.1f} |")

    # per chi-adic sde (finer than sde_3; N_D tracks sde_chi)
    P("\n## Counts per χ-adic sde of V (sde_χ = max_ij sde_χ(V_ij); 6·sde_3 is its maximum)\n")
    P("| sde_χ | special n | special N_φ | special T-cost7 | prescribed n | prescribed N_φ | prescribed T-cost7 | ΔN_φ | ΔT-cost7 |")
    P("|---|---|---|---|---|---|---|---|---|")
    res["per_sde_chi"] = []
    for c in sorted(set(pe.sde_chi) | set(n.sde_chi)):
        dn, dp = n[n.sde_chi == c], pe[pe.sde_chi == c]
        f1 = lambda d, q: (f"{d[q].mean():.1f} ± {d[q].std(ddof=1) / math.sqrt(len(d)):.1f}" if len(d) > 1  # noqa: E731
                           else (f"{d[q].mean():.1f}" if len(d) else "—"))
        dd = lambda q: (f"{dp[q].mean() - dn[q].mean():+.1f}" if len(dn) and len(dp) else "—")  # noqa: E731
        res["per_sde_chi"].append(dict(sde_chi=int(c), n_sp=len(dn), n_pr=len(dp),
                                       N_sp=float(dn.N_phi.mean()) if len(dn) else None,
                                       N_pr=float(dp.N_phi.mean()) if len(dp) else None,
                                       T_sp=float(dn.tcost_unitary.mean()) if len(dn) else None,
                                       T_pr=float(dp.tcost_unitary.mean()) if len(dp) else None))
        P(f"| {c} | {len(dn)} | {f1(dn, 'N_phi')} | {f1(dn, 'tcost_unitary')} | {len(dp)} | {f1(dp, 'N_phi')} | "
          f"{f1(dp, 'tcost_unitary')} | {dd('N_phi')} | {dd('tcost_unitary')} |")

    # per-sde differences prescribed - special
    P("\n## Prescribed − special at equal sde\n")
    P("| sde | ΔN_φ | ΔN_φ / N_φ | ΔT-cost7 | ΔT-cost4 | ε_presc/ε_special (median) | Δlog3(1/ε) |")
    P("|---|---|---|---|---|---|---|")
    res["per_sde_diff"] = []
    for s in sorted(set(pe.sde) & set(n.sde)):
        dn, dp = n[n.sde == s], pe[pe.sde == s]
        rec = dict(sde=int(s))
        cells = []
        for q in ["N_phi", "tcost_unitary", "tcost_meas"]:
            dlt = dp[q].mean() - dn[q].mean()
            se = math.sqrt(dp[q].var(ddof=1) / len(dp) + dn[q].var(ddof=1) / len(dn))
            rec[q] = (float(dlt), float(se))
            cells.append(f"{dlt:+.1f} ± {se:.1f}")
            if q == "N_phi":
                cells.append(f"{dlt / dn[q].mean():+.1%}")
        ratio = dp.achieved_eps.median() / dn.achieved_eps.median()
        dl = float(np.median(L3(dn.achieved_eps)) - np.median(L3(dp.achieved_eps)))
        rec.update(eps_ratio=float(ratio), dlog3=dl)
        res["per_sde_diff"].append(rec)
        P(f"| {s} | " + " | ".join(cells) + f" | {ratio:.1f}× | {dl:.2f} |")

    # achieved eps vs sde: per-sde fit of log3(1/eps) = c + d*sde
    P("\n## Achieved ε vs sde (\"fixed-θ ε penalty\")\n")
    P("| set | log3(1/ε) per unit sde | intercept | n |")
    P("|---|---|---|---|")
    res["eps_vs_sde"] = {}
    for name, d in [("special", n), ("prescribed (sde>=1)", pe[pe.sde >= 1]),
                    ("prescribed best-per-sde (min ε)", pe[pe.sde >= 1].groupby("sde").achieved_eps.min().reset_index())]:
        f = fit_jk(d.sde, L3(d.achieved_eps))
        res["eps_vs_sde"][name] = f
        P(f"| {name} | {f['b']:.3f} ± {f['db']:.3f} | {f['a']:.2f} ± {f['da']:.2f} | {f['n']} |")
    P("\nMin / median achieved ε per sde:\n")
    P("| sde | special min | special median | prescribed min | prescribed median | prescribed target ε values |")
    P("|---|---|---|---|---|---|")
    for s in sorted(set(pe.sde) | set(n.sde)):
        dn, dp = n[n.sde == s], pe[pe.sde == s]
        g = lambda d, fn: f"{fn(d.achieved_eps):.2e}" if len(d) else "—"  # noqa: E731
        tg = ",".join(f"{t:g}" for t in sorted(dp.target_eps.unique())) if len(dp) else "—"
        P(f"| {s} | {g(dn, np.min)} | {g(dn, np.median)} | {g(dp, np.min)} | {g(dp, np.median)} | {tg} |")

    # within-sde dependence on eps (does count track eps or sde?)
    P("\n## Within-sde slope dN_φ/dlog3(1/ε) (does N_φ track ε at fixed sde?)\n")
    P("| sde | set | n | slope ± jk | ")
    P("|---|---|---|---|")
    res["within_sde"] = []
    for s in [4, 6, 8]:
        for name, d in [("special", n[n.sde == s]), ("prescribed", pe[pe.sde == s])]:
            if len(d) > 5 and d.achieved_eps.nunique() > 3:
                f = fit_jk(L3(d.achieved_eps), d.N_phi)
                res["within_sde"].append(dict(sde=s, set=name, **f))
                P(f"| {s} | {name} | {f['n']} | {f['b']:.2f} ± {f['db']:.2f} |")

    (H / "compare_tables.md").write_text("\n".join(out) + "\n")
    json.dump(res, open(H / "compare_results.json", "w"), indent=1, default=float)
    print("\n".join(out))
    figure(p, pb, n, res)


def figure(p, pb, n, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_SP, C_PR = "#2a78d6", "#eb6834"
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)
    rng = np.random.default_rng(1)
    L10 = np.log10
    xs = np.linspace(0, 11.5, 50)
    for ax, q, ylab in [(axs[0], "N_phi", r"$N_\varphi$"), (axs[1], "tcost_unitary", "T-cost (unitary)")]:
        jit = lambda d: rng.uniform(-0.25, 0.25, len(d))  # noqa: E731
        ax.scatter(L10(1 / n.achieved_eps), n[q] + jit(n), s=2.5, lw=0, color=C_SP, alpha=0.35,
                   rasterized=False, label=r"special $\theta$ (Gnedin, 900)")
        ax.scatter(L10(1 / pb.achieved_eps), pb[q] + jit(pb), s=2.5, lw=0, color=C_PR, alpha=0.5,
                   label=rf"prescribed $\theta$ ({len(pb)} shown of {len(p)})")
        # per-sde means
        for d, c, mk in [(n, C_SP, "o"), (p, C_PR, "s")]:
            g = d.groupby("sde").agg(x=("achieved_eps", lambda e: np.median(L10(1 / e))), y=(q, "mean"))
            ax.plot(g.x, g.y, mk, ms=4, mfc="white", mec=c, mew=1.0, ls="none")
        fs = res["fits"][q]["special: all 900 (headline)"]
        fp = res["fits"][q]["prescribed balanced (tier2 thinned 1/100)"]
        l3 = xs * math.log(10) / math.log(3)
        ax.plot(xs, fs["a"] + fs["b"] * l3, color=C_SP, lw=1.2,
                label=rf"special fit {fs['a']:.1f}+{fs['b']:.2f}$\,L_3$")
        xp = np.linspace(*[v * math.log(3) / math.log(10) for v in fp["x_range"]], 20)
        ax.plot(xp, fp["a"] + fp["b"] * xp * math.log(10) / math.log(3), color=C_PR, lw=1.2, ls="--",
                label=rf"prescribed fit {fp['a']:.1f}+{fp['b']:.2f}$\,L_3$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_xlim(0, 11.5)
        ax.set_ylim(bottom=0)
    axs[0].legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    h, l = axs[1].get_legend_handles_labels()
    axs[1].legend(h[2:], l[2:], loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    axs[1].set_xlabel(r"$\log_{10}(1/\epsilon)$")
    axs[1].text(0.98, 0.04, "open markers: per-sde means\n" r"$L_3=\log_3(1/\epsilon)$",
                transform=axs[1].transAxes, ha="right", va="bottom", fontsize=6, color="0.35")
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(H / "prescribed_vs_special.pdf")
    fig.savefig(H / "prescribed_vs_special_preview.png", dpi=200)


if __name__ == "__main__":
    main()
