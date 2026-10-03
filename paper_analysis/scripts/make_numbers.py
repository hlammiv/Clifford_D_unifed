#!/usr/bin/env python3
"""make_numbers.py — single source for every number quoted in the paper.

Reads the canonical data in the repository (never edits it) and writes
  numbers.tex          LaTeX macros (\\NphiSlope, \\TcostAt, ...)
  tables/perf.tex      per-f summary table (Table II)
  tables/fits.tex      fit table (Table III)
  tables/headline.tex  T per R_z at eps = 1e-10 in three gadget models (Table IV)

Fits are y = a + b log3(1/eps); uncertainties are delete-one-angle jackknife
(each (f, theta) matrix is one jackknife unit).

Usage: python3 scripts/make_numbers.py
"""
from __future__ import annotations

import collections
import csv
import math
from pathlib import Path

import numpy as np


def _repo_root(start):
    """Directory holding nick_test/: the repository root, or ../unified next to the paper."""
    for p in [start, *start.parents]:
        if (p / "nick_test").is_dir():
            return p
        if (p / "unified" / "nick_test").is_dir():
            return p / "unified"
    raise FileNotFoundError("cannot locate the repository root (directory containing nick_test/)")

ROOT = Path(__file__).resolve().parents[1]
UNI = _repo_root(ROOT)
NICK_CSV = UNI / "nick_test" / "nick_tcost_2026-09-30.csv"
STACK_CSV = UNI / "symmetry_variants" / "stack_full30_rows.csv"
if not STACK_CSV.exists():
    STACK_CSV = STACK_CSV.with_suffix(".csv.gz")
EPS_CSV = UNI / "nick_request" / "topk_run" / "fast_all30_candidates.csv"
PRESCRIBED_JSON = ROOT / "analysis" / "prescribed_theta" / "compare_results.json"

# Gustafson et al., arXiv:2503.20203: N_R = a + b log10(1/eps)
CR = {"Householder": (3.20, 10.77), "Exhaustive": (2.193, 8.621)}
R_COST = {"unitary": 7.0, "merged": 5.0, "meas": 4.0}   # T per R in each gadget model
EPS_REF = 1e-10
L3REF = math.log(1 / EPS_REF, 3)
LOG3_10 = math.log(10, 3)


def _open(path):
    import gzip
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def L3(e):
    return math.log(1 / e, 3)


def jackfit(x, y):
    """Least-squares line with delete-one jackknife errors on (a, b, y(EPS_REF))."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    n = len(x)
    reps = np.empty((n, 3))
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        bi, ai = np.polyfit(x[m], y[m], 1)
        reps[i] = ai, bi, ai + bi * L3REF
    err = np.sqrt((n - 1) / n * np.sum((reps - reps.mean(0)) ** 2, axis=0))
    yhat = a + b * x
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(a=a, b=b, at=a + b * L3REF, da=err[0], db=err[1], dat=err[2], r2=r2, n=n)


def pm(v, dv, nd):
    """Value(error) in parenthetical notation: at most nd decimals, and the
    error is shown with at most two significant digits."""
    if dv > 0:
        nd = min(nd, max(0, 1 - int(math.floor(math.log10(dv)))))
    s = f"{v:.{nd}f}"
    e = round(dv * 10 ** nd)
    return f"{s}({max(e, 1)})"


def macro(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def load_nick():
    rows = [r for r in csv.DictReader(open(NICK_CSV)) if r["ok"] == "1"]
    for r in rows:
        for k in ("f", "N_D", "n_T3", "n_L4", "n_R_syl", "n_R_resid", "n_R", "Tcost"):
            r[k] = float(r[k])
        r["epsilon"] = float(r["epsilon"])
        r["ops"] = r["n_T3"] + r["n_L4"] + r["n_R"]
        r["Tmeas"] = r["n_T3"] + 4 * r["n_L4"] + 4 * r["n_R"]
    return rows


def load_stack():
    key = lambda f, th: (str(f), f"{float(th):.6g}")  # noqa: E731
    eps = {key(r["f"], r["theta"]): float(r["epsilon"])
           for r in csv.DictReader(open(EPS_CSV)) if r["rank"] == "0"}
    by = collections.defaultdict(list)
    for r in csv.DictReader(_open(STACK_CSV)):
        if r["ok"] == "1":
            by[key(r["f"], r["theta"])].append(r)
    cols = {"unitary": "T_unit", "merged": "T_merged", "meas": "T_meas"}
    pts = collections.defaultdict(list)
    for k, v in by.items():
        x = L3(eps[k])
        v0 = [r for r in v if r["variant"] == "0"][0]
        for m, c in cols.items():
            pts[(m, "asis")].append((x, float(v0[c])))
            pts[(m, "best")].append((x, min(float(r[c]) for r in v)))
    return pts, len(by), len(next(iter(by.values())))


def eps_penalty():
    """Fixed-angle error penalty Delta = log3(eps_prescribed / eps_special) at equal sde.

    Counts agree at equal sde (analysis/prescribed_theta); only eps(sde) differs.
    We take the LARGEST median penalty over the sde levels both sets share (4, 6, 8)
    and hold it constant to eps = 1e-10.  This is conservative: a linear fit of
    log3(1/eps) vs sde is steeper for prescribed angles (1.61 vs 1.45 per level),
    so extrapolating would shrink the penalty."""
    import json
    rows = json.load(open(PRESCRIBED_JSON))["per_sde"]
    med = collections.defaultdict(dict)
    for r in rows:
        if r["set"] in ("special", "prescribed (all)"):
            med[r["sde"]][r["set"]] = r["med_eps"]
    pen = {k: math.log(v["prescribed (all)"] / v["special"], 3) for k, v in med.items() if len(v) == 2}
    return max(pen.values()), pen


def composition(delta):
    """Mean class counts (n_T, n_4, n_R) of the cheapest phase copy at a prescribed angle and
    eps = 1e-10, for each gadget model.  Fits each class on the copy that minimises that model's
    T count (so the classes sum exactly to the headline value), shifted by the penalty delta."""
    key = lambda f, th: (str(f), f"{float(th):.6g}")  # noqa: E731
    eps = {key(r["f"], r["theta"]): float(r["epsilon"])
           for r in csv.DictReader(open(EPS_CSV)) if r["rank"] == "0"}
    by = collections.defaultdict(list)
    for r in csv.DictReader(_open(STACK_CSV)):
        if r["ok"] == "1":
            by[key(r["f"], r["theta"])].append(r)
    out = {}
    for m, col in (("unitary", "T_unit"), ("meas", "T_meas")):
        xs, cls = [], {"n_T3": [], "n_L4": [], "n_R": []}
        for k, v in by.items():
            best = min(v, key=lambda r: float(r[col]))
            xs.append(L3(eps[k]))
            for c in cls:
                cls[c].append(float(best[c]))
        out[m] = {c: float(np.polyval(np.polyfit(xs, ys, 1), L3REF + delta)) for c, ys in cls.items()}
    return out


def main():
    out = ["% Auto-generated by scripts/make_numbers.py -- do not edit by hand.\n"]
    rows = load_nick()
    x = [L3(r["epsilon"]) for r in rows]
    out.append(macro("Nmat", len(rows)))
    fmin, fmax = int(min(r["f"] for r in rows)), int(max(r["f"] for r in rows))
    out.append(macro("fmin", fmin) + macro("fmax", fmax))
    em = np.median([r["epsilon"] for r in rows if int(r["f"]) == fmax])
    out.append(macro("epsmin", f"{em:.0e}".replace("e-", "\\times10^{-") + "}"))   # median error at f = fmax

    fits = {}
    spec = [("Nphi", "N_D", r"$N_\varphi$"),
            ("Ops", "ops", "ops"),
            ("NT", "n_T3", r"$n_T$"),
            ("NL", "n_L4", r"$n_4$"),
            ("NR", "n_R", r"$n_R$"),
            ("Tcost", "Tcost", r"$N_T$ (unitary)"),
            ("Tmeas", "Tmeas", r"$N_T$ (meas.)")]
    for tag, col, _ in spec:
        f = jackfit(x, [r[col] for r in rows])
        fits[tag] = f
        out.append(macro(f"{tag}Int", pm(f["a"], f["da"], 2)))
        out.append(macro(f"{tag}Slope", pm(f["b"], f["db"], 3)))
        out.append(macro(f"{tag}At", pm(f["at"], f["dat"], 1)))
        out.append(macro(f"{tag}RR", f"{f['r2']:.2f}"))

    # C+R references, per log3
    for name, (a10, b10) in CR.items():
        b3 = b10 / LOG3_10
        nr = a10 + b10 * 10
        out.append(macro(f"CR{name}Slope", f"{b3:.3f}"))
        out.append(macro(f"CR{name}At", f"{nr:.1f}"))
        for m, w in R_COST.items():
            out.append(macro(f"CR{name}T{m}", f"{w * nr:.0f}"))

    # symmetry copies + gadget models (n = 210 subsample)
    pts, nmat, nvar = load_stack()
    out.append(macro("Nstack", nmat) + macro("Nvariants", nvar))
    head = {}
    for (m, s), p in pts.items():
        xs, ys = zip(*p)
        f = jackfit(xs, ys)
        head[(m, s)] = f
        t = f"{m}{s}"
        out.append(macro(f"T{t}Int", pm(f["a"], f["da"], 1)))
        out.append(macro(f"T{t}Slope", pm(f["b"], f["db"], 2)))
        out.append(macro(f"T{t}At", f"{f['at']:.0f}"))
        for name, (a10, b10) in CR.items():
            ratio = R_COST[m] * (a10 + b10 * 10) / f["at"]
            out.append(macro(f"Ratio{t}{name}", f"{ratio:.1f}"))

    # conservative (prescribed-angle) values: shift L3 by the fixed-angle penalty
    delta, pen = eps_penalty()
    out.append(macro("EpsPenalty", f"{delta:.2f}") + macro("EpsPenaltyFactor", f"{3 ** delta:.0f}"))
    out.append(macro("EpsPenaltySdes", ", ".join(str(k) for k in sorted(pen))))
    out.append(macro("NphiConsAt", f"{fits['Nphi']['a'] + fits['Nphi']['b'] * (L3REF + delta):.0f}"))
    cons = {}
    for (m, sel), f in head.items():
        if sel != "best":
            continue
        v = f["a"] + f["b"] * (L3REF + delta)
        cons[m] = v
        out.append(macro(f"T{m}consAt", f"{v:.0f}"))
        for name, (a10, b10) in CR.items():
            out.append(macro(f"Ratio{m}cons{name}", f"{R_COST[m] * (a10 + b10 * 10) / v:.1f}"))

    comp = composition(delta)
    import json as _json
    (ROOT / "tables" / "composition.json").parent.mkdir(exist_ok=True)
    (ROOT / "tables" / "composition.json").write_text(_json.dumps(
        {"delta": delta, "classes": comp,
         "CR_NR": {n: a + b * 10 for n, (a, b) in CR.items()}, "R_COST": R_COST}, indent=1))
    for m, d in comp.items():
        out.append(macro(f"Comp{m}T", f"{d['n_T3']:.0f}") + macro(f"Comp{m}L", f"{d['n_L4']:.0f}")
                   + macro(f"Comp{m}R", f"{d['n_R']:.0f}"))
    nphi_cons = 2 * comp["unitary"]["n_T3"] + comp["unitary"]["n_L4"] + comp["unitary"]["n_R"]
    out.append(macro("TtypeShare", f"{2 * comp['unitary']['n_T3'] / nphi_cons:.2f}"))

    # break-even c_R/c_T against a C+R device with its own R factory, and qubit comparison
    QUBIT_DET = 10 * 3.0 * math.log2(1 / EPS_REF)          # 10 R_z, Ross-Selinger 3 log2(1/eps)
    QUBIT_RUS = 10 * (9.2 + 3.817 * 10)                     # 10 R_z, BRS PRL 114, 080502
    out.append(macro("QubitDet", f"{QUBIT_DET:.0f}") + macro("QubitRUS", f"{QUBIT_RUS:.0f}"))
    for m in ("unitary", "merged", "meas"):
        for name, (a10, b10) in CR.items():
            out.append(macro(f"Breakeven{m}{name}", f"{cons[m] / (a10 + b10 * 10):.2f}"))
        q = 6 * cons[m]
        out.append(macro(f"Qutrit{m}", f"{q:,.0f}".replace(",", "{,}")))
        out.append(macro(f"QvsDet{m}", f"{q / QUBIT_DET:.2f}") + macro(f"QvsRUS{m}", f"{q / QUBIT_RUS:.2f}"))

    # prescribed vs special at sde 8 (the deepest shared level)
    import json
    pj = json.load(open(PRESCRIBED_JSON))
    at8 = {r["set"]: r for r in pj["per_sde"] if r["sde"] == 8}
    for tag, key in (("Sp", "special"), ("Pr", "prescribed (all)")):
        r = at8[key]
        out.append(macro(f"Ns{tag}", f"{r['n']:,}".replace(",", "{,}")))
        out.append(macro(f"Nphi{tag}Eight", pm(r["N_phi"][0], r["N_phi"][1], 1)))
        out.append(macro(f"Tunit{tag}Eight", pm(r["tcost_unitary"][0], r["tcost_unitary"][1], 1)))
        out.append(macro(f"Tmeas{tag}Eight", pm(r["tcost_meas"][0], r["tcost_meas"][1], 1)))
        mant, ex = f"{r['med_eps']:.1e}".split("e")
        out.append(macro(f"Eps{tag}Eight", f"{mant}\\times10^{{{int(ex)}}}"))
    out.append(macro("Nprescribed", "11{,}403"))
    rowsp, last = [], None
    for r in pj["per_sde_target"]:
        if r["sde"] != last and last is not None:
            rowsp.append("\\MidRule")
        tgt = "all" if r["target_eps"] == 0 else f"$10^{{{int(round(math.log10(r['target_eps'])))}}}$" \
            if abs(math.log10(r["target_eps"]) - round(math.log10(r["target_eps"]))) < 1e-9 else f"{r['target_eps']:g}"
        rowsp.append(f"{r['sde'] if r['sde'] != last else ''} & {tgt} & {r['n']:,} & "
                     f"${pm(r['dN'], r['dN_se'], 1)}$ & ${pm(r['dT'], r['dT_se'], 1)}$ \\\\".replace(",", "{,}") if r['n'] >= 1000 else
                     f"{r['sde'] if r['sde'] != last else ''} & {tgt} & {r['n']} & "
                     f"${pm(r['dN'], r['dN_se'], 1)}$ & ${pm(r['dT'], r['dT_se'], 1)}$ \\\\")
        last = r["sde"]
    (ROOT / "tables" / "prescribed_rows.tex").write_text("\n".join(rowsp) + "\n")

    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "numbers.tex").write_text("".join(out))

    # ---------- Table II: per-f summary ----------
    byf = collections.defaultdict(list)
    for r in rows:
        byf[int(r["f"])].append(r)
    lines = []
    for f in sorted(byf):
        rr = byf[f]
        mean = lambda k: np.mean([r[k] for r in rr])  # noqa: E731
        sd = np.std([r["Tcost"] for r in rr], ddof=1)
        eps = np.median([r["epsilon"] for r in rr])
        mant, ex = f"{eps:.2e}".split("e")
        lines.append(f"{f} & {len(rr)} & ${mant}\\times10^{{{int(ex)}}}$ & {mean('N_D'):.1f} & "
                     f"{mean('n_T3'):.1f} & {mean('n_L4'):.1f} & {mean('n_R'):.2f} & "
                     f"{mean('ops'):.1f} & {mean('Tcost'):.1f}({sd:.1f}) & {mean('Tmeas'):.1f} \\\\")
    (ROOT / "tables" / "perf_rows.tex").write_text("\n".join(lines) + "\n")

    # ---------- Table III: fits ----------
    lines = []
    for tag, _, label in spec:
        f = fits[tag]
        lines.append(f"{label} & {pm(f['a'], f['da'], 2)} & {pm(f['b'], f['db'], 3)} & "
                     f"{f['r2']:.2f} & {pm(f['at'], f['dat'], 1)} \\\\")
    (ROOT / "tables" / "fits_rows.tex").write_text("\n".join(lines) + "\n")

    # ---------- Table IV: headline ----------
    names = {"unitary": "Unitary (7 $T$)", "merged": "Unitary + merging", "meas": "Measurement (4 $T$)"}
    lines = []
    for m in ("unitary", "merged", "meas"):
        a, b = head[(m, "asis")], head[(m, "best")]
        crh = R_COST[m] * (CR["Householder"][0] + CR["Householder"][1] * 10)
        cre = R_COST[m] * (CR["Exhaustive"][0] + CR["Exhaustive"][1] * 10)
        c = cons[m]
        nrh, nre = (CR["Householder"][0] + CR["Householder"][1] * 10), (CR["Exhaustive"][0] + CR["Exhaustive"][1] * 10)
        lines.append(f"{names[m]} & {b['at']:.0f} & {c:.0f} & {crh:.0f} & {cre:.0f} & "
                     f"{crh / c:.1f} / {cre / c:.1f} & {c / nrh:.2f} / {c / nre:.2f} \\\\")
    (ROOT / "tables" / "headline_rows.tex").write_text("\n".join(lines) + "\n")

    print((ROOT / "numbers.tex").read_text())


if __name__ == "__main__":
    main()
