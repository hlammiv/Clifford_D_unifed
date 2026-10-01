#!/usr/bin/env python3
"""compute_tables.py — regenerate every number quoted in results_bundle/*.md.

Single source of truth: the per-matrix recount of Nick's 900 exact approximants
(nick_test/nick_tcost_2026-09-30.csv: fixed canonical_reducer with residual R
charged) plus the published C+R fits (Gustafson et al. arXiv:2503.20203) and
qubit baselines.  Writes markdown tables to stdout (and to tables_generated.md).

Conventions (see 01_conventions_and_counts.md):
  A  signed 𝒟 (Kalra / Evra–Parzanchevski; R ∈ C+D, R charged)        <- code default
  B  unsigned ζ₉ diagonals (draft's prose; −1 phases treated as free)  <- draft eq. (37)
Cost units:
  N_φ   per ζ₉ phase with 3∤a (+1 per R under A)  = the historical "N_D"
  ops   per non-Clifford operation (T-type, level-4, R each = 1)
  Tcost T-type 1 + level-4 7 + R 7  (7-T gadgets, one clean ancilla)

Usage:  python3 compute_tables.py [--csv PATH] [--w4 7] [--wr 7]
"""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "nick_test" / "nick_tcost_2026-09-30.csv"
LN3_10 = math.log(10, 3)                       # log3(10) = 2.0959

# Published fits (arXiv:2503.20203, per log10(1/eps))
CR = {"Householder": (3.20, 10.77), "Exhaustive": (2.193, 8.621)}
SU3_FLOOR = (-2.16, 10.27)                     # covering bound, not a fit
QUBIT_RUS = (9.2, 3.817)                       # BRS PRL 114, 080502 (per log10)
QUBIT_DET_PER_LOG2 = 3.0                       # Ross–Selinger leading term


def L3(e): return math.log(1 / e, 3)
def L10(e): return math.log10(1 / e)


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["ok"] == "1"]
    for r in rows:
        for k in ("f", "N_D", "n_T3", "n_L4", "n_R_syl", "n_R_resid", "n_R", "Tcost"):
            r[k] = float(r[k])
        r["epsilon"] = float(r["epsilon"])
    return rows


def fit(xs, ys):
    b, a = np.polyfit(xs, ys, 1)
    yhat = a + b * np.asarray(xs)
    r2 = 1 - np.sum((ys - yhat) ** 2) / np.sum((ys - np.mean(ys)) ** 2)
    return a, b, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--w4", type=float, default=7)
    ap.add_argument("--wr", type=float, default=7)
    a = ap.parse_args()
    rows = load(a.csv)
    out = io.StringIO()
    p = lambda *s: print(*s, file=out)  # noqa: E731

    # derived per-matrix quantities
    for r in rows:
        r["Nphi_A"] = r["N_D"]                          # code convention: R charged
        r["Nphi_B"] = r["N_D"] - r["n_R"]               # draft convention: R free
        r["ops_A"] = r["n_T3"] + r["n_L4"] + r["n_R"]
        r["ops_B"] = r["n_T3"] + r["n_L4"]
        r["T"] = r["n_T3"] + a.w4 * r["n_L4"] + a.wr * r["n_R"]
        r["x"] = L3(r["epsilon"])

    fs = sorted({int(r["f"]) for r in rows})
    p(f"# Generated tables (source: `{Path(a.csv).name}`, n = {len(rows)}; weights: level-4 {a.w4:g} T, R {a.wr:g} T)\n")

    # ---- per-f table
    p("## Per-f means\n")
    p("| f | n | median ε | N_φ (A) | N_φ (B) | T-type | level-4 | R syl | R resid | ops (A) | **T-cost** | T-cost sd |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in fs:
        g = [r for r in rows if int(r["f"]) == f]
        m = lambda k: np.mean([r[k] for r in g])  # noqa: E731
        p(f"| {f} | {len(g)} | {np.median([r['epsilon'] for r in g]):.2e} | {m('Nphi_A'):.1f} | {m('Nphi_B'):.1f} | "
          f"{m('n_T3'):.1f} | {m('n_L4'):.1f} | {m('n_R_syl'):.2f} | {m('n_R_resid'):.2f} | {m('ops_A'):.1f} | "
          f"**{m('T'):.1f}** | {np.std([r['T'] for r in g]):.1f} |")

    # ---- fits
    x = np.array([r["x"] for r in rows])
    p("\n## Fits vs log₃(1/ε) (all rows)\n")
    p("| quantity | intercept | slope per log₃ | slope per log₁₀ | R² | value at ε = 10⁻¹⁰ |")
    p("|---|---|---|---|---|---|")
    fits = {}
    for key, lab in (("Nphi_A", "N_φ, convention A (R charged)"), ("Nphi_B", "N_φ, convention B (R free)"),
                     ("ops_A", "non-Clifford ops, A"), ("ops_B", "non-Clifford ops, B"),
                     ("n_R", "R count"), ("n_L4", "level-4 count"), ("n_T3", "T-type count"),
                     ("T", "**T-cost**")):
        y = np.array([r[key] for r in rows])
        ia, sb, r2 = fit(x, y)
        fits[key] = (ia, sb)
        p(f"| {lab} | {ia:.2f} | {sb:.3f} | {sb * LN3_10:.2f} | {r2:.3f} | {ia + sb * L3(1e-10):.1f} |")

    # ---- C+R reference in the same units
    p("\n## C+R reference (Gustafson et al.), per log₃ and at ε = 10⁻¹⁰\n")
    p("| algorithm | N_R slope per log₃ | N_R at 10⁻¹⁰ | T-cost with R = 7 T at 10⁻¹⁰ | C+R / C+D (T-cost) |")
    p("|---|---|---|---|---|")
    cd10 = fits["T"][0] + fits["T"][1] * L3(1e-10)
    for name, (ic, sl) in CR.items():
        n10 = ic + sl * L10(1e-10)
        p(f"| {name} | {sl / LN3_10:.3f} | {n10:.1f} | {a.wr * n10:.0f} | {a.wr * n10 / cd10:.2f}× |")
    p(f"| SU(3) covering bound (not a fit) | {SU3_FLOOR[1] / LN3_10:.3f} | {SU3_FLOOR[0] + SU3_FLOOR[1] * 10:.1f} | — | — |")

    # ---- break-even vs R-factory C+R device
    p("\n## Break-even against a C+R device with its own R-state factory\n")
    for name, (ic, sl) in CR.items():
        n10 = ic + sl * L10(1e-10)
        p(f"- {name}: C+D cheaper iff c_R/c_T > {cd10 / n10:.2f} at 10⁻¹⁰ "
          f"(asymptotically > {fits['T'][1] / (sl / LN3_10):.2f}).")

    # ---- qutrit vs two-qubit emulation (6 rotations vs 10 qubit R_z)
    p("\n## Qutrit (6 rotations) vs two-qubit emulation (10 R_z), magic-state units\n")
    qrus = lambda e: QUBIT_RUS[0] + QUBIT_RUS[1] * L10(e)            # noqa: E731
    qdet = lambda e: QUBIT_DET_PER_LOG2 * math.log2(1 / e)           # noqa: E731
    cd = lambda e: fits["T"][0] + fits["T"][1] * L3(e)               # noqa: E731
    cd100 = lambda e: fits["T"][0] + 8.3 * L3(e)                     # noqa: E731 (extrapolated best-of-100)
    nphi = lambda e: fits["Nphi_A"][0] + fits["Nphi_A"][1] * L3(e)   # noqa: E731
    crh7 = lambda e: a.wr * (CR["Householder"][0] + CR["Householder"][1] * L10(e))  # noqa: E731
    p("| qutrit side | vs qubit RUS @1e-6 | @1e-10 | vs qubit deterministic @1e-6 | @1e-10 |")
    p("|---|---|---|---|---|")
    for lab, fn in (("C+D T-cost (as is)", cd), ("C+D T-cost, best-of-100 (slope 8.3, extrapolated)", cd100),
                    ("*C+D per-phase N_φ (mixed units; retracted as a cost)*", nphi),
                    ("C+R Householder, R = 7 T", crh7)):
        r = lambda e, q: 6 * fn(e) / (10 * q(e))  # noqa: E731
        p(f"| {lab} | {r(1e-6, qrus):.2f} | {r(1e-10, qrus):.2f} | {r(1e-6, qdet):.2f} | {r(1e-10, qdet):.2f} |")
    p(f"\nAbsolute at 10⁻¹⁰: qutrit C+D 6 × {cd(1e-10):.0f} = {6 * cd(1e-10):.0f} T₃; "
      f"2-qubit RUS {10 * qrus(1e-10):.0f} T₂; 2-qubit deterministic {10 * qdet(1e-10):.0f} T₂.")

    text = out.getvalue()
    print(text)
    (HERE / "tables_generated.md").write_text(text)


if __name__ == "__main__":
    main()
