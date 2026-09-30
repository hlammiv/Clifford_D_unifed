"""report.py — summarize candidates_*.csv for the R-count / T-cost study.

Tcost weights (editable): T3 syllable diag = W3, level-4 diag = W4, R syllable or residual R = WR.
Usage: python3 report.py candidates_e0.01.csv [candidates_e0.1.csv ...] [--w3 1 --w4 8 --wr 24]
Writes candidates_all.csv (concatenated, with Tcost column)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--w3", type=float, default=1); ap.add_argument("--w4", type=float, default=8)
ap.add_argument("--wr", type=float, default=24); ap.add_argument("--tol", type=float, default=0.10)
a = ap.parse_args()
df = pd.concat([pd.read_csv(c) for c in a.csvs], ignore_index=True)
print(f"rows {len(df)}, success {df.success.sum()}, greedy used {df.greedy.sum()}")
df = df[df.success == True].copy()
df["n_T3"], df["n_L4"], df["n_R"] = df.lvl3, df.lvl4, df.R_syl + df.R_resid_sign
df["Tcost"] = a.w3 * df.n_T3 + a.w4 * df.n_L4 + a.wr * df.n_R
out = Path(a.csvs[0]).parent / "candidates_all.csv"
df.to_csv(out, index=False)
print("wrote", out)
print("R_resid_classify nonzero:", int((df.R_resid_classify != 0).sum()),
      "| R_resid_sign nonzero:", int((df.R_resid_sign != 0).sum()))
print("HRSA C++ D (with simplifier) == reducer N_D:", float((df.hrsa_D == df.N_D).mean()))

for (eps, th), g in df.groupby(["eps", "theta"]):
    fb = g.loc[g.frob.idxmin()]
    near = g[g.frob <= fb.frob * (1 + a.tol)]
    bT = near.loc[near.Tcost.idxmin()]; bR = near.loc[near.n_R.idxmin()]
    hist = g.n_R.value_counts().sort_index().to_dict()
    print(f"\neps={eps} theta={th}: n={len(g)} f={g.f.iloc[0]} frob[min,med,max]="
          f"{g.frob.min():.4g},{g.frob.median():.4g},{g.frob.max():.4g}")
    print(f"  n_R hist {hist}; R_syl hist {g.R_syl.value_counts().sort_index().to_dict()}")
    print(f"  N_D med {g.N_D.median()} [{g.N_D.min()}-{g.N_D.max()}]; Tcost med {g.Tcost.median()} "
          f"[{g.Tcost.min()}-{g.Tcost.max()}]; T3 med {g.n_T3.median()} L4 med {g.n_L4.median()}")
    print(f"  frob-best: frob={fb.frob:.4g} N_D={fb.N_D} n_R={fb.n_R} T3={fb.n_T3} L4={fb.n_L4} Tcost={fb.Tcost}")
    print(f"  within {a.tol:.0%} of best frob: {len(near)} cands; min Tcost={bT.Tcost} (n_R={bT.n_R}, "
          f"N_D={bT.N_D}, frob={bT.frob:.4g}); min n_R={bR.n_R}; "
          f"frac with fewer R than frob-best={np.mean(near.n_R < fb.n_R):.2f}")
    print(f"  global: min Tcost={g.Tcost.min()} (frob {g.loc[g.Tcost.idxmin()].frob:.4g}); "
          f"min-N_D cand (HRSA_bestD pick) Tcost={g.loc[g.N_D.idxmin()].Tcost} n_R={g.loc[g.N_D.idxmin()].n_R}")

print("\n=== correlations (pooled, within-cell ranks) ===")
for c in ["N_D", "frob", "n_T3", "n_L4", "Tcost", "sde0", "n_syl"]:
    rs = [g.n_R.corr(g[c], method="spearman") for _, g in df.groupby(["eps", "theta"])]
    print(f"  spearman(n_R, {c}) per cell: {np.round(rs, 2)}")

print("\n=== predictors ===")
for col in ["col0_rel", "col0_dig", "row0_dig", "det_sign", "trail_det"]:
    df[col] = df[col].astype(str)
    t = df.groupby(col).agg(n=("n_R", "size"), mean_nR=("n_R", "mean"), mean_Rsyl=("R_syl", "mean"),
                            frac_R0=("n_R", lambda s: float((s == 0).mean())),
                            first_eps=("first_eps", "mean"), resid=("R_resid_sign", "mean"))
    print(f"-- by {col}\n{t.round(3)}")
# parity check: (-1)^R_syl == det_sign(V) * trail_det
par = ((-1) ** df.R_syl) == (df.det_sign.astype(int) * df.trail_det.astype(int))
print("\nparity identity (-1)^R_syl == sign det V * sign det T holds:", float(par.mean()))
print("R_syl parity vs det_sign(V):")
print(pd.crosstab(df.det_sign, df.R_syl % 2))
print("n_R parity vs det_sign(V):")
print(pd.crosstab(df.det_sign, df.n_R % 2))

print("\n=== chi-adic sign-class filter (col0_dig == '111', i.e. first column leading digits all equal mod chi) ===")
df["cls111"] = df.col0_dig.astype(str) == "111"
print(pd.crosstab([df.cls111], [df.first_eps, df.R_resid_sign], margins=True))
df["R_mid"] = df.R_syl - df.first_eps
print("mid-word R (R_syl - first_eps) by class:\n", df.groupby("cls111").R_mid.describe().round(2))
for (eps, th), g in df.groupby(["eps", "theta"]):
    fb = g.loc[g.frob.idxmin()]
    near = g[g.frob <= fb.frob * (1 + a.tol)]
    n1 = near[near.cls111]
    s = (f"eps={eps} th={th}: frac111={g.cls111.mean():.2f}; meanT 111={g[g.cls111].Tcost.mean():.0f} "
         f"vs other={g[~g.cls111].Tcost.mean():.0f}; meanN_D 111={g[g.cls111].N_D.mean():.1f} vs {g[~g.cls111].N_D.mean():.1f}; "
         f"near-best-frob: n={len(near)} n111={len(n1)} minT(all)={near.Tcost.min()} minT(111)="
         f"{n1.Tcost.min() if len(n1) else 'NA'} frobbest T={fb.Tcost} (111={fb.cls111})")
    print(s)
