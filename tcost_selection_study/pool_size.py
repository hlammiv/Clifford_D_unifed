"""pool_size.py -- how many candidates exist at a fixed denominator level f (HRSA, k3=1).

Uses the complete HRSA pools (max-solns 50000, never reached):
  f=2: raw/hrsa_t*_e0.1.log   f=3: raw/hrsa_t*_e0.01.log
Counts N(frob < x) and fits N ~ C x^p on the range where the pool is complete
(HRSA's acceptance proxy |u-u0|^2 < eps^2/8 is stricter than frob; the pool is complete in frob
only for x <~ 0.27 eps_run (99th pct of |u-u0|^2/frob^2 is 1.73)).  Extrapolates to N=1 -> lowest eps reachable at level f.
"""
import glob, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_tcost as A  # noqa: E402

out = []
for lg in sorted(glob.glob(str(HERE / "raw" / "hrsa_t*_e0.1.log"))) + sorted(glob.glob(str(HERE / "raw" / "hrsa_t*_e0.01.log"))):
    d = pd.DataFrame(A.parse_log(lg))
    if len(d) < 1000:
        continue
    th, eps, f = d.theta.iloc[0], d.eps.iloc[0], int(d.f.max())
    xs = np.geomspace(0.12 * eps, 0.26 * eps, 8)
    Ns = np.array([(d.frob < x).sum() for x in xs])
    p, lc = np.polyfit(np.log(xs), np.log(Ns), 1)
    x1 = np.exp(-lc / p)
    out.append(dict(theta=th, f=f, eps_run=eps, n_pool=len(d), p=p, eps_N1=x1,
                    N_at_frob_0p26eps=int(Ns[-1]), T_mean=d.Tcost.mean(), T_sd=d.Tcost.std()))
    print(f"theta={th} f={f}: pool={len(d)}  N(frob<x) ~ x^{p:.2f}; N=1 at frob={x1:.2e}; "
          f"T mean {d.Tcost.mean():.1f} sd {d.Tcost.std():.1f}")
O = pd.DataFrame(out)
O.to_csv(HERE / "pool_size.csv", index=False)
g = O.groupby("f").agg(p=("p", "mean"), eps_N1=("eps_N1", lambda v: np.exp(np.mean(np.log(v)))),
                       T_mean=("T_mean", "mean"), T_sd=("T_sd", "mean"))
print(g)
if {2, 3} <= set(g.index):
    r = g.eps_N1[2] / g.eps_N1[3]
    dL = np.log(r) / np.log(3)
    p = g.p.mean()
    print(f"level spacing: eps floor ratio f2/f3 = {r:.1f} -> dL = {dL:.2f} log3 units; "
          f"pool at top of a level Nmax ~ r^p = {r ** p:.2e}")
    print(f"mean step {g.T_mean[3] - g.T_mean[2]:.1f} T per level = {(g.T_mean[3] - g.T_mean[2]) / dL:.1f} T per log3; "
          f"sd step {g.T_sd[3] - g.T_sd[2]:.2f} -> {(g.T_sd[3] - g.T_sd[2]) / dL:.2f} per log3")
