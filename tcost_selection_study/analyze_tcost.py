"""analyze_tcost.py -- within-angle T-cost spread of HRSA candidate approximants.

Inputs
  raw/hrsa_t{TH}_e{EPS}.log  (HRSA_tester --no-direct --max-solns N): CANDDUMP + CANDTCOST lines
  ../r_count_study/candidates_all.csv  (previous study, eps=0.1 and 1e-2; Python canonical_reducer)

T-cost weights 1/7/7:  Tcost = n_T3 + 7 n_L4 + 7 n_R   (n_R includes residual sign R).
frob = ||P01 (I - u u^dag) - diag(e^{-i th/2}, e^{i th/2}, 1)||_F with u_i = n_i / 3^f.

Outputs: cands_new.csv (all candidates, new + old), summary.csv (per theta/eps),
bestK.csv (resampled best-of-K), printed report.
"""
from __future__ import annotations
import glob, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Z = np.exp(2j * np.pi / 9)
BASIS = np.array([Z ** k for k in range(6)])
KS = [1, 3, 10, 30, 100, 300, 1000]
RNG = np.random.default_rng(20260930)


def target(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2), 1.0])


def frob_of(n1, n2, n3, f, theta):
    u = np.array([np.dot(n, BASIS) for n in (n1, n2, n3)]) / 3 ** f
    H = np.eye(3) - np.outer(u, u.conj())
    V = H[[1, 0, 2], :]
    return float(np.linalg.norm(V - target(theta)))


def parse_log(path):
    m = re.search(r"hrsa_t([0-9.]+)_e([0-9.e-]+)\.log", Path(path).name)
    theta, eps = float(m.group(1)), float(m.group(2))
    dump, tc = {}, {}
    for line in open(path):
        if line.startswith("CANDDUMP"):
            p = line.split()
            s, D, f = int(p[1]), int(p[2]), int(p[9])
            ints = [int(x) for x in p[10:28]]
            dump[s] = (D, f, tuple(ints))
        elif line.startswith("CANDTCOST"):
            p = line.split()
            tc[int(p[1])] = tuple(int(x) for x in p[2:6])
    rows, seen = [], set()
    for s, (D, f, ints) in dump.items():
        if s not in tc or (f, ints) in seen:
            continue
        seen.add((f, ints))
        t3, l4, r, t = tc[s]
        assert t == t3 + 7 * l4 + 7 * r, (path, s)
        rows.append(dict(src="new_cpp", theta=theta, eps=eps, f=f, idx=s, N_D=D,
                         n_T3=t3, n_L4=l4, n_R=r, Tcost=t,
                         frob=frob_of(ints[0:6], ints[6:12], ints[12:18], f, theta)))
    return rows


def load_all():
    rows = []
    for lg in sorted(glob.glob(str(HERE / "raw" / "hrsa_t*_e*.log"))):
        rows += parse_log(lg)
    new = pd.DataFrame(rows)
    old = pd.read_csv(HERE.parent / "r_count_study" / "candidates_all.csv")
    old = old[old.success == True].copy()  # noqa: E712
    old["Tcost"] = old.n_T3 + 7 * old.n_L4 + 7 * old.n_R
    old["src"] = "old_py"
    old["idx"] = old.key.str.split("#").str[1].astype(int)
    old = old[["src", "theta", "eps", "f", "idx", "N_D", "n_T3", "n_L4", "n_R", "Tcost", "frob"]]
    # prefer new data where the same (theta, eps) exists in both
    have = set(zip(new.theta, new.eps)) if len(new) else set()
    old = old[[(t, e) not in have for t, e in zip(old.theta, old.eps)]]
    return pd.concat([new, old], ignore_index=True)


def best_of_k(t, K):
    """Exact E[min of K draws] from the empirical pool: with replacement (order-statistic
    formula) and without replacement (hypergeometric; NaN if K > n)."""
    t = np.sort(np.asarray(t, dtype=float)); n = len(t); k = np.arange(1, n + 1)
    wr = float(((1 - (k - 1) / n) ** K - (1 - k / n) ** K) @ t)
    if K <= n:
        from scipy.special import gammaln
        # P(min index >= k) = C(n-k+1, K)/C(n, K)
        def lc(a, b):
            return np.where(a >= b, gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1), -np.inf)
        surv = np.exp(lc(n - k + 1, K) - lc(n, K))
        surv_next = np.exp(lc(n - k, K) - lc(n, K))
        wo = float((surv - surv_next) @ t)
    else:
        wo = np.nan
    return wr, wo


# expected max of K iid standard normals
ENORM = {1: 0.0, 3: 0.8463, 10: 1.5388, 30: 2.0427, 100: 2.5076, 300: 2.8778, 1000: 3.2414}


def sd_model(eps):
    return 12.5 + 0.68 * np.log(1 / eps) / np.log(3)


def main():
    d = load_all()
    d.to_csv(HERE / "cands_all.csv", index=False)
    summ, bk = [], []
    for (eps, th), g in d.groupby(["eps", "theta"]):
        t = g.Tcost.values
        fmin = g.frob.min()
        r = dict(eps=eps, theta=th, src=g.src.iloc[0], f=int(g.f.max()), n=len(g),
                 L=np.log(1 / eps) / np.log(3),
                 T_mean=t.mean(), T_sd=t.std(ddof=1), T_min=t.min(), T_q10=np.quantile(t, .1),
                 T_q25=np.quantile(t, .25), T_med=np.median(t), T_q75=np.quantile(t, .75),
                 T_max=t.max(), ND_mean=g.N_D.mean(), ND_sd=g.N_D.std(ddof=1),
                 frob_min=fmin, frob_max=g.frob.max(), frob_over_eps_max=g.frob.max() / eps,
                 corr_T_frob=np.corrcoef(g.frob, t)[0, 1] if len(g) > 2 else np.nan,
                 sd_model=sd_model(eps))
        # frob restriction: best-frob decile / quartile of the pool (frob_min ~ 0 at f=3 makes a
        # "within x% of best frob" ratio cut degenerate), plus frob <= eps/2
        for tag, x in (("10", 0.10), ("25", 0.25), ("half_eps", None)):
            gg = g[g.frob <= (np.quantile(g.frob, x) if x is not None else eps / 2)]
            r[f"n_in{tag}"] = len(gg)
            r[f"Tmean_in{tag}"] = gg.Tcost.mean()
            r[f"Tsd_in{tag}"] = gg.Tcost.std(ddof=1) if len(gg) > 1 else np.nan
            r[f"Tmin_in{tag}"] = gg.Tcost.min()
        # frob tertiles
        q = np.quantile(g.frob, [1 / 3, 2 / 3])
        r["Tmean_frob_lo3"] = g.Tcost[g.frob <= q[0]].mean()
        r["Tmean_frob_hi3"] = g.Tcost[g.frob > q[1]].mean()
        summ.append(r)
        for K in KS:
            wr, wo = best_of_k(t, K)
            bk.append(dict(eps=eps, theta=th, n=len(t), K=K, bestK_wr=wr, bestK_wo=wo,
                           gauss_within=t.mean() - ENORM[K] * t.std(ddof=1),
                           gauss_model=t.mean() - ENORM[K] * sd_model(eps)))
    S = pd.DataFrame(summ)
    B = pd.DataFrame(bk)
    S.to_csv(HERE / "summary.csv", index=False)
    B.to_csv(HERE / "bestK.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(S[["eps", "theta", "src", "f", "n", "T_mean", "T_sd", "sd_model", "T_min", "T_q10", "T_med",
             "T_max", "ND_mean", "frob_min", "frob_max", "corr_T_frob", "n_in10", "Tmean_in10", "Tsd_in10",
             "n_in25", "Tmean_in25", "Tsd_in25", "n_inhalf_eps", "Tmean_inhalf_eps", "Tmean_frob_lo3", "Tmean_frob_hi3"]].round(3).to_string())
    print()
    # per-eps aggregate: pooled within-angle sd (mean of per-angle variances) vs model
    agg = S.groupby("eps").apply(lambda s: pd.Series(dict(
        n_angles=len(s), L=s.L.iloc[0], T_mean=s.T_mean.mean(),
        sd_within_pooled=np.sqrt((s.T_sd ** 2).mean()), sd_model=s.sd_model.iloc[0],
        ratio=np.sqrt((s.T_sd ** 2).mean()) / s.sd_model.iloc[0],
        T_mean_between_sd=s.T_mean.std(ddof=1))))
    print(agg.round(3).to_string())
    agg.to_csv(HERE / "sd_by_eps.csv")
    print()
    # best-of-K reduction (mean - bestK), averaged over angles
    B["gain_wr"] = B.merge(S[["eps", "theta", "T_mean"]], on=["eps", "theta"]).eval("T_mean - bestK_wr")
    B["gain_gauss_model"] = B.merge(S[["eps", "theta", "T_mean"]], on=["eps", "theta"]).eval("T_mean - gauss_model")
    B["gain_gauss_within"] = B.merge(S[["eps", "theta", "T_mean"]], on=["eps", "theta"]).eval("T_mean - gauss_within")
    B["gain_wo"] = B.merge(S[["eps", "theta", "T_mean"]], on=["eps", "theta"]).eval("T_mean - bestK_wo")
    B["sd"] = B.merge(S[["eps", "theta", "T_sd"]], on=["eps", "theta"]).T_sd
    B["gain_over_sd_wo"] = B.gain_wo / B.sd
    B.to_csv(HERE / "bestK.csv", index=False)
    Bv = B[B.K <= B.n]   # only K the pool can actually supply (without replacement)
    G = Bv.groupby(["eps", "K"])[["n", "bestK_wo", "gain_wo", "gain_over_sd_wo", "gain_gauss_within", "gain_gauss_model"]].mean()
    G["n_angles"] = Bv.groupby(["eps", "K"]).size()
    print(G.round(2).to_string())
    G.to_csv(HERE / "bestK_by_eps.csv")


if __name__ == "__main__":
    main()
