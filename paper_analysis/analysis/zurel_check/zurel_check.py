"""Re-run the Golay-type R-state row of the c_R/c_T factory model with the new qutrit
quantum-quadratic-residue (QQR) Strange-state codes of Zurel, Jana, de Silva,
arXiv:2603.18560 (QST 11, 045052, 2026).

Nothing in unified/ is modified: factory_cost.py is imported read-only (bytecode writing is
disabled so no __pycache__ is touched there).

Distillation map (paper Eqs. (118)-(121), Sec. 4.3.1, pp. 24-25), noise parameter
eps = depolarising parameter of rho_S(eps) = (1-eps)|S><S| + eps*1/3   (paper Eq. (107)):
    x = (4 eps - 3)/9 , y = (3 - eps)/18                                 (Eq. (108))
    eps' = 3/4 * (1 + 3 W_I(x,y) / W_{I^perp}(x,y))                      (Eq. (119))
    W_{I^perp}(x,y) = W_I(1, (eps-1)/2) / 3^(n-1)   (MacWilliams, Eq. (5) with q=9)
    P_s = W_{I^perp}(x,y)                                               (Eq. (121))
The model's error variable is infidelity = 2 eps / 3, which is what is passed in/out here.
P_s is the single-syndrome (all-zero outcome) post-selection probability used by the paper.

Run:  python3 zurel_check.py   (writes zurel_check_output.txt next to this file)
"""
import functools
import math
import os
import sys
from pathlib import Path

def _repo_root(start):
    """Directory holding nick_test/: the repository root, or ../unified next to the paper."""
    for p in [start, *start.parents]:
        if (p / "nick_test").is_dir():
            return p
        if (p / "unified" / "nick_test").is_dir():
            return p / "unified"
    raise FileNotFoundError("cannot locate the repository root (directory containing nick_test/)")


sys.dont_write_bytecode = True
sys.path.insert(0, str(_repo_root(Path(__file__).resolve()) / "factory_model"))
import factory_cost as fc            # noqa: E402  (read-only reuse)

import mpmath as mp                  # noqa: E402

from zurel_codes import QQR          # noqa: E402






mp.mp.dps = 120


# ---------------------------------------------------------------------------------
# exact maps from the paper's weight enumerators
# ---------------------------------------------------------------------------------
def qqr_raw(name, eps):
    """(eps_out, P_s) in the paper's depolarising-parameter convention."""
    n, _, A = QQR[name]
    e = mp.mpf(eps)
    x = (4 * e - 3) / 9
    y = (3 - e) / 18
    num = mp.fsum(a * x ** (n - w) * y ** w for w, a in A.items())
    den = mp.fsum(a * ((e - 1) / 2) ** w for w, a in A.items()) / mp.mpf(3) ** (n - 1)
    return mp.mpf(3) / 4 * (1 + 3 * num / den), den


@functools.lru_cache(maxsize=None)
def _qqr_inf(name, inf):
    eo, P = qqr_raw(name, 1.5 * inf)
    return float(eo) * 2 / 3, float(P)


def make_qqr_map(name):
    return lambda inf: _qqr_inf(name, float(inf))


def threshold(name):
    lo, hi = 1e-6, 0.999
    f = lambda e: float(qqr_raw(name, e)[0]) - e      # noqa: E731
    # find a sign change scanning downwards from near 1 (distilling region is f<0)
    grid = [i / 2000 for i in range(1, 2000)]
    last = None
    for g in grid:
        s = f(g) < 0
        if last is not None and last and not s:
            lo, hi = prev, g
            break
        last, prev = s, g
    else:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return lo


def leading_order(name):
    """eps' ~ c eps^m (paper convention): fit m and c at tiny eps."""
    e1, e2 = mp.mpf("1e-12"), mp.mpf("1e-13")
    o1, o2 = qqr_raw(name, e1)[0], qqr_raw(name, e2)[0]
    m = round(float(mp.log(o1 / o2) / mp.log(e1 / e2)))
    c = float(qqr_raw(name, mp.mpf("1e-20"))[0] / mp.mpf("1e-20") ** m)
    return m, c


# ---------------------------------------------------------------------------------
# code tables in factory_cost.best_sequence format: name -> (n, k, map(inf)->(inf_out,P), ngen)
# ---------------------------------------------------------------------------------
DISTILLING = ["QQR11", "QQR17", "QQR23", "QQR41", "QQR47"]     # paper Table 3 (p. 26)
NEW = ["QQR17", "QQR23", "QQR41", "QQR47"]


def codes_realistic(include_golay_model=True):
    d = {}
    if include_golay_model:
        d["Golay11"] = fc.CODES_S["Golay11"]          # exactly as in the model
    for nm in NEW:
        n = QQR[nm][0]
        d[nm] = (n, 1, make_qqr_map(nm), n - 1)
    return d


def codes_optimistic_exactmap():
    """Exact eps_out(eps) but acceptance forced to 1."""
    d = {}
    for nm in DISTILLING:
        n = QQR[nm][0]
        f = make_qqr_map(nm)
        d[nm] = (n, 1, (lambda f: lambda e: (f(e)[0], 1.0))(f), n - 1)
    return d


LEAD = {}


def codes_optimistic_leading():
    """Leading order eps_out = c eps^m (infidelity units), acceptance 1."""
    d = {}
    for nm in DISTILLING:
        n = QQR[nm][0]
        m, c = LEAD[nm]
        # convert: inf = 2 eps/3 -> inf' = (2/3) c (1.5 inf)^m
        f = (lambda m, c: lambda e: ((2 / 3) * c * (1.5 * e) ** m, 1.0))(m, c)
        d[nm] = (n, 1, f, n - 1)
    return d


def cost_R_strange(p, eps_gate, codes, max_levels=5):
    eps_S_target = eps_gate / (fc.INJ_R * fc.EPS_R_OVER_EPS_S)
    c, seq, lv = fc.best_sequence(codes, p, eps_S_target, max_levels=max_levels)
    return fc.INJ_R * fc.S_PER_RSTATE * c, seq, lv


def main(out):
    def pr(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        out.write(s + "\n")

    pr("Zurel-Jana-de Silva QQR Strange-state codes in the c_R/c_T model\n")
    pr("Per-code data (paper convention eps = depolarising parameter; infidelity = 2eps/3)")
    pr(f"{'code':7s} {'[[n,1,d]]':10s} {'thresh':>8s} {'m':>3s} {'c (eps^m)':>11s} "
       f"{'P_s(0)':>11s} {'P_s(.015)':>11s} {'eps_out(.015)':>13s} {'P_s(.0015)':>11s} "
       f"{'eps_out(.0015)':>14s} {'P_s(1.5e-4)':>11s}")
    for nm in QQR:
        n, dist, _ = QQR[nm]
        th = threshold(nm)
        m, c = leading_order(nm)
        if nm in DISTILLING:
            LEAD[nm] = (m, c)
        r0 = qqr_raw(nm, 0)[1]
        e2, P2 = qqr_raw(nm, 0.015)
        e3, P3 = qqr_raw(nm, 0.0015)
        _, P4 = qqr_raw(nm, 0.00015)
        pr(f"{nm:7s} [[{n},1,{dist}]]".ljust(19) + f"{th:8.5f} {m:3d} {c:11.4g} {float(r0):11.4g} "
           f"{float(P2):11.4g} {float(e2):13.4g} {float(P3):11.4g} {float(e3):14.4g} {float(P4):11.4g}")
    pr("  (eps-args .015/.0015/1.5e-4 are the depolarising parameters for raw infidelity "
       "p = 1e-2/1e-3/1e-4; threshold 0 = no non-trivial threshold)")
    pr("  1/P_s(0) for QQR11 = %.1f (=1728: Prakash's Golay value reproduced)\n"
       % (1 / float(qqr_raw('QQR11', 0)[1])))

    budget, ND, NR = 1e-2, 227.0, 111.0           # same as factory_cost.main
    realistic = codes_realistic()
    golay_only = fc.CODES_S
    optx = codes_optimistic_exactmap()
    optl = codes_optimistic_leading()
    rows = []
    pr("c_R/c_T grid (c_T = QRM8 as in the model's Golay row; R route = strange-state "
       "distillation -> SS->N->R conversion (x32) -> RUS injection (x3))")
    for p in [1e-2, 1e-3, 1e-4]:
        for Nrot in [1e2, 1e3, 1e4]:
            eT, eR = budget / (ND * Nrot), budget / (NR * Nrot)
            cT, sT, _ = fc.cost_T(p, eT)
            cG, sG, _ = cost_R_strange(p, eR, golay_only)
            cZ, sZ, lZ = cost_R_strange(p, eR, realistic)
            cX, sX, _ = cost_R_strange(p, eR, optx)
            cL, sL, _ = cost_R_strange(p, eR, optl)
            floor = fc.INJ_R * fc.S_PER_RSTATE * 11       # one P=1 round of the smallest code
            rows.append((p, Nrot, cT, cG / cT, cZ / cT, cX / cT, cL / cT, floor / cT))
            pr(f"\np={p:g} N_rot={Nrot:g}  eps_R/gate={eR:.1e}  c_T={cT:.4g} {sT}")
            pr(f"  Golay only (model)          : c_R={cG:10.4g}  c_R/c_T={cG/cT:10.4g}  {sG}")
            pr(f"  Golay + QQR17/23/41/47      : c_R={cZ:10.4g}  c_R/c_T={cZ/cT:10.4g}  {sZ}")
            for (nm, ei, eo, P) in (lZ or []):
                pr(f"      level {nm}: inf {ei:.3e} -> {eo:.3e}, P_s={P:.4g}, raw/out={QQR.get(nm, (11,))[0]/P:.4g}")
            pr(f"  optimistic: exact map, P=1  : c_R={cX:10.4g}  c_R/c_T={cX/cT:10.4g}  {sX}")
            pr(f"  optimistic: c*eps^m, P=1    : c_R={cL:10.4g}  c_R/c_T={cL/cT:10.4g}  {sL}")
            pr(f"  hard floor (one 11->1 round at P=1, x96 conversion+injection): "
               f"c_R/c_T >= {floor/cT:.3g}")
    pr("\nSummary (min..max over the 9-point grid):")
    labels = ["Golay only (model)", "Golay+QQR, real acceptance", "optimistic exact map P=1",
              "optimistic leading order P=1", "floor: 1 round n=11, P=1"]
    for i, lab in enumerate(labels):
        vals = [r[3 + i] for r in rows]
        pr(f"  {lab:30s}: {min(vals):10.4g} .. {max(vals):10.4g}")
    pr("  comparison: break-even c_R/c_T ~ 1.4-2.6 ; 7-T gadget c_R/c_T = 7 (QRM8 T route)")
    return rows


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "zurel_check_output.txt"), "w") as fh:
        main(fh)
