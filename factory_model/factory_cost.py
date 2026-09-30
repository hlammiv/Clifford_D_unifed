"""Crude cost model: raw (physical-level) magic states consumed per fault-tolerant
qutrit T gate vs per fault-tolerant R = diag(1,1,-1) gate, on a generic qutrit
stabilizer-code machine with perfect Cliffords (standard MSD accounting).

Tags in comments:  (L) literature   (I) our inference/computation   (?) unverified guess

Routes
  T  : QRM_3(2) 8->1  [Campbell-Anwar-Browne 1205.3104]  exact depolarising map (I, re-derived
       from the CAB weight-enumerator formula; reproduces threshold 0.211001 exactly)
       optional PS [[20,7,2]]_3 [Prakash-Saha 2403.06228] with an ASSUMED eps_out = A eps^2 (?)
  R(ii): Golay 11->1 strange-state distillation [Prakash 2003.02717] (L: exact delta_out(delta),
       P_succ = 1/1728 - 11/2592 delta), then SS->N (p=1/2), NN->R-state (p=1/4) (L, and
       simulated exactly in conversions.py: eps_R-state = 2.667 eps_S), then RUS injection
       with 3 R-states per R gate on average (L: BRS 1605.02756; I: Z2xZ2 random walk w/ adaptive frame)
  R(i) : [[5,1,3]]_3 H-state distillation (L: 1202.2326, eps2' = 0.77 eps2, linear) -> parity
       check -> equatorialisation (p=1/4, I computed). Only a LOWER bound is modelled.
  R(d) : R from 7 T with one clean ancilla (our result) -> deterministic, 7 T per R.
  R(iii): R-state from P9/T by RUS (BRS: 27/4 T per R-state; earlier agent: 9) x 3 per R gate.

Run:  python3 factory_cost.py
"""
import itertools
import math
import numpy as np

from conversions import qrm_map, R_state_infidelity

LOG = []

# ---------------------------------------------------------------------------------
# distillation "codes":  n inputs -> k outputs, map(eps) -> (eps_out, P_success)
# error variable = infidelity eps = 1 - <psi|rho|psi>, depolarising/twirled noise.
# ---------------------------------------------------------------------------------


def golay_map(eps):
    """Golay 11->1 for strange state; eps = infidelity = 2 delta/3 (L: Prakash eq 5.7-5.12)."""
    d = 1.5 * eps
    P = (3021*d**8 - 24816*d**7 + 92180*d**6 - 203280*d**5 + 292710*d**4
         - 283536*d**3 + 181764*d**2 - 71280*d + 13365)
    Q = (495*d**11 - 3960*d**10 + 13750*d**9 - 25245*d**8 + 18810*d**7 + 23628*d**6
         - 86328*d**5 + 121770*d**4 - 102465*d**3 + 53460*d**2 - 16038*d + 2187)
    dout = d**3 * P / (2 * Q)
    psucc = max(1/1728 - 11/2592 * d, 1e-9)          # (L) first order only; (?) beyond
    return (2/3) * dout, psucc


def make_ps(A):
    """PS [[20,7,2]]_3: per-output eps_out = A eps^2 (?), P = 1 - (18m-2k)/3 delta, m=3,k=7 (L)."""
    def f(eps):
        d = 1.5 * eps
        return A * eps**2, max(1 - (54 - 14) / 3 * d, 1e-9)
    return f


CODES_T = {"QRM8": (8, 1, qrm_map, 7)}          # (n, k, map, #stabiliser generators)
CODES_T_PS = {"QRM8": (8, 1, qrm_map, 7), "PS20": (20, 7, make_ps(3.0), 13)}
CODES_S = {"Golay11": (11, 1, golay_map, 10)}


def best_sequence(codes, eps_in, eps_target, max_levels=6):
    """Cheapest sequence (raw states per output) reaching eps_target. Brute force."""
    best = (math.inf, None, None)
    for L in range(0, max_levels + 1):
        for seq in itertools.product(codes.keys(), repeat=L):
            e, cost, ok = eps_in, 1.0, True
            levels = []
            for name in seq:
                n, k, fmap, _ = codes[name]
                eo, P = fmap(e)
                if not (eo < e):
                    ok = False
                    break
                cost *= n / (k * P)
                levels.append((name, e, eo, P))
                e = eo
            if ok and e <= eps_target and cost < best[0]:
                best = (cost, seq, levels)
    return best


# ---------------------------------------------------------------------------------
# per-gate raw cost
# ---------------------------------------------------------------------------------
EPS_R_OVER_EPS_S = R_state_infidelity(1e-8)[1] / (2e-8 / 3)      # = 2.667 (I, simulated)
INJ_R = 3.0            # expected R-states consumed per R gate (L/I)
S_PER_RSTATE = 4 * 8   # 2 S per attempt @ p=1/2 -> 4 S per N ; 2 N @ p=1/4 -> 8 N per R-state


def cost_T(p, eps_gate, codes=CODES_T):
    c, seq, lv = best_sequence(codes, p, eps_gate)
    return c, seq, lv


def cost_R_golay(p, eps_gate):
    # eps_gate ~ INJ_R * eps_Rstate (union bound over consumed states) ; eps_Rstate = 2.667 eps_S
    eps_S_target = eps_gate / (INJ_R * EPS_R_OVER_EPS_S)
    c, seq, lv = best_sequence(CODES_S, p, eps_S_target)
    return INJ_R * S_PER_RSTATE * c, seq, lv


def cost_R_anwar_lowerbound(p, eps_gate):
    """LOWER bound: 5 raw states/round (P_succ<=1), eps2 -> 0.77 eps2 per round (L),
    parity checking: ~4 rounds, >=4 states per output per round, delta grows 2x/round (L),
    equatorialisation 8 plus-states per R-state (I), 3 R-states per gate."""
    eps2_needed = eps_gate / INJ_R / 2**4          # delta doubles over 4 parity rounds
    if p <= eps2_needed:
        rounds = 0
    else:
        rounds = math.ceil(math.log(eps2_needed / p) / math.log(0.77))
    return INJ_R * 8 * 4**4 * 5.0**rounds, rounds


def cost_R_from_T(p, eps_gate, codes=CODES_T):
    c, _, _ = cost_T(p, eps_gate / 7, codes)
    return 7 * c


def cost_R_from_P9_RUS(p, eps_gate, t_per_state=9.0, codes=CODES_T):
    c, _, _ = cost_T(p, eps_gate / (INJ_R * t_per_state), codes)
    return INJ_R * t_per_state * c


# ---------------------------------------------------------------------------------
# crude space-time volume  (ASSUMPTIONS, all (?))
#   - qutrit surface-code patch, logical error per d cycles: pL(d) = 0.1 (p/p_th)^((d+1)/2), p_th=1e-2
#   - each distillation attempt: n_in patches (+ equal routing space) for n_gen*d cycles
#     (one lattice-surgery multi-patch measurement per stabiliser generator)
#   - each raw injection: 1 patch at the level-1 distance for d cycles
#   - conversions / gate injection: 2 patches (+routing) for d cycles
#   - distance at level l chosen so that n_in*n_gen*pL(d) <= 0.1*eps_out(l)
# ---------------------------------------------------------------------------------
P_TH = 1e-2


def dist_for(p, target):
    d = 3
    while 0.1 * (p / P_TH) ** ((d + 1) / 2) > target:
        d += 2
        if d > 101:
            return math.inf
    return d


def volume(p, levels, codes, out_multiplier, eps_final, extra_2patch_ops_per_output):
    """Expected patch-cycle volume (units: d^2 qutrits x cycles) per final gate."""
    if levels is None:
        return math.inf
    outputs = out_multiplier       # final-level states needed per gate
    vol = 0.0
    for (name, e_in, e_out, P) in reversed(levels):
        n, k, _, ngen = codes[name]
        d = dist_for(p, 0.1 * e_out / (n * ngen))
        attempts = outputs / (k * P)
        vol += attempts * (2 * n) * (ngen * d) * d**2
        outputs = attempts * n
    d1 = dist_for(p, 0.1 * levels[0][1]) if levels else 3
    vol += outputs * d1**3                                      # raw injections
    dc = dist_for(p, 0.1 * eps_final)
    vol += extra_2patch_ops_per_output * 4 * dc**3              # conversions + gate injection
    return vol


def main():
    budget = 1e-2
    ND, NR = 227.0, 111.0        # T per rotation (C+D) and R per rotation (C+R) at eps_synth=1e-10
    print("c_R/c_T model — raw magic states per logical gate (perfect Cliffords)\n")
    print(f"R-state infidelity per strange-state infidelity after SS->N->R: {EPS_R_OVER_EPS_S:.3f} (I)")
    rows = []
    for p in [1e-2, 1e-3, 1e-4]:
        for Nrot in [1e2, 1e3, 1e4]:
            eT = budget / (ND * Nrot)
            eR = budget / (NR * Nrot)
            cT, sT, lT = cost_T(p, eT)
            cTps, sTps, _ = cost_T(p, eT, CODES_T_PS)
            cRg, sRg, lRg = cost_R_golay(p, eR)
            cRa, nra = cost_R_anwar_lowerbound(p, eR)
            cR7 = cost_R_from_T(p, eR)
            cR7ps = cost_R_from_T(p, eR, CODES_T_PS)
            cRp9 = cost_R_from_P9_RUS(p, eR, 27/4)
            # space-time (only meaningful below threshold)
            if p < P_TH:
                vT = volume(p, lT, CODES_T, 1.0, eT, 1)
                vRg = volume(p, lRg, CODES_S, INJ_R * S_PER_RSTATE, eR,
                             INJ_R * (1 + 4 + 2))   # per gate: 3 inj, 12 N-attempts->... (crude)
                cT7, sT7, lT7 = cost_T(p, eR / 7)
                vR7 = 7 * volume(p, lT7, CODES_T, 1.0, eR / 7, 1)
            else:
                vT = vRg = vR7 = float("nan")
            rows.append((p, Nrot, eT, eR, cT, sT, cTps, cRg, sRg, cRa, nra, cR7, cR7ps, cRp9,
                         vT, vRg, vR7))
            print(f"\np={p:g}  N_rot={Nrot:g}  eps_T/gate={eT:.1e}  eps_R/gate={eR:.1e}")
            print(f"  (a) T  QRM8     : {cT:10.4g} raw/T   seq={sT}")
            print(f"  (a') T QRM8+PS20: {cTps:10.4g} raw/T   seq={sTps}  (PS eps_out=3eps^2 assumed)")
            print(f"  (b) R  Golay    : {cRg:10.4g} raw/R   seq={sRg}  -> c_R/c_T = {cRg/cT:8.3g}")
            print(f"  (c) R  [[5,1,3]]: >= {cRa:9.3g} raw/R  ({nra} linear rounds) -> c_R/c_T >= {cRa/cT:.3g}")
            print(f"  (d) R  = 7T     : {cR7:10.4g} raw/R   -> c_R/c_T = {cR7/cT:6.3g}  (PS-T: {cR7ps/cTps:.3g})")
            print(f"  (iii) R via 27/4 P9 RUS: {cRp9:10.4g} raw/R -> c_R/c_T = {cRp9/cT:.3g}")
            print(f"  per-rotation raw cost: C+D {ND*cT:.4g}  vs  C+R(Golay) {NR*cRg:.4g}  "
                  f"C+R(7T) {NR*cR7:.4g}")
            if p < P_TH:
                print(f"  crude volume (d^3 units): T {vT:.3g}; R-Golay {vRg:.3g} (ratio {vRg/vT:.3g}); "
                      f"R-7T {vR7:.3g} (ratio {vR7/vT:.3g})")
    # equal-target table
    print("\nEqual per-gate target table (c_R/c_T at eps_R = eps_T):")
    print(" p      eps     c_T(QRM)   c_R(Golay)   ratio   c_R(7T)/c_T")
    for p in [1e-2, 1e-3, 1e-4]:
        for e in [1e-6, 1e-8, 1e-10, 1e-12]:
            cT, _, _ = cost_T(p, e)
            cRg, _, _ = cost_R_golay(p, e)
            cR7 = cost_R_from_T(p, e)
            print(f" {p:6.0e} {e:6.0e} {cT:10.4g} {cRg:12.4g} {cRg/cT:9.3g} {cR7/cT:9.3g}")
    return rows


if __name__ == "__main__":
    main()
