# Qutrit Clifford+D vs Clifford+R: fault-tolerant cost results (2026-09-29/30)

Summary of results from the 2026-09-29/30 session, kept as a record for later work. The derivations and literature discussion are in `cd_approx_cr_notes.md` (§3–§12).

## 1. Headline gate-count result (unchanged)
- C+D exact-ring synthesis (Nick's data, f = 4…16, n = 900): **N_D = 3.89 + 5.12·log₃(1/ε)**.
- C+R: Householder 5.14·log₃(1/ε), Exhaustive 4.11·log₃(1/ε) (Gustafson et al., arXiv:2503.20203).
- **The tie holds only in "per-phase" N_D units.** N_D charges each ζ₉ phase with 3∤a, so a qutrit T gate is charged 2. Counted per non-Clifford *operation*, C+D uses about 62 vs about 105 for C+R at ε ≈ 3.6×10⁻¹⁰.
- **Why the per-phase slopes tie.** Every exact C+D matrix has two Galois "shadow" readings, ζ₉ → ζ₉², ζ₉⁴, which must also be unitary. The number of candidates is ∝ A(ε)·Norm in both rings, so the sde needed is the same. See notes §3–4.
- **RUS would not break the tie.** Its gain comes from relaxing the target region (the qubit case goes from ε³ to ε), which helps both rings equally. The shadow constraint still applies. Tabled.

## 2. What the non-Clifford gates are
- R = diag(1,1,−1) is the same gate in C+R (Gustafson) and C+D (Kalra's signed 𝒟). **R is in no level of the Clifford hierarchy**, and it cannot be transversal on any qutrit stabilizer code (Jochym-O'Connor–Kubica–Yoder, 1710.07256).
- ζ₉ diagonal diag(ζ^a, ζ^b, ζ^c) is **level 3 iff a+b+c ≡ 0 (mod 3)**. Up to Clifford that is exactly the qutrit T or T⁻¹. Otherwise it is **level 4**, a single-position phase up to Clifford.
- **Composition of Nick's f=14 circuits, per rotation:** 38.8 T-type, 16.3 level-4, 7.2 R syllables, plus a residual R (see §5).
- **Both R and level-4 are required.** Restricting the reducer to ε=0 prefixes, or to level-3 diagonals only, gives 0/10 decompositions (`nick_test/no_r_test.py`, `no_l4_test.py`).
- **No distillation protocol or transversal code exists for qutrit level-4 states.** Watson et al. 1503.08800 Thm 1 excludes d = 3, µ₀ = 4. **R-states have no direct high-order distillation either:** the known routes are 5-qutrit linear suppression, or Golay strange states at ~32 per R-state, plus RUS injection at ~3 R-states per R.

## 3. New gadget costs: exact, deterministic, 1 clean ancilla, verified independently

| Gate | T-count | Previous best | Optimality | Files |
|---|---|---|---|---|
| T-type diagonal | 1 | — | — | — |
| **R = diag(1,1,−1)** | **7** | 39 (borrowed ancilla, GRVY 2202.09235); 24 (clean ancilla) | **Optimal with 1 ancilla** (merge-free exhaustive search, ≤6 impossible). ≤6 impossible with 2 ancillas in the invariant-merged search | `r_from_d/verify_R7T.py`, `R_7T*.json`, `raw_lb_1anc.py` |
| **Level-4 diag(1,1,ζ)** | **7** | 8 (GRVY's 8-T controlled block on a \|0⟩ ancilla, `level4/l4_from_8T.py`) | **Optimal with 1 ancilla** (exact, merge-free orbit MITM, `level4/l4_orbits.py`). ≥6 with 2 ancillas (rigorous); 6 open. Impossible with 0 ancillas (determinant argument) | `level4/l4_7T_verify.py`, `r_from_d/D4_7T*.json` |

- **Method:** exact meet-in-the-middle over products of Clifford-conjugated T rotations (80 two-qutrit Paulis). Isometries are compared up to a left Clifford via a Pauli-multiset invariant in exact Z[ζ] coordinates (`r_from_d/mitm_core.py`).
- **Level-4 7-T circuit, in simple form:** `C2X ; T on ancilla ; C2X†`. The 3-T |2⟩-controlled-X writes [x=2] into the ancilla, T adds the phase ζ^[x=2], and C2X† uncomputes. Independently verified.
- **Other constructions found along the way:**
  - R from 9 level-4 D gates (`r_from_d/e2e.py`), now superseded.
  - R with 15 T plus a Strange-state catalyst.
  - Lower bounds from mana: ≥1 T.

## 4. Cost per rotation on a qutrit T-factory machine (weights T 1, level-4 7, R 7)

**Recount with the fixed reducer (residual R charged).** Nick's 900 matrices, f = 4…16 (`nick_test/nick_tcost_2026-09-30.csv`, script `nick_tcost_all.py`). All decomposed.

| f | n | median ε | N_D (corrected) | T-type | level-4 | R (syl + resid) | **T-cost** |
|---|---|---|---|---|---|---|---|
| 4 | 150 | 5.8e-3 | 30.3 | 10.6 | 5.5 | 2.9 (2.2 + 0.72) | **70** |
| 6 | 150 | 1.4e-4 | 45.5 | 16.7 | 8.0 | 3.7 (2.9 + 0.75) | **98** |
| 8 | 150 | 2.8e-6 | 61.9 | 23.3 | 10.1 | 4.6 (3.9 + 0.67) | **126** |
| 10 | 100 | 2.2e-7 | 75.0 | 28.7 | 12.1 | 5.0 (4.4 + 0.65) | **149** |
| 12 | 100 | 6.9e-9 | 90.9 | 34.3 | 14.9 | 7.1 (6.3 + 0.78) | **188** |
| 14 | 100 | 3.6e-10 | 103.4 | 39.0 | 17.0 | 7.9 (7.2 + 0.77) | **214** |
| 16 | 150 | 3.2e-11 | 121.6 | 46.4 | 19.8 | 8.5 (7.8 + 0.77) | **245** |

**Fits vs log₃(1/ε), n = 900:**
- N_D (corrected) = 3.94 + **5.16**·log₃(1/ε), R² = 0.912. The old uncorrected fit was 5.12, so the residual R adds ~0.7 per rotation roughly uniformly.
- **T-cost = 16.6 + 10.0·log₃(1/ε)**, R² = 0.874. That is **≈ 227 T at ε = 10⁻¹⁰**.
- R count = 1.0 + 0.34·log₃(1/ε), which is ~8 at 10⁻¹⁰.

**Device comparison (C+R counts from Gustafson's fits):**

| Device / synthesis | Slope per log₃(1/ε) | T at ε = 10⁻¹⁰ |
|---|---|---|
| **C+D synthesis on a T-factory device** | **10.0** | **~227** |
| C+R Householder, R built from 7 T | 36.0 | ~776 (3.4× C+D) |
| C+R Exhaustive, R built from 7 T | 28.8 | ~619 (2.7× C+D) |
| C+R device with an R-state factory | 5.14·c_R | 111·c_R |

- Against an R-state factory, C+D wins iff c_R ≳ **2.0 c_T** (Householder) or 2.5 c_T (Exhaustive). With ~3 R-states consumed per R in RUS injection, that means an R-state costing ≳ 0.7–0.85 T-states.
- These numbers use **Nick's selection** (frob-driven), not T-cost-aware selection. T-cost ranking cut 36–61% on HRSA candidates, so the C+D numbers above are likely upper bounds.
- **The C+D device is in effect a qutrit Clifford+T device with one reusable clean ancilla.** It needs only T-state factories, which exist (qutrit Reed–Muller, Prakash–Saha).
- Metaplectic/anyon hardware is dropped as too far out. Both devices are generic qutrit stabilizer-code machines.

**Factory-level c_R/c_T** (`factory_model/`, 2026-09-30; raw injected states per logical gate, perfect Cliffords, depolarising raw error p):

| R source | c_R/c_T | Dominated by |
|---|---|---|
| Golay strange-state factory (Prakash 2003.02717), as published | 3×10⁴ – 6×10⁸ | 1/1728 postselection per round |
| Same, optimistic (all syndromes usable) | 16 – 180 | 32 strange states per R-state × 3 R-states per R (RUS) |
| [[5,1,3]]₃ route (Anwar–Campbell–Browne) | ≥10²⁵ | linear suppression only |
| R-state from P9/T by RUS (BRS) | ≈20–27 | 27/4–9 T per state × 3 |
| **R = 7 T (ours)** | **7 exactly (~8–9 smoothed)** | deterministic |

- T via QRM₃(2) 8→1 was rebuilt exactly: ε' = 2.0ε², threshold 0.211 (matches CAB). Injection is deterministic.
- All conversion and injection steps for R use only Cliffords, stabilizer measurements and postselection. There is no hidden non-Clifford cost.
- **Conclusion.** Every modelled regime (p = 1e-2…1e-4, per-gate targets 1e-6…1e-12) is far above the c_R ≳ 2 c_T break-even. So **C+D beats C+R-with-an-R-factory robustly**, and a C+R device should itself make R from 7 T, which gives the 3.4× C+D advantage above.
- Structural reason: R is transversal on no stabilizer code, so it cannot have an efficient direct factory.
- **Caveats:** a smarter Golay decoder is untested; the Prakash–Saha error coefficient is assumed; no published qudit factory space-time estimates exist, so the volume numbers are assumptions.

**T-cost-aware exact synthesis does nothing (2026-09-30, null result).**
- Setup: `canonical_reducer.set_selection_cost("tcost")` chooses among sde-reducing prefixes by T-cost instead of per-phase D-count (`nick_test/nick_tcost_select-tcost_2026-09-30.csv`, all 900 matrices).
- T-cost 16.55 + 10.02·log₃ (vs 16.61 + 10.02). Per-f means differ by ≤0.1%, with 0–4% of matrices changing either way.
- n_L4 and n_R are identical at every f. Only N_D rises (+2 to +6), because phase-equivalent swaps are free in T-cost.
- **Interpretation:** for a fixed matrix the non-Clifford content of the canonical peel is essentially forced. T-cost is a property of the *matrix*, so the only lever is choosing a different approximant (§5, candidate selection).

## 5. Synthesis-side findings
- **Bug in `hrsa/canonical_reducer.py` `classify_monomial_and_d_cost`.** It hard-coded the residual R to 0. A trailing monomial with mixed entry signs needs one R, because a Clifford monomial has uniform signs.
  - This affected 76% of HRSA candidates and averages 0.7 R per rotation on Nick's f = 10 circuits.
  - Every N_D and R count before 2026-09-30 is low by up to 1 R. (Being fixed.)
- **Candidate selection by T-cost** (`r_count_study/`, 900 HRSA candidates at ε = 1e-2 and 0.1):
  - Within 10% of the best frob, the minimum T-cost is 36–61% lower than the frob-best candidate's.
  - Over all ε-passing candidates it is 2–6× below HRSA_bestD's min-N_D pick. The min-N_D picks are R-heavy (R-light words are longer in N_D).
  - R = 0 candidates exist: 5.6% at ε = 1e-2 and 11% at ε = 0.1.
- **Selection at tighter ε** (`tcost_selection_study/`, ε down to 0.003, thousands of HRSA candidates per angle):
  - Within-angle T-cost spread = 1.03–1.14× the cross-angle spread, so the assumption holds.
  - Best-of-10 saves ~27%, best-of-100 ~42%, best-of-1000 ~57% (one angle).
  - Pools are large (27k candidates per angle at f = 3).
  - **Extrapolated T-cost slope: 10.0 → ≈9.0 (K = 10), ≈8.3 (K = 100), ≈7.8 (K = 1000) per log₃(1/ε).** That is ~180 T at 10⁻¹⁰ for K = 100.
  - Confirming at f = 12–16 needs top-K dumps from Nick's pipeline.
- **Cheap predictor (empirical, 900/900).** Take the leading χ-adic digit of each column-0 numerator, up to global sign. Class `111` means no bookend R. Mixed classes need ≥2 R. Class `111` is necessary for R = 0.
- **Global phase** (−V, ζV) never changes the R count.
- **Euler merging** (several R_z's multiplied, then re-reduced) saves only ~1.3% (ratio 0.987). Dead.

## 6. Code changes (2026-09-30; `unified/`, branch `kprime-cap-stage2-lossy`, uncommitted)
- **Residual-R fix:** mixed-sign trailing monomials now cost 1 R.
  - Python: `hrsa/canonical_reducer.py` (`_unit_sign`, `classify_monomial_and_d_cost`).
  - C++: both `hrsa/decompose.cpp` (int path) and `hrsa/decompose_impl.h` (templated path), in `countMonomialD`'s non-Clifford-phase branch.
  - `test_canonical_reducer.py` passes. The Python fix agrees 12/12 with an independent sign check.
- **`HRSA_tester --rank-tcost [--tcost-w4 7 --tcost-wr 7]`:** HRSA_bestD picks the candidate with the lowest T-cost instead of the lowest D-count. It also prints `CANDTCOST s n_T3 n_L4 n_R tcost` per candidate. The default behaviour is unchanged.
  - Test at θ = 1.0, ε = 1e-2, 100 candidates: the D-count pick has 26 D and **91 T**; the T-cost pick has 57 D and **55 T** (−40%).
- Binaries in `hrsa/` rebuilt.

## 7. Open items
1. ~~Re-report Nick's N_D and T-costs with the fixed reducer~~ Done; see §4.
2. Add T-cost selection to Nick's and zeta9's final candidate selection, with the `111` pre-screen. **Ask Nick to dump the top-K candidates per θ** (see `paper_prep/README.md`, request to Nick).
3. Whether R or level-4 can be done in ≤6 T with 2 ancillas (level-4: ≥6 rigorous).
4. ~~Factory cost comparison~~ Done (§4). c_R/c_T ≥ 16 (optimistic) up to 10⁸ for R factories; 7 via R = 7 T.
5. Unify the paper's convention: the draft's unsigned C+D (no R, and eq. (37) charges no R) vs Kalra's signed set used in the code. Fix the draft error at l.606.
6. Clean up Nick's repository sync: local `zeta9` has been pulled to eaa8068, and the local edit is saved in a `git stash`.
