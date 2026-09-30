# Qutrit Clifford+D vs Clifford+R: equal phase count, 3.4× fewer T gates

*Version 2026-10 (data as of 2026-09-30). Replaces `unified/cd_vs_cr_one_pager_2026-06-04.md`/`.tex`. Henry Lamm (Fermilab/SQMS), with Nick's exact-ring C+D data. Numbers come from `unified/cd_ft_cost_results_2026-09-30.md` (the summary of record), and the derivations from `unified/cd_approx_cr_notes.md`.*

**TL;DR.**
- **Per non-Clifford phase, the two gate sets tie.** Approximating R_z(θ) to Frobenius error ε costs N_D ≈ 3.94 + **5.16**·log₃(1/ε) phases in qutrit Clifford+D (C+D), against **5.14**·log₃(1/ε) R gates in Clifford+R (C+R, Householder).
- **On a qutrit machine whose only magic state is T, C+D is about 3.4× cheaper.** C+D costs **T ≈ 16.6 + 10.0·log₃(1/ε)**, about **227 T at ε = 10⁻¹⁰**. C+R Householder costs **~776 T** when each R is built from 7 T.
- **Against a C+R device with its own R-state factory, C+D wins iff c_R ≳ 2 c_T.**

## 1. Why the per-phase counts tie: Galois shadows

**The shadows.**
- A C+D unitary has entries in Z[ζ₉, 1/3]. The Galois maps ζ₉ → ζ₉² and ζ₉ → ζ₉⁴ send it to two other matrices, its "shadows".
- The shadows are also exactly unitary, because Galois maps commute with complex conjugation.
- The hardware only ever uses the physical embedding σ₁, but every candidate must satisfy all three unitarity conditions.
- A demonstration is in `unified/nick_test/galois_shadows_demo.py`. The physical matrix hits R_z to 2×10⁻⁷, and both shadows are generic unitaries far from the target.

**The counting argument (heuristic).**
- Count the lattice points of Z[ζ₉] ⊂ C³ that satisfy "physical entry in the ε-region, both shadow entries in the unit disk".
- The count scales as A(ε)·N(δ), where N(δ) = 3^k is the norm of the denominator at sde level k.
- For Z[ω] ⊂ C, which has no shadows, the count is also A(ε)·3^k.
- So both rings gain the same factor of 3 in candidates per sde level. C+D's extra points at a fixed denominator 3^f are paid for by needing 6f levels instead of 2f, and they differ mostly in the shadows, which nobody measures.

**Consistency with every negative result.** Bocharov-polytope, Babai-CVP, lifting through the Z[ω] subring and SK warm-starts all fail for the same reason: the extra points sit in the shadow directions (notes §3–§6).

**RUS does not break the tie.** It relaxes the target region from a ball to a wedge in both rings equally, and the shadow constraint remains (notes §8b).

**Status of the argument.** It is a heuristic argument plus a rigorous partial theorem: the σ₁ covering radius is Θ(B^{−1/2}). A full proof for complete unitaries is open (notes §4, §7).

## 2. Why the tie is in the wrong unit

**The per-phase count double-charges T.**
- N_D charges 1 for every ζ₉ phase with 3∤a and 1 for every R.
- A qutrit T = diag(1, ζ₉, ζ₉⁻¹) carries two such phases, so it is charged 2.
- A C+D circuit is actually made of three physically different gates (results §2, f = 14 per rotation):

| Gate | Count at f = 14 | Clifford hierarchy | Direct FT resource |
|---|---|---|---|
| T-type diagonal (T^±1 up to Clifford) | 39.0 | level 3 | T-state factories exist (qutrit Reed–Muller; Prakash–Saha) |
| Level-4 diagonal (diag(1,1,ζ₉) up to Clifford) | 17.0 | level 4 | none known; transversality excluded for d = 3 (Watson et al. 1503.08800) |
| R = diag(1,1,−1) | 7.9 (incl. 0.77 residual) | **in no level** | none direct; no transversal R on any qutrit code (JOKY 1710.07256) |

**Both R and level-4 are unavoidable.** Removing either from the reducer gives 0/10 decompositions (`nick_test/no_r_test.py`, `nick_test/no_l4_test.py`).

## 3. New gadgets: R and level-4 from 7 T each

Both gadgets are exact and deterministic, and use one clean ancilla that is returned to |0⟩. Both were re-verified on 2026-09-30 to ~3×10⁻¹⁵ (`unified/r_from_d/verify_R7T.py`, `unified/level4/l4_7T_verify.py`).

| Gate | T-count | Previous best | Optimality |
|---|---|---|---|
| R = diag(1,1,−1) | **7** | 39 with a borrowed ancilla (GRVY 2202.09235); 24 with a clean ancilla (our construction from GRVY blocks) | optimal with 1 ancilla (exhaustive, merge-free); ≤6 T not found with 2 ancillas (merged search) |
| diag(1,1,ζ₉) (level 4) | **7** | 8 (GRVY's controlled block on a \|0⟩ ancilla) | optimal with 1 ancilla (exact orbit MITM); ≥6 with 2 ancillas; impossible with 0 ancillas |

- **Level-4 circuit:** a |2⟩-controlled X (3 T) writes [x=2] into the ancilla, a T on the ancilla adds the phase ζ₉^[x=2], and the inverse controlled X (3 T) uncomputes.
- **Method:** exact meet-in-the-middle over Clifford-conjugated two-qutrit T rotations (`r_from_d/mitm_core.py`).
- **Novelty:** see `unified/paper_prep/novelty_check.md`.

## 4. The cost comparison

**Per-rotation T-cost** = n_T + 7 n_4 + 7 n_R. These are Nick's 900 matrices (f = 4…16), all decomposed, recounted with the residual-R fix (`nick_test/nick_tcost_2026-09-30.csv`).

| f | median ε | N_D (per phase) | n_T / n_4 / n_R | **T-cost** |
|---|---|---|---|---|
| 4 | 5.8e-3 | 30.3 | 10.6 / 5.5 / 2.9 | **70** |
| 8 | 2.8e-6 | 61.9 | 23.3 / 10.1 / 4.6 | **126** |
| 12 | 6.9e-9 | 90.9 | 34.3 / 14.9 / 7.1 | **188** |
| 16 | 3.2e-11 | 121.6 | 46.4 / 19.8 / 8.5 | **245** |

**Device comparison at ε = 10⁻¹⁰.** C+R counts are Gustafson et al.'s fits (arXiv:2503.20203).

| Device / synthesis | Slope per log₃(1/ε) | Cost at 10⁻¹⁰ |
|---|---|---|
| **C+D synthesis, T-factory device** | **10.0** | **~227 T** |
| C+R Householder, R built from 7 T | 36.0 | ~776 T (3.4×) |
| C+R Exhaustive, R built from 7 T | 28.8 | ~619 T (2.7×) |
| C+R Householder, dedicated R-state factory | 5.14·c_R | ~111 c_R |

**Break-even against an R-state factory.**
- C+D wins iff c_R ≳ **2.0 c_T** against Householder, or 2.5 c_T against Exhaustive.
- RUS injection consumes about 3 R-states per R. So C+D wins whenever an R-state costs more than about 0.7–0.85 T-states.
- That looks likely, because R-states have no direct high-order distillation. The known routes are 5-qutrit linear suppression, or ~32 Golay strange states per R-state (notes §11).

## 5. Caveats

1. **The selection is not T-optimized.** Nick's approximants were chosen by Frobenius error. On HRSA candidates, choosing by T-cost cut T by 36–61% at nearly the same error (results §5). The C+D numbers above are therefore likely **upper bounds**. The `HRSA_tester --rank-tcost` flag exists, but Nick's and zeta9's selection have not been rerun yet.
2. **Factory costs are not modelled.** "T-cost" counts magic states consumed and does not include distillation overhead. The C+R break-even is a condition on c_R/c_T, not a measured ratio. A T-vs-R factory comparison at a fixed physical error rate is still open.
3. **One clean ancilla.** The 7-T gadgets need one ancilla qutrit in |0⟩, which is reused and returned clean. Without an ancilla, R is not in single-qutrit C+T at all (GRVY), and level-4 is impossible by a determinant argument.
4. **The gadgets are new and not yet peer reviewed.** Optimality is proven only for one ancilla. Whether 6 T is possible with 2 ancillas is open for both gadgets.
5. **Unit discipline.** The June one-pager's 5.12 and every N_D or R count computed before 2026-09-30 undercount R by up to 1 per rotation. The corrected per-phase slope is 5.16.
6. **The per-phase tie is empirical at the level of the constant.** The shadow argument explains why the slopes should match, not why the constants are equal.

**Paper framing.**
- The two gate sets are equally expressive per non-Clifford phase, and the Galois-shadow argument says why.
- On the hardware that exists in the qutrit literature, stabilizer codes with T-state factories, C+D synthesis plus two 7-T gadgets is 2.7–3.4× cheaper than C+R.
- It stays ahead of any R-factory device whose R-states cost more than about 2× a T-state per R gate.

**Sources.**
- Results and data: `unified/cd_ft_cost_results_2026-09-30.md` (§1–§5); `unified/nick_test/nick_tcost_2026-09-30.csv` (script `nick_test/nick_tcost_all.py`).
- Notes: `unified/cd_approx_cr_notes.md` §3–§4 (shadows), §9–§12 (R, hierarchy, fault tolerance).
- Gadgets: `unified/r_from_d/R_7T.json`, `unified/r_from_d/D4_7T.json`, `unified/level4/l4_7T_gates.json`.
- Convention: `unified/paper_prep/convention_memo.md`.
- References:
  - Gustafson et al. arXiv:2503.20203.
  - Kalra et al. arXiv:2311.08696.
  - Evra–Parzanchevski arXiv:2401.16120.
  - Glaudell–Ross–van de Wetering–Yeh arXiv:2202.09235.
  - Cui–Gottesman–Krishna arXiv:1608.06596.
  - Jochym-O'Connor–Kubica–Yoder arXiv:1710.07256.
  - Watson et al. arXiv:1503.08800.
