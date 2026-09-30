# Convention memo: what "C+D" means and how N_D is counted

*2026-09-30. Prepared for the C+D synthesis paper. Sources: `unified/cd_ft_cost_results_2026-09-30.md` (§1–§5), `unified/cd_approx_cr_notes.md` (§9, §11, §12), and the draft `Clifford_D_Householder/Synthesis_of_Single_Qutrit_Circuits_from_Clifford_D.zip` → `ESA_CliffordD_fixed.tex` (the same text as `summary/Synthesis_of_Single_Qutrit_Circuits_from_Clifford_D.pdf`, dated May 29 2026).*

---

## 1. The inconsistency

| Source | Definition of the diagonal set | Is R = diag(1,1,−1) a member? | Is R charged? |
|---|---|---|---|
| Draft, l.232 (Intro) | "diagonal matrices whose diagonal entries are **ninth roots of unity**" (unsigned) | No | — |
| Draft, l.590 | "**the** non-Clifford gate D = Diag(ζ₉, 1, ζ₉⁻¹)" (this is the qutrit T, up to Clifford) | No | — |
| Draft, l.597, l.603, l.695, l.707 | uses entries ±ζ₉^k; uses an undefined set 𝒟 | Implicitly yes | — |
| Draft, eq. (32), l.624 | Syllable H·Diag(ζ^{a₁},ζ^{a₂},ζ^{a₃})·**R^ε**·X^δ | Used as a generator | Alg. 6 "Update N_D from D": R not charged |
| Draft, eq. (37), l.709–712 | V = Diag(ᾱ, α, 1) with α = **±**ζ₉^a: N_D = 0 if 3∣a, else 2 | — | **No.** α = −1 gives Diag(−1,−1,1) = −R, charged 0 |
| Draft, l.241 | C+D = U(3, Z[ζ₉, 1/3]) (citing EP Thm 2.8) | Needs R ∈ C+D, because R ∈ U(3, Z[ζ₉,1/3]) | — |
| Kalra et al. 2311.08696 | 𝒟 = {diag(±ξ^a, ±ξ^b, ±ξ^c)}, ξ = ζ₉; "R := R_[0,0,1]" | **Yes** | — |
| Evra–Parzanchevski 2401.16120 | C+D = {H} ∪ 𝒟, same 𝒟 | **Yes** | — |
| Our code (`hrsa/canonical_reducer.py`, `hrsa/decompose*.{cpp,h}`) | Signed 𝒟 | Yes | **Yes**: +1 per R^ε in a syllable, and since 2026-09-30 +1 for a mixed-sign residual monomial |

The draft therefore uses two definitions. Its prose uses the unsigned set; its syllables, proofs and the EP citation need the signed set. Its gate count (Alg. 6, eq. (37)) charges nothing for R.

## 2. Empirical facts that constrain the choice

All from `cd_ft_cost_results_2026-09-30.md` unless marked.

1. **Synthesis needs R.** If the reducer's prefix table is restricted to ε = 0 (no R), 0 of 10 f = 10 matrices decompose, against 10/10 with R (`nick_test/no_r_test.py`; notes §12). All of these matrices have det = 1, so the determinant is not what forces R. Heuristic reason: modulo χ = 1−ζ₉, every unsigned D reduces to I and R reduces to diag(1,1,−1). So only R can change the signs of the leading residues (p,q,r), which is Kalra's R^ε normalisation.
2. **Synthesis needs level-4 diagonals.** With the table restricted to level-3 or Clifford diagonals, 0/10 decompose, with or without R (`nick_test/no_l4_test.py`).
3. **R is not in the unsigned group, up to the global-phase caveat.** Every element of ⟨H, Clifford, unsigned D⟩ has det ∈ μ₉, and det R = −1. That rules out R and ζ^k R exactly, but not −ζ^k R. A meet-in-the-middle search up to 6 syllables found no word equal to R up to global phase (`nick_test/r_from_d_search.py`). This is evidence, not a proof.
4. **Physical classification** (§2 of the results):
   - diag(ζ^a, ζ^b, ζ^c) is at **level 3** iff a+b+c ≡ 0 (mod 3). Up to Clifford it is then T or T⁻¹, and it carries 2 phases with 3∤a.
   - Otherwise it is at **level 4**: up to Clifford and global phase, a single-position phase with 1 such phase.
   - **R and every mixed-sign 𝒟 element lie in no level of the Clifford hierarchy.** Hierarchy diagonals have 3^m-th-root phases (Cui–Gottesman–Krishna 1608.06596). No qutrit stabilizer code has a transversal R (Jochym-O'Connor–Kubica–Yoder 1710.07256).
5. **Composition per rotation** (f = 14, ε ≈ 3.6×10⁻¹⁰, corrected reducer): 39.0 T-type, 17.0 level-4, 7.9 R, including 0.77 residual R.
6. **Gadget costs** with one clean ancilla, exact and deterministic: T-type 1 T, level-4 7 T, R 7 T (`r_from_d/verify_R7T.py`, `level4/l4_7T_verify.py`; both re-run 2026-09-30 and PASS to ~3×10⁻¹⁵).

## 3. Options

### Option A: unsigned C+D, R free (the draft as written)
- **What is true:** the gate set is at most level 4, so its FT story is "T plus level-4".
- **What breaks:**
  - The synthesis in the paper does not produce words in this gate set (fact 1).
  - The l.241 identification C+D = U(3, Z[ζ₉,1/3]) (EP Thm 2.8) is false for this set. U(3, Z[ζ₉,1/3]) contains R, and R is not in the unsigned group (fact 3, up to the stated caveat).
  - It silently drops the most expensive gate in any FT accounting.
- **Effect on the numbers:** the 900-matrix recount has R = 1.0 + 0.34·log₃(1/ε). That is 2.9 R at f = 4 and 8.5 R at f = 16, and about 8 at ε = 10⁻¹⁰, which option A would report as 0. The per-phase slope would fall from 5.16 to about 4.8 (5.16 − 0.34). This looks better than C+R, but only because R is not counted: **a phantom advantage.** In T-cost terms it removes about 7·8 ≈ 57 T of the 227 at ε = 10⁻¹⁰.
- **Verdict:** not defensible.

### Option B: signed 𝒟 (Kalra/EP), per-phase count N_D (current code)
- **Definition:** N_D = Σ_syllables #{phases with 3∤a} + n_R, which is ≈ 2 n_T + n_4 + n_R.
  - At f = 14: 2·39.0 + 17.0 + 7.9 = 102.9, against a measured N_D of 103.4.
  - The small gap comes from the residual monomial phases.
- **Pros:**
  - Matches the code and every CSV to date.
  - It is the unit in which the per-phase tie with C+R appears: N_D = 3.94 + 5.16·log₃(1/ε), against 5.14 for C+R Householder.
- **Cons:**
  - The draft names D = diag(ζ,1,ζ⁻¹), which is T, as *the* non-Clifford gate. A reader will read N_D as "number of D gates", but N_D charges every T twice.
  - It gives equal weight to three physically very different gates.
  - It is not a hardware cost.

### Option C: signed 𝒟, per-gate counts (n_T, n_4, n_R)
- **Definition:** each non-Clifford syllable diagonal counts as one gate of its type, and each R^ε and residual R counts as one R.
- **Numbers:** at f = 14 this is 39.0 + 17.0 + 7.9 ≈ 64 operations, against about 105 R for C+R Householder at the same ε. (This is the column sum of results §4; notes §9 give ≈62 from the pre-fix counts.)
- **Pros:** honest about what a device must execute.
- **Cons:** "C+D 64 vs C+R 105" invites the wrong conclusion that C+D is 1.6× cheaper. A qutrit T, a level-4 gate and an R do not cost the same.

### Option D: T-cost on a T-factory machine (weights T 1, level-4 7, R 7)
- **Definition:** T-cost = n_T + 7 n_4 + 7 n_R.
- **Fit:** 16.6 + 10.0·log₃(1/ε), R² = 0.874, n = 900. That is ≈ 227 T at ε = 10⁻¹⁰.
- **Pros:** one scalar with a physical meaning. The only gate a C+D device then has to distil is T, which already has protocols (qutrit Reed–Muller; Prakash–Saha 2403.06228).
- **Cons:**
  - It depends on the gadget costs, which are new and not yet peer reviewed (see `novelty_check.md`).
  - It assumes one reusable clean ancilla.
  - It ignores factory costs. The comparison with an R-factory device has to be stated as a break-even condition.

## 4. Recommendation

1. **Gate set: adopt the signed 𝒟 of Kalra et al. and Evra–Parzanchevski, and say so explicitly.** This is the only choice consistent with l.241 (EP Thm 2.8), the syllable form (32), Kalra's Theorem 5.7 as used in §IX, and the code. State explicitly that 𝒟 contains R, and that R and the mixed-sign elements lie outside the Clifford hierarchy.
2. **Primary reported data: the per-type counts (n_T, n_4, n_R)**, where n_R includes the residual R. Every table of synthesis results should carry these three columns.
3. **Headline FT metric: T-cost** = n_T + 7 n_4 + 7 n_R, with the two 7-T gadgets given explicitly, including circuits or a supplementary file, and the ancilla assumption stated.
4. **Keep the per-phase count, but rename it.**
   - Recommended symbol: **N_φ**, the "non-Clifford phase count", defined by N_φ = 2n_T + n_4 + n_R up to residual-phase bookkeeping.
   - It is the natural unit for the sde/lattice argument and for the per-phase tie with C+R.
   - If the authors prefer to keep the symbol N_D for continuity with earlier drafts and plots, define it in one displayed equation and never call it a "D-gate count".
5. **Comparisons with C+R always state the unit.** The per-phase slopes tie (5.16 vs 5.14). On a T-factory machine, C+D is 3.4× cheaper than C+R Householder with a 7-T R. Against a C+R device with its own R-factory, C+D wins iff c_R ≳ 2 c_T.
6. **Report the global phase and residual convention.**
   - Residual phases are counted modulo Clifford diagonals *and* a global phase ζ₉^k. For example, (0,1,1) ~ (2,0,0) is one level-4 phase, not two.
   - A mixed-sign residual monomial costs one R. A Clifford monomial has uniform signs up to global −1.
   - Check that the code's residual phase count takes this minimum over global phase before quoting it. If it does not, that would explain the 0.5 gap between N_D and 2n_T + n_4 + n_R.

### Suggested wording for the paper

> *Gate set.* Following Kalra et al. [Kal] and Evra–Parzanchevski [EvPar], we write $\mathcal{D}=\{\Diag(\pm\zeta_9^{a},\pm\zeta_9^{b},\pm\zeta_9^{c})\}$ and $\CD=\langle \mathcal{C}\cup\mathcal{D}\rangle$, where $\mathcal{C}$ is the single-qutrit Clifford group. With this definition $\CD=\U(3,\Z[\zeta_9,1/3])$ [EvPar, Thm. 2.8]. Note that $\mathcal{D}$ contains the metaplectic reflection $\bR=\Diag(1,1,-1)$. Modulo Cliffords and a global phase, every non-Clifford element of $\mathcal{D}$ is a product of one of three kinds of gate:
> (i) a *T-type* diagonal $\Diag(\zeta_9^a,\zeta_9^b,\zeta_9^c)$ with $a+b+c\equiv0\pmod 3$, which equals $T^{\pm1}$ up to Clifford and lies in the third level of the Clifford hierarchy;
> (ii) a *level-4* diagonal, a single-position phase $\Diag(1,1,\zeta_9^{\pm1})$ up to Clifford;
> (iii) $\bR$, which lies in no level of the hierarchy.
> We find that both (ii) and (iii) are unavoidable in our exact decompositions: restricting the syllable table to exclude either one makes peeling fail on every test matrix.
>
> *Cost.* For every compiled rotation we report the counts $(n_T,n_4,n_R)$, where $n_R$ includes a possible residual $\bR$ in the final monomial. On a device whose only non-Clifford resource is the qutrit T state, $\bR$ and every level-4 diagonal can each be implemented exactly with seven T gates and one clean ancilla (Sec. X). The resulting *T-cost* is $n_T+7n_4+7n_R$. To compare with the $\bR$-count of $\CR$ synthesis [Gust] we also quote the phase count $N_\varphi = \sum(\text{phases }\zeta_9^a\text{ with }3\nmid a)+n_R \approx 2n_T+n_4+n_R$. This charges a T gate twice and is not a hardware cost.

## 5. Numbers to use under the recommended convention

From results §4 (Nick's 900 matrices, corrected reducer). The per-phase column is the file's "N_D (corrected)".

| f | median ε | n_T | n_4 | n_R | N_φ (= "N_D corrected") | T-cost |
|---|---|---|---|---|---|---|
| 4 | 5.8e-3 | 10.6 | 5.5 | 2.9 | 30.3 | 70 |
| 8 | 2.8e-6 | 23.3 | 10.1 | 4.6 | 61.9 | 126 |
| 12 | 6.9e-9 | 34.3 | 14.9 | 7.1 | 90.9 | 188 |
| 14 | 3.6e-10 | 39.0 | 17.0 | 7.9 | 103.4 | 214 |
| 16 | 3.2e-11 | 46.4 | 19.8 | 8.5 | 121.6 | 245 |

Fits vs log₃(1/ε):

| Quantity | Fit | R² | Value at ε = 10⁻¹⁰ (log₃ = 20.96) |
|---|---|---|---|
| N_φ | 3.94 + 5.16·log₃(1/ε) | 0.912 | ≈112 |
| T-cost | 16.6 + 10.0·log₃(1/ε) | 0.874 | ≈227 |
| n_R | 1.0 + 0.34·log₃(1/ε) | — | ≈8 |

**Caveats to carry with every number.**
- These use Nick's frob-driven candidate selection, not T-cost-aware selection. On HRSA candidates, T-cost ranking cut 36–61%, so these C+D numbers are probably upper bounds.
- Every N_D or R count produced before 2026-09-30 is low by up to one R (the residual-R bug). This includes the June one-pager's 5.12 and the paper-data snapshot of 2026-05-23. Regenerate them before use.
