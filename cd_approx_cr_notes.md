# Why qutrit Clifford+D ≈ Clifford+R — working notes

*Started 2026-09-29. Status: living notes. Sections marked **[heuristic]** still need a rigorous version.*

---

## 1. The result

Single-qutrit targets R_z(θ) = diag(e^{−iθ/2}, e^{iθ/2}, 1). The error ε is Frobenius distance, and the cost is the number of non-Clifford gates, with Cliffords free.

| Gate set / algorithm | N vs ε | N at ε = 10⁻¹⁰ |
|---|---|---|
| **C+D, exact-ring (this work, f = 4…16, n = 900)** | **3.89 + 5.12·log₃(1/ε)** | **111** |
| C+R Householder (Gustafson et al., arXiv:2503.20203) | 3.20 + 5.14·log₃(1/ε) | 111 |
| C+R Exhaustive (same paper) | 2.19 + 4.11·log₃(1/ε) | 88 |
| SU(3) covering lower bound (gate-set independent) | 4.90·log₃(1/ε) − 2.16 | 100 |

- **Verdict:** C+D is statistically tied with C+R Householder (z = −0.34). It is about 1.24× more expensive than C+R Exhaustive.
- **Two-qubit C+T comparison:** relative to emulating a qutrit with two qubits (10·N_T^RUS per arbitrary qutrit gate), C+D costs 1.41× at ε = 10⁻¹⁰ and 1.69× asymptotically. C+R Householder costs 1.40× and 1.69×.
- **Retraction:** an earlier claim that "C+D beats C+R by 1.67×" came from a per-decade vs per-log₃ slip. It is retracted.
- **Data:**
  - `unified/nick_test/nick_nd_vs_eps_all.csv`: exact ring matrices, all exactly unitary, all decomposed with zero reconstruction residual.
  - Plot: `unified/nd_vs_eps_log10_allnick_2026-06-04.png`.

This is surprising because the gate sets are nested: C+R ⊊ C+T ⊆ C+D. C+D can reach strictly more unitaries exactly, so one expects it to approximate more cheaply. These notes explain why it doesn't.

---

## 2. Setup and notation

| | C+R | C+D |
|---|---|---|
| Ring | Z[ω, 1/3], ω = e^{2πi/3} | Z[ζ₉, 1/3], ζ₉ = e^{2πi/9} |
| Field degree over Q | 2 | 6 |
| Complex embeddings (up to conjugation) | **1** | **3**: σ₁, σ₂, σ₄ |
| Prime above 3 | χ = 1 − ω, 3 = unit·χ² | χ = 1 − ζ₉, 3 = unit·χ⁶ |
| Absolute norm of χ | 3 | 3 |
| Denominator 3^f ↔ sde_χ | 2f | 6f |

Gates:
- **R** = diag(1, 1, −1).
- **D** = diag(ζ₉^a, ζ₉^b, ζ₉^c). It costs one per phase with a ≢ 0 (mod 3), since ζ₉³ = ω is Clifford.
- **Exact synthesis:** peel one syllable at a time, lowering sde_χ.
  - C+R always has a syllable that lowers sde_χ by 1 (Property P).
  - C+D has Kalra's obstruction: when Δ = p₁²+q₁²+r₁² ≡ −1 (mod 3), no single syllable works. A two-syllable ("double-prefix") step always resolves it empirically.
  - The workaround adds no net cost (see §5).

---

## 3. What a "shadow" is

**Definition.** Every entry of an exact C+D unitary U is a polynomial in ζ₉ with rational coefficients. Nothing in the algebra singles out ζ₉ = e^{2πi/9}. Substituting another primitive 9th root of unity gives an equally valid algebraic object:

- σ₁: ζ₉ ↦ e^{2πi/9} gives the **physical** matrix, the one the hardware applies.
- σ₂: ζ₉ ↦ e^{4πi/9} gives **shadow #1**, σ₂(U).
- σ₄: ζ₉ ↦ e^{8πi/9} gives **shadow #2**, σ₄(U).
- σ₈, σ₇, σ₅ are just the complex conjugates of these three.

**Key fact: the shadows are unitary too.**
- Galois automorphisms of Q(ζ₉) commute with complex conjugation, because the Galois group is abelian.
- So σ(U)·σ(U)† = σ(U·U†) = σ(I) = I.
- Any exact identity satisfied by U is also satisfied by its shadows.

**What a shadow looks like as a circuit.** Apply σ to every gate of a circuit for U:

| Gate | σ₄ shadow (ζ₉ ↦ ζ₉⁴) | σ₂ shadow (ζ₉ ↦ ζ₉²) |
|---|---|---|
| Clifford C (entries in Q(ω)) | C, unchanged (σ₄ fixes ω and √−3) | C̄, complex conjugate, still Clifford |
| D = diag(ζ₉^a, ζ₉^b, ζ₉^c) | diag(ζ₉^{4a}, ζ₉^{4b}, ζ₉^{4c}) | diag(ζ₉^{2a}, ζ₉^{2b}, ζ₉^{2c}) |

So a C+D circuit is really a recipe that simultaneously builds three unitaries. They share the same Clifford skeleton, and the D-phases are multiplied by 1, 2 and 4 respectively. You only get to use one of them (σ₁), but all three are forced to exist and to be unitary.

**Concrete example.** Take one of Nick's f = 10 matrices, θ ≈ 0. Script: `unified/nick_test/galois_shadows_demo.py`.

```
sigma_1: unitarity err = 7.5e-16   |U - Rz|_F = 2.20e-07   |row 0| = [1.    0.    0.   ]
sigma_2: unitarity err = 1.6e-15   |U - Rz|_F = 2.50e+00   |row 0| = [0.486 0.569 0.663]
sigma_4: unitarity err = 1.6e-15   |U - Rz|_F = 2.22e+00   |row 0| = [0.357 0.386 0.851]
```

- The physical matrix hits the target to 2×10⁻⁷.
- Both shadows are perfectly unitary, but they are generic-looking unitaries that are nowhere near R_z(θ).
- Every θ sampled looks the same way.

**Comparison with C+R.** Z[ω] has only one embedding up to conjugation, because the only nontrivial automorphism is ω ↦ ω̄, i.e. complex conjugation. The "shadow" of a C+R unitary is Ū, which carries no independent constraint. **C+R has no shadows.**

**Comparison with qubit C+T.** Z[ζ₈] has two embeddings, so C+T has one shadow (ζ₈ ↦ ζ₈³, equivalently √2 ↦ −√2). This is exactly the "•-conjugate" condition in Ross–Selinger's grid problem: the shadow entries must also have magnitude ≤ 1. Ross–Selinger's 3·log₂(1/ε), and the room that repeat-until-success schemes found below it, both trace back to this shadow constraint.

---

## 4. Why shadows cancel the extra expressiveness **[heuristic]**

**Counting candidates.** Consider one entry of the approximant, u = x/δ with x ∈ Z[ζ₉] and denominator δ with absolute norm N(δ) = 3^k (so sde_χ = k).

Conditions:
1. **Physical:** σ₁(u) must lie in the target region, which has some area A(ε) in the unit disk.
2. **Shadows:** |σ₂(u)| ≤ 1 and |σ₄(u)| ≤ 1, forced by unitarity of the shadows.

Z[ζ₉] sits in C³ = R⁶ as a lattice with fixed covolume. The allowed region for x has volume

  A(ε)·|σ₁(δ)|² × π|σ₂(δ)|² × π|σ₄(δ)|² ∝ A(ε) · N(δ) = A(ε) · 3^k.

For C+R the same computation in C = R² gives A(ε)·|δ|² = A(ε)·3^k.

**So in both rings, the number of candidates at sde level k is about A(ε)·3^k.** Finding at least one needs k ≈ log₃(1/A(ε)), the same k for both gate sets.

Two remarks make this precise:
- **Only the norm matters, not how it is distributed.**
  - Units of Z[ζ₉] (rank 2) can move scale between the physical place and the shadows. For example, |σ₁(1−ζ₉)|² = 0.47 while |σ₄(1−ζ₉)|² = 3.88.
  - But the volume depends only on the product, which is the norm.
  - There is no way to put the whole norm budget into the physical place. The shadows always take their share.
- **The "more expressive" intuition, made precise.**
  - At a fixed physical scale (denominator 3^f), Z[ζ₉] has about 3^{6f} candidates versus 3^{2f} for Z[ω]. That is the expressiveness you expected.
  - But reaching denominator 3^f costs sde 6f in Z[ζ₉] versus 2f in Z[ω], three times as many peeling levels.
  - Per level, which is what gates pay for, both rings provide one factor of 3 in candidates.
  - The extra candidates are exactly paid for, and they differ from one another mostly in the shadows, which the physical gate doesn't see.

**From levels to gates.** Each sde level costs about a constant number of non-Clifford gates. So N_gates ≈ c · k ≈ c · log₃(1/A(ε)), and the two gate sets can differ only in the constant c.

**Consistency check with our data.**
- Nick's decompositions start at sde_χ ≈ 5.8–6.0·f: 58 at f = 10, 72 at f = 12, 82 at f = 14. This matches the "3^f ↔ 6f" row in §2.
- They use about 1.25 D per sde level: 74.4/58, 90.2/72, 102.7/82.
- Observation, not yet understood: the ratio 5.12/4.11 = 1.245 of our slope to C+R Exhaustive is about the same 1.25. That is what you would get if Exhaustive C+R pays about 1 R per level and C+D about 1.25 D per level at the same k. Worth checking against Gustafson's syllable statistics.

**What still needs to be made rigorous.**
- A(ε) for the full unitary, not a single entry. This means handling the other entries through the norm equation.
- Showing that norm-equation solvability has comparable density in both rings.
- The partial rigorous result so far: the σ₁-projection covering radius of Z[ζ₉] is Θ(B^{−1/2}), the trivial lattice rate. So the shadows do not secretly improve physical coverage. Source: memory note `path_c_research_dead_2026-05-25`.

---

## 5. Kalra's obstruction is not the reason

- The obstruction occurs at roughly every other level, and it is resolved by a double prefix, which usually drops sde by 2–3.
- Measured over Nick's decompositions: 0.91 syllables per sde level and 1.38 D per syllable.
- So the obstruction is absorbed with no visible overhead.
- The parity with C+R comes from §4, the lattice/norm accounting, not from the peeling procedure.

Sources:
- `summary/Synthesis_of_Single_Qutrit_Circuits_from_Clifford_D.pdf`, sections "The Kalra obstruction" and "Double-prefix resolution".
- `unified/hrsa/canonical_reducer.py:1149`.

---

## 6. Ways C+D could beat C+R that were tried and failed

| Idea | Why it failed |
|---|---|
| Port Bocharov's polytope enumeration to Z[ζ₉] | The shadows form a 4-D free ball: about 3×10²⁹ points at f = 12, ε = 10⁻⁴. |
| Babai closest-vector rounding | 2.6–9× worse than HRSA; joint entry selection needed. |
| "Path C" (lift through the Z[ω] subring) | The covering radius Θ(B^{−1/2}) cannot be improved. |
| Warm-start from Solovay–Kitaev output | SK matrices are in a different ideal class, so they give no useful start. |
| Merge Euler pieces: multiply several R_z's, then re-reduce (2026-09-29) | Ratio 0.987 against the naive sum over 10 trials (about 5 D saved of about 375). Log: `unified/nick_test/euler_recompile_f10_K5_2026-09-29.log`. |

All of these fail for the same reason: any extra points live in the shadow directions.

**What could plausibly beat C+R.** Anything that escapes the requirement that every shadow be unitary. Note: RUS does **not** do this (see §8b); it relaxes the target region instead and helps C+R equally. No concrete mechanism for this is known. The routes that could still break the tie are fault-tolerant gate cost (§8a) and targets that are exact in C+D (§8c).

---

## 7. Open items

1. Make §4 rigorous for full unitaries: A(ε) plus norm-equation density.
2. Check the per-level constant c for C+R using Gustafson's syllable/sde statistics. Is it about 1 R per level? This would test the 1.25 observation.
3. **[optional experiment]** Count distinct exact unitaries with sde ≤ k for each gate set and confirm both grow like 3^{k·(same exponent)}. This directly tests "same expressiveness per level".
4. Regenerate the June 4 one-pager with the 5.12 headline so the .md and .tex agree.
5. Decide whether §3–4 belongs in the paper as an explanatory section. It would answer "why isn't C+D better?" before a referee asks.

---

## 8. Can C+D ever beat C+R? (added 2026-09-29)

**Punchline, stated carefully.**
- It is *not* "C+D reaches more of SU(3), but those points are far from R_z."
- The counting in §4 does not depend on the target: it holds for any point of SU(3).
- Per unit of cost (sde level ≈ gate), C+D's physical matrices cover SU(3) exactly as densely as C+R's, everywhere.
- C+D does have more exact unitaries at a fixed denominator 3^f. But reaching that denominator takes 3× more levels.
- The shadows are the reason: every candidate must also have two unitary shadows. The shadows take 2/3 of the norm budget and can't be chosen to help. They come out as random-looking unitaries (§3 demo).

**In the stated cost model C+D cannot beat C+R's slope.** That model is deterministic, ancilla-free, single-qutrit, with one unit of cost per non-Clifford gate. At best C+D can tie the constant; it is currently 1.24× behind Exhaustive. There are three ways to break the tie.

### 8a. Charge the gates by their fault-tolerant cost *(cheapest to check; potentially largest)*

- **ζ₉ phases:** diag(ζ₉^a, …) with 3∤a are qutrit-T-type gates. They sit at the 3rd level of the qutrit Clifford hierarchy and are injectable from magic states that standard qutrit distillation produces (Campbell–Anwar–Browne 2012, Reed–Muller codes).
- **R:** R = diag(1,1,−1) is **not** in the qutrit Clifford hierarchy at any level. Its phase −1 is not a 3^m-th root of unity. It needs special protocols: repeat-until-success (Cui–Wang, Bocharov–Roetteler–Svore 2016), or a multi-qutrit C+T construction.
- **The opposite case:** on *metaplectic* anyon hardware the economics reverse. R is native, and P9 needs "rather costly magic state distillation" (Bocharov 1606.02315 §4). So which gate set wins is **architecture-dependent**.
- **Architecture to settle:** for code-based qutrit FT, which is presumably the SQMS case, C+D would win per gate. We need to confirm which architecture applies.
- **Caveat 1: our C+D circuits still use R.**
  - Nick's decompositions put R^ε in 8–10% of syllables, about 7 R gates per circuit at f=14. The other ≈1.26 cost units per syllable are ζ₉ phases.
  - Either cost these R's or find R-free decompositions: restrict the syllable prefixes to ε=0 and check whether peeling still terminates.
- **Caveat 2: the ζ₉-phase count isn't T-count 1:1.** A single qutrit T carries two nontrivial ζ₉ phases. A proper conversion is needed.

### 8b. Non-deterministic synthesis: RUS / ancilla + measurement *(research-scale; does NOT break the tie)*

> **Correction (2026-09-29).** An earlier version of this section said RUS "escapes the shadow tax" and therefore favours C+D by up to 3×. **That was wrong.** In the qubit RUS construction (BRS 1404.5320) the shadow is still constrained: stage 2 requires both |rz|² ≤ 2^L and |(rz)•|² ≤ 2^L.

- **Where RUS's gain comes from.** It relaxes the *target region*, not the shadows.
  - Deterministic synthesis needs the entry close to the target in phase *and* magnitude, a tiny region of area ~ε^a.
  - RUS needs only the phase. The magnitude sets the success probability, so the region becomes a wedge of area ~ε.
  - The count in §4 becomes A_wedge(ε)·Norm instead of A_det(ε)·Norm. The Norm factor, and with it the shadow tax, is unchanged.
- **Qubit check.** Ross–Selinger's region has area ~ε³ (operator norm), and the RUS wedge has area ~ε. That predicts about a 3× drop in slope. Observed: 3·log₂ → 1.15·log₂, i.e. 2.6×.
- **Consequence: RUS helps C+R and C+D by the same factor to first order.** The region gain does not depend on the ring, and both rings pay the same count per unit of norm. So RUS lowers both curves and the tie survives.
- **Second-order differences, direction unclear.**
  - Modifier rings: C+R has only r ∈ Z, which is enough since there are no shadows to balance. C+D has r ∈ Z[ζ₉+ζ₉⁻¹].
  - Constants in the success probability.
  - Cost of the stage-4 controlled circuit. C+R may be *easier*: R-injection RUS primitives already exist (Cui–Wang, BRS 2016).
- **Blockers (both gate sets):** the qutrit 3-branch controlled circuit (§10, stage 4). C+D also needs a correctable failure mechanism for D.
- **Mixing:** mixing approximants squares ε (Campbell; Hastings). It also helps both gate sets equally.

### 8c. Pick targets that are exact in C+D

- Rotations by multiples of 2π/9 (and products of them) are exact in C+D at O(1) cost. C+R must approximate them at about 5·log₃(1/ε).
- **Example:** qutrit QFT or phase-estimation controlled phases e^{2πi/9}.
- **To do:** audit the target application circuits (gauge-theory primitives, arithmetic) for how much of their non-Clifford content is exactly ζ₉-representable.

**Suggested order:** 8a first (literature plus a recount), 8c second (an application audit), 8b as a gated research question.

---

## 9. The R gate inside C+D (literature check, 2026-09-29)

**It is the same gate.**
- Gustafson's R is `R = Diag(1,1,-1)`, called "the metaplectic R" (2503.20203 `paper_main.tex` l.180, l.135).
- Our `gate_R()` (`hrsa/canonical_reducer.py:684`) is the same matrix.
- Their only non-Clifford gate is R. The D(a,b,c) in their normal form uses powers of ω, which makes it Clifford. Their cost is the R-count (l.235).

**R is a member of C+D, by Kalra's definition.**
- Kalra et al. 2311.08696 (l.182–203): 𝒟 = all diag(±ξ^a, ±ξ^b, ±ξ^c) with ξ = ζ₉, and "R := R_[0,0,1]".
- Evra–Parzanchevski 2401.16120 use the same definition: C+D = {H} ∪ 𝒟.
- The syllable H·D·R^ε·X^δ is Kalra's own (main theorem). The R^ε is what normalizes (p,q,r) to ±(1,1,1).

**How often R shows up in our decompositions.**
- Nick's matrices, as decomposed by our reducer, use R in 8–10% of syllables. At f = 14 that is about 7 R per circuit, against about 96 ζ₉-phase units.
- So a C+D circuit is mostly ζ₉ phases plus a handful of R's. A C+R circuit is about 111 R's.

**Our code and the in-tree draft count R differently.**
- `count_d_from_syllable` charges +1 for each R^ε.
- The draft (`ESA_CliffordD_fixed.tex`, and `summary/..._Clifford_D.pdf`) defines C+D using unsigned ninth-root diagonals and never charges for R. Its eq. (37) treats a −1 as free.
- Our N_D is therefore the more conservative count, by about 7 at f = 14. **This has to be made consistent before publication.**

**What N_D actually counts (checked 2026-09-29, Nick f=14 decompositions, per circuit on average).**

| Syllable diagonal type | Count | Phases with 3∤a | Charged to N_D |
|---|---|---|---|
| Clifford (all exponents ≡ 0 mod 3) | 19.6 | 0 | 0 |
| Level-3 (a₀+a₁+a₂ ≡ 0 mod 3) | 38.8 | **always 2** | 77.6 |
| Level-4 | 16.3 | **always 1** | 16.3 |
| R^ε | 7.2 | — | 7.2 |
| **Total** | | | **≈101** (mean N_D = 102.7) |

**Each level-3 syllable diagonal is exactly one qutrit T (or T²) times a Clifford.**
- Reduced mod Clifford diagonals (exponents mod 3) and global phase, a level-3 diagonal is (0, a, −a), i.e. T^a.
- **N_D therefore charges each T as 2.**
- Nothing in these circuits needs further synthesis. Every piece is a single gate.

**Why this matters: the tie depends on the counting unit.**
- *Per phase* (current N_D): about 103 for C+D vs about 105 for C+R at ε ≈ 3.6×10⁻¹⁰. **A tie.**
- *Per non-Clifford operation* (one T or T², one level-4 diagonal, or one R, each counting 1): about 39 + 16 + 7 ≈ 62 vs about 105. **About 1.7× fewer for C+D.** Whether those operations cost the same is exactly the fault-tolerance question (§8a).
- The in-tree draft names D = diag(ζ₉, 1, ζ₉⁻¹), which *is* T, as "the" non-Clifford gate. Its own convention would therefore count T as 1, not 2. **The paper must choose a unit explicitly.**
- §4's counting argument sets the *number of sde levels*. The gates spent per level (c) depend on this unit. The ≈1.25 D per level is per phase; per operation it is ≈0.76.

**Draft error, l.606.** "Diag(−ζ₉⁶, −ζ₉³, 1) = Diag(ω², ω, 1) … is Clifford" is wrong. Since −ζ₉⁶ = −ω², that matrix is a Clifford times R, and it is not Clifford.

**Clifford hierarchy (facts only; cost analysis deferred).**
- R is in no level of the hierarchy (Glaudell et al. 2202.09235; BRS 1605.02756; Cui–Gottesman–Krishna 1608.06596).
- R cannot be built in single-qutrit ancilla-free C+T. With one borrowed ancilla it needs T-count 39 (Glaudell).
- Not every ζ₉ diagonal is at level 3.
  - Howard–Vala: the level-3 diagonals are the 27 with Συ_k ≡ 0 (mod 3), for example T = diag(1, ζ, ζ⁸).
  - A numeric check (agent's own, not from the literature) finds diag(ζ^a, ζ^b, ζ^c) is at level ≤3 exactly when a+b+c ≡ 0 (mod 3). The other 54 phase classes, such as diag(1,1,ζ), are at level 4.
  - **So "+1 per phase with 3∤a" is not the same as T-count.** Keep this in mind for the fault-tolerant comparison (§8a).
- **Can R be made from D? (checked 2026-09-29)**
  - **With an ancilla: yes.**
    - Deterministically: R exactly in C+T with one borrowed ancilla at T-count 39 (Glaudell 2202.09235). T is a D gate.
    - By RUS: an R-state costs 27/4 P9 on average, and about 3 R-states make one R gate, so ~20 D per R (BRS 1605.02756).
  - **Single qutrit, no ancilla: open.**
    - Not in C+T (Glaudell, proven).
    - For the group generated by H and unsigned ζ₉ diagonals, which includes level-4 gates:
      - An exact meet-in-the-middle search over every word of up to 6 syllables found **no** word equal to R up to global phase. The search was validated by finding X, S and H† (`nick_test/r_from_d_search.py`).
      - With H = F/(i√3), det H = 1, so every element of that group has det ∈ μ₉. That excludes −I and ζ^k R, but not −ζ^k R. So the determinant does not settle the question.
    - Evidence, not proof. A proof might adapt Glaudell's adjoint-representation argument.
  - **Consequence.**
    - On a D-native machine, R is available from D states at ~20–39 D per R. A C+R circuit of about 105 R therefore costs about 2,000–4,000 D-states.
    - On an R-native (metaplectic) machine the reverse holds.

---

## 10. What RUS synthesis for C+D would take (scoping, 2026-09-29)

**Qubit template.** The template is Bocharov–Roetteler–Svore, PRL 114, 080502 (arXiv:1404.5320); locally `qutrits/RUS_qubit.pdf`. The 2024 notebook `qutrits/RUS_derive.nb` already reproduces its stages 1–2 for qubits.

| Stage | Qubit version | C+D version | Status | Size |
|---|---|---|---|---|
| 1. Target search | Find z whose *direction* z*/z ≈ e^{iθ}, with no magnitude constraint, via PSLQ | **Existing HRSA/zeta9 front-end with a changed acceptance region.** Accept a thin *wedge* (angle within ε, magnitude free up to 3^L) instead of an ε-ball at the target point, and rank by (angle error, success probability \|σ₁(u)\|²) instead of Frobenius distance. PSLQ/LLL (`cvp/`) is an optional faster alternative | Modify existing | days |
| 2. Modifier + norm equation | r ∈ Z[√2], solve \|y\|² = 2^L − \|rz\|² | **Existing zeta9 stage 3** (the same relative norm equation that already completes rows), plus an outer loop over modifiers r ∈ Z[ζ₉+ζ₉⁻¹]. Risk: how often it is solvable as f grows | Modify existing | days–1 week |
| 3. Exact synthesis of V | Single-qubit KMM | **`canonical_reducer`, unchanged** | Reuse | none |
| 4. Controlled unitary with Clifford-correctable failure | U = diag(V, V†). V† is V with Paulis inserted, so it is cheap | Needs a cheap 3-branch multiplexor Σ\|k⟩⟨k\|⊗V_k. The qubit V→V† trick does not carry over: qutrit Clifford controls act linearly in k | **Open research problem, gating.** Generic 2-qutrit exact synthesis exists (Glaudell–Ross–vdWetering–Yeh 2405.08136) but its worst-case cost is useless | months, possible negative result |

**Answer to "new algorithm or just connect to exact synthesis?"**
- Stages 1–3 are what we already have: the same enumerator, norm-equation solver and exact synthesis.
- The changes are the acceptance region (wedge instead of ball), the ranking metric, and a modifier loop.
- *(An earlier draft of this section said stage 1 was a "new algorithm". That overstated it.)*
- The real problem is stage 4, a cheap controlled circuit. It is not a search problem.
- Stages 1–3 (about 1–2 weeks of modification) would measure the wedge-vs-ball density gain before anyone commits to stage 4. Run the same measurement for Z[ω] to confirm the gain is the same for C+R.

**Corrections to the May feasibility memo.**
- arXiv:1409.3552 is Bocharov–Roetteler–Svore (PQF / "fallback"), not Martonosi.
- Its m ≡ 0 (mod 4) restriction belongs to the PQF construction. The RUS construction only needs complex conjugation, which Z[ζ₉] has. So "blocker 1" reduces to the stage-4 multiplexor question.

**Closest recent template:** Kliuchnikov–Brachter–da Silva, arXiv:2604.20033 (Apr 2026). It does one-ancilla RUS for all of U(2), using lattice methods and relative norm equations. It is qubit-only.

**Mixing is cheap but doesn't break the tie.**
- Mixing (Campbell 1612.02689; Hastings 1612.01011) needs no ancilla and can be built from existing code in about a week.
- It combines about 9 deterministic approximants (3 if all are diagonal) whose su(3) error generators surround 0. That gives O(ε²) diamond-norm error, roughly halving the log coefficient.
- It is gate-set independent, so C+D and C+R gain equally. Compare the two in diamond norm.

---

## 11. Fault-tolerance consequences of R being outside the Clifford hierarchy (literature, 2026-09-29)

Legend: (L) the literature states it; (I) our inference; (?) not verified.

**Membership.**
- Diagonal hierarchy gates have phases that are 3^m-th roots of unity (Cui–Gottesman–Krishna 1608.06596). (L)
- Hence R, and every *mixed-sign* 𝒟 element (a hierarchy diagonal times R), is outside every level. (I)
- Kalra's statement that "a D gate is in the Clifford hierarchy" (2311.08696 l.205) holds only for unsigned D. (I)

**QEC: transversal gates.**
- Transversal gates on any error-detecting qudit stabilizer code must lie in the Clifford hierarchy (Jochym-O'Connor–Kubica–Yoder 1710.07256, Cor. 6; stated for m-dimensional qudits). (L)
- **So R can never be transversal on a qutrit stabilizer code.** (I)
- The qutrit T₃ = diag(1, ζ₉, ζ₉⁸) *is* transversal on qutrit 3D color codes (Watson et al. 1503.08800). (L)
- Level-4 transversality for qutrits: (?).

**Gate injection.**

| Gate | Injection | Cost per gate | Source |
|---|---|---|---|
| T (level 3) | Deterministic, Clifford correction, constant depth | 1 T-state | Campbell–Anwar–Browne 1205.3104 (L) |
| Level-4 diagonal | Deterministic in ≤2 rounds; level-3 correction with prob. 2/3 | 1 level-4 state + 2/3 T-state (expected) | (I); no paper states the 2/3 |
| R | Random walk / RUS; variable latency | ~3 R-states (expected) | BRS 1605.02756 l.427; Anwar–Campbell–Browne 1202.2326 (L) |

**Magic-state distillation.**
- **T-states:**
  - Qutrit Reed–Muller QRM₃(2) is 8→1 with quadratic suppression and depolarizing threshold 0.211 (Campbell–Anwar–Browne). (L)
  - Prakash–Saha 2403.06228: [[20,7,2]]₃ with yield parameter 1.51. (L)
  - Krishna–Tillich's γ → 0 needs large primes, so it does not apply to qutrits. (L)
- **R-states: there is no direct protocol.** Three routes:
  - [[5,1,3]]₃ at *linear* suppression, then a probabilistic equatorialization step (Anwar–Campbell–Browne 1202.2326). (L)
  - Golay-code strange states (cubic suppression, threshold 0.387), then probabilistic conversions S,S → N (p = 1/2) and N,N → R-state (p = 1/4). That is about 32 strange states per R-state (2003.02717 App. A). (L/I)
  - Build it from T: an R-state via RUS needs 4 ancillas and an expected P9-count of 27/4 (BRS l.1254). With about 3 R-states per R gate, that is **~20 T per R** (I). The alternative is deterministic: R costs **39 T with one borrowed ancilla** (Glaudell 2202.09235). (L)
- **No head-to-head comparison of R-state vs T-state distillation exists.** (?)

**Metaplectic (anyon) hardware reverses all of this.**
- There, R-states are prepared exactly and probabilistically, "9/4 trials on average … much better than any state distillation method" (BRS l.465). (L)
- On that hardware, P9/T is the costly gate (Bocharov 1606.02315 §4). (L)

**Other stack issues.**
- RUS gives variable online latency and needs adaptive scheduling. BRS l.1256 calls for a magic-state coprocessor "of width somewhat greater than 27". (L)
- No qutrit-specific scheduling study was found. (?)

**Implications for C+D vs C+R.**

> **Correction (2026-09-29).** An earlier version of this section priced R in "T-equivalents" (~20–39 T). That is the route of *building R from T*. The intended architecture distills each gate's own state directly, so T-equivalents are the wrong currency. The right per-gate cost is:
>
> **cost(gate) = (raw states per distilled state at the target fidelity, for that gate's own protocol) × (states consumed per injection).**

| Gate | Direct distillation of its own state | States per injection |
|---|---|---|
| T-type D (level 3). Every level-3 syllable diagonal is T or T⁻¹ up to Clifford, so "distill D directly" here *is* distilling T-states | Yes. Transversal-gate codes: qutrit RM, Prakash–Saha | 1, deterministic |
| Level-4 D | **Unknown.** Needs a code with a transversal level-4 gate (allowed by JOKY) or another route (?) | 1 level-4 state + ⅔ T-state, deterministic ≤2 rounds (I) |
| R | Only through Clifford-eigenstate distillation plus probabilistic conversion: Golay strange state (cubic) → ~32 per R-state, or [[5,1,3]]₃ (linear) → equatorialization | **~3 R-states, random walk. Intrinsic: better distillation cannot remove it** |

**Structural reason there is no efficient direct R-state protocol (I).**
- Standard high-order distillation uses a code on which the target gate is transversal.
- JOKY Cor. 6 forbids a transversal R on any qutrit stabilizer code.
- So R-state distillation must go through Clifford-eigenstate states, which is exactly why the known routes do that.
- **Level-3 D does not have this obstruction. Level-4 D is not excluded by it.**

**Which counts are needed.**
- N_D should be counted *per injected gate* (T-type, level-4, R separately), not per phase.
- At f = 14: C+D uses 39 T-type + 16 level-4 + 7 R; C+R uses about 105 R.
- **Even with direct R distillation, C+R's 105 gates each carry the ~3× injection overhead and variable latency.**
- **Top action item:** find out whether the R's can be removed from C+D exact synthesis. Kalra's syllable uses R^ε only to normalize (p,q,r), and the draft's unsigned definition of C+D would forbid R altogether. Also check whether level-4 states have any distillation route.
- **Bug in the quai catalog:** `quai/hank/swiftbot/tools/distillation.py` tags Krishna–Tillich as usable for qutrits, and tags Anwar–Campbell–Browne as distilling "Howard-Vala-class T-magic state". It actually distills H/φ states that lead to the R-state. Not yet corrected.

---

## 12. Costing framework (agreed with PI, 2026-09-29)

**Compare a C+R device running C+R synthesis against a C+D device running C+D synthesis.**

Total cost = Σ over gate types of (count per synthesized gate) × (device cost per gate).

| | C+R device | C+D device |
|---|---|---|
| Counts at ε ≈ 3.6×10⁻¹⁰ | ~105 R | ~39 T-type + ~16 level-4 + ~7 R (Nick, f = 14) |
| Per-gate costs needed | c_R (own-state distillation + ~3× RUS injection) | c_T (own-state, deterministic); c_L4 (?); c_R on a D device (~20–39 D via ancilla, or native if supported) |

**Choices the comparison depends on.**
1. **What a "C+D device" provides.**
   - Kalra/EP C+D includes signed diagonals, so it includes R, and the device must support R.
   - The draft's unsigned C+D does not include R. Then our synthesis must be R-free, or the ~7 R's are paid at the ancilla price.
   - This decides whether the ~7 R's are cheap or expensive. The no-R test on `canonical_reducer` settles which case holds.
2. **What substrate the "C+R device" is.**
   - **Same substrate (primary comparison):** both are qutrit stabilizer-code machines, each distilling its own magic states. Then R has no transversal code (JOKY) and needs indirect distillation plus RUS injection, while T-type D has standard protocols. That favours the C+D device.
   - **Metaplectic anyons (secondary comparison):** R is native (probabilistic, about 9/4 trials per R-state) and P9/T is the costly gate. This is a cross-platform comparison and needs physical cost units.
3. **The level-4 cost c_L4** is unknown; no qutrit level-4 distillation protocol was found. It matters for about 16 gates per circuit.

**Update, 2026-09-29 (PI).** Metaplectic/anyon hardware is dropped as too far out. **Both devices are generic qutrit stabilizer-code machines.** The C+R device makes R through an R-state factory plus RUS injection.

**R-free synthesis test** (`nick_test/no_r_test.py`, f = 10, 10 matrices; repo code unchanged). With the reducer's prefix table restricted to ε = 0 (no R), **0/10 decompose**. Peeling stalls immediately, mostly at the first step (sde 58), for both single and double prefixes. With R allowed, 10/10 succeed, with 54 R syllables across the 10 circuits. All of Nick's matrices have det = 1 exactly, so the determinant is not what forces R.

- **Likely reason [heuristic].** Modulo χ = 1−ζ₉ (residue field F₃), ζ₉ ≡ 1, so every unsigned D reduces to the identity, while R reduces to diag(1,1,−1).
- Unsigned gates therefore cannot change the *signs* in the leading residue pattern (p,q,r); only R can. This is Kalra's R^ε normalization.
- This supports "R ∉ unsigned C+D" (together with the exhaustive search up to 6 syllables). It suggests a membership invariant, but it is not proven.

**Ways a C+D device can get its ~7 R per circuit.**
1. **Build the same R-state factory the C+R device uses.** On a common substrate nothing stops this. Then:
   **C+D cost = 39 c_T + 16 c_L4 + 7 c_R, against C+R cost = 105 c_R.**
   C+D wins unless 39 c_T + 16 c_L4 > 98 c_R. Since c_R carries a ~3× RUS injection overhead on top of an indirect distillation chain, this looks robust. **The one real unknown is c_L4.**
2. **Make R from D with an ancilla.** *(PI's preferred option.)*

   **New, verified 2026-09-29:** R costs **9 level-4 D gates + Cliffords, with one clean |0⟩ ancilla. It is deterministic, exact and unitary.**
   - Construction:
     - GRVY's |2⟩-controlled(−τ₁₂) (2202.09235) applied to an ancilla in |0⟩ gives R ⊗ |0⟩, because τ₁₂|0⟩ = |0⟩.
     - Its three 8-T blocks, |2⟩-ctrl(ζ⁷Z(1,1)), are each re-expressed as 3 single-qutrit D phase gadgets joined by SUM gates: diag(1,ζ,ζ), diag(1,1,ζ²), diag(1,1,ζ²).
     - The leftover phase is ω^{quadratic}, which is Clifford.
   - Found by the literature agent through exhaustive gadget search. Independently re-verified: exact R⊗|0⟩ on random inputs, and the residual checked to be Clifford. Scripts: `unified/r_from_d/` (`e2e.py`, `mincount.py`).
   - **All 9 are level-4.** No T-type-only diagonal solution exists in this search space.
   - Other options:
     - The same construction in T gates is **24 T** with a clean ancilla. The published 39 T needs only a *borrowed* ancilla.
     - 15 T with a reusable Strange-state catalyst.
     - RUS routes (BRS) cost ≈27 P9 per R. The agent argues BRS's stated 27/4 per R-state is an arithmetic slip and should be 9; I have not checked this.
   - **Lower bound (mana):** at least 1 T-state per R. The gap between 1 and 9–24 is open. Nobody has published an improvement on 39, or any lower bound.
   - **Consequence.** About 7 R per rotation become 63 level-4 D. A fully R-free C+D device then pays **39 c_T + (16 + 63) c_L4 ≈ 39 c_T + 79 c_L4** per rotation at ε ≈ 10⁻¹⁰. The C+R device pays 105 c_R.
   - **Sanity bound.** A C+D device could also emulate C+R synthesis at 9 D per R (≈945 level-4). Native C+D synthesis is far cheaper than that.
3. **Synthesize approximants that need fewer R.** Rank candidates by R-count, or restrict to the mod-χ sign class that needs none. This is an algorithmic lever on the approximation stage, at an unknown cost in candidate density.

**Level-4 cost c_L4: literature review (2026-09-29; scripts in `unified/level4/`).**
- **No qutrit level-4 distillation protocol exists.** All qudit protocols target level 3: Campbell–Anwar–Browne, Krishna–Tillich, Prakash–Saha, 2510.10852.
- **Transversal codes are ruled out by an existing theorem's hypothesis.** Watson et al. 1503.08800 Thm 1 gives transversal level-µ₀ gates on qudit color codes only when µ₀! ≢ 0 (mod d). For d = 3, µ₀ = 4 this fails, so qutrit level 4 is *explicitly excluded*. The authors "believe the qualification can be removed… future work."
- **No code with a transversal diag(1,1,ζ₉)-type gate is known.** No qudit cultivation or catalysis paper exists either; the recent √T cultivation and catalyst work is qubit-only.
- **What does work: exact multi-qutrit Clifford+T synthesis.**
  - Any unitary over Z[1/3, ζ₉] is exactly Clifford+T given ≤2 ancillas (2405.08136; Kalra et al. 2405.08147).
  - An ancilla-free |2⟩-controlled-ζ construction (Yeh–vdW 2204.00552, Cor. 4) gives diag(1,1,ζ) exactly.
  - **No optimized T-count is published.** The agent's rough tally is ~100–200 T (unverified). A ZX-optimized circuit at ~20–40 T is "plausible".
- **A no-go (agent's own, not independently checked):** CX, X and diagonal Clifford/T gadgets with basis-state ancillas cannot produce diag(1,1,ζ). Non-diagonal pieces are needed.
- **Catalytic identity (agent's own):** diag(1,1,ζ) = C₂(X·diag(1,1,ω)) acting on a reusable level-3 catalyst. But C₂(Λ) is itself level 4, so the catalyst does not lower the level.
- **Qubit calibration:** √T/T ≈ 2–3× with dedicated QRM distillation (Landahl–Cesare 31-to-1 vs 15-to-1). Catalyst routes run ~10–30×.

**Consequences.**
- ~~c_L4 ≈ 100–200 T~~ superseded: c_L4 ≤ 8 T (see RESULT below).
- **The 9-level-4 R construction is then ~900–1800 T, much worse than the 24-T clean-ancilla construction.** For R on a C+D device, use **24 T + 1 clean ancilla** unless c_L4 < 24/9 ≈ 2.7 c_T.
- **The C+D device is effectively a C+T device plus level-4 synthesis.** Per rotation: 39 T-type + 16 level-4 (at ~c_L4 each) + 7 R (at 24 T each) ≈ 39 + 16·c_L4 + 168 T-equivalents. The C+R device pays 105 c_R.
- **Next checks:**
  1. ~~Can synthesis avoid level-4 syllables?~~ **No** (`nick_test/no_l4_test.py`, f = 10, 10 matrices). With the prefix table restricted to level-3 or Clifford diagonals, 0/10 decompose, whether or not R is allowed; peeling stalls at the first step. The baseline gets 10/10. So Nick's approximants need level-4 gates as well as R. Heuristic, not proven: they lie outside single-qutrit C+T. Avoiding level-4 would need a *different approximation stage* that only produces single-qutrit C+T elements; its density cost is unknown.
  2. Find the true minimal T-count of diag(1,1,ζ₉) with ancilla (ZX-style search).

**RESULT (2026-09-29, verified): c_L4 ≤ 8 T with one clean ancilla** (`unified/level4/l4_from_8T.py`). This supersedes the "100–200 T" estimate above.
- **Construction.**
  - GRVY's 8-T block |2⟩-ctrl(ζ⁷ Z(1,1)) (2202.09235, Cor. after Lemma "tcspdagphase") is rebuilt from their tikz figures in our script.
  - With the target ancilla in |0⟩, Z(1,1)|0⟩ = |0⟩, so the controlled global phase ζ⁷ kicks back.
  - The result is **diag(1,1,ζ⁷) ⊗ |0⟩ = [diag(1,1,ζ)·diag(1,1,ω²)] ⊗ |0⟩**: exact, deterministic, unitary, and the ancilla comes back clean.
  - Numerically verified. The script first reproduces the 3-T |2⟩-ctrl X and the 8-T |2⟩-ctrl ζS†, then checks the end-to-end action on random states.
- **It covers every level-4 syllable diagonal.**
  - Modulo Clifford diagonals and global phase, all six level-4 classes are single-position phases: (0,0,1), (0,0,2), (0,1,0), (0,2,0), (0,1,1) ~ (2,0,0), (0,2,2) ~ (1,0,0).
  - Each is one 8-T gadget, controlled on |0⟩/|1⟩/|2⟩ via X-conjugation, or its adjoint.
- **Lower bound:** the mana of diag(1,1,ζ)|+⟩ is 0.33 against 0.46 for a T-state, so ≥1 T. The gap between 1 and 8 is open.
- **Why the earlier estimates missed it.** The level-4 agent looked at *ancilla-free* constructions. The R agent used 3 *level-4* gadgets per block, rather than noticing that the block itself is 8 T.

**Resulting per-rotation cost on a T-factory device** (C+D synthesis; T-type = 1 T, level-4 = 8 T, R = 24 T, clean ancilla reused):

| f (ε) | T-type | level-4 × 8 | R × 24 | **T total** | N_D (per phase) |
|---|---|---|---|---|---|
| 10 (2.2e-7) | 28.4 | 11.4 | 4.4 | **225** | 74.4 |
| 12 (6.9e-9) | 34.0 | 14.2 | 6.3 | **298** | 90.2 |
| 14 (3.6e-10) | 38.8 | 16.3 | 7.2 | **341** | 102.7 |

- **The C+D device is in effect a qutrit Clifford+T device with one clean ancilla.** It needs only T-state factories. R (~51% of the cost) and level-4 gates (~38%) dominate.
- **Break-even against a C+R device** at ε ≈ 3.6e-10: C+D wins iff c_R > 341/105 ≈ **3.2 c_T**. Each R consumes ~3 R-states in RUS injection, so this means an R-state costing more than ~1.1 T-states. R-states have no direct high-order distillation (Golay strange-state route: ~32 strange states per R-state), so this looks very likely. **Not yet a cost model:** it needs actual distillation factory costs.
- **For reference:** a C+R device that made R from T at 24 T each would pay ~2,500 T, about 7× C+D.
- **Cheapest improvements:**
  1. R-count reduction through candidate selection (option 3, running).
  2. A search for R below 24 T.
  3. A search for level-4 below 8 T.

**Option 3: choose approximants by T-cost** (`unified/r_count_study/`, 2026-09-30; open `candidates_all.csv` first).
- **Data.** HRSA_bestD `--max-solns 300` at θ ∈ {0.3, 0.5, 1.0, 1.7, 2.5}. 900 distinct exact candidates were decomposed: 500 at ε = 1e-2 (f = 3) and 400 at ε = 0.1 (f = 2). ε = 1e-3 did not finish in the time budget.
- **Bug found (confirmed in code).** `classify_monomial_and_d_cost` hard-codes the residual R count to 0 and treats a trailing monomial with *mixed entry signs* as free.
  - A qutrit Clifford monomial has uniform signs, so mixed signs cost one R.
  - This affects **76% of candidates**. **N_D and all R counts quoted before 2026-09-30 undercount R by up to 1 per rotation.** A recount of Nick's circuits is running.
- **Global phase** (V, −V, ζV, …) leaves the R count unchanged.
- **R per rotation (syllables + residual).**
  - Mean 3.8 at ε = 1e-2 and 2.8 at ε = 0.1.
  - R = 0 is reached by 5.6% and 11% of candidates respectively, at 4 of 5 angles.
  - R-light candidates are *longer* in N_D (Spearman −0.2 to −0.47). **So HRSA_bestD's min-N_D pick is systematically R-heavy.**
- **T-cost** (1·T3 + 8·L4 + 24·R):
  - Within 10% of the best-frob candidate, the minimum T-cost is **36–61% lower** than the frob-best candidate's (median ~48%, ε = 1e-2).
  - Over all ε-passing candidates the minimum is 2–6× below the min-N_D pick. Example at θ = 1.0, ε = 1e-2: min-N_D pick 229 T, min T-cost 53 T.
- **Cheap predictor (empirical, 900/900).** Take the leading χ-adic digits of the column-0 numerators, up to a global sign.
  - Class **`111`** means the first syllable needs no R and there is no residual R.
  - Any mixed class costs ≥2 "bookend" R.
  - Every R = 0 candidate is in class `111`. Within `111`, 33% have R = 0.
  - Necessary, not sufficient. Not a proven invariant.
- **Action:** rank by T-cost inside HRSA_bestD (and in Nick's/zeta9 final selection) instead of by N_D. Pre-screen with the `111` class.

**RESULT (2026-09-30, independently verified): R costs 7 T and level-4 costs 7 T, each with 1 clean ancilla** (`unified/r_from_d/verify_R7T.py`, `R_7T*.json`, `D4_7T*.json`).
- **Method.** Exact meet-in-the-middle over products of Clifford-conjugated T rotations R_P (80 two-qutrit Paulis), with isometries compared up to a left Clifford and a global phase. The comparison uses a Pauli-multiset invariant computed in exact Z[ζ] coordinates.
- **R = diag(1,1,−1): 7 T** (3 T + 4 T†, all on the system qutrit, 63 gates). Output U(ψ⊗|0⟩) = e^{−2πi/9}(Rψ)⊗|0⟩ exactly, with no correction needed.
  - **Optimal with 1 ancilla.** A merge-free exhaustive search of all rotation products of length ≤3 from both ends (`raw_lb_1anc.py`) finds no circuit with ≤6 T.
  - With 2 ancillas: no circuit ≤6 T, under the invariant-merged search (caveat: merging could in principle hide solutions).
- **diag(1,1,ζ): 7 T** (60 gates). No meet at ≤6 with 1 ancilla (merged search, same caveat). A merge-free check has been requested.
- **Independent verification.** I re-implemented the gate set (S = diag(1,1,ω), Z = diag(1,ω,ω²), H ∝ F) and parsed the published gate lists. Both circuits reproduce the targets to 1e-15, with the ancilla returned to |0⟩.
- **Previous best:** R 39 T with a borrowed ancilla (GRVY) / 24 T with a clean ancilla; level-4 8 T. **The R result appears to be new.**
- **Updated gadget costs:** T-type 1 T, **level-4 7 T, R 7 T**. From Nick's syllable counts (residual R still missing, recount running):
  - f = 14: 38.8 + 7·16.3 + 7·7.2 ≈ **203 T** per rotation (was 341).
  - C+D wins against a C+R device iff c_R > 203/105 ≈ **1.9 c_T**, i.e. an R-state costs more than ~0.65 T-states given ~3 R-states per R.
  - A C+R device making R from T would now pay 105 × 7 = **735 T**, about 3.6× C+D. *C+R on a T-factory machine is now a legitimate competitor and must be kept in the comparison.*

**Consequence for N_D.** Report counts by gate type (T-type / level-4 / R), not the per-phase N_D. The "C+D ≈ C+R" tie holds only in per-phase units.

## 13. Sources

- Gustafson et al., arXiv:2503.20203, `paper_main.tex`:
  - fits at lines 624–633;
  - covering bound at lines 150–169;
  - two-qubit overhead at line 646.
- Kalra et al., Theorem 5.7 (syllable obstruction).
- Ross–Selinger (qubit grid problem, •-conjugate).
- `unified/cd_vs_cr_one_pager_2026-06-04.md`.
- Memory notes (in `~/.claude/projects/.../memory/`):
  - `cd_vs_cr_corrected_2026-05-29`
  - `path_c_research_dead_2026-05-25`
  - `bocharov_z9_port_dead_2026-05-24`
  - `cliffd_rus_feasibility_2026-05-30`
  - `cd_state_2026-09-29`
