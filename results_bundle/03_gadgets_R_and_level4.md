# 03: The 7-T gadgets for R and level-4

All circuits are exact, deterministic and unitary over qutrit Clifford+T (T = diag(1, ζ₉, ζ₉⁸)). They use **one clean ancilla** in |0⟩, which is returned to |0⟩.

## Results

| Gate | T-count | Previous best | Optimality | Verify |
|---|---|---|---|---|
| **R = diag(1,1,−1)** | **7** (3 T + 4 T†, 63 gates) | 39 T with a *borrowed* ancilla (Glaudell–Ross–van de Wetering–Yeh 2202.09235); 24 T with a clean ancilla (implicit in the same paper) | **Optimal with ≤2 clean ancillas** | `r_from_d/verify_R7T.py` |
| **Level-4 diag(1,1,ζ₉)** (all level-4 classes are this up to Clifford) | **7** | 224 T with 2 borrowed qutrits (our count of Yeh–van de Wetering 2204.00552 Cor. 4); no T-count published | **Optimal with ≤2 clean ancillas**; impossible with 0 ancillas (determinant) | `level4/l4_7T_verify.py` |

## The circuits
- **Level-4: compute / phase / uncompute.**

  `C2X ; T(ancilla) ; C2X†`, where `C2X = H1 X0 X1 [Xdg1 T1 X1 SUM01]×3 Xdg0 Xdg1 Hdg1`

  - C2X is GRVY's 3-T |2⟩-controlled X. It writes [x = 2] into the ancilla.
  - The T on the ancilla applies ζ^{[x=2]}.
  - C2X† then uncomputes the ancilla.
  - Result: exactly diag(1,1,ζ) ⊗ |0⟩.
  - Other level-4 classes: conjugate the system by X^{2−j} to put the phase on position j, and use T† for ζ⁻¹.
- **R.** The 63-gate word is in `r_from_d/R_7T_compact.json`. It was found by exact meet-in-the-middle and gives R⊗|0⟩ up to the global phase e^{−2πi/9}. No correction is needed.

## How optimality was established
- **Search space.** Every t-T circuit is K·f(P_t)⋯f(P_1), with f(P) a Clifford-conjugated T rotation: 80 Paulis on 2 qutrits, 728 on 3. The search is an exact orbit meet-in-the-middle on the isometries (target ⊗ |0…0⟩), compared up to a left Clifford, a right single-qutrit Clifford and a global phase.
- **One ancilla.**
  - R: merge-free exhaustive search of all rotation products up to length 3 from each end. No circuit has ≤6 T (`r_from_d/raw_lb_1anc.py`).
  - Level-4: exact orbit MITM with no false merges (`level4/l4_orbits.py`).
- **Two ancillas.**
  - The search: `two_ancilla/orbits3.py`, 29 min on 10 processes.
  - It uses an exact 3-qutrit Clifford-equivalence test (`two_ancilla/c3core.py`): backtracking over Sp(6,3) plus a Pauli correction and a matrix check. Controls: 30/30 positive, 62/62 negative (`test_equiv.py`).
  - Result: no ≤6-T circuit for either gate. The two invariant collisions at t = 6 are refuted by `certify.py`.
- **Three or more ancillas are not searched.** That would mean 6,560 rotations per step, about 9× more per depth, so it is infeasible without new symmetry reduction.
- **Measurement- or catalyst-based constructions were not searched.** A 15-T R with a reusable Strange-state catalyst exists, which is worse anyway.
- **Lower bound from mana:** ≥1 T. That is weak, but it is the only analytic bound.

## Novelty (`paper_prep/novelty_check.md`)
- **R in 7 T:** ~85% confident it is new; the optimality claim is new.
- **Level-4 in 7 T:** ~70% confident. The compute/phase/uncompute kickback is standard, so the circuit is arguably folklore. The explicit count, the use of a single clean ancilla and the optimality are new.
- **Suggested wording:** "To our knowledge, the first explicit T-counts; optimal among circuits with at most two clean ancillas (exhaustive search); the general-ancilla case is open."

## Why these matter
- **R is in no level of the Clifford hierarchy** and cannot be transversal on any qutrit stabilizer code (Jochym-O'Connor–Kubica–Yoder 1710.07256). It has no efficient direct factory (`04`).
- **No level-4 qutrit distillation protocol or transversal code is known.** Watson et al. 1503.08800 Thm 1 explicitly excludes d = 3, µ₀ = 4.
- **So these gadgets make a C+D device a pure T-factory machine.** The only non-Clifford resource is level-3 T-states, plus one reusable ancilla.

## UPDATE 2026-10-01: measurement model, level-4 = 4 T
- **Construction:** compute b = [x=2] into a clean ancilla (3-T |2⟩-controlled X), apply T to the ancilla, then **measure the ancilla in the Fourier basis**. Each outcome m (probability ⅓) is fixed by the Clifford diag(1,1,ω^m).
- **Cost:** deterministic with Clifford feed-forward, **T-count 4** instead of 7. It is the qutrit analogue of Gidney's measurement-based uncomputation.
- **Verified:** 200 random states × 3 outcomes, residual ~1e-15 (`level4/l4_meas_uncompute.py`).
- **Pending:** searches for R and for level-4 below 4 in the measurement / catalyst model (`measurement_tricks/`), and for ≤6 T with ≥3 ancillas (`three_ancilla/`).

## UPDATE 2026-10-01: optimal with ANY number of clean ancillas (unitary model)
`three_ancilla/verify_lemmas.py` (all checks pass; re-run with threads capped).
- **Lemma 1.** A T-rotation either stays on the active register, or (after a Clifford) attaches one fresh ancilla in a fixed non-stabilizer magic state m (mana 0.4614). So a depth-a half uses ≤ a ancillas.
- **Lemma 2.** Idle |0⟩ ancillas cancel from a Clifford equivalence.
- **Theorem (t ≤ 6).** Split the circuit 3 + 3.
  - One half using 3 fresh ancillas and the other not: impossible. The first has no reference-trivial Pauli stabilizer; the second has one from an idle |0⟩.
  - Both halves using 3 fresh ancillas: needs mana(Choi(target)) = 0. But mana(Choi R) = 0.368 and mana(Choi L4) = 0.330.
  - Every other case uses ≤2 ancillas, which the exhaustive search excludes.
- **Conclusion: R and level-4 need exactly 7 T for unitary circuits with any number of clean ancillas.**
- Lemma 2 and the theorem are proved on paper and spot-checked numerically. The result inherits the correctness of the 2-ancilla search.
- **Not covered:** measurement/feed-forward and catalysts. With measurement, level-4 is already 4 T (above).

## UPDATE 2026-10-01 (later): measurement model, **R = 4 T** too
- **Construction.** A 22-gate circuit, 4 T, one clean ancilla, ancilla measured in the **computational** basis. The corrections diag(1,1,ω²) (m = 0, 1) and diag(1,ω²,1) (m = 2) are Clifford, so the gadget is deterministic.
- **Verified** by an exact ring check (agent) and an independent numeric re-check (300 states × 3 outcomes, residual 2e-15). Files: `measurement_tricks/r_meas_4T.json`, `verify_meas.py`.
- **Nothing at ≤3 T, R or L, in the searched measurement classes:**
  - 1 or 2 ancillas, unitary prefix + final stabilizer-basis measurement (`search2.log`);
  - adaptive single mid-measurement (`adaptive.log`).
- **Lower bounds** for these gadgets: mana gives ≥1 T; thauma gives ≥1 for R and ≥1 for L.

| Model | T-type | Level-4 | R |
|---|---|---|---|
| Unitary, any number of clean ancillas | 1 | 7 (optimal) | 7 (optimal) |
| **Measurement + Clifford feed-forward, 1 ancilla** | 1 | **4** | **4** |

## Final measurement/catalyst search results (2026-10-01; `measurement_tricks/`)
- **4 T is optimal for both R and level-4** in these classes (exhaustive):
  - 1 ancilla, t ≤ 6, unitary prefix + final stabilizer-basis measurement + one common Clifford (`search1.py`). The first hits are at t = 4: 246 for R, 219 for L.
  - 2 ancillas, t ≤ 3 (`search2.py`).
  - One mid-circuit measurement with outcome-dependent continuation, 2 ancillas, total t ≤ 3 (`search_adaptive.py`). Caveat: this search's own end-to-end positive control did not run within the time budget.
- **No outcome at t ≤ 3 ever produces the target**, so single-round repeat-until-success cannot beat 4 either.
- **Catalysts (searched t ≤ 1 only):** no hits for R or L with Strange, T|+⟩, L|+⟩ or R|+⟩ catalysts. Kickback-catalyst routes are argued to cost ≥4 T per use; no single-qutrit Clifford has eigenvalue ratio ζ.
- **Analytic lower bounds:** mana R ≥ 0.80, L ≥ 0.72; max-thauma (SDP) R ≥ 0.97, L ≥ 0.86. Neither beats t ≥ 1.
- **Not covered:** ≥3 ancillas with measurement, and multi-round adaptive trees.
- **Mechanism of the R gadget:** |x⟩|0⟩ → (−1)^{[x=2]}|x⟩ ⊗ D_x|+⟩ with D_x an ω-phase ancilla diagonal. Measuring the ancilla turns D_x into an ω^{f(x)} phase on the system, which is always Clifford for one qutrit. The −1 comes from interference, not kickback.
