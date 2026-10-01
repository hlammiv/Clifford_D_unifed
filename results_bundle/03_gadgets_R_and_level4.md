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
