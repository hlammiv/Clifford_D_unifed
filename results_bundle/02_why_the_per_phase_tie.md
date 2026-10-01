# 02: Why C+D ties C+R in gate count

The surprise: C+R ⊊ C+T ⊆ C+D, so C+D can reach strictly more unitaries exactly. Yet its per-phase gate count for approximating R_z(θ) matches C+R. The reason is the field's other embeddings.

## Galois "shadows"
- **Each entry has three readings.** An exact C+D unitary U has entries in Z[ζ₉, 1/3], each a list of 6 integers over 3^f. Reading them with ζ = e^{2πi/9} gives the physical gate. Reading them with ζ = e^{4πi/9} or e^{8πi/9} gives two "shadow" matrices, σ₂(U) and σ₄(U).
- **The shadows are unitary too.** Galois automorphisms commute with complex conjugation, so σ(U)σ(U)† = σ(UU†) = I.
- **As circuits:**
  - The σ₄ shadow keeps every Clifford and maps each D-phase ζ^a → ζ^{4a}.
  - The σ₂ shadow complex-conjugates the Cliffords and maps ζ^a → ζ^{2a}.
- **Example** (`nick_test/galois_shadows_demo.py`, Nick f=10, θ≈0):

  | Embedding | Distance to R_z | Unitary? | First row |
  |---|---|---|---|
  | Physical | 2.2×10⁻⁷ | yes | ≈ (1, 0, 0) |
  | σ₂ shadow | ≈ 2.5 | yes, exactly | (0.49, 0.57, 0.66) |
  | σ₄ shadow | ≈ 2.2 | yes, exactly | (0.36, 0.39, 0.85) |

  Both shadows are generic unitaries.
- **C+R has no shadows.** Z[ω] has only one embedding up to conjugation. Qubit C+T has one, which is Ross–Selinger's "•" condition.

## Counting argument [heuristic]
- **C+D.** Approximate one entry u = x/δ. The physical value must land in a target region of area A(ε), and both shadow values must have modulus ≤ 1. Z[ζ₉] is a lattice in C³ ≅ R⁶, so the admissible volume is

  A(ε)·|σ₁(δ)|²·π|σ₂(δ)|²·π|σ₄(δ)|² ∝ A(ε)·N(δ) = A(ε)·3^k   (k = sde_χ).

- **C+R.** The same count in Z[ω] gives A(ε)·|δ|² = A(ε)·3^k.
- **Same sde needed.** Both rings need k ≈ log₃(1/A(ε)).
- **Units can't help.** They redistribute magnitude between the physical value and the shadows, but the volume depends only on the norm.
- **Reconciling with "C+D is more expressive":**
  - At a fixed denominator 3^f, Z[ζ₉] has ~3^{6f} candidates against 3^{2f} for C+R.
  - But reaching 3^f costs sde 6f in Z[ζ₉] against 2f in Z[ω]. Nick's data confirm sde ≈ 6f.
  - Per sde level, which is what gates pay for, both rings gain one factor of 3.
- **Rigorous piece.** The σ₁-projection covering radius of Z[ζ₉] is Θ(B^{−1/2}), the trivial lattice rate (May 2026, Path C analysis).

## Consequences
- **The per-phase slopes match** up to a gates-per-sde-level constant. C+D spends ≈1.25 phase-units per level.
- **The extra C+D candidates differ mostly in the shadows**, which the hardware never sees. So better deterministic search can't beat C+R; the Bocharov-port, Babai-CVP, Path C and Euler-merging attempts all failed for this reason.
- **RUS does not break the tie.**
  - RUS's gain comes from relaxing the *target region*: phase only, area ~ε instead of ~ε^a. That predicts ~3× on qubits; 2.6× is observed.
  - The qubit RUS construction still bounds the shadow.
  - So RUS helps C+D and C+R by about the same factor.
- **What does break it is the unit.** Per phase, the gate sets tie. In fault-tolerant T-cost they don't (`04`), because C+R's every non-Clifford is an R (7 T each), while C+D uses mostly T-type gates (1 T each).
