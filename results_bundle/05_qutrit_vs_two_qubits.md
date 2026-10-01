# 05: Qutrit vs two-qubit emulation, in consistent units

**Setup** (as in Gustafson et al. §5): an arbitrary single-qutrit gate is 6 diagonal rotations, and its two-qubit emulation is 10 single-qubit R_z. The ratio is qutrit magic states ÷ qubit T gates.

| Qutrit side (×6) | vs 2-qubit **deterministic** (Ross–Selinger ≈ 3·log₂(1/ε)) @1e-6 / @1e-10 | vs 2-qubit **RUS** (9.2 + 3.817·log₁₀, BRS 2015) @1e-6 / @1e-10 |
|---|---|---|
| **C+D T-cost (as is)** | 1.43 / **1.36** | 2.66 / **2.87** |
| C+D T-cost, best-of-100 (extrapolated slope 8.3) | 1.21 / **1.15** | 2.26 / 2.41 |
| *C+D per-phase N_φ (mixed units)* | *0.69 / 0.67* | *1.29 / 1.42* |
| C+R Householder, R = 7 T | 4.76 / 4.67 | 8.87 / 9.83 |

Absolute counts at 10⁻¹⁰:

| Approach | Magic states per arbitrary gate |
|---|---|
| Qutrit C+D | 6 × 227 = 1,359 T₃ |
| 2-qubit deterministic | 997 T₂ |
| 2-qubit RUS | 474 T₂ |

## Retraction
- **The old figures mixed units.** The earlier "C+D has 1.37× overhead vs two qubits at 10⁻¹⁰" (June one-pager) and Gustafson et al.'s 1.12× / 1.40× compared per-phase or R gate counts with qubit T counts.
- **They are not cost statements.** The *mixed units* row above reproduces them.

## Correct framing
- **Like-for-like (deterministic vs deterministic):** a qutrit C+D gate needs ~1.36× the magic states of two-qubit emulation, or ~1.15× with T-cost-aware selection.
- **Against qubit RUS** (a stronger, probabilistic baseline): ~2.9× (~2.4×). Qutrit RUS was tabled; it does not break the C+D/C+R tie but could narrow this gap.
- **Do not claim parity with two qubits.** The robust claim is qutrit-to-qutrit: C+D ≈ 3.4× cheaper than C+R.

## Caveats
- **Per-state factory cost differs.** A qutrit T-state from QRM₃(2) 8→1 (2ε²) costs 0.28–4.3× a qubit T-state from 15→1 (35ε³) in raw states, depending on which side of a round boundary the target falls (p = 1e-3/1e-4). There is no robust direction.
- **Modern qubit factories favour qubits.** Cultivation and Litinski's factories widen the qubit advantage, and no qutrit analogues exist.
- **Physical footprint is not modelled.** One qutrit vs two qubits, and qutrit vs qubit code distances, are left out.
