# 04: Fault-tolerant cost: C+D device vs C+R device

**Setting.** Both devices are generic qutrit stabilizer-code machines (surface or color code) with magic-state factories. Metaplectic/anyon hardware is excluded as too far out. **Unit:** T magic states (level-3 qutrit T-states).

## T-cost per R_z rotation (C+D synthesis, Nick's approximants)
T-cost = n_T + 7·n_4 + 7·n_R (`03` gadgets) = **16.6 + 10.0·log₃(1/ε)**, R² = 0.874 → **227 T at ε = 10⁻¹⁰**.
- **Where the T-cost goes** (stable across f): level-4 ≈ 56%, R ≈ 25%, T-type ≈ 18%.
- **The counts are real circuits.** `compiler/cd_to_ct.py` emits the full Clifford+T circuit on system plus ancilla for any decomposition and verifies it exactly (err ~1e-13). The emitted T-count equals the formula (checked at f = 4 and 12).

## Against C+R (Gustafson's fits)

| C+R device | T per rotation at 10⁻¹⁰ | C+D advantage |
|---|---|---|
| R built from 7 T (best available on a T-factory machine) | Householder 776 / Exhaustive 619 | **3.4× / 2.7×** |
| R from its own R-state factory | 111 × c_R (Householder) | C+D wins iff c_R/c_T > **2.04** (Exhaustive: > 2.56) |

## Factory model: what c_R/c_T is (`factory_model/`)

Raw injected states per logical gate: perfect Cliffords, depolarising raw error p ∈ {1e-2, 1e-3, 1e-4}, per-gate targets 1e-6…1e-12.

| R source | c_R/c_T | Dominated by |
|---|---|---|
| Golay strange-state factory (Prakash 2003.02717) + conversions + RUS injection, as published | 3×10⁴ – 6×10⁸ | 1/1728 postselection per round |
| Same, optimistic (all syndromes usable) | 16 – 180 | 32 strange states per R-state × ~3 R-states per R |
| [[5,1,3]]₃ route (Anwar–Campbell–Browne 1202.2326) | ≥10²⁵ | linear suppression only |
| R-state from P9/T by RUS (BRS 1605.02756) | ≈20–27 | 27/4–9 T per state × 3 |
| **R = 7 T (ours)** | **7** (~8–9 once the staircase is smoothed) | deterministic |

- **T factory:** qutrit Reed–Muller QRM₃(2) 8→1, rebuilt exactly: ε' = 2.0ε², threshold 0.211. Injection is deterministic.
- **No hidden non-Clifford cost.** All R conversions and the injection use only Cliffords, stabilizer measurement and postselection.
- **Structural reason R factories are bad.** R is transversal on no stabilizer code, so high-order direct distillation is impossible.
- **Conclusion.** C+D beats C+R with an R factory in every modelled regime, by far. A sensible C+R device builds R from 7 T, which gives the 3.4× above.

## Selection can lower C+D further (`06`)
- **The measured headline** uses Nick's frob-selected matrices.
- **Choosing approximants by T-cost:**
  - best-of-100 saves ~42% at ε ≤ 0.01 on HRSA candidates;
  - the extrapolated slope is 10.0 → ~8.3 per log₃, i.e. ~180 T at 10⁻¹⁰;
  - confirming this at f = 12–16 needs Nick's top-K dumps.
- **Changing the reducer's step choice does nothing** (null result), because T-cost is a property of the matrix.

## Caveats
- The space-time volume of factories is not modelled beyond raw-state counts. No published qudit factory volume estimates exist.
- The Prakash–Saha error coefficient was assumed; a smarter Golay decoder is untested.
- The one clean ancilla is reused sequentially. Parallel rotations need one ancilla each.
