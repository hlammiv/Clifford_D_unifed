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

## UPDATE 2026-10-01: measurement model (level-4 = 4 T, R still 7 T)
Generated with (superseded below: both gadgets 4 T).
- **T-cost = 12.7 + 7.59·log₃(1/ε) → ~172 T at 10⁻¹⁰** (f = 14: 163; f = 16: 185).
- **vs C+R with R = 7 T:** 4.5× (Householder) / 3.6× (Exhaustive).
- **Break-even against an R factory:** c_R/c_T > 1.55.
- **Provisional:** if measurement tricks also lower R, C+R built from T gets cheaper too, and the ratio has to be recomputed with the same R cost on both sides.

## UPDATE 2026-10-01 (later): measurement model with level-4 = 4 T and R = 4 T (both sides)
`python3 compute_tables.py --w4 4 --wr 4` → `tables_generated_measurement_w4-4_wr-4.md`.
- **C+D T-cost = 9.7 + 6.58·log₃(1/ε) → ~148 T at 10⁻¹⁰** (f = 14: 139; f = 16: 160).
- **C+R with R = 4 T:** Householder 444 / Exhaustive 354 at 10⁻¹⁰, so **C+D 3.0× / 2.4× cheaper**. The C+R side benefits from the 4-T R as well; this is the like-for-like comparison.
- **Break-even against an R factory:** c_R/c_T > 1.33.

## UPDATE 2026-10-02: T-merging, T-depth, lower bound (`depth_optimality/`)
- **Free ~25% T saving in the unitary model.**
  - Consecutive 7-T gadgets that share the ancilla begin and end with ancilla-only T rotations, which cancel or fuse.
  - Standard rotation merging (`tmerge.py`) on 31 of Nick's matrices: T_merged / T = **0.746** (0.69–0.87). Exact: the merged circuit reproduces the 9×9 unitary to 1.7e-11.
  - Extrapolated: ~11.7 + 7.6·log₃ → **~170 T at 10⁻¹⁰** (vs 227 unmerged; measurement model 148).
  - The measurement model gains nothing from merging.
- **T-depth.**
  - Gadgets: 7-T = 3 rotation layers; 4-T = 2.
  - Per rotation: ~0.91 × T-count as emitted. The minimum over re-orderings is ~4.6·log₃ (fresh ancilla per gadget) or ~3.6·log₃ (measurement).
  - Floor: one layer per non-Clifford syllable.
- **Lower bound.**
  - Ancilla-free single-qutrit C+T has N(≤t) = (216/5)(8·6ᵗ − 3) operators (Glaudell–Ross–Taylor; checked to t = 8). That gives **T ≥ 4.90·log₃(1/ε)**: rigorous worst case over PU(3); for typical θ under an equidistribution assumption supported by enumeration.
  - Ours: 10.0 (2.04×), 7.6 merged (~1.55×), 6.58 measurement (1.34×).
  - **Caveats:** with ancillas or measurement the counting bound weakens (slope ~2 for one ancilla), so the 4.90 floor strictly applies only ancilla-free. The fully T-type ideal (3.15) is impossible because these approximants are not in C+T, which is a thin subgroup (Evra–Parzanchevski).

## UPDATE 2026-10-02 (later): all savings stacked (see README headline)
- **Symmetry-copy selection × rotation merging × gadget model** (`symmetry_variants/stack_check.py`, 210 matrices × 324 copies):

  | Model | C+D at 10⁻¹⁰, best copy | Fit |
  |---|---|---|
  | Unitary | **206** | 2.5 + 9.71·log₃ |
  | Merged | **154** | 3.3 + 7.21·log₃ |
  | Measurement | **136** | 1.8 + 6.42·log₃ |

- **Like-for-like C+R** (R = 7 / 5.0 merged / 4 T), Householder: 776 / 554 / 444. **The C+D advantage is 3.3–3.8× (Householder) and 2.6–3.0× (Exhaustive) in every model.**
