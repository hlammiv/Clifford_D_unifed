# zeta9 Stage 2 Math Audit — 2026-05-13

## Executive summary

**The math is correct. The wrapper has a `--norm` bug.**

Stage 2's joint constraint `Y_1 + Y_2 + Y_3 = (norm·3^{2f}, 0, 0)` in `Z[α]` coefficient form is mathematically correct for matching the *rows of the diagonal unitary matrix V*, **provided `norm` matches the choice of input files**. The wrapper (`/home/hlamm/Desktop/efficent_gates/unified/zeta9_compile.py`) passes `--norm 2` while feeding inputs `(u=1, u=0, u=0)`, whose σ₁ embeddings sum to ≈ `1·3^{2f}` — not `2·3^{2f}`. The constraint is therefore infeasible *by σ₁ arithmetic alone*, before any (m1, m2)-cancellation question even arises. The 53-row success at `f=2 ε=0.025` was a norm=1 run; every norm=2 manifest in `D/` is `rows_written: 0`. Fix: change `--norm 2` to `--norm 1` in both stage-1 and stage-2 invocations inside `zeta9_compile.py` (lines 124 and 145).

## 1. Algebraic conventions confirmed

### Ring and embeddings
- `Y_i = (m0, m1, m2)` represents `m0 + m1·α + m2·α² ∈ Z[α]` where `α = ζ_9 + ζ_9^{-1} = 2cos(2π/9)`. This is the maximal real subfield of `Q(ζ_9)`, totally real of degree 3.
  - Confirmed in `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/collect_targets.py:20-31` (ALPHA1/2/4 constants) and `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/tools.py:170-199` (basis conversion).
- The three real embeddings `σ_1, σ_2, σ_4` send α to `2cos(2π/9), 2cos(4π/9), 2cos(8π/9)` respectively.
- Stage 1 enforces:
  - `σ_1(Y) ∈ [t − ε_Y, t + ε_Y]` where `t = u·3^{2f}` and `u ∈ {0, 1}`,
  - `σ_2(Y), σ_4(Y) ∈ [0, norm·3^{2f}]` (positivity at the conjugate embeddings, the totally-positive-norm condition).

### Scale convention
For `x = b/3^f` with `b ∈ Z[ζ_9]`, we have `x·conj(x)·3^{2f} = b·conj(b) ∈ Z[α]`. Equivalently `|x|² = Y/3^{2f}` and Y is positive in all three embeddings. This is the standard Kalra ringZ9 layer-f convention; it matches HRSA's `householder_search.cpp` and the `bb_to_real_coeffs(n)` formula at `tools.py:216`.

### What does the search actually find?
Stage 5 (`search_diagonal_matrix_two_rows_streamed_mpi.py`) searches for **rows of the 3×3 diagonal-target unitary V ≈ diag(e^{−iθ/2}, e^{iθ/2}, 1)** — *not* a Householder reflection vector. Row 1 ≈ (d1, 0, 0), row 2 ≈ (0, d2, 0). The "Householder workflow" name in `zeta9/README.md` is a vestige of an earlier formulation. The wrapper `zeta9_compile.py:220` then *reconstructs* a Householder matrix `V = X_{(0,1)}·(I − u·u†)` from the row-1 result, but that's a post-processing step that does not affect the triples constraint.

**Each row is a UNIT VECTOR**, so `|row|² = |Y_1|² + |Y_2|² + |Y_3|² / 3^{2f} = 1`, i.e. `Σ Y_i = 1·3^{2f}` in Z[α]. The `norm` parameter should be **1**, not 2.

## 2. The norm parameter mismatch

`norm` semantically encodes the σ₁ scale of the target sum: `target_sum = (norm·3^{2f}, 0, 0)`. For the input pattern `(u=1, u=0, u=0)` used by the wrapper:

| Quantity | u=1 entry | u=0 entry | Sum (u=1, u=0, u=0) | norm=1 target | norm=2 target |
|---|---|---|---|---|---|
| σ₁(Y_i) ≈ | `3^{2f}` ± ε | `0` ± ε | `1·3^{2f}` ± 3ε | `3^{2f}` ✓ match | `2·3^{2f}` ✗ infeasible |

The σ₁ arithmetic alone makes norm=2 with `(u=1, u=0, u=0)` inputs *intrinsically* infeasible (gap is `3^{2f}` ≫ 3ε at any realistic ε).

If norm=2 were intended, the inputs would need to be `(u=1, u=1, u=0)` (no such file exists) or `(u=2, u=0, u=0)` (illegal — collect_targets requires `0 ≤ u ≤ norm`).

### Empirical confirmation: the existing TM manifests

```
TM_f=2_eps=0.025.manifest.json   norm: 1  total_scale_int: 81     rows_written: 53
TM_f=2_eps=0.05.manifest.json    norm: 2  total_scale_int: 162    rows_written: 0
TM_f=4_eps=0.0025.manifest.json  norm: 2  total_scale_int: 13122  rows_written: 0
TM_f=4_eps=0.025.manifest.json   norm: 2  total_scale_int: 13122  rows_written: 0
```

The ONLY successful Stage 2 cell uses norm=1. Every norm=2 cell produces 0 triples. This is a deterministic outcome of the σ₁ sum constraint, not a sparsity accident.

### Author-side scripts confirm norm=1

All upstream/authoritative scripts in `zeta9/` (`run_matrix`, `run_select_triples_matrix`, `zeta9_2x2/run_matrix_2x2`) use `--norm 1`. The `--norm 2` default in `collect_targets.py:1228` and `select_triples_optimized.py:1063` is a leftover from an older "Householder-reflection" formulation that was never matched to the current input pattern.

## 3. Empirical Y₁ distribution check

Loaded `D/Y1_f=*_u=*_eps=*.npy`:

```
f=2 u=0 ε=0.025:     23 rows | m0 ∈ [0, 88]    | m1 ∈ [-27, 0]   | m2 ∈ [-29, 9]
f=2 u=1 ε=0.025:    226 rows | m0 ∈ [-12, 81]  | m1 ∈ [0, 29]    | m2 ∈ [-10, 30]
f=4 u=0 ε=0.0025:   231k rows| m0 ∈ [0, 15233] | m1 ∈ [-4966, 0] | m2 ∈ [-4971, 1718]
f=4 u=1 ε=0.0025:  3.998M rows| m0 ∈ [-1057, 14177] | m1 ∈ [-2485, 2486] | m2 ∈ [-3349, 3350]
```

Observations:
- m0 (the rational part) is non-negative for u=0 (correct: it dominates a positive-definite norm near zero) and centered near 3^{2f} for u=1. Magnitudes are consistent with `|x|²·3^{2f} ≤ norm·3^{2f}` — values for f=4 max around 13122 ≈ 2·3^{2f}, consistent with the σ₂, σ₄ cap (NB: the cap is on σ₂, σ₄ but not directly on m0; the values still fall within plausible Z[α] elements of the embedding box).
- m1 distribution is **asymmetric for u=0** (always non-positive!) and for u=1 (f=2, always non-negative; f=4 only mildly asymmetric). This reflects the conjugate-embedding positivity constraints: u=0 Y's are concentrated near 0 in σ₁ but spread to positive values at σ₂, σ₄, biasing (m1, m2) into specific quadrants.
- m1 = 0 fraction: 0–5% across files; m2 = 0 fraction similar. Joint m1=m2=0 ≈ 10⁻⁴ — small but not vanishing.

The m1-cancellation feasibility for `m1_a + 2·m1_b` (with a from u=1, b from u=0) at f=4:
- Achievable range: `[m1_u1.min() + 2·m1_u0.min(), m1_u1.max() + 0] = [−12417, +2486]`. So m1 sum = 0 is *within range*, just very sparse.

The cancellation problem at norm=1 is genuinely hard but *not impossible* — the f=2 ε=0.025 case found 53 triples. At f=4 ε=0.0025, with ~4M × 231k × 231k = 2.1×10^17 a-priori triples, finding any with all three constraints (σ₁ near target, m1 sum = 0, m2 sum = 0 exactly) is highly nontrivial but well-defined and computationally addressable.

## 4. What `try.py` does

`/home/hlamm/Desktop/efficent_gates/unified/zeta9/try.py` does **not** use a different math formulation. It is a manual driver that:
1. Loads pre-computed best **row 1** from `D/X2_f={f}_eps={eps}_best.npz` (computed elsewhere — perhaps HRSA or an earlier zeta9 row-1-only run).
2. Computes the corresponding Householder matrix `D = X_{(0,1)}·(I − x†x)` for diagnostic display.
3. Calls `search_diagonal_matrix_two_rows_streamed_mpi.py` with `--fixed_row1_sage <row1>` to search only for **row 2** given that row 1.

This **assumes** the TM/RM/triples files in `D/` already exist (built by `run_matrix` with `--norm 1`). It bypasses stages 1–4 entirely. So try.py is irrelevant to the joint-norm question: it just consumes the norm-1 artifacts. Per the memory note `zeta9_findings_2026-05-07.md`, the integration mismatch was that try.py's triples file has *row-1* Y values, while stage-5's row-2 preselect needs *row-2* Y values — that's a separate issue (the triples list semantically is "norms whose row could be row 1 of the matrix near (d1, 0, 0)", not "norms for row 2 near (0, d2, 0)").

## 5. Where does "0 triples" come from?

**It is a wrapper configuration bug, not a math problem and not a sparsity wall.**

- **Math:** Correct, when `norm` matches input pattern.
- **Constraint setup:** Correct in `collect_targets.py` and `select_triples_optimized.py`.
- **Wrapper:** `zeta9_compile.py` passes `--norm 2` with inputs `(u=1, u=0, u=0)`. The σ₁ sum is ~`3^{2f}` but the target σ₁ is `2·3^{2f}`. Mismatch is enormous (gap ≈ 3^{2f}, tolerance ≈ 3ε ≪ 3^{2f}).
- **Easy fix:** In `zeta9_compile.py`, change both `--norm 2` occurrences (lines 124 and 145) to `--norm 1`. Delete the stale TM_f=*_eps=* and Y1_f=*_eps=* files built with norm=2 so they're regenerated.

After that fix, the f=2 ε=0.025 cell should reproduce 53 triples, and the f=4 ε=0.0025 cell may produce a (small) nonzero count once stage 2 completes with the right target.

### Secondary issue (not blocking)
The `--norm 2` default in `collect_targets.py:1228`, `select_triples.py:453`, `select_triples_optimized.py:1063`, and `find_roots_old.py:1352` is misleading given that the matrix workflow uses `--norm 1`. Changing the default to 1 would prevent future wrapper bugs. The "Householder reflection" wording in `zeta9/README.md` should also be clarified — the current pipeline searches for *rows of a diagonal unitary*, with a Householder matrix reconstructed only as a post-step in `zeta9_compile.py`.

## Files referenced

- `/home/hlamm/Desktop/efficent_gates/unified/zeta9_compile.py` (wrapper, lines 124, 145 — the bug)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/collect_targets.py` (stage 1)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/select_triples_optimized.py` (stage 2; target_sum at line 826)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py` (stage 5; row1_target/row2_target at lines 1119–1120)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/tools.py` (Z[α] arithmetic, basis conversions)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/run_matrix` (author's canonical script, uses `--norm 1`)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/try.py` (manual driver; not the source of the bug)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/D/TM_f=*.manifest.json` (empirical evidence: norm=1 ⇒ 53 rows; norm=2 ⇒ 0 rows)
