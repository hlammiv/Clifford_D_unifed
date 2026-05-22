# zeta9 Cell-Setup Audit (2026-05-13)

Read-only investigation of why current norm=2 zeta9 cells produce 0 valid stage-2 triples.

## TL;DR

The wrapper hard-codes `--norm 2` to stages 1 and 2, but its `select_triples` input pattern `(inputs1=Y_u=1, inputs2=Y_u=0, inputs3=Y_u=0)` corresponds to a **single unitary row** (norm = 1), not a Householder u-vector (norm = 2). The two settings are inconsistent. With `--norm 2` the target_sum is `(2·3^{2f}, 0, 0)` but the maximum achievable Y₁+Y₂+Y₃ from the provided inputs is ~`(3^{2f}, 0, 0)`, so 0 triples are emitted by construction. The 53-triple cell (`f=2 eps=0.025`) was a leftover from when the wrapper used `--norm 1`; it is a row-of-unitary cell, and that is the format the rest of the pipeline (stage 5's `search_diagonal_matrix_two_rows_streamed_mpi.py`) actually consumes. **The fix is to revert `--norm` to `1` in both stage-1 and stage-2 calls in `zeta9_compile.py`, and delete the stale norm=2 caches.**

## 1. What `zeta9_compile.py` actually does

Tracing `zeta9_compile.py --theta T --epsilon E --max-f F`:

1. Maps `f_u = max_f` (user u-denom) → `f_v = 2·max_f` (zeta9 internal "f" = V-denom). All stages 1-5 receive `f_v`.
2. `eps_pre = epsilon/2` unless `--eps-pre` is provided.
3. Stage 1 (`ensure_precompute`, lines 102-191): runs `zeta9.collect_targets` **twice**, once with `--u 0` and once with `--u 1`, both with `--norm 2` hard-coded (line 124). Outputs `Y1_f={fv}_u={u}_eps={eps_pre}.npy`.
4. Stage 2 (lines 134-153): runs `zeta9.select_triples_optimized` with `--norm 2` (line 145) and inputs:
   - `--inputs1 = Y1_*_u=1` (one copy)
   - `--inputs2 = Y1_*_u=0` (one copy)
   - `--inputs3 = Y1_*_u=0` (one copy)
5. Stages 3–4 are theta-independent precompute; stage 5 is the actual diagonal-target search.

There is no iteration over f or ε. The wrapper takes one cell per invocation.

Inside `collect_targets._collect_targets_single_range_mpi` (collect_targets.py:705-1053):
- `base_scale = 3^{2f}`, `total_scale = norm·3^{2f}`, `target_t = base_scale·u`.
- Generates Y = (n₀, n₁, n₂) where `s_1 = n₀ + n₁α + n₂α² ∈ [target_t − ε_Y, target_t + ε_Y]` and the other two Galois embeddings `s_2, s_4 ∈ [0, total_scale]`.
- The `--norm` parameter affects ONLY the per-Galois-conjugate upper bound `total_scale` — i.e. with `--norm 2` you get a *superset* of the `--norm 1` Y list. The Y values themselves and `target_t` are identical.

Inside `select_triples_optimized` (line 825-826):
```python
total_scale_int = int(round(norm * (3 ** (2 * f))))
target_sum = np.array([total_scale_int, 0, 0], dtype=ROW_DTYPE)
```
So with `--norm 2, f=2`: `target_sum = (162, 0, 0)`. With `--norm 1, f=2`: `target_sum = (81, 0, 0)`.

## 2. Is `--norm 2` correct?

**No, not for this pipeline.** Two pieces of evidence:

(a) The downstream consumer of stage-2 triples is `search_diagonal_matrix_two_rows_streamed_mpi.py`. Its row-1 and row-2 of a unitary are 3-component vectors with `Σ |U_ij|² = 1`. The "triples" are the three `Y_j = |U_ij|² · 3^{2f}` numerators of one unitary row, summing to `(3^{2f}, 0, 0)` (norm = 1). This is confirmed at `search_diagonal_matrix_two_rows_streamed_mpi.py:1115-1120`:
```python
row1_target = np.array([d1, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
row2_target = np.array([0.0 + 0.0j, d2, 0.0 + 0.0j], dtype=np.complex128)
```
The triples represent a row near `(d1, 0, 0)` (or, by the internal permutation at lines 1336, `(0, d2, 0)`). One large component (|·|²=1), two small (|·|²=0), summing to (3^{2f}, 0, 0).

(b) The memory `zeta9_audit_2026-05-09.md` documents: *"Phase 2 sum invariant Y₁+Y₂+Y₃ = (norm·3^(2f), 0, 0): 591/591 rows correct (sum = (81, 0, 0) for norm=1, f=2)"*. The working f=2 cell built at norm=1 had target (81, 0, 0). After 2026-05-12 the wrapper was changed to norm=2 to "match Householder", but the rest of the wrapper's input pattern (Y_u1, Y_u0, Y_u0) was not adjusted.

(c) The Householder formulation lives in `ep_compile.py` and the `householder_v_from_u()` helper inside `zeta9_compile.py`. Householder u = (e^{iθ/2}, −1, 0) does have `|u|² = 2`, BUT this code path is NEVER reached by the wrapper. The actual zeta9 stage-5 emits full 3×3 rows via `rows_coeffs_layer_f`, and `extract_best_v` (since 2026-05-12) reads those directly without going through Householder reconstruction. The Householder math is dead code in the current wrapper.

## 3. Why current cells produce 0 stage-2 triples (mathematical reason)

At `--norm 2`:
- `target_sum = (2·3^{2f}, 0, 0)`.
- `inputs1 = Y_u=1`: n₀-coefficient is close to `target_t = 3^{2f}` (i.e. `~3^{2f}`).
- `inputs2, inputs3 = Y_u=0`: n₀-coefficient is close to `target_t = 0` (i.e. `~0`).
- Maximum reachable `Y₁[0] + Y₂[0] + Y₃[0] ≈ 3^{2f} + 0 + 0 = 3^{2f}`, half of the target's first component `2·3^{2f}`.
- No triple from these inputs can sum to `(2·3^{2f}, 0, 0)`. → **0 hits, by construction.**

Empirical: at f=4, eps=0.0025, |Y_u=1| = 3,998,215 and |Y_u=0| = 231,442. Even with ~10¹² candidate triples the entire ensemble lies in a half-target shell. No combinatorial luck closes the gap.

The norm=1 f=2 eps=0.025 cell worked because target_sum = (81, 0, 0), and (Y_u1's n₀≈81) + (Y_u0's n₀≈0) + (Y_u0's n₀≈0) lands exactly on 81. That's the correct mathematical setup — one unitary row, not a Householder u-vector.

## 4. Is `try.py` / X2_*_best.npz the path to use?

**No.** `try.py:99-147` (`Load`) loads `D/X2_f={f}{mod}_eps={eps}_best.npz` files which are NOT present in the local working tree (`find … -name "X2_*best*"` returns nothing). The X2 cache is a historical artifact from the original zeta9 paper repo and represents Householder u-vectors that were Tier-1 picks for fixing as row-1 in stage 5. The current `zeta9_compile.py` wrapper does NOT use this path; it lets stage 5 enumerate row-1 candidates directly from the TM file. As of the 2026-05-12 memory `zeta9_e2e_works_2026-05-12.md`, the standalone path is confirmed to work (modulo the wrapper bug we're now diagnosing).

Note: `try.py`'s `Test` function passes a `fixed_row1_sage` (the matrix's actual row 1 from the X2 cache) and lets stage 5 find row 2. It still depends on a TM file built with norm=1. The norm conflict would bite `try.py` too if the wrapper-built TM was norm=2.

## 5. Concrete recommendation

**Change `zeta9_compile.py` line 124 and line 145 from `--norm 2` to `--norm 1`.** Then delete the stale norm=2 caches:
- `D/TM_f=2_eps=0.05*`, `D/TM_f=4_eps=0.0025*`, `D/TM_f=4_eps=0.025*` (manifests, parts, join_buckets, c_partitions)
- The corresponding `RM_*` (find_roots output) and `chunk_meta` artifacts
- The `Y1_*_eps={eps}.npy` files: keep these — they are supersets of the norm=1 Y lists, so they remain *valid* for norm=1 selection. (Stage 1 won't be re-run if the file exists.)

Re-run the wrapper for `f=2 eps=0.05`. With `--norm 1` and target_sum = (81, 0, 0), the stage 2 join should produce ~hundreds of triples (the audit found 591 at f=2 eps=0.05 historically). Stage 5 can then proceed and the standalone path works.

If you actually want to use Householder (norm=2) downstream, you'd need to BOTH (i) change the input pattern in `ensure_precompute` to `(Y_u=1, Y_u=1, Y_u=0)` so the three Y's sum to `(2·3^{2f}, 0, 0)`, AND (ii) build a separate Householder-aware stage 5 search — the current `search_diagonal_matrix_two_rows_streamed_mpi.py` is unitary-row-oriented and cannot consume Householder u-triples.

## Files referenced

- `/home/hlamm/Desktop/efficent_gates/unified/zeta9_compile.py` (the wrapper; lines 102-153 contain the bug)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/collect_targets.py` (stage 1; norm only affects per-conjugate bound)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/select_triples.py` and `select_triples_optimized.py` (stage 2; target_sum = norm·3^{2f})
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py` (stage 5; row-of-unitary norm = 1 confirmed at lines 1115-1124)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/try.py` (uses Householder path via X2 cache + fixed_row1; not applicable for fresh wrapper runs)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/D/TM_f=2_eps=0.025.manifest.json` (working cell, `norm: 1, total_scale_int: 81, rows_written: 53`)
- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/D/TM_f=2_eps=0.05.manifest.json` (broken cell, `norm: 2, total_scale_int: 162, rows_written: 0`)
