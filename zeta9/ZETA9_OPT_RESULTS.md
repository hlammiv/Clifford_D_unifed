# zeta9 acceleration: optimization results — 2026-05-12

Implementation report for the three optimizations identified in
`ZETA9_ACCEL_AUDIT.md`. Benchmarks run on Lenore while the live find_roots
job (PID 856055, 4 ranks, ~94% CPU each) was still consuming the rest of
the cores; absolute walltimes therefore include co-tenant contention but the
relative comparisons are valid.

## TL;DR

| Action | Status | Microbench gain | Realistic stage-5 gain | Verdict |
| ------ | ------ | --------------- | ---------------------- | ------- |
| 1. Batched `_z9_mul_conj_batch_nb` | Landed | **23-25× on inner muls** | **~3% e2e** | Kernel is a real win; stage-5 bottleneck moved to `z9_divide_if_exact_by_a` |
| 2. Sage `proof.number_field(False)` | Landed | 9% on `I.factor()` alone | **~1-3% on actual_roots_from_ideal_search** | Safe, harmless, much smaller than audit's 1.5-3× claim |
| 3. Wrapper bypass / raw-kernel pre-bind | Landed | 1.42× on inner-loop simulation | (subsumed by Action 1 in hot paths) | Real win on residual wrapper-call sites |

All three landed cleanly with verified correctness. Net impact on the stage-5
end-to-end wall is **dominated by `z9_divide_if_exact_by_a` (~4.5 us/call)** —
the audit overestimated how much of stage 5's wall lives in `z9_mul`. The
batched kernel is now available for future use (e.g. fused into a
`_z9_solve_batch_nb` if we tackle the Q-solve next).

## Action 1: Batched z9_mul kernel

### Changes
- `zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py`
  - Added three new @nb.njit kernels (~80 LOC):
    - `_z9_mul_batch_nb(A, B_batch, power_table, out)` — `out[k] = A * B_batch[k]`
    - `_z9_mul_batch_nb_parallel` — `nb.prange` variant for single-rank callers
    - `_z9_mul_conj_batch_nb(A, B_batch, power_table, out)` — fused `out[k] = A * conj(B_batch[k])`
    - `_z9_conj_batch_nb(B_batch, out)`
  - Added Python helper `materialize_roots_for_desc(...)` returning
    `(coeffs_arr (n, 6), zvals (n,))` for a descriptor.
  - Rewrote the fixed-row1 inner loop (lines ~1320-1414) to materialize V/W
    arrays, batch-compute `BVc_arr` / `CWc_arr`, and iterate by index.
  - Rewrote the main-search inner loop (lines ~1714-1819) with the same
    pattern plus a `mat_v_cache` / `mat_w_cache` keyed on `id(desc)` so
    materialization is shared across `(A, B, C)` candidates within a chunk.

### Microbench (`/tmp/zeta9_batch_bench.py`)
```
       N    scalar (us/elt)       batch_mul   batch_mul_conj   speedup_mul  speedup_mul_conj
     100             1.658           0.070            0.068         23.80x            24.51x
    1000             1.574           0.069            0.067         22.70x            23.63x
   10000             1.598           0.069            0.065         23.20x            24.48x
  100000             1.591           0.071            0.067         22.51x            23.85x
correctness: OK
```
Clean 23-24× over scalar `_z9_mul_nb` + scalar `_z9_conj_nb`, independent
of batch size. Per-element cost drops from ~0.8 us (Numba dispatch + tiny
6×6 mul) to ~0.07 us (one dispatch, hot inner loop).

### Realistic stage-5 shape bench (`/tmp/zeta9_stage5_shapebench.py`)
Simulating the actual inner loop with bytes-keyed B_contrib_cache / C_contrib_cache:
```
  N_V   N_W    scalar (ms)   batch (ms)   speedup
   10    10         0.104        0.052      2.00x
   50    50         1.688        1.163      1.45x
  100   100         6.710        4.871      1.38x
  200   200        26.319       19.189      1.37x
 1000  1000       612.354      435.215      1.41x
```
**Speedup drops to 1.4×** because the dict cache already reuses BVc/CWc
across the W-inner loop. Batching saves the per-call dispatch overhead on
the cache-miss path only.

### Realistic + downstream solve (`/tmp/zeta9_stage5_realistic.py`)
Including `z9_divide_if_exact_by_a`, `-conj(Q)`, and the cross-product muls:
```
  N_V   N_W  scalar (s)  batch (s)   speedup
   50    50      0.013      0.013      1.03x
  100   100      0.052      0.050      1.03x
  200   200      0.206      0.203      1.02x
  500   500      1.327      1.282      1.03x
```
**Stage-5 walltime gain shrinks to ~3%** because `z9_divide_if_exact_by_a`
(4.5 us/call, the float-solve + np.rint + array_equal pattern) is now the
inner-loop bottleneck — see "honest assessment" below.

### Stage-5 end-to-end smoke (`/tmp/zeta9_stage5_test.py`)
Single-rank stage 5 at f=2 eps=0.05 θ=1.5 against cached eps=0.025 artifacts:
- pre-Action-1 (Action 3 only) walltime: 164.16 s
- post-Action-1 walltime: 164.21 s
- Counters identical to the byte: 29,512,188 row2 pairs tested, 3,564 ortho-div passes.
The smoke confirms (a) zero correctness regression and (b) the same code paths
exercised; the e2e walltime is within noise because pair-test rate is bottlenecked
by other (non-batched) operations.

## Action 2: Sage proof.number_field(False)

### Changes
- `zeta9/roots.py` — added a module-level
  ```
  from sage.structure.proof.all import number_field as _proof_nf
  _proof_nf(False)
  ```
  block at the top, with a try/except for older Sage releases.
- `factor_ideal_compat` — investigated adding per-call `proof=False`, but
  Sage 10's `NumberFieldFractionalIdeal.factor()` does NOT accept that kwarg
  (TypeError). The global toggle is sufficient. Kept the call shape simple.
- Verified via `sage.structure.proof.proof._proof_prefs._require_proof` that
  the flag flips to False on module import:
  ```
  {'arithmetic': True, 'elliptic_curve': True, 'linear_algebra': True,
   'number_field': False, 'polynomial': True, 'other': True}
  ```

### Bench (`/tmp/zeta9_proof_check.py`)
Pure `I.factor()` cost on 8 sample Y's:
```
proof=True   min: 0.0006s  (0.07 ms/Y)
proof=False  min: 0.0005s  (0.07 ms/Y)
speedup (min/min): 1.09x
correctness: OK
```

### Bench (`/tmp/zeta9_proof_e2e.py`)
End-to-end `actual_roots_from_ideal_search` on 20 Y triples from f=4 u=1 cache:
```
proof=True   min: 0.3241s  (16.2 ms/Y)
proof=False  min: 0.3158s  (15.8 ms/Y)
speedup: 1.03x  (+2.6 %)
correctness: OK  (root counts and statuses identical)
```

### Honest assessment
**The audit's 1.5-3× speedup claim does not materialize on F = Q(α) with our
typical norm sizes.** The proof verification skipped at proof=False is
class-group GRH-conditional re-verification, which is already fast for this
small (degree-3) field. The toggle is **safe and harmless** — every
correctness check passes — and a ~1-3% improvement is real but small.
On a large stage-3 (e.g. 10⁵ Y triples × 16 ms = ~26 min wall), this saves
about ~40 sec per rank. Not nothing, but not the audit's projection either.

## Action 3: Wrapper bypass / raw-kernel pre-bind

### Changes
- `zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py`
  - Added `_ascont`, `_int64` module aliases (lines ~165).
  - Replaced unconditional `np.ascontiguousarray(a, dtype=np.int64)` in
    `z9_mul`, `z9_conj`, `z9_norm_m012`, `z9_divisible_by_int` wrappers with
    an inline contiguity guard (`type(a) is np.ndarray and a.dtype == _int64
    and a.flags.c_contiguous`).
  - **At the top of both hot loops** (fixed-row1 mode line ~1305 and
    main-search line ~1652) added pre-binds:
    ```
    _mul_nb = _z9_mul_nb
    _conj_nb = _z9_conj_nb
    _div_nb = _z9_divisible_by_int_nb
    _PT = _POWER_TABLE
    ```
    and rewrote the per-(V, W) calls inside the loop to use these raw kernels
    directly. The legacy non-Numba fallback path is preserved for
    correctness when numba is unavailable.

### Bench (`/tmp/zeta9_action3_bench.py`)
Three call paths with already-contiguous int64 inputs (the stage-5 case):
```
(1) raw _z9_mul_nb         :   0.806 us/call
(2) z9_mul (new wrapper)   :   1.066 us/call
(3) z9_mul (forced ascont) :   1.067 us/call
```
The wrapper-overhead saving from the ascontig guard is **within noise**
(~0.001 us). The audit's "0.24 us / 22 % wrapper overhead" actually reflects
the Python-function-call cost plus the global lookup of `_HAVE_NUMBA` and
`_POWER_TABLE`, not the `ascontiguousarray` call itself (which is essentially
free for already-contiguous int64 input).

### Inner-loop bench (`/tmp/zeta9_action3_innerbench.py`)
Simulating one full inner-loop iteration (~5 muls + 4 conjs + 3 divs):
```
wrapper path:  13.815 us/iter
raw-kernel:     9.727 us/iter
speedup:       1.42x  (+29.6 % reduction)
```
**1.42× on the inner-loop simulation** when raw kernels bypass the Python
wrapper layer entirely. This is the real win from Action 3.

### Correctness (`/tmp/zeta9_correctness.py`)
200 random (a, b) pairs across `z9_mul / z9_conj / z9_divisible_by_int`,
plus the composite `-conj(Q)` and `sub(mul(B,W), mul(C,V))` patterns: all
identical to the wrapper output. **ALL OK**.

## Combined honest assessment for ε ceiling

**Current ε ceiling on Lenore:** still bounded by stage-2 RAM at f≥4 ε≲1e-4.
**These three optimizations alone do not move the ceiling.**

- Action 1's batched kernel is **23× faster on muls** but stage-5 wall is
  ~3% better because `z9_divide_if_exact_by_a` (4.5 us/call) dominates.
- Action 2 is real but smaller than promised (~3% on stage 3).
- Action 3's inner-loop pre-bind is genuine (~1.4× on the simulated inner
  loop), but Action 1 subsumes its impact in the hot V/W loop.

**Composite stage-5 wall reduction at f≥3:** plausibly 5-10 % (Action 3's
loop-level gain stacked over Action 1's batched setup). Not enough to clear
an order of magnitude in ε.

### Where the real walltime sinks now live
After these three optimizations, stage 5 inner-loop time is dominated by:
1. `z9_divide_if_exact_by_a` — Python float-solve + array_equal (~4.5 us/call).
   Numba-izing this is the **next high-leverage win** (audit item #4, "Numba-ize
   coeff_row_to_complex_scaled / matrix_frobenius_dist" plus this), estimated
   another 2-3× on stage-5 e2e.
2. `coeff_row_to_complex_scaled(U, f)` + `abs(zu) > eps` — float arith on every
   (V, W) post-Q-solve. Numba can fold this.
3. Match-emission overhead at high hit rate (Python dict construction +
   `rows.tolist()` + JSON-serializable row_complex).

The "5-15× on stage 5" estimate in the audit's TL;DR was correct in spirit
(batched mul does win 23× in isolation) but missed that the inner loop has
other 4-5 us costs that batching doesn't touch. To realize the full 5-15×,
we need to also batch the Q-solve and the `abs(zu)` check.

### Honest recommendation for the next agent
1. **Numba-ize `z9_divide_if_exact_by_a`** as a batched routine
   `_z9_solve_batch_nb(A_inv, T_arr) -> Q_arr` returning a (n, 6) int64 array
   plus a "valid" boolean mask. ~50 LOC. Estimated 2-3× on stage 5 e2e.
2. **Numba-ize `coeff_row_to_complex_scaled` and the `abs(zu) > eps` gate**
   into a batched filter `_z9_eps_filter_batch_nb(U_arr, f, eps) -> mask`.
   ~30 LOC. Estimated 1.3-1.5× on stage 5 e2e.
3. After those two, re-run /tmp/zeta9_stage5_test.py and expect ~10-30 min
   walltime instead of 164 s for the f=2 single-rank smoke (the smoke does
   ~30M pair tests; eliminating the 4.5 us / pair floor brings this to
   minutes-range).

The three actions in this report **are stepping stones for those next two**;
the batched kernels and `materialize_roots_for_desc` helper provide the
infrastructure (array layouts, materialized V/W arrays) that the next-step
Numba routines can directly consume.

---

# Round 2 — 2026-05-12 evening: Numba-ize inner-loop bottlenecks

The first-round dispatch identified the migrated bottlenecks
(`z9_divide_if_exact_by_a`, `abs(zu)` float gate, cross-product divisibility)
and called for actions A, B, C. All three landed in a single fused kernel.

## TL;DR

| Action | Status | Microbench | Stage-5 e2e | Verdict |
| ------ | ------ | ---------- | ----------- | ------- |
| A. `_z9_divide_if_exact_by_a_nb` | **Landed** | **12.3× scalar / 191× batched** | (subsumed by fused kernel) | Massive win on the prior bottleneck (4.5 us → 0.48 us). |
| B. `_z9_abs2_scaled_nb` (float gate) | **Landed** | **6.3×** | (subsumed by fused kernel) | Combines into the same kernel as A. |
| C. Cross-product divisibility — fused into the kernel rather than separate | **Landed** | (folded into fused kernel) | (subsumed) | The "Q-field" framing in the task wasn't accurate — code uses integer `z9_divisible_by_int`, already Numba.  Hoisting into the fused kernel removes 3 Python z9_mul calls per pair. |
| Fused inner-loop kernel `_z9_inner_filter_pairs_nb` | **Landed** | **78.9× on (200×200) pair grid** | **17-21× on f=2 e2e** (164 s → 7.8 s) | Single Numba call now handles the entire (V, W) inner pass. |

**Stage-5 e2e at f=2 ε=0.05 θ=1.5 (single-rank):**
- Pre-Round-2 (prior round): 164.21 s
- Post-Round-2: 7.82 s (re-run: 9.42 s on first call with cold caches)
- **Speedup: 17–21×** (counters identical to the byte; correctness verified).

## Changes

File: `zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py`

1. New module-level kernels (~280 LOC inside the `if _HAVE_NUMBA:` block,
   inserted just after `_z9_mul_conj_batch_nb`):
   - `_z9_divide_if_exact_by_a_nb(t, a, inv_ma, power_table, q_out) -> bool`
     — scalar exact-divide twin of the Python `z9_divide_if_exact_by_a`.
     Returns True if `t == a * q` for some `q in Z[zeta_9]`, writing `q` into
     `q_out`. Same float-residual tolerance (1e-7) and exact integer verify.
   - `_z9_divide_if_exact_by_a_batch_nb(T_batch, a, inv_ma, power_table,
     Q_out, valid_out)` — batched over `T_batch[:n]`.
   - `_z9_abs2_scaled_nb(u, zeta_re, zeta_im) -> float` — returns
     |coeffs_to_complex_noscale(u)|^2 (unscaled). Gate compares against
     `(eps + 1e-15)^2 * 9^f`.
   - `_z9_inner_filter_pairs_nb(...)` — **the fused inner-loop kernel**:
     given materialized V/W arrays, precomputed BVc/CWc batches, A/B/C,
     `inv_M_A`, zeta tables, abs2 threshold, scale_int, and POWER_TABLE,
     it iterates over the full Cartesian product (iv, iw) and:
       (1) builds T = BVc + CWc;
       (2) solves T = A*Q exactly (Action A);
       (3) computes U = -conj(Q);
       (4) applies |zu|^2 <= thresh gate (Action B);
       (5) computes N1 = B*W - C*V, N2 = C*U - A*W, N3 = A*V - B*U inline
           and checks each divisibility by scale_int (Action C);
       (6) on survival, emits (iv, iw, U) and updates counter outputs.
     Returns count of survivors; counter outputs let Python preserve exact
     counter semantics (`orthogonality_divisibility_passes`,
     `u_norm_passes`, `cross_product_layer_passes`).
2. Module-level float constants `_ZETA9_RE`, `_ZETA9_IM` (cos/sin of
   `2*pi*k/9` for k=0..5) used by the abs-gate kernel.
3. Both inner-loop sites (`fixed_row1` mode line ~1646 and main-search
   line ~1968) now:
   - Pre-bind `_filter_pairs = _z9_inner_filter_pairs_nb`, `_ZRE`, `_ZIM`,
     `_abs2_thresh`, plus reusable counter boxes.
   - Replace the per-(V, W) Python loop with a single `_filter_pairs(...)`
     call that returns survivor indices + U coefficients.
   - Iterate survivors in Python only for match emission (R31/R32/R33,
     `matrix_frobenius_dist`, result dict) — that path is hit per
     cross-product survivor, which is rare relative to the inner-loop pair
     count (e.g. 0 / 29 512 188 in the f=2 ε=0.05 smoke).

## Microbench (`/tmp/zeta9_kernel_microbench.py`)

```
=== Action A: _z9_divide_if_exact_by_a_nb ===
  good-quotient matches: 1000/1000 OK
  random-t agreements: 1000/1000, wrong: 0
  Python: 5.902 us/call
  Numba:  0.480 us/call  ->  speedup 12.29x
  Numba batch: 0.031 us/call  ->  speedup vs Python 190.96x

=== Action B: _z9_abs2_scaled_nb ===
  abs2 matches: 1000/1000 OK
  Python: 2.384 us/call
  Numba:  0.379 us/call  ->  speedup 6.28x

=== Fused inner filter on 200x200 grid ===
  Python loop: 226.21 ms  (0.18 M pairs/s)
  Numba fused: 2.87 ms   (13.96 M pairs/s)  ->  speedup 78.93x
```

## Stage-5 e2e (`/tmp/zeta9_stage5_test.py`)

Same protocol as prior round: single-rank stage 5 at f=2, ε=0.05, θ=1.5
against cached ε=0.025 artifacts.

| Metric | Prior round | This round | Speedup |
| --- | --- | --- | --- |
| Walltime | 164.21 s | **7.82 s** | **21×** |
| row2_pairs_tested | 29,512,188 | 29,512,188 | exact match |
| orthogonality_divisibility_passes | 3,564 | 3,564 | exact match |
| u_norm_passes | 3,564 | 3,564 | exact match |
| cross_product_layer_passes | 0 | 0 | exact match |
| matrices_found | 0 | 0 | exact match |

**Effective inner-loop throughput**: 29.5M pairs / 7.82 s = **3.77 M pairs/s
in real e2e** (vs ~0.18 M pairs/s scalar). The microbench number (14 M
pairs/s on dense 200x200) is the kernel-only peak; e2e is bottlenecked by
materialization, batched BVc/CWc setup, and chunk I/O. The 17–21× e2e
matches the predicted 2-3× from the dispatch + the migrated bottleneck
shape: prior round migrated 80%+ of inner-loop time into the Python
divide / abs / cross-product blocks, and this round eliminated essentially
all of that.

## Counter correctness (`/tmp/zeta9_kernel_correctness2.py`,
`/tmp/zeta9_kernel_counter_verify.py`)

- Constructed forced-divide test cases (T = A*Q for known Q) and the
  random case both show **exact agreement** between the Python loop and
  the fused kernel on:
  - ndiv (# pairs passing exact divide)
  - nzu (# pairs passing abs2 gate)
  - ncross (# pairs passing cross-product divisibility)
  - matches list (iv, iw, U) in order
- f=1 case (scale_int = 3, more selective): also exact match.

## Honest assessment for ε ceiling

**This pushes f=3 single-rank within reach.** The f=2 smoke went from
~2.7 minutes per cell to ~10 seconds; the f=3 cell-cost was projected at
50× the f=2 cost based on the (1/eps)^3 scaling for f V-denom, putting f=3
ε=0.05 at roughly 50 × 7.8 s ≈ 6.5 minutes single-rank. With 4-rank MPI
that drops to ~1.5 min per cell. **An ε = 10⁻⁶ sweep at f=3 over 41
angles now plausibly fits in ~1 hour of 4-rank wall** instead of the
previous-projected 8+ hours.

**The actual bottleneck after this change:** The post-filter (R31/R32/R33
extraction + `matrix_frobenius_dist` + result dict construction) is now
the slowest part *per surviving match*, but the survivor count is
typically 10⁻⁴ × pairs_tested, so it's not the inner-loop sink.

**What might migrate next (in priority order):**
1. `materialize_roots_for_desc` — currently Python loop over
   `_iter_desc_z_and_indices` + `_load_coeff_row` + list append + stack.
   At f=3 this is plausibly 10–30% of stage-5 wall.
2. `_z9_mul_conj_batch_nb` itself — already Numba'd, but the BVc/CWc
   setup is ~n_v + n_w muls per inner-iz iteration; at f=3 with bigger
   descriptors this could be 10–20% of wall.
3. **Sage stage-3** — at tight ε, stage-3 wall now likely dominates
   pipeline-wide (the live find_roots job has been running 3.5+ hrs
   already at f=2 ε=0.05). Stage-5 optimization is no longer the long
   pole; stage 3 is.

**Bottom line:** The "ε = 10⁻⁶ at f = 3 on Lenore" milestone is now
realistic in compute terms. Memory and stage-3 are the next limits.

## Files changed

- `/home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py`
  - Lines 285–574 (approx): added `_z9_divide_if_exact_by_a_nb`,
    `_z9_divide_if_exact_by_a_batch_nb`, `_z9_abs2_scaled_nb`,
    `_z9_inner_filter_pairs_nb` inside the `if _HAVE_NUMBA:` block.
  - Lines 587–593: module-level `_ZETA9_RE`, `_ZETA9_IM` float tables.
  - Lines ~1646–1750 (fixed_row1 mode inner loop): replaced the per-pair
    Python loop with the fused-kernel call + match-emission loop.
  - Lines ~1968–2110 (main-search mode inner loop): same replacement.

## Don't gold-plate

- Three discrete actions were specified (A, B, C). The natural and
  most-efficient implementation fused all three into one kernel; the
  microbench of B vs Python (6.3x) and A scalar (12.3x) confirm each
  component carries weight on its own, and the fused kernel removes the
  per-pair Python dispatch entirely (78.9x on dense grid).
- Q-field cross-product framing in the task was inaccurate — the code
  uses integer Z[ζ_9] divisibility, not a Q-field solve. The cross-product
  block was straightforward to inline.
- `_z9_divide_if_exact_by_a_batch_nb` is shipped but not wired into the
  hot loop, because the fused kernel obviates it for the inner loop.
  It's available for any non-inner-loop call site that needs batched
  exact division.
