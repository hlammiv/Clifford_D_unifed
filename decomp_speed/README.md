# decomp_speed — fast, bit-identical Clifford+D decomposition

`fast_decompose.decompose_fast(V, ...)` is a drop-in replacement for
`hrsa/canonical_reducer.decompose_canonical` (same signature, same returned
dict: syllables, D_count, trailing_clifford, n_iter, error, ...).  No
existing repo file is modified.

## Use

```python
import sys; sys.path.insert(0, "/home/hlamm/Desktop/efficent_gates/unified/decomp_speed")
import fast_decompose as fd
r = fd.decompose_fast(V)          # V: 3x3 of cr._FastZ9Frac (ingest_decompose.build_ring)
fd.install()                      # or: monkey-patch cr.decompose_canonical for existing callers
```

`install()` makes `nick_test/nick_tcost_all.work` and
`nick_request/analyze_topk.work` use the fast path (patch before forking
workers).  Ready-made launcher with the identical CLI:

```
python3 analyze_topk_fast.py ../nick_request/topk_run/samples/fits_topk_f=8.sample30.txt \
    --k 100 --procs 2 --colmajor --out ../nick_request/topk_run/fast_f8
```

Requires numpy + numba (numba kernels are cached on disk in `__pycache__`;
first call in a fresh process ~0.5 s).  Without numba, or when coefficients
exceed the int64 guard (max|coef| ≥ 7e14 ≈ 3^31, i.e. f ≳ 30), it falls back to
exact numpy/Python-int object arrays (same results, slower).

## Where the time went (original, cProfile)

| f  | wall   | double-prefix calls | time in `_try_double_prefix` |
|----|--------|---------------------|------------------------------|
| 8  | 189 s  | 16 (of 28 iters)    | 169 s (90%)                  |
| 16 | 422 s  | 30 (of 56 iters)    | 382 s (91%)                  |

Inside it: `sde_chi_z9` / `_z9_sde_chi` / `_formal_derivative_coefs` ≈ 60%,
`_FastZ9.__mul__` + `_reduce_9_fast` ≈ 25%.  The double prefix is the
common case, not a rare fallback (more than half of all peel steps): each
call evaluates up to 4374² candidate pairs in Python.  Table build is only
~4 s here; the single-prefix scan and the apply step are minor.

## What changed (algorithmically identical)

1. **No prefix table.** P = H·D(a)·R^ε·X^δ, so (P·V)[0][0] = (c/3)·Σ_j s_j ζ^{a_j} V[π_δ(j)][0],
   c = −1−2ζ³.  All 4374 candidate numerators are signed sums of a 9×3×6 table
   {ζ^a·n_k}.  Only the winning P is built (memoised, via the original
   `_build_prefix`) for the apply step.
2. **Closed-form sde_χ.** The original `sde_chi_z9` recursion is exactly the
   χ-adic valuation (χ = 1−ζ₉): with b = Taylor coefficients of x at ζ = 1,
   v(x) = min_i (6·v₃(b_i) + i).  sde_chi_full = 6f − v (f ≥ 1 always holds
   for candidate entries; zero → 0, as in the original).
3. **Double prefix as a numba kernel** that literally transcribes
   `_try_double_prefix`'s loop (order, pruning `k1+k2 ≥ best`, break at 0),
   with the inner test sde < s replaced by "Taylor coefficient b_i divisible
   by 3^⌈(T−i)/6⌉, T = 6(f+2)−s+1" (early exit on the first coefficient).
   All 4374 outer column-0 vectors are precomputed in one numpy pass.
4. Single-prefix scan likewise in numba; selection (`_SELECTION` "d" or
   "tcost", first-minimal index; greedy mode's tie/break rules) reproduced
   exactly.  Apply step (`_prefix_times_V`, `_reduce_by_three`), final
   `sde_chi_full` and `classify_monomial_and_d_cost` are the original functions.

## Verification

* `test_kernels.py`: for real V's at f = 4, 8, 12 (several peel steps), the fast
  new-sde vector equals the original `sde_chi_full(_prefix_times_V_00(P,V))`
  for all 4374 prefixes (int64, object and numba paths), and the double-prefix
  result (idx1, idx2, mid_s) equals `_try_double_prefix` exactly (also at a
  perturbed s).
* `test_greedy.py`: `greedy_single=True` and `skip_double=True` give identical
  results on 9 matrices (f = 4, 6, 8) → 18/18 agree.
* `verify.py`: 49 matrices, f = 4…16 (per f: 4 from `nick_test/fits_f=*.txt`,
  2 top-K samples read `--colmajor`, 1 top-K sample with `--select tcost`),
  run through `nick_tcost_all.work`.  Compared: full syllable list, D_count,
  n_iter, success/error, exact trailing Clifford, and the work() record
  (N_D, n_T3, n_L4, n_R_syl, n_R_resid, n_R, Tcost, epsilon).
  **49/49 bit-identical, 0 mismatches** (`python3 verify.py --compare orig_all.jsonl fast.jsonl`).
  Two f=16 matrices were run twice with the original (deterministic, identical).

## Speed (single core, busy shared machine, through `nick_tcost_all.work`)

| f  | n | original s/matrix | fast s/matrix | speedup |
|----|---|-------------------|---------------|---------|
| 4  | 7 | 34.2  | 0.18 (incl. 0.5 s numba warm-up on the 1st) | 186× |
| 6  | 7 | 56.2  | 0.17 | 335× |
| 8  | 7 | 68.1  | 0.24 | 282× |
| 10 | 7 | 77.8  | 0.31 | 249× |
| 12 | 7 | 100.6 | 0.35 | 284× |
| 14 | 7 | 81.8  | 0.75 | 109× |
| 16 | 7 | 125.0 | 1.18 | 106× |

Bare `decompose_fast` (no work() overhead), warm process: f=4 ≈ 0.05 s,
f=8 ≈ 0.15 s, f=12 ≈ 0.25 s, f=16 ≈ 0.5 s.  The remaining time is ~50% the
double-prefix numba kernel, the rest the (original) big-int apply step and
numpy glue.

Note: the machine had load ≈ 15/20 during all runs; absolute times are
inflated, ratios are what matter.

## C++ decompose_tool

Not used: the Python fast path is already ~100–300× faster than the original
and well below the C++ subprocess/JSON overhead regime, so the persistent-mode
and cpp_int questions were not pursued.

## Files

* `fast_decompose.py` — the implementation (`decompose_fast`, `install`)
* `analyze_topk_fast.py` — analyze_topk launcher using the fast path
* `verify.py` — matrix set, orig/fast runners, `--compare`
* `test_kernels.py`, `test_greedy.py` — kernel / mode-level equality tests
* `profile_orig.py`, `prof_*.log` — original profiles; `prof_fast.py`
* `orig_*.jsonl`, `fast.jsonl` — verification dumps
