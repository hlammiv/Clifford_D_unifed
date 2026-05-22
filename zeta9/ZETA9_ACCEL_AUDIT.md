# zeta9 acceleration audit — 2026-05-12

Goal: move zeta9's ε ceiling from ~10⁻⁵ toward 10⁻⁶ / 10⁻⁷. Each order of magnitude
buys one SK recursion level saved (~5× on N_D / word length).

This document is an inventory plus a ranked action list. It deliberately stays in
the constant-factor regime (no rewrites, no SK build) — those are tracked elsewhere.

---

## Implementation status (2026-05-12 afternoon)

Actions #1, #2 (was #6 in audit numbering), and #3 from the TL;DR were
implemented. Verdict: **all three landed cleanly with correctness preserved,
but realized speedups are smaller than the audit predicted.** The full
report and benchmarks live in `ZETA9_OPT_RESULTS.md`.

Headline finding: **batching z9_mul wins 23× in microbench but only ~3 % on
stage 5 e2e**, because `z9_divide_if_exact_by_a` (4.5 us/call) is now the
inner-loop bottleneck. Audit item #4 ("Numba-ize the float-arith gates")
moves to highest priority as the gate to ε≤10⁻⁶ feasibility.

## Implementation status (2026-05-12 evening — Round 2)

Action #4 (Numba-ize the migrated bottlenecks) was implemented as a single
**fused inner-loop kernel** `_z9_inner_filter_pairs_nb` that covers:
  - `z9_divide_if_exact_by_a` (exact divide, the 4.5 us/call bottleneck)
  - `coeff_row_to_complex_scaled + abs(zu) > eps` (float gate)
  - cross-product divisibility (three N_i = B*W - C*V style sub-products
    each tested mod 3^f)

**Result: stage-5 e2e f=2 ε=0.05 θ=1.5 went from 164.21 s → 7.82 s
(21× speedup) with byte-identical counters.** Microbench shows 12-13× on
the scalar divide kernel, ~6× on the abs gate, and 78× on the fused-kernel
e2e over a 200×200 (V, W) grid.

This **clears the f=3 ε ≈ 10⁻⁶ wall** on Lenore compute-wise. The next
limit is Sage stage-3 ideal-factorization wall, not stage-5. Full details
in `ZETA9_OPT_RESULTS.md` (Round 2 section).

---

## TL;DR ranked action list

| # | Action                                                                              | Effort     | Expected gain                              | Where it helps | Status |
| - | ----------------------------------------------------------------------------------- | ---------- | ------------------------------------------ | -------------- | ------ |
| 1 | **Stage 5 batched z9_mul kernel** (`_z9_mul_conj_batch_nb`)                         | ~80 LOC    | 5-15× on stage-5 inner loop                | stage 5        | **DONE 2026-05-12**: 23-25× on inner muls (microbench), but ~3% e2e because `z9_divide_if_exact_by_a` is now the bottleneck. See ZETA9_OPT_RESULTS.md. |
| 2 | **Default `--check_local_p3k` ON for ε ≤ 1e-4 in `zeta9_compile.py`**               | ~5 LOC     | ~10× pruning of Y-triples → stage 3 + 5    | stages 1,3,5   | DONE (wrapper plumbing, see Part B at end) |
| 3 | **Pre-bind raw `_z9_*_nb` kernels at top of stage-5 hot loop**                      | ~15 LOC    | 1.3× on stage 5 (wrapper overhead removal) | stage 5        | **DONE 2026-05-12**: 1.42× on inner-loop simulation; subsumed by Action 1 in the V/W hot path. See ZETA9_OPT_RESULTS.md. |
| 4 | **Numba-ize `coeff_row_to_complex_scaled` / `matrix_frobenius_dist`**               | ~30 LOC    | 1.5-3× on the |zu|≤ε + Frobenius gates     | stage 5        | **DONE 2026-05-12 (Round 2)**: fused into `_z9_inner_filter_pairs_nb` along with the exact-divide and cross-product divisibility. Stage-5 e2e: 164 s → 7.8 s (21×). See ZETA9_OPT_RESULTS.md Round 2. |
| 5 | **Cache complete `desc_v` lookups by (Y, target) bytes, not Python tuples**         | ~10 LOC    | 1.1-1.5× (currently hashes tuple of ints)  | stage 5        | |
| 6 | **Sage `--proof=False` everywhere (or pari_nfinit only)**                           | ~20 LOC    | 1.5-3× on stage 3 per-Y solve              | stage 3        | **DONE 2026-05-12**: module-level `proof.number_field(False)` in roots.py. Actual stage-3 gain 1-3 %, not 1.5-3× (audit overestimated). See ZETA9_OPT_RESULTS.md. |
| 7 | **Sage ideal cache across Y triples (LRU keyed on `(m0,m1,m2)` mod ideal class)**   | ~50 LOC    | depends, plausibly 2× stage 3              | stage 3        | |
| 8 | **Better MPI load balancing for stage 5** (`chunk_idx % size` → work-stealing)      | ~40 LOC    | 2-4× when n_chunks < n_ranks               | stage 5        | |
| 9 | **f-power sweep: validate 0.05 → 0.005 → 0.0005 → 0.00005 plan with RAM probe**     | analysis   | data for next leg, not a speedup           | planning       | |

How much closer to ε = 10⁻⁶ does the full list get us?
The "ε ceiling" is set by walltime / RAM at the next f-level. Items 1-3 are local
to stage 5 — they shrink walltime for a single cell; the f-jump from 2 → 3 → 4 (V-
denom) costs another ~10-50× in stage-5 row-count. Items 1 + 2 stacked give an
estimated 50-150× on the per-cell stage-5 wall while the upstream Y-triple universe
shrinks ~10×. Net: an ε = 1e-5 cell that today takes ~60 min is plausibly ~1-3 min
after, which makes an ε = 1e-6 sweep at f=3 feasible (~30 min/cell × 41 angles).
ε = 1e-7 is still gated by item #9: stage-2 join-bucket RAM at f=4 will likely
exceed Lenore's 15 GB.

---

## Stage-by-stage inventory

The five-stage pipeline as exercised by `unified/zeta9_compile.py`:

| Stage | Driver                                                              | Cost driver                          | Numba? |
| ----- | ------------------------------------------------------------------- | ------------------------------------ | ------ |
| 1     | `zeta9.collect_targets` (MPI, 2 runs for u=0,1)                     | Polygon scan + ideal sieve           | yes (line 4 `import numba as nb`) |
| 2     | `zeta9.select_triples_optimized` (MPI)                              | 3-way hash join + I/O                | no, pure NumPy/I/O (fine) |
| 3     | `zeta9.find_roots_exact_v2` (MPI Sage)                              | Sage ideal factorization per Y       | no (Sage-bound) |
| 4a    | `zeta9.build_phase_sidecar_binned_mpi`                              | Mostly I/O                           | no |
| 4b    | `zeta9.build_triple_chunk_metadata`                                 | I/O                                  | no |
| 5     | `zeta9.search_diagonal_matrix_two_rows_streamed_mpi`                | z9_mul-heavy nested loop             | **yes (already wired)** |

Source line counts (informational):
- `search_diagonal_matrix_two_rows_streamed_mpi.py`: 1922 (the stage-5 hot path)
- `roots.py`: 2097 (stage 3 internals)
- `select_triples_optimized.py`: 658 (stage 2)
- `collect_targets.py`: 1262 (stage 1)
- `find_roots_exact_v2.py`: 668 (stage 3 driver)

---

## 1. Stage-by-stage runtime profile

### Live process on Lenore at audit time
PID 856055 (mpirun) → 4 ranks of `find_roots_exact_v2.py` on `f=2_eps=0.05`.
ps shows ~94% CPU for ranks 1-3 and ~3% for rank 0 (the coordinator), elapsed ~3h.
Stage-2 final file `D/TM_f=2_eps=0.05` is absent at audit time (only `.parts/` and
`.join_buckets/` are present), so this 3-hour job is **stage 2 still running**, not
stage 3 as previously assumed. The wrapper had not yet entered stage 3.

This single observation is the most important finding in the audit:
**stage 2 (select_triples), not stage 3 (find_roots), is the dominant wall on the
current cell.** Memory note `zeta9_perf_audit_2026-05-12.md` claimed bucket cache
caps were harmful and "set big or off" — the current wrapper has bucket_cache_entries=64
and n_join_buckets=8192. With ~15 GB RAM total and Sage already resident, that
may still be RAM-constrained on Lenore.

### Representative-cell profile (recommended next probe, do NOT run while find_roots
is alive — see co-tenancy at end)

```bash
# All artifacts already exist for f=2 eps=0.025 (May 7 cache):
ls D/{Y1_f=2_u=0_eps=0.025.npy,Y1_f=2_u=1_eps=0.025.npy,TM_f=2_eps=0.025,RM_f=2_eps=0.025_local.roots.json}

# Then run stage 5 in isolation with cProfile:
/home/hlamm/miniforge3/envs/sage/bin/python -m cProfile -o /tmp/stage5.prof \
  zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py \
  --triples_file D/TM_f=2_eps=0.025 --triples_json D/TM_f=2_eps=0.025.manifest.json \
  --rootdb_prefix D/RM_f=2_eps=0.025_local --f 2 \
  --theta 1.5708 --eps 0.05 --output_prefix /tmp/stage5_probe \
  --row2_cache /tmp/stage5_probe.row2.npy \
  --chunk_meta_json D/RM_f=2_eps=0.025_triples_chunk_meta.json \
  --max_matches 100 --quiet
```

Estimated walltime: stage 5 at f=2 with these tighter caches and existing artifacts
should be a few minutes single-rank. Use `python -m pstats /tmp/stage5.prof` →
`sort cumtime` → `stats 30` to confirm `z9_mul`/`z9_conj`/`matrix_frobenius_dist`
dominate. (This audit is sized off the live profile artifact at `xout.profile.json`
and the codebase, not a fresh probe — see "Don't oversubscribe" below.)

### Inner-loop micro-benchmark (executed during audit)

Test machine: this Lenore terminal at audit time, before the find_roots job started
saturating cores (it's still saturating now but the bench was single-threaded and
respected co-tenancy).

```
== JIT-ON ==
  z9_mul       :   1.075 us/call    (52.6× over JIT-OFF)
  z9_conj      :   0.835 us/call    ( 1.6× over JIT-OFF)
  z9_norm_m012 :   0.691 us/call    (85.5× over JIT-OFF)
  z9_divis_int :   0.392 us/call    ( 1.0×)

== JIT-OFF ==
  z9_mul       :  56.530 us/call
  z9_conj      :   1.370 us/call
  z9_norm_m012 :  59.070 us/call
  z9_divis_int :   0.413 us/call

== raw _z9_mul_nb (bypass wrapper) ==
  _z9_mul_nb (raw)   :   0.814 us/call   (wrapper overhead ≈ 0.23us / 22%)
  _z9_conj_nb (raw)  :   0.572 us/call
  _z9_norm_nb (raw)  :   0.433 us/call
```

Interpretation:
- The Numba kernels (item from memory note `zeta9_perf_audit_2026-05-12.md`) are
  **already implemented** in `search_diagonal_matrix_two_rows_streamed_mpi.py`
  lines 109-161. The audit memo is stale — that work was done.
- Per-call overhead of ~0.2-0.3 us comes from `np.ascontiguousarray(..., dtype=np.int64)`
  in the wrapper at lines 167-169, 187, 211-212, 255. Each stage-5 inner-loop iter
  does ~6 calls × ~0.25us = 1.5us of pure wrapper churn.
- Remaining time per call is Numba JIT dispatch overhead (~700 ns floor). The only
  way past that floor is **batching** — call `_z9_mul_nb` once per N pairs, not once
  per element.

---

## 2. The Numba z9_mul opportunity → **already done; remaining wins are batching**

`search_diagonal_matrix_two_rows_streamed_mpi.py` lines:
- 104  `_POWER_TABLE = np.vstack([z9_reduce_power_coeff(k) for k in range(11)]).astype(np.int64)`
- 109-113  `_HAVE_NUMBA` import guard
- 116-131 `_z9_mul_nb` — @nb.njit(cache=True, inline="always")
- 133-142 `_z9_conj_nb`
- 144-149 `_z9_divisible_by_int_nb`
- 151-161 `_z9_norm_m012_nb`
- 164-181 `z9_mul` wrapper (dispatches to `_z9_mul_nb` if available)
- 184-199 `z9_conj` wrapper
- 252-267 `z9_norm_m012` wrapper

**Status: Numba is wired in, called from the hot inner loops at lines 1166-1211
(fixed_row1 mode) and 1436-1482 (main streamed mode), and verified to be active
in the live sage env (numba 0.62.1).**

The memory note `zeta9_perf_audit_2026-05-12.md` predicted "20-100× stage 5";
microbench confirms 53× on z9_mul and 85× on z9_norm_m012. So this is captured.

### What remains: two real opportunities

**(a) Skip the wrapper in the hot loop.** Each call to `z9_mul(a, b)` does
`np.ascontiguousarray(...)` twice (lines 167-168). For inputs that are already
contiguous int64 arrays (A, B, C, U, V, W in stage 5 already are), this is wasted
work. Pre-bind the raw kernel at the top of the hot loop:

```python
# Inside search_diagonal_two_rows_streamed, right before the main while-loop:
_mul = _z9_mul_nb        # local capture (Python attr-lookup cost saved too)
_conj = _z9_conj_nb
_pt = _POWER_TABLE
# ... then in the inner loop:
BVc = _mul(B, _conj(V), _pt)   # replaces z9_mul(B, z9_conj(V))
```

Expected gain: 1.05us → 0.81us per z9_mul call, ~22% on the kernel; given there
are ~6 such calls per inner iter, expected ~15-20% stage-5 wall reduction. Cheap
to implement (~15 LOC of mechanical substitution in lines 1166-1211 and 1436-1482).

**(b) Batched pairwise mul.** The inner loop computes:
- `BVc = z9_mul(B, conj(V))` for each V in `desc_v` roots
- `CWc = z9_mul(C, conj(W))` for each W in `desc_w` roots
- Then for each (V, W): `T = BVc + CWc; Q = solve_for_U(T); ...`

This is exactly the pattern that wants `_z9_mul_left_nb(B_left_matrix, V_arr)`
where `V_arr` is shape (n_v, 6) and the output is (n_v, 6). A single Numba
invocation amortizes JIT dispatch over all V's. Sketch:

```python
@_nb.njit(cache=True)
def _z9_mul_batch_left_nb(left_M, V_batch, out):
    # left_M: (6,6) precomputed M(B) such that M(B)@v == B*v
    # V_batch: (n, 6)
    # out: (n, 6)
    n = V_batch.shape[0]
    for k in range(n):
        for i in range(6):
            s = 0
            for j in range(6):
                s += left_M[i, j] * V_batch[k, j]
            out[k, i] = s
```

Then in stage 5:
```python
B_left = _z9_mul_matrix_left_arr(B)  # 6x6 left-mul matrix for B
V_arr = np.stack([V for V, _ in roots_v])
BVc_arr = np.empty_like(V_arr)
_z9_mul_batch_left_nb(B_left, V_arr, BVc_arr)
# Then iterate over BVc_arr instead of calling z9_mul per V
```

Expected gain: 0.8us/call → ~0.1us/element amortized → **8× on the inner mul
hot spot**. Combined with the wrapper-skip above, plausibly 5-15× on the
end-to-end stage-5 wall. Effort: ~80 LOC. This is action #1 in the TL;DR.

Note: applying the same pattern to `z9_mul_matrix_left` itself (line 224) costs
1 inv_M_A inversion per A, which is fine — already amortized over the (V, W)
inner loop.

---

## 3. Sage / ideal-factorization in stage 3

`find_roots_exact_v2.py` → `solve_assigned_Ys_exact` → `actual_roots_from_ideal_search`
in `roots.py:1559`.

The Sage work per Y (m0, m1, m2):
1. `analyze_M` (line 272 in roots.py) — embeds M into F=Q(α), builds the ideal
   `F.ideal(M)`, factors it via `factor_ideal_compat` (line 28).
2. `enumerate_candidate_ideals` — local choice over each prime above 3 / Galois.
3. For each candidate ideal:
   - `principal_generator_compat` — proves principal + extracts a generator.
   - `lift_K_to_F_if_possible` — checks `ratio ∈ F`.
   - `UnitNormSolver.solve` (cached) — unit-correction via log lattice.
4. Optional legacy fallback (line 1614) when no roots found.

Bottleneck: **per-Y Sage overhead** (ideal construction + factorization + principal
generator extraction). Each Y costs O(seconds), and stage 3 processes O(10⁴-10⁶)
unique Y's. Weighted partition by `Y_weight` (line 43) gives MPI balance but the
per-call cost is the floor.

Fast paths worth probing:
- `F.ideal(M).factor(proof=False)` — Sage's default is proof=True for class
  groups, which is way more expensive than we need. The audit could not find
  an explicit `proof=` keyword on `factor_ideal_compat` calls in roots.py (grep
  for `proof=` only shows it on `principal_generator_compat` line 1448). Adding
  `proof=False` to `factor_ideal_compat` and downstream is action #6 (~20 LOC).
- PARI direct: `pari(M).nffactor(...)` or `pari(...).nfeltdiveuc(...)`. Sage's
  high-level wrappers add `Element` boxing per intermediate. Going through
  `pari.nfinit(F.polynomial())` once and reusing for factorization across all
  Y's may save 30-50% per call. Effort: ~80 LOC, plus dependency on PARI version
  compatibility. Defer until item #6 is benchmarked.
- **Ideal cache across Y's.** The class group of F = Q(α) is finite (computed
  once); many Y values will produce the same ideal class. Cache `(m012) → roots`
  in `~/.zeta9_root_cache.sqlite` so reruns and Galois orbits hit cache. The
  existing `--global_rootdb_prefix` already does roughly this at the npz level,
  but only across pipeline runs, not within a run for sibling Y's. Action #7.

What this audit cannot do without burning a Sage-instrumented run: confirm which
of `factor()` / `principal_generator()` / `solve(ratioF)` dominates. The right
profile is to run `find_roots_exact_pipeline` for 100 Y's with `cProfile` and look
at cumtime per Sage function. **Do not do this while the live job is running.**

---

## 4. --check_local_p3k integration

The flag is wired through `collect_targets.py` (lines 715, 877, 885, 894, 901,
1104, 1123, 1175, 1233, 1252) and `roots.py` (lines 345-358, 450-466, etc).

Definition at `roots.py:1016` — `quick_local_p3k_norm_screen(m0, m1, m2)`. Default
`local_p3k=2`. The screen rejects Y's whose norm isn't in the local image mod 3^(2k);
~10× pruning when k=2 per memory note `zeta9_p3k_screen.md`.

**Current wrapper (`unified/zeta9_compile.py`) does NOT pass `--check_local_p3k`
to stage 1.** Adding it conditionally is trivial:

```python
# In zeta9_compile.py around line 115 (stage 1 cmd construction):
cmd = mpirun + [py, "-m", "zeta9.collect_targets",
                "--f", str(f), "--norm", "2", "--u", str(u),
                "--eps", label, "--eps_bin_width", "0",
                "--output", str(out.with_suffix("")),
                "--resume"]
if args.epsilon <= 1.0e-4 or args.epsilon <= eps_pre <= 1.0e-4:
    cmd.append("--check_local_p3k")
```

Caveat from memory note: caches at different `--check_local_p3k` settings are
**not interchangeable** (the cache won't include p3k-pruned Y's). So toggling the
flag invalidates `Y1_f=*_u=*_eps=*.npy` cache files. The wrapper should add
`_p3k` to the artifact name when the flag is on, or simply force a non-resume
when the flag toggles. Easiest: filename-tag.

This is action #2 — ~5-20 LOC, requires one validation run at ε ≤ 1e-4 to confirm
the ~10× pruning carries through to stage 5 success rate. **Validate at ε=1e-4
before pushing to ε=1e-5/1e-6.**

**Post-hoc validation note (2026-05-12):** running `quick_local_p3k_norm_screen`
as a *post-filter* on the existing `Y1_f=2_u=0_eps=0.025.npy` (23 Ys) and
`Y1_f=2_u=1_eps=0.025.npy` (226 Ys) shows **100% pass** — these arrays are
already-stage-1-passed Ys, so the screen has nothing more to reject. The screen
matters as an *in-loop* filter inside `collect_targets`'s enumeration; it cuts
the candidate count *before* the more expensive `quick_screen_M` runs. The
~10× speedup claim from memory note `zeta9_p3k_screen.md` is upstream-of-stage-1
output. We cannot empirically confirm the 10× factor without running stage 1
itself at tight ε, which is RAM-expensive.

---

## 5. f-power scaling

Empirical Y1 universe (from D/ directory listing):

| f | u | eps    | Y1 file size | rows (est)         |
| - | - | ------ | ------------ | ------------------ |
| 2 | 0 | 0.05   | 6.5 KB       | ~270               |
| 2 | 1 | 0.05   | 21 KB        | ~870               |
| 2 | 0 | 0.025  | 0.7 KB       | ~30                |
| 4 | 0 | 0.0025 | 5.6 MB       | ~230 000           |
| 4 | 1 | 0.0025 | 96 MB        | ~4 000 000         |
| 4 | 0 | 0.025  | 62 MB        | ~2 600 000         |
| 4 | 1 | 0.025  | 96 MB        | ~4 000 000         |

The Y1 universe grows ~10⁴× from f=2 → f=4 even at the same ε. At f=6 (V-denom
3¹³ ≈ 1.6 M) and ε=1e-6 the per-coord box is ~10⁻⁶ × 3⁶ ≈ 7×10⁻⁴ — naively
the Y count scales as (1/ε)³ × box volume, so a rough projection:

- f=4 ε=0.0025 Y1 = 96 MB
- f=6 ε=2.5e-4 Y1 ≈ 96 MB × (10×) × small box factor ≈ 1-5 GB per u
- f=6 ε=2.5e-5 ≈ 10-50 GB per u  → **exceeds Lenore's 15 GB RAM**.

Conclusion: with current ε-binning, **f=5 is feasible on Lenore (ε ~ 10⁻⁶), f=6
needs more RAM or staged streaming + `--check_local_p3k`.** That's why item #2
matters: p3k pruning before Y1 is materialized would shave 10× off these sizes.

Stage-2 join-bucket dir at f=4 ε=0.0025 is 627 KB metadata over ~2 GB of bucket
.bin files (visible via `D/TM_f=4_eps=0.0025.join_buckets/`). At f=5 these would
be ~20-50 GB unless n_join_buckets is upped from 8192. The current wrapper hard-
codes 8192; for f≥5 that should be argparse-controlled.

---

## 6. Stage 5 V-filter / bucket cache

Stage-2 bucket cache is controlled in the wrapper (line 141-142):
```python
"--bucket_cache_entries", "64",      # default 16; we have RAM headroom
"--n_join_buckets", "8192",          # default; was 32768
```

These reverted from earlier-tighter caps (4 / 32768) which caused 0.01% hit rate
and disk thrashing — per memory note `zeta9_perf_audit_2026-05-12.md` and the
inline comment.

64 cache entries × ~tens of MB per bucket at f=4 ≈ ~1-2 GB. RAM budget on Lenore
(15 GB total, ~7 GB available with co-tenant load + Sage):
- Sage process: ~1-2 GB resident per rank
- 4 ranks × ~1 GB stage-2 state = 4 GB
- bucket cache ≈ 1-2 GB
- **Total ~7-9 GB peak; fits but tight.**

For stage 5: the row2_cache `.row2.npy` is a per-run cache and can be tens of MB.
Not RAM-binding unless we go to f≥5.

Recommendation: at f≥5, raise `bucket_cache_entries` to 128 only if `free -g`
shows > 8 GB available before launch. The wrapper should probe this rather than
hardcode 64.

---

## 7. Lenore-specific tuning

Snapshot at audit time (`uptime` + `free -g`):
- nproc = 20
- load avg = 3.28 / 6.89 / 16.07 (1/5/15 min)
- Memory: 15 GB total, 7 GB used, 2 GB free, 6 GB buff/cache, **7 GB available**
- Swap: 3 GB total, 1 GB used
- Find_roots ranks 1-3 at 93-94% CPU; rank 0 at 3% (coordinator)

Per co-tenancy rule from memory note `feedback_lenore_cotenancy.md`:
```
safe_parallelism = (nproc - existing_load) / per_cell_cores
                 = (20 - 4) / 4   # 4 ranks × ~1 cell core each
                 = 4 ranks
```

**Safe to launch alongside live job: up to 4 additional ranks.** Do NOT use the
`--oversubscribe` flag for these; the find_roots job already uses it.

Suggested probe (does not interfere):
```bash
mpirun -n 4 /home/hlamm/miniforge3/envs/sage/bin/python \
  /home/hlamm/Desktop/efficent_gates/unified/zeta9/zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py \
  --triples_file D/TM_f=2_eps=0.025 \
  --triples_json D/TM_f=2_eps=0.025.manifest.json \
  --rootdb_prefix D/RM_f=2_eps=0.025_local \
  --f 2 --theta 1.5708 --eps 0.05 \
  --output_prefix /tmp/probe \
  --row2_cache /tmp/probe.row2.npy \
  --chunk_meta_json D/RM_f=2_eps=0.025_triples_chunk_meta.json \
  --max_matches 100 --quiet
```

Memory budget for this probe ≈ 4 × 300 MB ≈ 1.2 GB, well within 7 GB headroom.

---

# Part B — implementation: action #2 (`--check_local_p3k` wired into wrapper)

The audit's TL;DR identified action #2 as the highest-impact-easy item once
the Numba kernels turned out to be already implemented:

- Action #1 (batched z9_mul) — biggest perf gain (5-15× on stage 5) but ~80 LOC
  and requires correctness re-validation. Tomorrow's work.
- Action #2 (`--check_local_p3k` wiring) — ~10× *pruning* of the upstream Y
  universe at tight ε. Smaller code change (~25 LOC of wrapper plumbing), but
  cache-tag-invalidates, so existing artifacts at the same `eps_pre` and
  without `_p3k` remain valid for unfiltered runs.
- Action #3 (pre-bind raw kernels) — 1.3× but 15 LOC and zero risk. Worth doing
  alongside #1.

Action #2 was chosen for this implementation because:
1. Higher leverage: 10× pruning of stage-1 output → 10× smaller stage-2 join →
   10× smaller stage-3 Y universe → smaller stage-5. This stacks across stages.
2. Required for ε ≲ 10⁻⁵ per memory note `zeta9_p3k_screen.md`; the flag exists
   in the stage-1 driver already and is unused by the wrapper.
3. Doesn't touch the stage-5 hot path → no risk of breaking the live job.
4. Smaller LOC + named cache namespaces preserve all current `D/` artifacts.

## Changes applied

File: `/home/hlamm/Desktop/efficent_gates/unified/zeta9_compile.py`

1. `artifact_paths(workdir, f, eps_pre, check_local_p3k=False)` — new keyword
   adds `_p3k` suffix to all stage-1..stage-4 filenames when True. The global
   rootdb (`RM_f={f}_global`) is unchanged so its cache reuse stays sound for
   both modes.
2. `ensure_precompute(...)` — same flag, conditionally appends `--check_local_p3k`
   to the stage-1 mpirun command.
3. `run_stage5(...)` — same flag, picks `_p3k`-tagged artifacts when set.
4. CLI: `--check-local-p3k` / `--no-check-local-p3k` (defaulting to None →
   auto-enable iff `epsilon <= 1e-4`).

Diff summary: +27 lines, -7 lines across one file.

## Validation

Dry-run sanity:
```
$ python zeta9_compile.py --theta 1.5708 --epsilon 0.05 --max-f 1 --dry-run
[zeta9_compile] ... check_local_p3k=False
[zeta9_compile] stage1 (u=0) cached: D/Y1_f=2_u=0_eps=0.025.npy           # unchanged
$ python zeta9_compile.py --theta 1.5708 --epsilon 1e-5 --max-f 2 --dry-run
[zeta9_compile] ... check_local_p3k=True
[zeta9_compile] stage1 u=0: ... --output D/Y1_f=4_u=0_eps=5e-06_p3k --resume --check_local_p3k
                                                            ^^^^                  ^^^^^^^^^^^^^^^^^
```

Existing-cache compatibility: tested `--epsilon 0.05 --dry-run` and confirmed
all stages still see the May-7 cached artifacts (`Y1_f=2_u=0_eps=0.025.npy`,
`TM_f=2_eps=0.025`, `RM_f=2_eps=0.025_local.roots.json`, etc.) without `_p3k`
suffix — existing reruns at coarse ε pay zero penalty.

## Before/after benchmark

I cannot safely run a stage-1 benchmark at tight ε while the live `find_roots`
job is consuming ~4 cores and ~1 GB RAM. The empirical 10× pruning claim comes
from memory note `zeta9_p3k_screen.md` (recorded 2026-05-11). A post-filter
benchmark on the existing `Y1_f=2_u=*_eps=0.025.npy` arrays shows 100% pass —
expected, because those arrays are post-stage-1 output (already passed the
parallelogram strip and inert-prime sieve). The screen prunes upstream of
that output, inside `collect_targets`'s candidate enumeration.

**Empirical validation deferred until the live find_roots job finishes** (or
on a different host). Recommended probe: `--theta 1.5708 --epsilon 1e-4 --max-f 2`
twice — once with `--no-check-local-p3k`, once with `--check-local-p3k` — and
compare stage-1 walltime + Y1 file size.

## Honest assessment: how much closer to ε=10⁻⁶?

- **One concrete order of magnitude unlocked.** Without the p3k screen, our
  empirical ε ceiling on Lenore is ~10⁻⁵ (per `zeta9_perf_audit_2026-05-12.md`),
  because stages 1-2 RAM-blow at f=4 ε=2.5e-5. With p3k cutting the Y universe
  10×, ε=10⁻⁵ at f=4 becomes feasible RAM-wise, and ε=10⁻⁶ at f=5 looks
  *probably* feasible (still need n_join_buckets re-tuning).
- **No measured speedup yet** — this is constructive plumbing; the actual gain
  is verified by stage 1 runtime when the live job is gone.
- **Not enough for ε=10⁻⁷.** That needs items #1 (batched z9_mul, ~5-15× on
  stage 5) AND #6 (Sage proof=False, 1.5-3× on stage 3) AND the bigger n_join_buckets.

## What's next after this

In priority order, after the live find_roots job completes:
1. Empirical p3k validation at ε=1e-4 (confirm the 10× claim end-to-end).
2. Action #1: batched z9_mul kernel (estimated 5-15× stage 5; ~80 LOC).
3. Action #3: pre-bind raw kernels in stage-5 inner loop (1.3× stage 5; ~15 LOC).
4. Action #6: `proof=False` on all Sage ideal factorizations in `roots.py`.
5. Then re-bench at ε=1e-6 to see if we cleared the next order of magnitude.
