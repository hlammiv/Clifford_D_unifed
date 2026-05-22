# Householder Stage 5 — Design (2026-05-13)

## 1. Survey of the original zeta9 codebase

**The original zeta9 does NOT contain a Householder stage 5.** The shipped stage-5 scripts are all "row-of-unitary" searches:

| File | Role |
|---|---|
| `zeta9/zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py` | Streamed two-row search: enumerate row-1 ≈ (d₁, 0, 0), then row-2 ≈ (0, d₂, 0), row-3 via cross product. |
| `zeta9/zeta9/search_diagonal_matrix_two_rows.py` | Older non-streamed twin of the above. |
| `zeta9/zeta9/search_diagonal_matrix_from_rootdb.py` | Same row-of-unitary semantics, sidecar-backed candidate enumeration. |
| `zeta9/zeta9_2x2/enumerate_vectors_fit_matrix_2x2_mpi.py` | 2-vector fitting for a **2×2** diagonal target. Searches (x, y) such that the column completes to a unit 2×2; not 3×3 and not Householder. |

The only places that mention "Householder" semantics:

- `zeta9/try.py` (and unified `zeta9/try.py`): the helper function `MatrixDinC(coeff, f)` constructs `D = X_{(0,1)} · (I − x*x^T)` from a single 3-vector `x` (loaded from a precomputed `X2_*_best.npz`). This is a **post-processing diagnostic**, not a stage-5 search. The `Test` function still calls the row-of-unitary stage 5 (`search_diagonal_matrix_two_rows_streamed_mpi.py`).
- `unified/zeta9_compile.py:259-294` (`householder_v_from_u`): exact-integer ringZ9 reconstruction of V from a u-vector. Per `CELL_SETUP_AUDIT.md:49` this code path is dead — `extract_best_v` reads the full 3×3 `rows_coeffs_layer_f` emitted by the row-of-unitary stage 5 directly, never invoking the Householder helper.
- `unified/zeta9/MATH_AUDIT_2026-05-13.md:23`: "the 'Householder workflow' name in `zeta9/README.md` is a vestige of an earlier formulation."

**Conclusion: there is no existing Householder stage 5 to copy. The design below is new.**

## 2. Mathematical specification

### 2.1 Setup

Stage 1 enumerates ringZ9 elements `a` with `σ₁(|a|²) ≈ u·3^{2f}` for `u ∈ {0, 1}`. Stage 2 (in Householder mode, `--norm 2`, inputs `(Y₁_u=1, Y₁_u=1, Y₁_u=0)`) emits triples `(Y₁, Y₂, Y₃) ∈ Z[α]³` with `Y₁ + Y₂ + Y₃ = (2·3^{2f}, 0, 0)`. Stage 3 reifies, for each Y in the triples file, the set of ringZ9 elements `a` with `a · conj(a) = Y`. Stage 4 builds the phase-binned sidecar for those roots.

Empirical check on `D/TM_f=2_eps=0.05_hh` (16,027 rows, this is the smoke-test cell):
- Y₀ slot: `σ₁(Y)/3^{2f} ∈ [0.994, 1.006]`  →  `|a₀| ≈ 1`
- Y₁ slot: `σ₁(Y)/3^{2f} ∈ [0.994, 1.006]`  →  `|a₁| ≈ 1`
- Y₂ slot: `σ₁(Y)/3^{2f} ∈ [0.000, 0.003]`  →  `|a₂| ≈ 0`

(The `order: [2, 0, 1]` in the manifest is the bucket-key permutation used during the streamed join; it does not change the semantics of which Y is "small". After dedup, the small-norm Y is consistently in slot 2 of the file row.)

### 2.2 Householder reflection: convention

The Householder formula used throughout zeta9 (see `try.py:12-32`, `zeta9_compile.py:259-294`) is

```
D = X_{(0,1)} · (I − x̄ ⊗ x)         where (x̄ ⊗ x)_{ij} = conj(x_i) · x_j
```

(Note: this is **not** the textbook `I − 2 u u†/|u|²`; the factor `2` is absorbed by choosing `|x|² = 2` and the convention puts the conjugate on the **first** factor of the outer product, making the role of x "row-like" rather than column-like. This is the convention bequeathed by the original zeta9 paper code; we keep it.)

### 2.3 Target vector

For `x = (a, b, 0)` with `|a| = |b| = 1`:

```
(I − x̄⊗x)_00 = 1 − |a|²  = 0
(I − x̄⊗x)_01 = − ā · b
(I − x̄⊗x)_10 = − b̄ · a
(I − x̄⊗x)_11 = 1 − |b|²  = 0
(I − x̄⊗x)_22 = 1
```

After `X_{(0,1)}` swaps rows 0 and 1:

```
D_00 = − b̄·a     D_01 = 0           D_02 = 0
D_10 = 0           D_11 = − ā·b      D_12 = 0
D_22 = 1
```

Note `D_00 = conj(D_11)` automatically (Hermiticity of the outer product). With diagonal target `diag(d₁, d₂, 1)` (and `d₂ = conj(d₁)` for our R^Z gate), the conditions are:

```
|a|² = 1,  |b|² = 1,  b̄·a = −d₁   ⇔   a·b̄ = −d₁ = −e^{−iθ/2}
```

So we want to find ringZ9 `(a₀, a₁, a₂)` (at layer f, denominator `3^f`) with:

| condition | meaning |
|---|---|
| `\|a₀\|² ≈ 1` | radial constraint on slot 0 |
| `\|a₁\|² ≈ 1` | radial constraint on slot 1 |
| `\|a₂\| < ε` | small third component |
| `(a₀ · conj(a₁)) / 3^{2f} ≈ −d₁ = −e^{−iθ/2}` | **the phase constraint** |

This is the key difference from the diagonal stage 5: there is **no per-coordinate phase constraint** on a₀ or a₁ individually; only the **product phase** of `a₀·conj(a₁)` is fixed. The lattice has much more freedom, which is the source of the "cheaper N_D" claim in the task description.

### 2.4 Constructing V

Once `(a₀, a₁, a₂)` is found, build `V = X_{(0,1)} · (I − ā⊗a)` exactly in ringZ9 at layer `2f` (since the outer product squares the denominator). Concretely:

```
H_{ij} = δ_{ij} · 3^{2f} − conj(a_i) · a_j      (each H_{ij} ∈ Z[ζ_9], denom 3^{2f})
V row 0 = H row 1
V row 1 = H row 0
V row 2 = H row 2
```

Then emit `V` as `rows_coeffs_layer_f` of shape `(3, 3, 6)` with denominator `3^{2f}`, which is **the same on-disk schema as the diagonal stage 5 emits**. `extract_best_v` in `zeta9_compile.py` consumes this schema directly with no changes.

Note: V is automatically unitary by construction (Householder is exact at the algebra level). The only thing the search optimizes is the closeness of V to the target diagonal, i.e., `‖V − diag(d₁, d₂, 1)‖_F`.

### 2.5 Frobenius distance

```
‖V − diag(d₁, d₂, 1)‖_F² = Σ_ij |V_ij|² − 2·Re[ trace(V · diag(d₁, d₂, 1)*) ] + 3
                          = 3 − 2·Re[ d₁·conj(V_00) + d₂·conj(V_11) + conj(V_22) ] + 3
```

But the off-diagonals contribute as well when V is not exactly diagonal; just compute the Frobenius norm directly via the same `matrix_frobenius_dist` helper used by the diagonal stage 5.

### 2.6 What about Y₃ = (0,0,0) (i.e. `a₂` exactly zero)?

The find_roots stage maps `Y = (0,0,0)` to the single root `a = 0`, which is the dominant case in the smoke-test cell (most triples have small but non-zero Y₂). For non-zero small Y₂, the rootdb returns the lattice roots; we filter them by `|complex_embedding(a₂)/3^f| < ε`.

## 3. Algorithm pseudocode

```
function search_householder_streamed(triples_file, rootdb_prefix, f, theta, eps):
    d1, d2, d3 = exp(-iθ/2), exp(iθ/2), 1
    target_a0_times_conj_a1 = -d1       # complex; a0 · conj(a1) ≈ this
    scale = 3^f

    paths = state_paths(rootdb_prefix)
    phase_meta = load(paths.phase_sidecar_meta)
    chunk_meta = load(chunk_meta_json)

    for chunk in stream(triples_file, by chunk):
        triples = read(chunk)
        locator = chunk.locator   # Y -> (batch_idx, row_idx) lookup in sidecar
        if locator is None:
            locator = build_locator(unique(triples), paths)

        # Per-chunk caches for descriptors.
        a0_desc_cache: Y -> desc (radial only, target |a|=1)
        a1_desc_cache: Y -> desc (radial only, target |a|=1)
        a2_desc_cache: Y -> desc (target=0+0j, radial |a|<eps)

        for tri in triples:
            Y0, Y1, Y2 = tri[0:3], tri[3:6], tri[6:9]

            desc_a0 = a0_desc_cache.get_or_build(Y0, target=1.0+0j)   # radial shell
            desc_a1 = a1_desc_cache.get_or_build(Y1, target=1.0+0j)
            desc_a2 = a2_desc_cache.get_or_build(Y2, target=0.0+0j)
            if desc_a0 is None or desc_a1 is None or desc_a2 is None:
                continue

            roots_a0, z_a0 = materialize(desc_a0)   # (n0, 6) int + (n0,) complex
            roots_a1, z_a1 = materialize(desc_a1)
            roots_a2, z_a2 = materialize(desc_a2)
            if n0 == 0 or n1 == 0 or n2 == 0:
                continue

            # Filter (a0, a1) pairs by product phase constraint.
            # a0 · conj(a1) / 3^{2f} ≈ -d1   (complex)
            # Brute-force inner loop, since #(a0)*#(a1) is typically small after radial filter.
            for i0 in range(n0):
                for i1 in range(n1):
                    prod = z_a0[i0] * conj(z_a1[i1])
                    if |prod - target_a0_times_conj_a1| > eps:
                        continue
                    # Phase constraint satisfied; iterate a2 candidates (often just 1).
                    for i2 in range(n2):
                        if |z_a2[i2]| > eps:
                            continue
                        # Emit V.
                        V_int = build_V_exact(roots_a0[i0], roots_a1[i1], roots_a2[i2], f)
                        dist = frobenius(V_int, diag(d1, d2, d3))
                        push_to_heap(V_int, dist, ...)

    gather and save heap as xout.npz with rows_coeffs_layer_f, dist_fro, f=2f, ...
```

### 3.1 Performance considerations

- Radial-only candidate descriptors are coarser than the diagonal stage 5's radial+phase descriptors. The sidecar always supports radial-only enumeration by passing `target=ρ_real_positive` and `eps=ρ_target`, which collapses the phase-shell logic to "all phases at this radius". (See `_candidate_desc_from_binned_sidecar` lines 188-197: when `target` magnitude ≤ eps, every root with `|a|−|target|<eps` passes; we exploit a different branch for radial-only by setting target on the real positive axis with eps = 2·|target|.)
  - **Implementation note**: simpler to enumerate all roots at the locator and apply the radial filter ourselves. Since we'll do batched Numba ops anyway, the per-Y root list is the natural granularity.
- Each (a₀, a₁) pair must hit a complex phase shell; the brute-force inner is `n₀ × n₁` complex-distance checks. For `f=2` cells this is small (tens of roots per Y); for `f=4` it could be hundreds, still tractable.
- The exact V construction is `2·9 = 18` ringZ9 multiplications per surviving triple. Negligible.

### 3.2 Output schema

Identical to the diagonal stage 5 — `xout.npz` containing:

```
rows_coeffs_layer_f : (n_results, 3, 3, 6) int64
dist_fro            : (n_results,) float64
diag                : (3,) complex128
f                   : (1,) int64        # = 2*f_user (V-denominator exponent)
eps                 : (1,) float64
```

This means `extract_best_v(xout_npz_path, f_u)` in `zeta9_compile.py` works unchanged.

## 4. Files

- New: `unified/zeta9/zeta9/search_householder_two_rows_streamed_mpi.py` (this design)
- Edit: `unified/zeta9_compile.py:run_stage5` — dispatch on `mode`.
- Untouched: `search_diagonal_matrix_two_rows_streamed_mpi.py` (diagonal mode keeps using it).

## 5. Smoke test plan

```
cd /home/hlamm/Desktop/efficent_gates/unified
python3 zeta9_compile.py --theta $(python3 -c 'import math; print(math.pi/2)') \
    --epsilon 0.5 --max-f 1 --mode householder --mpi 4 --json /tmp/hh_test.json
```

Note: `--max-f 1` so `f_v = 2` (the cached cell). `--eps-pre 0.05` matches the cached eps_pre, or `--epsilon 0.1` with default eps_pre=epsilon/2 = 0.05 ✓.

Success criterion: `matrices_found > 0` in the stage-5 output, and the emitted V (in xout.npz) has `‖V − diag(target)‖_F` reported in the summary.
