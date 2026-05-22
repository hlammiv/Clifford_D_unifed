# Unified Qutrit Clifford+D Compiler Output Schema

This document specifies a single JSON schema that the three independent
qutrit Clifford+D compilation backends in this repository must emit so
that head-to-head comparisons (in particular, plots of the D-gate count
N_D versus the Frobenius accuracy ε at fixed θ) can be produced from a
common file format.

The three target backends are:

- **esa**   — `qutrits/ESA_Clifford_D_v6/ESA_test.cpp` (exhaustive
  search algorithm). Produces a 3x3 unitary V over Z[ζ_9, 1/3] but
  has no D-gate decomposition.
- **hrsa**  — `Clifford_D_Householder/HRSA_test.cpp` (Householder
  reduction search algorithm with a Direct-search and Diagonal-search
  fallback ladder). Has a real `decompose()` that yields D-counts and
  a syllable list.
- **zeta9** — `zeta9/zeta9/` MPI Householder pipeline
  (`collect_targets` → `select_triples_optimized` → `find_roots_exact_v2`
  → `fit_vectors_mpi_sidecar_binned`). Saves NPZ files holding
  `best_u_coeffs[n,3,6]` and `best_Y[n,3,3,6]`-shaped arrays of
  Z[ζ_9, 1/3] integer coefficients but, as of 2026-05, has no
  decompose stage and therefore no D-count.

Each compiler invocation MUST emit a single JSON object whose top-level
keys are the section names defined below. Unknown sections SHOULD be
ignored by readers, but unknown fields within a section SHOULD be
preserved when round-tripping. All angles are in radians, distances are
unitless Frobenius norms, times are seconds, memory is kilobytes (KB).

The accompanying file `compile_qutrit_schema_example.json` shows a fully
populated, schema-valid HRSA run at θ = π/4, ε = 10^-3.

---

## 1. `identification`

Who produced this record and how.

| Field          | JSON type | Required  | Filled by | Semantics |
|----------------|-----------|-----------|-----------|-----------|
| `backend`      | string    | required  | all       | One of `"esa"`, `"hrsa"`, `"zeta9"`. Lower-case, no version suffix. |
| `version`      | string    | required  | all       | Free-form. SHOULD be a git revision (`git rev-parse --short HEAD`) or, if not in a git tree, a release-style string such as `"ESA_v6"` or `"hrsa-2026-04"`. |
| `command_line` | array of string | required | all  | argv (including `argv[0]`). For zeta9, the underlying `python -m zeta9.fit_vectors_mpi_sidecar_binned ...` argv. |
| `host`         | string    | required  | all       | Hostname (`gethostname()` / `socket.gethostname()`). |
| `timestamp`    | string    | required  | all       | ISO-8601 UTC, e.g. `"2026-05-05T17:42:11Z"`. End time of the run. |
| `schema_version` | string  | required  | all       | This document's version, currently `"1.0"`. |

## 2. `inputs`

The user-facing knobs that name the problem instance.

| Field          | JSON type | Required  | Filled by | Semantics |
|----------------|-----------|-----------|-----------|-----------|
| `theta`        | number    | required  | all       | Target rotation angle in **radians**. |
| `epsilon`      | number    | required  | all       | Frobenius accuracy budget for the *matrix* distance ‖V − target‖_F. |
| `max_f`        | integer   | required  | all       | Maximum sde_3 / f-level explored by the search. For zeta9 this is `--f`. |
| `c`            | number\|null | required | all      | Householder contraction factor in (0, 1]. **null** for ESA and zeta9, which do not expose this knob; defaults to `1.0` for HRSA. |
| `max_solns`    | integer\|null | required | all     | Number of HRSA candidates kept before picking the lowest D-count. **null** for ESA and zeta9. |
| `max_direct`   | integer\|null | required | all     | Maximum k for HRSA's Direct-search phase (gate-sequence enumeration). `-1` means "skipped via `--no-direct`". **null** for ESA and zeta9, which do not have a Direct phase. |

Notes:

- HRSA's CLI also accepts `epsilon` as a *vector-Frobenius* tolerance
  internally (the condition `ε² / (8 c²)`), but the schema's `epsilon`
  is always the **matrix** Frobenius target, matching how all three
  backends end up reporting "did the result pass?".
- zeta9's `--norm` (default 2) is an internal screening parameter and
  is therefore not in the schema; if a backend wants to record it, it
  goes under `errors` as an informational string or in a future
  optional `backend_specific` map.

## 3. `target`

The gate the compiler was asked to approximate. Captured explicitly so
that downstream tools can re-verify the achieved Frobenius distance
without rebuilding `target` from `theta`.

| Field      | JSON type | Required | Filled by | Semantics |
|------------|-----------|----------|-----------|-----------|
| `gate`     | string    | required | all       | Identifier. Default `"R_Z_01_theta"` = the rotation R_{(0,1)}^Z(θ) used by HRSA and ESA: diag(e^{−iθ/2}, e^{+iθ/2}, 1) up to the convention. Other allowed values: `"R_Z_02_theta"`, `"R_Z_12_theta"`. |
| `convention` | string  | required | all       | Either `"esa"` (target = diag(e^{+iθ/2}, e^{−iθ/2}, 1), as in `ESA_test.cpp`) or `"hrsa"` (target = X_{(0,1)} (I − u u†) ≈ R_{(0,1)}^Z(θ), as in `HRSA_test.cpp`). The two conventions differ by an overall X swap of the first two rows; readers MUST honor this when comparing. zeta9 SHOULD set whichever convention its run scripts target. |
| `matrix`   | 3×3 array of `[re, im]` pairs | required | all | The target as floating-point complex doubles, row-major. Used for self-checking. |

## 4. `achieved`

A flat summary of the outcome, suitable for the per-row line in
`results.csv`-style aggregation.

| Field             | JSON type | Required | Filled by | Semantics |
|-------------------|-----------|----------|-----------|-----------|
| `success`         | boolean   | required | all       | True iff a unitary V was produced and `achieved_frob ≤ epsilon`. |
| `achieved_frob`   | number    | required | all       | ‖V − target‖_F as a double. **NaN-equivalent**: emit `null` if no V was produced. |
| `epsilon_passed`  | boolean   | required | all       | Equivalent to `achieved_frob < epsilon` when `success` is true; redundant but cheap, kept for grep-ability. |
| `f_level`         | integer\|null | required | all   | The sde_3 / f at which the solution lives. For HRSA Direct hits this is 0. **null** if no solution. |
| `method`          | string    | required | all       | Free-form tag identifying which sub-algorithm produced the answer. Allowed values: `"Direct"`, `"Diagonal"`, `"HRSA"`, `"ESA"`, `"zeta9-householder"`, `"none"`. |

## 5. `unitary`

The exact algebraic representation of V. All nine entries share a single
denominator `f`, so V = (1/3^f) · M where M is a 3×3 matrix over Z[ζ_9].
This matches what every search backend actually produces (HRSA, ESA, and
zeta9 all carry one common denominator level f).

Each ring element is a length-6 integer coefficient vector in the basis
{1, ζ_9, ζ_9², ζ_9³, ζ_9⁴, ζ_9⁵}. The 6-coefficient form matches
`Clifford_D_Householder/Z9chi.h::getStdArray()` (size 6) and the zeta9
npz files (trailing dim 6). The internal C++ `element[9]` array MUST be
reduced via Φ_9(ζ_9) = 0 before serialization.

| Field   | JSON type | Required | Filled by | Semantics |
|---------|-----------|----------|-----------|-----------|
| `ring`  | string    | required | all       | Constant `"Z[zeta_9, 1/3]"`. Reserved for future schema versions. |
| `basis` | string    | required | all       | Constant `"1, zeta9, zeta9^2, zeta9^3, zeta9^4, zeta9^5"`. Coefficient ordering MUST match this. |
| `f`     | integer   | required | all       | The common denominator exponent: every entry is divided by 3^f. |
| `V`     | 3×3×6 array of integers | required | all | Row-major 3×3 list-of-lists, each entry being a length-6 integer coefficient vector. The complex value of entry (i, j) is (Σ_{k=0..5} V[i][j][k] · ζ_9^k) / 3^f. |

This is exactly the shape consumed by `decompose_tool` (`{"f": int, "V":
[[ [6 ints]×3 ]×3]}`). Re-emitters and consumers must agree on the same
f-once-per-matrix layout — do NOT introduce per-entry `denom_exp`. If a
backend ever needed mixed denominators per entry, the right approach is
to multiply through by the common 3^f_max so the schema-level f covers
all of them.

## 6. `decomposition`

The Clifford+D gate sequence and counts. Several fields are optional
because not every backend computes them.

| Field                | JSON type | Required | Filled by | Semantics |
|----------------------|-----------|----------|-----------|-----------|
| `N_D`                | integer\|null | required | hrsa | Total D-gate count. **null** for ESA and zeta9, which lack a decompose stage as of 2026-05. |
| `N_C`                | integer\|null | optional | hrsa | Total Clifford count, if exposed. As of 2026-05 `decompose.h::DecompResult` does not store this independently of `steps.size()`; HRSA SHOULD set it to `len(steps)` minus the number of trailing identity Cliffords, and otherwise emit `null`. |
| `N_total`            | integer\|null | optional | hrsa | `N_D + N_C` if both are known, else `null`. |
| `sde_chi_initial`    | integer\|null | optional | hrsa | `DecompResult::sde_chi` of the input V before peeling. **null** if not recorded. |
| `sde_chi_final`      | integer\|null | optional | hrsa | sde_chi after the final peel; should be 0 on success. |
| `syllables`          | array\|null | optional | hrsa | Optional list of `GateStep`-shaped objects. **null** for ESA and zeta9. As of 2026-05 only `D_count` is reliably exposed by `decompose.h::DecompResult` (the `steps` vector exists internally but is not currently formatted for output by `HRSA_test.cpp`); if a backend chooses not to dump it, set this to `null` rather than `[]`. |

Each syllable, when present, is an object:

```
{
  "H_factor":  bool,        // whether this step's prefix includes the qutrit Hadamard H
  "D_diag":    [a0, a1, a2],// integer triple in {0..8}^3, parameters of D_clifford = diag(ω^a0, ω^a1, ω^a2), ω = ζ_9^3
  "R_exp":     int,         // 0, 1, or 2: exponent on the non-Clifford D gate diag(ζ_9, 1, ζ_9^{-1})
  "X_exp":     int          // 0, 1, or 2: exponent on the X cyclic shift
}
```

This mirrors `GateStep` in `decompose.h`. The convention is that the
total V equals the **left-to-right** product of (H^{H_factor} · D(D_diag)
· Dgate^{R_exp} · X^{X_exp}) over the `syllables` list, followed by a
trailing Clifford which is determined by lookup and is NOT in the list
(this matches the iterative peel-from-the-left algorithm in
`decompose.cpp`).

## 7. `sanity_checks`

Numerical self-consistency checks the backend computed before writing
the file. Readers should re-verify these but populating them helps
catch silent failures.

| Field                          | JSON type | Required | Filled by | Semantics |
|--------------------------------|-----------|----------|-----------|-----------|
| `unitarity_residual`           | number    | required | all       | ‖V V† − I‖_F evaluated in floating point. SHOULD be ≤ 1e-12 for any honest output. |
| `frobenius_check_passed`       | boolean   | required | all       | True iff `achieved_frob ≤ epsilon`. Logically redundant with `achieved.epsilon_passed`; included so verifiers can grep for one section. |
| `decompose_roundtrip_passed`   | boolean\|null | required | hrsa | True iff multiplying together the `syllables` list (and the trailing Clifford) reproduces `unitary.V` *exactly* in the ring (no floating-point comparison). **null** when `syllables` is null. |

## 8. `performance`

Wall, CPU, memory, and parallelism. Values are best-effort: a backend
that can't measure CPU time independently of wall (e.g. `getrusage` not
available) MAY set `cpu_seconds = wall_seconds`.

| Field          | JSON type | Required | Filled by | Semantics |
|----------------|-----------|----------|-----------|-----------|
| `wall_seconds` | number    | required | all       | Elapsed wall clock from start of `main` (or, for zeta9, the outermost MPI driver). |
| `cpu_seconds`  | number    | required | all       | Sum of user+sys CPU across all threads/ranks of this run, in seconds. |
| `max_rss_kb`   | integer   | required | all       | Peak resident set size in kilobytes. On Linux this is `getrusage(RUSAGE_SELF).ru_maxrss` (which is already in KB). For MPI runs this is the max over ranks. |
| `threads`      | integer   | required | all       | OpenMP/std::thread thread count actually used (1 for single-threaded). |
| `mpi_ranks`    | integer\|null | optional | zeta9 | MPI world size. **null** for ESA and HRSA, which are not MPI parallel. |

## 9. `errors`

A list of human-readable warnings or non-fatal errors. Empty list (not
null) when the run was clean. Common entries:

- `"truncated lookup table at f=5"` — ESA's table-walk hit its
  enumeration cap.
- `"MPI buffer overflow during gather; recovered"` — zeta9
  `fit_vectors_mpi_sidecar_binned` recovered from a chunked-gather edge
  case.
- `"decompose: tryDoublePrefix exhausted at sde_chi=K"` — HRSA's
  `decompose()` could not peel one syllable and fell back to two.
- `"direct search returned k=2 but Frobenius did not pass"` —
  informational only.

| Field    | JSON type        | Required | Filled by | Semantics |
|----------|------------------|----------|-----------|-----------|
| (root)   | array of string  | required | all       | Possibly empty; never null. Order is chronological. Strings SHOULD be ≤ 240 chars; longer messages MUST be truncated with a trailing `"..."`. |

---

## Per-backend responsibility summary

| Section          | esa                                | hrsa                          | zeta9                              |
|------------------|------------------------------------|-------------------------------|------------------------------------|
| identification   | full                               | full                          | full                               |
| inputs           | `c`/`max_solns`/`max_direct` null  | full                          | `c`/`max_solns`/`max_direct` null  |
| target           | full (uses `convention="esa"`)     | full (`"hrsa"`)               | full (set per run)                 |
| achieved         | `f_level` from final loop          | full                          | `f_level` = `--f`                  |
| unitary          | full (3×3 ring repr.)              | full                          | full (decode `best_Y` from npz)    |
| decomposition    | `N_D=null`, all syllable fields null | full                       | `N_D=null` (no decompose yet)      |
| sanity_checks    | `decompose_roundtrip_passed=null`  | full                          | `decompose_roundtrip_passed=null`  |
| performance      | full (`mpi_ranks=null`)            | full (`mpi_ranks=null`)       | full (`mpi_ranks` from `comm.size`)|
| errors           | full                               | full                          | full                               |
