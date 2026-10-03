# Clifford+D synthesis of single-qutrit rotations

Code and derived data for

> P. J. Fox, N. Y. Gnedin, E. J. Gustafson, C. Herbig, H. Lamm, and E. M. Murairi,
> *Fault-tolerant synthesis of single-qutrit rotations from Clifford+D* (2026).

The repository synthesizes single-qutrit rotations `Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}, 1)` as exact
unitaries over `Z[ζ₉, 1/3]` (the gate set Clifford+D), decomposes them into Clifford+D words, and
compiles those into qutrit Clifford+T circuits with the 7-T and 4-T gadgets for `R = diag(1,1,-1)`
and the level-4 phases.

## Reproducing the paper

Requirements: Python ≥ 3.10 with the packages in [`requirements.txt`](requirements.txt).
The search backends additionally need a C++17 compiler (`hrsa/`, `esa/`) and Sage/PARI with
mpi4py (`zeta9/`); reproducing the tables and figures does not.

```bash
cd results_bundle && ./verify_all.sh          # gadgets, emitter, certificates, tables (8 checks)
cd ../paper_analysis
python3 scripts/make_numbers.py               # every number in the paper -> numbers.tex, tables/
python3 scripts/plot_figures.py               # every data figure -> fig/
```

| Paper item | Script | Input data |
|---|---|---|
| Numbers in the text, Tables III–V | `paper_analysis/scripts/make_numbers.py` | `nick_test/nick_tcost_2026-09-30.csv`, `symmetry_variants/stack_full30_rows.csv.gz`, `nick_request/topk_run/fast_all30_candidates.csv`, `paper_analysis/analysis/prescribed_theta/compare_results.json` |
| Figs. 2–5, App. E figures | `paper_analysis/scripts/plot_figures.py` | as above, plus `paper_analysis/analysis/special_angles/ang_f*.npz` |
| Exact approximants (900 matrices) | — | `nick_test/fits_f={4..16}.txt` (θ and 3^f·V, six integers per entry) |
| Gate counts of the 900 approximants | `nick_test/nick_tcost_all.py` | `nick_test/fits_f=*.txt` |
| Phase-copy selection | `symmetry_variants/variant_search.py`, `stack_analyze.py` | `nick_test/fits_f=*.txt` |
| Decomposition (Alg. 3) | `hrsa/canonical_reducer.py`, fast version `decomp_speed/fast_decompose.py` | — |
| Householder enumeration (Alg. 1) | `hrsa/` (C++, `make`; `HRSA_tester`) | — |
| Norm-equation pipeline (Alg. 2) | `zeta9/`, wrapper `zeta9_compile.py` | — |
| 7-T `R` gadget | `r_from_d/verify_R7T.py` | `r_from_d/R_7T_compact.json` |
| 7-T and 4-T level-4 gadgets | `level4/l4_7T_verify.py`, `level4/l4_meas_uncompute.py` | — |
| 4-T measured `R` gadget | `measurement_tricks/verify_meas.py` | `measurement_tricks/r_meas_4T.json` |
| Optimality (Theorem 1, App. C) | `two_ancilla/`, `three_ancilla/verify_lemmas.py`, `r_from_d/raw_lb_1anc.py`, `measurement_tricks/search*.py` | — |
| Clifford+T emission and merging | `compiler/cd_to_ct.py`, `depth_optimality/tmerge.py` | — |
| Merged `R` chains, 5N+2 T (Sec. V) | `paper_analysis/analysis/cr_merge/cr_merge.py` | `r_from_d/R_7T_compact.json` |
| Factory model, Table II | `factory_model/factory_cost.py` | — |
| Quadratic-residue codes check (Sec. V) | `paper_analysis/analysis/zurel_check/zurel_check.py` | — |
| Prescribed-angle comparison (Sec. VI, App. E) | `paper_analysis/analysis/prescribed_theta/redecompose.py`, `compare.py` | `prescribed_counts.csv` (derived) |
| Special-angle statistics (App. E) | `paper_analysis/analysis/special_angles/analyze_angles.py` | `ang_f*.npz` (derived) |
| C+D vs C+R summary notes | `results_bundle/*.md` | — |

### Data provenance

- The 900 exact approximants in `nick_test/fits_f=*.txt` were produced by N. Gnedin's
  phase-free special-angle search, a branch of the `zeta9` pipeline. That code is available
  from the authors on request.
- Only derived data are committed. The raw inputs behind two derived files are available on request:
  - the full special-angle output (265 MB), from which `special_angles/ang_f*.npz` is extracted by `extract_angles.py`;
  - the 11,403 prescribed-angle unitaries (about 100 MB of JSON), from which `prescribed_theta/prescribed_counts.csv` is built.
- All gate counts use the decomposer with the residual-`R` fix of 2026-09-30. Counts produced
  before that date in files under `sweep_*` and older notes are lower by up to one `R`.

### Repository status

Development material from earlier stages remains in the tree: HRSA/zeta9 sweeps, the CVP
backend, the SK bootstrap, and dated notes and plots. The paper uses only the items in the table above.

## Backends

| Backend | Source | Method | Strength |
|---|---|---|---|
| **esa** | `esa/` | Exhaustive search algorithm over `Z[ζ₉, 1/3]` | Reference / ground truth (slow) |
| **hrsa** | `hrsa/` | Householder Reduction Search Algorithm, bidirectional BFS with R-extended dispatcher | Fastest at moderate ε (≥10⁻³) |
| **zeta9** | `zeta9/zeta9/` | Lattice-first norm-equation pipeline (collect_targets → select_triples → find_roots → search_householder) over `Q(ζ₉)` | Reaches tight ε (10⁻⁴ and below) where the other two stall |

All three emit a **uniform JSON schema** documented in
[`compile_qutrit_schema.md`](compile_qutrit_schema.md) so cross-backend
comparisons can be done from a common file format.

### Exploratory: Solovay-Kitaev (SK) bootstrap pipeline

An exploratory fourth backend, not used in the paper (see `rz_db/` and `u_net/`):

- `rz_db/` — SQLite-backed R_z(θ) lookup DB. Loads from existing HRSA/zeta9
  sweep CSVs. Used by the SK pipeline to avoid re-spawning HRSA per Euler leaf.
- `u_net/` — U(3) net builder via Haar sampling + Euler decomposition into
  R_z leaves. Will support **scaffolded SK** with multiple decade-ε tiers.
- See `rz_db/PHASE_D_TODO.md` for lazy-population rule that ANY SK
  consumer of the R_z DB must honor.

## Layout

```
unified/
├── zeta9_compile.py        # single-shot wrapper for the zeta9 pipeline
├── hybrid_compile.py       # HRSA + zeta9 hybrid driver (work in progress)
├── sweep_hrsa.py           # angle × ε sweep harness for HRSA
├── sweep_zeta9_calibration.py  # min-frob calibration sweep for zeta9
├── plot_zeta9_calibration.py   # quick-look plotter
├── v_validate.py           # independent post-hoc validator
├── verify_conventions.py   # check θ-sign / basis conventions across backends
├── compile_qutrit_schema.md
├── HYBRID_DESIGN.md
├── hrsa/                   # HRSA C++ source + tester binaries (build via Makefile)
├── esa/                    # ESA C++ source + binaries
├── zeta9/
│   ├── zeta9/              # zeta9 Python package (collect_targets, select_triples_optimized,
│   │                       #   find_roots_exact_v2, search_householder_*_streamed_mpi, …)
│   ├── D/                  # generated data cache  (gitignored)
│   └── *.md                # design / audit notes
├── rz_db/                  # R_z lookup DB for the SK pipeline (Phase A)
│   ├── rz_lookup.py        # RzLookupDB SQLite class
│   ├── build_rz_db.py      # CSV → DB ingestor
│   ├── PHASE_D_TODO.md     # mandatory lazy-population rule for SK consumers
│   └── test_rz_lookup.py   # 12 tests, all passing
├── u_net/                  # U(3) net builder (scaffolded SK; Phases B-D)
│   ├── haar_sampler.py     # Haar SU(3) sampling + dedup + coverage estimate
│   └── test_haar_sampler.py
├── sweep_zeta9_batched.py  # C1 batched θ-sweep driver (one mpirun, many queries)
├── sweep_hrsa_grid.py      # HRSA grid sweep
├── plot_nd_vs_eps_v2.py    # paper-data N_D vs ε plotter (two-panel, color by method)
└── nd_vs_eps_v2_*.png      # rendered plots
```

## Quick start

### Build native binaries

```bash
cd hrsa && make
cd ../esa && make
```

(HRSA depends on a `Z[ζ₉, 1/3]` arithmetic library; ESA needs the same plus
its own perf probes.)

### Set up the Sage env (zeta9 only)

zeta9's stages 1–5 use Sage (cypari2 / PARI) + mpi4py:

```bash
conda create -n sage -c conda-forge sage mpi4py mpich python-flint
```

Then ensure `$SAGE_ENV/bin` is on `$PATH` before invoking the wrapper.
The wrapper sets `--sage-env` to a default; see `zeta9_compile.py --help`.

### Compile a single Rz(θ) target with zeta9

```bash
./zeta9_compile.py --theta 0.5 --epsilon 1e-3 --max-f 2 --mpi 4 \
                   --workdir ./zeta9 --json out.json
```

`--max-f N` is the **u-denominator cap**; the lattice V-denominator is
`2N` (see `zeta9_compile.py` docstring and `compile_qutrit_schema.md`).

### Compile with HRSA

```bash
./hrsa/HRSA_tester 0.5 1e-3 3 --json out.json
```

The third positional arg is the u-denominator cap (HRSA's max_f, same
semantic as the zeta9 wrapper's `--max-f`).

### Run a calibration sweep

```bash
./sweep_zeta9_calibration.py --n_thetas 100 --max_f_min 0 --max_f_max 2 \
    --eps 0.5 --mpi 4 --out_dir /tmp/sweep_out
./plot_zeta9_calibration.py /tmp/sweep_out/summary.csv out.png
```

## Conventions

θ-sign and basis conventions are documented in [`verify_conventions.py`](verify_conventions.py).
The canonical target is `Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}, 1)`, as in the paper;
ESA and HRSA match it; the zeta9 stage-5 search uses a Householder-row
layout described in [`zeta9/HOUSEHOLDER_STAGE5_DESIGN.md`](zeta9/HOUSEHOLDER_STAGE5_DESIGN.md).

## License

A license will be added after Fermilab software-release approval. Until then, all rights are reserved.
