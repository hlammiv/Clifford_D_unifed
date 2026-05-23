# qutrit Clifford+D compiler — efficient_gates

Three independent backends for compiling single-qudit unitaries (currently
focused on Rz(θ) targets) over the qutrit **Clifford + D** gate set.
The goal is to compare gate counts (N_D) at matched Frobenius accuracy ε
across the three approaches and to push ε as low as practically possible.

## Backends

| Backend | Source | Method | Strength |
|---|---|---|---|
| **esa** | `esa/` | Exhaustive search algorithm over `Z[ζ₉, 1/3]` | Reference / ground truth (slow) |
| **hrsa** | `hrsa/` | Householder Reduction Search Algorithm, bidirectional BFS with R-extended dispatcher | Fastest at moderate ε (≥10⁻³) |
| **zeta9** | `zeta9/zeta9/` | Lattice-first norm-equation pipeline (collect_targets → select_triples → find_roots → search_householder) over `Q(ζ₉)` | Reaches tight ε (10⁻⁴ and below) where the other two stall |

All three emit a **uniform JSON schema** documented in
[`compile_qutrit_schema.md`](compile_qutrit_schema.md) so cross-backend
comparisons can be done from a common file format.

### In-progress: Solovay-Kitaev (SK) bootstrap pipeline

A fourth backend is under construction (see `rz_db/` and `u_net/`):

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
The current canonical sign for `Rz(θ) = diag(e^{iθ/2}, e^{-iθ/2}, 1)`
matches ESA and HRSA; the zeta9 stage-5 search uses a Householder-row
layout described in [`zeta9/HOUSEHOLDER_STAGE5_DESIGN.md`](zeta9/HOUSEHOLDER_STAGE5_DESIGN.md).

## License

(TBD — pick before public release.)
