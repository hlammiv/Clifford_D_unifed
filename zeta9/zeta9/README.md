# zeta9_householder

Self-contained package for the Householder search workflow.

Contents:
- `collect_targets.py`: MPI target collection with optional inert-parity and exact ideal-theoretic screening through `zeta9_root.quick_screen_M(...)`.
- `approximate_vector.py`: MPI two-phase vector matching, with root finding routed through `zeta9_root.actual_roots_from_ideal_search(...)`.
- `tools.py`: bundled `Z[zeta_9]` arithmetic and legacy exact backend helpers used by the Householder workflow.

CLI changes:
- `--norm_factor` was replaced by `--norm` (integer, default `2`).
- `--eps` is required.
- `--inert_prime_bound` now defaults to `29`.
- `zeta9_2x2.py` was removed; internal imports now use `tools.py`.
- `__init__.py` is intentionally minimal to avoid `python -m ...` runtime warnings from eager submodule imports.

Examples:

```bash
mpirun -n 8 python -m zeta9_householder.collect_targets \
  --f 5 --u 0.5 --eps 1e-6 --output Y.npy \
  --norm 2 --use_inert_parity --inert_prime_bound 29 \
  --use_exact_ideal_screen
```

```bash
mpirun -n 8 python -m zeta9_householder.approximate_vector \
  --file1 Y0.npy --file2 Y1.npy --file3 Y2.npy \
  --f 5 --t 1.576 --eps 0.175 --norm 2 --save out.npz
```
