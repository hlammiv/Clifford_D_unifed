# Top-K candidate dump: request and patch

Hi Nick,

Right now we only see the single min-Frobenius matrix for each θ. The T-cost of
a Clifford+D word can vary a lot between approximants whose ε is almost the
same. We'd like to choose among the near-best ones, so could you dump **every
candidate with frob ≤ 1.25 × best, up to K = 100 per θ**?

`topk_dump.patch` (one commit on top of your `main` @ eaa8068) adds this as
opt-in flags. Without the new flags, every script behaves exactly as before.

```bash
git switch -c topk-dump
git am topk_dump.patch          # or: git apply topk_dump.patch
```

## Option A: special-θ path (the one that made the `fits_f=N.txt` you sent us)

This needs no new search. It post-processes the `out.phase_free_special_theta.npz`
files you already have:

```bash
python3 dump_special_thetas.py --topk 100 --tol 1.25 --fs 4,6,8,10,12,14,16
```

For each special θ that `Dump()` writes, it ranks **every** matrix stored in the
npz (all groups × all root pairs) by ‖V − R_z(θ)‖_F. It then writes the ones with
frob ≤ tol·best (at most K) to `fits_topk_f=N.txt`, using the same `eps=` labels
that `Dump()` uses. With no `--topk`, the script runs the old `Dump()` calls.

The pool here is only as large as `--top_n` in `find_phase_free_special_theta_exactdiff`.
On the files you sent us, the matrices at f ≥ 6 had no neighbour within 1.25×. If you
can, please re-run that step with a larger `--top_n` (say 10–100× larger) so the pool
is denser.

## Option B: fixed-θ path (`zeta9/fit_theta_all_best_one_theta.py`)

New flags:

| flag | default | meaning |
|---|---|---|
| `--topk K` | 0 (off) | keep up to K candidates per θ |
| `--tol T` | 1.25 | keep matrix frob ≤ T × best matrix frob |
| `--topk_vec_slack S` | 1.5 | pass 2 enumerates root combinations with vector distance ≤ T·S·best |
| `--topk_txt PATH` | `<output_prefix>.topk.txt` | text output |
| `--topk_append` | off | append, so one file collects a whole θ loop |
| `--topk_format` | `matrix` | `matrix` or `vector` (18-integer Householder vector) |

Candidates are ranked by the true matrix Frobenius distance to R_z(θ). Copies that
give an identical exact matrix (for example x and a unit multiple of x) are written
once. The usual `.theta_all_best.npz` is still written and is unchanged. A new
`.theta_topk.npz` is also written. The driver `run_topk` loops over a θ list (for
example the θ column of a `fits_f=N.txt`) using the `D/V_f=..` layout of
`plot_eps_vs_theta.py`. Edit `f`, `lab`, `eps` and `mpi` at the top of it.

## Output format

This is the `fits_f=N.txt` format with two extra trailing columns and several rows per θ:

```
#f=8
# columns: theta, 3^f*Mij (3x3 6-element-groups), frob, rank
0.0032278824570467165 , 3349 152 ... , 574 343 -343 0 -2906 -3249 , 2.97e-05 , 0
0.0032278824570467165 , ...                                          , 3.41e-05 , 1
```

The element order is the same as `Dump()` (j outer, i inner). For `--topk_format vector`,
the header is `#f=N norm=2/3^f vector` and each row is `theta , 3^f x_0 , 3^f x_1 , 3^f x_2 , frob , rank`
with V = X01 (I − λ x̄ xᵀ).

**Size:** each matrix row is about 250–530 bytes (f = 4 … 16, frob and rank included), so
150 θ × 7 f × K = 100 comes to about 40 MB at most. Fewer candidates usually pass the 1.25× cut, so the real files will be smaller.

## What we tested (we can't run the pipeline end-to-end without your D/ data)

* The integer matrix builder (`tools.householder_int_matrix`) matches your Sage `MatrixC6`.
* The top-K enumeration inside pass 2, on synthetic **exact** Householder vectors, matches a
  brute-force search for 1 and 2 ranks.
* `fit_one_theta_all_best` runs with only the file I/O stubbed. Every written matrix is
  exactly unitary over Z[ζ9, 1/3], and the frob column checks out.
* `--topk 0` gives bit-identical `.theta_all_best.npz` output to `main`.
* `dump_special_thetas.py --topk` was run on a synthetic npz.

**Not tested:** real rootdb/sidecar I/O, MPI with more than one process under `mpirun`, and
run time on real triples files. A wide `--tol` or `--topk_vec_slack` enumerates more root
combinations in pass 2.

## Two things we noticed in passing (not changed)

* With the default `--equal_atol 1e-24`, pass 2 of `fit_theta_all_best_one_theta.py` can
  miss the best solution itself. `best_sq - (a+b)` and `c` differ by about one ulp (~1e-20
  when best_sq ~ 1e-3), which is wider than the searchsorted window. On our synthetic data
  it returned `n_solutions = 0` at every θ. A relative tolerance (or `--equal_atol 1e-12`)
  fixes it. The top-K branch compares `(a+b)+c` exactly as pass 1 does, so it is not affected.
* `dump_special_thetas.py` writes the 9 groups column-major (so it stores the transpose of D).
  This is harmless for ε and unitarity. The `tools.parse_fixed_row1_sage` row splitter fails on
  sparse rows (for example constant entries), and it also raises a `NameError` because
  `argparse` isn't imported in `tools.py`. Real data is dense, so this probably never bites.

Thanks!
Henry
