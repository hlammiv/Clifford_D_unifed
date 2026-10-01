# T-cost-aware candidate selection study (2026-09-30)

Question: if a synthesis pipeline returns many exact approximants per angle, how much does picking the one with the lowest fault-tolerant T-cost save, and does the slope estimate from Nick's data hold up?

**T-cost** = n_T3 + 7·n_L4 + 7·n_R. That is: T-type diagonals cost 1, level-4 diagonals 7, and R (syllable R plus residual R) 7, using the 7-T gadgets with one clean ancilla.

## Data
- Generated with `HRSA_tester θ ε max_f --no-direct --max-solns N`, which prints a `CANDTCOST` line per candidate (`run_cands.sh`).
- Angles θ ∈ {0.3, 0.5, 1.0, 1.7, 2.5}; ε ∈ {0.1, 0.01, 0.005, 0.003} (f = 2…4).
- All candidates are in `cands_all.csv`. Full candidate pools for θ = 1.0 are in `pool_t1.0_e0.01_full.csv` and `pool_t1.0_e0.1_full.csv`. Raw logs are in `raw/`.
- `xcheck_py_t1.0_e0.01.csv` cross-checks the C++ counts against the Python reducer on 30 candidates.

## Results
- **The within-angle spread matches the assumption.** The pooled within-angle sd of T-cost is 1.03–1.14× the cross-angle sd model fitted to Nick's matrices, sd ≈ 12.5 + 0.68·log₃(1/ε) (`sd_by_eps.csv`). So the assumption behind the slope estimate is fair, slightly pessimistic.
- **Best-of-K savings** (`bestK_by_eps.csv`) at ε = 0.003–0.01, where the mean T-cost is ≈ 93–100:

  | K | 3 | 10 | 30 | 100 | 1000 |
  |---|---|---|---|---|---|
  | T saved | ~15 | ~26 | ~34 | ~40 | ~55 (one angle) |
  | % | ~15% | ~27% | ~35% | ~42% | ~57% |

- **Pool size is not limiting** (`pool_size.csv`).
  - At θ = 1.0, HRSA finds 6,223 candidates at f = 2 (ε = 0.1) and 27,126 at f = 3 (ε = 0.01).
  - About 1,000 of the f = 3 candidates lie within 0.26 ε.
  - The pool grows with f.
- **Extrapolated T-cost slope**, from 10.0·log₃(1/ε) today (`extrapolate.out`; model A/B):

  | K | 10 | 100 | 1000 |
  |---|---|---|---|
  | slope (per log₃) | ≈ 9.0 | ≈ 8.3–8.5 | ≈ 7.8–8.0 |

  Bracketing the sd growth (slope 0.5–1.16 per log₃) gives K = 100 → 7.1–8.75.

## Caveats
- The measurements stop at ε = 0.003 (f ≤ 4). Extrapolating to 1e-10 assumes the spread keeps growing the way it does here.
- HRSA candidates, not Nick's exact-ring pipeline. Confirming at f = 12–16 needs top-K dumps from Nick's pipeline.
- The agent that ran this study was cut off by a network error before writing its own report. This README was written from its saved outputs.

## Scripts
`run_cands.sh`, `analyze_tcost.py`, `pool_size.py`, `extrapolate.py`.
