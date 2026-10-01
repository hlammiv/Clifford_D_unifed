# 06: Synthesis-side findings

## Residual-R bug (fixed)
- **The bug.** The canonical reducer charged nothing for the final signed-permutation residual. When its entry signs are mixed it needs one R, because Clifford monomials have uniform signs and ζ₉ phases cannot make −1.
- **Where it was:** Python `hrsa/canonical_reducer.py` (hard-coded 0) and both C++ paths (`decompose.cpp`, `decompose_impl.h`, non-Clifford-phase branch).
- **Size:** it affected 76% of HRSA candidates and ~0.7–0.8 R per rotation on Nick's circuits. **N_φ and R counts from before 2026-09-30 are low by up to 1.**
- **Fixed** in commit 6f43f53. The reducer regression test passes, and the Python fix agrees 12/12 with an independent sign check.

## Both R and level-4 are structurally required
- **Neither can be removed by restricting the reducer.**

  | Prefixes allowed | Decomposed |
  |---|---|
  | No R | 0/10 |
  | Only T-type or Clifford diagonals (no level-4) | 0/10 |
  | All (baseline) | 10/10 |

  Scripts: `nick_test/no_r_test.py`, `no_l4_test.py`. Peeling stalls at the first step.
- **Heuristic reason:** mod χ = 1−ζ₉ every ζ₉ phase ≡ 1, so only R can fix the residue sign pattern.

## T-cost is a property of the matrix
- **The test:** letting the reducer choose among sde-reducing prefixes by T-cost (`canonical_reducer.set_selection_cost("tcost")`).
- **Null result** on all 900: 16.55 + 10.02·log₃ vs 16.61 + 10.02. n_4 and n_R are unchanged at every f.

## Choosing approximants by T-cost works
- **The bias in the current rule.** HRSA_bestD's min-N_φ pick is systematically R-heavy (R-light words are longer in N_φ).
- **`HRSA_tester --rank-tcost`:** picks by T-cost instead. Test cell θ = 1.0, ε = 1e-2: 91 T → 55 T.
- **Low ε** (`r_count_study/`, 900 candidates at ε = 1e-2 and 0.1): within 10% of the best frob, the minimum T-cost is 36–61% below the frob-best candidate.
- **Tighter ε** (`tcost_selection_study/`, ε down to 0.003):
  - **The spread assumption holds.** The within-angle T-cost spread is 1.03–1.14× the cross-angle spread seen in Nick's data.
  - **Savings:**

    | Candidates per angle | Saving |
    |---|---|
    | 10 | ~27% |
    | 100 | ~42% |
    | 1,000 | ~57% |

  - **Pools are large:** ~27k candidates per angle at f = 3.
  - **Extrapolated slope:** 10.0 → ~9.0 (K = 10), ~8.3 (K = 100), ~7.8 (K = 1000).
- **Cheap predictor (empirical, 900/900):**
  - Take the leading χ-adic digit of each column-0 numerator, up to sign.
  - Class `111` means no bookend R; mixed classes need ≥2 R.
  - `111` is necessary for R = 0 (and about 1/3 of `111` candidates have R = 0).
- **Global phase never changes the R count.**

## Dead ends (recorded so they aren't retried)
- **Euler merging** (multiply several R_z, then re-reduce): 0.987× of the naive sum.
- **RUS** helps C+R equally (`02`).
- Bocharov port, Babai-CVP, Path C, and SK warm-start were already ruled out in May 2026.

## Tooling added
- `compiler/cd_to_ct.py`: decomposition → verified Clifford+T circuit (system + 1 ancilla).
- `nick_test/nick_tcost_all.py`: parallel recount of any fits file.
- `nick_request/analyze_topk.py`: ingest for Nick's top-K dumps (best-of-K curves and fits).
