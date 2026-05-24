# Phase 3 v2 smoke sweep results

Date: 2026-05-24
Module: `unified/cvp/diophantine_v2.py` (`solve_joint_x1_x3`)

## Setup

- Algorithm: joint (x_1, x_3) Babai-CVP enumeration (Selinger 2012 §7 port)
- Candidate pools: `n_x1 = 32`, `n_x3 = 32`
- f-bumping: walk `f ∈ {8, 10, 12, 14, 16, 18, 20}`, stop at first hit
- Early stop: first Frob-passing triple per cell
- 5 cells at ε=10⁻³, 5 at ε=10⁻⁴

## Results

| theta | eps | f | hits | wall (s) | best frob | N_D |
|---|---|---|---|---|---|---|
| 0.1335 | 0.001 | — | 0 | 58.7 | — | — |
| 0.1806 | 0.001 | 14 | 1 | 10.8 | 0.00091602 | 229 |
| 0.2121 | 0.001 | 14 | 1 | 9.2 | 0.00045607 | 198 |
| 0.2749 | 0.001 | 12 | 1 | 5.8 | 0.00056354 | 183 |
| 0.5000 | 0.001 | — | 0 | 59.7 | — | — |
| 0.1000 | 0.0001 | — | 0 | 56.2 | — | — |
| 0.1300 | 0.0001 | — | 0 | 55.3 | — | — |
| 0.1800 | 0.0001 | — | 0 | 63.3 | — | — |
| 0.2700 | 0.0001 | — | 0 | 60.8 | — | — |
| 0.5000 | 0.0001 | — | 0 | 52.2 | — | — |

## HRSA baseline comparison (matched cells)

HRSA sweep data: `unified/sweep_hrsa_grid_2026-05-22/summary.csv` (max-solns=20).

| theta | eps | v2 N_D | HRSA N_D | v2 / HRSA |
|---|---|---|---|---|
| 0.1335 | 0.001 | — (miss) | 37 (f=3) | — |
| 0.1806 | 0.001 | 229 (f=14) | 50 (f=3) | 4.6× WORSE |
| 0.2121 | 0.001 | 198 (f=14) | 29 (f=3) | 6.8× WORSE |
| 0.2749 | 0.001 | 183 (f=12) | 50 (f=3) | 3.7× WORSE |
| 0.5000 | 0.001 | — (miss) | (no hit) | — |

HRSA has no ε=10⁻⁴ data; the v2 0/5 at ε=10⁻⁴ is unmatched.

## Summary

- **ε=10⁻³ hit rate:** 3/5 (60%)
- **ε=10⁻⁴ hit rate:** 0/5 (0%)
- **CVP N_D vs HRSA at winning cells:** 3.7× to 6.8× WORSE
- **No cell where CVP beats HRSA in N_D**
- **Per-cell wall time at f=12-14:** 5-11 s when successful
- **Per-cell wall time when bumping all the way to f=20:** 50-65 s of wasted search

## Verdict

Phase 3 v2 does **not** meet the success criteria:
- Hit rate at ε=10⁻⁴ is 0%, not the > 50% required
- v2 N_D is multiplicatively worse than HRSA at every cell where both succeed
- The orbit-phase resolution floor of `~sin(π/18)·√2 / 3^f → ε` limits
  achievable Frob; reaching ε=10⁻⁴ would require f ≥ 20+ with a much
  larger candidate pool (currently 32×32 = 1024 pairs, would need 10K-100K)

## Notes on what works

- The algorithm IS correctness-clean: every returned triple satisfies the
  full bb-sum identity in Z[α] and reifies via `reify_householder(strict=True)`
  without raising — that's the primary deliverable vs v1.
- Joint enumeration with the ζ_18 torsion-orbit expansion + total-positivity
  pre-screen gives substantial speedup vs naive `for x_1: for x_3: PARI`:
  ~85-95% of pairs are filtered out before PARI is called.
- At eps=10⁻³, v2 succeeds where v1's `solve_x2_x3_ring_unitary` returns 0:
  v1 reported 0% hit rate at ε ≤ 10⁻³ per Phase 5 memory; v2 hits 60%.

## Recommendation

Do NOT proceed with Phase 6 paper-sweep. The CVP/HRSA N_D ratio (3-6× worse)
plus 0% hit at ε=10⁻⁴ means v2 is not a competitive synthesis path for the
paper. Consider:

1. **Wait for K'-cap to unlock zeta9 f=6** — that path has the right
   asymptotic, not a multiplicative loss.
2. **SK scaffolded driver (Phase E)** — already shipping to ε=10⁻⁵ at
   ~30K gates; tier-2 may reach 10⁻⁶ with ~100K gates.
3. **Investigate Babai pool diversity** — v2's intrinsic N_D loss may
   stem from the Babai candidates having higher q than necessary;
   a wider enumeration could help, but the orbit-floor remains an
   independent algorithmic obstacle.
