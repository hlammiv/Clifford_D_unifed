# Generated tables (source: `nick_tcost_2026-09-30.csv`, n = 900; weights: level-4 4 T, R 4 T)

## Per-f means

| f | n | median ε | N_φ (A) | N_φ (B) | T-type | level-4 | R syl | R resid | ops (A) | **T-cost** | T-cost sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 150 | 5.76e-03 | 30.3 | 27.4 | 10.6 | 5.5 | 2.19 | 0.72 | 19.1 | **44.5** | 7.8 |
| 6 | 150 | 1.38e-04 | 45.5 | 41.8 | 16.7 | 8.0 | 2.93 | 0.75 | 28.3 | **63.3** | 8.6 |
| 8 | 150 | 2.83e-06 | 61.9 | 57.3 | 23.3 | 10.1 | 3.91 | 0.67 | 38.0 | **82.0** | 9.4 |
| 10 | 100 | 2.20e-07 | 75.0 | 70.0 | 28.7 | 12.1 | 4.39 | 0.65 | 45.8 | **97.3** | 10.9 |
| 12 | 100 | 6.89e-09 | 90.9 | 83.9 | 34.3 | 14.9 | 6.28 | 0.78 | 56.2 | **122.0** | 9.6 |
| 14 | 100 | 3.56e-10 | 103.4 | 95.5 | 39.0 | 17.0 | 7.15 | 0.77 | 63.9 | **138.8** | 12.0 |
| 16 | 150 | 3.21e-11 | 121.6 | 113.1 | 46.4 | 19.8 | 7.78 | 0.77 | 74.7 | **159.7** | 13.7 |

## Fits vs log₃(1/ε) (all rows)

| quantity | intercept | slope per log₃ | slope per log₁₀ | R² | value at ε = 10⁻¹⁰ |
|---|---|---|---|---|---|
| N_φ, convention A (R charged) | 3.94 | 5.161 | 10.82 | 0.912 | 112.1 |
| N_φ, convention B (R free) | 2.94 | 4.824 | 10.11 | 0.879 | 104.1 |
| non-Clifford ops, A | 2.86 | 3.152 | 6.61 | 0.966 | 68.9 |
| non-Clifford ops, B | 1.86 | 2.815 | 5.90 | 0.932 | 60.9 |
| R count | 1.00 | 0.337 | 0.71 | 0.371 | 8.1 |
| level-4 count | 1.29 | 0.808 | 1.69 | 0.686 | 18.2 |
| T-type count | 0.57 | 2.008 | 4.21 | 0.774 | 42.7 |
| **T-cost** | 9.74 | 6.584 | 13.80 | 0.929 | 147.7 |

## C+R reference (Gustafson et al.), per log₃ and at ε = 10⁻¹⁰

| algorithm | N_R slope per log₃ | N_R at 10⁻¹⁰ | T-cost with R = 7 T at 10⁻¹⁰ | C+R / C+D (T-cost) |
|---|---|---|---|---|
| Householder | 5.139 | 110.9 | 444 | 3.00× |
| Exhaustive | 4.113 | 88.4 | 354 | 2.39× |
| SU(3) covering bound (not a fit) | 4.900 | 100.5 | — | — |

## Break-even against a C+R device with its own R-state factory

- Householder: C+D cheaper iff c_R/c_T > 1.33 at 10⁻¹⁰ (asymptotically > 1.28).
- Exhaustive: C+D cheaper iff c_R/c_T > 1.67 at 10⁻¹⁰ (asymptotically > 1.60).

## Qutrit (6 rotations) vs two-qubit emulation (10 R_z), magic-state units

| qutrit side | vs qubit RUS @1e-6 | @1e-10 | vs qubit deterministic @1e-6 | @1e-10 |
|---|---|---|---|---|
| C+D T-cost (as is) | 1.73 | 1.87 | 0.93 | 0.89 |
| C+D T-cost, best-of-100 (slope 8.3, extrapolated) | 2.13 | 2.33 | 1.15 | 1.11 |
| *C+D per-phase N_φ (mixed units; retracted as a cost)* | 1.29 | 1.42 | 0.69 | 0.67 |
| C+R Householder, R = 4 T | 5.07 | 5.62 | 2.72 | 2.67 |

Absolute at 10⁻¹⁰: qutrit C+D 6 × 148 = 886 T₃; 2-qubit RUS 474 T₂; 2-qubit deterministic 997 T₂.
