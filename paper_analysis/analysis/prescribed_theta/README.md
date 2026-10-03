# Prescribed-θ vs special-θ Clifford+D gate counts

**Question:** the headline fits are made from Gnedin's 900 exact approximants at
*special* angles θ (θ comes out of the search). Do exact approximants at *prescribed* θ
have the same gate counts (a) at fixed sde and (b) as a function of ε?

**Verdict (short):**
* **Per sde: yes.** At sde_3 = 8, the only level where both sets are generic and large,
  prescribed and special counts agree to about 1%: N_φ 61.1 vs 61.9 (Δ = −0.8 ± 0.7),
  unitary T-cost 126.5 vs 125.9 (Δ = +0.6 ± 1.6), measurement T-cost 82.1 vs 82.0. Every
  per-class mean agrees too (n_T 22.9/23.4, n_4 10.05/10.07, n_R 4.76/4.58). At sde 6
  and sde 4 the counts also agree when the prescribed target ε is tight (comparable to
  special ε): ΔN_φ = −1.2 ± 0.7 at sde 6 (target 10⁻³) and +2.6 ± 1.2 at sde 4
  (target 10⁻²). Prescribed words found at loose targets (0.05–0.5) at the same sde have
  fewer phases (ΔN_φ −4 to −10) but more level-4/R gates, so their T-cost is higher
  (ΔT-cost +3 to +12). That bias comes from the loose-ε search. It is not a θ effect.
* **Versus ε: same slope, but shifted.** A prescribed θ needs about one more sde level
  to reach the same ε. The N_φ slope is the same: 5.05 ± 0.09 (balanced prescribed set)
  vs 5.161 ± 0.063 (special). The intercept is higher by about 9 (12.6 ± 0.4 vs
  3.94 ± 0.72). At ε ≈ 3·10⁻⁵ (sde 8) prescribed data sit +8.7 N_φ, +15.9 T-cost(7)
  and +10.5 T-cost(4) above the headline fits. **The headline vs-ε fits are therefore
  optimistic for an arbitrary prescribed angle.** They are correct as counts per sde.
  The special angles buy roughly a 12× smaller ε at the same sde, i.e. about 2.3 units
  of log₃(1/ε).

## Data

* Prescribed: every `unified/sweep_*/**.json` holding an exact `unitary.V` (13 614 JSONs:
  956 without a unitary, 12 658 with V). Deduplicating by exact V leaves
  **11 403 distinct unitaries**. Sources: zeta9_tier2 ε=10⁻⁴ (10 000 cells, all used,
  no subsampling), f4 ε=10⁻³/10⁻⁴, zeta9_cal, hrsa cascade/grid/dense/redo_max20/f6_*/
  sweep_out*/simp. `sweep_hrsa_f2_2026-05-23` holds only a summary CSV (no matrices).
  `rz_db/` holds code only; no DB file exists.
* Special: `unified/nick_test/nick_tcost_2026-09-30.csv` (900 rows, f = 4…16).

## Method

* `redecompose.py` re-decomposes every distinct V with the post-residual-R-fix fast
  decomposer (`decomp_speed/fast_decompose.install()`) and calls
  `nick_test/nick_tcost_all.work` verbatim. Counts and ε are therefore computed exactly
  as for the special set (plain Frobenius ‖V − diag(e^{−iθ/2}, e^{iθ/2}, 1)‖).
  Wall time is 130 s on 14 processes.
* Columns: n_T = n_T3, n_4 = n_L4, n_R = syllable R + residual R,
  **N_phi = N_D** (reducer D-count, convention A). The headline 3.94 + 5.161 L₃ is
  fitted to this column. The textbook formula 2n_T + n_4 + n_R (`N_phi_formula`) differs
  from N_D by −1…+1, with a mean of +0.54: a T-type syllable can carry 2 or 3 phases, and
  a residual phase of D-cost 1 is booked as T-type. ops = n_T + n_4 + n_R;
  tcost_unitary = n_T + 7n_4 + 7n_R; tcost_meas = n_T + 4n_4 + 4n_R.
* Checks. All 11 403 V are exactly unitary over Z[ζ₉, 1/3], and all decompositions
  succeed. The stored `achieved_frob` matches the recomputed ε to within 10⁻⁶ + 10⁻³ε
  for 12 602/12 626 occurrences (zeta9: median rel. diff 9·10⁻¹³; HRSA prints 6
  digits, so the max abs. diff is 5·10⁻⁷). **24 zeta9_tier2 cells
  (`cell_0026`–`cell_0049`) have inconsistent θ metadata.** Their V is a rotation by
  θ ≈ 0.017–0.031, but `inputs.theta`/`target` say 3.33–6.22, so the recomputed ε is
  2.1–2.8. These 23 distinct V are flagged `eps_ok = 0` and excluded. No row has
  achieved ε > target ε.
* Dedup: each V keeps its best-ε occurrence. 72 V occur at more than one θ (by sde: 10 at sde 0, 17 at sde 2, 42 at sde 4, 2 at sde 6, 1 at sde 8).
* Denominators: every JSON stores V = (1/3^f)·Z[ζ₉] numerators, the same convention as
  Gnedin's `fits_f=*.txt`. HRSA's own f (the `HRSA(f=k)` method string and max_f) is the
  u-denominator, with V-f = 2k. The stored V-f equals sde_3(V) everywhere except for 16 zeta9_tier2
  matrices, which are stored at f = 8 but reduce to sde_3 = 7. Both CSVs report `sde` = sde_3(V) and `sde_chi` = max_ij sde_χ(V_ij).
  Gnedin's f equals sde_3 for all 900 rows. The prescribed sde distribution is
  {0: 10, 2: 57, 4: 562, 5: 1, 6: 673, 7: 16, 8: 10 061}.
* The stored (pre-fix) N_D was lower than the re-decomposed count by +0.82 on average:
  72% higher now, 28% equal, 0.5% lower (range −2…+10). The stored values came from
  older decomposer variants.
* `compare.py`: OLS fits vs L₃ = log₃(1/ε) with jackknife errors (delete-one for
  n ≤ 200, otherwise 50 random blocks). Fits use eps_ok, sde ≥ 1 and ε ≤ 0.3. sde 0
  rows are exact ζ₉-diagonal monomials and are not approximations. The "balanced" fit
  thins zeta9_tier2 to every 100th cell (1 564 points); otherwise that one ε cluster
  carries 87% of the weight.

## Key numbers

### Per sde_3 (mean ± s.e.m.)

| sde | set | n | median ε | N_φ | T-cost (7) | T-cost (4) |
|---|---|---|---|---|---|---|
| 4 | special | 150 | 5.8e-3 | 30.3 ± 0.5 | 69.8 ± 1.3 | 44.5 ± 0.6 |
| 4 | prescribed, all | 562 | 6.7e-2 | 25.5 ± 0.3 | 75.1 ± 0.7 | 46.2 ± 0.4 |
| 4 | prescribed, target 0.01 | 26 | 6.2e-3 | 32.9 | 66.2 | — |
| 6 | special | 150 | 1.4e-4 | 45.5 ± 0.6 | 98.2 ± 1.5 | 63.3 ± 0.7 |
| 6 | prescribed, all | 673 | 6.1e-4 | 43.1 ± 0.4 | 102.0 ± 0.8 | 64.8 ± 0.4 |
| 6 | prescribed, target 0.001 | 444 | 5.0e-4 | 44.3 | 99.6 | — |
| 8 | special | 150 | 2.8e-6 | 61.9 ± 0.7 | 125.9 ± 1.6 | 82.0 ± 0.8 |
| 8 | prescribed | 10 061 | 3.5e-5 | 61.1 ± 0.1 | 126.5 ± 0.2 | 82.1 ± 0.1 |

At the finer χ-adic level the agreement is clearer. At sde_χ = 48 (the generic
sde_3 = 8 value) N_φ is 61.5 ± 0.1 for prescribed (n = 7 951) vs 61.9 ± 0.7 for special.
At sde_χ = 36 it is 43.6 vs 45.5. Within sde 8 the prescribed N_φ does not depend on
ε: the slope is −0.08 ± 0.22 per L₃. **Counts are set by the denominator, not by θ.**
The full tables (per class, per target ε, per sde_χ) are in `compare_tables.md`.

### Fits vs log₃(1/ε) (a + b·L₃, jackknife errors)

| set | N_φ | T-cost (7) | T-cost (4) |
|---|---|---|---|
| special, all 900 (headline) | 3.94 ± 0.72 + 5.161 ± 0.063 L₃ | 16.61 ± 1.86 + 10.017 ± 0.146 L₃ | 9.74 ± 0.91 + 6.584 ± 0.070 L₃ |
| special, ε ≥ 10⁻⁴ (n = 257) | 8.1 ± 1.7 + 4.74 ± 0.28 L₃ | 29.0 ± 3.5 + 8.76 ± 0.59 L₃ | 17.3 ± 1.7 + 5.82 ± 0.28 L₃ |
| special, common ε range [4.6e-6, 7.4e-3] (n = 300) | 10.0 ± 1.4 + 4.36 ± 0.23 L₃ | 32.4 ± 3.4 + 8.08 ± 0.53 L₃ | 19.5 ± 1.7 + 5.37 ± 0.26 L₃ |
| prescribed, all (n = 11 364) | 13.17 ± 0.37 + 5.082 ± 0.041 L₃ | 54.5 ± 1.1 + 7.65 ± 0.12 L₃ | 32.2 ± 0.6 + 5.29 ± 0.07 L₃ |
| prescribed, balanced (n = 1 564) | 12.60 ± 0.42 + 5.052 ± 0.087 L₃ | 52.4 ± 1.5 + 7.91 ± 0.25 L₃ | 31.0 ± 0.8 + 5.42 ± 0.13 L₃ |
| prescribed, common range, balanced (n = 977) | 18.5 ± 1.3 + 4.26 ± 0.19 L₃ | 75.1 ± 3.2 + 4.80 ± 0.45 L₃ | 44.6 ± 1.6 + 3.57 ± 0.23 L₃ |

The special set reaches ε ≥ 10⁻⁴ only with f = 4 and part of f = 6; its largest ε is
7.4·10⁻³, so nothing overlaps the prescribed points at ε ~ 0.02–0.3. In the common
range both sets have flatter slopes than the full fit; a restricted range (two or three
sde clusters) does that.

* The N_φ slope matches: 5.05–5.08 vs 5.16.
* The prescribed T-cost slope is lower (7.9 vs 10.0). The low-sde prescribed points
  (HRSA, loose targets) carry extra level-4/R and raise the intercept. At sde 8 the
  prescribed T-cost is equal to the special one.

Mean residual of the prescribed data from the headline fits (data − fit):

| sde | n | median ε | ΔN_φ | ΔT-cost(7) | ΔT-cost(4) |
|---|---|---|---|---|---|
| 4 | 562 | 6.7e-2 | +5.7 to +7.8 | +30 | +18 |
| 6 | 673 | 6.1e-4 | +7.5 ± 0.4 | +23.9 ± 1.0 | +14.6 ± 0.5 |
| 8 | 10 061 | 3.5e-5 | +8.7 ± 0.1 | +15.9 ± 0.2 | +10.5 ± 0.1 |

### Achieved ε vs sde: the fixed-θ ε penalty

| sde | special min / median ε | prescribed min / median ε | median ratio | Δ log₃(1/ε) |
|---|---|---|---|---|
| 4 | 3.2e-3 / 5.8e-3 | 4.1e-3 / 6.7e-2 | 11.7× | 2.24 |
| 6 | 6.9e-5 / 1.4e-4 | 2.0e-4 / 6.1e-4 | 4.4× | 1.35 |
| 8 | 2.8e-6 / 2.8e-6 | 4.6e-6 / 3.5e-5 | 12.2× | 2.28 |

Fitting log₃(1/ε) = c + d·sde gives d = 1.447 ± 0.004 for special and 1.610 ± 0.008
for prescribed (sde ≥ 1). The intercepts are −0.58 and −3.50, so at the sde levels in
use the prescribed ε is 2–2.3 log₃-units worse.

Caveat: prescribed searches stop at the first denominator that meets the target, so the
prescribed median ε is partly set by the target (e.g. sde 8 is reached at a 10⁻⁴ target
with ε ≈ 1.7–5·10⁻⁵). The sde 8 comparison is the cleanest: 9 961 generic θ at
target 10⁻⁴ give median ε = 3.4·10⁻⁵. The best of 10⁴ angles reaches 4.6·10⁻⁶, while
the special angles sit on a near-constant envelope of 2.8·10⁻⁶. That envelope is about
12× (2.3 L₃ units, about 1.6 sde levels) better than a typical prescribed angle at the
same denominator. Times the N_φ slope, 5.16 × 2.28 ≈ 11.8. That is consistent with the
+8.7 N_φ offset vs ε, given that the prescribed sde 8 N_φ is 0.8 lower.

## Implication for the paper

* "Gate counts per denominator (sde) are independent of θ" holds: about 1% at sde 8 for
  N_φ and both T-cost models, n ≈ 10⁴ vs 150.
* The headline count-vs-ε fits describe special angles. For a prescribed θ the
  expected cost at a given ε is higher by about one sde level: roughly +9 N_φ, +16
  unitary T-cost, +10 measurement T-cost at ε ~ 10⁻⁵ (same N_φ slope). Either quote
  the fits per sde, with ε(sde) as a separate statement, or add this offset when
  quoting cost-vs-ε for arbitrary angles.

## Files

* `redecompose.py`: builds `prescribed_counts.csv` (one row per distinct V),
  `eps_check.csv` (stored vs recomputed ε per occurrence), `nick_sde.csv`
  (sde_3/sde_χ of the 900 special matrices) and `redecompose.log`.
* `compare.py`: builds `compare_tables.md`, `compare_results.json`,
  `prescribed_vs_special.pdf` (3.4 in wide, vector; N_φ and unitary T-cost vs
  log₁₀(1/ε)) and a PNG preview.
