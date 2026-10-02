# Qutrit Clifford+D vs Clifford+R: results bundle (2026-10-01)

This bundle collects every result from the 2026-09-29 → 10-01 work, with the code to recompute it. **Convention A (signed, R charged) is adopted** (decision 2026-10-01, below). Convention B is shown in `01` for comparison only.

## Headline (updated 2026-10-02): T per R_z rotation at ε = 10⁻¹⁰, all models like-for-like

C+D: Nick's exact approximants, 30 θ × 7 f-levels. "Best copy" = the cheapest of 324 exact ±ζ-phase conjugations, all with identical ε. C+R: Gustafson et al. fits, with each R costed in the same gadget model.

| Gadget model | C+D as given | **C+D best copy** | C+R Householder | C+R Exhaustive | C+D advantage (best copy) |
|---|---|---|---|---|---|
| Unitary (level-4 = R = 7 T) | 223 (18.7 + 9.73·log₃) | **206** (2.5 + 9.71·log₃) | 776 (R = 7 T) | 619 | **3.8× / 3.0×** |
| Unitary + rotation merging (shared ancilla) | 166 (14.1 + 7.23·log₃) | **154** (3.3 + 7.21·log₃) | 554 (merged R = 5.0 T) | 442 | **3.6× / 2.9×** |
| Measurement + Clifford feed-forward (4 T) | 146 (10.9 + 6.44·log₃) | **136** (1.8 + 6.42·log₃) | 444 (R = 4 T) | 354 | **3.3× / 2.6×** |

- **The savings stack.**
  - Copy selection lowers the intercept by ~10–16 T per rotation in every model. The slope is unchanged.
  - Merging lowers the slope (9.7 → 7.2). The best copy for unmerged cost stays the best after merging.
  - Source: `symmetry_variants/stack_summary.txt` (68,040 decompositions; script `stack_analyze.py`).
- **Merging on the C+R side:** R chains merge to **5.0 T per R**, measured on C+R-shaped gadget chains. So the merged row compares like with like.
- **Sample size:** fits use n = 210 (30 θ per f). The as-given unitary fit (9.73·log₃, 223 at 10⁻¹⁰) agrees with the 900-matrix fit (10.0·log₃, 227).
- **Not included yet:** best-of-K over genuinely different approximants. That awaits Nick's larger `--top_n` rerun; at f = 4 it gave a further ~37%.

## Headline (original, 2026-10-01)

| | C+D (this work) | C+R (Gustafson et al. 2503.20203) |
|---|---|---|
| Per-phase gate count, convention A (R charged) | 3.94 + **5.16**·log₃(1/ε) | Householder 5.14, Exhaustive 4.11 ·log₃(1/ε) |
| Per-phase gate count, convention B (R free) | 2.94 + **4.82**·log₃(1/ε) | same |
| **Fault-tolerant T-cost per rotation** (qutrit T-factory machine) | 16.6 + **10.0**·log₃(1/ε) → **~227 T at 10⁻¹⁰** | R built from 7 T: ~776 (Householder) / ~619 (Exhaustive) at 10⁻¹⁰ → **C+D 3.4× / 2.7× cheaper** |
| **T-cost, measurement model** (level-4 = R = 4 T, both sides) | 9.7 + **6.58**·log₃(1/ε) → **~148 T at 10⁻¹⁰** | R = 4 T: ~444 / ~354 → **C+D 3.0× / 2.4× cheaper** |

- **Gadget costs:**
  - unitary model: R = level-4 = **7 T**, optimal for any number of clean ancillas;
  - measurement + Clifford feed-forward: **4 T** each, optimal within the searched classes.
- **Per-phase gate count:** the tie under A is the old "C+D ≈ C+R" result.
- **T-cost:** the robust comparison. It does not depend on the A/B choice, because the 7 R per rotation are paid in T either way.

## Decisions (PI, 2026-10-01)
1. **Counting convention: A (signed 𝒟, R ∈ C+D, R charged).** Report the per-class counts (n_T, n_4, n_R) as the primary data, with T-cost as the headline. N_φ (per-phase, R charged) is kept only for comparison with the literature. Convention B appears only to explain why the draft's numbers differ.
2. **Cost models: report both.**
   - Unitary gadgets (level-4 = R = 7 T, optimal for any number of clean ancillas): ~227 T at 10⁻¹⁰, 3.4× / 2.7× cheaper than C+R.
   - Measurement + Clifford feed-forward gadgets (4 T each): ~148 T at 10⁻¹⁰, 3.0× / 2.4× cheaper than C+R with R = 4 T.
3. **Draft corrections: tabled.** A new document will probably be written using this bundle as its base, rather than patching `ESA_CliffordD_fixed.tex`. `paper_prep/draft_corrections.md` stays as a reference list of errors not to repeat.
4. **The 7-T and 4-T gadgets go into this paper,** not a separate note.
5. **Nick:** request sent 2026-10-01; awaiting the top-K dump.

## Files

| File | Contents |
|---|---|
| `01_conventions_and_counts.md` | Conventions A/B, the three cost units (N_φ, ops, T-cost), what the counts mean, per-f tables |
| `02_why_the_per_phase_tie.md` | Galois "shadows": why C+D ties C+R in gate count; why RUS doesn't break it |
| `03_gadgets_R_and_level4.md` | **R = 7 T** and **level-4 = 7 T** with one clean ancilla; optimality with ≤2 ancillas; novelty |
| `04_fault_tolerant_cost.md` | T-cost per rotation, the device comparison, factory model (c_R/c_T) |
| `05_qutrit_vs_two_qubits.md` | Qutrit vs two-qubit emulation in consistent units (retracts the old 1.37×) |
| `06_synthesis_findings.md` | Residual-R bug, candidate selection by T-cost, null results, the "111" predictor, Euler merging |
| `tables_generated_measurement_w4-4_wr-4.md` | Same tables in the measurement model (`compute_tables.py --w4 4 --wr 4`) |
| `07_open_items.md` | What's open, and the request to Nick |
| `tables_generated.md` | All numeric tables, written by `compute_tables.py` (don't hand-edit) |
| `compute_tables.py` | Recomputes every table from `nick_test/nick_tcost_2026-09-30.csv` and the published fits. `--w4/--wr` change the gadget weights |
| `verify_all.sh` | Re-runs the checks: gadgets, emitter, certificate, tables. `--full` adds the slow ones |

## Reproduce

```bash
cd unified/results_bundle
python3 compute_tables.py          # all tables -> tables_generated.md
./verify_all.sh                    # quick checks (8/8 pass, 2026-10-01)
./verify_all.sh --full             # + reducer regression, equivalence controls, R lower bound, factory model
# recount from scratch (≈1 h, 12 procs):
python3 ../nick_test/nick_tcost_all.py --procs 12
```

Everything is on branch `cd-ft-cost` of github.com/hlammiv/Clifford_D_unifed. The detailed chronological log is in `../cd_approx_cr_notes.md`, and the earlier summary in `../cd_ft_cost_results_2026-09-30.md`.
