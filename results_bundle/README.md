# Qutrit Clifford+D vs Clifford+R: results bundle (2026-10-01)

This bundle collects every result from the 2026-09-29 → 10-01 work, with the code to recompute it. **The counting convention is not decided yet.** Results that depend on it are given under both conventions, A and B, defined in `01`.

## Headline

| | C+D (this work) | C+R (Gustafson et al. 2503.20203) |
|---|---|---|
| Per-phase gate count, convention A (R charged) | 3.94 + **5.16**·log₃(1/ε) | Householder 5.14, Exhaustive 4.11 ·log₃(1/ε) |
| Per-phase gate count, convention B (R free) | 2.94 + **4.82**·log₃(1/ε) | same |
| **Fault-tolerant T-cost per rotation** (qutrit T-factory machine) | 16.6 + **10.0**·log₃(1/ε) → **~227 T at 10⁻¹⁰** | R built from 7 T: ~776 (Householder) / ~619 (Exhaustive) at 10⁻¹⁰ → **C+D 3.4× / 2.7× cheaper** |

- **Per-phase gate count:** the tie under A is the old "C+D ≈ C+R" result.
- **T-cost:** the robust comparison. It does not depend on the A/B choice, because the 7 R per rotation are paid in T either way.

## Files

| File | Contents |
|---|---|
| `01_conventions_and_counts.md` | Conventions A/B, the three cost units (N_φ, ops, T-cost), what the counts mean, per-f tables |
| `02_why_the_per_phase_tie.md` | Galois "shadows": why C+D ties C+R in gate count; why RUS doesn't break it |
| `03_gadgets_R_and_level4.md` | **R = 7 T** and **level-4 = 7 T** with one clean ancilla; optimality with ≤2 ancillas; novelty |
| `04_fault_tolerant_cost.md` | T-cost per rotation, the device comparison, factory model (c_R/c_T) |
| `05_qutrit_vs_two_qubits.md` | Qutrit vs two-qubit emulation in consistent units (retracts the old 1.37×) |
| `06_synthesis_findings.md` | Residual-R bug, candidate selection by T-cost, null results, the "111" predictor, Euler merging |
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
