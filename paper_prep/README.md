# Qutrit Clifford+D vs Clifford+R: memo and notes package (2026-09-30)

All files live in `unified/` on branch `cd-ft-cost`.

## Read in this order
1. **`paper_prep/one_pager_2026-10.md`**: the one-page summary for collaborators. The per-phase gate count ties with C+R, but on a qutrit T-factory machine C+D is ~3.4× cheaper.
2. **`tcost_vs_eps_2026-09-30.png`** (from `plot_tcost_vs_eps.py`): the headline figure. Left: per-phase tie. Right: T-cost.
3. **`cd_ft_cost_results_2026-09-30.md`**: the summary of record, with every number, table and file pointer.
4. **`paper_prep/convention_memo.md`**: **decision needed.** Signed 𝒟 (R ∈ C+D) vs the draft's unsigned definition, and how to report counts (n_T, n_4, n_R and T-cost; rename per-phase N_D → N_φ).
5. **`paper_prep/draft_corrections.md`**: 15 find/replace LaTeX patches for `ESA_CliffordD_fixed.tex`. The l.606 one is required; others fix the R accounting, the "sde 0 = Clifford" claim and a target typo.
6. **`paper_prep/novelty_check.md`**: novelty of the 7-T constructions.
   - R = 7 T, optimal with one clean ancilla, against 39 T published: about 85% confident it is new.
   - Level-4 diag(1,1,ζ₉) = 7 T, against 224 T for the only explicit prior construction: likely new as an explicit count. The kickback gadget itself is standard.
7. **`cd_approx_cr_notes.md`**: the chronological working log, with derivations: Galois shadows, why RUS does not break the tie, the literature reviews. Superseded blocks are marked ⚠.

## Supporting studies (each has scripts and data)
| Directory | What |
|---|---|
| `r_from_d/` | R = 7 T, with the exhaustive search for optimality (`verify_R7T.py`, `raw_lb_1anc.py`) |
| `level4/` | Level-4 = 7 T, with the optimality search (`l4_7T_verify.py`, `l4_orbits.py`). `novelty/` counts the prior construction (224 T) |
| `compiler/cd_to_ct.py` | Emits verified Clifford+T circuits from C+D decompositions. Emitted T-count = formula |
| `factory_model/` | R-state vs T-state factory costs: c_R/c_T ≥ 16 (optimistic) up to 10⁸; R = 7 T gives 7 |
| `nick_test/nick_tcost_2026-09-30.csv` | T-cost recount of Nick's 900 approximants with the residual-R fix |
| `r_count_study/`, `tcost_selection_study/` | Candidate selection by T-cost. Best-of-100 saves ~42%; extrapolated slope 10.0 → ~8.3 |
| `hrsa/` | Residual-R bug fix (Python and both C++ paths), `HRSA_tester --rank-tcost`, `canonical_reducer.set_selection_cost` |

## Decisions and open items
- **For the PI:**
  - the counting convention (memo);
  - whether to apply the draft corrections;
  - whether the 7-T R / level-4 results become a separate short note.
- **For Nick:** the request below.
- **Settled:** 7 T is optimal for R and level-4 with ≤2 clean ancillas (`two_ancilla/`).
- **Qutrit vs two qubits** (results §4): ~1.36× vs deterministic qubit synthesis and ~2.9× vs qubit RUS, in magic-state units. The old 1.37× mixed units and is retracted.

## Request to Nick: top-K candidates per angle

**Ready-to-send package: `nick_request/`.** It contains `INSTRUCTIONS_for_Nick.md` and `topk_dump.patch` (local zeta9 branch `topk-dump`, commit 853a8da; tests in `nick_request/tests/` pass). Our side is `analyze_topk.py`.

**Goal.** Test T-cost-aware selection at f = 12–16 on his exact-ring pipeline. On HRSA at f ≤ 4, best-of-100 saved ~42% of the T-cost, and the extrapolated slope falls from 10.0 to ~8.3 per log₃(1/ε).

**Ask.** For each θ and f he already ran, keep every exact approximant within some tolerance of the best Frobenius error, rather than only the best. Suggested tolerance: frob ≤ 1.25 × best, capped at K = 1000 per θ. Use the same text format as `fits_f=*.txt` (θ, then 9 groups of 6 integers = 3^f·M), and add one column with the achieved Frobenius distance. If it is easier, the 18-integer Householder vector x (D = X₀₁(I − 3^f x x†)) is enough, at one third of the size.

**File size**, measured from his current files at ~200 B per matrix (f = 4) to ~490 B (f = 16), ≈ 25 B per f-step:

| Scope | Matrices | Text | gzip (×0.37) |
|---|---|---|---|
| 150 θ × 7 f-levels × K = 100 | 1.05×10⁵ | ~37 MB | ~14 MB |
| 150 θ × 7 f × K = 1000 | 1.05×10⁶ | ~370 MB | ~140 MB |
| ~1000 θ × 7 f × K = 1000 (everything) | 7×10⁶ | ~2.5 GB | ~0.9 GB |
| Same with the Householder vector only | | ÷3 | ÷3 |

**Storage is not the issue.** The open question is whether his pipeline still holds the candidate pools; `fit_theta_all_best_one_theta.py` keeps only the best. If his `D/` intermediates still exist, the dump is a re-scan. If not, the final selection stage has to be re-run. The analysis cost on our side scales from `nick_test/nick_tcost_all.py`, which does the 900 current matrices in ~1 h on 12 cores. 10⁵ candidates would be ~4–5 days on this machine, or hours on Lenore. Sub-sampling θ (e.g. 20 angles × K = 100 per f) keeps it to ~half a day.
