# Novelty check: 7-T qutrit R and 7-T level-4 diagonal (one clean ancilla)

*2026-09-30. Literature search by a sub-agent: arXiv full texts grepped with pdftotext, Semantic Scholar forward citations, and web search. The key numbers were spot-checked by hand against the downloaded texts. PDFs and texts are in the session scratchpad and not in the repo.*

**Claims under test** (`unified/cd_ft_cost_results_2026-09-30.md` §3; both circuits re-verified 2026-09-30, deviation ≤ 3×10⁻¹⁵):
- **(a)** R = diag(1,1,−1) exactly, deterministically and unitarily, with **T-count 7** and one clean ancilla. It is optimal for one ancilla: an exhaustive merge-free search finds nothing with ≤ 6 T. Files: `unified/r_from_d/verify_R7T.py`, `R_7T.json`, `raw_lb_1anc.py`.
- **(b)** diag(1,1,ζ₉), a level-4 diagonal, with **T-count 7** and one clean ancilla. It is optimal for one ancilla (exact orbit MITM), and simple in form: |2⟩-controlled X (3 T) → T on the ancilla → uncompute (3 T). Files: `unified/level4/l4_7T_verify.py`, `l4_orbits.py`.

---

## 1. What exists

| arXiv | Paper | What it says relevant to (a)/(b) |
|---|---|---|
| 2202.09235 | Glaudell–Ross–van de Wetering–Yeh (GRVY), *Qutrit metaplectic gates are a subset of Clifford+T* (TQC 2022) | See the list below this table. |
| 2204.00552 | Yeh–van de Wetering, *Constructing all qutrit controlled Clifford+T gates in Clifford+T* | Gives only polynomial T-counts, O(k^3.585), for \|2⟩^⊗k-controlled gates. **Cor. 4:** \|2⟩^⊗k-controlled ζI "without ancillae", equivalently \|2⟩^⊗(k−1)-controlled Z(0,1/3). For k = 1 that is diag(1,1,ζ₉) on one qutrit, with a second qutrit acted on trivially (in effect a borrowed qutrit). **It proves (b) exists; it gives no explicit T-count.** Built from Lemma 11 (Eq. (26)) and Lemma 9, and shown only as a picture. |
| 2204.13681 | van de Wetering–Yeh, *Building qutrit diagonal gates from phase gadgets* | Uses R = Z(0,3/2) as a primitive and counts in R-count. Its constructions are O(n^2.585) or O(2ⁿ). No Clifford+T count for R and no ancilla construction of diag(1,1,ζ₉). |
| 2405.08136 | Glaudell–Ross–van de Wetering–Yeh, *Exact synthesis of multiqutrit Clifford-cyclotomic circuits* | App. A: (−1)[x] (R for n = 1) needs 1 borrowed ancilla over Clifford+T (Prop. A.16). (ω₂)[x] with ω₂ = ζ₉, i.e. diag(1,1,ζ₉), needs 2 borrowed ancillae (Lemma A.15). **Ancilla counts only, no T-counts.** |
| 2405.08147 | Kalra–Saikia–Valluri–Winnick–Yard, *Multi-qutrit exact synthesis* | Every unitary over Z[1/3, ζ₉] is exact Clifford+T with ≤ 2 ancillae (Thm 2). diag(1,1,ζ) is **not** in single-qutrit C+T. Cites GRVY for R. No T-counts. |
| 1605.02756 | Bocharov–Roetteler–Svore, *Factoring with qutrits* | C₂(INC) costs 3 P9 with no ancilla, which is the source of the 3-T \|2⟩-controlled X. App. A: an R\|2⟩ magic state by a **probabilistic** RUS circuit on 4 qutrits with expected P9-count 27/4 (checked: l.1301 of the text), then RUS injection of R. **Non-unitary and probabilistic**, but numerically the closest competitor to (a). |
| Yeh, Oxford DPhil thesis (2025) | *Transdimensional quantum computation* | Thm 8.35 still quotes **T-count 39** for R (checked: l.7288). This is the state of the art as the authors themselves see it. |
| 2603.04548 | Li–Yeh, *Transversal AND in quantum codes* | Qutrit AND with T-count 3, CCX 12, CⁿX 6n. R is treated as a separate primitive. |
| 2609.29884 | Saha–Arzani (Sep 2026) | Multi-controlled Toffoli at 6n+3 P9. Nothing on R or ζ₉ phases. |
| Also checked, nothing relevant | 2303.12979, 2504.12710, 2507.09781 (qutrit Toffoli / T-count / ZX-type optimization); 2503.20203, 2410.16414 (Gustafson et al.); 2311.08696 (Kalra); 2401.16120 (EP); 2510.11526; 2604.23007; 2607.08200; 2506.10945 | No T-count for R or for a ζ₉ single-position phase. |

**GRVY 2202.09235 in detail** (each count checked against the text):
- Lemma 17: |2⟩-controlled X, **3 T**, no ancilla.
- Lemma 18: |2⟩-controlled τ₁₂, 15 T.
- Lemma 19: |2⟩-controlled S†, up to a controlled phase ζ, **8 T**.
- Cor. 20: |2⟩-controlled Z(1,1), 8 T.
- Lemma 21: |2⟩-controlled −H†, 24 T.
- **Thm 22: R with T-count 39 using one borrowed ancilla** (15 + 3·8).
- Prop. 35: R is not in single-qutrit C+T.
- The conclusion calls 39 "rather inefficient" and names a lower-T R as future work.

## 2. Closest prior results

**(a) R.**
- Best published deterministic unitary construction: **39 T, one borrowed ancilla** (GRVY Thm 22; repeated in Yeh's 2025 thesis).
- Our "24 T with a clean ancilla" is not stated anywhere. It is an immediate corollary of GRVY Eq. (16): with the ancilla in |0⟩ the 15-T controlled τ₁₂ drops out. **Cite it as "implicit in GRVY", not as prior art.**
- Probabilistic alternative: BRS, R-state at expected 27/4 P9, plus RUS injection at about 3 R-states per R.
- **No lower bound or optimality result for the T-count of R was found for any number of ancillas.**

**(b) diag(1,1,ζ₉).**
- Existence without counts: Yeh–vdW Cor. 4 (k = 1), and GRVY 2405.08136 Lemma A.15.
- Implicit **8 T** with a clean ancilla: GRVY Lemma 19 applied to a |0⟩ target, since S†|0⟩ = |0⟩ and the controlled ζ kicks back. This is our previous-best "8".
- The 7-T circuit composes published pieces (BRS/GRVY 3-T |2⟩-controlled X) in the textbook compute–phase–uncompute pattern.
- We found it neither written down nor counted.
- **Open check.** Hand-count the T gates in Yeh–vdW Cor. 4 at k = 1 (Eq. (26) plus Lemma 9, dirty target). The sub-agent estimates 8–11. If it came out ≤ 7, "prior best 8" would need rewording. Because that construction uses a borrowed qutrit rather than a clean one, it would still not undercut a 7-T clean-ancilla claim unless it is ≤ 6.

## 3. Confidence that our results are new

| Claim | Confidence new | Notes |
|---|---|---|
| (a) R in 7 T with 1 clean ancilla | **~85%** | A 5.6× drop from the published 39, and the 2025 thesis still quotes 39. Risk: circuits that exist only in figures of 2204.13681 or 2405.08136, and 2026 preprints not yet indexed. |
| (a′) optimality: ≤ 6 T impossible with 1 ancilla | **~90%** | No qutrit T-count lower bounds for R appear anywhere. This is the most clearly new element. |
| (b) diag(1,1,ζ₉) in 7 T with 1 clean ancilla | **~70% as a stated result; low as a technique** | Referees are likely to call the circuit a folklore corollary of the 3-T controlled X. |
| (b′) optimality with 1 ancilla, and impossible with 0 | ~80% for optimality | The 0-ancilla impossibility is essentially Kalra et al. 2405.08147 (not in single-qutrit C+T) or a determinant argument. Cite it rather than claim it. |

## 4. Recommendations for the paper

1. **Present (a) as a result (a lemma or theorem)** together with the optimality statement. Compare it with GRVY's 39 (borrowed ancilla) and, explicitly, with the implicit 24 T from GRVY's construction on a clean ancilla. Contrast it with BRS's probabilistic R-state route, which is non-unitary and has variable latency.
2. **Present (b) as a remark or short lemma.** Give the circuit C2X · T_anc · C2X†, attribute the controlled X to BRS/GRVY Lemma 17, and state the 8 → 7 improvement over the Lemma 19 kickback. Claim novelty for the one-ancilla optimality, not for the circuit.
3. **State the ancilla model precisely:** one *clean* ancilla, returned to |0⟩, reusable. Optimality is only for one ancilla; 2 ancillas with ≤ 6 T is open (level-4: ≥ 6 is rigorous).
4. **Before submission:**
   - Hand-count Yeh–vdW Cor. 4 at k = 1.
   - Run a Semantic Scholar or Google Scholar forward-citation sweep of 2405.08136 and 2204.13681. These were rate-limited in this pass.
   - Look for any 2026 qutrit ZX/T-count optimization preprint, e.g. a PyZX-qudit or qutrit-ZX T-count reduction paper, that may have rediscovered a small R circuit.

## Update 2026-09-30: T-count of the prior level-4 construction
A direct count of Yeh–van de Wetering arXiv:2204.00552 Cor. 4 at k=1 gives **224 T with 2 borrowed qutrits** (script `level4/novelty/yeh_vdw_cor4_tcount.py`, exact matrix check). Breakdown: |2>-ctrl S† 8 T; |22>-ctrl X01 51 T; |22>-ctrl X 108 T; |2>-ctrl Z(0,1) 216 T; total 216 + 8.

No paper states any explicit T-count for diag(1,1,ζ₉).

Suggested wording: "To our knowledge, this is the first explicit T-count for an exact Clifford+T implementation of diag(1,1,ζ₉). Prior constructions (2204.00552 Cor. 4; 2405.08136 Lemma A.[w2]) establish existence with borrowed ancillae but state no T-count; a direct count of the former gives 224 T."

Caveats to state:
- The compute/phase/uncompute kickback gadget itself is standard.
- Optimality rests on our exhaustive search (l4_orbits.py): 2-qutrit Clifford group, ancilla returned to |0>.

## Update 2026-10-01: optimality with two ancillas
An exact search (`two_ancilla/`) rules out ≤6 T with two clean ancillas for both R and diag(1,1,ζ₉).

Suggested wording: "T-count 7 is optimal among circuits using at most two clean ancillas (exhaustive search over all Clifford-conjugated T-rotation products, t ≤ 6); the general-ancilla case remains open."
