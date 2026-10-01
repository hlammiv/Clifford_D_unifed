# 07: Open items

## Decisions (PI)
1. **Counting convention:** A (signed, R charged) vs B (unsigned, R free). See `01`. The recommendation is A, with per-class counts plus T-cost as the headline. Deferred for now; both are reported.
2. **Draft corrections:** 15 patches in `paper_prep/draft_corrections.md`. l.606 is required; others include l.584, l.586, l.329, Algorithm 6 and eq. (37). Apply them once the convention is chosen.
3. **Whether the 7-T gadgets become a standalone short note or an appendix** (`03`).

## Waiting on Nick
- **Top-K candidates per θ** (frob ≤ 1.25 × best, up to K = 100–1000). The package is `nick_request/` (`INSTRUCTIONS_for_Nick.md`, `topk_dump.patch`, also zipped as `nick_stuff.zip`).
  - At f ≥ 6 his current pools have no near-neighbours, so he needs a larger `--top_n`.
  - Size: ~40 MB for 150 θ × 7 f × K = 100.
- **Our side:** `nick_request/analyze_topk.py --k 100 --max-theta 20 --procs 8`, about half a day locally on a subsample.
- **This confirms or refutes the slope estimate** of 10.0 → ~8.3 T per log₃ at f = 12–16.

## Open research questions
- **Gadget optimality with ≥3 ancillas, measurement or catalysts.** ≤2 clean ancillas is settled (7 T). Three ancillas is infeasible with the current search; it would need new symmetry reduction.
- **A non-trivial analytic T-count lower bound.** Mana gives ≥1; thauma or stabilizer extent are not computed.
- **A rigorous version of the shadow counting argument** (`02`): A(ε) for full unitaries and the density of solvable norm equations.
- **Factory space-time volume for qutrit T-states vs qubit T-states.** It is needed for a firm qutrit-vs-qubit statement (`05`).
- **Qutrit RUS** (tabled). It cannot break the C+D/C+R tie, but could narrow the gap to qubit RUS.
- **An approximation stage that targets low T-cost directly**, rather than filtering error-optimal candidates.
