"""cvp — qutrit Babai-CVP synthesis (Phases 1-4).

See `qutrit_babai_cvp_plan_2026-05-24` for the full 7-phase plan.

Implemented so far:

* Phase 1 — Gram matrix + LLL basis (`cvp.gram`)
* Phase 2 — 6-D Babai CVP for x_1 (`cvp.babai`)
* Phase 3 — Diophantine (x_2, x_3) via zeta9 PARI (`cvp.diophantine`)
* Phase 4 — Reify (x_1, x_2, x_3, f) -> Householder V + canonical D-count
            (`cvp.reify`)
"""
from cvp.gram import B, B_LLL_CACHE, U_lll, B_lll, lll_reduce, q_form
from cvp.babai import babai_x1, minkowski_embedding
from cvp.diophantine import (
    bb_to_real_coeffs,
    enumerate_x3,
    solve_q_norm,
    solve_x2_x3,
    solve_x2_x3_ring_unitary,
    verify_pair,
    householder_frobenius,
)
from cvp.reify import (
    reify_householder,
    decompose_to_gates,
    cvp_compile_one,
)

__all__ = [
    # Phase 1
    "B",
    "B_lll",
    "U_lll",
    "B_LLL_CACHE",
    "lll_reduce",
    "q_form",
    # Phase 2
    "babai_x1",
    "minkowski_embedding",
    # Phase 3
    "bb_to_real_coeffs",
    "enumerate_x3",
    "solve_q_norm",
    "solve_x2_x3",
    "solve_x2_x3_ring_unitary",
    "verify_pair",
    "householder_frobenius",
    # Phase 4
    "reify_householder",
    "decompose_to_gates",
    "cvp_compile_one",
]
