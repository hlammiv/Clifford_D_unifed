"""cvp — qutrit Babai-CVP synthesis (Phase 1: Gram matrix + LLL basis).

See `qutrit_babai_cvp_plan_2026-05-24` for the full 7-phase plan. Phase 1
delivers only the Gram-matrix foundation: an integer Gram `B`, a numpy-only
`q_form` matching HRSA's `ringZ9::quad()` bit-exactly, and a one-time LLL
reduction of `B` cached at module-import time.
"""
from cvp.gram import B, B_LLL_CACHE, U_lll, B_lll, lll_reduce, q_form
from cvp.babai import babai_x1, minkowski_embedding

__all__ = [
    "B",
    "B_lll",
    "U_lll",
    "B_LLL_CACHE",
    "lll_reduce",
    "q_form",
    "babai_x1",
    "minkowski_embedding",
]
