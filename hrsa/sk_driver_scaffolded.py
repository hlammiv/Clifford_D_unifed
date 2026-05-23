"""sk_driver_scaffolded.py — Solovay-Kitaev driver with multi-tier U-net scaffold.

Phase E of the SK U-net bootstrap.  Plugs :class:`u_net.scaffolded_net.ScaffoldedNet`
into the SK recursion so each level dispatches to the right-density U-net,
and short-circuits ("tier-skip") whenever a denser tier is dense enough to
already meet the residual precision without further commutator descent.

Algorithm (per call at recursion depth ``depth``)
-------------------------------------------------
1. Pick the loosest tier that satisfies ``eps_u <= target_eps * 0.5``;
   ``scaffold.closest(target, target_eps)`` returns its nearest entry.
2. Measure ``residual_frob = |target_U - V_aligned|_F`` after global-phase
   alignment (we work with SU(3) up to a phase; global phase is free).
3. If ``residual_frob <= target_eps`` → done; return.
4. If ``depth >= max_depth`` → bail out and return the base case so the caller
   can still measure how far we got.
5. Factor ``E = target_U @ V_base.dagger`` as a near-identity commutator
   ``E ≈ [A, B]`` via :func:`hrsa.su3_commutator.factor_commutator`.
6. Recurse on ``A`` and ``B`` at a tighter precision target
   ``sqrt(target_eps)`` (the SK contraction relation).  TIER-SKIP: if the
   recursive call lands inside a denser tier than the current one, accept
   that result directly without further commutator descent (the recursion
   handles this implicitly because it picks the appropriate tier at depth=0).
7. Assemble ``W = V_A · V_B · V_A.dagger · V_B.dagger · V_base`` and return.

The N_D budget is the per-leaf sum from the chosen U-net entries.  Each of
``V_A``, ``V_B`` appears twice (once as itself, once as the conjugate
transpose); we charge the dagger the same as the un-conjugated factor.

Public API
----------
sk_driver_scaffolded(target_U, target_eps, scaffold,
                     max_recurse_per_tier=3, depth=0) -> dict
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_UNIFIED = _HERE.parent
for p in (_UNIFIED, _HERE, _UNIFIED / "u_net"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from su3_commutator import factor_commutator                # noqa: E402
from u_net.scaffolded_net import ScaffoldedNet              # noqa: E402


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Per-call recursion ceiling.  At depth 0 we just dispatch to the matching
# tier; each additional level adds one commutator factorization.  Three is
# enough to take a base-case residual ~0.5 down to ~10⁻⁴ via the usual
# SK contraction (each level squares the error, roughly).
DEFAULT_MAX_RECURSE = 3

# Outside-basin guard: if the residual ``|E - I|_F`` exceeds this after the
# base case, ``factor_commutator`` is unreliable (the analytic seed for SU(3)
# requires E near the identity).  We bail out and return the base case alone.
_CONTRACT_CHECK_MAX = 0.9


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _strip_global_phase(V: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Rotate V by the global phase that minimises ``|V - U|_F``.

    For unitaries, the optimal phase is ``phi = -arg(Tr(U^† V))``; the
    minimised distance is ``|V - U|_F`` after multiplying V by
    ``e^{i phi}``.  Returns V unchanged if the inner product is tiny.
    """
    inner = np.trace(U.conj().T @ V)
    if abs(inner) < 1e-14:
        return V
    return V * (inner.conjugate() / abs(inner))


def _frob(M: np.ndarray) -> float:
    """Frobenius norm helper."""
    return float(np.linalg.norm(M, ord="fro"))


# ---------------------------------------------------------------------------
#  Recursive driver
# ---------------------------------------------------------------------------

def _sk_recurse(target_U: np.ndarray,
                target_eps: float,
                scaffold: ScaffoldedNet,
                max_recurse_per_tier: int,
                depth: int,
                state: dict) -> dict:
    """Internal recursive helper.  See ``sk_driver_scaffolded`` for the full
    contract.  Mutates ``state`` (a per-invocation accumulator).
    """
    state["tree_size"] += 1

    target_U = np.asarray(target_U, dtype=np.complex128)
    if target_U.shape != (3, 3):
        raise ValueError(f"target_U must be (3,3), got {target_U.shape}")

    # --- 1. Base case: nearest entry in the scaffold-selected tier. -------
    base = scaffold.closest(target_U, target_eps)
    V_raw = base["V_complex"]
    V_aligned = _strip_global_phase(V_raw, target_U)
    base_frob = _frob(V_aligned - target_U)

    state["log"].append({
        "depth": depth,
        "target_eps": float(target_eps),
        "tier_eps_u": float(base["tier_eps_u"]),
        "tier_index": int(base["tier_index"]),
        "coverage_ok": bool(base["coverage_ok"]),
        "base_frob": float(base_frob),
        "stored_frob": float(base["achieved_frob"]),
        "N_D_base": int(base["N_D"]),
        "event": "tier_dispatch",
    })

    # --- 2. Already inside the target?  Done. -----------------------------
    if base_frob <= target_eps:
        state["log"].append({"depth": depth, "event": "inside_target_eps",
                             "achieved_frob": float(base_frob)})
        return {
            "V": V_aligned,
            "N_D": int(base["N_D"]),
            "achieved_frob": float(base_frob),
            "depth_reached": depth,
            "tier_eps_u_used": float(base["tier_eps_u"]),
            "coverage_ok": bool(base["coverage_ok"]),
            "success": True,
        }

    # --- 3. Recursion ceiling? -------------------------------------------
    if depth >= max_recurse_per_tier:
        state["log"].append({"depth": depth, "event": "max_recurse_reached",
                             "achieved_frob": float(base_frob)})
        return {
            "V": V_aligned,
            "N_D": int(base["N_D"]),
            "achieved_frob": float(base_frob),
            "depth_reached": depth,
            "tier_eps_u_used": float(base["tier_eps_u"]),
            "coverage_ok": bool(base["coverage_ok"]),
            "success": False,
            "note": "max recursion depth reached without meeting target_eps",
        }

    # --- 4. Compute residual E = target_U · V_aligned^† ; should be ≈ I. -
    E = target_U @ V_aligned.conj().T
    err_E = _frob(E - np.eye(3, dtype=np.complex128))
    if not np.isfinite(err_E) or err_E > _CONTRACT_CHECK_MAX:
        state["log"].append({"depth": depth, "event": "outside_basin",
                             "err_E": float(err_E)})
        return {
            "V": V_aligned,
            "N_D": int(base["N_D"]),
            "achieved_frob": float(base_frob),
            "depth_reached": depth,
            "tier_eps_u_used": float(base["tier_eps_u"]),
            "coverage_ok": bool(base["coverage_ok"]),
            "success": False,
            "note": f"residual outside contraction basin (|E-I|_F={err_E:.3f})",
        }

    # --- 5. Commutator factorization. ------------------------------------
    try:
        A_target, B_target, residual_AB = factor_commutator(E)
    except Exception as exc:
        state["log"].append({"depth": depth, "event": "factor_commutator_threw",
                             "exception": repr(exc)})
        return {
            "V": V_aligned,
            "N_D": int(base["N_D"]),
            "achieved_frob": float(base_frob),
            "depth_reached": depth,
            "tier_eps_u_used": float(base["tier_eps_u"]),
            "coverage_ok": bool(base["coverage_ok"]),
            "success": False,
            "note": f"factor_commutator failed: {exc!r}",
        }

    if not np.isfinite(residual_AB) or residual_AB >= err_E:
        # Commutator did not reduce error; recursion can't help.  Return base.
        state["log"].append({"depth": depth, "event": "commutator_no_reduction",
                             "residual_AB": float(residual_AB),
                             "err_E": float(err_E)})
        return {
            "V": V_aligned,
            "N_D": int(base["N_D"]),
            "achieved_frob": float(base_frob),
            "depth_reached": depth,
            "tier_eps_u_used": float(base["tier_eps_u"]),
            "coverage_ok": bool(base["coverage_ok"]),
            "success": False,
            "note": "commutator factorization did not reduce residual",
        }

    # --- 6. Recurse on A and B at tighter precision. ---------------------
    #
    # SK contraction: a residual of magnitude eps factors into A, B both of
    # magnitude sqrt(eps).  Approximating each to within sqrt(eps') lets the
    # assembled commutator hit eps'.  So the recursive precision target is
    # the square root of the *residual* magnitude — NOT of the caller's
    # eps_total, which can be much smaller.
    sub_eps = max(math.sqrt(max(err_E, 1e-16)), 1e-10)

    # TIER-SKIP NOTE: the recursive _sk_recurse call invokes scaffold.closest
    # with target_eps=sub_eps.  pick_tier returns the LOOSEST qualifying tier
    # — so if a much denser tier already exists, the recursion happily
    # dispatches there directly at depth 0 and terminates without further
    # commutator levels.  No explicit "tier-skip" branch is needed; the
    # selection rule is self-organising.
    state["log"].append({"depth": depth, "event": "recurse_sub",
                         "sub_eps": float(sub_eps), "err_E": float(err_E),
                         "residual_AB": float(residual_AB)})

    Va_res = _sk_recurse(A_target, sub_eps, scaffold,
                          max_recurse_per_tier, depth + 1, state)
    Vb_res = _sk_recurse(B_target, sub_eps, scaffold,
                          max_recurse_per_tier, depth + 1, state)

    if Va_res.get("V") is None or Vb_res.get("V") is None:
        return {
            "V": V_aligned,
            "N_D": int(base["N_D"]),
            "achieved_frob": float(base_frob),
            "depth_reached": depth,
            "tier_eps_u_used": float(base["tier_eps_u"]),
            "coverage_ok": bool(base["coverage_ok"]),
            "success": False,
            "note": "sub-recursion returned no V",
        }

    Va = Va_res["V"]
    Vb = Vb_res["V"]

    # --- 7. Reassemble:  W = V_A V_B V_A^† V_B^† V_base. ------------------
    W = Va @ Vb @ Va.conj().T @ Vb.conj().T @ V_aligned
    W = _strip_global_phase(W, target_U)
    W_frob = _frob(W - target_U)

    # Charge N_D: base + 2*(V_A + V_B).  Daggers are charged the same as the
    # un-conjugated factors (Hermitian conjugation is structurally free but
    # we don't have a separate cost slot for it).
    N_D_total = int(base["N_D"]) + 2 * (int(Va_res["N_D"]) + int(Vb_res["N_D"]))
    depth_reached = max(int(Va_res["depth_reached"]),
                        int(Vb_res["depth_reached"]))

    state["log"].append({"depth": depth, "event": "assembled",
                         "W_frob": float(W_frob),
                         "N_D_total": int(N_D_total)})

    return {
        "V": W,
        "N_D": int(N_D_total),
        "achieved_frob": float(W_frob),
        "depth_reached": int(depth_reached),
        "tier_eps_u_used": float(base["tier_eps_u"]),
        "coverage_ok": bool(base["coverage_ok"] and Va_res.get("coverage_ok", False)
                           and Vb_res.get("coverage_ok", False)),
        "success": bool(W_frob <= target_eps),
        "residual_AB": float(residual_AB),
        "base_frob": float(base_frob),
    }


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def sk_driver_scaffolded(target_U: np.ndarray,
                         target_eps: float,
                         scaffold: ScaffoldedNet,
                         max_recurse_per_tier: int = DEFAULT_MAX_RECURSE,
                         depth: int = 0) -> dict:
    """Run a scaffolded SK descent on ``target_U`` to precision ``target_eps``.

    Parameters
    ----------
    target_U : (3, 3) complex unitary.
    target_eps : Frobenius precision target ``|W - target_U|_F``.
    scaffold : a populated :class:`ScaffoldedNet`.
    max_recurse_per_tier : recursion ceiling (default 3).  ``depth=0`` is the
        bare tier dispatch; each additional level adds one commutator
        factorization.
    depth : starting recursion depth (callers should leave this at the
        default of 0; the recursive helper passes ``depth + 1`` internally).

    Returns
    -------
    dict with::

        V              : (3, 3) complex   the assembled approximation
        N_D            : int              total D-gate budget
        achieved_frob  : float            |W - target_U|_F (post phase alignment)
        depth_reached  : int              deepest recursion level hit
        tier_eps_u_used: float            eps_u of the tier at depth=0
        coverage_ok    : bool             True iff every tier in the recursion
                                          met the COVERAGE_SLACK criterion
        success        : bool             True iff achieved_frob <= target_eps
        wall_seconds   : float
        tree_size      : int              total recursive calls (incl. base)
        log            : list[dict]       per-event recursion log (tier picks,
                                          commutator residuals, etc.)
    """
    state = {"tree_size": 0, "log": []}
    t0 = time.perf_counter()
    res = _sk_recurse(target_U, target_eps, scaffold,
                      max_recurse_per_tier, depth, state)
    dt = time.perf_counter() - t0
    out = dict(res)
    out["wall_seconds"] = float(dt)
    out["tree_size"] = int(state["tree_size"])
    out["log"] = state["log"]
    return out


__all__ = ["sk_driver_scaffolded", "DEFAULT_MAX_RECURSE"]
