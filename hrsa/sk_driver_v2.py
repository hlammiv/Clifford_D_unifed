"""Solovay-Kitaev driver v2: Euler-decomposition base case + recursive contraction.

The base case (depth 0) of v1 was `closest_in_net(target, 5184)`, which has a
hard granularity floor of ~0.3 and produces nothing in the (0.1, 0.9) shell
that SK's commutator factorization actually needs.  This v2 replaces the base
case with a *constructed* approximation:

    base_case(U) = synthesize_factor_list( euler_decompose(U), eps_per_leaf )

where `euler_decompose` writes U as a chain of 24*outer-iter factors
(('CLIFFORD', _), ('Z', 0, 1, θ), ('CLIFFORD', _))×8 per outer pass, and
`synthesize_leaf` dispatches each Z-rotation to HRSA (or to a 5184-net lookup
or to zeta9 at tighter ε).

The recursive case is unchanged from v1:  given U_0 = base_case(target), compute
the residual E = U_0^† · target and SK-factor it into [A, B], then recursively
synthesise A and B, assembling W = V_A V_B V_A^† V_B^† U_0.

Key design choices
------------------
* `eps_per_leaf` is the per-leaf precision passed to `synthesize_leaf`.  With
  ~32–48 leaves per Euler decomposition, a target total error ε_total at this
  recursion depth is split as `eps_per_leaf = ε_total / (2 * n_leaves)`
  (factor-of-2 slack vs RSS bound).
* Words are not stored as flat index lists in v2 (since most factors aren't in
  the 5184-net).  Instead we return the realised matrix V_total alongside the
  N_D budget.

Public API
----------
sk_synthesize_v2(target, depth, eps_total) -> dict
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from su3_euler import euler_decompose, verify_decomposition
from sk_leaves import synthesize_factor_list, _build_RZ
from su3_commutator import factor_commutator
import sk_ringZ9 as rk


# ---------------------------------------------------------------------------
#  Base case: Euler decompose + leaf synth
# ---------------------------------------------------------------------------

_CONTRACT_CHECK_MAX = 0.9  # broader than v1's 0.7 since base case is constructive


def _euler_base_case(target: np.ndarray, eps_total: float,
                     state: dict) -> dict:
    """Run the Euler base case at the requested precision target.

    Returns dict with:
      'V'         : (3,3) complex synthesised matrix
      'N_D'       : int  total D-gate budget across all leaves
      'frob_error': float |V - target|_F
      'n_leaves'  : int  number of Z-rotation leaves
      'success'   : bool whether all leaves met their per-leaf threshold
      'euler_err' : float residual of the Euler decomposition itself
    """
    state["base_case_calls"] += 1

    # 1. Euler decomposition.
    factors, diag = euler_decompose(target, return_diagnostics=True)
    euler_err = diag["final_frob_error"]

    # 2. Choose eps_per_leaf.  With n_leaves Z-rotations whose synthesis errors
    # we treat as independent and assume worst-case linear accumulation, set
    # eps_per_leaf = eps_total / (2 * n_leaves).  RSS bound is tighter but we
    # don't rely on cancellation.
    # Count Z-leaves in the factor list.
    n_leaves = sum(1 for f in factors if f[0] == "Z")
    if n_leaves == 0:
        # Target is already identity (within Euler convergence tol).
        I3 = np.eye(3, dtype=complex)
        return {
            "V": I3,
            "V_ring": rk.ringmat_identity(),
            "ring_valid": True,
            "N_D": 0,
            "frob_error": float(np.linalg.norm(I3 - target, ord="fro")),
            "n_leaves": 0,
            "success": True,
            "euler_err": euler_err,
        }

    # HRSA wall-time is highly sensitive to eps_per_leaf (eps=0.05 ≈ 3 s, eps=0.005 ≈ several minutes
    # at max_f=4).  We keep eps_per_leaf >= 0.05 by default — the corresponding
    # worst-case base-case error from a 32-leaf Euler is 32·0.05 = 1.6 (linear)
    # or √32·0.05 ≈ 0.28 (RSS).  The 0.28 RSS lands at the edge of SK's
    # contraction basin; one SK level brings it to ~0.08, two to ~0.013, etc.
    # Callers that want tighter eps_per_leaf must accept much longer wall-times.
    eps_per_leaf = max(eps_total / (2.0 * n_leaves), 0.05)

    # 3. Synthesise.
    synth = synthesize_factor_list(factors, eps_per_leaf=eps_per_leaf)
    V = synth["V_total"]
    err = float(np.linalg.norm(V - target, ord="fro"))

    return {
        "V": V,
        "V_ring": synth.get("V_ring_total"),
        "ring_valid": bool(synth.get("ring_valid", False)),
        "N_D": synth["N_D_total"],
        "frob_error": err,
        "n_leaves": synth["n_leaves"],
        "success": synth["success"],
        "euler_err": euler_err,
        "eps_per_leaf": eps_per_leaf,
    }


# ---------------------------------------------------------------------------
#  Recursion
# ---------------------------------------------------------------------------

def _sk_recurse(target: np.ndarray, depth: int, eps_total: float,
                state: dict) -> dict:
    """Internal recursive helper.  Returns the same dict shape as _euler_base_case
    plus a 'depth' key."""
    state["tree_size"] += 1

    if depth == 0:
        out = _euler_base_case(target, eps_total, state)
        out["depth"] = 0
        return out

    # (a) Build U_0 via the Euler base case at the *current* depth's loose eps.
    # Heuristic: at recursion depth k, the base case eps_target is eps_total^{2^{-k}}.
    # We use a multiplicative descent to land in the contraction basin at every
    # depth.
    base_eps = max(eps_total ** (1.0 / (2.0 ** depth)) if eps_total < 1 else 0.3,
                   0.05)
    base = _euler_base_case(target, base_eps, state)
    V0 = base["V"]
    if V0 is None:
        # Base case totally failed; bail out.
        return {"V": None, "V_ring": None, "ring_valid": False,
                "N_D": None, "frob_error": None,
                "n_leaves": 0, "success": False, "depth": depth}

    # (b) Residual E = V_0^† · target.
    E = V0.conj().T @ target
    err_E = float(np.linalg.norm(E - np.eye(3), ord="fro"))

    if err_E > _CONTRACT_CHECK_MAX or not np.isfinite(err_E):
        # Outside the contraction basin; just return the base case.
        out = dict(base)
        out["depth"] = depth
        out["note"] = "outside contraction basin; base only"
        return out

    # (c) Factor E ≈ [A, B].
    try:
        A, B, residual_AB = factor_commutator(E)
    except Exception:
        out = dict(base); out["depth"] = depth
        out["note"] = "factor_commutator threw"
        return out

    if not np.isfinite(residual_AB) or residual_AB > err_E:
        out = dict(base); out["depth"] = depth
        out["note"] = "commutator did not reduce residual"
        return out

    # (d) Recurse on A and B with tighter precision target.  Each subproblem
    # gets ~ sqrt(eps_total) precision; the SK contraction then yields
    # eps_total on the assembled word.
    sub_eps = max(math.sqrt(max(eps_total, 1e-16)), 1e-8)
    Va_res = _sk_recurse(A, depth - 1, sub_eps, state)
    Vb_res = _sk_recurse(B, depth - 1, sub_eps, state)
    if Va_res.get("V") is None or Vb_res.get("V") is None:
        out = dict(base); out["depth"] = depth
        out["note"] = "sub-recursion failed"
        return out

    # (e) Assemble W = V_A · V_B · V_A^† · V_B^† · V_0.
    Va, Vb = Va_res["V"], Vb_res["V"]
    W = Va @ Vb @ Va.conj().T @ Vb.conj().T @ V0

    # Assemble the ringZ9 version in parallel.  All three sub-results
    # (Va_res, Vb_res, base) must have valid ringZ9 representations; otherwise
    # the assembled W is no longer exact.
    Va_ring = Va_res.get("V_ring")
    Vb_ring = Vb_res.get("V_ring")
    V0_ring = base.get("V_ring")
    ring_valid_W = bool(
        Va_res.get("ring_valid") and Vb_res.get("ring_valid")
        and base.get("ring_valid")
        and Va_ring is not None and Vb_ring is not None and V0_ring is not None
    )
    W_ring = None
    if ring_valid_W:
        try:
            Va_dag_ring = rk.ringmat_dagger(Va_ring)
            Vb_dag_ring = rk.ringmat_dagger(Vb_ring)
            W_ring = rk.chain_mul([Va_ring, Vb_ring, Va_dag_ring, Vb_dag_ring, V0_ring])
            # Cheap reduction: divide out shared factors of 3 to keep denoms small.
            W_ring = rk.ringmat_reduce_by_three(W_ring)
        except Exception:
            W_ring = None
            ring_valid_W = False

    err_W = float(np.linalg.norm(W - target, ord="fro"))

    N_D_total = (base.get("N_D", 0) or 0) \
                + 2 * ((Va_res.get("N_D") or 0) + (Vb_res.get("N_D") or 0))
    # Each of V_A, V_B appears twice (V_A and V_A^†).  We assume Hermitian
    # conjugation is free (it would be in a structured word representation, but
    # since our base case returns a single matrix product we don't have an
    # honest cost for the daggers; treat them as identical-cost mirror images).

    return {
        "V": W,
        "V_ring": W_ring,
        "ring_valid": ring_valid_W,
        "N_D": N_D_total,
        "frob_error": err_W,
        "n_leaves": (Va_res.get("n_leaves", 0) + Vb_res.get("n_leaves", 0)) * 2
                    + (base.get("n_leaves") or 0),
        "depth": depth,
        "success": True,
        "base_err": base.get("frob_error"),
        "residual_AB": residual_AB,
    }


def sk_synthesize_v2(target: np.ndarray, depth: int,
                     eps_total: float = 0.01) -> dict:
    """Run SK with the Euler-base-case at the requested depth.

    Parameters
    ----------
    target    : (3,3) complex unitary to approximate
    depth     : SK recursion depth (0 = pure Euler base case)
    eps_total : end-to-end Frobenius target |W − target|_F.  Used to
                allocate per-leaf precision in the base case.

    Returns dict with:
      'V'           : assembled approximation
      'final_error' : |V − target|_F
      'depth'       : echoed depth
      'tree_size'   : int total recursive calls made
      'N_D'         : int total D-gate budget
      'base_case_calls': int number of base-case (Euler) invocations
      'n_leaves'    : int Z-rotation leaf count summed over the tree
    """
    state = {"tree_size": 0, "base_case_calls": 0}
    t0 = time.perf_counter()
    res = _sk_recurse(target, depth, eps_total, state)
    dt = time.perf_counter() - t0
    return {
        "V": res.get("V"),
        "V_ring": res.get("V_ring"),
        "ring_valid": bool(res.get("ring_valid", False)),
        "final_error": res.get("frob_error"),
        "depth": depth,
        "tree_size": state["tree_size"],
        "base_case_calls": state["base_case_calls"],
        "N_D": res.get("N_D"),
        "n_leaves": res.get("n_leaves", 0),
        "wall_seconds": dt,
        "note": res.get("note"),
        "success": res.get("success", False),
    }


# ---------------------------------------------------------------------------
#  __main__ : self-test
# ---------------------------------------------------------------------------

def _fmt_row(label, r):
    err = r.get("final_error")
    err_s = f"{err:.3e}" if err is not None else "n/a"
    nd = r.get("N_D")
    nd_s = f"{nd}" if nd is not None else "n/a"
    return (f"  {label}: err={err_s:>10}  N_D={nd_s:>5}  "
            f"tree={r['tree_size']:>4}  base_calls={r['base_case_calls']:>3}  "
            f"t={r['wall_seconds']:.2f}s")


if __name__ == "__main__":
    print("sk_driver_v2 self-test: depth 0..2 on R_Z_{(0,1)} of pi/4")
    target = _build_RZ(0, 1, math.pi / 4.0)
    for d in range(3):
        r = sk_synthesize_v2(target, depth=d, eps_total=0.05)
        print(_fmt_row(f"depth={d}", r))
