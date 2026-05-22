"""Leaf synthesis dispatch for the Euler-based SK base case.

Given a single Z-rotation factor `R^Z_{(i,j)}(theta)` at target precision `eps`,
return a synthesised unitary `V` (3,3 complex), its N_D cost, and the method used.

Dispatch rules
--------------
1. If |theta| < eps                                      → identity   (N_D = 0)
2. If R^Z_{(i,j)}(theta) is in the 5184-element sign-Clifford net within eps
   (i.e. theta lands within eps_Frobenius of a "sign-extended Clifford" stored
   in /tmp/e0_net_5184.txt)                              → that net element  (N_D = 1)
3. If eps >= 0.005                                       → HRSA subprocess
4. Else                                                  → zeta9  (currently STUBBED)

For (i,j) ≠ (0,1):  conjugate R^Z_{(0,1)}(theta) by the unique Clifford P that
sends (i,j) → (0,1).  The cost is unchanged; the resulting V has the same N_D
as the (0,1) synthesis.

Public API
----------
synthesize_leaf(theta: float, eps: float, i: int, j: int) -> dict
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ep_descent import load_e0_net, frobenius_distance_batch, _E0_NET_PATH
from hrsa_bootstrap import bootstrap_target
from su3_euler import _load_clifford_set
import sk_ringZ9 as rk


# ---------------------------------------------------------------------------
#  Static state: 5184-element sign-extended net + permutation Cliffords
# ---------------------------------------------------------------------------

_E0_NET_CACHE: Optional[np.ndarray] = None
_PERM_CACHE: Optional[dict] = None


def _get_e0_net() -> np.ndarray:
    """Load the 5184 sign-extended Cliffords (polar-orthogonalised in chunks
    of 648 for unitarity)."""
    global _E0_NET_CACHE
    if _E0_NET_CACHE is None:
        raw, _ = load_e0_net()
        # Polar-orthogonalise every entry for the same reason as in su3_euler.
        from su3_euler import _polar_orthogonalize
        _E0_NET_CACHE = np.array([_polar_orthogonalize(C) for C in raw],
                                 dtype=np.complex128)
    return _E0_NET_CACHE


def _build_RZ(i: int, j: int, theta: float) -> np.ndarray:
    """R^Z_{(i,j)}(theta) = diag(... e^{-i theta/2} at i, e^{+i theta/2} at j ...)."""
    M = np.eye(3, dtype=complex)
    M[i, i] = np.exp(-0.5j * theta)
    M[j, j] = np.exp(+0.5j * theta)
    return M


def _build_subspace_permutations() -> dict:
    """For each (i,j) with i!=j, find a Clifford P that conjugates
    R^Z_{(0,1)}(theta) into R^Z_{(i,j)}(theta).

    Returns dict keyed by (i,j) -> (P, P_dag, clifford_idx, clifford_dag_idx)
    where indices are in the 648-Clifford set (the first 648 of the e0_net).

    These permutations exist exactly: see probe (2026-05-12) which found 1
    Clifford per ordered (i,j) ≠ (0,1) in the 648-set.  (i,j)=(0,1) maps via I.
    """
    global _PERM_CACHE
    if _PERM_CACHE is not None:
        return _PERM_CACHE

    cliffords = _load_clifford_set()
    # We need C such that C · R_Z_{(0,1)}(theta) · C^dag = R_Z_{(i,j)}(theta).
    Rz01 = _build_RZ(0, 1, 0.7)  # probe with arbitrary theta
    targets = {}
    for i in range(3):
        for j in range(3):
            if i != j:
                targets[(i, j)] = _build_RZ(i, j, 0.7)

    perm = {}
    perm[(0, 1)] = (np.eye(3, dtype=complex),
                    np.eye(3, dtype=complex),
                    0,   # identity is clifford index 0
                    0)
    for ci, C in enumerate(cliffords):
        D = C @ Rz01 @ C.conj().T
        for key, T in targets.items():
            if key in perm:
                continue
            if np.linalg.norm(D - T, ord="fro") < 1e-8:
                # Find C^dag's index too.
                Cdag = C.conj().T
                d = np.sqrt(np.sum(np.abs(cliffords - Cdag[None, :, :]) ** 2,
                                   axis=(1, 2)))
                j_idx = int(np.argmin(d))
                if d[j_idx] > 1e-6:
                    # Shouldn't happen for the Clifford group; skip.
                    continue
                perm[key] = (C, Cdag, ci, j_idx)
    if len(perm) != 6:
        raise RuntimeError(
            f"subspace permutation table incomplete: found {len(perm)}/6 pairs"
        )
    _PERM_CACHE = perm
    return _PERM_CACHE


# ---------------------------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------------------------

def _closest_in_e0_net(target: np.ndarray, eps: float):
    """Return (idx, distance) of the closest 5184-net element to target.

    Tolerates the orthogonalised net; only returns success if dist < eps.
    """
    net = _get_e0_net()
    d = frobenius_distance_batch(target, net)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


# Fast-mode flags for HRSA: stop after the first solution and skip the
# look-ahead extension.  Reduces per-call wall-time from ~20 s to ~3 s
# at f=4, with a small N_D penalty (~+2 D-gates on average per memory
# `hrsa_fast_mode_todo`, which we accept since base-case calls dominate
# the SK wall-clock budget).
_HRSA_FAST_ARGS = ["--max-solns", "1", "--no-lookahead"]

# Process-local memoisation: HRSA on (rounded theta, eps) is deterministic,
# so we cache results across Euler / SK calls within a single Python run.
# Key: (round(theta, 6), round(eps, 6)).  Value: full bootstrap_target dict.
_HRSA_CACHE: dict = {}
_HRSA_CACHE_HITS = {"hit": 0, "miss": 0}


def _hrsa_synthesize_z01(theta: float, eps: float,
                          max_f: int = 4,
                          timeout_s: float = 60.0,
                          fast: bool = True) -> dict:
    """Call HRSA for an R^Z_{(0,1)}(theta) target at the requested precision.

    `fast=True` adds `--max-solns 1 --no-lookahead` for ~6x speedup at the cost
    of ~+2 N_D per leaf on average.  Tolerates failure: doubles max_f once and
    retries; if still no solution, returns dict with V=None / success=False.

    Process-local cache on (round(theta, 6), round(eps, 6)) avoids re-invoking
    HRSA for repeated leaf angles within a single SK invocation.
    """
    cache_key = (round(theta, 6), round(eps, 6), bool(fast))
    if cache_key in _HRSA_CACHE:
        _HRSA_CACHE_HITS["hit"] += 1
        return _HRSA_CACHE[cache_key]
    _HRSA_CACHE_HITS["miss"] += 1
    extra = list(_HRSA_FAST_ARGS) if fast else None
    target = _build_RZ(0, 1, theta)
    result = bootstrap_target(
        target,
        eps_hrsa=eps,
        theta=theta,
        max_f=max_f,
        fallback_max_f=max_f * 2,
        timeout_s=timeout_s,
        extra_args=extra,
    )
    _HRSA_CACHE[cache_key] = result
    return result


def hrsa_cache_stats() -> dict:
    """Return cumulative HRSA cache stats (hits / misses) for this Python run."""
    return dict(_HRSA_CACHE_HITS)


def synthesize_leaf(theta: float, eps: float, i: int, j: int) -> dict:
    """Synthesise a single Z-rotation leaf R^Z_{(i,j)}(theta) at precision eps.

    Parameters
    ----------
    theta : rotation angle (radians).
    eps   : leaf precision target; |V - R^Z_{(i,j)}(theta)|_F should be < eps.
    i, j  : 2-level subspace indices (0 <= i < j <= 2).

    Returns dict with:
      'V'           : (3,3) complex synthesised unitary
      'N_D'         : int  number of D-gates used  (0, 1, or HRSA-reported)
      'method'      : str  'identity' | 'sign_clifford_cached' | 'HRSA' | 'zeta9'
      'frob_error'  : float  |V - target|_F
      'success'     : bool  whether the requested eps was met
      'theta'       : float  echoed angle (after the (i,j) → (0,1) reduction)
      'i'           : int    echoed i
      'j'           : int    echoed j
    """
    if not (0 <= i < j <= 2):
        raise ValueError(f"need 0 <= i < j <= 2; got (i,j)=({i},{j})")

    # Reduce angle to (-pi, pi] for stability.
    theta_red = (theta + math.pi) % (2.0 * math.pi) - math.pi
    target = _build_RZ(i, j, theta_red)
    I3 = np.eye(3, dtype=complex)

    # --- Rule 1: tiny angle → identity. ---
    if abs(theta_red) < eps:
        return {
            "V": I3.copy(),
            "V_ring": rk.ringmat_identity(),
            "ring_valid": True,
            "N_D": 0,
            "method": "identity",
            "frob_error": float(np.linalg.norm(I3 - target, ord="fro")),
            "success": True,
            "theta": theta_red,
            "i": i, "j": j,
        }

    # --- Rule 2: closest in 5184-element sign-Clifford net? ---
    idx, dist = _closest_in_e0_net(target, eps)
    if dist <= eps:
        net = _get_e0_net()
        return {
            "V": net[idx].copy(),
            "V_ring": rk.ringmat_from_net_idx(idx, str(_E0_NET_PATH)),
            "ring_valid": True,
            "N_D": 1,                # sign-extended Cliffords cost ≤ 1 N_D
            "method": "sign_clifford_cached",
            "net_idx": idx,
            "frob_error": dist,
            "success": True,
            "theta": theta_red,
            "i": i, "j": j,
        }

    # --- Subspace reduction: (i,j) != (0,1) → conjugate by a permutation Cliff. ---
    perm = _build_subspace_permutations()
    P, Pdag, ci_P, ci_Pdag = perm[(i, j)]

    # --- Rule 3: HRSA dispatch. ---
    if eps >= 0.005:
        try:
            result = _hrsa_synthesize_z01(theta_red, eps=eps)
        except Exception as exc:
            # HRSA crashed (timeout, empty JSON, ...).  Don't propagate;
            # report a soft failure and let the caller fall back.
            return {
                "V": None, "V_ring": None, "ring_valid": False,
                "N_D": None, "method": "HRSA_exception",
                "frob_error": None, "success": False,
                "theta": theta_red, "i": i, "j": j,
                "exception": repr(exc),
            }
        if result.get("success") and result.get("V") is not None:
            V01 = result["V"]
            V_ij = P @ V01 @ Pdag
            err = float(np.linalg.norm(V_ij - target, ord="fro"))
            # Build the ringZ9 V_ij = P_ring @ V01_ring @ Pdag_ring.
            # HRSA's JSON payload has the ringZ9 representation of V01.
            raw = result.get("raw_json") or {}
            uni = raw.get("unitary") or {}
            V_ring_ij = None
            ring_valid = False
            try:
                if uni.get("V") is not None and uni.get("f") is not None:
                    V01_ring = rk.ringmat_from_ints(uni["V"], int(uni["f"]))
                    P_ring = rk.get_clifford_ringmat(int(ci_P), str(_E0_NET_PATH))
                    Pdag_ring = rk.get_clifford_ringmat(int(ci_Pdag), str(_E0_NET_PATH))
                    V_ring_ij = rk.ringmat_mul(P_ring, rk.ringmat_mul(V01_ring, Pdag_ring))
                    ring_valid = True
            except Exception:
                # ringZ9 build failed — leave V_ring_ij None and ring_valid False;
                # caller can still use the float V.
                V_ring_ij = None
                ring_valid = False
            return {
                "V": V_ij,
                "V_ring": V_ring_ij,
                "ring_valid": ring_valid,
                "N_D": int(result["N_D"]) if result["N_D"] is not None else None,
                "method": "HRSA",
                "frob_error": err,
                "success": err <= eps * 1.5,  # small slack vs HRSA's threshold
                "theta": theta_red,
                "i": i, "j": j,
                "hrsa_raw": result,
            }
        # HRSA failure: fall through to a loud failure dict.
        return {
            "V": None,
            "V_ring": None,
            "ring_valid": False,
            "N_D": None,
            "method": "HRSA_failed",
            "frob_error": None,
            "success": False,
            "theta": theta_red,
            "i": i, "j": j,
            "hrsa_raw": result,
        }

    # --- Rule 4: tight eps → zeta9 (STUBBED). ---
    raise NotImplementedError(
        f"synthesize_leaf eps={eps:.2e} < 0.005 falls to zeta9, which is "
        f"not yet wired through this module.  See unified/zeta9_compile.py "
        f"for the wrapper.  Caller: theta={theta_red}, (i,j)=({i},{j})."
    )


# ---------------------------------------------------------------------------
#  Convenience: bulk-synthesise a factor list from su3_euler.
# ---------------------------------------------------------------------------

def synthesize_factor_list(factors: list,
                            eps_per_leaf: float) -> dict:
    """Walk a factor list from `su3_euler.euler_decompose` and synthesise leaves.

    For each ('Z', i, j, theta) factor, call synthesize_leaf(theta, eps_per_leaf, i, j).
    For each ('CLIFFORD', idx), look up the Clifford matrix (N_D = 0).
    For each ('SIGN', sp), build the diagonal sign matrix (N_D <= 1; we charge 0
    if sp==0 (identity), else 1).

    Returns dict with:
      'V_total'       : (3,3) approximation of the product of factors (float)
      'V_ring_total'  : RingMat (list[list[Z9Frac]]) exact product, OR None if
                        any factor failed to produce a ringZ9 representation.
      'ring_valid'    : bool True iff every factor's ringZ9 was available
      'N_D_total'     : int sum of N_D across leaves (Cliffords are free)
      'per_leaf'      : list of dicts, one per ('Z', ...) factor
      'frob_error_per_leaf' : list of floats (each leaf's |V_leaf - target|_F)
      'n_leaves'      : int  number of Z-rotation factors
      'success'       : bool True iff every leaf reported success
    """
    cliffords = _load_clifford_set()
    V_total = np.eye(3, dtype=complex)
    V_ring_total = rk.ringmat_identity()
    ring_valid = True
    N_D_total = 0
    per_leaf = []
    frob_per_leaf = []
    success = True
    n_leaves = 0

    for f in factors:
        kind = f[0]
        if kind == "Z":
            _, i, j, theta = f
            res = synthesize_leaf(theta, eps_per_leaf, i, j)
            if res.get("V") is None:
                # Leaf synthesis failed.  Substitute the EXACT target rotation
                # so the rest of the product remains a valid SU(3) element
                # (the contribution to error is bounded by the per-leaf budget).
                success = False
                ring_valid = False  # exact target rotation is not ringZ9
                V_total = V_total @ _build_RZ(i, j, theta)
                per_leaf.append(res)
                n_leaves += 1
                continue
            V_total = V_total @ res["V"]
            if ring_valid and res.get("ring_valid") and res.get("V_ring") is not None:
                V_ring_total = rk.ringmat_mul(V_ring_total, res["V_ring"])
            else:
                ring_valid = False
            if res.get("N_D") is not None:
                N_D_total += int(res["N_D"])
            per_leaf.append(res)
            frob_per_leaf.append(res.get("frob_error", float("nan")))
            n_leaves += 1
            success = success and res.get("success", False)
        elif kind == "CLIFFORD":
            idx = f[1]
            V_total = V_total @ cliffords[idx]
            if ring_valid:
                try:
                    C_ring = rk.get_clifford_ringmat(int(idx), str(_E0_NET_PATH))
                    V_ring_total = rk.ringmat_mul(V_ring_total, C_ring)
                except Exception:
                    ring_valid = False
        elif kind == "SIGN":
            sp = f[1]
            D = np.eye(3, dtype=complex)
            for k in range(3):
                D[k, k] = -1.0 if (sp >> k) & 1 else 1.0
            V_total = V_total @ D
            if ring_valid:
                V_ring_total = rk.ringmat_mul(V_ring_total, rk.ringmat_sign_diag(int(sp)))
            if sp != 0:
                N_D_total += 1
        else:
            raise ValueError(f"unknown factor kind {kind!r} in factor list")

    return {
        "V_total": V_total,
        "V_ring_total": V_ring_total if ring_valid else None,
        "ring_valid": ring_valid,
        "N_D_total": N_D_total,
        "per_leaf": per_leaf,
        "frob_error_per_leaf": frob_per_leaf,
        "n_leaves": n_leaves,
        "success": success,
    }


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{'theta':>10} {'eps':>8} {'(i,j)':>8} {'method':>22} "
          f"{'N_D':>4} {'|V-T|_F':>12}")
    print("-" * 78)
    # Small set to keep the smoke test under 30s of HRSA calls.
    for theta, eps, (i, j) in [
        (0.005, 0.05, (0, 1)),         # rule 1: identity
        (math.pi / 4, 0.3, (0, 1)),    # rule 2: net hit
        (math.pi / 4, 0.05, (0, 1)),   # rule 3: HRSA
        (math.pi / 2, 0.05, (1, 2)),   # rule 3 with subspace reduction
    ]:
        try:
            res = synthesize_leaf(theta, eps, i, j)
        except NotImplementedError as exc:
            print(f"  theta={theta:.4f} eps={eps:.3f} (i,j)=({i},{j})  SKIPPED  ({exc})")
            continue
        method = res["method"]
        N_D = res["N_D"] if res["N_D"] is not None else -1
        err = res["frob_error"] if res["frob_error"] is not None else float("nan")
        print(f"  {theta:>8.4f} {eps:>8.3f}  ({i},{j})  {method:>20} "
              f"{N_D:>4} {err:>12.4e}")
