"""sk_driver_rz.py — R_z-specific SK driver for qutrit Clifford+D synthesis.

Targets: R_z(θ) at arbitrary θ, pushing ε beyond zeta9's native reach
(typically max-f=2 ε≈10⁻⁴ → SK ε≈10⁻⁵ via one commutator level).

Algorithm (single-level SK for R_z target at ε):
  1. Lookup R_z(θ) in R_z DB at eps=ε directly. If hit, return.
  2. Find tightest available base V_base in R_z DB (across all ε tiers).
  3. Compute residual E = R_z(θ) · V_base†.   ||E - I||_F = eps_base.
  4. If eps_base ≤ ε, return V_base (no SK needed).
  5. Commutator-factor E ≈ A · B · A† · B† via su3_commutator.factor_commutator.
     A, B are general U(3) near identity, ||A - I||_F ≈ √eps_base.
  6. Approximate A, B by Euler-decomposing each into ~57 R_z leaves and
     looking up each leaf in the R_z DB at per-leaf eps = √eps_base/√57.
  7. Assemble V_total = V_A · V_B · V_A† · V_B† · V_base. Return.

Crucially, this driver does NOT need a pre-built U-net. The commutator
factors A, B are synthesized ON DEMAND via Euler-decomp + R_z DB lookup.

For SK contraction at ε ≤ eps_base^{3/2} to fire, the R_z DB must be
dense enough at tight ε (≤ √eps_base / √57) across the canonical [0, π/3]
fundamental domain (after Clifford 6-fold + negation symmetry).
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_UNIFIED = _HERE.parent
if str(_UNIFIED) not in sys.path:
    sys.path.insert(0, str(_UNIFIED))

from su3_commutator import factor_commutator  # type: ignore
from su3_euler import euler_decompose, _load_clifford_set  # type: ignore
from u_net.u_net_builder import (  # type: ignore
    _reify_factors,
    _v_blob_to_complex,
    _theta_canonical,
    _rz_clifford_2k3,
)
from rz_db.rz_lookup import RzLookupDB


def R_z_target(theta: float) -> np.ndarray:
    """HRSA convention: R^Z_{(0,1)}(θ) = diag(e^{-iθ/2}, e^{+iθ/2}, 1)."""
    return np.diag([
        np.exp(-1j * theta / 2.0),
        np.exp(+1j * theta / 2.0),
        1.0 + 0j,
    ]).astype(np.complex128)


def _rz_db_lookup_with_fold(theta: float, eps: float, db: RzLookupDB):
    """R_z DB lookup with Clifford 6-fold + negation fold.

    Returns dict {V (3,3 complex), N_D, achieved_frob, theta_canonical,
    k_cliff, daggered} or None on miss.
    """
    theta_can, k_cliff, daggered = _theta_canonical(theta)
    res = db.lookup(theta_can, eps)
    if res is None:
        return None
    V = _v_blob_to_complex(res["V"], res["v_f"])
    if daggered:
        V = V.conj().T
    if k_cliff != 0:
        V = _rz_clifford_2k3(k_cliff) @ V
    return {
        "V": V,
        "N_D": res["N_D"],
        "achieved_frob": res["achieved_frob"],
        "theta_canonical": theta_can,
        "k_cliff": k_cliff,
        "daggered": daggered,
    }


def _rz_tightest_base(theta: float, db: RzLookupDB,
                      candidate_eps=(1e-4, 1e-3, 1e-2, 1e-1, 5e-1)):
    """Find the TIGHTEST achievable R_z DB approximation of R_z(theta).

    Tries each candidate eps in order; returns the FIRST hit (which is the
    tightest tier that has coverage at this theta). The achieved_frob
    determines actual precision.
    """
    for eps in candidate_eps:
        hit = _rz_db_lookup_with_fold(theta, eps, db)
        if hit is not None:
            return hit
    return None


def _approximate_via_euler(U_target: np.ndarray, eps_u: float,
                            db: RzLookupDB,
                            cliffords: np.ndarray) -> dict:
    """Approximate an arbitrary U(3) element via Euler-decomp + R_z DB.

    Uses the existing u_net_builder._reify_factors machinery. Per-leaf
    budget is ε_leaf = eps_u / √n_leaves (L2 estimate; per the
    `tier0_slack_design_dead_2026-05-23` finding).

    Returns {V, N_D_sum, n_leaves, n_db_misses, success}.
    """
    factors = euler_decompose(U_target)
    m = _reify_factors(
        factors=factors,
        eps_u=eps_u,
        db=db,
        cliffords=cliffords,
        allow_live_fallback=False,
        timeout=60,
    )
    V = m.get("V_total")
    return {
        "V": V,
        "N_D_sum": int(m.get("N_D_sum", 0)),
        "n_leaves": int(m.get("n_leaves", 0)),
        "n_db_misses": int(m.get("n_db_misses", 0)),
        "success": V is not None,
    }


def sk_rz_compile(theta: float, eps_target: float,
                  db: RzLookupDB,
                  cliffords=None,
                  verbose: bool = False) -> dict:
    """Single-level SK synthesis of R_z(theta) at target Frobenius accuracy.

    Returns dict with:
      V (3,3 complex)             — synthesized unitary
      N_D                         — total D-gate count
      achieved_frob               — ||V - R_z(theta)||_F
      success                     — achieved_frob ≤ eps_target
      path                        — 'direct' | 'sk_one_level' | 'no_base'
      eps_base                    — base lookup's actual precision (if used)
      log                         — list of step records
    """
    if cliffords is None:
        cliffords = _load_clifford_set()

    R_target = R_z_target(theta)
    log: list[dict] = []

    # === Step 1: direct DB lookup at target precision ===
    direct = _rz_db_lookup_with_fold(theta, eps_target, db)
    if direct is not None:
        V = direct["V"]
        achieved = float(np.linalg.norm(V - R_target, ord="fro"))
        if achieved <= eps_target:
            log.append({"step": "direct_hit", "achieved": achieved})
            return {
                "V": V,
                "N_D": int(direct["N_D"] or 0),
                "achieved_frob": achieved,
                "success": True,
                "path": "direct",
                "eps_base": achieved,
                "log": log,
            }
        log.append({"step": "direct_hit_but_loose", "achieved": achieved,
                    "eps_target": eps_target})

    # === Step 2: find tightest available base ===
    base = _rz_tightest_base(theta, db)
    if base is None:
        log.append({"step": "no_base"})
        return {"V": None, "N_D": -1, "achieved_frob": float("inf"),
                "success": False, "path": "no_base",
                "eps_base": None, "log": log}

    V_base = base["V"]
    eps_base = float(np.linalg.norm(V_base - R_target, ord="fro"))
    log.append({"step": "base", "tier_eps": base["achieved_frob"],
                "actual_eps_base": eps_base, "N_D_base": base["N_D"]})

    if eps_base <= eps_target:
        return {
            "V": V_base,
            "N_D": int(base["N_D"] or 0),
            "achieved_frob": eps_base,
            "success": True,
            "path": "direct",
            "eps_base": eps_base,
            "log": log,
        }

    # === Step 3: SK commutator level ===
    # Residual E = R_target · V_base†  (this is R_z(δ) for δ = θ - θ_base)
    E = R_target @ V_base.conj().T
    err_E = float(np.linalg.norm(E - np.eye(3), ord="fro"))
    log.append({"step": "residual", "err_E": err_E})

    A_target, B_target, comm_residual = factor_commutator(E, polish=True)
    log.append({"step": "commutator_factor", "residual": comm_residual,
                "norm_A_minus_I": float(np.linalg.norm(A_target - np.eye(3), ord="fro")),
                "norm_B_minus_I": float(np.linalg.norm(B_target - np.eye(3), ord="fro"))})

    # Per-Euler-leaf precision target: ε_A ≈ eps_target / eps_base would balance
    # the SK contraction. Practically, use ε_A = eps_target / (factor of safety).
    # The contraction math: ||V_total - R_target||_F ≤ ε_A · eps_base + C · eps_base^{3/2}
    # Solving for ε_A: ε_A = max(eps_target / 2, eps_base^{1.5}) / eps_base.
    eps_A = max(eps_target * 0.5, eps_base ** 1.5) / max(eps_base, 1e-15)

    if verbose:
        print(f"[sk_rz] eps_base={eps_base:.3e}, eps_target={eps_target:.3e}, "
              f"eps_A_required={eps_A:.3e}")

    V_A = _approximate_via_euler(A_target, eps_A, db, cliffords)
    V_B = _approximate_via_euler(B_target, eps_A, db, cliffords)

    log.append({"step": "factor_A", "n_leaves": V_A["n_leaves"],
                "n_misses": V_A["n_db_misses"], "N_D": V_A["N_D_sum"],
                "success": V_A["success"]})
    log.append({"step": "factor_B", "n_leaves": V_B["n_leaves"],
                "n_misses": V_B["n_db_misses"], "N_D": V_B["N_D_sum"],
                "success": V_B["success"]})

    if not (V_A["success"] and V_B["success"]):
        return {"V": V_base, "N_D": int(base["N_D"] or 0),
                "achieved_frob": eps_base, "success": False,
                "path": "sk_factor_failed", "eps_base": eps_base, "log": log}

    # === Step 4: assemble V_total = V_A · V_B · V_A† · V_B† · V_base ===
    A = V_A["V"]
    B = V_B["V"]
    V_total = A @ B @ A.conj().T @ B.conj().T @ V_base

    achieved = float(np.linalg.norm(V_total - R_target, ord="fro"))
    # N_D: each of A, B used once + their daggers (free) → 2·N_D_A + 2·N_D_B
    # Wait — A and A^† each cost N_D_A (dagger of Clifford+D is itself Clifford+D
    # with same gate count). So total Clifford+D word uses 2·N_D_A + 2·N_D_B
    # for the commutator + N_D_base for V_base.
    N_D_total = (int(base["N_D"] or 0)
                 + 2 * V_A["N_D_sum"]
                 + 2 * V_B["N_D_sum"])

    log.append({"step": "assemble", "achieved": achieved,
                "N_D_total": N_D_total})

    return {
        "V": V_total,
        "N_D": N_D_total,
        "achieved_frob": achieved,
        "success": achieved <= eps_target,
        "path": "sk_one_level",
        "eps_base": eps_base,
        "log": log,
    }


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--db", default=os.environ.get("RZ_DB_PATH", "/tmp/rz_test.sqlite"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    db = RzLookupDB(args.db)
    res = sk_rz_compile(args.theta, args.eps, db, verbose=args.verbose)
    print(f"theta={args.theta}  eps_target={args.eps}")
    print(f"  achieved_frob = {res['achieved_frob']:.6e}")
    print(f"  N_D           = {res['N_D']}")
    print(f"  path          = {res['path']}")
    print(f"  success       = {res['success']}")
    if args.verbose:
        for step in res["log"]:
            print(f"    {step}")
