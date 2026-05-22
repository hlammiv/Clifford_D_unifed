"""EP-direct compiler for qutrit Clifford+D.

Pipeline for one (θ, ε, f) cell:

  1. Per-component rounding: round each of (u_1, u_2, u_3) to Z[ζ_9, 1/3^f]
     near its Householder target.  Uses lattice_round.round_single_entry.
       target_1 = e^{+iθ/2}    (per HRSA convention; |·|² ≈ 1)
       target_2 = -1            (|·|² ≈ 1)
       target_3 = 0             (|·|² ≈ 0)
     The sum |u_1|² + |u_2|² + |u_3|² = 2 enforces the Householder norm.

  2. Triple combination + joint norm filter: form (u_1, u_2, u_3) by taking
     one candidate from each per-component list, then check the joint norm
     in Z[α] equals (2·3^{2f}, 0, 0).

  3. ε-test: verify σ_1(u) is within ε of the target Householder vector.

  4. Build V via Householder formula (HRSA reification).

  5. (Optional) Decompose V to count gates via HRSA's decompose_tool CLI.

This is "Path B" — the EP-style continuous-to-integer rounder that bypasses
zeta9's enumeration entirely.  We bet that for any cell with a valid
Householder approximation, per-component independent rounding produces enough
joint-norm hits to be efficient.
"""

from __future__ import annotations
import cmath
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running as script or module
_here = Path(__file__).parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from zeta9.lattice_round import (
    minkowski_basis,
    round_single_entry,
    y_norm,
)


# ---------------------------------------------------------------------------
#  Householder targets for R^Z_{(0,1)}(θ)
# ---------------------------------------------------------------------------

def householder_targets(theta: float) -> tuple[complex, complex, complex]:
    """Return (target_1, target_2, target_3) for the Householder vector
    approximating R^Z_{(0,1)}(θ) = diag(e^{-iθ/2}, e^{+iθ/2}, 1) per HRSA
    convention.  The Householder is V = X_{(0,1)} · (I - u u†) with
    u·sqrt(2) ≈ (e^{+iθ/2}, -1, 0) so that |u|² = 2."""
    return (
        cmath.exp(0.5j * theta),
        -1.0 + 0.0j,
        0.0 + 0.0j,
    )


# ---------------------------------------------------------------------------
#  Joint norm check
# ---------------------------------------------------------------------------

def total_norm(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray) -> tuple[int, int, int]:
    """Compute |u_1|² + |u_2|² + |u_3|² in Z[α].  Each input is 6 ints."""
    y1 = y_norm(a1)
    y2 = y_norm(a2)
    y3 = y_norm(a3)
    return (y1[0] + y2[0] + y3[0], y1[1] + y2[1] + y3[1], y1[2] + y2[2] + y3[2])


# ---------------------------------------------------------------------------
#  Sigma_1 distance for triple
# ---------------------------------------------------------------------------

def sigma1_distance(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray,
                     theta: float, f: int) -> float:
    """L2 distance ‖σ_1(u) − u_target‖ in C^3 for u = (a/3^f).

    Targets from householder_targets(θ).
    """
    M = minkowski_basis()
    sigma1 = M[:2, :]                                          # (2, 6)
    targets = np.array(householder_targets(theta))             # (3,) complex
    denom = 3**f
    d2 = 0.0
    for j, a in enumerate((a1, a2, a3)):
        ma = sigma1 @ np.asarray(a, dtype=np.float64)          # (2,)
        u_j = complex(ma[0] / denom, ma[1] / denom)
        d2 += abs(u_j - targets[j]) ** 2
    return math.sqrt(d2)


# ---------------------------------------------------------------------------
#  The EP-direct compiler
# ---------------------------------------------------------------------------

def ep_compile(theta: float, epsilon: float, f: int,
                *, per_component_candidates: int = 64,
                eps_per_component: Optional[float] = None,
                verbose: bool = False) -> Optional[dict]:
    """Compile R^Z(θ) into a qutrit Clifford+D Householder triple via the
    EP-direct rounding approach.

    Args:
      theta, epsilon, f: as usual.
      per_component_candidates: max candidates per (u_1, u_2, u_3) target.
      eps_per_component: ε budget per component (default epsilon/√3, so
                       triple's L2 σ_1 distance budget is epsilon).
      verbose: print phase counts.

    Returns:
      dict with success / triple / σ_1 residual / norm.  None if no hit.
    """
    if eps_per_component is None:
        eps_per_component = epsilon / math.sqrt(3.0)

    targets = householder_targets(theta)
    f_pow_sq = 3 ** (2 * f)
    target_norm = (2 * f_pow_sq, 0, 0)

    # Phase 1: per-component rounding
    cand_lists = []
    for j, t in enumerate(targets):
        cands = round_single_entry(t, eps=eps_per_component, f=f,
                                    max_candidates=per_component_candidates)
        cand_lists.append(cands)
        if verbose:
            print(f"  [u_{j+1}] target={t}, got {len(cands)} candidates within ε={eps_per_component:.4f}")

    if not all(cand_lists):
        return {"success": False, "reason": "one component had no candidates",
                "per_comp_counts": [len(c) for c in cand_lists]}

    # Phase 2: combine and joint-norm filter
    # Iterate over all triples (might be expensive for big candidate sets;
    # prune aggressively).
    hits = []
    for c1 in cand_lists[0]:
        a1 = c1[0]
        y1 = y_norm(a1)
        # Need y2 + y3 = target - y1
        needed_y23 = (target_norm[0] - y1[0],
                      target_norm[1] - y1[1],
                      target_norm[2] - y1[2])
        for c2 in cand_lists[1]:
            a2 = c2[0]
            y2 = y_norm(a2)
            # Now y3 must equal needed_y23 - y2
            needed_y3 = (needed_y23[0] - y2[0],
                         needed_y23[1] - y2[1],
                         needed_y23[2] - y2[2])
            for c3 in cand_lists[2]:
                a3 = c3[0]
                y3 = y_norm(a3)
                if y3 == needed_y3:
                    # Joint norm satisfied!
                    sig1_dist = sigma1_distance(a1, a2, a3, theta, f)
                    if sig1_dist <= epsilon:
                        hits.append({
                            "a1": a1.tolist(),
                            "a2": a2.tolist(),
                            "a3": a3.tolist(),
                            "y1": y1, "y2": y2, "y3": y3,
                            "sigma1_dist": sig1_dist,
                        })

    if verbose:
        print(f"  joint-norm passers (within ε): {len(hits)}")

    if not hits:
        return {"success": False, "reason": "no triple satisfies joint norm + ε",
                "per_comp_counts": [len(c) for c in cand_lists]}

    hits.sort(key=lambda h: h["sigma1_dist"])
    best = hits[0]
    return {
        "success": True,
        "best": best,
        "n_hits": len(hits),
        "per_comp_counts": [len(c) for c in cand_lists],
    }


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--theta", type=float, default=math.pi / 2)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--f", type=int, default=2)
    p.add_argument("--max-cands", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"=== EP-direct compile: θ={args.theta:.4f} ε={args.epsilon} f={args.f} ===")
    r = ep_compile(args.theta, args.epsilon, args.f,
                    per_component_candidates=args.max_cands,
                    verbose=True)
    if r is None:
        print("  (no result)")
    elif r["success"]:
        b = r["best"]
        print(f"  SUCCESS: {r['n_hits']} hits, best σ_1 dist = {b['sigma1_dist']:.6f}")
        print(f"    u_1 = {b['a1']}  Y_1 = {b['y1']}")
        print(f"    u_2 = {b['a2']}  Y_2 = {b['y2']}")
        print(f"    u_3 = {b['a3']}  Y_3 = {b['y3']}")
    else:
        print(f"  failed: {r['reason']}")
        print(f"  per-component counts: {r.get('per_comp_counts')}")
