"""Joint-norm-aware integer rounder for qutrit Householder triples.

Problem:
  Round the continuous Householder vector u = (e^{iθ/2}, -1, 0) ∈ C³ to
  integer Z[ζ_9]/3^f coefficients (a_1, a_2, a_3) such that
    σ_1(a_j)/3^f ≈ u_j   (close to the target Householder vector)
    Y(a_1) + Y(a_2) + Y(a_3) = (2·3^{2f}, 0, 0)   in Z[α]
  where Y(a) = (m_0, m_1, m_2) is the Z[α]-valued norm of a, α = 2cos(2π/9).

Why a "joint" rounder?
  ep_compile.py's per-component independent rounding finds candidates that
  match σ_1 individually but produce wildly disparate Y signatures with
  no chance of cancelling in (m_1, m_2).  At θ=π/2, ε=0.3, f∈{2,4} the
  intersection is empty (0 hits).  This module peels components one at
  a time and uses each Y to constrain the next.

Strategy:
  Step 1: enumerate a_3 candidates around σ_1 target 0.
  Step 2: for each a_3, need Y(a_1)+Y(a_2) = target_norm − Y(a_3).
  Step 3: enumerate a_2 candidates around σ_1 target −1.
  Step 4: needed_Y_1 = target_norm − Y(a_3) − Y(a_2);
          enumerate a_1 near σ_1 target e^{iθ/2} and filter by Y match.
  Step 5: keep triples with total σ_1 L2 distance < ε.

Reuses zeta9.lattice_round.{minkowski_basis, round_single_entry, y_norm}.
"""

from __future__ import annotations
import cmath
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running as script or module.
_here = Path(__file__).parent
if str(_here.parent) not in sys.path:
    sys.path.insert(0, str(_here.parent))

from zeta9.lattice_round import (
    minkowski_basis,
    round_single_entry,
    y_norm,
)


# ---------------------------------------------------------------------------
#  Targets
# ---------------------------------------------------------------------------

def householder_targets(theta: float) -> tuple[complex, complex, complex]:
    """(target_1, target_2, target_3) for Householder of R^Z_{(0,1)}(θ).
    Matches ep_compile's convention so the two backends are comparable."""
    return (
        cmath.exp(0.5j * theta),
        -1.0 + 0.0j,
        0.0 + 0.0j,
    )


# ---------------------------------------------------------------------------
#  σ_1 distance for a triple
# ---------------------------------------------------------------------------

def _sigma1_distance(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray,
                     theta: float, f: int) -> float:
    M = minkowski_basis()
    sigma1 = M[:2, :]
    targets = householder_targets(theta)
    denom = 3 ** f
    d2 = 0.0
    for j, a in enumerate((a1, a2, a3)):
        ma = sigma1 @ np.asarray(a, dtype=np.float64)
        u_j = complex(ma[0] / denom, ma[1] / denom)
        d2 += abs(u_j - targets[j]) ** 2
    return math.sqrt(d2)


# ---------------------------------------------------------------------------
#  Plausibility check for the residual Y target before we hunt a_1
# ---------------------------------------------------------------------------

def _is_plausible_y(y: tuple[int, int, int]) -> bool:
    """A necessary (but not sufficient) check that y could be Y(a) for some
    a ∈ Z[ζ_9].

    Y(a) = |a|² lies in the totally positive cone of Z[α] (where α=2cos(2π/9)).
    With numerical embeddings α ≈ 1.532, α' = 2cos(4π/9) ≈ 0.347,
    α'' = 2cos(8π/9) ≈ -1.879 (so α'' < 0), the three Galois conjugates of
    m_0 + m_1·α + m_2·α² must all be ≥ 0.  α² embeddings: α²≈2.347,
    (α')²≈0.121, (α'')²≈3.532.

    Returns True if all three real embeddings of (m_0, m_1, m_2) are ≥ 0.
    """
    m0, m1, m2 = y
    # Galois embeddings of α and α²
    alphas = (
        2.0 * math.cos(2.0 * math.pi / 9.0),
        2.0 * math.cos(4.0 * math.pi / 9.0),
        2.0 * math.cos(8.0 * math.pi / 9.0),
    )
    for a in alphas:
        v = m0 + m1 * a + m2 * (a * a)
        if v < -1e-9:
            return False
    return True


# ---------------------------------------------------------------------------
#  The joint rounder
# ---------------------------------------------------------------------------

def joint_round(theta: float, eps: float, f: int,
                *,
                max_candidates: int = 64,
                eps_relax_factor: float = 1.5,
                conj_bound_factor: float = 1.5,
                verbose: bool = False) -> list[dict]:
    """Peel a_3, then a_2, then constrain a_1 by Y-match.

    Args:
      theta, eps, f: synthesis cell.
      max_candidates: per-component candidate cap (passed to round_single_entry).
      eps_relax_factor: a_1 search uses eps*eps_relax_factor/sqrt(3) so that
                       widened-radius candidates can still close the joint
                       constraint; final triples are filtered by total
                       σ_1 L2 distance ≤ eps.
      conj_bound_factor: forwarded to round_single_entry.
      verbose: print per-step diagnostics.

    Returns:
      A list of dicts:
        {'a1', 'a2', 'a3'  : list[int] of length 6
         'sigma1_dist'     : float (triple L2 σ_1 distance)
         'y1', 'y2', 'y3'  : tuple[int,int,int]
         'y_sum'           : tuple[int,int,int]  (should equal target_norm)}
      Sorted by sigma1_dist ascending.
    """
    target_norm = (2 * 3 ** (2 * f), 0, 0)
    t1, t2, t3 = householder_targets(theta)

    # Per-component eps budget.  Step 1 & 2 split eps^2 evenly across three
    # components for the σ_1 L2 distance.  Step 4 widens a_1 to recover joint
    # solutions where independent rounding would have rejected.
    eps_per = eps / math.sqrt(3.0)
    eps_a1 = eps_relax_factor * eps_per

    if verbose:
        print(f"  target_norm = {target_norm}")
        print(f"  eps_per = {eps_per:.4f}, eps_a1 (relaxed) = {eps_a1:.4f}")

    # ----- Step 1: a_3 candidates near σ_1 = 0 -----
    cand_a3 = round_single_entry(
        t3, eps=eps_per, f=f,
        max_candidates=max_candidates,
        conj_bound_factor=conj_bound_factor,
    )
    if verbose:
        print(f"  step 1 (a_3 near 0): {len(cand_a3)} candidates")
    if not cand_a3:
        return []

    # ----- Step 3: a_2 candidates near σ_1 = -1 -----
    cand_a2 = round_single_entry(
        t2, eps=eps_per, f=f,
        max_candidates=max_candidates,
        conj_bound_factor=conj_bound_factor,
    )
    if verbose:
        print(f"  step 3 (a_2 near -1): {len(cand_a2)} candidates")
    if not cand_a2:
        return []

    # ----- Step 4 prelim: a_1 candidates (relaxed eps) near σ_1 = e^{iθ/2} -----
    cand_a1 = round_single_entry(
        t1, eps=eps_a1, f=f,
        max_candidates=max_candidates * 4,
        conj_bound_factor=conj_bound_factor,
    )
    if verbose:
        print(f"  step 4 prelim (a_1 near e^{{iθ/2}}): {len(cand_a1)} candidates (relaxed)")
    if not cand_a1:
        return []

    # Precompute Y for every a_1 candidate -> bucket by Y for O(1) lookup.
    a1_by_y: dict[tuple[int, int, int], list[tuple[np.ndarray, float]]] = {}
    for a1, res_s1, _max_s, _ns in cand_a1:
        y1 = y_norm(a1)
        a1_by_y.setdefault(y1, []).append((a1, res_s1))

    # Stats
    pairs_total = 0
    pairs_plausible = 0   # needed_y_1 passes plausibility
    pairs_with_a1 = 0     # needed_y_1 has at least one a_1 candidate
    hits: list[dict] = []

    # Precompute y values for a_3 and a_2 candidates
    a3_with_y = [(c[0], y_norm(c[0])) for c in cand_a3]
    a2_with_y = [(c[0], y_norm(c[0])) for c in cand_a2]

    for a3, y3 in a3_with_y:
        needed_y_12 = (target_norm[0] - y3[0],
                       target_norm[1] - y3[1],
                       target_norm[2] - y3[2])
        for a2, y2 in a2_with_y:
            needed_y_1 = (needed_y_12[0] - y2[0],
                          needed_y_12[1] - y2[1],
                          needed_y_12[2] - y2[2])
            pairs_total += 1
            if not _is_plausible_y(needed_y_1):
                continue
            pairs_plausible += 1
            matches = a1_by_y.get(needed_y_1)
            if not matches:
                continue
            pairs_with_a1 += 1
            for a1, _res_s1 in matches:
                # Final filter: total σ_1 distance ≤ eps
                d = _sigma1_distance(a1, a2, a3, theta, f)
                if d > eps + 1e-12:
                    continue
                y1 = needed_y_1  # by construction
                y_sum = (y1[0] + y2[0] + y3[0],
                         y1[1] + y2[1] + y3[1],
                         y1[2] + y2[2] + y3[2])
                hits.append({
                    "a1": [int(x) for x in a1],
                    "a2": [int(x) for x in a2],
                    "a3": [int(x) for x in a3],
                    "sigma1_dist": d,
                    "y1": y1, "y2": y2, "y3": y3,
                    "y_sum": y_sum,
                })

    if verbose:
        print(f"  step 4 stats: {pairs_total} (a_3,a_2) pairs, "
              f"{pairs_plausible} plausible needed_Y_1, "
              f"{pairs_with_a1} with at least one a_1 in our pool, "
              f"{len(hits)} final triples passing σ_1 ≤ ε")

    hits.sort(key=lambda h: h["sigma1_dist"])
    return hits


# ---------------------------------------------------------------------------
#  Diagnostic: how often is needed_Y_1 even plausible?
# ---------------------------------------------------------------------------

def diagnose(theta: float, eps: float, f: int,
             max_candidates: int = 64) -> dict:
    """Same step 1-3 enumeration as joint_round but returns step-4 viability
    stats without trying to find a_1.  Useful when joint_round returns 0."""
    target_norm = (2 * 3 ** (2 * f), 0, 0)
    t1, t2, t3 = householder_targets(theta)
    eps_per = eps / math.sqrt(3.0)
    cand_a3 = round_single_entry(t3, eps=eps_per, f=f,
                                  max_candidates=max_candidates)
    cand_a2 = round_single_entry(t2, eps=eps_per, f=f,
                                  max_candidates=max_candidates)
    pairs_total = pairs_plausible = 0
    needed_y_examples: list[tuple[int, int, int]] = []
    for c3 in cand_a3:
        y3 = y_norm(c3[0])
        needed12 = (target_norm[0] - y3[0],
                    target_norm[1] - y3[1],
                    target_norm[2] - y3[2])
        for c2 in cand_a2:
            y2 = y_norm(c2[0])
            needed1 = (needed12[0] - y2[0],
                       needed12[1] - y2[1],
                       needed12[2] - y2[2])
            pairs_total += 1
            if _is_plausible_y(needed1):
                pairs_plausible += 1
                if len(needed_y_examples) < 5:
                    needed_y_examples.append(needed1)
    return {
        "n_a3": len(cand_a3),
        "n_a2": len(cand_a2),
        "pairs_total": pairs_total,
        "pairs_plausible": pairs_plausible,
        "plausible_frac": (pairs_plausible / pairs_total) if pairs_total else 0.0,
        "examples_needed_Y_1": needed_y_examples,
    }


# ---------------------------------------------------------------------------
#  __main__
# ---------------------------------------------------------------------------

def _run_cell(theta: float, eps: float, f: int, label: str,
              max_candidates: int = 64) -> None:
    print(f"\n=== {label}: θ={theta:.4f} ε={eps} f={f} (max_cands={max_candidates}) ===")
    hits = joint_round(theta, eps, f, max_candidates=max_candidates, verbose=True)
    if hits:
        best = hits[0]
        print(f"  RESULT: n_hits = {len(hits)}, best σ_1 dist = {best['sigma1_dist']:.6f}")
        print(f"    a1 = {best['a1']}  Y_1 = {best['y1']}")
        print(f"    a2 = {best['a2']}  Y_2 = {best['y2']}")
        print(f"    a3 = {best['a3']}  Y_3 = {best['y3']}")
        print(f"    Y_sum = {best['y_sum']}   (target = {(2*3**(2*f), 0, 0)})")
    else:
        print(f"  RESULT: n_hits = 0")
        d = diagnose(theta, eps, f, max_candidates=max_candidates)
        print(f"  DIAG: a_3 cands = {d['n_a3']}, a_2 cands = {d['n_a2']}")
        print(f"        pairs total = {d['pairs_total']}, "
              f"plausible needed_Y_1 = {d['pairs_plausible']} "
              f"({100*d['plausible_frac']:.1f}%)")
        if d["examples_needed_Y_1"]:
            print(f"        examples of needed_Y_1 (m0, m1, m2):")
            for ex in d["examples_needed_Y_1"]:
                print(f"          {ex}")


if __name__ == "__main__":
    print("joint_round.py — joint-norm-aware integer rounding for qutrit Householder triples")

    # ep_compile reports 0 hits at all three of these cells.
    _run_cell(math.pi / 2, 0.3, 2, "Cell A (matches ep_compile failing case)")
    _run_cell(math.pi / 2, 0.3, 4, "Cell B (matches ep_compile failing case)")
    _run_cell(math.pi / 4, 0.3, 2, "Cell C")
