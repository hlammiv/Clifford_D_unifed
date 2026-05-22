"""Depth-1 expansion of the ε_0-net for qutrit Solovay-Kitaev.

Given the 5184-element ε_0-net (8 signs × 648 Cliffords) with covering radius
ε_0 ≈ 0.55 (Frobenius), we densify by composing products g_i · g_j.

Naive 5184² = 27M is too much; we focus the left factor on a "near-identity"
subset of the net (small rotations, where refinement is needed) and let the
right factor range over the full net.  Output is deduplicated and capped at a
configurable size, keeping the most diverse representatives.

Pure numpy + scipy.  No project-specific imports.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.stats import unitary_group


# ---------------------------------------------------------------------------
#  ε_0-net loader (matches ep_descent.load_e0_net)
# ---------------------------------------------------------------------------

_E0_NET_PATH = Path("/tmp/e0_net_5184.txt")


def load_e0_net(path: Path = _E0_NET_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load the 5184 sign-extended Clifford set from the dump file.

    Each non-comment line: idx sign_pattern clifford_idx + 18 reals (3×3 complex
    matrix, row-major Re/Im pairs) + 54 ints (V coefficients).

    Returns:
      U_arr: shape (N, 3, 3) complex array of the unitaries
      meta:  shape (N, 2) int array of (sign_pattern, clifford_idx)
    """
    U_arr = []
    meta = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            sp = int(parts[1])
            ci = int(parts[2])
            mat = np.zeros((3, 3), dtype=complex)
            for i in range(3):
                for j in range(3):
                    base = 3 + 2 * (3 * i + j)
                    re = float(parts[base])
                    im = float(parts[base + 1])
                    mat[i, j] = complex(re, im)
            U_arr.append(mat)
            meta.append((sp, ci))
    U_arr = np.asarray(U_arr, dtype=complex)
    meta = np.asarray(meta, dtype=np.int64)
    return U_arr, meta


# ---------------------------------------------------------------------------
#  Near-identity selection
# ---------------------------------------------------------------------------

def find_near_identity_elements(net: np.ndarray,
                                threshold: float = 0.7) -> np.ndarray:
    """Return indices of net elements with ‖g − I‖_F < threshold.

    If fewer than 200 satisfy the cutoff, return the 200 smallest-distance
    indices instead (whichever set is larger).
    """
    I = np.eye(3, dtype=complex)
    d = np.sqrt(np.sum(np.abs(net - I[None, :, :]) ** 2, axis=(1, 2)))
    below = np.flatnonzero(d < threshold)
    if below.size >= 200:
        # Sort by distance ascending so callers can take prefixes.
        order = below[np.argsort(d[below])]
        return order.astype(np.int64)
    # Fallback: 200 smallest by distance.
    smallest = np.argsort(d)[:200]
    return smallest.astype(np.int64)


# ---------------------------------------------------------------------------
#  Expansion
# ---------------------------------------------------------------------------

def _hash_matrix(M: np.ndarray, decimals: int = 6) -> bytes:
    """Hash a 3×3 complex matrix by rounding to `decimals` places.

    Used for fast deduplication of products that are numerically identical.
    """
    re = np.round(M.real, decimals).astype(np.float64)
    im = np.round(M.imag, decimals).astype(np.float64)
    return re.tobytes() + im.tobytes()


def expand_net(net: np.ndarray,
               near_id_indices: np.ndarray,
               max_size: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Build products g_i · g_j with i ∈ near_id, j ∈ full net.

    Deduplication:
      - Exact: hash of rounded (6 decimal) real+imag entries.
      - Diversity: skip any product within 1e-4 Frobenius of an already-kept
        element in the expanded set.

    The full net itself is included as the leading entries (with word pair
    (j, -1) marking "no left factor") so the depth-1 net is a superset of the
    depth-0 net.

    Returns:
      U_expanded: shape (M, 3, 3) complex, M ≤ max_size
      word_pairs: shape (M, 2) int — (i, j) such that U_expanded[k] = net[i] @ net[j].
                  For seeded depth-0 entries, i = -1.
    """
    N = len(net)
    diversity_tol = 1e-4
    diversity_tol_sq = diversity_tol ** 2

    seen_hashes: set[bytes] = set()
    kept_U: list[np.ndarray] = []
    kept_pairs: list[tuple[int, int]] = []

    # Seed with the full base net so the expansion is monotone (depth-1 ⊇ depth-0).
    for j in range(N):
        if len(kept_U) >= max_size:
            break
        U = net[j]
        h = _hash_matrix(U)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        kept_U.append(U)
        kept_pairs.append((-1, j))

    # Diversity check requires a stacked array; rebuild as we go.  For
    # efficiency we maintain a growing (k, 3, 3) array and recompute the
    # vectorized distance.
    def _too_close(U: np.ndarray, stack: np.ndarray) -> bool:
        if stack.shape[0] == 0:
            return False
        diff = stack - U[None, :, :]
        d2 = np.sum(diff.real ** 2 + diff.imag ** 2, axis=(1, 2))
        return bool(np.any(d2 < diversity_tol_sq))

    stack = np.asarray(kept_U, dtype=complex) if kept_U else np.zeros((0, 3, 3), dtype=complex)

    # Iterate products in row-major order over (near_id × full_net).  We add to
    # `kept_U` and periodically refresh the diversity stack to avoid rebuilding
    # on every insertion.
    REFRESH_EVERY = 256
    pending_since_refresh = 0

    for i in near_id_indices:
        if len(kept_U) >= max_size:
            break
        gi = net[int(i)]
        # Batched product of gi against all net elements.
        # (3,3) @ (N,3,3) → (N,3,3)
        prods = np.einsum("ij,njk->nik", gi, net)
        for j in range(N):
            if len(kept_U) >= max_size:
                break
            U = prods[j]
            h = _hash_matrix(U)
            if h in seen_hashes:
                continue
            if _too_close(U, stack):
                seen_hashes.add(h)  # mark to skip future exact duplicates
                continue
            seen_hashes.add(h)
            kept_U.append(U)
            kept_pairs.append((int(i), j))
            pending_since_refresh += 1
            if pending_since_refresh >= REFRESH_EVERY:
                stack = np.asarray(kept_U, dtype=complex)
                pending_since_refresh = 0

    U_expanded = np.asarray(kept_U, dtype=complex)
    word_pairs = np.asarray(kept_pairs, dtype=np.int64)
    return U_expanded, word_pairs


# ---------------------------------------------------------------------------
#  Covering radius
# ---------------------------------------------------------------------------

def covering_radius(net: np.ndarray,
                    n_samples: int = 200,
                    seed: int = 0) -> float:
    """Estimate covering radius (95th percentile of min Frobenius distance).

    Samples random SU(3) elements via scipy.stats.unitary_group.rvs(3) followed
    by phase-projection to SU(3): U ← U · det(U)^{-1/3} (principal cube root).
    For each sample compute min_g ‖U − g‖_F over g ∈ net, then return the 95th
    percentile.

    NOTE: Cliffords actually live in PU(3), so projecting samples to SU(3) and
    measuring Frobenius distance is approximate — there will be some baseline
    discrepancy for phase choices not represented in the net.
    """
    rng = np.random.default_rng(seed)
    # scipy's unitary_group exposes a `random_state` argument.
    ug = unitary_group
    mins = np.empty(n_samples, dtype=float)
    for k in range(n_samples):
        U = ug.rvs(3, random_state=rng)
        det = np.linalg.det(U)
        # Principal cube root via complex log/exp.
        phase = np.exp(-np.log(det) / 3.0)
        U = U * phase
        diff = net - U[None, :, :]
        d = np.sqrt(np.sum(diff.real ** 2 + diff.imag ** 2, axis=(1, 2)))
        mins[k] = d.min()
    return float(np.percentile(mins, 95))


# ---------------------------------------------------------------------------
#  __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src = Path("/tmp/e0_net_5184.txt")
    out = Path("/tmp/e1_net_expanded.npz")
    print(f"Loading base ε_0-net from {src} ...")
    net, meta = load_e0_net(src)
    print(f"  base net: {net.shape[0]} unitaries, shape {net.shape}")

    print("Estimating base-net covering radius (200 SU(3) samples) ...")
    cr_base = covering_radius(net, n_samples=200, seed=0)
    print(f"  base covering radius (95th pct, Frobenius): {cr_base:.6f}")
    print("  (Cliffords live in PU(3); some discrepancy vs SU(3) samples expected.)")

    print("Finding near-identity elements (threshold=0.7) ...")
    near = find_near_identity_elements(net, threshold=0.7)
    print(f"  near-identity count: {near.size}")

    print("Expanding to depth-1 net (cap = 20000) ...")
    U_exp, pairs = expand_net(net, near, max_size=20000)
    print(f"  expanded size: {U_exp.shape[0]}")

    print("Estimating expanded-net covering radius (200 SU(3) samples) ...")
    cr_exp = covering_radius(U_exp, n_samples=200, seed=0)
    print(f"  expanded covering radius (95th pct, Frobenius): {cr_exp:.6f}")
    if cr_exp > 0:
        print(f"  ratio base/expanded: {cr_base / cr_exp:.3f}")

    print(f"Saving to {out} ...")
    np.savez(out, unitaries=U_exp, word_pairs=pairs)
    print(f"  wrote {out}")
