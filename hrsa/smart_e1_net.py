"""Smart densification of the ε_0-net near the identity for qutrit SK.

Background:
  /tmp/e0_net_5184.txt is the 5184-element sign-extended Clifford "ε_0-net".
  Naive Cartesian products (see e1_net_expand.py) capped at 20k entries only
  reduce the 95th-percentile covering radius from ~1.185 to ~1.052 (uniform
  SU(3) sampling).  Most products land far from the identity, where SK
  recursion does NOT need them.

This file builds a denser net specifically in the near-identity shell that
the Solovay–Kitaev recursion visits, using a priority-BFS / greedy-coverage
construction:

  1.  Frontier = base-net elements with ‖g − I‖_F < 1.0  (≈ 200-500 of them).
  2.  Accepted = Frontier  (initial set).
  3.  Repeatedly:  for each g_acc ∈ Accepted, batch-multiply by ALL base-net
      elements (5184).  Keep g_new only if
          ‖g_new − I‖_F  < max_radius                 (not too far from I)
        AND  min_{g ∈ Accepted} ‖g_new − g‖_F > min_separation   (non-redundant).
  4.  Stop when Accepted reaches max_size or no new elements added in a sweep.

Near-neighbour test uses scipy.spatial.cKDTree on the 18 real coordinates of
each 3×3 complex matrix (re/im interleaved).  Tree is rebuilt every ~10k
insertions to amortize.

The result is saved to /tmp/smart_e1_net.npz with arrays 'unitaries' and
'words' (object array of variable-length tuples of base-net indices).

Pure numpy + scipy.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
#  ε_0-net loader (shares format with ep_descent.py / e1_net_expand.py)
# ---------------------------------------------------------------------------

_E0_NET_PATH = Path("/tmp/e0_net_5184.txt")


def load_e0_net(path: Path = _E0_NET_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load the 5184 sign-extended Clifford set from the dump file.

    Returns:
      U_arr: shape (N, 3, 3) complex
      meta:  shape (N, 2) int — (sign_pattern, clifford_idx)
    """
    U_arr: List[np.ndarray] = []
    meta: List[Tuple[int, int]] = []
    with open(path) as fh:
        for line in fh:
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
    return np.asarray(U_arr, dtype=complex), np.asarray(meta, dtype=np.int64)


# ---------------------------------------------------------------------------
#  Distance utilities
# ---------------------------------------------------------------------------

_I3 = np.eye(3, dtype=complex)


def _frob_to_identity(U_batch: np.ndarray) -> np.ndarray:
    """‖U − I‖_F for each U in (N, 3, 3) array."""
    diff = U_batch - _I3[None, :, :]
    return np.sqrt(np.sum(diff.real ** 2 + diff.imag ** 2, axis=(1, 2)))


def _matrices_to_coords(U_batch: np.ndarray) -> np.ndarray:
    """Flatten (N, 3, 3) complex to (N, 18) real coords (re, im interleaved).

    Frobenius distance on matrices = Euclidean distance on these 18-vectors.
    """
    flat = U_batch.reshape(U_batch.shape[0], 9)
    return np.concatenate([flat.real, flat.imag], axis=1)


# ---------------------------------------------------------------------------
#  Priority-BFS / greedy coverage near identity
# ---------------------------------------------------------------------------

def priority_bfs_near_identity(
    net: np.ndarray,
    *,
    max_radius: float = 1.2,
    min_separation: float = 0.05,
    max_size: int = 500_000,
    max_depth: int = 3,
    seed_radius: float = 1.0,
    batch_size: int = 1000,
    tree_refresh: int = 10_000,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Build a depth-limited dense net near the identity by priority-BFS.

    The "frontier" at depth d is the set of accepted elements available as the
    LEFT factor in next-depth products.  Products g_acc · g_base are kept only
    when ‖g_acc · g_base − I‖_F < max_radius AND they are at least
    min_separation away (Frobenius) from every other accepted element.

    For a base net like the 5184 sign-extended Cliffords — extremely sparse
    near I — pure "near-I × near-I" expansion does nothing because there are
    only ~7 near-I base elements.  We therefore seed the frontier with the
    full base net so that cancellations g_i · g_j ≈ I have a chance to fire,
    and the radius filter on the OUTPUT keeps the accepted set near I.

    Args:
      net: (N, 3, 3) complex base net.
      max_radius: discard products with ‖g − I‖_F > max_radius (focus near I).
      min_separation: minimum Frobenius distance from any existing accepted
        element (greedy ε-net rule).
      max_size: stop when this many accepted elements have been collected.
      max_depth: how many BFS shells to expand.
      seed_radius: include all base elements with ‖g − I‖_F < seed_radius in
        the INITIAL accepted set (they're trivially near I).  The FRONTIER
        used as left factors is the full base net regardless — that's how we
        get cancellation products.
      batch_size: number of left-factor matrices per matmul batch.
      tree_refresh: rebuild the KD-tree every this-many insertions.
      verbose: print per-shell stats.

    Returns:
      U_arr: (M, 3, 3) complex accepted unitaries.
      words: list of length M; each entry is a tuple of base-net indices
        whose left-to-right product equals U_arr[i].
    """
    N = net.shape[0]
    d_net_id = _frob_to_identity(net)

    # ----- Initial accepted set: all base elements near identity -----
    seed_idx = np.flatnonzero(d_net_id < seed_radius)
    if seed_idx.size == 0:
        seed_idx = np.argsort(d_net_id)[:1]  # at least identity
    seed_idx = seed_idx[np.argsort(d_net_id[seed_idx])]
    if verbose:
        print(f"  [seed] {seed_idx.size} base elements with ‖g−I‖_F < {seed_radius}")

    accepted_U = [net[i] for i in seed_idx]
    accepted_words: List[Tuple[int, ...]] = [(int(i),) for i in seed_idx]

    # The frontier used as LEFT factor for shell-1 expansion is the full base
    # net (we want cancellation products g_i · g_j ≈ I).  After shell 1, the
    # frontier becomes the depth-1 accepted set — which is already near-I.
    shell_frontier_U: np.ndarray = net.copy()
    shell_frontier_words: List[Tuple[int, ...]] = [(int(j),) for j in range(N)]

    # KD-tree over current accepted coords.
    coords_list: List[np.ndarray] = [_matrices_to_coords(np.asarray(accepted_U))]
    tree = cKDTree(coords_list[0])
    tree_anchor = coords_list[0].shape[0]  # tree built up to this index

    def _rebuild_tree() -> None:
        nonlocal tree, tree_anchor
        stack = np.concatenate(coords_list, axis=0)
        coords_list[:] = [stack]
        tree = cKDTree(stack)
        tree_anchor = stack.shape[0]

    def _query_min_dist(new_coords: np.ndarray) -> np.ndarray:
        """Smallest distance from each row of new_coords to any accepted point.

        Uses the KD-tree for the "old" accepted; checks freshly-added (since
        last refresh) by direct numpy comparison.
        """
        d_tree, _ = tree.query(new_coords, k=1)
        if len(coords_list) > 1 or coords_list[0].shape[0] > tree_anchor:
            # Combine entries added since last tree rebuild.
            tail = np.concatenate(coords_list, axis=0)[tree_anchor:]
            if tail.shape[0]:
                # Pairwise distance: (Nnew, Ntail).
                diff = new_coords[:, None, :] - tail[None, :, :]
                d_tail = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
                return np.minimum(d_tree, d_tail)
        return d_tree

    insertions_since_tree = 0

    for depth in range(1, max_depth + 1):
        if len(accepted_U) >= max_size:
            break

        frontier_U = shell_frontier_U
        frontier_words = shell_frontier_words
        added_this_shell = 0
        accepted_before_shell = len(accepted_U)
        if verbose:
            print(f"  [shell {depth}] accepted={accepted_before_shell}  frontier={frontier_U.shape[0]}")

        for b_start in range(0, frontier_U.shape[0], batch_size):
            if len(accepted_U) >= max_size:
                break
            b_end = min(b_start + batch_size, frontier_U.shape[0])
            left = frontier_U[b_start:b_end]  # (B, 3, 3)
            # Batched product: (B, 3, 3) @ (N, 3, 3) -> (B, N, 3, 3).
            #   prods[bi, j] = left[bi] @ net[j]
            prods = np.einsum("bij,njk->bnik", left, net)
            BN = prods.shape[0] * prods.shape[1]
            prods_flat = prods.reshape(BN, 3, 3)

            # Pre-filter by radius to identity (cheap).
            d_to_I = _frob_to_identity(prods_flat)
            keep_mask = d_to_I < max_radius
            if not np.any(keep_mask):
                continue
            kept_local = np.flatnonzero(keep_mask)

            cand_coords = _matrices_to_coords(prods_flat[kept_local])
            min_d = _query_min_dist(cand_coords)
            ok_mask = min_d > min_separation
            if not np.any(ok_mask):
                continue
            ok_local = kept_local[ok_mask]

            # Greedy: process in order of increasing distance-to-I (closest first).
            order = np.argsort(d_to_I[ok_local])
            ok_local = ok_local[order]

            new_coords_buffer: List[np.ndarray] = []
            for li in ok_local:
                if len(accepted_U) >= max_size:
                    break
                # Re-check against ALL accepted (including ones added this same
                # batch) — use a small running-list comparison for the tail.
                row = _matrices_to_coords(prods_flat[li:li + 1])
                d_old = _query_min_dist(row)[0]
                if new_coords_buffer:
                    tail = np.concatenate(new_coords_buffer, axis=0)
                    diff = row - tail
                    d_new = float(np.sqrt(np.sum(diff * diff, axis=1)).min())
                    d_old = min(d_old, d_new)
                if d_old <= min_separation:
                    continue

                bi = li // N
                ji = li % N
                left_word = frontier_words[b_start + bi]
                new_word = left_word + (int(ji),)
                accepted_U.append(prods_flat[li].copy())
                accepted_words.append(new_word)
                new_coords_buffer.append(row)
                insertions_since_tree += 1
                added_this_shell += 1

            if new_coords_buffer:
                coords_list.append(np.concatenate(new_coords_buffer, axis=0))
            if insertions_since_tree >= tree_refresh:
                _rebuild_tree()
                insertions_since_tree = 0

        if verbose:
            print(f"  [shell {depth}] added {added_this_shell}, total accepted = {len(accepted_U)}")
        if added_this_shell == 0:
            break

        # For the NEXT shell, the frontier becomes the elements just added.
        # These are guaranteed near-I, so further multiplications can produce
        # finer near-I products via small commutators.
        new_slice = slice(accepted_before_shell, len(accepted_U))
        shell_frontier_U = np.asarray(accepted_U[new_slice], dtype=complex)
        shell_frontier_words = accepted_words[new_slice]

    U_arr = np.asarray(accepted_U, dtype=complex)
    return U_arr, accepted_words


# ---------------------------------------------------------------------------
#  Near-identity covering radius
# ---------------------------------------------------------------------------

# Eight Gell-Mann-like Hermitian generators of su(3); used as random directions
# for "near-identity" target sampling via U = exp(i * c * H).  We don't bother
# orthonormalizing — only the direction matters, and we'll randomize coefficients.
_LAMBDA = np.array([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1, 0, 0], [0, 1, 0], [0, 0, -2]] / np.sqrt(3.0),
], dtype=complex)


def _sample_near_identity(rng: np.random.Generator,
                          target_norm: float,
                          max_tries: int = 32) -> np.ndarray:
    """Generate U ∈ SU(3) with ‖U − I‖_F ≈ target_norm.

    Strategy: pick a random Hermitian H = Σ c_i λ_i, c_i ~ N(0,1), normalize,
    then set U = exp(i * t * H) and binary-search t until ‖U − I‖_F matches
    the target within 5 %.  For small radii, ‖exp(itH) − I‖_F ≈ t * ‖H‖_F.
    """
    cs = rng.standard_normal(8)
    cs /= np.linalg.norm(cs) + 1e-30
    H = np.einsum("k,kij->ij", cs, _LAMBDA)
    # ‖H‖_F should be O(1) since Σ c_i² = 1 and the λ_i are roughly unit-norm.
    # Bisect on t.
    t_lo, t_hi = 0.0, np.pi
    for _ in range(max_tries):
        t_mid = 0.5 * (t_lo + t_hi)
        U = expm(1j * t_mid * H)
        d = float(np.sqrt(np.sum(np.abs(U - _I3) ** 2)))
        if d > target_norm:
            t_hi = t_mid
        else:
            t_lo = t_mid
        if abs(d - target_norm) < 0.05 * target_norm:
            return U
    return expm(1j * t_hi * H)


def covering_radius_near_identity(
    net: np.ndarray,
    radius_band: Tuple[float, float] = (0.0, 1.0),
    n_samples: int = 200,
    seed: int = 0,
) -> float:
    """95th-percentile min-Frobenius distance over near-identity SU(3) samples.

    Targets are constructed so that ‖U_target − I‖_F is uniformly distributed
    over `radius_band`.  For each target, find the closest element of `net`
    and record its Frobenius distance.

    Args:
      net: (M, 3, 3) complex unitary net.
      radius_band: (lo, hi) target ‖U − I‖_F range.
      n_samples: number of random targets.
      seed: rng seed.

    Returns:
      95th percentile of the per-target min-Frobenius distances.
    """
    rng = np.random.default_rng(seed)
    lo, hi = radius_band
    mins = np.empty(n_samples, dtype=float)
    for k in range(n_samples):
        r = rng.uniform(lo, hi)
        # Avoid r=0 exactly (degenerate).
        r = max(r, 1e-6)
        U = _sample_near_identity(rng, r)
        diff = net - U[None, :, :]
        d = np.sqrt(np.sum(diff.real ** 2 + diff.imag ** 2, axis=(1, 2)))
        mins[k] = float(d.min())
    return float(np.percentile(mins, 95))


# ---------------------------------------------------------------------------
#  Comparison helper
# ---------------------------------------------------------------------------

def compare_nets(base_net: np.ndarray,
                 expanded_net: np.ndarray,
                 radius_band: Tuple[float, float] = (0.0, 1.0),
                 n_samples: int = 200,
                 seed: int = 0) -> dict:
    """Return a dict comparing base vs expanded near-identity covering."""
    cr_base = covering_radius_near_identity(base_net, radius_band, n_samples, seed)
    cr_exp = covering_radius_near_identity(expanded_net, radius_band, n_samples, seed)
    out = {
        "radius_band": radius_band,
        "n_samples": n_samples,
        "base_size": int(base_net.shape[0]),
        "expanded_size": int(expanded_net.shape[0]),
        "base_p95_near_id": cr_base,
        "expanded_p95_near_id": cr_exp,
    }
    if cr_exp > 0:
        out["ratio_base_over_expanded"] = cr_base / cr_exp
    return out


# ---------------------------------------------------------------------------
#  Word saving helper
# ---------------------------------------------------------------------------

def _save_words(path: Path, U_arr: np.ndarray, words: Sequence[Sequence[int]]) -> None:
    obj = np.empty(len(words), dtype=object)
    for i, w in enumerate(words):
        obj[i] = np.asarray(w, dtype=np.int32)
    word_lens = np.asarray([len(w) for w in words], dtype=np.int32)
    np.savez(path, unitaries=U_arr, words=obj, word_lens=word_lens)


# ---------------------------------------------------------------------------
#  __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    src = _E0_NET_PATH
    out = Path("/tmp/smart_e1_net.npz")

    print(f"Loading base ε_0-net from {src} ...")
    net, _meta = load_e0_net(src)
    print(f"  base net: {net.shape[0]} unitaries, shape {net.shape}")

    # ---- Baseline coverage near identity ----
    print("\nEstimating BASE-net covering radius (|U - I|_F in (0, 1.0), 200 samples) ...")
    cr_base_near = covering_radius_near_identity(net, radius_band=(0.0, 1.0),
                                                 n_samples=200, seed=0)
    print(f"  base p95 near-identity covering radius (Frob): {cr_base_near:.6f}")

    # Also report the smaller "SK-recursion" shell.
    cr_base_sk = covering_radius_near_identity(net, radius_band=(0.1, 0.5),
                                               n_samples=200, seed=1)
    print(f"  base p95 covering radius in |U-I|∈(0.1, 0.5): {cr_base_sk:.6f}")

    # ---- Default smart expansion ----
    print("\nBuilding smart depth-limited net (max_size=500000, min_sep=0.05, max_radius=1.2) ...")
    t0 = time.time()
    U_smart, words_smart = priority_bfs_near_identity(
        net,
        max_radius=1.2,
        min_separation=0.05,
        max_size=500_000,
        max_depth=3,
        seed_radius=1.0,
        batch_size=1000,
        tree_refresh=10_000,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s, expanded size = {U_smart.shape[0]}")

    print("\nEstimating SMART-net covering radius (|U - I|_F in (0, 1.0), 200 samples) ...")
    cr_smart_near = covering_radius_near_identity(U_smart, radius_band=(0.0, 1.0),
                                                  n_samples=200, seed=0)
    print(f"  smart p95 near-identity covering radius (Frob): {cr_smart_near:.6f}")
    if cr_smart_near > 0:
        print(f"  improvement ratio (base/smart): {cr_base_near / cr_smart_near:.3f}")

    cr_smart_sk = covering_radius_near_identity(U_smart, radius_band=(0.1, 0.5),
                                                n_samples=200, seed=1)
    print(f"  smart p95 covering radius in |U-I|∈(0.1, 0.5): {cr_smart_sk:.6f}")
    if cr_smart_sk > 0:
        print(f"  SK-shell improvement ratio (base/smart): {cr_base_sk / cr_smart_sk:.3f}")

    print(f"\nSaving smart net to {out} ...")
    _save_words(out, U_smart, words_smart)
    print(f"  wrote {out}")

    # ---- Aggressive sweep ----
    print("\nBuilding AGGRESSIVE smart net (max_size=200000, min_sep=0.02) ...")
    t0 = time.time()
    U_agg, words_agg = priority_bfs_near_identity(
        net,
        max_radius=1.2,
        min_separation=0.02,
        max_size=200_000,
        max_depth=3,
        seed_radius=1.0,
        batch_size=1000,
        tree_refresh=10_000,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s, expanded size = {U_agg.shape[0]}")

    cr_agg_near = covering_radius_near_identity(U_agg, radius_band=(0.0, 1.0),
                                                n_samples=200, seed=0)
    cr_agg_sk = covering_radius_near_identity(U_agg, radius_band=(0.1, 0.5),
                                              n_samples=200, seed=1)
    print(f"  aggressive p95 near-identity covering radius: {cr_agg_near:.6f}")
    print(f"  aggressive p95 SK-shell (0.1,0.5) covering radius: {cr_agg_sk:.6f}")
    if cr_agg_near > 0:
        print(f"  aggressive improvement (base/agg) near-I: {cr_base_near / cr_agg_near:.3f}")
    if cr_agg_sk > 0:
        print(f"  aggressive improvement (base/agg) SK-shell: {cr_base_sk / cr_agg_sk:.3f}")

    # ---- Distribution of distance-to-identity in the smart net (diagnostic) ----
    d_smart = _frob_to_identity(U_smart)
    print("\nSmart-net distance-to-identity distribution:")
    for lo, hi in [(0.0, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 0.5),
                   (0.5, 0.9), (0.9, 1.0), (1.0, 1.2)]:
        n = int(((d_smart >= lo) & (d_smart < hi)).sum())
        print(f"  [{lo:.2f}, {hi:.2f}): {n}")

    # ---- One-line conclusion (honest) ----
    print("\n=== CONCLUSION ===")
    in_sk_shell = int(((d_smart >= 0.1) & (d_smart < 0.5)).sum())
    print(
        f"Smart depth-≤3 net (size {U_smart.shape[0]}): {in_sk_shell} elements in "
        f"|U-I|∈(0.1,0.5).  p95 covering radius in that shell: "
        f"{cr_base_sk:.4f} (base) -> {cr_smart_sk:.4f} (smart)  "
        f"x{(cr_base_sk / cr_smart_sk if cr_smart_sk > 0 else float('nan')):.2f}.  "
    )
    if in_sk_shell == 0:
        print(
            "  Single-/double-/triple-products of the 5184 sign-extended Cliffords "
            "land in a DISCRETE set of distances-to-I (~0, ~1, ~sqrt(4/3)), with no "
            "depth-≤3 elements in the (0.1, 0.5) SK shell.  Smart densification at "
            "this depth CANNOT enable deeper SK recursion; commutator constructions "
            "(a·b·a^{-1}·b^{-1}) or much deeper words are required to populate the "
            "near-I shell that SK actually visits."
        )
    else:
        print(
            "  Smart densification populates the SK shell; this directly reduces "
            "the residual left to the next recursion level."
        )
