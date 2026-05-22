"""Recursive Solovay-Kitaev driver for qutrit Clifford+D synthesis.

Wires together the existing pieces:
  - ep_descent.load_e0_net  : 5184 sign-extended Cliffords (the ε_0-net)
  - ep_descent.closest_in_net / frobenius_distance
  - su3_commutator.group_commutator / factor_commutator
  - su3_lie (transitively, via su3_commutator)

Implements the classical Solovay-Kitaev recursion:
    SK_k(target):
        U_0 = SK_{k-1}(target)              # base-case at depth 0
        E   = U_0^dagger · target           # residual close to I
        A, B = factor_commutator(E)         # [A,B] ≈ E with |A-I|≈|B-I|≈c√|E-I|
        Va  = SK_{k-1}(A)
        Vb  = SK_{k-1}(B)
        return Va · Vb · Va^dagger · Vb^dagger · U_0

(We perform the depth-0 lookup at every depth >= 1 too, since the recursion
calls SK_{k-1} on the target itself for U_0.)

The word is stored as a left-to-right list of net indices.  When we need a
Hermitian conjugate of a sub-word we reverse the list and substitute each
index by its inverse via a precomputed inverse_lookup.

Caveats: SK contracts only when the initial residual lies in the
contraction basin of factor_commutator (empirically |E-I|_F ≲ 0.3 for
this code-path).  For coarser targets, depth=1 can be worse than depth=0
and that's a legitimate report-it observation.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ep_descent import (
    load_e0_net,
    closest_in_net,
    frobenius_distance,
    target_R_Z_01,
)
from su3_commutator import group_commutator, factor_commutator


# ---------------------------------------------------------------------------
#  Inverse lookup
# ---------------------------------------------------------------------------

_HASH_DECIMALS = 10


def _matrix_key(M: np.ndarray, decimals: int = _HASH_DECIMALS) -> bytes:
    """Quantize a 3x3 complex matrix to `decimals` digits and return the
    rounded byte representation suitable for use as a dict key."""
    R = np.round(M.real, decimals=decimals)
    I = np.round(M.imag, decimals=decimals)
    # Normalize -0.0 → 0.0 so equal matrices hash identically.
    R = R + 0.0
    I = I + 0.0
    return R.tobytes() + b"|" + I.tobytes()


def build_inverse_lookup(e0_net: np.ndarray, *,
                         verify_tol: float = 1e-6,
                         strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (inv, residual) where:
      inv[i]      = index j of the net element best approximating e0_net[i]^dagger
      residual[i] = Frobenius distance |e0_net[j] - e0_net[i]^dagger|_F

    If `strict=True`, raises when any residual exceeds `verify_tol` (the
    prompt's original spec — kept for callers that genuinely expect a
    group-closed net).

    EMPIRICAL NOTE: the 5184 'sign-extended Cliffords' built by
    e0_net_dump.cpp are NOT actually closed under inverse — at this writing,
    2916/5184 elements have no Frobenius match for g^dagger within the
    net (worst residual = 1.0).  We therefore default to non-strict and
    return the residual array so callers can decide what to do.
    """
    n = e0_net.shape[0]
    # Hash-based exact lookup first (fast path for the cases that DO close).
    table = {}
    for i in range(n):
        k = _matrix_key(e0_net[i])
        if k not in table:
            table[k] = i
    inv = np.full(n, -1, dtype=np.int64)
    residual = np.zeros(n, dtype=float)
    for i in range(n):
        dagger = e0_net[i].conj().T
        k = _matrix_key(dagger)
        j = table.get(k)
        if j is None:
            # Hash miss: brute-force nearest in Frobenius norm.
            d = np.sqrt(np.sum(np.abs(e0_net - dagger[None, :, :]) ** 2, axis=(1, 2)))
            j = int(np.argmin(d))
            r = float(d[j])
        else:
            r = float(np.linalg.norm(e0_net[j] - dagger, ord="fro"))
        inv[i] = j
        residual[i] = r
        if strict and r > verify_tol:
            raise ValueError(
                f"No inverse for net element {i}: "
                f"min Frobenius distance to net = {r:.3e} > tol {verify_tol}"
            )
    return inv, residual


def reverse_inverse(word: list[int], inverse_lookup: np.ndarray) -> list[int]:
    """Return the index-list representing W^dagger given index-list W.

    If W = [g_0, g_1, ..., g_{k-1}] denotes the product g_0 g_1 ... g_{k-1}
    in left-to-right order, then W^dagger = g_{k-1}^dagger ... g_0^dagger,
    so we reverse the list and replace each index by its inverse.
    """
    return [int(inverse_lookup[i]) for i in reversed(word)]


def product_from_word(word: list[int], e0_net: np.ndarray) -> np.ndarray:
    """Compute the actual matrix M = g_{i_0} · g_{i_1} · ... · g_{i_{k-1}}."""
    M = np.eye(3, dtype=complex)
    for idx in word:
        M = M @ e0_net[idx]
    return M


# ---------------------------------------------------------------------------
#  SK recursion
# ---------------------------------------------------------------------------

# If the residual E is so far from I that factor_commutator cannot
# meaningfully contract it, the recursion is harmful.  Refuse to recurse
# when |E - I|_F exceeds this and just return the depth-0 approximation.
# 0.7 is a generous bound; the empirical contraction basin is closer to 0.3.
_CONTRACT_CHECK_MAX = 0.7


def _sk_recurse(target: np.ndarray, depth: int,
                e0_net: np.ndarray, inverse_lookup: np.ndarray,
                state: dict) -> tuple[list[int], np.ndarray]:
    """Internal recursive helper.

    Returns (word_indices, product_matrix).  `state` is a mutable dict
    we use to accumulate tree_size.
    """
    state["tree_size"] += 1

    # Base case: depth-0, look up nearest net element.
    if depth == 0:
        idx, _ = closest_in_net(target, e0_net)
        return [idx], e0_net[idx]

    # Recursive case.
    # (a) Find U_0 = closest net element to target.
    idx0, d0 = closest_in_net(target, e0_net)
    U0 = e0_net[idx0]

    # (b) Residual E = U_0^dagger · target  (so target ≈ U_0 · E).
    E = U0.conj().T @ target
    err_E = float(np.linalg.norm(E - np.eye(3), ord="fro"))

    # Contract-check: if E is too far from I, don't recurse — the
    # commutator factorization won't meaningfully shrink anything.
    if err_E > _CONTRACT_CHECK_MAX or not np.isfinite(err_E):
        return [idx0], U0

    # (c) Factor E ≈ [A, B].  If the residual isn't small, refuse to
    # recurse and fall back to depth-0.
    try:
        A, B, residual_AB = factor_commutator(E)
    except Exception:
        return [idx0], U0
    # Sanity check on factor quality: if residual_AB > |E-I|, the
    # commutator factorization didn't help.
    if not np.isfinite(residual_AB) or residual_AB > err_E:
        return [idx0], U0

    # (d) Recurse on A and B.
    word_A, _Va = _sk_recurse(A, depth - 1, e0_net, inverse_lookup, state)
    word_B, _Vb = _sk_recurse(B, depth - 1, e0_net, inverse_lookup, state)

    # (e) Assemble:  Va · Vb · Va^dagger · Vb^dagger · U_0.
    word = (
        word_A
        + word_B
        + reverse_inverse(word_A, inverse_lookup)
        + reverse_inverse(word_B, inverse_lookup)
        + [idx0]
    )

    # (f) Multiply out for the actual product.
    product = product_from_word(word, e0_net)
    return word, product


def sk_synthesize(target: np.ndarray, depth: int,
                  e0_net: np.ndarray,
                  inverse_lookup: Optional[np.ndarray] = None,
                  e0_meta: Optional[np.ndarray] = None) -> dict:
    """Run the Solovay-Kitaev recursion at the requested depth.

    Returns a dict with keys:
      'word_indices' : list[int] left-to-right product order
      'product'      : (3,3) complex assembled matrix
      'final_error'  : float Frobenius |product - target|
      'depth'        : int
      'tree_size'    : int number of recursive calls
    """
    if inverse_lookup is None:
        inverse_lookup, _ = build_inverse_lookup(e0_net)
    state = {"tree_size": 0}
    word, product = _sk_recurse(target, depth, e0_net, inverse_lookup, state)
    err = float(np.linalg.norm(product - target, ord="fro"))
    return {
        "word_indices": word,
        "product": product,
        "final_error": err,
        "depth": depth,
        "tree_size": state["tree_size"],
    }


# ---------------------------------------------------------------------------
#  Depth sweep / evaluation
# ---------------------------------------------------------------------------

def evaluate_recursion(target: np.ndarray, max_depth: int,
                       e0_net: np.ndarray,
                       inverse_lookup: np.ndarray) -> list[dict]:
    """Run sk_synthesize for depths 0..max_depth and time each call."""
    out = []
    for depth in range(max_depth + 1):
        t0 = time.perf_counter()
        res = sk_synthesize(target, depth, e0_net, inverse_lookup)
        t1 = time.perf_counter()
        out.append({
            "depth": depth,
            "final_error": res["final_error"],
            "word_length": len(res["word_indices"]),
            "tree_size": res["tree_size"],
            "walltime_seconds": t1 - t0,
        })
    return out


# ---------------------------------------------------------------------------
#  __main__: characterize SK on Z-rotations
# ---------------------------------------------------------------------------

def _fmt_row(d: dict) -> str:
    return (
        f"  depth={d['depth']}  err={d['final_error']:.4e}  "
        f"|word|={d['word_length']:>5d}  tree={d['tree_size']:>6d}  "
        f"t={d['walltime_seconds']:.3f}s"
    )


if __name__ == "__main__":
    print("Loading ε_0-net from /tmp/e0_net_5184.txt...")
    e0_net, e0_meta = load_e0_net(Path("/tmp/e0_net_5184.txt"))
    print(f"  loaded {e0_net.shape[0]} unitaries.")

    print("Building inverse lookup...")
    t0 = time.perf_counter()
    inverse_lookup, inv_residual = build_inverse_lookup(e0_net, strict=False)
    dt = time.perf_counter() - t0
    n_unique_inv = len(np.unique(inverse_lookup))
    n_exact = int(np.sum(inv_residual < 1e-6))
    n_loose = int(np.sum(inv_residual >= 1e-6))
    print(f"  done in {dt:.2f}s; {n_unique_inv}/{e0_net.shape[0]} distinct inverse indices.")
    print(f"  exact inverses (residual<1e-6): {n_exact}/{e0_net.shape[0]}")
    print(f"  inexact (no group-closure):    {n_loose}/{e0_net.shape[0]}, "
          f"worst residual = {inv_residual.max():.4f}")
    # Sanity: every index should appear as an inverse of exactly one other index
    # (since g↦g^dagger is an involution on the net).
    if n_unique_inv != e0_net.shape[0]:
        print(f"  NOTE: inverse lookup is not a permutation — "
              f"{e0_net.shape[0] - n_unique_inv} duplicate targets "
              f"(expected since net is not group-closed).")
    # Quick verification on the elements that DID hash-match exactly.
    exact_idx = np.where(inv_residual < 1e-6)[0]
    rng = np.random.default_rng(0)
    if len(exact_idx) > 0:
        sample_idx = rng.choice(exact_idx, size=min(8, len(exact_idx)), replace=False)
        max_res = 0.0
        for i in sample_idx:
            j = int(inverse_lookup[i])
            prod = e0_net[i] @ e0_net[j]
            r = float(np.linalg.norm(prod - np.eye(3), ord="fro"))
            max_res = max(max_res, r)
        print(f"  spot-check on exact-inverse elements: max |g·g^{{-1}} - I|_F = {max_res:.2e}\n")
    else:
        print()

    thetas = [
        ("π/6",   math.pi / 6),
        ("π/4",   math.pi / 4),
        ("π/3",   math.pi / 3),
        ("π/2",   math.pi / 2),
        ("2π/3",  2 * math.pi / 3),
        ("π",     math.pi),
    ]
    max_depth = 4

    # Expected word lengths for the 4·W_{k-1}+1 recursion:
    expected_lengths = [1]
    for _ in range(max_depth):
        expected_lengths.append(4 * expected_lengths[-1] + 1)
    print(f"Expected |word| at depths 0..{max_depth}: {expected_lengths}\n")

    summary = []
    for label, th in thetas:
        target = target_R_Z_01(th)
        err_I = float(np.linalg.norm(target - np.eye(3), ord="fro"))
        print(f"=== θ = {label}  ({th:.4f} rad)   |target - I|_F = {err_I:.4f} ===")
        rows = evaluate_recursion(target, max_depth, e0_net, inverse_lookup)
        for r in rows:
            print(_fmt_row(r))
        # Monotonicity check.
        errs = [r["final_error"] for r in rows]
        monotone = all(errs[k + 1] <= errs[k] + 1e-12 for k in range(len(errs) - 1))
        first_help = None
        for k in range(1, len(errs)):
            if errs[k] < errs[0]:
                first_help = k
                break
        print(f"  monotone decreasing: {monotone};  "
              f"first depth that beats depth-0: "
              f"{first_help if first_help is not None else 'never (in 0..%d)' % max_depth}")
        summary.append({
            "theta_label": label,
            "theta": th,
            "errs": errs,
            "monotone": monotone,
            "first_help": first_help,
        })
        print()

    # One-line conclusions per target.
    print("=== Summary ===")
    for s in summary:
        e0e = s["errs"][0]
        e_best = min(s["errs"])
        best_depth = s["errs"].index(e_best)
        print(
            f"  θ={s['theta_label']:>5s}: depth-0 err={e0e:.4e}, "
            f"best={e_best:.4e} at depth={best_depth}; "
            f"monotone={s['monotone']}, "
            f"first_help={s['first_help']}"
        )

    n_helps = sum(1 for s in summary if s["first_help"] is not None)
    print(f"\nSK contracts on {n_helps}/{len(summary)} tested targets within depth ≤ {max_depth}.")
