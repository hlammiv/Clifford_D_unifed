"""Iterative tree descent for qutrit Clifford+D approximation.

Implements the EP-style approximate synthesis: given a continuous target unitary
U ∈ PU(3), find a Clifford+D word approximating U to within ε via recursive
descent toward the identity using group commutators of base-case approximations.

Architecture:
  - Base case: 5184 sign-extended Cliffords (depth-0 net) — provides ε_0 ≈ 0.3
    coverage of PU(3).  Loaded from /tmp/e0_net_5184.txt (built by e0_net_dump).
  - Iteration: given current error E = U · W^{-1} (W = current word), find the
    base-case element g closest to I·E^{-1} that reduces the error, append.
  - Termination: when |E - I|_F < ε, stop.

This is EP-style if we use the Bruhat-Tits structure (Iwasawa dispatch); for
the prototype we just do brute-force closest-net-element at each step.  The
recursion eventually plateaus at the base-case granularity; for tighter ε we
need either deeper bidir BFS (memory wall) or proper recursive commutators
(Kuperberg-style higher-cancellation).

Status: WIP scaffolding.  Today's deliverable is the ε_0-net loader + nearest
lookup + greedy single-step descent.  Recursion to tighter ε is the next
phase.
"""
from __future__ import annotations
import argparse
import cmath
import math
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
#  ε_0-net loader
# ---------------------------------------------------------------------------

_E0_NET_PATH = Path("/tmp/e0_net_closed.txt")


def load_e0_net(path: Path = _E0_NET_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load the 5184 sign-extended Clifford set from the dump file.

    Returns:
      U_arr: shape (5184, 3, 3) complex array of the unitaries
      meta:  shape (5184, 2) int array of (sign_pattern, clifford_idx)
    """
    U_arr, meta, _, _ = load_e0_net_with_ring(path)
    return U_arr, meta


def load_e0_net_with_ring(
    path: Path = _E0_NET_PATH,
    max_f: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the sign-extended Clifford set and also the per-entry ringZ9 ints.

    Each entry's value is (sum_k V_ring[i,j,k]·ξ^k) / 3^{f_per_idx}. The
    denominator exponent f depends on the element (some Cliffords need 1/3
    factors, others don't); the dump file does NOT store f explicitly, so we
    infer it by matching the ringZ9 eval against the dump's float column.

    Returns:
      U_arr:      shape (N, 3, 3) complex array of the unitaries (float)
      meta:       shape (N, 2) int array of (sign_pattern, clifford_idx)
      V_ring:     shape (N, 3, 3, 6) int array of Z[ξ] coefficients on
                  {1, ξ, ξ², ξ³, ξ⁴, ξ⁵}
      f_per_idx:  shape (N,) int array; f-exponent for each net element

    Format of the dump file:
        idx  sign_pattern  clifford_idx  Re00 Im00 Re01 Im01 ... Re22 Im22
        V_00_a0 V_00_a1 ... V_00_a5  V_01_a0 ... V_22_a5
    """
    import cmath
    xi = cmath.exp(2j * cmath.pi / 9.0)
    U_list = []
    meta_list = []
    V_ring_list = []
    f_list: list[int] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            sp = int(parts[1]); ci = int(parts[2])
            mat = np.zeros((3, 3), dtype=complex)
            for i in range(3):
                for j in range(3):
                    base = 3 + 2 * (3*i + j)
                    re = float(parts[base])
                    im = float(parts[base+1])
                    mat[i, j] = complex(re, im)
            V_int = np.zeros((3, 3, 6), dtype=np.int64)
            for i in range(3):
                for j in range(3):
                    off = 21 + 6 * (3 * i + j)
                    for k in range(6):
                        V_int[i, j, k] = int(parts[off + k])
            # Infer f: eval ringZ9 ints at f=0, then pick f = log_3(|eval[i,j]| /
            # |dump[i,j]|) using a nonzero entry. Round and verify.
            eval_f0 = np.zeros((3, 3), dtype=complex)
            for i in range(3):
                for j in range(3):
                    s = 0+0j
                    for k in range(6):
                        c = int(V_int[i, j, k])
                        if c == 0:
                            continue
                        s += c * (xi ** k)
                    eval_f0[i, j] = s
            f_use = 0
            best_err = float("inf")
            for f_try in range(max_f + 1):
                err = float(np.max(np.abs(eval_f0 / (3 ** f_try) - mat)))
                if err < best_err:
                    best_err = err
                    f_use = f_try
                if err < 1e-10:
                    break
            if best_err >= 1e-10:
                raise RuntimeError(
                    f"load_e0_net_with_ring: could not infer f for net element "
                    f"sp={sp}, ci={ci}; best (f={f_use}, residual={best_err:.2e}) "
                    f"at max_f={max_f}. Increase max_f or check dump integrity."
                )
            U_list.append(mat)
            meta_list.append((sp, ci))
            V_ring_list.append(V_int)
            f_list.append(f_use)
    U_arr = np.array(U_list, dtype=complex)
    meta = np.array(meta_list, dtype=np.int64)
    V_ring = np.array(V_ring_list, dtype=np.int64)
    f_per_idx = np.array(f_list, dtype=np.int64)
    return U_arr, meta, V_ring, f_per_idx


# ---------------------------------------------------------------------------
#  Distance metric
# ---------------------------------------------------------------------------

def frobenius_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Frobenius norm ‖U − V‖_F for two 3×3 complex matrices."""
    d = U - V
    return float(np.sqrt(np.abs(np.trace(d.conj().T @ d))))


def frobenius_distance_batch(U: np.ndarray, V_batch: np.ndarray) -> np.ndarray:
    """Frobenius distance from U to each row of V_batch (shape (N, 3, 3))."""
    d = V_batch - U[None, :, :]
    # ‖d‖_F² = sum of |d_ij|² over all entries.
    return np.sqrt(np.sum(np.abs(d) ** 2, axis=(1, 2)))


def closest_in_net(target: np.ndarray, net: np.ndarray) -> tuple[int, float]:
    """Return (index, distance) of net element closest to target."""
    d = frobenius_distance_batch(target, net)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


# ---------------------------------------------------------------------------
#  Target unitary
# ---------------------------------------------------------------------------

def target_R_Z_01(theta: float) -> np.ndarray:
    """Target qutrit z-rotation R^Z_{(0,1)}(θ) = diag(e^{-iθ/2}, e^{+iθ/2}, 1)."""
    return np.diag([
        cmath.exp(-0.5j * theta),
        cmath.exp(+0.5j * theta),
        1.0 + 0.0j,
    ]).astype(complex)


# ---------------------------------------------------------------------------
#  Single-step descent demo
# ---------------------------------------------------------------------------

def single_step_descent(target: np.ndarray, net: np.ndarray) -> dict:
    """Find the closest net element to the target.  This is the depth-0
    approximation — the base case for further recursion."""
    idx, dist = closest_in_net(target, net)
    return {
        "approx_idx": idx,
        "frob_distance": dist,
        "approx": net[idx],
    }


# ---------------------------------------------------------------------------
#  Iterative descent (greedy)
# ---------------------------------------------------------------------------

def greedy_descent(target: np.ndarray, net: np.ndarray,
                   *, max_iters: int = 10, eps: float = 1e-6,
                   verbose: bool = False) -> dict:
    """Greedy iterative approximation.

    At each step we have a residual U = (current_word)^{-1} · target.
    Find the net element g closest to U; multiply word from left by g.
    Continue until residual is identity (within ε).

    NOTE: this is the LEAST-clever descent; it doesn't use group commutators
    so the convergence is linear in net granularity rather than recursive.
    A real implementation would replace this with recursive higher-commutators
    (Kuperberg c=1.44 or EP c=1).

    Returns dict with word_indices, final_error, etc.
    """
    word = []
    residual = target.copy()
    history = []
    for it in range(max_iters):
        idx, d = closest_in_net(residual, net)
        history.append(d)
        if verbose:
            print(f"  iter {it}: residual frob = {d:.6f}, closest net idx = {idx}")
        if d < eps:
            break
        # Multiply residual on the left by g^{-1} = g^† (unitary)
        g = net[idx]
        residual = g.conj().T @ residual
        word.append(idx)
    return {
        "word_indices": word,
        "final_residual_frob": history[-1] if history else None,
        "history": history,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
#  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--theta", type=float, default=math.pi / 4,
                   help="rotation angle in radians")
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--net-path", default=str(_E0_NET_PATH))
    args = p.parse_args()

    print(f"Loading ε_0-net from {args.net_path}...")
    net, meta = load_e0_net(Path(args.net_path))
    print(f"  loaded {len(net)} unitaries, shape {net.shape}")

    target = target_R_Z_01(args.theta)
    print(f"\nTarget = R^Z_(0,1)({args.theta:.4f}):")
    print(target)

    print(f"\n=== Depth-0 (single-step) approximation ===")
    r0 = single_step_descent(target, net)
    print(f"  closest net element idx = {r0['approx_idx']}")
    print(f"  Frobenius distance = {r0['frob_distance']:.6f}")
    print(f"  sign_pattern, clifford_idx = {meta[r0['approx_idx']].tolist()}")

    print(f"\n=== Greedy iterative descent ({args.max_iters} iters max) ===")
    r1 = greedy_descent(target, net, max_iters=args.max_iters, verbose=True)
    print(f"  final residual Frob = {r1['final_residual_frob']:.6f}")
    print(f"  word length = {len(r1['word_indices'])}")
    print(f"  Greedy converges linearly; needs recursive commutators for tight ε.")
