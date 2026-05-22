#!/usr/bin/env python3
"""decompose_y1_driver.py — produce N_D distribution for one (ε, θ) cell from
ideal-cache Y1 + γ output.

Pipeline
========
1. Load Y1 .npy (M = γγ̄ coefficients, shape (N, 3)) and companion .gamma.npy
   (γ in O_K standard basis, shape (N, 6)).
2. For target θ, build V_target = R_Z_01(θ) = diag(e^{-iθ/2}, e^{iθ/2}, 1).
3. Pair-join over (γ_a, γ_b) ∈ Y1²:
   - Compute σ_K_1(γ_a · γ̄_b)/3^{2f} and check distance to -e^{-iθ/2}.
   - Surviving pairs are V-Frob ≤ ε to V_target.
4. For each surviving pair, build V via householder_v_from_u(u=(γ_a, γ_b, 0))
   and run hrsa_decompose → N_D.
5. Write per-pair N_D values to JSONL (one line per pair).

Usage
=====
    decompose_y1_driver.py \\
        --y1 path/to/Y1_eps=1e-5_all.npy \\
        --gamma path/to/Y1_eps=1e-5_all.gamma.npy \\
        --f 6 --eps 1e-5 --theta 1.5707963 \\
        --output cell.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# run_hrsa_decompose lives in zeta9_compile; v_validate.py was removed so we
# can't reuse householder_v_from_u — we reimplement it below using inline
# ringZ9 ops backed by ideal_cache_conv._mul_in_OK.
sys.path.insert(0, "/home/hlamm/Desktop/efficent_gates/unified")
from zeta9_compile import run_hrsa_decompose
from zeta9.ideal_cache_conv import _mul_in_OK


# ---------------------------------------------------------------------------
# ringZ9 helpers (length-6 integer lists in basis {1, ω, ω², ω³, ω⁴, ω⁵}
# where ω = ζ_9). All ops on NUMERATORS only — caller tracks the common
# denominator 3^k.
# ---------------------------------------------------------------------------
def rz_zero(): return [0, 0, 0, 0, 0, 0]
def rz_one(): return [1, 0, 0, 0, 0, 0]
def rz_neg(a): return [-x for x in a]
def rz_add(a, b): return [a[i] + b[i] for i in range(6)]
def rz_sub(a, b): return [a[i] - b[i] for i in range(6)]


def rz_conj(a):
    """ζ_9 → ζ_9⁻¹ = ζ_9⁸; reduce back to the standard basis using
    ζ_9⁶ = -1 - ζ_9³, ζ_9⁷ = -ζ_9 - ζ_9⁴, ζ_9⁸ = -ζ_9² - ζ_9⁵."""
    c0, c1, c2, c3, c4, c5 = a
    return [c0 - c3, -c2, -c1, -c3, -c2 + c5, -c1 + c4]


def rz_mul(a, b):
    """Wrap _mul_in_OK (Numba) → Python list."""
    r0, r1, r2, r3, r4, r5 = _mul_in_OK(
        int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]),
        int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5]),
    )
    return [int(r0), int(r1), int(r2), int(r3), int(r4), int(r5)]


def householder_v_from_u(u_coeffs, f):
    """V = X_{0,1} · (I - u u†) given u as 3 ringZ9 numerators at denom 3^f.
    Returns (V_int, v_f) where V_int[r][c] is the ringZ9 numerator at denom 3^v_f.
    """
    # uu[i][j] = u_i · conj(u_j), at denom 3^(2f)
    uu = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            uu[i][j] = rz_mul(u_coeffs[i], rz_conj(u_coeffs[j]))
    f2 = 2 * f
    one_3_2f = [3 ** f2, 0, 0, 0, 0, 0]
    H = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i == j:
                H[i][j] = rz_sub(one_3_2f, uu[i][j])
            else:
                H[i][j] = rz_neg(uu[i][j])
    # V = X_{(0,1)} · H: swap rows 0 and 1
    V = [[None] * 3 for _ in range(3)]
    for j in range(3):
        V[0][j] = H[1][j]
        V[1][j] = H[0][j]
        V[2][j] = H[2][j]
    return V, f2

# ζ_9 = exp(2πi/9). σ_K_1 embedding sends ζ_9 → ζ_9 (the principal complex
# embedding). For γ = sum_k γ_k ζ_9^k, σ_K_1(γ) = sum_k γ_k · exp(2πi·k/9).
_OMEGA = complex(math.cos(2 * math.pi / 9), math.sin(2 * math.pi / 9))
_OMEGA_POWERS = np.array([_OMEGA ** k for k in range(6)], dtype=np.complex128)


def sigma_K_1_embed_batch(gamma_int):
    """Return σ_K_1(γ) for each row in gamma_int (shape (N, 6))."""
    return gamma_int.astype(np.float64) @ _OMEGA_POWERS


def sigma_K_1_embed_conj_batch(gamma_int):
    """Return conj(σ_K_1(γ)) for each row."""
    return np.conj(sigma_K_1_embed_batch(gamma_int))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--y1", required=True, help="Y1 .npy (M rows)")
    p.add_argument("--gamma", required=True, help="γ companion .npy")
    p.add_argument("--f", type=int, required=True, help="U denominator exp (f)")
    p.add_argument("--eps", type=float, required=True, help="Frob tolerance")
    p.add_argument("--theta", type=float, required=True, help="R_Z target angle (rad)")
    p.add_argument("--output", required=True, help="Output JSONL with per-pair results")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap on surviving pairs to decompose (default: all)")
    p.add_argument("--decompose_timeout", type=float, default=120.0)
    args = p.parse_args()

    f = args.f
    eps = args.eps
    theta = args.theta
    scale = 3 ** (2 * f)
    inv_scale = 1.0 / scale

    M = np.load(args.y1)
    G = np.load(args.gamma)
    assert M.shape[0] == G.shape[0], (M.shape, G.shape)
    N = M.shape[0]
    print(f"loaded Y1: {N} rows", flush=True)

    # ---- Exact Householder structural constraint ----
    # For u = (γ_a, γ_b, 0)/3^f to satisfy |u|² = 2 EXACTLY (required for V
    # to be in <Clifford, D>), need γ_a·γ̄_a + γ_b·γ̄_b = 2·3^{2f}, i.e.
    # M_a + M_b = (2·3^{2f}, 0, 0) in the (1, α, α²) F-basis.
    target_sum = np.array([2 * scale, 0, 0], dtype=np.int64)
    M_dict = {(int(M[i, 0]), int(M[i, 1]), int(M[i, 2])): i for i in range(N)}
    triple_pairs = []  # (a_idx, b_idx) with exact M-sum
    for i in range(N):
        needed = (int(target_sum[0] - M[i, 0]),
                  int(target_sum[1] - M[i, 1]),
                  int(target_sum[2] - M[i, 2]))
        j = M_dict.get(needed)
        if j is not None:
            triple_pairs.append((i, j))
    print(f"exact M-sum join: {len(triple_pairs)} pairs satisfying "
          f"M_a + M_b = (2·3^{{2f}}, 0, 0)", flush=True)

    # ---- Frob filter ----
    # V[0,0] = -σ_K_1(γ_b · γ̄_a)/3^{2f}; target = d_1 = e^{-iθ/2}.
    d1 = complex(math.cos(-theta / 2), math.sin(-theta / 2))
    pair_tol = eps / math.sqrt(2.0)
    sig_a = sigma_K_1_embed_batch(G)  # σ_K_1 of each γ (numerical)

    surviving = []  # list of (a_idx, b_idx, frob)
    for (ai, bi) in triple_pairs:
        V00 = -sig_a[bi] * sig_a[ai].conjugate() * inv_scale
        diff = V00 - d1
        absdiff = abs(diff)
        if absdiff <= pair_tol:
            surviving.append((ai, bi, float(math.sqrt(2.0) * absdiff)))
    print(f"frob filter: {len(surviving)} of {len(triple_pairs)} pairs "
          f"within Frob ≤ {eps:.1e} of V_target(θ={theta:.4f})", flush=True)

    # Sort by Frob, optionally cap
    surviving.sort(key=lambda x: x[2])
    if args.max_pairs is not None:
        surviving = surviving[: args.max_pairs]
        print(f"capped to top {len(surviving)} by Frob", flush=True)

    # Decompose each surviving V
    results = []
    t0 = time.perf_counter()
    for k, (ai, bi, fr) in enumerate(surviving):
        u_coeffs = np.array([G[ai], G[bi], np.zeros(6, dtype=np.int64)],
                            dtype=np.int64)
        try:
            V_int, v_f = householder_v_from_u(u_coeffs.tolist(), f)
        except Exception as e:
            results.append({"a_idx": ai, "b_idx": bi, "frob": fr,
                            "error": f"build V failed: {e}"})
            continue
        try:
            dec = run_hrsa_decompose(V_int, v_f, timeout=args.decompose_timeout)
            results.append({"a_idx": ai, "b_idx": bi, "frob": fr,
                            "N_D": dec.get("N_D"),
                            "sde_chi_initial": dec.get("sde_chi_initial"),
                            "sde_chi_final": dec.get("sde_chi_final"),
                            "success": dec.get("success", False)})
        except Exception as e:
            results.append({"a_idx": ai, "b_idx": bi, "frob": fr,
                            "error": f"decompose failed: {e}"})
        if (k + 1) % 25 == 0:
            print(f"  decomposed {k+1}/{len(surviving)}  "
                  f"(elapsed {time.perf_counter()-t0:.1f}s)", flush=True)
    t_dec = time.perf_counter() - t0

    # Write JSONL
    with open(args.output, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    # Summary
    successes = [r for r in results if r.get("success")]
    N_D_vals = [r["N_D"] for r in successes if r["N_D"] is not None]
    print(f"\n[done] {t_dec:.1f}s decompose wall", flush=True)
    print(f"  pairs decomposed: {len(results)}", flush=True)
    print(f"  successful: {len(successes)}", flush=True)
    if N_D_vals:
        print(f"  N_D: min={min(N_D_vals)}  median={int(np.median(N_D_vals))}  "
              f"max={max(N_D_vals)}  mean={np.mean(N_D_vals):.1f}", flush=True)
    print(f"  output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
