"""End-to-end: HRSA bootstrap → SK recursion on residual.

target ≈ V · W_SK where V comes from HRSA at eps_hrsa=0.3 and W_SK
approximates the residual E = V†·target. Total error |product - target|
= |W_SK - E| because V is unitary.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).parent
sys.path.insert(0, str(_here))

from hrsa_bootstrap import bootstrap_target
from sk_driver import sk_synthesize, build_inverse_lookup
from ep_descent import load_e0_net


def main():
    print("Loading e0_net...")
    e0_net, e0_meta = load_e0_net()
    inv, inv_res = build_inverse_lookup(e0_net)
    n_exact = int((inv_res < 1e-6).sum())
    print(f"  net: {len(e0_net)} elements, {n_exact} with exact inverse in net")

    angles = [math.pi/6, math.pi/4, math.pi/3, math.pi/2, 2*math.pi/3, math.pi]
    print(f"\n{'theta':>8} | {'N_D':>3} | {'|E-I|':>10} | "
          + " | ".join(f"d{d}" for d in range(6)))
    print("-" * 100)
    for theta in angles:
        target = np.diag([
            np.exp(-0.5j * theta),
            np.exp(+0.5j * theta),
            1.0 + 0.0j,
        ]).astype(complex)
        b = bootstrap_target(target, eps_hrsa=0.3, theta=theta, max_f=4)
        if b.get('V') is None:
            print(f"  θ={theta:.4f}: HRSA failed: {b}")
            continue
        E = b['residual_E']
        errs = []
        for depth in range(6):
            r = sk_synthesize(E, depth, e0_net, inverse_lookup=inv)
            errs.append(r['final_error'])
        nd = b['N_D']
        e_norm = b['residual_norm']
        print(f"  {theta:>6.4f} | {nd:>3d} | {e_norm:>10.4e} | "
              + " | ".join(f"{x:.2e}" for x in errs))


if __name__ == "__main__":
    main()
