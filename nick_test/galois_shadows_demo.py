"""galois_shadows_demo.py — show the Galois "shadows" of Nick's exact C+D matrices.

Every exact Clifford+D unitary U has entries in Z[zeta_9, 1/3]. The field
Q(zeta_9) has three inequivalent complex embeddings, sigma_k: zeta_9 -> zeta_9^k
for k in {1, 2, 4} (k = 8, 7, 5 are their complex conjugates). sigma_1 is the
physical matrix; sigma_2(U) and sigma_4(U) are the "shadows".

For each sampled row this prints, per embedding: unitarity error, Frobenius
distance to the target R_z(theta), and |row 0| entries. Expected: all three
are exactly unitary; only sigma_1 is close to the target.

Usage:  python3 galois_shadows_demo.py [--f 10] [--rows 0 40 80]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "hrsa"))

from ingest_decompose import parse_fits_file  # noqa: E402


def embed(groups, f, k):
    """sigma_k(M): substitute zeta_9 -> zeta_9^k in every entry."""
    basis = np.exp(2j * np.pi * k * np.arange(6) / 9)
    return np.array([np.dot(g, basis) for g in groups]).reshape(3, 3) / 3**f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=int, default=10)
    ap.add_argument("--rows", type=int, nargs="+", default=[0, 40, 80])
    args = ap.parse_args()

    f, rows = parse_fits_file(_HERE / f"fits_f={args.f}.txt")
    for ri in args.rows:
        _, theta, groups = rows[ri]
        target = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2), 1])
        print(f"theta = {theta:.4f}  (f = {f})")
        for k in (1, 2, 4):
            U = embed(groups, f, k)
            unit_err = np.linalg.norm(U @ U.conj().T - np.eye(3))
            dist = np.linalg.norm(U - target)
            print(f"  sigma_{k}: unitarity err = {unit_err:.1e}   "
                  f"|U - Rz|_F = {dist:.2e}   |row 0| = {np.round(abs(U[0]), 3)}")


if __name__ == "__main__":
    main()
