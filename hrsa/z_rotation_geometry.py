"""Covering geometry of the 5184-element epsilon_0-net on the z-rotation
submanifolds of SU(3).

The 5184-element net (sign-extended Cliffords) is a depth-0 cover of full
SU(3) with 95th-percentile covering radius ~1.18 on generic SU(3).  Our
actual compilation targets, however, live on three 1-parameter submanifolds
of z-rotations:

    R^Z_{(0,1)}(theta) = diag(e^{-i theta/2}, e^{+i theta/2}, 1)
    R^Z_{(0,2)}(theta) = diag(e^{-i theta/2}, 1, e^{+i theta/2})
    R^Z_{(1,2)}(theta) = diag(1, e^{-i theta/2}, e^{+i theta/2})

This module measures the covering radius restricted to those circles, which
is the quantity that actually controls whether an SK / EP recursion can
converge from depth 0.

Pure numpy + scipy.  No project-specific imports beyond the net loader.
"""
from __future__ import annotations

import cmath
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group


# ---------------------------------------------------------------------------
#  Loader (mirrors ep_descent.load_e0_net)
# ---------------------------------------------------------------------------

_E0_NET_PATH = Path("/tmp/e0_net_5184.txt")


def load_e0_net(path: Path = _E0_NET_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load the 5184 sign-extended Clifford set from the dump file.

    Returns:
      U_arr: shape (N, 3, 3) complex array of the unitaries.
      meta:  shape (N, 2) int array of (sign_pattern, clifford_idx).
    """
    U_arr: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []
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
    return np.asarray(U_arr, dtype=complex), np.asarray(meta, dtype=np.int64)


# ---------------------------------------------------------------------------
#  Z-rotation target families
# ---------------------------------------------------------------------------

def z_rotation_R_Z(i: int, j: int, theta: float) -> np.ndarray:
    """Qutrit z-rotation R^Z_{(i,j)}(theta).

    diagonal is 1 everywhere except positions (i, i) = e^{-i theta/2}
    and (j, j) = e^{+i theta/2}.
    """
    if i == j or not (0 <= i < 3 and 0 <= j < 3):
        raise ValueError(f"invalid (i, j) = ({i}, {j}) for SU(3) z-rotation")
    d = np.ones(3, dtype=complex)
    d[i] = cmath.exp(-0.5j * theta)
    d[j] = cmath.exp(+0.5j * theta)
    return np.diag(d)


# ---------------------------------------------------------------------------
#  Frobenius distance helpers (vectorised)
# ---------------------------------------------------------------------------

def _frob_batch(U: np.ndarray, net: np.ndarray) -> np.ndarray:
    """Frobenius distance from U (3,3) to each row of net (N,3,3)."""
    d = net - U[None, :, :]
    return np.sqrt(np.sum((d.real * d.real + d.imag * d.imag), axis=(1, 2)))


# ---------------------------------------------------------------------------
#  Coverage curve along a 1-parameter z-rotation family
# ---------------------------------------------------------------------------

def coverage_curve(net: np.ndarray, i: int, j: int,
                   n_theta: int = 1024) -> dict:
    """Measure how well the net covers the R^Z_{(i,j)} circle.

    For each of n_theta uniform points theta in [0, 2 pi), find the Frobenius
    distance to the closest net element.

    Returns dict with: 'theta', 'min_frob', 'best_idx',
                       'pct_50', 'pct_75', 'pct_95', 'max', 'mean'.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    min_frob = np.empty(n_theta, dtype=np.float64)
    best_idx = np.empty(n_theta, dtype=np.int64)
    for k, th in enumerate(thetas):
        U = z_rotation_R_Z(i, j, float(th))
        d = _frob_batch(U, net)
        b = int(np.argmin(d))
        best_idx[k] = b
        min_frob[k] = float(d[b])
    return {
        "theta":    thetas,
        "min_frob": min_frob,
        "best_idx": best_idx,
        "pct_50":   float(np.percentile(min_frob, 50)),
        "pct_75":   float(np.percentile(min_frob, 75)),
        "pct_95":   float(np.percentile(min_frob, 95)),
        "max":      float(np.max(min_frob)),
        "mean":     float(np.mean(min_frob)),
    }


# ---------------------------------------------------------------------------
#  Worst / best theta identification
# ---------------------------------------------------------------------------

def _argextrema_separated(values: np.ndarray, n: int,
                          *, mode: str, min_sep: int) -> list[int]:
    """Pick n indices that locally extremise `values`, enforcing min_sep
    separation in index space so we don't return n adjacent samples.
    """
    order = np.argsort(values)
    if mode == "max":
        order = order[::-1]
    chosen: list[int] = []
    for idx in order:
        if all(abs(int(idx) - c) >= min_sep
               and abs(int(idx) - c) <= len(values) - min_sep
               for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= n:
            break
    return chosen


def find_worst_thetas(curve: dict, n: int = 5) -> list[tuple[float, float]]:
    """Return the n theta values where covering distance is largest.

    Indices are spaced by at least len/(4n) samples to avoid returning a
    single broad maximum n times.
    """
    vals = curve["min_frob"]
    sep = max(1, len(vals) // (4 * max(n, 1)))
    idxs = _argextrema_separated(vals, n, mode="max", min_sep=sep)
    return [(float(curve["theta"][k]), float(vals[k])) for k in idxs]


def find_best_thetas(curve: dict, n: int = 5) -> list[tuple[float, float]]:
    """Return the n theta values where covering distance is smallest."""
    vals = curve["min_frob"]
    sep = max(1, len(vals) // (4 * max(n, 1)))
    idxs = _argextrema_separated(vals, n, mode="min", min_sep=sep)
    return [(float(curve["theta"][k]), float(vals[k])) for k in idxs]


# ---------------------------------------------------------------------------
#  Generic-SU(3) comparison
# ---------------------------------------------------------------------------

def _project_to_su3(U: np.ndarray) -> np.ndarray:
    """Phase-project a U(3) matrix to SU(3) by dividing out det(U)^{1/3}."""
    det = np.linalg.det(U)
    # principal cube root of the unit-modulus determinant
    phase = det ** (1.0 / 3.0)
    return U / phase


def compare_to_generic(net: np.ndarray,
                       *, n_samples: int = 200,
                       n_theta_submanifold: int = 1024,
                       seed: int = 12345) -> dict:
    """Compare 95th-pct covering radius on each z-rotation submanifold to a
    generic-SU(3) Haar sample.
    """
    sub01 = coverage_curve(net, 0, 1, n_theta=n_theta_submanifold)
    sub02 = coverage_curve(net, 0, 2, n_theta=n_theta_submanifold)
    sub12 = coverage_curve(net, 1, 2, n_theta=n_theta_submanifold)

    rng = np.random.default_rng(seed)
    # scipy's unitary_group uses numpy global state if we don't pass a seed,
    # but the modern signature accepts a Generator/seed via random_state.
    samples = unitary_group.rvs(3, size=n_samples, random_state=rng)
    gen_dists = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        U = _project_to_su3(samples[k])
        d = _frob_batch(U, net)
        gen_dists[k] = float(np.min(d))

    return {
        "submanifold_95pct_01": sub01["pct_95"],
        "submanifold_95pct_02": sub02["pct_95"],
        "submanifold_95pct_12": sub12["pct_95"],
        "generic_95pct":        float(np.percentile(gen_dists, 95)),
    }


# ---------------------------------------------------------------------------
#  Smoke test / report
# ---------------------------------------------------------------------------

def _fmt_table(rows: list[tuple[str, dict]]) -> str:
    hdr = f"{'family':>10}  {'pct50':>8}  {'pct75':>8}  {'pct95':>8}  {'max':>8}  {'mean':>8}"
    out = [hdr, "-" * len(hdr)]
    for name, c in rows:
        out.append(
            f"{name:>10}  "
            f"{c['pct_50']:8.4f}  {c['pct_75']:8.4f}  {c['pct_95']:8.4f}  "
            f"{c['max']:8.4f}  {c['mean']:8.4f}"
        )
    return "\n".join(out)


if __name__ == "__main__":
    print(f"Loading e0-net from {_E0_NET_PATH}...")
    net, meta = load_e0_net()
    print(f"  loaded {len(net)} unitaries, shape {net.shape}")

    print("\nComputing coverage curves (n_theta=1024) on each submanifold...")
    families = [("R_Z(0,1)", 0, 1), ("R_Z(0,2)", 0, 2), ("R_Z(1,2)", 1, 2)]
    curves: dict[str, dict] = {}
    for name, i, j in families:
        curves[name] = coverage_curve(net, i, j, n_theta=1024)

    print("\nCovering radius on z-rotation submanifolds:")
    print(_fmt_table([(n, curves[n]) for n, _, _ in families]))

    tightest_name = min(families, key=lambda f: curves[f[0]]["pct_95"])[0]
    radii_95 = [curves[n]["pct_95"] for n, _, _ in families]
    spread = max(radii_95) - min(radii_95)
    print(f"\nTightest 95th-pct: {tightest_name} "
          f"(pct95={curves[tightest_name]['pct_95']:.4f})")
    print(f"Spread across families: {spread:.4e} "
          f"({'consistent with Galois symmetry' if spread < 1e-3 else 'asymmetric — investigate'})")

    print("\nWorst-5 theta on each submanifold:")
    for name, _, _ in families:
        worst = find_worst_thetas(curves[name], n=5)
        worst_str = ", ".join(f"theta={th:6.4f} d={d:6.4f}" for th, d in worst)
        print(f"  {name:>10}: {worst_str}")

    print("\nBest-5 theta on R_Z(0,1):")
    for th, d in find_best_thetas(curves["R_Z(0,1)"], n=5):
        print(f"    theta={th:6.4f}  d={d:.6e}")

    print("\nGeneric-SU(3) comparison (200 Haar samples)...")
    comp = compare_to_generic(net, n_samples=200, n_theta_submanifold=1024)
    print(f"  submanifold 95pct (0,1): {comp['submanifold_95pct_01']:.4f}")
    print(f"  submanifold 95pct (0,2): {comp['submanifold_95pct_02']:.4f}")
    print(f"  submanifold 95pct (1,2): {comp['submanifold_95pct_12']:.4f}")
    print(f"  generic SU(3) 95pct    : {comp['generic_95pct']:.4f}")

    out_path = Path("/tmp/z_rot_coverage.npz")
    np.savez(
        out_path,
        theta_01=curves["R_Z(0,1)"]["theta"],
        min_frob_01=curves["R_Z(0,1)"]["min_frob"],
        theta_02=curves["R_Z(0,2)"]["theta"],
        min_frob_02=curves["R_Z(0,2)"]["min_frob"],
        theta_12=curves["R_Z(1,2)"]["theta"],
        min_frob_12=curves["R_Z(1,2)"]["min_frob"],
    )
    print(f"\nSaved coverage curves to {out_path}")

    # Feasibility rule of thumb: commutator factoring gives |A - I| ~ 1.4 * sqrt(radius);
    # for the second iteration to land inside the net's coverage we need
    # 1.4 * sqrt(r) < r, i.e. r > 1.96.  In the small-radius regime where we
    # want CONVERGENCE we instead need the post-commutator residual to be
    # SMALLER than the starting radius, i.e. 1.4 * sqrt(r) < r  =>  r > 1.96,
    # OR equivalently we need r < ~0.5 so that 1.4*sqrt(r) < ~1 (one iteration
    # of the recursion brings us into a regime where the cover is dense).
    worst_pct95 = max(radii_95)
    if worst_pct95 < 0.5:
        verdict = ("SK recursion CAN converge from depth 0 "
                   f"(worst submanifold pct95={worst_pct95:.3f} < 0.5)")
    else:
        verdict = ("SK recursion CANNOT converge from depth 0 "
                   f"(worst submanifold pct95={worst_pct95:.3f} >= 0.5); "
                   "need a denser base net or a smarter base case")
    print(f"\nConclusion: {verdict}")
