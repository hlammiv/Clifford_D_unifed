"""Phase B validation: verify ``su3_euler.euler_decompose`` is U-net ready.

The SK U-net bootstrap relies on the claim that every SU(3) target U can be
expressed as a product of free Cliffords interleaved with rotations of the form
R^Z_{(0,1)}(theta) only.  If that holds, the Phase-A R_z lookup DB (which
caches only the (0,1) family) is sufficient to score every Z-leaf in the
decomposition.

This script generates Haar-random SU(3) targets, runs ``euler_decompose`` on
each, and asserts:

1. Every Z-factor returned has the form ``('Z', 0, 1, theta)``.
2. ``verify_decomposition`` recovers the target within 1e-10 Frobenius distance
   (after global-phase removal).
3. The factor count distribution is bounded (median / max n_z_leaves are
   reported but not asserted past sanity checks).

Run standalone or via ``pytest -v``.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from su3_euler import euler_decompose, verify_decomposition


N_TRIALS = 100
SEED = 20260523
FROB_TOL = 1e-10


def _haar_su3(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return ``n`` Haar-random SU(3) matrices.

    ``scipy.stats.unitary_group`` samples Haar U(3); we project to SU(3) by
    dividing out the global phase ``det(U)^{1/3}``.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    rs = np.random.RandomState(seed)
    Us = unitary_group.rvs(3, size=n, random_state=rs)
    if n == 1:
        Us = Us[None, :, :]
    out = np.empty_like(Us)
    for k in range(n):
        d = np.linalg.det(Us[k])
        phase = d ** (1.0 / 3.0)
        out[k] = Us[k] / phase
    return out


def _classify(factors):
    """Return (n_z_leaves, n_clifford, n_sign, bad_z_subspaces).

    ``bad_z_subspaces`` is a list of (i, j) tuples appearing in Z-factors that
    are NOT ``(0, 1)`` — should be empty.
    """
    n_z = n_c = n_s = 0
    bad = []
    for f in factors:
        kind = f[0]
        if kind == "Z":
            n_z += 1
            i, j = f[1], f[2]
            if (i, j) != (0, 1):
                bad.append((i, j))
        elif kind == "CLIFFORD":
            n_c += 1
        elif kind == "SIGN":
            n_s += 1
        else:
            raise AssertionError(f"unexpected factor kind {kind!r} in {f!r}")
    return n_z, n_c, n_s, bad


def run_validation(n_trials: int = N_TRIALS, seed: int = SEED, verbose: bool = True):
    rng = np.random.default_rng(seed)
    targets = _haar_su3(rng, n_trials)

    n_z_leaves_dist: list[int] = []
    n_factors_dist: list[int] = []
    frob_errs: list[float] = []
    bad_subspace_offenders: list[tuple[int, list]] = []
    decompose_failures: list[tuple[int, float]] = []

    t0 = time.perf_counter()
    for trial in range(n_trials):
        U = targets[trial]
        factors = euler_decompose(U)
        n_z, n_c, n_s, bad = _classify(factors)
        n_z_leaves_dist.append(n_z)
        n_factors_dist.append(len(factors))
        err = verify_decomposition(U, factors, ignore_global_phase=True)
        frob_errs.append(err)
        if bad:
            bad_subspace_offenders.append((trial, bad))
        if err >= FROB_TOL:
            decompose_failures.append((trial, err))
    dt = time.perf_counter() - t0

    success = (not bad_subspace_offenders) and (not decompose_failures)

    summary = {
        "n_trials": n_trials,
        "wall_s": dt,
        "all_z_in_01_subspace": not bad_subspace_offenders,
        "all_frob_within_tol": not decompose_failures,
        "frob_tol": FROB_TOL,
        "median_n_z_leaves": int(statistics.median(n_z_leaves_dist)),
        "max_n_z_leaves": int(max(n_z_leaves_dist)),
        "min_n_z_leaves": int(min(n_z_leaves_dist)),
        "mean_n_z_leaves": float(statistics.mean(n_z_leaves_dist)),
        "median_n_factors": int(statistics.median(n_factors_dist)),
        "max_n_factors": int(max(n_factors_dist)),
        "median_frob_err": float(statistics.median(frob_errs)),
        "max_frob_err": float(max(frob_errs)),
        "bad_subspace_count": len(bad_subspace_offenders),
        "decompose_failure_count": len(decompose_failures),
        "success_rate": (n_trials - len(decompose_failures) - len(bad_subspace_offenders))
        / n_trials,
    }

    if verbose:
        print(f"Phase B validation: {n_trials} Haar SU(3) targets")
        print(f"  wall time:                {dt:6.2f}s "
              f"({dt / n_trials * 1000:.1f} ms / target)")
        print(f"  all Z-factors in (0,1):   {summary['all_z_in_01_subspace']}")
        print(f"  all |product − U|_F<{FROB_TOL:.0e}: "
              f"{summary['all_frob_within_tol']}")
        print(f"  n_z_leaves   min/med/max: "
              f"{summary['min_n_z_leaves']}/{summary['median_n_z_leaves']}/"
              f"{summary['max_n_z_leaves']}")
        print(f"  n_factors    median/max:  "
              f"{summary['median_n_factors']}/{summary['max_n_factors']}")
        print(f"  frob error   median/max:  "
              f"{summary['median_frob_err']:.3e} / {summary['max_frob_err']:.3e}")
        print(f"  success rate:             {summary['success_rate']*100:.1f}%")
        if bad_subspace_offenders:
            print("  BAD SUBSPACES detected:")
            for ti, bad in bad_subspace_offenders[:5]:
                print(f"    trial {ti}: {bad[:5]}{'...' if len(bad) > 5 else ''}")
        if decompose_failures:
            print("  DECOMPOSITION FAILURES:")
            for ti, e in decompose_failures[:5]:
                print(f"    trial {ti}: frob err {e:.3e}")

    return success, summary


# ---------------------------------------------------------------------------
#  pytest entry points
# ---------------------------------------------------------------------------

def test_all_z_factors_target_01_subspace():
    """Every Z-factor returned by euler_decompose must live in subspace (0,1).

    This is the key claim Phase B is verifying: the R_z lookup DB (which caches
    only (0,1) rotations) covers every Z-leaf and no other (i,j) family is
    ever requested by the decomposition algorithm.
    """
    success, summary = run_validation(n_trials=N_TRIALS, seed=SEED, verbose=True)
    assert summary["all_z_in_01_subspace"], (
        f"euler_decompose emitted Z-factors outside (0,1) on "
        f"{summary['bad_subspace_count']} / {summary['n_trials']} trials; "
        f"R_z DB plan blocked — see stdout for offending subspaces"
    )


def test_decomposition_recovers_target():
    """verify_decomposition must drop below FROB_TOL on every trial."""
    success, summary = run_validation(n_trials=N_TRIALS, seed=SEED, verbose=False)
    assert summary["all_frob_within_tol"], (
        f"euler_decompose failed |product − U|_F < {FROB_TOL:.0e} on "
        f"{summary['decompose_failure_count']} / {summary['n_trials']} trials; "
        f"worst error = {summary['max_frob_err']:.3e}"
    )


def test_leaf_count_within_expected_range():
    """Sanity check: n_z_leaves should be O(10^2), not pathological."""
    _success, summary = run_validation(n_trials=N_TRIALS, seed=SEED, verbose=False)
    # Each outer iter emits 8 Z-factors (the basis size); max_outer_iter=12,
    # so the hard ceiling is 96.  Allow some slack (and warn-not-fail above
    # ~96).
    assert summary["max_n_z_leaves"] <= 96, (
        f"max_n_z_leaves={summary['max_n_z_leaves']} exceeds outer-iter cap of 96; "
        f"euler_decompose may be looping at noise floor"
    )
    # We expect at least 8 leaves per non-trivial target.
    assert summary["min_n_z_leaves"] >= 8, (
        f"min_n_z_leaves={summary['min_n_z_leaves']}; Haar samples should never "
        f"be near identity by chance"
    )


if __name__ == "__main__":
    ok, summary = run_validation()
    print()
    print("Summary dict:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
