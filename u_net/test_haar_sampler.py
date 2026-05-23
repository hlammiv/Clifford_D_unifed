"""Tests for ``u_net.haar_sampler``.

Coverage:

* ``haar_su3`` returns valid SU(3) matrices (|det − 1| < 1e-10, U U† ≈ I).
* ``haar_su3_with_dedup`` enforces minimum Frobenius separation.
* ``coverage_radius_estimate`` returns a plausible coverage radius for 1k
  Haar samples vs SU(3) (median ~0.6–1.2 by the analytic SU(3) volume).

Run with ``pytest u_net/test_haar_sampler.py -v``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from haar_sampler import (
    coverage_radius_estimate,
    haar_su3,
    haar_su3_with_dedup,
)


# ---------------------------------------------------------------------------
#  haar_su3
# ---------------------------------------------------------------------------

def test_haar_su3_returns_su3_matrices():
    """1000 samples must satisfy |det − 1| < 1e-10 and U U† ≈ I."""
    samples = haar_su3(1000, seed=42)
    assert samples.shape == (1000, 3, 3)
    assert samples.dtype == np.complex128

    dets = np.linalg.det(samples)
    det_err = np.abs(dets - 1.0).max()
    assert det_err < 1e-10, (
        f"max |det − 1| = {det_err:.3e}; samples are not in SU(3)"
    )

    # U U† − I, batched.
    eye3 = np.eye(3, dtype=np.complex128)
    prod = samples @ samples.conj().transpose(0, 2, 1)
    diff = prod - eye3[None]
    unitarity_err = np.sqrt(np.einsum("kij,kij->k", diff.conj(), diff).real).max()
    assert unitarity_err < 1e-10, (
        f"max |U U† − I|_F = {unitarity_err:.3e}; samples are not unitary"
    )


def test_haar_su3_seed_reproducible():
    a = haar_su3(50, seed=123)
    b = haar_su3(50, seed=123)
    assert np.array_equal(a, b)

    c = haar_su3(50, seed=124)
    # Different seed should give different matrices (with overwhelming probability).
    assert not np.array_equal(a, c)


def test_haar_su3_rejects_bad_n():
    with pytest.raises(ValueError):
        haar_su3(0, seed=0)
    with pytest.raises(ValueError):
        haar_su3(-3, seed=0)


def test_haar_su3_distribution_sanity():
    """Average ``|U|_F`` for Haar U(3) is sqrt(3) for any unitary; just a sanity
    check that the sampler isn't returning all-zero or identity-like outputs."""
    samples = haar_su3(500, seed=7)
    norms = np.sqrt(np.einsum("kij,kij->k", samples.conj(), samples).real)
    # |U|_F = sqrt(3) exactly for any unitary U(3).
    assert np.allclose(norms, math.sqrt(3.0), atol=1e-10)


# ---------------------------------------------------------------------------
#  haar_su3_with_dedup
# ---------------------------------------------------------------------------

def test_dedup_enforces_separation():
    """With dedup_tol=0.5, no pair of accepted samples should be closer than 0.5
    in Frobenius distance."""
    samples = haar_su3_with_dedup(200, dedup_tol=0.5, seed=7, max_tries=10_000)
    assert samples.shape[0] >= 1
    # Compute all pairwise distances and check.
    m = samples.shape[0]
    if m < 2:
        return
    diffs = samples[:, None, :, :] - samples[None, :, :, :]
    dists = np.sqrt(np.einsum("ijab,ijab->ij", diffs.conj(), diffs).real)
    # Mask diagonal.
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    assert min_dist >= 0.5 - 1e-9, (
        f"min pairwise dist = {min_dist:.6f}; dedup tol 0.5 violated"
    )


def test_dedup_returns_su3():
    samples = haar_su3_with_dedup(50, dedup_tol=0.3, seed=13, max_tries=5_000)
    dets = np.linalg.det(samples)
    assert np.abs(dets - 1.0).max() < 1e-10


def test_dedup_caps_at_max_tries():
    """A tight dedup_tol that exhausts max_tries should return fewer than n."""
    # tol=4.0 is bigger than the SU(3) diameter (2*sqrt(3) ≈ 3.46), so the
    # second sample can never be accepted.
    samples = haar_su3_with_dedup(50, dedup_tol=4.0, seed=0, max_tries=200)
    assert samples.shape[0] == 1


def test_dedup_zero_tol_returns_n():
    """With dedup_tol=0, every Haar draw should be accepted (probability-1)."""
    samples = haar_su3_with_dedup(20, dedup_tol=0.0, seed=5, max_tries=100)
    assert samples.shape[0] == 20


# ---------------------------------------------------------------------------
#  coverage_radius_estimate
# ---------------------------------------------------------------------------

def test_coverage_radius_basic_keys():
    samples = haar_su3(100, seed=2)
    stats = coverage_radius_estimate(samples, n_test=200, seed=99)
    for key in ("n_samples", "n_test", "min", "median", "p95", "p99", "max",
                "fraction_within_0p5", "fraction_within_1p0"):
        assert key in stats
    assert stats["n_samples"] == 100
    assert stats["n_test"] == 200
    assert 0.0 <= stats["min"] <= stats["median"] <= stats["p95"] <= stats["p99"] \
        <= stats["max"]


def test_coverage_radius_at_1000_samples_is_plausible():
    """With 1000 samples and 1000 held-out targets the median NN distance
    should sit in roughly [0.4, 1.5] — well below the SU(3) Frobenius diameter
    (2 sqrt 3 ≈ 3.46) and well above 0."""
    samples = haar_su3(1000, seed=42)
    stats = coverage_radius_estimate(samples, n_test=1000, seed=43)
    median = stats["median"]
    assert 0.4 < median < 1.5, (
        f"median NN distance {median:.3f} outside plausible [0.4, 1.5] for "
        f"1000 Haar SU(3) samples; sampler or estimator broken"
    )
    # Sanity: with 1000 Haar samples in 8-dimensional SU(3), the median NN is
    # ~1.04, so a sizable fraction (but not majority) of test points sit within
    # 1.0 of their nearest sample.  We just want non-trivial coverage.
    assert stats["fraction_within_1p0"] > 0.2, (
        f"fraction_within_1p0 = {stats['fraction_within_1p0']:.3f}; sampler"
        f" coverage is suspiciously low"
    )
