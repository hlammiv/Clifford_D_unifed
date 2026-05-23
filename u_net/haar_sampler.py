"""Haar-measure SU(3) sampler, dedup helper, coverage estimator.

Phase C of the SK U-net bootstrap.  We need a uniform sampler on SU(3) to seed
the U-net (Phase D, separate file).  This module provides:

* :func:`haar_su3` — n Haar-random SU(3) matrices via QR of a complex Ginibre
  matrix (Mezzadri 2007), with the global ``det`` divided out.
* :func:`haar_su3_with_dedup` — same, but rejecting any sample within
  ``dedup_tol`` Frobenius distance of an already-accepted one (useful when
  seeding a U-net so the kernels are not redundant).
* :func:`coverage_radius_estimate` — Monte-Carlo estimate of how well a set of
  samples covers SU(3) (max / quantile of nearest-neighbour Frobenius distance
  from a held-out Haar batch).

Only ``numpy`` is required.  ``scipy.stats.unitary_group`` is intentionally not
used here so the sampler is self-contained and reproducible.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
#  Haar sampler
# ---------------------------------------------------------------------------

def _ginibre_qr_unitary(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Return ``n_samples`` Haar-uniform U(3) matrices (Mezzadri 2007).

    Sample G complex Ginibre (i.i.d. complex normal); QR-factor G = QR;
    rotate Q by the diagonal sign matrix of R to remove the QR sign ambiguity.
    The result is Haar-uniform on U(3).
    """
    # Real + imaginary i.i.d. standard normal => complex normal with var 1.
    real = rng.standard_normal((n_samples, 3, 3))
    imag = rng.standard_normal((n_samples, 3, 3))
    G = (real + 1j * imag) / math.sqrt(2.0)
    Q, R = np.linalg.qr(G)
    # Sign-correction: diag(R) / |diag(R)|.
    diag_R = np.einsum("...ii->...i", R)
    phases = diag_R / np.abs(diag_R)
    # Right-multiply Q by diag(phases) so each column gets its sign correction.
    Q = Q * phases[:, None, :]
    return Q.astype(np.complex128)


def haar_su3(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Return ``n`` Haar-random SU(3) matrices.

    Uses Mezzadri's QR algorithm on complex Ginibre matrices to get U(3),
    then divides each U by ``det(U)^(1/3)`` (principal cube root) to project
    to SU(3).  Because Haar U(3) factors as Haar SU(3) × U(1) (phase), the
    projected matrices are Haar-distributed on SU(3).

    Parameters
    ----------
    n : int
        Number of SU(3) matrices to return.
    seed : int or None
        Seed for the underlying ``numpy.random.default_rng``.

    Returns
    -------
    (n, 3, 3) complex128 ndarray.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = np.random.default_rng(seed)
    Us = _ginibre_qr_unitary(rng, n)
    dets = np.linalg.det(Us)
    # Principal cube root: for complex z = r e^{i arg(z)} the principal root is
    # r^{1/3} e^{i arg(z)/3}.  numpy's ** with complex base uses principal log
    # via cmath conventions.
    cube_roots = dets ** (1.0 / 3.0)
    out = Us / cube_roots[:, None, None]
    return out


# ---------------------------------------------------------------------------
#  Dedup sampler
# ---------------------------------------------------------------------------

def _frob_dist_batch(target: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """|target − candidates[k]|_F for k = 0 .. len(candidates)-1.

    Shapes: target (3, 3), candidates (k, 3, 3); returns (k,) float array.
    """
    if candidates.shape[0] == 0:
        return np.empty(0, dtype=float)
    diff = candidates - target[None, :, :]
    return np.sqrt(np.einsum("kij,kij->k", diff.conj(), diff).real)


def haar_su3_with_dedup(
    n: int,
    dedup_tol: float,
    seed: Optional[int] = None,
    max_tries: Optional[int] = None,
) -> np.ndarray:
    """Sample up to ``n`` SU(3) matrices, rejecting near-duplicates.

    Each candidate is generated via :func:`haar_su3` and accepted only if its
    Frobenius distance to every previously-accepted sample exceeds
    ``dedup_tol``.  Returns when either ``n`` samples are accepted or
    ``max_tries`` candidates have been drawn — whichever comes first.

    Parameters
    ----------
    n : int
        Target sample count.
    dedup_tol : float
        Minimum Frobenius separation between any two accepted samples.
    seed : int or None
        Seed for the underlying RNG (deterministic).
    max_tries : int or None
        Hard cap on candidate draws (default ``10 * n``).

    Returns
    -------
    (m, 3, 3) complex128 ndarray with ``m <= n``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if dedup_tol < 0:
        raise ValueError(f"dedup_tol must be >= 0, got {dedup_tol}")
    if max_tries is None:
        max_tries = 10 * n
    rng = np.random.default_rng(seed)

    accepted = np.empty((n, 3, 3), dtype=np.complex128)
    n_accepted = 0
    n_tried = 0
    # Batched candidate draws to avoid per-sample numpy overhead.
    batch_size = max(64, min(1024, n))
    while n_accepted < n and n_tried < max_tries:
        remaining = max_tries - n_tried
        draw = min(batch_size, remaining)
        # Use the same RNG so the stream is reproducible.
        sub_seed = int(rng.integers(0, 2**31 - 1))
        cand = haar_su3(draw, seed=sub_seed)
        n_tried += draw
        for k in range(draw):
            if n_accepted == 0:
                accepted[0] = cand[k]
                n_accepted = 1
                continue
            d = _frob_dist_batch(cand[k], accepted[:n_accepted])
            if d.min() > dedup_tol:
                accepted[n_accepted] = cand[k]
                n_accepted += 1
                if n_accepted == n:
                    break
    return accepted[:n_accepted]


# ---------------------------------------------------------------------------
#  Coverage estimator
# ---------------------------------------------------------------------------

def _nearest_distance_to_set(target: np.ndarray, samples: np.ndarray) -> float:
    """Minimum Frobenius distance from ``target`` to any row of ``samples``."""
    d = _frob_dist_batch(target, samples)
    return float(d.min())


def coverage_radius_estimate(
    samples: np.ndarray,
    n_test: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """Estimate the coverage radius of ``samples`` on SU(3) by Monte Carlo.

    Generate ``n_test`` fresh Haar SU(3) targets; for each, compute the
    Frobenius distance to the nearest matrix in ``samples``.  Return summary
    statistics on the distribution of nearest-neighbour distances.

    The reported ``coverage_radius`` is the worst observed distance — the
    radius of a Frobenius ball around the nearest sample that the random test
    matrix sits within.  A coverage radius significantly larger than the
    typical NN gap suggests holes in the sample set.

    Parameters
    ----------
    samples : (m, 3, 3) complex ndarray
        The candidate covering set.
    n_test : int
        Number of held-out Haar SU(3) targets to probe with.
    seed : int or None
        RNG seed (deterministic).

    Returns
    -------
    dict with keys: ``min``, ``median``, ``p95``, ``p99``, ``max``,
    ``fraction_within_0p5``, ``fraction_within_1p0``, ``n_samples``,
    ``n_test``.
    """
    if samples.ndim != 3 or samples.shape[1:] != (3, 3):
        raise ValueError(f"samples must be (m, 3, 3), got {samples.shape}")
    if samples.shape[0] == 0:
        raise ValueError("samples is empty")
    if n_test <= 0:
        raise ValueError(f"n_test must be positive, got {n_test}")

    targets = haar_su3(n_test, seed=seed)

    # Vectorise: compute all pairwise distances test x samples in batches to
    # avoid O(n_test * m * 18) memory blowup.
    dists = np.empty(n_test, dtype=float)
    # Batch over test targets, all samples at once.
    chunk = max(1, min(n_test, 256))
    for start in range(0, n_test, chunk):
        end = min(n_test, start + chunk)
        diff = targets[start:end, None, :, :] - samples[None, :, :, :]
        # Frobenius norm per (test, sample) pair.
        sq = np.einsum("tsij,tsij->ts", diff.conj(), diff).real
        d = np.sqrt(sq)
        dists[start:end] = d.min(axis=1)

    return {
        "n_samples": int(samples.shape[0]),
        "n_test": int(n_test),
        "min": float(dists.min()),
        "median": float(np.median(dists)),
        "p95": float(np.quantile(dists, 0.95)),
        "p99": float(np.quantile(dists, 0.99)),
        "max": float(dists.max()),
        "fraction_within_0p5": float((dists < 0.5).mean()),
        "fraction_within_1p0": float((dists < 1.0).mean()),
    }
