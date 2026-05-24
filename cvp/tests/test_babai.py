"""tests/test_babai.py — sanity checks for cvp.babai.babai_x1.

Run::

    python3 -m pytest cvp/tests/test_babai.py -v
"""
from __future__ import annotations

import math
import random
from typing import Sequence, Tuple

import numpy as np
import pytest

from cvp import babai_x1, q_form


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_OMEGA = [
    complex(math.cos(2 * math.pi * j / 9), math.sin(2 * math.pi * j / 9))
    for j in range(6)
]


def _sigma_1(a: Sequence[int]) -> complex:
    """Principal complex embedding ``σ_1(a) = Σ a_j · e^{2πi j / 9}``."""
    return sum(a[j] * _OMEGA[j] for j in range(6))


def _target(theta: float, f: int) -> complex:
    return (3.0 ** f) * complex(math.cos(theta / 2.0), math.sin(theta / 2.0))


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_theta0_f3_top_is_27():
    """θ = 0, f = 3: target z = 27. The integer 27 ≡ (27, 0, 0, 0, 0, 0)
    has σ_1 = 27 exactly and q = 27² = 729 (well inside 2·3⁶ = 1458). It
    should be the top candidate."""
    cands = babai_x1(0.0, 3, n_candidates=10)
    assert len(cands) > 0, "no candidates returned"
    top = cands[0]
    assert top == (27, 0, 0, 0, 0, 0), f"top is {top}, expected (27,0,0,0,0,0)"
    # σ_1 hit is exact (within float epsilon).
    assert abs(_sigma_1(top) - 27.0) < 1e-9


def test_returns_at_most_n_candidates():
    """The length of the returned list is at most n_candidates."""
    for n in (1, 5, 20, 100):
        cands = babai_x1(math.pi / 3, 4, n_candidates=n)
        assert len(cands) <= n


def test_candidate_types():
    """Every returned candidate is a length-6 tuple of Python ints."""
    cands = babai_x1(0.7, 4, n_candidates=10)
    assert len(cands) > 0
    for a in cands:
        assert isinstance(a, tuple)
        assert len(a) == 6
        for x in a:
            assert isinstance(x, int)


# ---------------------------------------------------------------------------
# Budget + tolerance contracts
# ---------------------------------------------------------------------------

def test_q_budget_respected():
    """Every returned candidate satisfies q(a) ≤ 2·3^{2f}."""
    rng = random.Random(20260524)
    for _ in range(5):
        theta = rng.uniform(0.0, math.pi)
        f = rng.choice([3, 5, 6])
        budget = 2 * 3 ** (2 * f)
        cands = babai_x1(theta, f, n_candidates=30)
        assert len(cands) > 0, f"no cands at θ={theta}, f={f}"
        for a in cands:
            q = q_form(a)
            assert q <= budget, f"q(a)={q} > budget {budget} at θ={theta}, f={f}"


def test_default_eps_tolerance():
    """Top candidate's σ_1 deviation is below the default ε = 8 · 3^{-f}
    bound (the Babai approximation guarantee)."""
    rng = random.Random(31415)
    for _ in range(5):
        theta = rng.uniform(0.0, math.pi)
        f = rng.choice([4, 5, 6])
        cands = babai_x1(theta, f, n_candidates=10)
        assert len(cands) > 0
        z = _target(theta, f)
        eps_default = 8.0 * (3.0 ** (-f))
        top_err = abs(_sigma_1(cands[0]) - z) / (3.0 ** f)
        assert top_err <= eps_default, (
            f"top σ_1 error {top_err:.4g} > eps_default {eps_default:.4g} "
            f"at θ={theta}, f={f}"
        )


def test_custom_eps_respected():
    """All returned candidates satisfy |σ_1(a)/3^f − target| ≤ eps when
    a custom ``eps`` is supplied."""
    theta = math.pi / 7
    f = 5
    eps = 0.01
    cands = babai_x1(theta, f, n_candidates=50, eps=eps)
    assert len(cands) > 0
    z = _target(theta, f)
    for a in cands:
        err = abs(_sigma_1(a) - z) / (3.0 ** f)
        assert err <= eps + 1e-12, f"σ_1 error {err} > eps {eps}"


# ---------------------------------------------------------------------------
# Ordering contract
# ---------------------------------------------------------------------------

def test_candidates_sorted_by_principal_error():
    """The returned list is monotone non-decreasing in |σ_1(a) − z|."""
    theta = 2.137
    f = 6
    cands = babai_x1(theta, f, n_candidates=30)
    assert len(cands) >= 5
    z = _target(theta, f)
    errs = [abs(_sigma_1(a) - z) for a in cands]
    for i in range(1, len(errs)):
        assert errs[i] >= errs[i - 1] - 1e-12, (
            f"candidates out of order at index {i}: {errs[i-1]} > {errs[i]}"
        )


# ---------------------------------------------------------------------------
# Cross-check vs HRSA-style brute enumeration (at f=1, eps=0.5)
# ---------------------------------------------------------------------------

def _hrsa_brute(theta: float, eps_tol: float, f: int):
    """Brute enumerate all ``a ∈ Z[ζ_9]`` with ``q(a) ≤ 8·3^{2f}`` and
    ``|σ_1(a)/3^f − e^{iθ/2}| < eps_tol``. Mirrors HRSA's entryEnumeration
    pre-filter (using HRSA's loose ``A = 8·3^{2f}`` enumeration ball)."""
    target_re = math.cos(theta / 2)
    target_im = math.sin(theta / 2)
    eps_cond = eps_tol * eps_tol
    f_pow = 3 ** f
    A = 8 * f_pow * f_pow
    inv_3f = 1.0 / f_pow
    cos_vals = [math.cos(2 * math.pi * j / 9) for j in range(6)]
    sin_vals = [math.sin(2 * math.pi * j / 9) for j in range(6)]
    out = []
    max_a3 = int(math.ceil(math.sqrt(A / 3)))
    for a3 in range(-max_a3, max_a3 + 1):
        b3 = A - 3 * a3 * a3
        if b3 < 0:
            continue
        max_a4 = int(math.ceil(math.sqrt(b3)))
        for a4 in range(-max_a4, max_a4 + 1):
            b4 = b3 - 3 * a4 * a4
            if b4 < 0:
                continue
            max_a5 = int(math.ceil(math.sqrt(b4)))
            for a5 in range(-max_a5, max_a5 + 1):
                b5 = b4 - 3 * a5 * a5
                if b5 < 0:
                    continue
                max_b = int(math.ceil(math.sqrt(b5)))
                for b0 in range(-max_b, max_b + 1):
                    if (b0 + a3) % 2 != 0:
                        continue
                    bb = b5 - b0 * b0
                    if bb < 0:
                        continue
                    a0 = (b0 + a3) // 2
                    max_b1 = int(math.ceil(math.sqrt(bb)))
                    for b1 in range(-max_b1, max_b1 + 1):
                        if (b1 + a4) % 2 != 0:
                            continue
                        bb1 = bb - b1 * b1
                        if bb1 < 0:
                            continue
                        a1 = (b1 + a4) // 2
                        max_b2 = int(math.ceil(math.sqrt(bb1)))
                        for b2 in range(-max_b2, max_b2 + 1):
                            if (b2 + a5) % 2 != 0:
                                continue
                            a2 = (b2 + a5) // 2
                            re = (cos_vals[0] * a0 + cos_vals[1] * a1
                                  + cos_vals[2] * a2 + cos_vals[3] * a3
                                  + cos_vals[4] * a4 + cos_vals[5] * a5) * inv_3f
                            im = (sin_vals[0] * a0 + sin_vals[1] * a1
                                  + sin_vals[2] * a2 + sin_vals[3] * a3
                                  + sin_vals[4] * a4 + sin_vals[5] * a5) * inv_3f
                            dx = re - target_re
                            dy = im - target_im
                            if dx * dx + dy * dy < eps_cond:
                                out.append((a0, a1, a2, a3, a4, a5))
    return out


def test_top10_overlap_vs_hrsa_f1():
    """At θ = π/3, f = 1, eps = 0.5, babai_x1's top-10 (sorted by
    principal-place error) overlaps HRSA's top-10 in at least 5 entries.

    The HRSA reference is generated by an in-test brute enumeration of
    the same q-budget ball with the same σ_1-distance cutoff. We
    intersect after applying the canonical ``q(a) ≤ 2·3^{2f}`` Householder
    constraint (HRSA's loose ``A = 8·3^{2f}`` ball admits some excess
    candidates that fail the joint-identity later)."""
    theta = math.pi / 3
    f = 1
    eps_pct = 0.5
    eps_tol = eps_pct / (2 * math.sqrt(2))  # HRSA: ε/(2√2 c), c=1

    hrsa_all = _hrsa_brute(theta, eps_tol, f)
    q_budget = 2 * 3 ** (2 * f)
    hrsa = [a for a in hrsa_all if q_form(a) <= q_budget]
    hrsa.sort(key=lambda a: abs(_sigma_1(a) - _target(theta, f)))

    babai = babai_x1(theta, f, n_candidates=200, eps=eps_tol)

    # All babai candidates must be valid (inside HRSA's set).
    assert set(babai).issubset(set(hrsa)), (
        "babai produced candidates outside the HRSA-valid set"
    )

    overlap = len(set(babai[:10]) & set(hrsa[:10]))
    assert overlap >= 5, (
        f"top-10 overlap is {overlap}/10, expected ≥ 5. "
        f"HRSA top-10: {hrsa[:10]}; babai top-10: {babai[:10]}"
    )


# ---------------------------------------------------------------------------
# Quick perf smoke (not a strict bound — just a check it isn't seconds)
# ---------------------------------------------------------------------------

def test_perf_smoke_f6_under_2s():
    """babai_x1(θ=π/3, f=6, n=100) returns in well under 2 seconds.
    This is the spec's perf target for Phase 2 at HRSA's sweet spot."""
    import time
    t0 = time.time()
    cands = babai_x1(math.pi / 3, 6, n_candidates=100)
    dt = time.time() - t0
    assert len(cands) > 0
    assert dt < 2.0, f"babai_x1 f=6 wall = {dt*1000:.0f} ms (> 2000 ms)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
