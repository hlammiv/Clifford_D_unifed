"""tests/test_gram.py — bit-exact validation of cvp.gram against HRSA quad.

Run::

    python3 -m pytest cvp/tests/test_gram.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from cvp.gram import B, B_lll, U_lll, q_form


def _hrsa_quad(coefs):
    """Reference re-implementation of HRSA ringZ9::quad() in pure Python.

    Mirrors hrsa/cyclotomic_int9.h:337
        quad(a) = sum_i a_i^2 - a_0 a_3 - a_1 a_4 - a_2 a_5
    """
    a = [int(x) for x in coefs]
    assert len(a) == 6
    return (
        a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        + a[3] * a[3] + a[4] * a[4] + a[5] * a[5]
        - a[0] * a[3] - a[1] * a[4] - a[2] * a[5]
    )


def test_q_form_matches_hrsa_quad():
    """q_form agrees with HRSA quad on 10k random integer vectors."""
    rng = np.random.default_rng(20260524)
    n = 10_000
    samples = rng.integers(-1000, 1001, size=(n, 6), dtype=np.int64)
    mismatches = 0
    for c in samples:
        if q_form(c) != _hrsa_quad(c):
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/{n} bit-exact mismatches vs HRSA quad"


def test_q_form_positive_definite():
    """q_form is non-negative and vanishes only at the origin."""
    # Zero vector
    assert q_form([0] * 6) == 0

    rng = np.random.default_rng(31415)
    n = 5000
    samples = rng.integers(-500, 501, size=(n, 6), dtype=np.int64)
    for c in samples:
        v = q_form(c)
        assert v >= 0, f"q_form returned {v} < 0 for {c.tolist()}"
        if v == 0:
            assert all(int(x) == 0 for x in c), (
                f"q_form vanished at non-zero vector {c.tolist()}"
            )


def test_q_form_unit_vectors():
    """q_form(2 ζ_9^k) == 4 (sanity check vs HRSA quad on unit-scaled basis)."""
    for k in range(6):
        e = [0] * 6
        e[k] = 2
        assert q_form(e) == 4, f"q_form(2 e_{k}) = {q_form(e)} != 4"
        # Cross-check the unscaled e_k as well: q_form(e_k) == 1.
        e[k] = 1
        assert q_form(e) == 1, f"q_form(e_{k}) = {q_form(e)} != 1"


def test_lll_unimodular_and_reduces():
    """U is unimodular and B_lll is at most as Frobenius-heavy as B."""
    det_U = int(round(np.linalg.det(U_lll.astype(np.float64))))
    assert det_U in (-1, 1), f"det(U) = {det_U}, expected ±1"

    # Integer reconstruction: B_lll == U @ B @ U.T exactly.
    recon = U_lll @ B @ U_lll.T
    assert np.array_equal(recon, B_lll), "B_lll != U @ B @ U.T"

    # Reduced basis Frobenius norm should not exceed the original.
    fro_B = float(np.linalg.norm(B.astype(np.float64), "fro"))
    fro_Blll = float(np.linalg.norm(B_lll.astype(np.float64), "fro"))
    assert fro_Blll <= fro_B + 1e-9, (
        f"||B_lll||_F={fro_Blll} > ||B||_F={fro_B}; LLL did not reduce"
    )

    # B_lll is symmetric positive-definite (same lattice, different basis).
    assert np.array_equal(B_lll, B_lll.T), "B_lll not symmetric"
    eigs = np.linalg.eigvalsh(B_lll.astype(np.float64))
    assert (eigs > 0).all(), f"B_lll has non-positive eigenvalues: {eigs}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
