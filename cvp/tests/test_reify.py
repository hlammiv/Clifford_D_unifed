"""tests/test_reify.py — Phase 4 of the qutrit Babai-CVP pipeline.

Tests the Householder reify (x_1, x_2, x_3) -> V and the canonical_reducer
decompose wrapper. The decompose step relies on
``hrsa.canonical_reducer.get_prefix_table()`` which is ~30 s on first call;
tests that need it share the module-scoped ``decomp_ref`` fixture so the
table is built once.

Run::

    python3 -m pytest cvp/tests/test_reify.py -v
"""
from __future__ import annotations

import cmath
import math
from typing import Tuple

import numpy as np
import pytest

from cvp.diophantine import householder_frobenius  # noqa: F401
from cvp.gram import q_form
from cvp.reify import (
    cvp_compile_one,
    decompose_to_gates,
    reify_householder,
)


# ---------------------------------------------------------------------------
# HRSA f=2 reference triple (same as test_diophantine).
# theta = pi/2, frob = 0.0322809 per HRSA reference.
# ---------------------------------------------------------------------------

_REF_X1: Tuple[int, ...] = (4, -2, 4, 5, -4, -2)
_REF_X2: Tuple[int, ...] = (0, -4, 1, 3, 1, 4)
_REF_X3: Tuple[int, ...] = (-6, 2, 3, -6, 1, -2)
_REF_F: int = 2
_REF_THETA: float = math.pi / 2


def _omega(k: int) -> complex:
    return cmath.exp(2j * math.pi * k / 9)


def _sigma1_blob(V_blob, f_blob) -> np.ndarray:
    """Map a [3][3][6] int blob at exponent f_blob to a 3x3 complex matrix
    using the principal embedding ζ -> e^(2πi/9)."""
    V = np.zeros((3, 3), dtype=np.complex128)
    denom = 3.0 ** f_blob
    for i in range(3):
        for j in range(3):
            s = sum(V_blob[i][j][k] * _omega(k) for k in range(6))
            V[i, j] = s / denom
    return V


# ---------------------------------------------------------------------------
# Pure reify tests (no canonical_reducer needed — fast)
# ---------------------------------------------------------------------------


def test_reify_q_check_holds_on_reference_triple():
    """Phase 3 reference triple satisfies the Householder norm equation."""
    r = reify_householder(_REF_X1, _REF_X2, _REF_X3, _REF_F)
    assert r["q_check"] is True
    assert r["unitary_in_ring"] is True
    assert r["f_blob"] == 2 * _REF_F


def test_reify_q_check_raises_on_bad_triple():
    """An obviously wrong triple raises ValueError when strict=True."""
    bad = (1, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="norm equation"):
        reify_householder(bad, bad, bad, _REF_F, strict=True)


def test_reify_unitarity_bit_exact_in_ring():
    """V V^† = I_3 holds bit-exactly in Z[ζ_9, 1/3^{2f}] (no float roundoff).

    This is the gold-standard correctness check: the q-sum equation
    *algebraically forces* V to be unitary. Any failure here points to a
    bug in the reify, the ring arithmetic, or the q-form definition.
    """
    r = reify_householder(_REF_X1, _REF_X2, _REF_X3, _REF_F)
    assert r["unitary_in_ring"] is True

    # Also confirm the float-embedded V is unitary to machine precision.
    V = _sigma1_blob(r["V_blob"], r["f_blob"])
    err = np.max(np.abs(V @ V.conj().T - np.eye(3)))
    assert err < 1e-12, f"|V V^† - I|_∞ = {err}"


def test_reify_frob_matches_hrsa_reference():
    """The Frobenius residual ||V - R_z(θ)||_F equals HRSA's 0.0322809."""
    r = reify_householder(_REF_X1, _REF_X2, _REF_X3, _REF_F, theta=_REF_THETA)
    # HRSA's reference value from cvp/tests/test_diophantine.py.
    assert abs(r["frob_residual"] - 0.0322809) < 1e-4

    # Independent re-check from the V_blob float embedding.
    V = _sigma1_blob(r["V_blob"], r["f_blob"])
    R = np.diag([
        np.exp(-1j * _REF_THETA / 2),
        np.exp(+1j * _REF_THETA / 2),
        1.0 + 0j,
    ])
    frob_from_blob = float(np.linalg.norm(V - R, "fro"))
    # Must agree to high precision with householder_frobenius().
    assert abs(frob_from_blob - r["frob_residual"]) < 1e-10


def test_reify_householder_convention_matches_hrsa():
    """V = X_(0,1) · (I - u u^†). Row swap exchanges H rows 0 and 1.

    Verifies the convention by constructing H and V independently and
    cross-checking against the reify output. This pins the convention
    described in hrsa/householder_search.h §41 and decompose_impl.h:69.
    """
    f = _REF_F
    f_blob = 2 * f
    r = reify_householder(_REF_X1, _REF_X2, _REF_X3, f)
    V = _sigma1_blob(r["V_blob"], f_blob)

    # Independent reconstruction: u = (x_1, x_2, x_3) / 3^f, H = I - u u^†.
    u = np.array([
        sum(xi * _omega(k) for k, xi in enumerate(_REF_X1)),
        sum(xi * _omega(k) for k, xi in enumerate(_REF_X2)),
        sum(xi * _omega(k) for k, xi in enumerate(_REF_X3)),
    ], dtype=np.complex128) / (3.0 ** f)
    H = np.eye(3, dtype=np.complex128) - np.outer(u, np.conj(u))
    X01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.complex128)
    V_expected = X01 @ H

    err = np.max(np.abs(V - V_expected))
    assert err < 1e-12, (
        f"reify V disagrees with X_(0,1)·(I-uu†) by {err}; "
        "Householder convention is broken."
    )


def test_reify_handles_zero_x3_corner_case():
    """When x_3 = 0 the reify must still produce a valid unitary.

    Choose x_1 = (9,0,...), x_2 = (9,0,...), x_3 = 0; this gives
    q_1+q_2+q_3 = 81+81+0 = 162 = 2·3^{2·2}, satisfying the norm
    equation exactly.
    """
    x_1 = (9, 0, 0, 0, 0, 0)
    x_2 = (9, 0, 0, 0, 0, 0)
    x_3 = (0, 0, 0, 0, 0, 0)
    assert q_form(x_1) + q_form(x_2) + q_form(x_3) == 2 * 3 ** (2 * 2)
    r = reify_householder(x_1, x_2, x_3, f=2)
    assert r["q_check"] is True
    assert r["unitary_in_ring"] is True


# ---------------------------------------------------------------------------
# Decompose tests (require canonical_reducer's 4374-entry prefix table —
# ~30 s build on first call; module-scoped fixture amortises this).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reified_ref():
    """Reified V_blob for the HRSA f=2 reference triple."""
    return reify_householder(_REF_X1, _REF_X2, _REF_X3, _REF_F, theta=_REF_THETA)


@pytest.fixture(scope="module")
def decomp_ref(reified_ref):
    """canonical_reducer output for the reference V. Built once per module."""
    return decompose_to_gates(reified_ref["V_blob"], reified_ref["f_blob"],
                              greedy=True, verify=True)


def test_decompose_succeeds_on_reference_triple(decomp_ref):
    """canonical_reducer reaches sde_chi = 0 with no error."""
    assert decomp_ref["success"] is True
    assert decomp_ref["sde_chi_final"] == 0
    assert decomp_ref["N_D"] >= 0


def test_decompose_reconstruction_is_bit_exact(decomp_ref):
    """The syllable product reproduces the input V bit-for-bit in the ring.

    ``max_coef_residual == 0`` is the canonical correctness witness for
    canonical_reducer (see ``canonical_reducer_2026-05-24`` memory).
    """
    assert decomp_ref["matches_input"] is True
    assert decomp_ref["reconstruction_residual"] == 0


def test_decompose_N_D_within_HRSA_tolerance(decomp_ref):
    """N_D should agree with HRSA's reported value ± 2.

    HRSA at f=2 θ=π/2 ε=0.05 with this exact triple reports N_D in the
    single digits (canonical_reducer typically matches C++ ± 2 per memory
    ``canonical_reducer_2026-05-24``).  We assert a generous upper bound
    here since the C++ ground-truth N_D for this specific triple is not
    pinned in the repo; the strict bit-exactness check above is the
    primary correctness gate.
    """
    # Sanity ceiling: at f=2, sde_chi_initial ≤ 6·2·2 = 24 (for f_blob=4),
    # so N_D ≤ ~24 in the worst case (one D per peel iteration).
    assert decomp_ref["N_D"] <= 30, (
        f"N_D={decomp_ref['N_D']} unreasonably large for f=2 single triple"
    )


def test_cvp_compile_one_returns_schema_shape(decomp_ref):
    """``cvp_compile_one`` returns the unified compile_qutrit_schema_v1 shape.

    Reuses the cached prefix table from ``decomp_ref`` indirectly — the
    second decompose call inside cvp_compile_one is fast.
    """
    rec = cvp_compile_one(
        _REF_X1, _REF_X2, _REF_X3, _REF_F, _REF_THETA, eps=0.05,
    )
    assert set(rec) >= {
        "inputs", "achieved", "unitary", "decomposition",
        "sanity_checks", "performance",
    }
    assert rec["achieved"]["success"] is True
    assert rec["sanity_checks"]["q_sum_ok"] is True
    assert rec["sanity_checks"]["unitary_in_ring"] is True
    assert rec["sanity_checks"]["reconstruction_residual"] == 0
    assert rec["achieved"]["method"] == "cvp-babai"
    assert rec["unitary"]["f"] == 2 * _REF_F
    # frob_residual matches reify
    assert abs(rec["achieved"]["achieved_frob"] - 0.0322809) < 1e-4
    # epsilon_passed: 0.0322809 < 0.05 -> True
    assert rec["achieved"]["epsilon_passed"] is True
