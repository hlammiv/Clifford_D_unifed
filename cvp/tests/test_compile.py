"""tests/test_compile.py — Phase 5 driver smoke tests.

Tests ``cvp_compile.cvp_compile``: the top-level Phase 5 driver that loops
f, drives Phases 2-4, and returns a unified-schema JSON record.

Per the operating-point caveat from Phase 3, these tests stay in the loose-ε
regime where the algorithm is known to converge (eps >= 0.05 at f ~ 2-4).
Tight-ε behavior (eps <= 1e-3) is covered by the offline 20-cell validation
sweep in ``unified/cvp_validate_20.py``, not unit tests.

Run::

    python3 -m pytest cvp/tests/test_compile.py -v

Some tests need Sage (via the persistent ``_NormEqWorker``). They skip
cleanly when Sage isn't available.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pytest

from cvp_compile import (
    _babai_f_min,
    _n_candidates_recommend,
    _orbit_floor_f,
    cvp_compile,
    f_start_recommend,
)
from cvp.diophantine import _NormEqWorker


# ---------------------------------------------------------------------------
# Sage gate (mirrors test_diophantine.py)
# ---------------------------------------------------------------------------


_SAGE_ENV = Path(os.environ.get("SAGE_ENV", "/home/hlamm/miniforge3/envs/sage"))


def _sage_available() -> bool:
    py = _SAGE_ENV / "bin" / "python"
    if not py.exists():
        return False
    try:
        out = subprocess.run(
            [str(py), "-c", "import cypari2"],
            env={
                "PATH": f"{_SAGE_ENV}/bin:" + os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", "/home/hlamm"),
            },
            capture_output=True,
            timeout=10,
        )
        return out.returncode == 0
    except Exception:
        return False


_HAS_SAGE = _sage_available()

needs_sage = pytest.mark.skipif(
    not _HAS_SAGE,
    reason=f"Sage env not found at {_SAGE_ENV} (cypari2 import failed).",
)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no worker spinup)
# ---------------------------------------------------------------------------


def test_f_start_recommend_uses_max_of_two_bounds():
    """f_start picks the larger of Babai bound and orbit floor."""
    # At eps=1e-3, Babai bound = ceil(log_3(8/1e-3)) = ceil(log_3(8000)) = 9.
    # Orbit floor = ceil(log_3(pi/(9·1e-3))) = ceil(log_3(349.06)) = 6.
    # Max(9, 6, 2) = 9.
    assert f_start_recommend(1e-3) == 9
    # At eps=0.5, Babai bound = ceil(log_3(16)) = 3.
    # Orbit floor = ceil(log_3(pi/4.5)) = ceil(log_3(0.698)) = clamped to 2.
    # Max(3, 2, 2) = 3.
    assert f_start_recommend(0.5) >= 2


def test_babai_and_orbit_bounds_are_monotone_in_eps():
    """Both bounds grow as eps shrinks."""
    for e1, e2 in [(0.1, 0.01), (1e-3, 1e-4), (1e-6, 1e-8)]:
        assert _babai_f_min(e2) >= _babai_f_min(e1)
        assert _orbit_floor_f(e2) >= _orbit_floor_f(e1)


def test_n_candidates_doubles_per_retry_and_caps():
    """Selinger scaling doubles and caps."""
    # Use a loose eps so the base is well below the cap.
    eps = 0.5  # base = max(10, ceil(4·sqrt(2)/0.5)) = max(10, 12) = 12
    n0 = _n_candidates_recommend(eps, retry_count=0, cap=1000)
    n1 = _n_candidates_recommend(eps, retry_count=1, cap=1000)
    n2 = _n_candidates_recommend(eps, retry_count=2, cap=1000)
    assert n1 == 2 * n0
    assert n2 == 4 * n0
    # Cap kicks in.
    big = _n_candidates_recommend(1e-9, retry_count=10, cap=1000)
    assert big == 1000


# ---------------------------------------------------------------------------
# End-to-end driver tests (need Sage)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def worker():
    if not _HAS_SAGE:
        pytest.skip("Sage env not available")
    w = _NormEqWorker()
    w.start()
    yield w
    w.close()


@needs_sage
def test_cvp_compile_loose_eps_succeeds(worker):
    """At θ=π/2 ε=0.2 the driver should converge fast at f_start=2."""
    rec = cvp_compile(
        theta=math.pi / 2, eps=0.2,
        f_start=2, max_f_iters=3,
        max_x1_to_try=5, max_pairs_per_x1=4,
        worker=worker,
    )
    assert rec["achieved"]["success"] is True
    assert rec["achieved"]["epsilon_passed"] is True
    assert rec["achieved"]["achieved_frob"] <= 0.2 + 1e-9
    assert rec["decomposition"]["N_D"] >= 0
    assert rec["sanity_checks"]["q_sum_ok"] is True
    assert rec["sanity_checks"]["unitary_in_ring"] is True
    assert rec["sanity_checks"]["reconstruction_residual"] == 0


@needs_sage
def test_cvp_compile_small_theta_succeeds(worker):
    """θ=0.1 ε=0.5 — loose ε, should hit at f=2 quickly."""
    rec = cvp_compile(
        theta=0.1, eps=0.5,
        f_start=2, max_f_iters=3,
        max_x1_to_try=5, max_pairs_per_x1=4,
        worker=worker,
    )
    assert rec["achieved"]["success"] is True
    assert rec["achieved"]["achieved_frob"] <= 0.5 + 1e-9


@needs_sage
def test_cvp_compile_returns_unified_schema_shape(worker):
    """Returned dict has all expected top-level keys (matches the unified
    schema example, minus optional fields)."""
    rec = cvp_compile(
        theta=math.pi / 2, eps=0.2,
        f_start=2, max_f_iters=2,
        max_x1_to_try=4, max_pairs_per_x1=4,
        worker=worker,
    )
    expected_top = {
        "identification", "inputs", "target", "achieved", "unitary",
        "decomposition", "sanity_checks", "performance",
        "attempted_f_levels", "errors",
    }
    assert expected_top <= set(rec.keys())

    # identification block
    ident = rec["identification"]
    assert ident["backend"] == "cvp-babai"
    assert ident["schema_version"] == "1.0"
    assert "host" in ident and "timestamp" in ident

    # achieved.method == "cvp-babai"
    assert rec["achieved"]["method"] == "cvp-babai"
    # unitary block: V is 3x3x6 integer blob
    V = rec["unitary"]["V"]
    assert len(V) == 3 and all(len(row) == 3 for row in V)
    assert all(len(V[i][j]) == 6 for i in range(3) for j in range(3))
    # decomposition has the expected fields
    assert set(rec["decomposition"]) >= {
        "N_D", "syllables", "sde_chi_initial", "sde_chi_final",
    }
    # performance has wall_seconds
    assert rec["performance"]["wall_seconds"] > 0


@needs_sage
def test_cvp_compile_failure_records_fallback(worker):
    """An impossibly tight eps at low f exhausts cleanly and reports a
    fallback method."""
    # Force the failure path: tiny f_max ensures no f-level reaches the
    # required orbit-floor regime for eps=1e-4.
    rec = cvp_compile(
        theta=math.pi / 2, eps=1e-4,
        f_start=2, f_max=3, max_f_iters=2,
        max_x1_to_try=2, max_pairs_per_x1=2,
        worker=worker, fallback_method="sk",
    )
    assert rec["achieved"]["success"] is False
    assert rec["achieved"]["epsilon_passed"] is False
    assert rec.get("fallback_method") == "sk"
    assert isinstance(rec["attempted_f_levels"], list)
    assert len(rec["attempted_f_levels"]) >= 1
