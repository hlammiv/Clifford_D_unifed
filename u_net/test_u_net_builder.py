"""Tests for ``u_net.u_net_builder`` + ``u_net.u_net_lookup``.

Phase D smoke + correctness tests.  We build a tiny tier-0 (eps_u=0.5) U-net
against the pre-built R_z DB at ``/tmp/rz_test.sqlite``, then verify the
UNetLookup companion returns bit-identical hits and reasonable distances on
fresh Haar targets.

Run with ``pytest u_net/test_u_net_builder.py -v``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_UNIFIED = _HERE.parent
for p in (_HERE, _UNIFIED, _UNIFIED / "hrsa", _UNIFIED / "rz_db"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from haar_sampler import haar_su3                                  # noqa: E402
from u_net_builder import build_u_net, live_synthesize_rz          # noqa: E402
from u_net_lookup import UNetLookup, _matrix_to_real18             # noqa: E402


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

RZ_DB_PATH = "/tmp/rz_test.sqlite"
E0_NET_PATH = "/tmp/e0_net_5184.txt"


def _skip_if_missing_prereqs():
    """Skip if the pre-baked R_z DB or e0_net dump aren't available."""
    if not Path(RZ_DB_PATH).exists():
        pytest.skip(
            f"prereq missing: build the R_z DB first: "
            f"python rz_db/build_rz_db.py --db {RZ_DB_PATH} "
            f"sweep_zeta9_cal_2026-05-22/summary.csv "
            f"sweep_hrsa_grid_2026-05-22/summary.csv"
        )
    if not Path(E0_NET_PATH).exists():
        pytest.skip(
            f"prereq missing: build the e0_net dump first: "
            f"cd hrsa && make e0_net_dump && ./e0_net_dump {E0_NET_PATH}"
        )


@pytest.fixture(scope="module")
def tier0_net(tmp_path_factory):
    """Build a tiny tier-0 (eps_u=0.5) U-net for shared use by the tests below."""
    _skip_if_missing_prereqs()
    out_dir = tmp_path_factory.mktemp("u_net_tier0")
    out_h5 = out_dir / "unet_tier0.h5"
    meta = build_u_net(
        n_samples=50,
        eps_u=0.5,
        rz_db_path=RZ_DB_PATH,
        output_h5=str(out_h5),
        seed=20260523,
        dedup_tol=0.0,             # do not reject samples at tier-0
        allow_live_fallback=False, # offline; we want DB-only coverage
        log_miss_rate=True,
    )
    # Account for the .npz fallback when h5py is missing.
    if not out_h5.exists():
        out_h5 = out_h5.with_suffix(".npz")
    return out_h5, meta


# ---------------------------------------------------------------------------
#  Embedding sanity
# ---------------------------------------------------------------------------

def test_matrix_to_real18_preserves_frob():
    """The 18-dim real embedding's Euclidean norm equals the Frobenius norm."""
    rng = np.random.default_rng(0)
    M = (rng.standard_normal((5, 3, 3)) + 1j * rng.standard_normal((5, 3, 3))) \
        / np.sqrt(2.0)
    emb = _matrix_to_real18(M)
    assert emb.shape == (5, 18)
    # ||M||_F = sqrt(sum |m_ij|^2) = sqrt(sum re^2 + im^2) = ||emb||_2
    frob = np.sqrt(np.einsum("kij,kij->k", M.conj(), M).real)
    euc = np.sqrt(np.einsum("ki,ki->k", emb, emb))
    np.testing.assert_allclose(euc, frob, rtol=1e-12)


def test_matrix_to_real18_pairwise_dist_matches_frob():
    """Pairwise Euclidean distance in the embedding = pairwise Frobenius."""
    rng = np.random.default_rng(1)
    M = (rng.standard_normal((4, 3, 3)) + 1j * rng.standard_normal((4, 3, 3)))
    emb = _matrix_to_real18(M)
    # pairwise frob
    diff = M[:, None] - M[None, :]
    frob_d = np.sqrt(np.einsum("ijab,ijab->ij", diff.conj(), diff).real)
    diff_emb = emb[:, None, :] - emb[None, :, :]
    euc_d = np.sqrt(np.einsum("ijk,ijk->ij", diff_emb, diff_emb))
    np.testing.assert_allclose(euc_d, frob_d, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
#  Builder smoke
# ---------------------------------------------------------------------------

def test_build_tier0_persists_some_targets(tier0_net):
    """Building tier-0 against the existing R_z DB should persist > 0 targets."""
    out_h5, meta = tier0_net
    assert out_h5.exists(), f"U-net file not written to {out_h5}"
    assert meta["n_persisted"] > 0, (
        f"tier-0 build persisted 0/{meta['n_samples_requested']} targets; "
        f"R_z DB may be under-covered for eps_u=0.5. meta={meta}"
    )
    # Should be able to find the JSON meta companion next to the data file.
    meta_path = out_h5.with_suffix(".json")
    assert meta_path.exists(), f"meta JSON missing at {meta_path}"
    on_disk = json.loads(meta_path.read_text())
    assert on_disk["n_persisted"] == meta["n_persisted"]
    assert on_disk["eps_u"] == meta["eps_u"]
    # Miss rate must be reported (regardless of value).
    assert "miss_rate" in meta
    assert 0.0 <= meta["miss_rate"] <= 1.0


def test_build_emits_within_slack_only(tier0_net):
    """Every persisted target must have achieved_frob <= 1.5 * eps_u."""
    out_h5, meta = tier0_net
    look = UNetLookup(str(out_h5))
    eps_u = meta["eps_u"]
    assert (look.achieved_frob <= 1.5 * eps_u + 1e-12).all(), (
        f"some persisted targets exceed the 1.5*eps_u slack: "
        f"max={look.achieved_frob.max():.4f} vs {1.5 * eps_u:.4f}"
    )


def test_build_offline_reports_miss_rate(tier0_net):
    """With ``allow_live_fallback=False`` the build must still report a sensible
    miss rate (DB-miss count over total lookups) — per the PHASE_D_TODO
    contract that miss_rate is instrumented per ε-tier."""
    _out_h5, meta = tier0_net
    assert meta["n_lookups"] > 0, "no DB lookups attempted; build was a no-op"
    assert 0.0 <= meta["miss_rate"] <= 1.0
    # In the tier-0 (eps_u=0.5) build against an existing DB, n_lookups should
    # roughly equal n_z_leaves * n_samples (~ 50 * 57 = 2850).  The exact
    # miss count is data-dependent but should be < 100% of the lookup count
    # (otherwise no target would have been persisted at all).
    assert meta["n_misses"] < meta["n_lookups"], (
        f"every DB lookup missed: n_misses={meta['n_misses']} "
        f"vs n_lookups={meta['n_lookups']}; DB does not cover any leaf"
    )


# ---------------------------------------------------------------------------
#  Lookup smoke
# ---------------------------------------------------------------------------

def test_lookup_loads_and_reports_stats(tier0_net):
    out_h5, meta = tier0_net
    look = UNetLookup(str(out_h5))
    s = look.stats()
    assert s["N"] == meta["n_persisted"]
    assert s["eps_u"] == meta["eps_u"]
    for key in ("achieved_frob_p50", "achieved_frob_p95",
                "N_D_p50", "N_D_p95", "N_D_mean"):
        assert key in s


def test_lookup_self_query_distance_is_zero(tier0_net):
    """Querying a stored target must return distance < 1e-10."""
    out_h5, _meta = tier0_net
    look = UNetLookup(str(out_h5))
    if look.targets.shape[0] == 0:
        pytest.skip("empty U-net")
    # Pick the first stored target verbatim.
    target = look.targets[0]
    res = look.closest(target)
    assert res["idx"] == 0
    assert res["distance"] < 1e-10, (
        f"self-query distance {res['distance']:.3e} > 1e-10; "
        f"BallTree round-trip or embedding broken"
    )


def test_lookup_self_query_all_entries(tier0_net):
    """Every stored target must round-trip via closest() with distance < 1e-10."""
    out_h5, _meta = tier0_net
    look = UNetLookup(str(out_h5))
    if look.targets.shape[0] == 0:
        pytest.skip("empty U-net")
    # Use batched query for speed.
    out = look.closest_batch(look.targets)
    assert (out["distance"] < 1e-10).all(), (
        f"some self-queries exceed 1e-10: "
        f"max distance = {out['distance'].max():.3e}"
    )
    # The nearest neighbour to each stored target should be itself.
    np.testing.assert_array_equal(out["idx"], np.arange(look.targets.shape[0]))


def test_lookup_fresh_haar_distance_bounded(tier0_net):
    """Coverage sanity: 100 fresh Haar targets all sit within a loose bound."""
    out_h5, _meta = tier0_net
    look = UNetLookup(str(out_h5))
    if look.targets.shape[0] == 0:
        pytest.skip("empty U-net")
    samples = haar_su3(100, seed=99)
    out = look.closest_batch(samples)
    # For 50 Haar samples in 8-dim SU(3), nearest-neighbour distance is large
    # (~1.5 on average).  We just want a soft sanity bound below the SU(3)
    # diameter (2*sqrt(3) ≈ 3.46).
    assert out["distance"].max() <= 3.5, (
        f"max NN distance {out['distance'].max():.3f} exceeds SU(3) diameter"
    )
    assert out["distance"].min() >= 0.0


# ---------------------------------------------------------------------------
#  Live-fallback API surface (does not actually call HRSA/zeta9 unless we ask)
# ---------------------------------------------------------------------------

def test_live_synthesize_rz_rejects_bad_method():
    with pytest.raises(ValueError):
        live_synthesize_rz(0.5, 0.1, method="bogus", timeout=1)


def test_live_synthesize_rz_returns_none_when_binary_missing(monkeypatch):
    """If HRSA_TESTER is replaced with a non-existent path, the call must
    return None instead of raising."""
    import u_net_builder as ub
    bad_path = Path("/tmp/__definitely_not_HRSA_tester__")
    monkeypatch.setattr(ub, "HRSA_TESTER", bad_path)
    res = live_synthesize_rz(0.5, 0.1, method="hrsa", timeout=2)
    assert res is None


if __name__ == "__main__":
    # Quick CLI invocation for the standalone build smoke test.
    sys.exit(pytest.main([__file__, "-v"]))
