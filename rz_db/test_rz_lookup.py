"""Tests for rz_db.rz_lookup.RzLookupDB."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `python -m pytest rz_db/test_rz_lookup.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rz_db.rz_lookup import RzLookupDB, THETA_TO_FROB, V_SHAPE, _pack_V, _unpack_V  # noqa: E402


def _make_V(seed: int = 0) -> np.ndarray:
    """Deterministic (3,3,6) int matrix for tests."""
    rng = np.random.default_rng(seed)
    return rng.integers(-100, 100, size=V_SHAPE, dtype=np.int64)


def test_pack_unpack_roundtrip():
    V = _make_V(1)
    blob = _pack_V(V)
    V2 = _unpack_V(blob)
    assert np.array_equal(V, V2)
    assert V2.dtype == np.int64
    assert V2.shape == V_SHAPE


def test_pack_rejects_wrong_shape():
    with pytest.raises(ValueError):
        _pack_V(np.zeros((3, 3, 5), dtype=np.int64))


def test_db_roundtrip(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    V = _make_V(42)
    db.insert(
        theta=1.234,
        eps_target=0.1,
        V=V,
        v_f=4,
        achieved_frob=0.05,
        N_D=20,
        method="zeta9-householder",
        source="test",
    )
    db.commit()
    assert db.count() == 1
    res = db.lookup(1.234, 0.2)
    assert res is not None
    assert res["theta"] == pytest.approx(1.234)
    assert res["eps_target"] == pytest.approx(0.1)
    assert res["v_f"] == 4
    assert res["N_D"] == 20
    assert res["method"] == "zeta9-householder"
    assert res["source"] == "test"
    # V must be byte-identical (not just numerically equal).
    assert np.array_equal(res["V"], V)
    assert res["V"].dtype == np.int64
    db.close()


def test_nearest_theta_picks_closer(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    Va = _make_V(1)
    Vb = _make_V(2)
    db.insert(theta=1.0, eps_target=0.1, V=Va, v_f=2, achieved_frob=0.01,
              N_D=5, method="m", source="a")
    db.insert(theta=2.0, eps_target=0.1, V=Vb, v_f=2, achieved_frob=0.01,
              N_D=5, method="m", source="b")
    db.commit()

    r1 = db.lookup(1.4, 0.5)
    assert r1 is not None and r1["source"] == "a"

    r2 = db.lookup(1.6, 0.5)
    assert r2 is not None and r2["source"] == "b"

    # Exact tie at midpoint: both within budget; tie-break is N_D then budget
    # then achieved_frob. Equal N_D and budget -> equal achieved_frob -> order
    # falls back to whichever is iterated first; both are acceptable returns,
    # so we only assert one of them is returned.
    r3 = db.lookup(1.5, 0.5)
    assert r3 is not None and r3["source"] in ("a", "b")
    db.close()


def test_eps_budget_enforced(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    V = _make_V(7)
    db.insert(theta=1.0, eps_target=0.05, V=V, v_f=2, achieved_frob=0.05,
              N_D=10, method="m", source="solo")
    db.commit()

    # Budget at theta=1.0001 is |0.0001|*0.5 + 0.05 = 0.05005
    # eps_max=0.06 -> fits, returns entry
    r = db.lookup(1.0001, 0.06)
    assert r is not None
    assert r["source"] == "solo"
    assert r["budget"] == pytest.approx(0.0001 * THETA_TO_FROB + 0.05, abs=1e-12)

    # eps_max=0.04 -> 0.05005 > 0.04 -> no entry
    r2 = db.lookup(1.0001, 0.04)
    assert r2 is None

    # eps_max=0.05 -> eps_target tier 0.05 has slack 0; only theta=1.0 itself fits
    r3 = db.lookup(1.0, 0.05)
    assert r3 is not None
    db.close()


def test_eps_max_below_eps_target_returns_none(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    V = _make_V(7)
    db.insert(theta=1.0, eps_target=0.1, V=V, v_f=2, achieved_frob=0.0,
              N_D=10, method="m", source="solo")
    db.commit()
    # eps_target > eps_max -> tier ineligible
    assert db.lookup(1.0, 0.05) is None
    db.close()


def test_prefer_smaller_N_D(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    V1 = _make_V(11)
    V2 = _make_V(22)
    # Two entries that both fit the query, but at different eps tiers with
    # different N_D. The looser eps tier has smaller N_D and should win.
    db.insert(theta=1.0, eps_target=0.1, V=V1, v_f=1, achieved_frob=0.05,
              N_D=5, method="m", source="loose")
    db.insert(theta=1.0, eps_target=0.01, V=V2, v_f=3, achieved_frob=0.005,
              N_D=30, method="m", source="tight")
    db.commit()
    r = db.lookup(1.0, 0.2)
    assert r is not None
    assert r["source"] == "loose"  # smaller N_D wins
    assert r["N_D"] == 5
    db.close()


def test_idempotent_insert(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    V = _make_V(5)
    V2 = _make_V(6)
    db.insert(theta=1.0, eps_target=0.1, V=V, v_f=2, achieved_frob=0.05,
              N_D=10, method="m", source="first")
    db.insert(theta=1.0, eps_target=0.1, V=V2, v_f=3, achieved_frob=0.04,
              N_D=8, method="m", source="second")
    db.commit()
    assert db.count() == 1  # upsert
    r = db.lookup(1.0, 0.2)
    assert r is not None
    assert r["source"] == "second"
    assert r["N_D"] == 8
    assert np.array_equal(r["V"], V2)
    db.close()


def test_count_and_stats(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    V = _make_V(0)
    # 6 rows in eps=0.1 tier, 4 in eps=0.01 tier.
    for i in range(6):
        db.insert(theta=0.1 * (i + 1), eps_target=0.1, V=V, v_f=1,
                  achieved_frob=0.05, N_D=10, method="m", source="s1")
    for i in range(4):
        db.insert(theta=0.5 * (i + 1), eps_target=0.01, V=V, v_f=3,
                  achieved_frob=0.005, N_D=20, method="m", source="s2")
    db.commit()

    assert db.count() == 10
    stats = db.stats()
    assert stats["total"] == 10
    assert stats["eps_tiers"][0.1]["n"] == 6
    assert stats["eps_tiers"][0.01]["n"] == 4
    # max gap in eps=0.1 tier is 0.1; in eps=0.01 tier is 0.5
    assert stats["eps_tiers"][0.1]["theta_max_gap"] == pytest.approx(0.1)
    assert stats["eps_tiers"][0.01]["theta_max_gap"] == pytest.approx(0.5)
    assert stats["by_source"]["s1"] == 6
    assert stats["by_source"]["s2"] == 4
    db.close()


def test_lookup_on_empty_db(tmp_path):
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    assert db.count() == 0
    assert db.lookup(1.0, 0.5) is None
    db.close()


def test_cache_invalidated_after_insert(tmp_path):
    """Insert AFTER a lookup must still surface in the next lookup."""
    db = RzLookupDB(str(tmp_path / "rz.sqlite"))
    V = _make_V(0)
    db.insert(theta=1.0, eps_target=0.1, V=V, v_f=1, achieved_frob=0.0,
              N_D=1, method="m", source="first")
    db.commit()
    # populate cache
    r1 = db.lookup(1.0, 0.5)
    assert r1 is not None
    # insert a new closer entry
    db.insert(theta=1.5, eps_target=0.1, V=V, v_f=1, achieved_frob=0.0,
              N_D=1, method="m", source="second")
    db.commit()
    r2 = db.lookup(1.4, 0.5)
    assert r2 is not None
    assert r2["source"] == "second"
    db.close()


def test_context_manager(tmp_path):
    path = tmp_path / "rz.sqlite"
    V = _make_V(0)
    with RzLookupDB(str(path)) as db:
        db.insert(theta=1.0, eps_target=0.1, V=V, v_f=1, achieved_frob=0.0,
                  N_D=1, method="m", source="ctx")
    # Reopen and verify persistence (commit happened in __exit__).
    db2 = RzLookupDB(str(path))
    assert db2.count() == 1
    db2.close()
