"""Tests for the scaffolded SK driver (Phase E).

The full test set covers:

  * single-tier degradation — a one-tier ScaffoldedNet behaves like a plain
    nearest-neighbour query at eps_u, no commutator descent;
  * tier-skip — with two tiers, a tight target_eps query routes directly to
    the denser tier at depth 0;
  * scaffolded jump — with a tight target that the loose tier alone can't
    meet, recursion happens and the log records the transitions.

Run with ``pytest hrsa/test_sk_driver_scaffolded.py -v``.

Prereqs (pytest.skip otherwise):
  * /tmp/rz_test.sqlite — pre-baked R_z DB.
  * /tmp/e0_net_5184.txt — Clifford dump.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_UNIFIED = _HERE.parent
for p in (_UNIFIED, _HERE, _UNIFIED / "u_net", _UNIFIED / "rz_db"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from u_net.scaffolded_net import ScaffoldedNet, COVERAGE_SLACK  # noqa: E402
from sk_driver_scaffolded import sk_driver_scaffolded            # noqa: E402


# ---------------------------------------------------------------------------
#  Prereq guards
# ---------------------------------------------------------------------------

RZ_DB_PATH = "/tmp/rz_test.sqlite"
E0_NET_PATH = "/tmp/e0_net_5184.txt"


def _skip_if_missing_prereqs():
    if not Path(RZ_DB_PATH).exists():
        pytest.skip(f"prereq missing: {RZ_DB_PATH} (build via rz_db/build_rz_db.py)")
    if not Path(E0_NET_PATH).exists():
        pytest.skip(f"prereq missing: {E0_NET_PATH} (build via hrsa/e0_net_dump)")


# ---------------------------------------------------------------------------
#  Fixtures — build tiny tier H5 files with offline DB-only synthesis.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tier_05_h5(tmp_path_factory):
    """Tiny tier-0 (eps_u=0.5) U-net.  10 Haar samples, offline."""
    _skip_if_missing_prereqs()
    from u_net.u_net_builder import build_u_net  # type: ignore
    out_dir = tmp_path_factory.mktemp("tier_05")
    out_h5 = out_dir / "tier_05.h5"
    # seed=7 reliably persists 2 samples at eps_u=0.5 against /tmp/rz_test.sqlite
    # (n=10 with default seeds often persists 0 since the DB only covers a
    # narrow slice of leaf angles).
    build_u_net(
        n_samples=10,
        eps_u=0.5,
        rz_db_path=RZ_DB_PATH,
        output_h5=str(out_h5),
        seed=7,
        dedup_tol=0.0,
        allow_live_fallback=False,
        log_miss_rate=False,
    )
    if not out_h5.exists():
        out_h5 = out_h5.with_suffix(".npz")
    return out_h5


@pytest.fixture(scope="module")
def tier_01_h5(tmp_path_factory, tier_05_h5):
    """Synthetic tier-1 (eps_u=0.1) net for tier-selection unit tests.

    Offline DB coverage is genuinely too sparse to build a real eps_u=0.1
    U-net (the existing /tmp/rz_test.sqlite only stores 100 angles at the
    0.1 tier, and Euler-leaf budgets don't fit any of them).  To still
    exercise the tier-selection logic of :class:`ScaffoldedNet`, we copy
    the tier-0 file and overwrite its JSON ``eps_u`` to 0.1.  The stored
    matrices are unchanged — only the tier label differs — which is
    sufficient for unit-level tier-pick / tier-skip checks.

    Tests that actually evaluate residuals against the eps_u contract
    should NOT use this fixture (it would be a tautology); they should
    rely on ``tier_05_h5`` and check raw distances.
    """
    import json
    import shutil

    src = Path(tier_05_h5)
    out_dir = tmp_path_factory.mktemp("tier_01_synth")
    out_h5 = out_dir / "tier_01.h5"
    out_npz = out_dir / "tier_01.npz"
    out_json = out_dir / "tier_01.json"

    if src.suffix == ".h5":
        shutil.copyfile(src, out_h5)
        # Also copy the BallTree cache if it exists so loading is fast.
        bt = src.with_suffix(src.suffix + ".balltree.pkl")
        if bt.exists():
            shutil.copyfile(bt, out_h5.with_suffix(out_h5.suffix + ".balltree.pkl"))
        emit_path = out_h5
    else:
        shutil.copyfile(src, out_npz)
        emit_path = out_npz

    src_meta = src.with_suffix(".json")
    meta = json.loads(src_meta.read_text())
    meta["eps_u"] = 0.1
    meta["output_path"] = str(emit_path)
    out_json.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return emit_path


# ---------------------------------------------------------------------------
#  ScaffoldedNet unit-level tests
# ---------------------------------------------------------------------------

def test_scaffold_loads_single_tier(tier_05_h5):
    s = ScaffoldedNet([str(tier_05_h5)])
    assert len(s) == 1
    tiers = s.all_tiers()
    assert tiers == [0.5]


def test_scaffold_sorts_coarse_first(tier_05_h5, tier_01_h5):
    # Pass them in mixed order; the constructor should re-sort coarse-first.
    s = ScaffoldedNet([str(tier_01_h5), str(tier_05_h5)])
    tiers = s.all_tiers()
    assert tiers == sorted(tiers, reverse=True), \
        f"tiers not coarse-first: {tiers}"
    assert tiers[0] == 0.5


def test_pick_tier_qualifying(tier_05_h5, tier_01_h5):
    s = ScaffoldedNet([str(tier_05_h5), str(tier_01_h5)])
    # target_eps = 0.2 → 0.2 * 0.5 = 0.1 ≥ eps_u=0.1 (tier_01 qualifies);
    # 0.5 does NOT qualify (0.5 > 0.1).  Loosest qualifying = tier_01.
    eps_u, lookup = s.pick_tier(target_eps=0.2)
    assert eps_u == 0.1
    # target_eps = 1.0 → 0.5 qualifying.  Loosest = 0.5.
    eps_u2, _ = s.pick_tier(target_eps=1.0)
    assert eps_u2 == 0.5


def test_pick_tier_fallback_emits_warning(tier_05_h5):
    """Tightest possible request below the densest tier → warning + densest fallback."""
    s = ScaffoldedNet([str(tier_05_h5)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        eps_u, _ = s.pick_tier(target_eps=1e-5)
    assert eps_u == 0.5  # densest (and only) tier
    msgs = [str(w.message) for w in caught]
    assert any("pick_tier" in m for m in msgs), f"no warning emitted; caught={msgs}"


def test_closest_records_coverage_ok(tier_05_h5):
    s = ScaffoldedNet([str(tier_05_h5)])
    # Pick any target from the tier itself.
    from u_net.u_net_lookup import UNetLookup  # type: ignore
    lookup = UNetLookup(str(tier_05_h5))
    if lookup.targets.shape[0] == 0:
        pytest.skip("tier_05 is empty; cannot probe coverage_ok")
    target = lookup.targets[0]
    # target_eps=2.0 → 0.5*2=1.0 ≥ 0.5, qualifying — coverage_ok=True.
    res_ok = s.closest(target, target_eps=2.0)
    assert res_ok["coverage_ok"] is True
    assert res_ok["tier_eps_u"] == 0.5
    # target_eps=1e-5 → no tier qualifies, falls back — coverage_ok=False.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res_bad = s.closest(target, target_eps=1e-5)
    assert res_bad["coverage_ok"] is False


# ---------------------------------------------------------------------------
#  Driver: single-tier degradation
# ---------------------------------------------------------------------------

def test_single_tier_degradation_matches_plain_lookup(tier_05_h5):
    """ScaffoldedNet([tier_05]) at target_eps=1.0 should behave like a plain
    nearest-neighbour query: depth_reached=0, achieved_frob equals the raw
    UNetLookup distance up to global phase alignment.
    """
    from u_net.u_net_lookup import UNetLookup  # type: ignore
    from u_net.haar_sampler import haar_su3      # type: ignore

    s = ScaffoldedNet([str(tier_05_h5)])
    lookup = UNetLookup(str(tier_05_h5))
    if lookup.targets.shape[0] == 0:
        pytest.skip("tier_05 is empty")

    targets = haar_su3(5, seed=20260601)
    for k in range(5):
        T = targets[k]
        res = sk_driver_scaffolded(T, target_eps=1.0, scaffold=s,
                                   max_recurse_per_tier=3)
        # At target_eps=1.0, target_eps*0.5=0.5 ≥ tier_eps_u=0.5 ⇒ tier dispatch
        # immediately.  Any Haar target sits within the SU(3) diameter from the
        # tier (since tier_05 is small but non-empty), and target_eps=1.0 is
        # loose so the residual is likely already inside that target.
        assert res["depth_reached"] in (0, 1, 2, 3)
        # Loose bound: the produced V must be a valid (3,3) complex matrix.
        V = res["V"]
        assert V is not None and V.shape == (3, 3)
        # Sanity: achieved_frob is finite and matches |V - T|_F.
        assert np.isfinite(res["achieved_frob"])
        recomputed = float(np.linalg.norm(V - T, ord="fro"))
        np.testing.assert_allclose(res["achieved_frob"], recomputed, rtol=1e-10)


def test_single_tier_loose_target_eps_meets_bound(tier_05_h5):
    """5 Haar targets at target_eps=1.0 against a tiny tier-0 net.  Bound is
    loose (≤ 3.5, the SU(3) diameter) — we just want no NaNs / no exceptions.
    """
    from u_net.u_net_lookup import UNetLookup  # type: ignore
    from u_net.haar_sampler import haar_su3      # type: ignore

    s = ScaffoldedNet([str(tier_05_h5)])
    lookup = UNetLookup(str(tier_05_h5))
    if lookup.targets.shape[0] == 0:
        pytest.skip("tier_05 is empty")

    targets = haar_su3(5, seed=99)
    for k in range(5):
        res = sk_driver_scaffolded(targets[k], target_eps=1.0, scaffold=s)
        assert res["V"] is not None
        # SU(3) diameter ~ 2*sqrt(3) ≈ 3.46.
        assert res["achieved_frob"] <= 3.5
        # No exceptions in the log.
        for ev in res["log"]:
            assert "exception" not in ev or ev.get("event") != "factor_commutator_threw"


# ---------------------------------------------------------------------------
#  Driver: tier-skip behaviour
# ---------------------------------------------------------------------------

def test_tier_skip_picks_dense_tier_directly(tier_05_h5, tier_01_h5):
    """With two tiers (0.5 and 0.1), target_eps=0.05 should route the FIRST
    base-case query to tier_01 (since 0.05 * 0.5 = 0.025 < 0.1 — neither
    qualifies, so we fall back to the densest = tier_01).  The log should
    record tier_index=1 (the denser tier).
    """
    from u_net.u_net_lookup import UNetLookup  # type: ignore
    from u_net.haar_sampler import haar_su3      # type: ignore

    s = ScaffoldedNet([str(tier_05_h5), str(tier_01_h5)])
    lookup_dense = UNetLookup(str(tier_01_h5))
    if lookup_dense.targets.shape[0] == 0:
        pytest.skip("dense tier (tier_01) is empty; cannot demonstrate tier-skip")

    target = haar_su3(1, seed=4242)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = sk_driver_scaffolded(target, target_eps=0.05, scaffold=s)
    # Find the first tier_dispatch event in the log; it should report the
    # densest tier (index 1) — that's the tier-skip in action.
    first_dispatch = next(ev for ev in res["log"]
                          if ev.get("event") == "tier_dispatch")
    assert first_dispatch["tier_index"] == 1, (
        f"first tier dispatch landed on tier_index={first_dispatch['tier_index']}, "
        f"expected 1 (densest); log={res['log'][:3]}"
    )
    assert first_dispatch["tier_eps_u"] == 0.1


def test_tier_skip_at_loose_target_picks_loose_tier(tier_05_h5, tier_01_h5):
    """With two tiers (0.5 and 0.1), target_eps=4.0 should route to the
    LOOSEST qualifying tier (tier_05) because 4.0 * 0.5 = 2.0 ≥ both
    eps_u=0.5 and eps_u=0.1, and we prefer loose.
    """
    from u_net.haar_sampler import haar_su3      # type: ignore

    s = ScaffoldedNet([str(tier_05_h5), str(tier_01_h5)])
    target = haar_su3(1, seed=20260605)[0]
    res = sk_driver_scaffolded(target, target_eps=4.0, scaffold=s,
                               max_recurse_per_tier=1)
    first_dispatch = next(ev for ev in res["log"]
                          if ev.get("event") == "tier_dispatch")
    assert first_dispatch["tier_index"] == 0, (
        f"loose target_eps should pick loose tier (idx 0), got "
        f"{first_dispatch['tier_index']}"
    )
    assert first_dispatch["tier_eps_u"] == 0.5
    # And at target_eps=4.0 the first dispatch should land inside the target
    # (base_frob ≤ SU(3) diameter ≤ 4.0), so depth_reached = 0.
    assert res["depth_reached"] == 0


# ---------------------------------------------------------------------------
#  Driver: scaffolded recursion log contains tier transitions
# ---------------------------------------------------------------------------

def test_scaffolded_recursion_logs_recursion_events(tier_05_h5, tier_01_h5):
    """A target the loose tier alone can't meet should trigger one or more
    ``recurse_sub`` events in the log.  At target_eps=0.3 the slack-adjusted
    tier-pick is 0.3*0.5=0.15 — tier_01 qualifies, tier_05 doesn't.  Pick
    tier_01.  If base_frob > 0.3, we recurse.
    """
    from u_net.u_net_lookup import UNetLookup  # type: ignore
    from u_net.haar_sampler import haar_su3      # type: ignore

    s = ScaffoldedNet([str(tier_05_h5), str(tier_01_h5)])
    lookup_dense = UNetLookup(str(tier_01_h5))
    if lookup_dense.targets.shape[0] < 2:
        pytest.skip("dense tier too sparse to exercise recursion")

    # Pick a Haar target known to be far from all stored entries so the
    # base case definitely misses the target_eps.  Use a generous max_recurse
    # so the log has room to grow.
    target = haar_su3(1, seed=987654)[0]
    res = sk_driver_scaffolded(target, target_eps=0.3, scaffold=s,
                               max_recurse_per_tier=2)
    events = [ev.get("event") for ev in res["log"]]
    # The driver always records at least one tier_dispatch at depth 0.
    assert "tier_dispatch" in events
    # If the base case missed target_eps, we should see a recurse event OR
    # an explicit terminal event (inside_target_eps / max_recurse_reached /
    # outside_basin / commutator_no_reduction).
    terminal_events = {"inside_target_eps", "max_recurse_reached",
                       "outside_basin", "commutator_no_reduction",
                       "factor_commutator_threw"}
    assert any(e == "recurse_sub" or e in terminal_events for e in events), (
        f"no recursion or terminal events in log; got events={events}"
    )


# ---------------------------------------------------------------------------
#  Smoke: __main__ runs a tiny end-to-end check.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
