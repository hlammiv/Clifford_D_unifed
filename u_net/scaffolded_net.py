"""scaffolded_net.py — Multi-tier U-net wrapper for Solovay-Kitaev recursion.

Phase E of the SK U-net bootstrap.  A :class:`ScaffoldedNet` loads a set of
per-tier ``UNetLookup`` instances (each at a different coverage radius
``eps_u``) and dispatches queries to the loosest tier that meets the caller's
precision budget.  The driver in :mod:`hrsa.sk_driver_scaffolded` uses this to
pick the right U-net per SK recursion depth and, when possible, skip
intermediate levels by jumping straight to a denser tier.

Tier-selection rule
-------------------
A tier with coverage ``eps_u`` is *usable* for target precision
``target_eps`` iff

    eps_u <= target_eps * COVERAGE_SLACK

where ``COVERAGE_SLACK = 0.5`` gives SK the contraction headroom it needs to
land inside the basin after one commutator level.  Among usable tiers we
pick the LOOSEST (largest ``eps_u``) — cheaper net, fewer N_D — and let SK
contraction tighten the residual.  If no tier qualifies, we fall back to the
densest available tier and mark the query ``coverage_ok=False`` so the
driver knows the base case is below spec and recursion is required.

Public API
----------
class ScaffoldedNet:
    __init__(tier_h5_paths)
    pick_tier(target_eps)        -> (eps_u, UNetLookup)
    closest(target, target_eps)  -> dict
    all_tiers()                  -> list[float]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from u_net_lookup import UNetLookup  # noqa: E402


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Slack factor: a tier is usable for target_eps iff eps_u <= target_eps * SLACK.
# 0.5 gives a 2× safety margin so one SK contraction level suffices.
COVERAGE_SLACK = 0.5


# ---------------------------------------------------------------------------
#  ScaffoldedNet
# ---------------------------------------------------------------------------

class ScaffoldedNet:
    """Stack of :class:`UNetLookup` instances ordered coarse-to-fine.

    Parameters
    ----------
    tier_h5_paths : list[str]
        One HDF5 (or NPZ fallback) per tier produced by
        ``u_net.u_net_builder.build_u_net``.  Order is irrelevant; we sort by
        ``eps_u`` descending (coarsest first) at load time.

    Attributes
    ----------
    tiers : list[tuple[float, UNetLookup]]
        Coarse-to-fine list of (eps_u, lookup) pairs.  ``eps_u`` is the
        tier's nominal coverage radius as reported by ``UNetLookup.stats()``.
    """

    def __init__(self, tier_h5_paths: list[str]):
        if not tier_h5_paths:
            raise ValueError("ScaffoldedNet requires at least one tier H5 path")

        # Load each tier and pair it with its nominal eps_u.
        unsorted: list[tuple[Optional[float], UNetLookup, str]] = []
        for path in tier_h5_paths:
            lookup = UNetLookup(str(path))
            stats = lookup.stats()
            eps_u = stats.get("eps_u")
            unsorted.append((eps_u, lookup, str(path)))

        # Validate every tier reports an eps_u (the builder writes it into the
        # JSON meta companion; missing means the file is malformed).
        bad = [p for (e, _, p) in unsorted if e is None]
        if bad:
            raise ValueError(
                f"ScaffoldedNet: the following tiers are missing eps_u in "
                f"their JSON meta companion: {bad}"
            )

        # Sort by eps_u descending — coarsest tier first.  ``pick_tier`` then
        # walks coarse→fine and returns the first qualifying tier.
        sorted_tiers = sorted(unsorted, key=lambda t: -float(t[0]))  # type: ignore
        self.tiers: list[tuple[float, UNetLookup]] = [
            (float(e), look) for (e, look, _) in sorted_tiers  # type: ignore
        ]
        # Stash source paths for diagnostics.
        self._tier_paths: list[str] = [p for (_, _, p) in sorted_tiers]

    # ----- introspection ----------------------------------------------------
    def all_tiers(self) -> list[float]:
        """Return the sorted (coarse-first) list of tier ``eps_u`` values."""
        return [eps for (eps, _) in self.tiers]

    def __len__(self) -> int:
        return len(self.tiers)

    def __repr__(self) -> str:
        return (f"ScaffoldedNet(tiers={self.all_tiers()}, "
                f"slack={COVERAGE_SLACK})")

    # ----- tier dispatch ----------------------------------------------------
    def pick_tier(self, target_eps: float) -> tuple[float, UNetLookup]:
        """Return ``(eps_u, lookup)`` of the loosest tier matching ``target_eps``.

        A tier qualifies iff ``eps_u <= target_eps * COVERAGE_SLACK``.  Among
        qualifying tiers we pick the LOOSEST so the net is as cheap as
        possible (denser tiers carry larger N_D per entry).

        If no tier qualifies — i.e. even the densest tier is too coarse to
        meet ``target_eps * slack`` — we issue a ``UserWarning`` and return
        the densest tier (smallest ``eps_u``) anyway.  The companion
        :meth:`closest` records this case via the ``coverage_ok`` flag.
        """
        if target_eps <= 0:
            raise ValueError(f"target_eps must be > 0, got {target_eps}")

        # tiers is coarse→fine; walk it and pick the first qualifier.
        qualifying = [(eps, look) for (eps, look) in self.tiers
                      if eps <= target_eps * COVERAGE_SLACK]
        if qualifying:
            # tiers are sorted coarse→fine, so qualifying[0] is the loosest
            # qualifier — exactly what we want.
            return qualifying[0]

        # Nothing qualifies; return the densest tier (last entry).
        densest_eps, densest_lookup = self.tiers[-1]
        warnings.warn(
            f"ScaffoldedNet.pick_tier: target_eps={target_eps:g} is tighter "
            f"than any available tier (densest eps_u={densest_eps:g}); "
            f"falling back to densest tier.",
            stacklevel=2,
        )
        return densest_eps, densest_lookup

    # ----- nearest-neighbour query -----------------------------------------
    def closest(self, target: np.ndarray, target_eps: float) -> dict:
        """Tier-aware nearest-neighbour query.

        Parameters
        ----------
        target : (3, 3) complex unitary.
        target_eps : caller's precision budget.  Used to select the tier;
            does NOT bound the returned distance (the recursion logic in
            ``sk_driver_scaffolded`` does that).

        Returns
        -------
        dict with the keys from :meth:`UNetLookup.closest` plus::

            tier_eps_u  : float — eps_u of the tier this query was routed to.
            coverage_ok : bool  — False iff ``pick_tier`` had to fall back
                          to a tier coarser than ``target_eps * 0.5``.
            tier_index  : int   — 0 is coarsest.
        """
        if target_eps <= 0:
            raise ValueError(f"target_eps must be > 0, got {target_eps}")

        # Capture warnings raised by ``pick_tier`` so we can report
        # ``coverage_ok`` without leaking a warning per call (the driver may
        # invoke ``closest`` thousands of times in a single SK descent).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            tier_eps, lookup = self.pick_tier(target_eps)
            coverage_ok = not any(
                issubclass(w.category, UserWarning) and "pick_tier" in str(w.message)
                for w in caught
            )

        result = lookup.closest(target)
        # Recover the tier index for diagnostics.
        tier_index = next(i for i, (e, _l) in enumerate(self.tiers)
                          if e == tier_eps and _l is lookup)
        result["tier_eps_u"] = float(tier_eps)
        result["tier_index"] = int(tier_index)
        result["coverage_ok"] = bool(coverage_ok)
        return result


__all__ = ["ScaffoldedNet", "COVERAGE_SLACK"]
