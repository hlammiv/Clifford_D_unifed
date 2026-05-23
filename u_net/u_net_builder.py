"""u_net_builder.py — Build a U-net at coverage radius ε_U from Haar SU(3) samples.

Phase D of the SK U-net bootstrap.  For each Haar-sampled target U:

1. Skip if within ``dedup_tol`` of an already-built entry.
2. ``su3_euler.euler_decompose(U)`` produces a factor sequence of
   ``('Z', 0, 1, theta)`` / ``('CLIFFORD', idx)`` / ``('SIGN', sp)``.
3. Per Z-leaf, query the Phase-A ``RzLookupDB`` at ``eps_leaf = eps_u / (2 * n_leaves)``.
   If the leaf misses and ``allow_live_fallback=True``, dispatch a live HRSA
   (eps>=1e-3) or zeta9 (eps<1e-3) synthesis and insert the result back into
   the DB (lazy population per ``rz_db/PHASE_D_TODO.md``).
4. Per ``('CLIFFORD', idx)`` factor, use ``su3_euler._load_clifford_set()[idx]``.
5. Per ``('SIGN', sp)`` factor, build ``diag(±1, ±1, ±1)`` from bit-encoded ``sp``.
6. Multiply all leaves left-to-right to get ``V_total`` (complex 3×3).
7. ``achieved_frob = |V_total - U|_F`` (after global-phase removal).
8. Drop the target if ``achieved_frob > 1.5 * eps_u`` (slack failure).
9. Persist ``(target, V_total, N_D_sum, achieved_frob, n_leaves)`` to HDF5.

The HDF5 file is a single dataset bundle; an adjacent ``.json`` carries
build metadata.

If ``h5py`` is unavailable we fall back to ``np.savez_compressed``; the
:class:`UNetLookup` companion handles both formats.

Public API
----------
build_u_net(...) -> dict
live_synthesize_rz(theta, eps_target, method='auto', timeout=120) -> dict | None
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Wire u_net + rz_db + hrsa into sys.path so this module is runnable from anywhere.
_HERE = Path(__file__).resolve().parent
_UNIFIED = _HERE.parent
for p in (_UNIFIED, _HERE, _UNIFIED / "hrsa", _UNIFIED / "rz_db"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rz_db.rz_lookup import RzLookupDB, THETA_TO_FROB, V_SHAPE  # noqa: E402
from haar_sampler import haar_su3                                # noqa: E402

# h5py is preferred but optional.
try:
    import h5py  # type: ignore
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False


# ---------------------------------------------------------------------------
#  Constants — paths + cost model
# ---------------------------------------------------------------------------

DEFAULT_E0_NET = "/tmp/e0_net_5184.txt"
HRSA_TESTER = _UNIFIED / "hrsa" / "HRSA_tester"
ZETA9_COMPILE = _UNIFIED / "zeta9_compile.py"

# eps threshold separating HRSA (eps >= 1e-3) from zeta9 (eps < 1e-3).
# Matches the dispatch in sk_leaves.synthesize_leaf (rule 3 vs rule 4).
EPS_HRSA_FLOOR = 1e-3

# Default per-leaf precision: split the U-net budget eps_u equally across the
# n_leaves Z-rotations the Euler decomposition emits, with a safety factor of 2.
# n_leaves itself depends on the target's distance from identity; we recompute
# the eps_leaf budget AFTER decomposing each target so the splits are tight.


# ---------------------------------------------------------------------------
#  Helpers — Euler factor reification
# ---------------------------------------------------------------------------

_CLIFFORD_SET_CACHE: Optional[np.ndarray] = None


def _get_clifford_set(e0_net_path: str = DEFAULT_E0_NET) -> np.ndarray:
    """Lazy-load the 648-element pure Clifford set used by ``su3_euler``.

    We delegate to ``su3_euler._load_clifford_set`` (which polar-orthogonalises
    the dump for exact unitarity).  This requires ``e0_net_path`` to exist;
    if it doesn't the caller must build it via ``e0_net_dump``.
    """
    global _CLIFFORD_SET_CACHE
    if _CLIFFORD_SET_CACHE is not None:
        return _CLIFFORD_SET_CACHE
    # su3_euler hard-codes /tmp/e0_net_5184.txt; we cannot pass an alternative
    # path through its public surface, so we just verify the file exists and
    # call it.  If a non-default path was requested but the file is at the
    # default location anyway, we accept it (e0_net is a static asset).
    expected = Path(DEFAULT_E0_NET)
    if not expected.exists():
        raise FileNotFoundError(
            f"e0_net dump missing at {expected}; build with "
            f"`cd hrsa && make e0_net_dump && ./e0_net_dump {expected}`."
        )
    if str(e0_net_path) != str(expected):
        # Soft warning — su3_euler will read /tmp anyway.
        print(
            f"[u_net_builder] note: e0_net_path={e0_net_path!r} but su3_euler "
            f"hard-codes {expected}; using the latter.",
            file=sys.stderr,
        )
    from su3_euler import _load_clifford_set  # type: ignore
    _CLIFFORD_SET_CACHE = _load_clifford_set()
    return _CLIFFORD_SET_CACHE


def _v_blob_to_complex(V_blob: np.ndarray, v_f: int) -> np.ndarray:
    """Convert a (3,3,6) int64 ringZ9 numerator + denom-exponent f to a (3,3) complex."""
    if V_blob.shape != V_SHAPE:
        raise ValueError(f"V_blob shape {V_blob.shape} != {V_SHAPE}")
    zeta9 = np.exp(2j * np.pi / 9)
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            s = 0.0 + 0.0j
            for k in range(6):
                a = int(V_blob[i, j, k])
                if a:
                    s += a * (zeta9 ** k)
            out[i, j] = s / (3 ** int(v_f))
    return out


def _sign_factor_matrix(sp: int) -> np.ndarray:
    """Build the diag(±1, ±1, ±1) matrix from the bit-encoded sign pattern ``sp``."""
    M = np.eye(3, dtype=np.complex128)
    for k in range(3):
        if (sp >> k) & 1:
            M[k, k] = -1.0
    return M


def _build_RZ(theta: float) -> np.ndarray:
    """R^Z_{(0,1)}(theta) = diag(e^{-i theta/2}, e^{+i theta/2}, 1)."""
    M = np.eye(3, dtype=np.complex128)
    M[0, 0] = np.exp(-0.5j * theta)
    M[1, 1] = np.exp(+0.5j * theta)
    return M


def _strip_global_phase(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Multiply M by the global phase that best aligns it with U.

    For unitaries M, U: pick phi maximising Re Tr(U^† e^{i phi} M);
    equivalently phi = -arg(Tr(U^† M)).
    """
    inner = np.trace(U.conj().T @ M)
    if abs(inner) < 1e-14:
        return M
    return M * (inner.conjugate() / abs(inner))


# ---------------------------------------------------------------------------
#  Live fallback — HRSA + zeta9 subprocess dispatchers
# ---------------------------------------------------------------------------

def _live_hrsa(theta: float, eps: float, timeout: int) -> Optional[dict]:
    """Run hrsa/HRSA_tester for a single (theta, eps) pair and parse its JSON.

    Returns a result dict {V (3,3,6 int64), v_f, achieved_frob, N_D, method}
    or None on subprocess failure / no solution / timeout.
    """
    if not HRSA_TESTER.exists() or not os.access(HRSA_TESTER, os.X_OK):
        return None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        json_path = Path(tf.name)
    try:
        cmd = [
            str(HRSA_TESTER),
            f"{theta:.10f}",
            f"{eps:g}",
            "8",  # max_f
            "--json", str(json_path),
            # max-solns 20: invoke HRSA_bestD (best-of-K within first successful
            # f-level by min D-count). Default 1 gives loose first-hit semantics
            # — see memory hrsa_first_hit_bias.md.
            "--max-solns", "20",
            "--no-lookahead",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if not json_path.exists():
            return None
        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
    finally:
        try:
            json_path.unlink()
        except OSError:
            pass

    uni = data.get("unitary")
    if uni is None:
        return None
    try:
        V = np.array(uni["V"], dtype=np.int64)
    except (KeyError, TypeError, ValueError):
        return None
    if V.shape != V_SHAPE:
        return None
    v_f = uni.get("f")
    if v_f is None:
        return None

    achieved = data.get("achieved") or {}
    decomp = data.get("decomposition") or {}
    achieved_frob = achieved.get("achieved_frob")
    if achieved_frob is None:
        return None
    N_D = decomp.get("N_D")
    try:
        N_D = int(N_D) if N_D is not None and int(N_D) >= 0 else None
    except (TypeError, ValueError):
        N_D = None

    return {
        "V": V,
        "v_f": int(v_f),
        "achieved_frob": float(achieved_frob),
        "N_D": N_D,
        "method": str(achieved.get("method") or "HRSA"),
    }


def _live_zeta9(theta: float, eps: float, timeout: int) -> Optional[dict]:
    """Run unified/zeta9_compile.py and parse its --json output."""
    if not ZETA9_COMPILE.exists():
        return None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        json_path = Path(tf.name)
    try:
        cmd = [
            sys.executable, str(ZETA9_COMPILE),
            "--theta", f"{theta:.10f}",
            "--epsilon", f"{eps:g}",
            "--max-f", "6",
            "--json", str(json_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if not json_path.exists():
            return None
        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
    finally:
        try:
            json_path.unlink()
        except OSError:
            pass

    uni = data.get("unitary")
    if uni is None:
        return None
    try:
        V = np.array(uni["V"], dtype=np.int64)
    except (KeyError, TypeError, ValueError):
        return None
    if V.shape != V_SHAPE:
        return None
    v_f = uni.get("f")
    if v_f is None:
        return None

    achieved = data.get("achieved") or {}
    decomp = data.get("decomposition") or {}
    achieved_frob = achieved.get("achieved_frob")
    if achieved_frob is None:
        return None
    N_D = decomp.get("N_D")
    try:
        N_D = int(N_D) if N_D is not None and int(N_D) >= 0 else None
    except (TypeError, ValueError):
        N_D = None

    return {
        "V": V,
        "v_f": int(v_f),
        "achieved_frob": float(achieved_frob),
        "N_D": N_D,
        "method": str(achieved.get("method") or "zeta9"),
    }


def live_synthesize_rz(theta: float, eps_target: float,
                       method: str = "auto", timeout: int = 180) -> Optional[dict]:
    """Live HRSA / zeta9 dispatch for a single ``R^Z_{(0,1)}(theta)`` target.

    Parameters
    ----------
    theta : rotation angle (radians).
    eps_target : leaf precision budget.
    method : 'auto' (default) routes by eps_target threshold, 'hrsa' / 'zeta9'
        force one backend.
    timeout : subprocess wall-clock cap in seconds.

    Returns
    -------
    dict with keys (V, v_f, achieved_frob, N_D, method) or None on failure.
    The caller is responsible for inserting the result into the R_z DB.
    """
    if method == "auto":
        method = "hrsa" if eps_target >= EPS_HRSA_FLOOR else "zeta9"

    if method == "hrsa":
        return _live_hrsa(theta, eps_target, timeout)
    if method == "zeta9":
        return _live_zeta9(theta, eps_target, timeout)
    raise ValueError(f"unknown method {method!r}; expected 'auto' / 'hrsa' / 'zeta9'")


# ---------------------------------------------------------------------------
#  Leaf resolution — DB lookup with lazy fallback
# ---------------------------------------------------------------------------

def _theta_canonical(theta: float) -> tuple[float, bool]:
    """Fold theta into [0, pi/2] using R_z symmetries.

    The R_z DB stores only positive theta in roughly [0, pi/2].  Two clean
    symmetries let us reduce a broader angle range into that window:

      * ``R_z(θ + 2π) = -R_z(θ)`` (global phase factor -1, irrelevant for our
        SU(3) net since we strip global phase from V_total).
      * ``R_z(-θ) = R_z(θ)^†`` (so a stored approximation V(|θ|) gives V^†
        as the approximation for R_z(-θ)).

    Returns ``(theta_abs, daggered)`` where ``theta_abs ∈ [0, π]`` and
    ``daggered`` indicates whether the caller should conjugate-transpose
    the looked-up V.

    NOTE: this does NOT reduce |theta| > π/2 into [0, π/2]; that requires an
    extra free-Clifford absorption (R_z(π) = diag(i, -i, 1) * global phase,
    a Clifford up to phase).  We leave that to a future iteration; leaves
    with |theta| > π/2 will miss the existing DB and rely on live fallback
    or be dropped.
    """
    # Wrap into (-π, π].
    t = (theta + math.pi) % (2.0 * math.pi) - math.pi
    if t < 0:
        return -t, True
    return t, False


def _resolve_z_leaf(theta: float, eps_leaf: float,
                    db: RzLookupDB,
                    allow_live_fallback: bool,
                    timeout: int) -> Optional[dict]:
    """Resolve a single ``('Z', 0, 1, theta)`` factor against the R_z DB.

    Returns dict {V (3,3 complex), N_D, method, source, achieved_frob, hit_or_miss}
    or None if no synthesis is available and allow_live_fallback is False
    (or the live fallback also fails).
    """
    theta_abs, daggered = _theta_canonical(theta)
    res = db.lookup(theta_abs, eps_leaf)
    if res is not None:
        V_complex = _v_blob_to_complex(res["V"], res["v_f"])
        if daggered:
            V_complex = V_complex.conj().T
        return {
            "V": V_complex,
            "N_D": res["N_D"],
            "method": res["method"],
            "source": res["source"],
            "achieved_frob": res["achieved_frob"],
            "hit_or_miss": "hit",
        }

    if not allow_live_fallback:
        return None

    # Live fallback: pick HRSA / zeta9 by eps.  Use theta_abs (positive)
    # so the inserted entry serves future symmetry-folded queries too.
    live = live_synthesize_rz(theta_abs, eps_leaf, method="auto", timeout=timeout)
    if live is None:
        return None

    # Insert into the DB so the next builder run sees it (lazy population).
    try:
        db.insert(
            theta=float(theta_abs),
            eps_target=float(eps_leaf),
            V=live["V"],
            v_f=int(live["v_f"]),
            achieved_frob=float(live["achieved_frob"]),
            N_D=live["N_D"],
            method=str(live["method"]),
            source="live_fallback",
        )
        db.commit()
    except Exception as exc:
        print(f"[u_net_builder] DB insert failed for theta={theta_abs:.6f} "
              f"eps={eps_leaf:.3g}: {exc!r}", file=sys.stderr)

    V_complex = _v_blob_to_complex(live["V"], int(live["v_f"]))
    if daggered:
        V_complex = V_complex.conj().T
    return {
        "V": V_complex,
        "N_D": live["N_D"],
        "method": str(live["method"]),
        "source": "live_fallback",
        "achieved_frob": float(live["achieved_frob"]),
        "hit_or_miss": "miss",
    }


def _reify_factors(factors: list,
                   eps_u: float,
                   db: RzLookupDB,
                   cliffords: np.ndarray,
                   allow_live_fallback: bool,
                   timeout: int) -> dict:
    """Walk an Euler factor list and produce the (V_total, N_D_sum, n_leaves) tuple.

    Returns dict ``{V_total (or None on full failure), N_D_sum, n_leaves,
    n_db_misses, n_live_recovered, n_unresolved}``.  ``V_total is None`` iff
    any leaf could not be resolved either from DB or via live fallback.

    ``n_db_misses`` counts EVERY DB miss (whether or not the live fallback
    rescued it) so the meta miss-rate matches the PHASE_D_TODO contract.
    """
    # Count the Z leaves so we can split the eps_u budget tightly.
    n_z_leaves = sum(1 for f in factors if f[0] == "Z")
    if n_z_leaves == 0:
        # No Z leaves — pure Clifford / sign / identity target.
        V_total = np.eye(3, dtype=np.complex128)
        for f in factors:
            if f[0] == "CLIFFORD":
                V_total = V_total @ cliffords[int(f[1])]
            elif f[0] == "SIGN":
                V_total = V_total @ _sign_factor_matrix(int(f[1]))
        return {"V_total": V_total, "N_D_sum": 0, "n_leaves": 0,
                "n_db_misses": 0, "n_live_recovered": 0, "n_unresolved": 0}

    # Per-leaf eps budget.  L2 estimate: total error ≈ eps_leaf · √n_leaves
    # for random orientations (Phase D's eps_leaf=eps_u design assumed cancellation
    # that doesn't materialize on arbitrary Haar — verified empirically 2026-05-23,
    # see memory `tier0_slack_design_dead_2026-05-23`).  Solving for eps_leaf:
    #   eps_leaf = eps_u / √n_leaves
    # gives total ≈ eps_u, well inside the 1.5·eps_u post-product slack ceiling.
    eps_leaf = max(float(eps_u) / max(1.0, float(n_z_leaves) ** 0.5), 1e-12)
    V_total = np.eye(3, dtype=np.complex128)
    N_D_sum = 0
    n_db_misses = 0
    n_live_recovered = 0
    n_unresolved = 0
    fully_failed = False
    for f in factors:
        kind = f[0]
        if kind == "Z":
            _, i, j, theta = f
            # Phase B established every Z-leaf is (0,1); refuse anything else.
            if (i, j) != (0, 1):
                fully_failed = True
                n_unresolved += 1
                continue
            # We need a sub-call that distinguishes DB-hit vs DB-miss-but-live-recovered
            # vs total-failure.  Do the DB lookup inline so we can update counters.
            theta_abs, daggered = _theta_canonical(theta)
            res = db.lookup(theta_abs, eps_leaf)
            if res is not None:
                V = _v_blob_to_complex(res["V"], res["v_f"])
                if daggered:
                    V = V.conj().T
                V_total = V_total @ V
                if res["N_D"] is not None:
                    N_D_sum += int(res["N_D"])
                continue

            # DB miss.
            n_db_misses += 1
            if not allow_live_fallback:
                fully_failed = True
                n_unresolved += 1
                continue
            live = live_synthesize_rz(theta_abs, eps_leaf, method="auto",
                                      timeout=timeout)
            if live is None:
                fully_failed = True
                n_unresolved += 1
                continue
            # Insert into DB for future runs.
            try:
                db.insert(
                    theta=float(theta_abs),
                    eps_target=float(eps_leaf),
                    V=live["V"],
                    v_f=int(live["v_f"]),
                    achieved_frob=float(live["achieved_frob"]),
                    N_D=live["N_D"],
                    method=str(live["method"]),
                    source="live_fallback",
                )
                db.commit()
            except Exception as exc:
                print(f"[u_net_builder] DB insert failed for theta={theta_abs:.6f} "
                      f"eps={eps_leaf:.3g}: {exc!r}", file=sys.stderr)
            V = _v_blob_to_complex(live["V"], int(live["v_f"]))
            if daggered:
                V = V.conj().T
            V_total = V_total @ V
            if live["N_D"] is not None:
                N_D_sum += int(live["N_D"])
            n_live_recovered += 1
        elif kind == "CLIFFORD":
            V_total = V_total @ cliffords[int(f[1])]
        elif kind == "SIGN":
            V_total = V_total @ _sign_factor_matrix(int(f[1]))
        else:
            raise ValueError(f"unknown factor kind {kind!r}")

    return {
        "V_total": None if fully_failed else V_total,
        "N_D_sum": N_D_sum,
        "n_leaves": n_z_leaves,
        "n_db_misses": n_db_misses,
        "n_live_recovered": n_live_recovered,
        "n_unresolved": n_unresolved,
    }


# ---------------------------------------------------------------------------
#  Dedup helper — Frobenius distance vs accepted set
# ---------------------------------------------------------------------------

def _frob_dist_to_set(target: np.ndarray, accepted: np.ndarray, n: int) -> float:
    if n == 0:
        return float("inf")
    diff = accepted[:n] - target[None, :, :]
    sq = np.einsum("kij,kij->k", diff.conj(), diff).real
    return float(np.sqrt(sq).min())


# ---------------------------------------------------------------------------
#  HDF5 / NPZ persistence helpers
# ---------------------------------------------------------------------------

def _write_h5(path: Path,
              targets: np.ndarray,
              V_complex: np.ndarray,
              N_D: np.ndarray,
              achieved_frob: np.ndarray,
              n_leaves: np.ndarray) -> None:
    if _HAVE_H5PY:
        with h5py.File(str(path), "w") as f:
            f.create_dataset("targets", data=targets, compression="gzip")
            f.create_dataset("V_complex", data=V_complex, compression="gzip")
            f.create_dataset("N_D", data=N_D, compression="gzip")
            f.create_dataset("achieved_frob", data=achieved_frob, compression="gzip")
            f.create_dataset("n_leaves", data=n_leaves, compression="gzip")
    else:
        npz_path = path.with_suffix(".npz")
        np.savez_compressed(npz_path,
                            targets=targets,
                            V_complex=V_complex,
                            N_D=N_D,
                            achieved_frob=achieved_frob,
                            n_leaves=n_leaves)
        print(f"[u_net_builder] h5py unavailable; wrote NPZ to {npz_path}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def build_u_net(
    n_samples: int,
    eps_u: float,
    rz_db_path: str,
    output_h5: str,
    seed: int = 42,
    dedup_tol: Optional[float] = None,
    e0_net_path: str = DEFAULT_E0_NET,
    allow_live_fallback: bool = True,
    log_miss_rate: bool = True,
    live_timeout_s: int = 120,
) -> dict:
    """Build a U-net at coverage radius ``eps_u`` from ``n_samples`` Haar SU(3) draws.

    Parameters
    ----------
    n_samples : int
        Number of Haar SU(3) targets to attempt.
    eps_u : float
        U-net coverage radius (Frobenius distance budget per target).
    rz_db_path : str
        Path to the SQLite R_z lookup DB.
    output_h5 : str
        Path to the output HDF5 (or ``.npz`` if h5py is unavailable).
    seed : int
        RNG seed for the Haar sampler.
    dedup_tol : float or None
        Reject a target if any earlier accepted target is within this Frobenius
        distance.  Defaults to ``eps_u / 4``.
    e0_net_path : str
        Path to the 5184-element sign-extended Clifford dump (used by
        ``su3_euler`` for Clifford reification).
    allow_live_fallback : bool
        If True, on a DB miss dispatch HRSA / zeta9 and insert the result.
        If False, drop targets whose leaves miss the DB.
    log_miss_rate : bool
        If True, print a one-line miss-rate summary on stderr at end.
    live_timeout_s : int
        Wall-clock cap (seconds) for each live HRSA/zeta9 subprocess.

    Returns
    -------
    Meta dict (also persisted as ``<output_h5>.json``) with keys::

        eps_u, n_samples_requested, n_persisted, miss_rate, dedup_tol,
        build_time_s, rz_db_path, rz_db_source_count, n_lookups,
        n_misses, n_dropped_dedup, n_dropped_unresolved, n_dropped_slack,
        N_D_total, N_D_mean, achieved_frob_p50, achieved_frob_p95
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if eps_u <= 0:
        raise ValueError(f"eps_u must be positive, got {eps_u}")
    if dedup_tol is None:
        dedup_tol = float(eps_u) / 4.0

    output_h5_p = Path(output_h5)
    output_h5_p.parent.mkdir(parents=True, exist_ok=True)
    meta_json_p = output_h5_p.with_suffix(".json")

    t0 = time.time()

    # Lazy imports to defer heavy module loads until build is actually invoked.
    from su3_euler import euler_decompose  # type: ignore

    cliffords = _get_clifford_set(e0_net_path)
    db = RzLookupDB(rz_db_path)
    rz_source_count = db.count()

    # Sample n_samples Haar SU(3) targets up front.  We reject in-loop on
    # dedup / leaf-failure, so the persisted count may be less than n_samples.
    haar = haar_su3(n_samples, seed=seed)

    persisted_targets = np.empty((n_samples, 3, 3), dtype=np.complex128)
    persisted_V = np.empty((n_samples, 3, 3), dtype=np.complex128)
    persisted_N_D = np.zeros(n_samples, dtype=np.int32)
    persisted_frob = np.zeros(n_samples, dtype=np.float64)
    persisted_n_leaves = np.zeros(n_samples, dtype=np.int32)

    n_persisted = 0
    n_lookups = 0
    n_misses = 0
    n_dropped_dedup = 0
    n_dropped_unresolved = 0
    n_dropped_slack = 0

    for trial in range(n_samples):
        target = haar[trial]

        # 1. Dedup against accepted set.
        if dedup_tol > 0 and n_persisted > 0:
            d = _frob_dist_to_set(target, persisted_targets, n_persisted)
            if d < dedup_tol:
                n_dropped_dedup += 1
                continue

        # 2. Euler-decompose.
        factors = euler_decompose(target)

        # 3-6. Reify factors against the R_z DB (with optional lazy fallback).
        reified = _reify_factors(factors, eps_u, db, cliffords,
                                 allow_live_fallback=allow_live_fallback,
                                 timeout=live_timeout_s)
        # Count lookup attempts (per Z-leaf) regardless of resolution success.
        n_z_attempted = sum(1 for f in factors if f[0] == "Z")
        n_lookups += n_z_attempted
        n_misses += int(reified["n_db_misses"])
        if reified["V_total"] is None:
            n_dropped_unresolved += 1
            continue

        # 7. Slack failure check.
        V_total = reified["V_total"]
        V_aligned = _strip_global_phase(V_total, target)
        achieved_frob = float(np.linalg.norm(V_aligned - target, ord="fro"))
        if achieved_frob > 1.5 * eps_u:
            n_dropped_slack += 1
            continue

        # 8. Persist.
        persisted_targets[n_persisted] = target
        persisted_V[n_persisted] = V_aligned
        persisted_N_D[n_persisted] = int(reified["N_D_sum"])
        persisted_frob[n_persisted] = achieved_frob
        persisted_n_leaves[n_persisted] = int(reified["n_leaves"])
        n_persisted += 1

    build_time_s = time.time() - t0
    miss_rate = (float(n_misses) / n_lookups) if n_lookups > 0 else 0.0

    # Trim and write.
    targets_out = persisted_targets[:n_persisted].copy()
    V_out = persisted_V[:n_persisted].copy()
    N_D_out = persisted_N_D[:n_persisted].copy()
    frob_out = persisted_frob[:n_persisted].copy()
    n_leaves_out = persisted_n_leaves[:n_persisted].copy()

    if n_persisted == 0:
        # Avoid h5py "empty dataset" gotchas — emit a stub file with shape (0,...).
        targets_out = np.zeros((0, 3, 3), dtype=np.complex128)
        V_out = np.zeros((0, 3, 3), dtype=np.complex128)
        N_D_out = np.zeros((0,), dtype=np.int32)
        frob_out = np.zeros((0,), dtype=np.float64)
        n_leaves_out = np.zeros((0,), dtype=np.int32)

    _write_h5(output_h5_p, targets_out, V_out, N_D_out, frob_out, n_leaves_out)

    N_D_total = int(N_D_out.sum()) if n_persisted > 0 else 0
    N_D_mean = float(N_D_out.mean()) if n_persisted > 0 else 0.0
    frob_p50 = float(np.median(frob_out)) if n_persisted > 0 else 0.0
    frob_p95 = float(np.quantile(frob_out, 0.95)) if n_persisted > 0 else 0.0

    meta = {
        "eps_u": float(eps_u),
        "n_samples_requested": int(n_samples),
        "n_persisted": int(n_persisted),
        "miss_rate": float(miss_rate),
        "dedup_tol": float(dedup_tol),
        "build_time_s": float(build_time_s),
        "rz_db_path": str(rz_db_path),
        "rz_db_source_count": int(rz_source_count),
        "rz_db_post_count": int(db.count()),
        "n_lookups": int(n_lookups),
        "n_misses": int(n_misses),
        "n_dropped_dedup": int(n_dropped_dedup),
        "n_dropped_unresolved": int(n_dropped_unresolved),
        "n_dropped_slack": int(n_dropped_slack),
        "N_D_total": int(N_D_total),
        "N_D_mean": float(N_D_mean),
        "achieved_frob_p50": float(frob_p50),
        "achieved_frob_p95": float(frob_p95),
        "seed": int(seed),
        "allow_live_fallback": bool(allow_live_fallback),
        "h5_format": "h5py" if _HAVE_H5PY else "npz",
        "output_path": str(output_h5_p),
    }

    with open(meta_json_p, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    if log_miss_rate:
        print(
            f"[u_net_builder] eps_u={eps_u:g}: {n_lookups} lookups, "
            f"{n_misses} misses ({100.0 * miss_rate:.1f}%); "
            f"persisted {n_persisted}/{n_samples} targets in {build_time_s:.1f}s.",
            file=sys.stderr,
        )

    db.close()
    return meta
