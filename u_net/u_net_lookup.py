"""u_net_lookup.py — BallTree-backed nearest-neighbour query for a U-net file.

Companion to :mod:`u_net_builder`.  Loads an HDF5 (or fallback NPZ) U-net
file produced by ``build_u_net`` and serves nearest-target queries against
its embedded SU(3) matrices.

We embed each 3×3 complex matrix as an 18-dim real vector (concatenated
``real`` + ``imag`` flattened entries) for the sklearn BallTree, which lives
in flat Euclidean space.  Euclidean distance in this embedding equals the
Frobenius distance of the original complex matrices, since
``||A - B||_F^2 = sum |a_ij - b_ij|^2 = sum (Re d_ij)^2 + sum (Im d_ij)^2``.

The BallTree is persisted next to the H5 file as ``<h5_path>.balltree.pkl``
and mtime-invalidated on rebuild.

Public API
----------
class UNetLookup:
    closest(target) -> dict
    stats() -> dict
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import h5py  # type: ignore
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

try:
    from sklearn.neighbors import BallTree
    _HAVE_BALLTREE = True
except ImportError:
    _HAVE_BALLTREE = False


def _matrix_to_real18(M: np.ndarray) -> np.ndarray:
    """Embed a (..., 3, 3) complex array as a (..., 18) real array.

    The flatten order is C-contiguous (row-major); the real part fills entries
    [0:9] and the imaginary part fills [9:18].
    """
    M = np.asarray(M)
    if M.ndim < 2 or M.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) array, got shape {M.shape}")
    flat = M.reshape(M.shape[:-2] + (9,))
    return np.concatenate([flat.real, flat.imag], axis=-1)


def _load_h5_or_npz(path: Path) -> dict:
    """Read a U-net file written by ``u_net_builder._write_h5``.

    Returns dict with arrays ``targets``, ``V_complex``, ``N_D``,
    ``achieved_frob``, ``n_leaves``.
    """
    if path.suffix == ".h5" and path.exists() and _HAVE_H5PY:
        with h5py.File(str(path), "r") as f:
            return {
                "targets": np.array(f["targets"]),
                "V_complex": np.array(f["V_complex"]),
                "N_D": np.array(f["N_D"]),
                "achieved_frob": np.array(f["achieved_frob"]),
                "n_leaves": np.array(f["n_leaves"]),
            }
    # Fall back to NPZ.  The builder writes either suffix; try both.
    npz_path = path.with_suffix(".npz")
    if npz_path.exists():
        data = np.load(str(npz_path))
        return {
            "targets": data["targets"],
            "V_complex": data["V_complex"],
            "N_D": data["N_D"],
            "achieved_frob": data["achieved_frob"],
            "n_leaves": data["n_leaves"],
        }
    # Last attempt: maybe the caller passed `.h5` but only NPZ exists, or
    # passed `.npz` but only H5 exists.
    h5_path = path.with_suffix(".h5")
    if h5_path.exists() and _HAVE_H5PY:
        with h5py.File(str(h5_path), "r") as f:
            return {
                "targets": np.array(f["targets"]),
                "V_complex": np.array(f["V_complex"]),
                "N_D": np.array(f["N_D"]),
                "achieved_frob": np.array(f["achieved_frob"]),
                "n_leaves": np.array(f["n_leaves"]),
            }
    raise FileNotFoundError(
        f"could not find U-net data at {path} (tried .h5 and .npz suffixes)"
    )


def _load_meta(h5_path: Path) -> dict:
    """Look up the JSON meta companion."""
    meta_path = h5_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    # Some on-disk fallbacks may use .npz; the meta is keyed off the original
    # h5 stem either way.  Return an empty stub so .stats() still works.
    return {}


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

class UNetLookup:
    """In-memory U-net with BallTree-accelerated nearest-target queries.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 produced by ``u_net_builder.build_u_net`` (or NPZ
        fallback if h5py was unavailable at build time).

    Attributes
    ----------
    targets       : (N, 3, 3) complex128
    V_complex     : (N, 3, 3) complex128
    N_D           : (N,) int32
    achieved_frob : (N,) float64
    n_leaves      : (N,) int32
    meta          : dict (loaded from the .json companion)
    """

    def __init__(self, h5_path: str):
        self.h5_path = Path(h5_path).resolve()
        data = _load_h5_or_npz(self.h5_path)
        self.targets: np.ndarray = data["targets"]
        self.V_complex: np.ndarray = data["V_complex"]
        self.N_D: np.ndarray = data["N_D"]
        self.achieved_frob: np.ndarray = data["achieved_frob"]
        self.n_leaves: np.ndarray = data["n_leaves"]
        self.meta = _load_meta(self.h5_path)
        self._tree: Optional[object] = None
        self._real_embedding: Optional[np.ndarray] = None
        if self.targets.shape[0] > 0:
            self._build_or_load_tree()

    # ----- BallTree management ----------------------------------------------
    def _tree_path(self) -> Path:
        return self.h5_path.with_suffix(self.h5_path.suffix + ".balltree.pkl")

    def _build_or_load_tree(self) -> None:
        """Build / cache an sklearn BallTree on the 18-dim real embedding."""
        embedding = _matrix_to_real18(self.targets)
        self._real_embedding = embedding

        if not _HAVE_BALLTREE:
            # No sklearn; fall back to linear scan in :meth:`closest`.
            self._tree = None
            return

        cache = self._tree_path()
        try:
            if cache.exists():
                # Invalidate the pickle if the source data is newer.
                src_mtime = self.h5_path.stat().st_mtime if self.h5_path.exists() \
                    else 0.0
                if cache.stat().st_mtime >= src_mtime:
                    with open(cache, "rb") as fh:
                        cached = pickle.load(fh)
                    # Sanity-check the cached tree size matches our data.
                    if isinstance(cached, dict) and cached.get("n") == \
                            int(self.targets.shape[0]):
                        self._tree = cached["tree"]
                        return
        except (OSError, pickle.PickleError, EOFError):
            # Corrupt cache — rebuild and overwrite.
            pass

        tree = BallTree(embedding, leaf_size=40, metric="euclidean")
        self._tree = tree
        try:
            with open(cache, "wb") as fh:
                pickle.dump({"tree": tree, "n": int(self.targets.shape[0])}, fh)
        except OSError as exc:
            print(f"[u_net_lookup] failed to cache BallTree at {cache}: {exc!r}",
                  file=sys.stderr)

    # ----- queries -----------------------------------------------------------
    def closest(self, target: np.ndarray) -> dict:
        """Return the nearest U-net entry to ``target``.

        ``target`` is a (3, 3) complex matrix.  Returns dict with keys::

            idx           : int   index into the U-net arrays
            V_complex     : (3,3) complex128  the synthesised approximation
            N_D           : int   total D-gate count of the chosen leaf product
            achieved_frob : float U-net leaf's |V - target_stored|_F
            distance      : float exact Frobenius distance to the stored target
        """
        if self.targets.shape[0] == 0:
            raise RuntimeError("U-net is empty; cannot perform closest() lookup")
        target = np.asarray(target, dtype=np.complex128)
        if target.shape != (3, 3):
            raise ValueError(f"target must be (3,3), got {target.shape}")

        q = _matrix_to_real18(target).reshape(1, -1)
        if self._tree is not None:
            dist_arr, idx_arr = self._tree.query(q, k=1)
            idx = int(idx_arr[0, 0])
        else:
            # Linear-scan fallback.
            assert self._real_embedding is not None
            diffs = self._real_embedding - q
            sq = np.einsum("ij,ij->i", diffs, diffs)
            idx = int(np.argmin(sq))

        # Recompute Frobenius distance EXACTLY on the original complex matrices
        # to avoid 1e-15 drift from the BallTree's internal float32 conversions.
        diff = self.targets[idx] - target
        dist = float(np.sqrt((diff.conj() * diff).real.sum()))

        return {
            "idx": idx,
            "V_complex": self.V_complex[idx].copy(),
            "N_D": int(self.N_D[idx]),
            "achieved_frob": float(self.achieved_frob[idx]),
            "distance": dist,
        }

    def closest_batch(self, targets: np.ndarray) -> dict:
        """Vectorised :meth:`closest` over many targets.

        ``targets`` is (M, 3, 3) complex.  Returns dict with arrays
        ``idx`` (M,), ``distance`` (M,) — caller can index back into
        ``self.V_complex`` / ``self.N_D`` as needed.
        """
        if self.targets.shape[0] == 0:
            raise RuntimeError("U-net is empty; cannot perform closest_batch()")
        targets = np.asarray(targets, dtype=np.complex128)
        if targets.ndim != 3 or targets.shape[1:] != (3, 3):
            raise ValueError(f"targets must be (M, 3, 3), got {targets.shape}")

        q = _matrix_to_real18(targets)
        if self._tree is not None:
            dist_arr, idx_arr = self._tree.query(q, k=1)
            idxs = idx_arr[:, 0].astype(np.int64)
        else:
            assert self._real_embedding is not None
            # Chunked argmin to avoid (M, N) blowup.
            M = targets.shape[0]
            idxs = np.empty(M, dtype=np.int64)
            chunk = max(1, min(M, 256))
            for start in range(0, M, chunk):
                end = min(M, start + chunk)
                diffs = self._real_embedding[None, :, :] - q[start:end, None, :]
                sq = np.einsum("mij,mij->mi", diffs, diffs)
                idxs[start:end] = np.argmin(sq, axis=1)

        diffs_c = self.targets[idxs] - targets
        dists = np.sqrt(np.einsum("kij,kij->k", diffs_c.conj(), diffs_c).real)
        return {
            "idx": idxs,
            "distance": dists,
        }

    # ----- stats -------------------------------------------------------------
    def stats(self) -> dict:
        """Summary statistics for this U-net.

        Returns
        -------
        dict with keys::

            N                 : int   number of persisted targets
            eps_u             : float (or None) the U-net's nominal coverage radius
            achieved_frob_p50 : float median |V - target|_F across entries
            achieved_frob_p95 : float p95 of the same
            N_D_p50           : int   median N_D across entries
            N_D_p95           : int   p95 of N_D
            N_D_mean          : float mean N_D
        """
        N = int(self.targets.shape[0])
        eps_u = self.meta.get("eps_u")
        out: dict = {
            "N": N,
            "eps_u": float(eps_u) if eps_u is not None else None,
        }
        if N == 0:
            out.update({
                "achieved_frob_p50": 0.0,
                "achieved_frob_p95": 0.0,
                "N_D_p50": 0,
                "N_D_p95": 0,
                "N_D_mean": 0.0,
            })
        else:
            out["achieved_frob_p50"] = float(np.median(self.achieved_frob))
            out["achieved_frob_p95"] = float(np.quantile(self.achieved_frob, 0.95))
            out["N_D_p50"] = int(np.median(self.N_D))
            out["N_D_p95"] = int(np.quantile(self.N_D, 0.95))
            out["N_D_mean"] = float(self.N_D.mean())
        return out
