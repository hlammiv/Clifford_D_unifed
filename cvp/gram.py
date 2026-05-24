"""gram.py — q-induced Gram matrix on Z[ζ_9] and its LLL reduction.

Phase 1 of the qutrit Babai-CVP plan. Provides:

* ``B`` — the 6x6 integer Gram matrix in the convention ``B = 2 G`` where
  ``G`` is the bilinear form such that ``quad(a) = a^T G a`` matches
  HRSA's ``ringZ9::quad()``. With this convention every entry of ``B`` is
  an integer (diagonal 2, off-diagonal -1 at indices (i, i+3) for
  i=0,1,2, zero elsewhere).
* ``q_form(coefs)`` — pure-Python/numpy quadratic form on a length-6
  integer coefficient vector. Returns a Python ``int`` (so it never
  overflows at the f=4050 SK scale). Bit-exactly matches HRSA's quad.
* ``lll_reduce()`` — runs fpylll LLL on the lattice defined by ``B`` and
  returns the LLL-reduced Gram matrix ``B_lll`` plus the unimodular
  ``U`` such that ``B_lll = U @ B @ U.T``.
* Constants ``B_lll`` and ``U_lll`` produced once at import-time and
  also serialized to ``gram_z9.npz`` for cross-process reuse.

Derivation
----------

The number field is ``Q(ζ_9)`` with minimal polynomial
``Φ_9(x) = x^6 + x^3 + 1``. The integral basis is ``{1, ζ, ζ^2, ..., ζ^5}``
(``ζ = ζ_9``). HRSA stores an element ``a ∈ Z[ζ_9]`` as a length-6 coef
vector ``element[0..5]`` and defines

    quad(a) = sum_i a_i^2 - a_0 a_3 - a_1 a_4 - a_2 a_5.        (1)

This is ``a^T G a`` with ``G_{ii} = 1``, ``G_{i, i+3} = G_{i+3, i} = -1/2``
for ``i = 0, 1, 2``, and zero elsewhere. Equivalently, ``B := 2 G`` has
all-integer entries (``+2`` on the diagonal, ``-1`` on the off-diagonal
pairs), and ``quad(a) = a^T B a / 2``.

The same matrix arises from the trace pairing
``B_{ij} = (1/6) Tr(ζ^i ζ^{-j})`` (using
``Tr(ζ^0) = 6``, ``Tr(ζ^{±3}) = -3``, ``Tr(ζ^{±k}) = 0`` for
``k ∈ {1, 2, 4}``), which confirms the two derivations agree.

Re-generate the cached npz with::

    python3 -m cvp.gram
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

try:
    from fpylll import IntegerMatrix, LLL
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "fpylll is required for cvp.gram. Install with `pip install fpylll`."
    ) from exc


# ---------------------------------------------------------------------------
# Gram matrix
# ---------------------------------------------------------------------------

def _build_B() -> np.ndarray:
    """Construct ``B = 2 G`` as a 6x6 int64 array.

    Diagonal 2, ``B[i, i+3] = B[i+3, i] = -1`` for i = 0, 1, 2.
    """
    B = 2 * np.eye(6, dtype=np.int64)
    for i in range(3):
        B[i, i + 3] = -1
        B[i + 3, i] = -1
    return B


B: np.ndarray = _build_B()
B.setflags(write=False)


def q_form(coefs: Sequence[int]) -> int:
    """Pure-Python quadratic form on length-6 integer coefficient vectors.

    Returns ``a^T G a == quad(a)`` matching HRSA's ``ringZ9::quad()``
    bit-for-bit. Computes via ``a^T B a / 2`` (integer division is exact
    since ``a^T B a`` is always even for integer ``a``).

    Uses pure Python ints throughout so arbitrary-size coefficients
    (f=4050 SK output reaches ~10^1934 per coordinate) do not overflow.
    """
    if len(coefs) != 6:
        raise ValueError(f"q_form expects length-6 vector, got {len(coefs)}")
    a = [int(x) for x in coefs]
    # quad(a) = sum a_i^2 - a_0 a_3 - a_1 a_4 - a_2 a_5
    s = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3] + a[4] * a[4] + a[5] * a[5]
    s -= a[0] * a[3] + a[1] * a[4] + a[2] * a[5]
    return s


# ---------------------------------------------------------------------------
# LLL reduction
# ---------------------------------------------------------------------------

# Scale factor used to embed the (irrational) Cholesky factor of G in Z.
# 10^10 is comfortable for 6 dimensions and well below int64 limits while
# preserving the relevant geometry.
_LLL_SCALE = 10 ** 10


def _cholesky_basis_integer(scale: int = _LLL_SCALE) -> np.ndarray:
    """Return a scaled integer approximation of the Cholesky factor of G.

    The lattice spanned by rows of ``round(scale * L)`` (where ``L L^T = G``)
    is geometrically the qutrit lattice scaled by ``scale``. LLL on this
    integer basis produces a unimodular ``U`` that simultaneously reduces
    ``B = 2G`` since the transformation is independent of the scale.
    """
    G = B.astype(np.float64) / 2.0
    L = np.linalg.cholesky(G)
    return np.round(L * scale).astype(np.int64)


def lll_reduce(scale: int = _LLL_SCALE) -> Tuple[np.ndarray, np.ndarray]:
    """LLL-reduce the qutrit lattice.

    Returns
    -------
    B_lll : ndarray, shape (6, 6), dtype int64
        Reduced Gram matrix, ``B_lll = U @ B @ U.T`` exactly (integer
        arithmetic — no floating-point round trip).
    U : ndarray, shape (6, 6), dtype int64
        Unimodular change-of-basis (``det(U) = ±1``). If ``c'`` are
        coordinates in the new basis and ``c`` in the original, then
        ``c = U.T @ c'`` (rows of U are the reduced basis expressed in
        the original basis).
    """
    Lint = _cholesky_basis_integer(scale)

    Bmat = IntegerMatrix.from_matrix(Lint.tolist())
    U_mat = IntegerMatrix.identity(6)
    LLL.reduction(Bmat, U=U_mat)

    U = np.array([[U_mat[i, j] for j in range(6)] for i in range(6)], dtype=np.int64)
    # B_lll computed exactly in integer arithmetic, no Cholesky round trip.
    B_lll = U @ B @ U.T
    return B_lll.astype(np.int64), U


B_lll, U_lll = lll_reduce()
B_lll.setflags(write=False)
U_lll.setflags(write=False)


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

_CACHE_PATH = Path(__file__).with_name("gram_z9.npz")


def _save_cache(path: Path = _CACHE_PATH) -> Path:
    """Persist ``B``, ``B_lll``, ``U_lll`` to a single .npz file."""
    np.savez(path, B=B, B_lll=B_lll, U_lll=U_lll)
    return path


def _load_cache(path: Path = _CACHE_PATH):
    """Return ``(B, B_lll, U_lll)`` from the cached npz, or ``None`` if
    the cache is missing."""
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["B"].copy(), data["B_lll"].copy(), data["U_lll"].copy()


# Build / refresh cache opportunistically at import time so downstream
# modules can pick it up via ``np.load(B_LLL_CACHE)`` without re-running
# fpylll.
if not _CACHE_PATH.exists():
    _save_cache()

B_LLL_CACHE: Path = _CACHE_PATH


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description="Build / verify the qutrit Z[ζ_9] Gram matrix + LLL cache."
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force regeneration of gram_z9.npz even if present.",
    )
    args = parser.parse_args()

    if args.regen or not _CACHE_PATH.exists():
        path = _save_cache()
        print(f"wrote {path}")
    else:
        print(f"cache already present at {_CACHE_PATH}")

    print("B (= 2 G):")
    print(B)
    print(f"det(B) = {int(round(np.linalg.det(B.astype(np.float64))))}")
    print()
    print("B_lll:")
    print(B_lll)
    print(f"det(U_lll) = {int(round(np.linalg.det(U_lll.astype(np.float64))))}")
    print(f"||B||_F     = {np.linalg.norm(B.astype(float), 'fro'):.6f}")
    print(f"||B_lll||_F = {np.linalg.norm(B_lll.astype(float), 'fro'):.6f}")


if __name__ == "__main__":
    _main()
