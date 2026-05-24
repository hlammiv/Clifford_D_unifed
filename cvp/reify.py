"""reify.py — Phase 4 of the qutrit Babai-CVP plan.

Given a (x_1, x_2, x_3, f) triple produced by Phase 3 (Householder norm
equation q(x_1) + q(x_2) + q(x_3) = 2·3^{2f} satisfied exactly in the ring),
construct the target unitary

    V = X_(0,1) · H,   H = I_3 - u u^†,   u = (x_1, x_2, x_3) / 3^f

so that V approximates ``R_(0,1)^Z(θ) = diag(e^{-iθ/2}, e^{iθ/2}, 1)`` in
Frobenius norm. This matches HRSA ``buildUnitary`` (decompose_impl.h:68 and
householder_search.cpp:545) bit-for-bit, in particular::

    H[i][j] = δ_ij - u_i · conj(u_j)        (column-vector outer product)
    V       = SWAP_(0,1) · H                 (Clifford row swap, theta-free)

Each entry of H lives in Z[ζ_9, 1/3^{2f}] (the constant 1 on the diagonal
contributes ``3^{2f}`` after clearing denominators; the off-diagonal terms
contribute ``x_i · x̄_j`` directly). We pack V as a length-(3,3,6) integer
blob at exponent ``f_blob = 2f`` for canonical_reducer.

Public API
----------

``reify_householder(x_1, x_2, x_3, f, theta=None) -> dict``
    Build V as described. Verifies ``V V^† = I_3`` bit-exactly in the ring
    (no floating-point) and (optionally) reports the principal-embedding
    Frobenius residual vs ``R_(0,1)^Z(θ)``.

``decompose_to_gates(V_blob, f, ...) -> dict``
    Thin wrapper around ``hrsa.canonical_reducer.decompose_canonical`` that
    returns ``{N_D, syllables, wall_s, reconstruction_residual}`` — a
    clean per-call view of canonical_reducer's output.

``cvp_compile_one(x_1, x_2, x_3, f, theta, eps=None) -> dict``
    The full Phase 4 pipeline: reify + decompose, packaged in the unified
    ``compile_qutrit_schema_v1`` JSON shape so the result is interchangeable
    with HRSA / zeta9 sweep outputs.
"""
from __future__ import annotations

import math
import sys as _sys
import time
from pathlib import Path
from typing import Optional, Sequence

# canonical_reducer + ep_level live in unified/hrsa/. canonical_reducer adds
# both unified/ and unified/hrsa/ to sys.path at import-time, but we add them
# here too so we can import ep_level directly (canonical_reducer's hot loop
# uses _FastZ9Frac internally and auto-promotes ep_level.Z9Frac inputs).
_UNIFIED_DIR = Path(__file__).resolve().parent.parent
_HRSA_DIR = _UNIFIED_DIR / "hrsa"
if str(_HRSA_DIR) not in _sys.path:
    _sys.path.insert(0, str(_HRSA_DIR))
if str(_UNIFIED_DIR) not in _sys.path:
    _sys.path.insert(0, str(_UNIFIED_DIR))

from ep_level import Z9, Z9Frac  # noqa: E402

from hrsa.canonical_reducer import (  # noqa: E402
    decompose_canonical,
    verify_decomposition,
)

from cvp.diophantine import householder_frobenius  # noqa: E402
from cvp.gram import q_form  # noqa: E402

__all__ = [
    "reify_householder",
    "decompose_to_gates",
    "cvp_compile_one",
]


# ---------------------------------------------------------------------------
# Ring-level helpers: construct V exactly in Z[ζ_9].
# ---------------------------------------------------------------------------


def _z9_from_coefs(coefs: Sequence[int]) -> Z9:
    """Construct a Z9 ring element from a length-6 integer coef vector."""
    if len(coefs) != 6:
        raise ValueError(f"coefs must have length 6, got {len(coefs)}")
    return Z9(tuple(int(c) for c in coefs))


def _build_V_ring(
    x_1: Sequence[int],
    x_2: Sequence[int],
    x_3: Sequence[int],
    f: int,
) -> list[list[Z9Frac]]:
    """Build V = X_(0,1) · (I - u u^†) as a 3x3 nested list of Z9Frac.

    Each entry sits at denom_pow3 = 2f. The (i,j) ring entry is
    ``3^{2f} · δ_ij - x_i · conj(x_j)`` with denom 3^{2f}, except the
    row-swap exchanges rows 0 and 1 of H to produce V.
    """
    f = int(f)
    if f < 0:
        raise ValueError(f"f must be ≥ 0, got {f}")

    x1 = _z9_from_coefs(x_1)
    x2 = _z9_from_coefs(x_2)
    x3 = _z9_from_coefs(x_3)
    xs = (x1, x2, x3)
    xs_conj = tuple(x.conjugate() for x in xs)

    # 3^{2f} as a Z9 ring element (used only to lift the diagonal "1").
    three_pow_2f = Z9.from_int(3 ** (2 * f))

    H: list[list[Z9Frac]] = [[None] * 3 for _ in range(3)]  # type: ignore
    for i in range(3):
        for j in range(3):
            # numerator of (δ_ij - x_i · conj(x_j) / 3^{2f}), denom 3^{2f}.
            cross = xs[i] * xs_conj[j]
            num = (three_pow_2f - cross) if i == j else (Z9.zero() - cross)
            H[i][j] = Z9Frac(num, 2 * f)

    # V = SWAP_(0,1) · H: exchanges rows 0 and 1 of H, leaves row 2.
    V: list[list[Z9Frac]] = [
        list(H[1]),
        list(H[0]),
        list(H[2]),
    ]
    return V


def _check_unitarity_in_ring(V: list[list[Z9Frac]]) -> bool:
    """V · V^† == I_3 bit-exactly in the ring (no float roundoff).

    Returns True iff every entry of V·V^† equals the corresponding entry of
    I_3 under the Z9Frac equality (which aligns denominators internally).
    """
    n = len(V)
    for i in range(n):
        for j in range(n):
            s = Z9Frac.zero()
            for k in range(n):
                s = s + V[i][k] * V[j][k].conjugate()
            expected = Z9Frac.one() if i == j else Z9Frac.zero()
            if not (s == expected):
                return False
    return True


def _V_to_blob(V: list[list[Z9Frac]], f_target: int) -> list[list[list[int]]]:
    """Convert a Z9Frac 3x3 matrix to the [3][3][6] integer blob at a shared
    denominator exponent ``f_target``.

    All entries must have ``denom_pow3 <= f_target``; lower-denom entries are
    scaled up by 3^(f_target - denom_pow3) so the blob is a uniform-f
    representation accepted by canonical_reducer.
    """
    blob: list[list[list[int]]] = []
    for i in range(3):
        row: list[list[int]] = []
        for j in range(3):
            entry = V[i][j]
            scale_exp = int(f_target) - int(entry.denom_pow3)
            if scale_exp < 0:
                raise ValueError(
                    f"entry V[{i}][{j}] has denom_pow3={entry.denom_pow3} > "
                    f"f_target={f_target}"
                )
            scale = 3 ** scale_exp if scale_exp > 0 else 1
            coefs = entry.num.coefs
            if scale != 1:
                coefs = tuple(int(c) * scale for c in coefs)
            row.append([int(c) for c in coefs])
        blob.append(row)
    return blob


# ---------------------------------------------------------------------------
# Public reify
# ---------------------------------------------------------------------------


def reify_householder(
    x_1: Sequence[int],
    x_2: Sequence[int],
    x_3: Sequence[int],
    f: int,
    theta: Optional[float] = None,
    *,
    strict: bool = True,
) -> dict:
    """Reify the Householder unitary V = X_(0,1) · (I - u u^†).

    Parameters
    ----------
    x_1, x_2, x_3 : length-6 integer coef vectors (Z[ζ_9] elements).
    f : denominator exponent. u = (x_1, x_2, x_3) / 3^f.
    theta : optional. If supplied, the principal-embedding Frobenius residual
        ``||V - R_(0,1)^Z(θ)||_F`` is computed and returned in ``frob_residual``.
    strict : if True (default), raise ValueError when the q-sum equation
        ``q(x_1) + q(x_2) + q(x_3) == 2·3^{2f}`` fails. Phase-3 outputs
        always pass this check; the defensive raise catches caller bugs.

    Returns
    -------
    dict with keys:
        V_blob : nested list, shape (3, 3, 6), integer coefficients at
                 denominator exponent ``f_blob = 2f`` (always; do not
                 confuse with input ``f``).
        f_blob : int, = 2*f.
        q_check : bool, ``q(x_1)+q(x_2)+q(x_3) == 2·3^{2f}``.
        unitary_in_ring : bool, ``V V^† == I_3`` bit-exactly.
        frob_residual : float, only when ``theta`` is supplied.

    The returned V_blob is directly consumable by
    :func:`hrsa.canonical_reducer.decompose_canonical` (via load_input or by
    constructing a Z9Frac matrix at ``f = f_blob``).
    """
    if len(x_1) != 6 or len(x_2) != 6 or len(x_3) != 6:
        raise ValueError("x_1, x_2, x_3 must each have length 6")
    f = int(f)
    if f < 0:
        raise ValueError(f"f must be ≥ 0, got {f}")

    # Defensive q-sum check (Phase 3 guarantees this, but the cost is O(1)).
    q1 = q_form(x_1)
    q2 = q_form(x_2)
    q3 = q_form(x_3)
    q_budget = 2 * 3 ** (2 * f)
    q_check = (q1 + q2 + q3) == q_budget
    if strict and not q_check:
        raise ValueError(
            f"Householder norm equation violated: q1={q1} q2={q2} q3={q3}, "
            f"sum={q1+q2+q3} != 2·3^{{2f}}={q_budget}"
        )

    V = _build_V_ring(x_1, x_2, x_3, f)
    unitary_ok = _check_unitarity_in_ring(V)
    if strict and not unitary_ok:
        raise ValueError(
            "V V^† != I in the ring; this is impossible when q-sum holds — "
            "indicates a reify or ring-arithmetic bug."
        )

    f_blob = 2 * f
    blob = _V_to_blob(V, f_blob)

    out: dict = {
        "V_blob": blob,
        "f_blob": f_blob,
        "q_check": q_check,
        "unitary_in_ring": unitary_ok,
    }
    if theta is not None:
        out["frob_residual"] = householder_frobenius(
            tuple(int(c) for c in x_1),
            tuple(int(c) for c in x_2),
            tuple(int(c) for c in x_3),
            theta=float(theta), f=f,
        )
    return out


# ---------------------------------------------------------------------------
# Decompose wrapper
# ---------------------------------------------------------------------------


def _blob_to_z9frac_mat(
    V_blob: Sequence[Sequence[Sequence[int]]],
    f_blob: int,
) -> list[list[Z9Frac]]:
    """Inverse of ``_V_to_blob``: a 3x3 nested list of Z9Frac at shared denom."""
    out: list[list[Z9Frac]] = []
    for i in range(3):
        row: list[Z9Frac] = []
        for j in range(3):
            coefs = V_blob[i][j]
            if len(coefs) != 6:
                raise ValueError(
                    f"V_blob[{i}][{j}] must be length-6, got {len(coefs)}"
                )
            row.append(Z9Frac(_z9_from_coefs(coefs), int(f_blob)))
        out.append(row)
    return out


def decompose_to_gates(
    V_blob: Sequence[Sequence[Sequence[int]]],
    f_blob: int,
    *,
    greedy: bool = True,
    max_iter: Optional[int] = None,
    verify: bool = True,
) -> dict:
    """Decompose V into Clifford+D syllables via canonical_reducer.

    Parameters
    ----------
    V_blob : nested list, shape (3, 3, 6), integer coefs (as produced by
        :func:`reify_householder`).
    f_blob : denominator exponent of every entry.
    greedy : passes ``greedy_single=True`` to canonical_reducer. Per memory
        ``canonical_reducer_2026-05-24``, greedy is recommended at f≫8 and
        does no harm at f≤8; we leave it on by default.
    max_iter : explicit peel-iteration cap (default: sde_chi+50).
    verify : if True, also call ``verify_decomposition`` on success and
        include the bit-exact reconstruction residual.

    Returns
    -------
    dict with:
        N_D : int, total D-gate count (syllable + residual). -1 on failure.
        syllables : list of GateStep dicts (one per peel iteration).
        wall_s : float, total wall time (decompose + verify).
        reconstruction_residual : int, ``max_coef_residual`` from the
            verification step. 0 means the syllable product reproduces the
            input V bit-for-bit. Only present when verify=True and success.
        success : bool.
        sde_chi_initial : int.
        sde_chi_final : int.
        n_iter : int.
        error : str, only present on failure.
    """
    V = _blob_to_z9frac_mat(V_blob, int(f_blob))

    t0 = time.time()
    result = decompose_canonical(
        V,
        max_iter=max_iter,
        verbose=False,
        greedy_single=bool(greedy),
        skip_double=False,
    )
    decomp_wall = time.time() - t0

    out: dict = {
        "N_D": int(result["D_count"]),
        "syllables": result["syllables"],
        "wall_s": float(decomp_wall),
        "success": bool(result["success"]),
        "sde_chi_initial": int(result["sde_chi_initial"]),
        "sde_chi_final": int(result["sde_chi_final"]),
        "n_iter": int(result.get("n_iter", 0)),
    }
    if "error" in result:
        out["error"] = result["error"]

    if verify and result["success"]:
        t1 = time.time()
        ver = verify_decomposition(
            V, result["syllables"], result["trailing_clifford"]
        )
        out["wall_s"] += time.time() - t1
        out["reconstruction_residual"] = int(ver["max_coef_residual"])
        out["matches_input"] = bool(ver["matches"])

    return out


# ---------------------------------------------------------------------------
# Top-level Phase 4 entry point
# ---------------------------------------------------------------------------


def cvp_compile_one(
    x_1: Sequence[int],
    x_2: Sequence[int],
    x_3: Sequence[int],
    f: int,
    theta: float,
    *,
    eps: Optional[float] = None,
    greedy: bool = True,
) -> dict:
    """Full Phase 4 pipeline for one (x_1, x_2, x_3) triple.

    Reifies the Householder unitary, decomposes via canonical_reducer, and
    returns a JSON-ready record in the unified
    ``compile_qutrit_schema_v1`` shape.

    Schema (subset matching ``compile_qutrit_schema_example.json`` keys):

    .. code-block::

        {
          "inputs":       {"theta", "epsilon", "f"},
          "achieved":     {"success", "achieved_frob", "epsilon_passed",
                           "f_level", "method"},
          "unitary":      {"ring": "Z[zeta_9, 1/3]", "basis": "...",
                           "f": f_blob, "V": V_blob},
          "decomposition":{"N_D", "syllables", "sde_chi_initial",
                           "sde_chi_final"},
          "sanity_checks":{"q_sum_ok", "unitary_in_ring",
                           "reconstruction_residual",
                           "frobenius_check_passed"},
          "performance":  {"reify_wall_s", "decompose_wall_s",
                           "total_wall_s"}
        }

    ``method`` is ``"cvp-babai"`` to distinguish from HRSA / zeta9 sweep
    records. The caller is expected to attach ``identification`` (host,
    timestamp, command_line, ...) externally.
    """
    if f < 0:
        raise ValueError(f"f must be ≥ 0, got {f}")
    theta = float(theta)

    t_total = time.time()
    t_reify = time.time()
    reify = reify_householder(x_1, x_2, x_3, f, theta=theta, strict=True)
    reify_wall = time.time() - t_reify

    decomp = decompose_to_gates(
        reify["V_blob"], reify["f_blob"],
        greedy=greedy, verify=True,
    )
    total_wall = time.time() - t_total

    achieved_frob = float(reify["frob_residual"])
    epsilon_passed = (eps is not None) and (achieved_frob <= float(eps) + 1e-12)

    record: dict = {
        "inputs": {
            "theta": theta,
            "epsilon": (float(eps) if eps is not None else None),
            "f": int(f),
        },
        "achieved": {
            "success": bool(decomp["success"]),
            "achieved_frob": achieved_frob,
            "epsilon_passed": (epsilon_passed if eps is not None else None),
            "f_level": int(f),
            "method": "cvp-babai",
        },
        "unitary": {
            "ring": "Z[zeta_9, 1/3]",
            "basis": "1, zeta9, zeta9^2, zeta9^3, zeta9^4, zeta9^5",
            "f": int(reify["f_blob"]),
            "V": reify["V_blob"],
        },
        "decomposition": {
            "N_D": int(decomp["N_D"]),
            "syllables": decomp["syllables"],
            "sde_chi_initial": int(decomp["sde_chi_initial"]),
            "sde_chi_final": int(decomp["sde_chi_final"]),
        },
        "sanity_checks": {
            "q_sum_ok": bool(reify["q_check"]),
            "unitary_in_ring": bool(reify["unitary_in_ring"]),
            "reconstruction_residual": int(
                decomp.get("reconstruction_residual", -1)
            ),
            "frobenius_check_passed": (
                epsilon_passed if eps is not None else None
            ),
        },
        "performance": {
            "reify_wall_s": float(reify_wall),
            "decompose_wall_s": float(decomp["wall_s"]),
            "total_wall_s": float(total_wall),
        },
    }
    if "error" in decomp:
        record["errors"] = [decomp["error"]]
    return record


# ---------------------------------------------------------------------------
# Module entry point — quick smoke test
# ---------------------------------------------------------------------------


def _main():  # pragma: no cover - manual sanity
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Smoke-test cvp.reify on the HRSA f=2 reference triple."
    )
    parser.add_argument("--theta", type=float, default=math.pi / 2)
    parser.add_argument("--f", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.05)
    args = parser.parse_args()

    # HRSA f=2 reference: see cvp/tests/test_diophantine.py.
    x_1 = (4, -2, 4, 5, -4, -2)
    x_2 = (0, -4, 1, 3, 1, 4)
    x_3 = (-6, 2, 3, -6, 1, -2)

    rec = cvp_compile_one(x_1, x_2, x_3, args.f, args.theta, eps=args.eps)
    print(json.dumps({k: v for k, v in rec.items() if k != "unitary"},
                     indent=2, default=str))
    print(f"unitary.f={rec['unitary']['f']}, V[0][0]={rec['unitary']['V'][0][0]}")


if __name__ == "__main__":  # pragma: no cover
    _main()
