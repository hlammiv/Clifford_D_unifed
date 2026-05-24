#!/usr/bin/env python3
"""cvp_compile.py — Phase 5 driver for the qutrit Babai-CVP pipeline.

Top-level entry point that orchestrates Phases 2-4 with a trial-f outer loop:

    f = f_start, f_start+1, ..., f_start + max_f_iters
        babai_x1(theta, f, n_candidates, eps)
          for each x_1 candidate:
              solve_x2_x3(x_1, theta, f, eps)
                  for each (x_2, x_3) pair:
                      reify_householder + decompose_to_gates
                      stop at first pair whose reified V has
                      ||V - R_z(theta)||_F <= eps

Returns a JSON record in the unified compile_qutrit_schema_v1 shape so the
result is interchangeable with HRSA / zeta9 sweep outputs.

f_start selection
-----------------

The plan's bound ``f_min = ceil(log_3(8·c/eps))`` is the Babai
approximation-factor bound. Empirically the ``Z[ζ_9]`` orbit phase
resolution sets a *separate* lower bound: at small ``f`` even a tight
σ_1 hit translates to a Frob residual ≳ π/18 · 3^{-f}. We start at
``max(babai_bound, orbit_floor_estimate)`` and walk up.

Usage::

    >>> from cvp_compile import cvp_compile
    >>> rec = cvp_compile(theta=math.pi/2, eps=1e-3)
    >>> rec["achieved"]["success"], rec["decomposition"]["N_D"]

Or from the CLI::

    python3 cvp_compile.py --theta 1.5707963 --eps 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

_UNIFIED_DIR = Path(__file__).resolve().parent
if str(_UNIFIED_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIFIED_DIR))

from cvp.babai import babai_x1
from cvp.diophantine import (
    _NormEqWorker,
    householder_frobenius,
    solve_x2_x3_ring_unitary,
)
from cvp.gram import q_form
from cvp.reify import decompose_to_gates, reify_householder

__all__ = ["cvp_compile", "f_start_recommend"]

_SCHEMA_VERSION = "1.0"
_THIS_VERSION = "cvp-2026-05-24-phase5"


# ---------------------------------------------------------------------------
# f_start heuristic
# ---------------------------------------------------------------------------


def _babai_f_min(eps: float, c: float = 1.0) -> int:
    """Plan's Babai-approximation bound: ``ceil(log_3(8·c/eps))``.

    Reflects the 2^(d/2)=2^9 approximation factor of plain Babai on the
    6-D LLL-reduced lattice. Loose upper bound; the orbit floor below
    often dominates at moderate ε.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return int(math.ceil(math.log(max(8.0 * c / eps, 3.0001), 3.0)))


def _orbit_floor_f(eps: float) -> int:
    """Empirical orbit-resolution floor.

    At low ``f``, the principal-embedding σ_1 of ``Z[ζ_9]`` has discrete
    phase resolution ≈ π/9 per unit-orbit element (the 18-fold unit group
    ±ζ^k). After scaling by 3^f the *achievable* Frobenius residual is
    bounded below by ≈ π/9·3^{-f} on the unit grid. Inverting gives

        f_floor ≈ ceil(log_3(π / (9·eps))).

    This is a *lower* bound on the f at which the pipeline can satisfy
    a Frob ≤ eps constraint; the actual lower bound depends on the slab
    geometry but is well-approximated by this formula for the working
    regime eps ∈ [1e-8, 1e-2].
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return int(math.ceil(math.log(max(math.pi / (9.0 * eps), 3.0001), 3.0)))


def f_start_recommend(eps: float, c: float = 1.0) -> int:
    """Return ``max(babai_bound, orbit_floor, 2)``.

    Two is the absolute minimum f for which the q-budget ``2·3^{2f} = 162``
    can host a nontrivial Householder triple (q-budget=2 at f=0 is too tight).
    """
    bf = _babai_f_min(eps, c)
    of = _orbit_floor_f(eps)
    return max(bf, of, 2)


# ---------------------------------------------------------------------------
# Selinger n_candidates scaling
# ---------------------------------------------------------------------------


def _n_candidates_recommend(
    eps: float, retry_count: int = 0, cap: int = 1000
) -> int:
    """Selinger 2012 §6: ``n = max(10, ceil(4·√2/eps))``.

    Doubled per failed-f retry up to ``cap`` so each escalation explores a
    strictly larger candidate pool before walking f up another step.
    """
    base = max(10, int(math.ceil(4.0 * math.sqrt(2.0) / max(eps, 1e-16))))
    n = base * (2 ** retry_count)
    return min(n, cap)


# ---------------------------------------------------------------------------
# Frobenius residual check (independent of reify's internal computation)
# ---------------------------------------------------------------------------


def _target_R_z(theta: float) -> np.ndarray:
    return np.diag([
        np.exp(-1j * theta / 2.0),
        np.exp(+1j * theta / 2.0),
        1.0 + 0j,
    ])


def _V_blob_to_complex(V_blob, f_blob: int) -> np.ndarray:
    """Map [3][3][6] int blob at denom 3^{f_blob} to 3x3 complex via σ_1."""
    omega = [
        complex(math.cos(2 * math.pi * j / 9.0),
                math.sin(2 * math.pi * j / 9.0))
        for j in range(6)
    ]
    denom = 3.0 ** f_blob
    V = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            s = sum(V_blob[i][j][k] * omega[k] for k in range(6))
            V[i, j] = s / denom
    return V


def _frob_residual_from_blob(V_blob, f_blob: int, theta: float) -> float:
    V = _V_blob_to_complex(V_blob, f_blob)
    R = _target_R_z(theta)
    return float(np.linalg.norm(V - R, "fro"))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def cvp_compile(
    theta: float,
    eps: float,
    *,
    f_start: Optional[int] = None,
    f_max: Optional[int] = None,
    max_f_iters: int = 6,
    n_candidates_initial: Optional[int] = None,
    n_candidates_cap: int = 1000,
    max_x1_to_try: int = 16,
    max_pairs_per_x1: int = 8,
    max_x3: int = 256,
    max_m_triples_per_x3: int = 512,  # unused since switch to ring-unitary; kept for API stability
    c: float = 1.0,
    greedy: bool = True,
    worker: Optional[_NormEqWorker] = None,
    verbose: bool = False,
    fallback_method: str = "sk",
    command_line: Optional[List[str]] = None,
) -> dict:
    """Run the full qutrit Babai-CVP pipeline for one (theta, eps) cell.

    Parameters
    ----------
    theta : float
        Target rotation angle. Target unitary is ``R_(0,1)^Z(θ)``.
    eps : float
        Frobenius tolerance.
    f_start : int, optional
        Starting denominator exponent. Defaults to ``f_start_recommend(eps)``.
    f_max : int, optional
        Hard upper cap on f. Defaults to ``f_start + max_f_iters``.
    max_f_iters : int
        Number of f values to try before declaring failure (only used to
        derive default ``f_max``).
    n_candidates_initial : int, optional
        Initial x_1 candidate count. Defaults to Selinger's
        ``max(10, ceil(4·sqrt(2)/eps))``. Doubled per retry.
    n_candidates_cap : int
        Upper bound on candidate count after doubling.
    max_x1_to_try : int
        At each f, only the top-``max_x1_to_try`` x_1 candidates (by σ_1
        proximity) feed into Phase 3. Higher = more thorough but costlier.
    max_pairs_per_x1 : int
        Cap on (x_2, x_3) pairs returned per x_1 (already Frob-sorted).
    max_x3, max_m_triples_per_x3 : int
        Forwarded to ``solve_x2_x3`` to bound Phase 3 PARI calls.
    c : float
        Householder contraction factor (HRSA convention; default 1.0).
    greedy : bool
        Forwarded to ``decompose_to_gates`` (greedy peeler).
    worker : optional
        Pre-warmed ``_NormEqWorker``. If None, one is spun up and torn down
        within this call (~3 s startup amortised).
    verbose : bool
        Emit per-f progress to stderr.
    fallback_method : str
        Recorded in the failure record as the suggested next-method
        (``"sk"`` or ``"hrsa"``). This driver does NOT actually invoke
        the fallback; that's Phase 6 (hybrid driver).
    command_line : list[str], optional
        Recorded under ``identification.command_line``. Defaults to
        ``sys.argv`` if running as a script.

    Returns
    -------
    dict
        Unified ``compile_qutrit_schema_v1`` record. On success the
        record's ``achieved.epsilon_passed == True`` and the
        ``decomposition.N_D`` is the canonical_reducer D-count. On
        failure, ``achieved.success == False`` and ``attempted_f_levels``
        records the f's tried.
    """
    t_total_start = time.time()
    theta = float(theta)
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if f_start is None:
        f_start = f_start_recommend(eps, c=c)
    f_start = int(f_start)
    if f_max is None:
        f_max = f_start + int(max_f_iters)
    f_max = int(f_max)

    if n_candidates_initial is None:
        n_candidates_initial = _n_candidates_recommend(eps, retry_count=0,
                                                       cap=n_candidates_cap)

    attempted_f_levels: List[dict] = []
    best_failure: Optional[dict] = None  # best (lowest-frob) reified-but-too-loose result

    owns_worker = worker is None
    if owns_worker:
        worker = _NormEqWorker()
        worker.start()

    success_record: Optional[dict] = None
    n_candidates_used: int = 0

    try:
        for retry_count, f in enumerate(range(f_start, f_max + 1)):
            n_cands = _n_candidates_recommend(eps, retry_count=retry_count,
                                              cap=n_candidates_cap)
            n_candidates_used = max(n_candidates_used, n_cands)
            t_f0 = time.time()

            # Phase 2: x_1 candidates
            x1_cands = babai_x1(theta, f, n_candidates=n_cands, eps=eps)
            t_babai = time.time() - t_f0

            if verbose:
                print(
                    f"[cvp_compile] f={f} eps={eps}: babai {len(x1_cands)} cands "
                    f"in {t_babai:.2f}s",
                    file=sys.stderr,
                )

            n_pairs_seen = 0
            n_x1_tried = 0
            best_frob_this_f = float("inf")
            best_triple_this_f = None

            for x1 in x1_cands[:max_x1_to_try]:
                n_x1_tried += 1
                t_x1 = time.time()
                pairs = solve_x2_x3_ring_unitary(
                    x_1=x1, theta=theta, f=f, eps=eps,
                    max_pairs=max_pairs_per_x1,
                    max_x3=max_x3,
                    worker=worker,
                    c=c,
                )
                t_pairs = time.time() - t_x1
                n_pairs_seen += len(pairs)

                for x2, x3 in pairs:
                    frob = householder_frobenius(
                        x1, x2, x3, theta=theta, f=f,
                    )
                    if frob < best_frob_this_f:
                        best_frob_this_f = frob
                        best_triple_this_f = (x1, x2, x3, frob)

                    if frob <= eps:
                        # Found a passing triple — reify + decompose.
                        if verbose:
                            print(
                                f"[cvp_compile]   HIT at f={f}: frob={frob:.4g} "
                                f"<= eps={eps}; decomposing...",
                                file=sys.stderr,
                            )
                        rec = _build_success_record(
                            x_1=x1, x_2=x2, x_3=x3,
                            f=f, theta=theta, eps=eps, greedy=greedy,
                            n_candidates_used=n_cands,
                            attempted_f_levels=attempted_f_levels,
                            command_line=command_line,
                            t_total_start=t_total_start,
                        )
                        success_record = rec
                        attempted_f_levels.append({
                            "f": f, "n_x1_tried": n_x1_tried,
                            "n_pairs": n_pairs_seen,
                            "best_frob": best_frob_this_f,
                            "wall_s": time.time() - t_f0,
                            "outcome": "hit",
                        })
                        break

                if success_record is not None:
                    break

                if verbose and pairs:
                    fr0 = householder_frobenius(x1, *pairs[0],
                                                theta=theta, f=f)
                    print(
                        f"[cvp_compile]   x1 #{n_x1_tried} q={q_form(x1)}: "
                        f"{len(pairs)} pairs in {t_pairs:.2f}s, "
                        f"best_frob={fr0:.4g}",
                        file=sys.stderr,
                    )

            if success_record is not None:
                break

            # No hit at this f
            attempted_f_levels.append({
                "f": f, "n_x1_tried": n_x1_tried,
                "n_pairs": n_pairs_seen,
                "best_frob": (best_frob_this_f
                              if best_frob_this_f != float("inf") else None),
                "wall_s": time.time() - t_f0,
                "outcome": "no_hit",
            })

            if best_triple_this_f is not None:
                if (best_failure is None
                        or best_triple_this_f[3] < best_failure["frob"]):
                    best_failure = {
                        "x_1": best_triple_this_f[0],
                        "x_2": best_triple_this_f[1],
                        "x_3": best_triple_this_f[2],
                        "f": f,
                        "frob": best_triple_this_f[3],
                    }

            if verbose:
                print(
                    f"[cvp_compile] f={f}: no hit, best_frob={best_frob_this_f:.4g} "
                    f"(wall {time.time()-t_f0:.2f}s)",
                    file=sys.stderr,
                )

    finally:
        if owns_worker and worker is not None:
            worker.close()

    if success_record is not None:
        return success_record

    # All f's exhausted — build a failure record.
    return _build_failure_record(
        theta=theta, eps=eps, f_start=f_start, f_max=f_max,
        attempted_f_levels=attempted_f_levels,
        best_failure=best_failure,
        n_candidates_used=n_candidates_used,
        fallback_method=fallback_method,
        command_line=command_line,
        t_total_start=t_total_start,
    )


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _identification(command_line: Optional[List[str]] = None) -> dict:
    return {
        "backend": "cvp-babai",
        "version": _THIS_VERSION,
        "command_line": command_line if command_line is not None else list(sys.argv),
        "host": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": _SCHEMA_VERSION,
    }


def _target_block(theta: float) -> dict:
    R = _target_R_z(theta)
    mat = [
        [[float(R[i, j].real), float(R[i, j].imag)] for j in range(3)]
        for i in range(3)
    ]
    return {
        "gate": "R_Z_01_theta",
        "convention": "hrsa",  # diag(e^{-iθ/2}, e^{+iθ/2}, 1)
        "matrix": mat,
    }


def _build_success_record(
    *,
    x_1, x_2, x_3, f: int, theta: float, eps: float, greedy: bool,
    n_candidates_used: int,
    attempted_f_levels: list,
    command_line: Optional[List[str]],
    t_total_start: float,
) -> dict:
    t_reify_start = time.time()
    reify = reify_householder(x_1, x_2, x_3, f, theta=theta, strict=True)
    reify_wall = time.time() - t_reify_start

    t_decomp_start = time.time()
    decomp = decompose_to_gates(
        reify["V_blob"], reify["f_blob"],
        greedy=greedy, verify=True,
    )
    decomp_wall = time.time() - t_decomp_start

    # Independent (numpy) Frobenius check from the reified V blob — this is
    # our "v_validate-style" sanity check so the success criterion does not
    # depend on the internal householder_frobenius computation.
    frob_blob = _frob_residual_from_blob(
        reify["V_blob"], reify["f_blob"], theta,
    )
    achieved_frob = float(reify["frob_residual"])
    epsilon_passed = (achieved_frob <= eps + 1e-12) and (frob_blob <= eps + 1e-12)

    record = {
        "identification": _identification(command_line),
        "inputs": {
            "theta": theta,
            "epsilon": eps,
            "f_start": (int(attempted_f_levels[0]["f"])
                        if attempted_f_levels else int(f)),
            "f_hit": int(f),
            "f_max": int(f),  # only reached up to this f; not the configured cap
            "n_candidates_used": int(n_candidates_used),
            "max_f_iters_attempted": len(attempted_f_levels),
        },
        "target": _target_block(theta),
        "achieved": {
            "success": bool(decomp["success"]) and epsilon_passed,
            "achieved_frob": achieved_frob,
            "achieved_frob_blob": frob_blob,
            "epsilon_passed": bool(epsilon_passed),
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
            "reconstruction_residual": int(decomp.get("reconstruction_residual", -1)),
            "frobenius_check_passed": bool(epsilon_passed),
        },
        "performance": {
            "wall_seconds": float(time.time() - t_total_start),
            "reify_wall_s": float(reify_wall),
            "decompose_wall_s": float(decomp_wall),
            "threads": 1,
            "mpi_ranks": None,
        },
        "attempted_f_levels": attempted_f_levels,
        "errors": [],
    }
    if "error" in decomp:
        record["errors"].append(decomp["error"])
    return record


def _build_failure_record(
    *,
    theta: float, eps: float, f_start: int, f_max: int,
    attempted_f_levels: list,
    best_failure: Optional[dict],
    n_candidates_used: int,
    fallback_method: str,
    command_line: Optional[List[str]],
    t_total_start: float,
) -> dict:
    record = {
        "identification": _identification(command_line),
        "inputs": {
            "theta": theta,
            "epsilon": eps,
            "f_start": int(f_start),
            "f_max": int(f_max),
            "n_candidates_used": int(n_candidates_used),
            "max_f_iters_attempted": len(attempted_f_levels),
        },
        "target": _target_block(theta),
        "achieved": {
            "success": False,
            "achieved_frob": (float(best_failure["frob"])
                              if best_failure else None),
            "epsilon_passed": False,
            "f_level": (int(best_failure["f"])
                        if best_failure else None),
            "method": "cvp-babai",
        },
        "fallback_method": fallback_method,
        "attempted_f_levels": attempted_f_levels,
        "performance": {
            "wall_seconds": float(time.time() - t_total_start),
            "threads": 1,
            "mpi_ranks": None,
        },
        "errors": [
            f"no (x_1,x_2,x_3) triple with frob <= eps={eps} found "
            f"across f in [{f_start}, {f_max}]"
        ],
    }
    if best_failure is not None:
        record["best_partial"] = {
            "x_1": list(best_failure["x_1"]),
            "x_2": list(best_failure["x_2"]),
            "x_3": list(best_failure["x_3"]),
            "f": int(best_failure["f"]),
            "frob": float(best_failure["frob"]),
        }
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main():  # pragma: no cover - CLI smoke
    ap = argparse.ArgumentParser(
        description="Phase 5 qutrit Babai-CVP compiler driver."
    )
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--f-start", type=int, default=None)
    ap.add_argument("--f-max", type=int, default=None)
    ap.add_argument("--max-f-iters", type=int, default=6)
    ap.add_argument("--max-x1", type=int, default=16,
                    help="x_1 candidates to feed Phase 3 per f-level")
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default=None,
                    help="If set, write the JSON record here.")
    ap.add_argument("--fallback", choices=("sk", "hrsa"), default="sk")
    args = ap.parse_args()

    rec = cvp_compile(
        theta=args.theta, eps=args.eps,
        f_start=args.f_start, f_max=args.f_max,
        max_f_iters=args.max_f_iters,
        max_x1_to_try=args.max_x1,
        max_pairs_per_x1=args.max_pairs,
        verbose=args.verbose,
        fallback_method=args.fallback,
        command_line=list(sys.argv),
    )

    # Trim V_blob for stdout to keep things readable.
    short = {k: v for k, v in rec.items() if k != "unitary"}
    print(json.dumps(short, indent=2, default=str))
    if rec["achieved"]["success"]:
        print(
            f"\nSUCCESS: f={rec['achieved']['f_level']} "
            f"N_D={rec['decomposition']['N_D']} "
            f"frob={rec['achieved']['achieved_frob']:.4g} "
            f"wall={rec['performance']['wall_seconds']:.2f}s",
            file=sys.stderr,
        )
    else:
        bp = rec.get("best_partial")
        bp_str = (
            f"best_partial frob={bp['frob']:.4g} at f={bp['f']}" if bp
            else "no candidates"
        )
        print(
            f"\nFAILURE: tried f in [{rec['inputs']['f_start']}, "
            f"{rec['inputs'].get('f_max', '?')}], {bp_str}",
            file=sys.stderr,
        )

    if args.out:
        Path(args.out).write_text(json.dumps(rec, default=str))
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    _main()
