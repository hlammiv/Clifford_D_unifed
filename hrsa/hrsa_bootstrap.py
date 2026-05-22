"""HRSA bootstrap adapter for Solovay-Kitaev refinement.

Given a target unitary and a loose threshold eps_hrsa (chosen so the residual
E = V^dagger · target falls inside the SK contraction basin, typically
|E - I|_F <~ 0.3), run HRSA_tester to obtain an exact ringZ9 unitary V, then
hand both V and E back to the caller.  The companion module sk_driver.py is
responsible for the SK recursion on E.

Key conventions
---------------
* Target convention (project-wide): R^Z(theta) = diag(e^{-i theta/2},
  e^{+i theta/2}, 1).  See unified/v_validate.py.
* ringZ9 basis: a length-6 integer tuple (a_0, ..., a_5) encodes
  Sum_{k=0..5} a_k * zeta_9^k, zeta_9 = exp(2 pi i / 9).
  Phi_9(zeta) = zeta^6 + zeta^3 + 1 = 0, so the higher powers are redundant
  and HRSA emits everything in this 6-coefficient canonical form.
* A V matrix entry V[i][j] = (1 / 3^f) * Sum_k a_k * zeta_9^k where a_k come
  from the per-entry integer list of length 6 and f is the unitary's shared
  exponent (data["unitary"]["f"] in the JSON schema).

Public API
----------
run_hrsa_json(theta, epsilon, max_f, timeout_s)
ringZ9_to_complex(coefs, f)
V_matrix_from_hrsa(json_blob)
compute_residual(target, V)
bootstrap_target(target, eps_hrsa, theta, max_f, ...)

The __main__ block runs a smoke test at four representative angles and writes
a JSON summary to /tmp/hrsa_bootstrap_results.json.
"""
from __future__ import annotations

import argparse
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


HERE = Path(__file__).resolve().parent
HRSA_TESTER = HERE / "HRSA_tester"
MAKEFILE_DIR = HERE                                       # `make HRSA_tester`
ZETA9 = np.exp(2j * np.pi / 9)


# ---------------------------------------------------------------------------
# 1. Subprocess driver
# ---------------------------------------------------------------------------

def _ensure_binary_built() -> None:
    """If HRSA_tester is missing, try to `make` it.  Raise on failure."""
    if HRSA_TESTER.exists() and os.access(HRSA_TESTER, os.X_OK):
        return
    print(f"[hrsa_bootstrap] HRSA_tester not found at {HRSA_TESTER}; running make ...",
          flush=True)
    rc = subprocess.run(["make", "HRSA_tester"], cwd=str(MAKEFILE_DIR)).returncode
    if rc != 0 or not HRSA_TESTER.exists():
        raise RuntimeError(f"failed to build HRSA_tester (make returncode {rc})")


def run_hrsa_json(theta: float,
                  epsilon: float,
                  max_f: int = 8,
                  timeout_s: float = 120.0,
                  extra_args: Optional[list] = None,
                  json_path: Optional[Path] = None) -> dict:
    """Invoke HRSA_tester, parse the resulting JSON, attach a 'V_complex' helper.

    Returns the JSON dict augmented with:
      - 'V_complex'    : a (3,3) np.complex128 reconstruction of the unitary
                         (None if no solution found).
      - '_invocation'  : {'cmd': [...], 'wall_seconds': float, 'returncode': int}.
    """
    _ensure_binary_built()

    delete_after = json_path is None
    if json_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.close()
        json_path = Path(tmp.name)
    else:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(HRSA_TESTER),
           f"{theta:.10f}",
           f"{epsilon:g}",
           str(int(max_f)),
           "--json", str(json_path)]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    wall = time.time() - t0

    if not json_path.exists():
        raise RuntimeError(
            f"HRSA_tester did not write JSON at {json_path}; returncode={rc}; "
            f"stderr tail: {proc.stderr[-400:] if rc != -1 else 'TIMEOUT'}")

    with json_path.open() as fh:
        data = json.load(fh)

    # Attach complex V (or None) for downstream convenience.
    try:
        V = V_matrix_from_hrsa(data)
    except Exception:
        V = None
    data["V_complex"] = V
    data["_invocation"] = {
        "cmd": cmd,
        "wall_seconds": wall,
        "returncode": rc,
        "json_path": str(json_path),
    }

    if delete_after:
        try:
            json_path.unlink()
        except OSError:
            pass

    return data


# ---------------------------------------------------------------------------
# 2. ringZ9 -> complex
# ---------------------------------------------------------------------------

def ringZ9_to_complex(coefs, f: int) -> complex:
    """Evaluate (Sum_{k=0..5} a_k * zeta_9^k) / 3^f at zeta_9 = exp(2 pi i / 9).

    `coefs` must be an iterable of length 6 of ints (the canonical basis in
    which HRSA emits ringZ9 numerators).  Returns a Python complex.
    """
    if len(coefs) != 6:
        raise ValueError(f"expected 6-coef ringZ9 numerator, got {len(coefs)}")
    s = 0.0 + 0.0j
    for k, a in enumerate(coefs):
        if a:
            s += a * (ZETA9 ** k)
    return s / (3 ** f)


def V_matrix_from_hrsa(json_blob: dict) -> np.ndarray:
    """Reconstruct the (3,3) complex unitary V from an HRSA JSON blob.

    Expects the schema produced by HRSA_test.cpp (schema_version 1.0):
        json_blob["unitary"]["f"]     : shared denom-exponent
        json_blob["unitary"]["V"][i][j] : list of 6 ints (ringZ9 numerator).
    Raises ValueError if `unitary` is null (no solution found).
    """
    uni = json_blob.get("unitary")
    if uni is None:
        raise ValueError("HRSA JSON has no 'unitary' (no solution found).")
    f = int(uni["f"])
    Vint = uni["V"]
    V = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            V[i, j] = ringZ9_to_complex(Vint[i][j], f)
    return V


# ---------------------------------------------------------------------------
# 3. Residual + SK hand-off
# ---------------------------------------------------------------------------

def compute_residual(target: np.ndarray, V: np.ndarray) -> np.ndarray:
    """E = V^dagger @ target.  SK recursion factors E (a near-identity unitary)
    into a group commutator [A, B] with |A - I|, |B - I| ~ c * sqrt(|E - I|)."""
    return V.conj().T @ target


def frob_norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, "fro"))


# ---------------------------------------------------------------------------
# 4. Top-level bootstrap entry point
# ---------------------------------------------------------------------------

def _theta_from_target(target: np.ndarray) -> float:
    """Recover theta from target = diag(e^{-i theta/2}, e^{+i theta/2}, 1).

    Uses arg(target[0, 0]) = -theta/2, so theta = -2 * arg(target[0, 0]).
    """
    return float(-2.0 * np.angle(target[0, 0]))


def target_R_Z(theta: float) -> np.ndarray:
    """Project-wide convention: R^Z(theta) = diag(e^{-i th/2}, e^{+i th/2}, 1)."""
    return np.diag([np.exp(-0.5j * theta),
                    np.exp(+0.5j * theta),
                    1.0 + 0.0j]).astype(np.complex128)


def bootstrap_target(target: np.ndarray,
                     eps_hrsa: float = 0.3,
                     theta: Optional[float] = None,
                     max_f: int = 4,
                     fallback_max_f: int = 6,
                     timeout_s: float = 120.0,
                     extra_args: Optional[list] = None) -> dict:
    """Run HRSA at the loose threshold eps_hrsa and package the result for SK.

    Parameters
    ----------
    target       : (3,3) complex unitary to approximate (must be a z-rotation
                   given the current HRSA interface; non-z targets need a
                   pre-Clifford reduction we do not perform here).
    eps_hrsa     : loose Frobenius threshold; pick small enough that
                   |E - I|_F lands in the SK contraction basin (~0.3).
    theta        : optional; if given, skip the target-to-theta extraction.
    max_f        : initial V-denominator cap passed to HRSA.
    fallback_max_f: if `max_f` yields no solution, retry once at this value.

    Returns a dict with:
        'hrsa_word'    : list of dicts ('decomposition.syllables' from the
                         HRSA JSON; may be None if HRSA returned a Clifford-
                         only / Direct match without syllables).
        'V'            : (3,3) complex unitary, exact reconstruction.
        'frob_error'   : |V - target|_F.
        'residual_E'   : V^dagger @ target.
        'residual_norm': |E - I|_F.
        'unitarity_err': |V V^dagger - I|_F  (should be at machine zero).
        'N_D'          : decomposition.N_D (number of D gates).
        'theta'        : the angle used.
        'f_level'      : achieved f-level (V denom exponent).
        'method'       : HRSA-reported method ('Direct', 'HRSA', 'Bidir', ...).
        'achieved_frob': HRSA-reported floating-point Frobenius distance.
        'success'      : True iff HRSA found a solution satisfying eps_hrsa.
        'raw_json'     : the full HRSA JSON for downstream debugging.
    """
    if theta is None:
        theta = _theta_from_target(target)
    # If caller passed theta but not target, materialise the canonical target.
    if target is None:
        target = target_R_Z(theta)

    # First attempt.
    data = run_hrsa_json(theta=theta, epsilon=eps_hrsa,
                         max_f=max_f, timeout_s=timeout_s,
                         extra_args=extra_args)
    achieved = data.get("achieved", {}) or {}
    if not achieved.get("success", False) and fallback_max_f and fallback_max_f > max_f:
        print(f"[hrsa_bootstrap] theta={theta:+.4f} eps={eps_hrsa}: no solution at "
              f"max_f={max_f}; retrying at max_f={fallback_max_f}", flush=True)
        data = run_hrsa_json(theta=theta, epsilon=eps_hrsa,
                             max_f=fallback_max_f, timeout_s=timeout_s,
                             extra_args=extra_args)
        achieved = data.get("achieved", {}) or {}

    success = bool(achieved.get("success", False))
    V = data.get("V_complex")
    decomposition = data.get("decomposition") or {}
    syllables = decomposition.get("syllables")
    N_D = decomposition.get("N_D")

    result = {
        "theta": theta,
        "eps_hrsa": eps_hrsa,
        "success": success,
        "hrsa_word": syllables,
        "V": V,
        "N_D": N_D,
        "f_level": achieved.get("f_level"),
        "method": achieved.get("method"),
        "achieved_frob": achieved.get("achieved_frob"),
        "raw_json": data,
    }

    if V is not None:
        I3 = np.eye(3, dtype=np.complex128)
        result["frob_error"] = frob_norm(V - target)
        E = compute_residual(target, V)
        result["residual_E"] = E
        result["residual_norm"] = frob_norm(E - I3)
        result["unitarity_err"] = frob_norm(V @ V.conj().T - I3)
    else:
        result["frob_error"] = None
        result["residual_E"] = None
        result["residual_norm"] = None
        result["unitarity_err"] = None

    return result


# ---------------------------------------------------------------------------
# 5. JSON serialisation helper for the __main__ smoke test
# ---------------------------------------------------------------------------

def _matrix_to_jsonable(M):
    if M is None:
        return None
    return [[[float(M[i, j].real), float(M[i, j].imag)] for j in range(M.shape[1])]
            for i in range(M.shape[0])]


def _result_to_jsonable(r):
    safe = {}
    for k, v in r.items():
        if k == "raw_json":
            # Drop the verbose HRSA blob; keep just the inputs/achieved/decomposition.
            jb = v if isinstance(v, dict) else {}
            safe["raw_summary"] = {
                "inputs": jb.get("inputs"),
                "achieved": jb.get("achieved"),
                "decomposition": jb.get("decomposition"),
                "_invocation": jb.get("_invocation"),
            }
        elif isinstance(v, np.ndarray):
            safe[k] = _matrix_to_jsonable(v)
        elif isinstance(v, (np.floating, np.integer)):
            safe[k] = v.item()
        else:
            safe[k] = v
    return safe


# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------

def _smoke_test(out_path: Path,
                eps_hrsa: float = 0.3,
                max_f: int = 4,
                fallback_max_f: int = 6,
                thetas=None) -> None:
    if thetas is None:
        thetas = [math.pi / 4.0,
                  math.pi / 2.0,
                  2.0 * math.pi / 3.0,
                  math.pi]
    print(f"[hrsa_bootstrap] HRSA_tester = {HRSA_TESTER}", flush=True)
    _ensure_binary_built()

    rows = []
    residual_norms = []
    print(f"{'theta':>10} {'method':>8} {'N_D':>4} {'f':>3} {'frob(V-T)':>11} "
          f"{'|E-I|':>11} {'|VV*-I|':>11}", flush=True)

    for theta in thetas:
        target = target_R_Z(theta)
        result = bootstrap_target(target,
                                  eps_hrsa=eps_hrsa,
                                  theta=theta,
                                  max_f=max_f,
                                  fallback_max_f=fallback_max_f)
        rows.append(result)
        method = result["method"] or "-"
        N_D = result["N_D"] if result["N_D"] is not None else -1
        f_level = result["f_level"] if result["f_level"] is not None else -1
        frob_err = result["frob_error"]
        res_norm = result["residual_norm"]
        unit_err = result["unitarity_err"]
        if res_norm is not None:
            residual_norms.append(res_norm)
        print(f"{theta:>10.4f} {method:>8} {N_D:>4} {f_level:>3} "
              f"{(frob_err if frob_err is not None else float('nan')):>11.4e} "
              f"{(res_norm if res_norm is not None else float('nan')):>11.4e} "
              f"{(unit_err if unit_err is not None else float('nan')):>11.4e}",
              flush=True)

        if result["V"] is not None and unit_err is not None and unit_err > 1e-10:
            print(f"  WARNING: V is not unitary to 1e-10 at theta={theta}",
                  flush=True)

    if residual_norms:
        avg_res = sum(residual_norms) / len(residual_norms)
    else:
        avg_res = float("nan")
    print(f"[hrsa_bootstrap] average |E - I|_F across {len(residual_norms)} "
          f"successful targets: {avg_res:.4e}", flush=True)
    # SK recursion prediction: after N levels the residual norm scales as
    # res^(2^N) (up to constants).  Just report what eps would result after
    # a couple of recursion levels if the constant in factor_commutator were 1.
    if residual_norms:
        for N in (1, 2, 3, 4):
            print(f"  predicted residual after {N} SK level(s): "
                  f"{avg_res ** (2 ** N):.4e}", flush=True)

    summary = {
        "eps_hrsa": eps_hrsa,
        "max_f": max_f,
        "fallback_max_f": fallback_max_f,
        "results": [_result_to_jsonable(r) for r in rows],
        "avg_residual_norm": avg_res,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[hrsa_bootstrap] wrote results to {out_path}", flush=True)


# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="HRSA bootstrap adapter for SK.")
    p.add_argument("--eps", type=float, default=0.3,
                   help="loose HRSA epsilon (default 0.3)")
    p.add_argument("--max-f", type=int, default=4,
                   help="initial HRSA max_f (default 4)")
    p.add_argument("--fallback-max-f", type=int, default=6,
                   help="fallback HRSA max_f on miss (default 6)")
    p.add_argument("--out", type=Path,
                   default=Path("/tmp/hrsa_bootstrap_results.json"))
    p.add_argument("--theta", action="append", type=float,
                   help="run a specific theta (repeatable); default = "
                        "[pi/4, pi/2, 2pi/3, pi].")
    args = p.parse_args()
    _smoke_test(out_path=args.out,
                eps_hrsa=args.eps,
                max_f=args.max_f,
                fallback_max_f=args.fallback_max_f,
                thetas=args.theta)


if __name__ == "__main__":
    _cli()
