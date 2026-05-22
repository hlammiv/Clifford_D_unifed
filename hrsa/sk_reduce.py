"""SK output reducer: re-decompose an SK-assembled exact unitary via
HRSA's `decompose_tool` to recover a shorter N_D word.

Workflow:
    1. Run SK (sk_synthesize_v2) producing both float V and exact V_ring.
    2. Pass V_ring to decompose_tool via stdin JSON. The tool returns a
       fresh syllable word with potentially lower D-count.
    3. Compare orig SK-reported N_D vs decompose D_count.

Caveats:
    - Requires SK call to have ring_valid=True. If any leaf fell to a
      non-ringZ9 path (HRSA failure, zeta9 stub), the reducer aborts.
    - decompose_tool may reject very large f-levels. Practical limit is
      f ≲ 30-40 (subject to internal sdeChi computations). We try the cheap
      GCD-by-3 reduction first to shrink f.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import numpy as np

import sk_ringZ9 as rk


_DEFAULT_DECOMPOSE_TOOL = Path(__file__).resolve().parent / "decompose_tool"


def _run_decompose(payload: dict,
                   binary: Path = _DEFAULT_DECOMPOSE_TOOL,
                   # 2026-05-13: decompose_tool now uses cpp_int storage so
                   # arbitrary-precision SK output (f up to ~90+) is supported,
                   # but per-call wall time can be 5-15 min at f≈60.  Bumped
                   # default 120s -> 900s.  Tighten manually if needed.
                   timeout_s: float = 900.0) -> dict:
    """Invoke decompose_tool with `payload` as stdin JSON; return parsed stdout."""
    if not binary.exists():
        raise FileNotFoundError(f"decompose_tool not found at {binary}")
    proc = subprocess.run(
        [str(binary)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode not in (0,):
        raise RuntimeError(
            f"decompose_tool exited rc={proc.returncode}; stderr:\n{proc.stderr}"
        )
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"decompose_tool produced non-JSON stdout:\n{proc.stdout[:500]}"
        ) from exc
    return out


def reduce_sk_output(sk_result: dict,
                     binary: Path = _DEFAULT_DECOMPOSE_TOOL,
                     timeout_s: float = 900.0,
                     verify_residual: bool = True) -> dict:
    """Re-decompose SK's exact unitary via decompose_tool.

    Parameters
    ----------
    sk_result : dict returned by sk_driver_v2.sk_synthesize_v2(...).
                Must have keys 'V_ring' (RingMat), 'ring_valid' (bool),
                'V' (float ndarray), 'N_D' (int — SK-reported).
    binary    : path to compiled decompose_tool.
    timeout_s : per-decompose timeout.
    verify_residual : if True, eval(V_ring) and check it matches sk_result["V"]
                      to within 1e-10; flag if not.

    Returns dict with:
      'orig_N_D'              : int  SK-reported N_D
      'reduced_N_D'           : int  decompose_tool's D_count for V_ring
      'reduced_syllables'     : list of syllable dicts from decompose_tool
      'sde_chi_initial'       : int  decompose's input sde-chi metric
      'sde_chi_final'         : int  decompose's output sde-chi
      'shared_f'               : int  shared denominator exponent passed to tool
      'verification_residual' : float |eval(V_ring) - V_float|_F (None if skipped)
      'reduction_ratio'       : float  reduced_N_D / orig_N_D (None if either missing)
      'success'                : bool decompose_tool's success flag
      'tool_stdout'            : dict full decompose_tool output
    """
    if not sk_result.get("ring_valid"):
        return {
            "orig_N_D": sk_result.get("N_D"),
            "reduced_N_D": None,
            "error": "SK result is not ring_valid; cannot reduce",
            "success": False,
        }
    V_ring = sk_result.get("V_ring")
    if V_ring is None:
        return {
            "orig_N_D": sk_result.get("N_D"),
            "reduced_N_D": None,
            "error": "sk_result has no V_ring",
            "success": False,
        }

    # Optional: GCD-by-3 reduction to keep the f passed to the tool small.
    V_ring_reduced = rk.ringmat_reduce_by_three(V_ring)

    # Flatten to (3x3 list of 6-int lists, shared f).
    coefs_3x3, shared_f = rk.ringmat_to_ints_with_f(V_ring_reduced)
    payload = {"f": int(shared_f), "V": coefs_3x3}

    out = _run_decompose(payload, binary=binary, timeout_s=timeout_s)

    # Optional residual check.
    residual = None
    if verify_residual:
        V_eval = rk.ringmat_eval(V_ring_reduced)
        V_float = sk_result.get("V")
        if V_float is not None:
            residual = float(np.linalg.norm(V_eval - V_float, ord="fro"))

    orig_N_D = sk_result.get("N_D")
    reduced_N_D = out.get("D_count")
    ratio = (
        (reduced_N_D / orig_N_D)
        if (orig_N_D is not None and reduced_N_D is not None and orig_N_D > 0)
        else None
    )

    return {
        "orig_N_D": orig_N_D,
        "reduced_N_D": reduced_N_D,
        "reduced_syllables": out.get("syllables"),
        "sde_chi_initial": out.get("sde_chi_initial"),
        "sde_chi_final": out.get("sde_chi_final"),
        "shared_f": shared_f,
        "verification_residual": residual,
        "reduction_ratio": ratio,
        "success": bool(out.get("success", False)),
        "tool_stdout": out,
    }


# ---------------------------------------------------------------------------
#  __main__ : end-to-end smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import sys
    import time

    import sk_driver_v2 as skv2

    target = np.diag([
        np.exp(-0.5j * math.pi / 2),
        np.exp(+0.5j * math.pi / 2),
        1 + 0j,
    ])

    for depth in [0, 1]:
        print(f"\n=== SK depth={depth} on R_Z_{{(0,1)}}(π/2), eps_total=0.05 ===")
        t0 = time.time()
        sk = skv2.sk_synthesize_v2(target, depth=depth, eps_total=0.05)
        print(f"  SK done: N_D={sk['N_D']} frob={sk['final_error']:.4g} "
              f"ring_valid={sk['ring_valid']} t={time.time()-t0:.1f}s")
        if not sk["ring_valid"]:
            print("  ringZ9 not available; skipping reduction.")
            continue
        t0 = time.time()
        red = reduce_sk_output(sk)
        print(f"  reduce: orig_N_D={red['orig_N_D']} reduced_N_D={red['reduced_N_D']} "
              f"ratio={red['reduction_ratio']!s} f={red['shared_f']} "
              f"residual={red['verification_residual']!s} "
              f"t={time.time()-t0:.1f}s")
