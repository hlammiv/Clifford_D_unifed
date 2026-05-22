#!/usr/bin/env python3
"""verify_conventions.py — sanity-check the unified theta convention across the
three qutrit Clifford+D backends.

Canonical convention (per the in-tree paper, ESA_CliffordD_Notes.tex §3):

    Input theta  ==>  target gate is  R^Z_{(0,1)}(theta)
                  =  Diag(e^{-i theta/2}, e^{+i theta/2}, 1).

Each backend produces a 3x3 unitary V over Z[zeta_9, 1/3] that approximates
this target. The three backends use different internal conventions, so the
purpose of this script is to confirm that — after the 2026-05 fixes — they
all converge on the SAME target matrix when given the SAME theta.

Tests:

  HRSA      : run unified/hrsa/HRSA_tester for several thetas (Phase 2 finds
              fast diagonal solutions); grep "passes: YES" from the output.
  ESA       : run unified/esa/ESA_convention_test, which exercises the
              candidate-filter step directly (millisecond runtime); grep
              "overall=PASS" from the output.
  zeta9     : analytical check — construct the target vector exactly as
              plot_eps_vs_theta.py builds it, apply the same Householder
              construction try.py uses (V = X_(0,1) * (I - conj(u) ⊗ u)),
              and confirm V == canonical target Diag(e^{-i theta/2},
              e^{+i theta/2}, 1) within 1e-12.

Usage: python3 verify_conventions.py
       (no arguments; runs a fixed sweep of test angles)
"""

import math
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def canonical_target(theta):
    """Diag(e^{-i theta/2}, e^{+i theta/2}, 1)."""
    M = np.zeros((3, 3), dtype=np.complex128)
    M[0, 0] = np.exp(-0.5j * theta)
    M[1, 1] = np.exp(+0.5j * theta)
    M[2, 2] = 1.0
    return M


# ---------------------------------------------------------------------------
# zeta9: analytical convention check (no external process)
# ---------------------------------------------------------------------------
def check_zeta9_analytic(theta, tol=1e-12):
    """Build the target vector u as plot_eps_vs_theta.py does, then form
    V = X_(0,1) * (I - conj(u) ⊗ u) per try.py:MatrixDinC.  Verify V equals
    the canonical target within tol."""
    norm_sq = 2  # zeta9 uses --norm 2 for Householder
    scale = (norm_sq / 2) ** 0.5
    u = np.array([np.exp(-0.5j * theta) * scale, -1.0 * scale, 0.0],
                 dtype=np.complex128)

    X01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.complex128)
    I3 = np.eye(3, dtype=np.complex128)
    outer = np.outer(np.conj(u), u)         # try.py:MatrixDinC convention
    V = X01 @ (I3 - outer)

    M = canonical_target(theta)
    diff = np.linalg.norm(V - M, "fro")
    return (diff < tol, diff, V)


# ---------------------------------------------------------------------------
# HRSA: spawn the C++ binary, parse output
# ---------------------------------------------------------------------------
def check_hrsa(theta, epsilon=0.05, max_f=3, no_direct=True, timeout=60):
    """Run HRSA_tester and check the Frobenius distance line."""
    binary = os.path.join(ROOT, "hrsa", "HRSA_tester")
    if not os.path.isfile(binary):
        return ("MISSING_BINARY", float("nan"), "build hrsa first")
    cmd = [binary, str(theta), str(epsilon), str(max_f), "1.0"]
    if no_direct:
        cmd.append("--no-direct")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd=os.path.join(ROOT, "hrsa"))
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", float("nan"), "")

    text = out.stdout + out.stderr
    # The matrixFrobeniusCheck function (called only on the HRSA Phase 3 path)
    # prints the distance line; the diagSearch path prints its own.
    # Both formats end with "passes: YES" or "passes: NO" / nothing.
    m = re.search(r"Matrix Frobenius distance.*?=\s*([\d.eE+-]+).*?passes:\s*(\w+)",
                  text, re.DOTALL)
    if m:
        dist = float(m.group(1))
        verdict = m.group(2)
        return (verdict, dist, "")

    # diagSearch / Phase 2 fallback: look for "Frobenius =" and check the
    # ε-comparison ourselves.
    m = re.search(r"best Frobenius so far\s*=\s*([\d.eE+-]+)", text)
    if m:
        dist = float(m.group(1))
        verdict = "YES" if dist < epsilon else "NO"
        return (verdict, dist, "(via Phase 2 best-Frobenius)")

    return ("NO_PARSE", float("nan"), text[-400:])


# ---------------------------------------------------------------------------
# ESA: spawn the C++ binary, parse output
# ---------------------------------------------------------------------------
def check_esa(theta, epsilon=0.5, f=1, timeout=15):
    """Run ESA_convention_test and parse the overall=PASS|FAIL line."""
    binary = os.path.join(ROOT, "esa", "ESA_convention_test")
    if not os.path.isfile(binary):
        return ("MISSING_BINARY", float("nan"), "build esa first")
    cmd = [binary, str(theta), str(epsilon), str(f)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd=os.path.join(ROOT, "esa"))
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", float("nan"), "")

    text = out.stdout + out.stderr
    m = re.search(r"overall=(\w+)", text)
    if not m:
        return ("NO_PARSE", float("nan"), text[-400:])

    # Worst per-entry distance
    dists = [float(d) for d in re.findall(r"dist=([\d.eE+-]+)\s+cands_size", text)]
    worst = max(dists) if dists else float("nan")
    return (m.group(1), worst, "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_angles = [0.0, 0.5, math.pi / 4, math.pi / 3, 1.5]

    print("=" * 78)
    print("Convention verification: input theta should map to the canonical target")
    print("    R^Z_{(0,1)}(theta) = Diag(e^{-i theta/2}, e^{+i theta/2}, 1)")
    print("=" * 78)

    overall_pass = True

    # --- zeta9 (analytic) ---
    print("\n[zeta9] Analytic target-construction check (no run, just math):")
    for th in test_angles:
        ok, diff, _ = check_zeta9_analytic(th)
        tag = "PASS" if ok else "FAIL"
        print(f"  theta={th:.6f}  ||V_zeta9 - canonical||_F = {diff:.2e}  {tag}")
        if not ok:
            overall_pass = False

    # --- HRSA (run binary) ---
    print("\n[HRSA] Run unified/hrsa/HRSA_tester (Phase 2 / Phase 3 paths):")
    for th in test_angles:
        verdict, dist, note = check_hrsa(th)
        passed = verdict == "YES"
        tag = "PASS" if passed else f"FAIL ({verdict})"
        print(f"  theta={th:.6f}  Frobenius dist = {dist:.4e}  {tag}  {note}")
        if not passed:
            overall_pass = False

    # --- ESA (run candidate-filter test) ---
    print("\n[ESA] Run unified/esa/ESA_convention_test (candidate filter):")
    for th in test_angles:
        verdict, worst, note = check_esa(th)
        passed = verdict == "PASS"
        tag = "PASS" if passed else f"FAIL ({verdict})"
        print(f"  theta={th:.6f}  worst per-entry dist = {worst:.4e}  {tag}  {note}")
        if not passed:
            overall_pass = False

    print()
    print("=" * 78)
    print(f"Overall: {'PASS — all backends agree on canonical convention.' if overall_pass else 'FAIL — see above.'}")
    print("=" * 78)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
