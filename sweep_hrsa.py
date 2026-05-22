#!/usr/bin/env python3
# sweep_hrsa.py — HRSA accuracy sweep over 14 angles × several epsilons.
#
# For each (theta, epsilon), runs HRSA_tester with V-denom max=4 (CLI max_f=2),
# captures JSON output, runs v_validate independently, records results.
#
# Outputs CSV to stdout and writes individual JSON files under sweep_out/.

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

UNIFIED = Path(__file__).resolve().parent
HRSA_TESTER = UNIFIED / "hrsa" / "HRSA_tester"
V_VALIDATE = UNIFIED / "v_validate.py"
OUT_DIR = Path(os.environ.get("SWEEP_OUT_DIR", UNIFIED / "sweep_out"))

# 14 angles spanning (0, π], hitting some Clifford-special and generic values.
ANGLES = [
    ("0",         0.0),
    ("pi/12",     math.pi / 12),
    ("pi/9",      math.pi / 9),
    ("pi/6",      math.pi / 6),
    ("pi/4",      math.pi / 4),
    ("pi/3",      math.pi / 3),
    ("5pi/12",    5 * math.pi / 12),
    ("pi/2",      math.pi / 2),
    ("7pi/12",    7 * math.pi / 12),
    ("2pi/3",     2 * math.pi / 3),
    ("3pi/4",     3 * math.pi / 4),
    ("5pi/6",     5 * math.pi / 6),
    ("11pi/12",   11 * math.pi / 12),
    ("pi",        math.pi),
]
EPSILONS = [0.1, 0.05, 0.01, 0.001]   # all > 1e-4
HRSA_CLI_F = int(os.environ.get("HRSA_CLI_F", "4"))   # u-denom cap (V-denom up to 2*this)
PER_RUN_TIMEOUT = 1800                  # 30 min per run

OUT_DIR.mkdir(exist_ok=True)


def run_one(label, theta, eps):
    json_path = OUT_DIR / f"hrsa_{label.replace('/', '_')}_eps{eps:g}.json"
    log_path = OUT_DIR / f"hrsa_{label.replace('/', '_')}_eps{eps:g}.log"
    cmd = [str(HRSA_TESTER), repr(theta), str(eps), str(HRSA_CLI_F),
           "--json", str(json_path)]
    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                           timeout=PER_RUN_TIMEOUT)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = time.time() - t0

    result = {
        "label": label, "theta": theta, "eps": eps, "wall": wall,
        "timed_out": timed_out,
        "success": False, "method": None, "frob": None, "f_level": None,
        "N_D": None, "validate_pass": None, "actual_sde_3": None,
    }

    if json_path.exists():
        try:
            d = json.loads(json_path.read_text())
            ach = d.get("achieved", {})
            result["success"] = bool(ach.get("success", False))
            result["method"] = ach.get("method")
            result["frob"] = ach.get("achieved_frob")
            result["f_level"] = ach.get("f_level")
            result["N_D"] = (d.get("decomposition") or {}).get("N_D")
        except Exception as e:
            result["error"] = f"json parse: {e}"

        # Independent validation
        vp = subprocess.run([str(V_VALIDATE), str(json_path)],
                            capture_output=True, text=True)
        if vp.stdout:
            try:
                vd = json.loads(vp.stdout)
                result["validate_pass"] = bool(vd.get("all_checks_pass", False))
                result["actual_sde_3"] = vd.get("actual_sde_3")
            except Exception:
                result["validate_pass"] = False
        else:
            result["validate_pass"] = False

    return result


def main():
    results = []
    print("# HRSA sweep: 14 angles × {} epsilons = {} runs".format(
        len(EPSILONS), len(ANGLES) * len(EPSILONS)))
    print("# CLI max_f={} (u-denom; V-denom up to {})".format(HRSA_CLI_F, 2 * HRSA_CLI_F))
    print()
    print("label,theta,eps,success,method,frob,f_level,N_D,wall,validate_pass,actual_sde_3,timed_out")
    sys.stdout.flush()

    for label, theta in ANGLES:
        for eps in EPSILONS:
            r = run_one(label, theta, eps)
            results.append(r)
            print("{},{:.6f},{:g},{},{},{},{},{},{:.2f},{},{},{}".format(
                r["label"], r["theta"], r["eps"],
                r["success"], r["method"], r["frob"], r["f_level"], r["N_D"],
                r["wall"], r["validate_pass"], r["actual_sde_3"], r["timed_out"]))
            sys.stdout.flush()

    # Summary
    print()
    print("# === Summary ===")
    n_success = sum(1 for r in results if r["success"])
    n_validate_pass = sum(1 for r in results if r["validate_pass"])
    n_validate_fail = sum(1 for r in results if r["success"] and not r["validate_pass"])
    n_timeout = sum(1 for r in results if r["timed_out"])
    print(f"# {n_success}/{len(results)} HRSA reported success")
    print(f"# {n_validate_pass}/{len(results)} v_validate confirmed")
    print(f"# {n_validate_fail} cases where HRSA reported success but v_validate FAILED")
    print(f"# {n_timeout} timed out (> {PER_RUN_TIMEOUT}s)")

    # Write JSON summary alongside
    with open(OUT_DIR / "sweep_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
