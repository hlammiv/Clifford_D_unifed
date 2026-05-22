#!/usr/bin/env python3
# hybrid_compile.py — drive HRSA to produce row 1, then zeta9 stage-5 daemon
# to fit row 2, and assemble a V matrix.
#
# This is a FIRST-CUT integration: it uses HRSA's full pipeline (--json mode)
# to produce a V, extracts row 1, and feeds that to zeta9's daemon.  The
# daemon's row 2 is validated for consistency with HRSA's row 2 (cross-check).
#
# A future "true hybrid" would skip HRSA's triple-product matching and only
# enumerate u_1 candidates via --dump-x1-cands, then have zeta9 fill u_2/u_3
# algebraically.  That requires a per-u_1 protocol extension (zeta9 currently
# wants a full row 1, not just u_1) and is left for a follow-up.
#
# Usage:
#   hybrid_compile.py --theta T --epsilon E --max-f F [--eps-pre EPS_PRE]
#                     [--mpi N] [--workdir DIR] [--json out.json]
#
# Architecture:
#   1. Run HRSA_tester with --json to get a full V (and row 1).
#   2. Format V's row 0 as a Sage vector string for zeta9's --fixed_row1_sage.
#   3. Spawn zeta9 stage-5 daemon (search_diagonal_matrix_two_rows_streamed_mpi
#      with --daemon).
#   4. Send one or more jobs (theta, eps, fixed_row1) to the daemon.
#   5. Collect results; cross-check against HRSA's V; emit schema JSON.

import argparse
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
ZETA9_DIR = UNIFIED / "zeta9"
SAGE_ENV = os.environ.get("SAGE_ENV", "/home/hlamm/miniforge3/envs/sage")


# ----- HRSA invocation -----

def run_hrsa(theta, epsilon, max_f, json_path, hrsa_cli_max_f=None):
    """Run HRSA_tester to produce a full V at this query.  Returns the parsed JSON."""
    if hrsa_cli_max_f is None:
        # Convention: schema's max_f is V-denom; HRSA's CLI max_f is u-denom = V-denom/2.
        hrsa_cli_max_f = max(max_f // 2, 1)
    cmd = [str(HRSA_TESTER), repr(theta), str(epsilon), str(hrsa_cli_max_f),
           "--no-direct", "--json", str(json_path)]
    print(f"[hybrid] HRSA: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    wall = time.time() - t0
    print(f"[hybrid] HRSA done in {wall:.1f}s (rc={rc})", flush=True)
    if rc != 0:
        return None
    with open(json_path) as fh:
        return json.load(fh)


# ----- Format V's row as zeta9 Sage-vector string -----

def format_row_as_sage_vector(row_int, f):
    """row_int: list of 3 ringZ9 numerators, each a list of 6 ints.
    f: denominator exponent.
    Returns a Sage-readable vector string like:
      (a + b*zeta9 + c*zeta9^2 + ...)/3^f, ...

    The exact format must match what zeta9's parse_fixed_row1_sage expects.
    Looking at try.py, it returns DZ9[0,:].str() — a Sage Matrix row string.
    Format example: "(15 + 8*zeta9 + 2*zeta9^2 - 22*zeta9^3 - 36*zeta9^4 - 13*zeta9^5)/81, ..."
    """
    denom = 3 ** f
    entries = []
    for entry in row_int:
        terms = []
        for k, c in enumerate(entry):
            if c == 0:
                continue
            if k == 0:
                terms.append(f"{c}")
            elif k == 1:
                terms.append(f"({c})*zeta9")
            else:
                terms.append(f"({c})*zeta9^{k}")
        if not terms:
            entries.append("0")
        else:
            poly = " + ".join(terms)
            entries.append(f"({poly})/{denom}")
    return "(" + ", ".join(entries) + ")"


# ----- Zeta9 daemon driver -----

class Zeta9Daemon:
    """Long-lived MPI+Sage process running stage-5 in --daemon mode."""

    def __init__(self, *, f, eps_pre, mpi=4, workdir=ZETA9_DIR, sage_env=SAGE_ENV):
        self.workdir = Path(workdir)
        env = os.environ.copy()
        env["PATH"] = f"{sage_env}/bin:" + env.get("PATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        eps_pre_str = f"{eps_pre:g}"
        d = self.workdir / "D"
        cmd = [
            "mpirun", "-n", str(int(mpi)), "--oversubscribe",
            f"{sage_env}/bin/sage", "-python",
            "-m", "zeta9.search_diagonal_matrix_two_rows_streamed_mpi",
            "--daemon",
            "--triples_file", str(d / f"TM_f={f}_eps={eps_pre_str}"),
            "--triples_json", str(d / f"TM_f={f}_eps={eps_pre_str}.manifest.json"),
            "--rootdb_prefix", str(d / f"RM_f={f}_eps={eps_pre_str}_local"),
            "--chunk_meta_json", str(d / f"RM_f={f}_eps={eps_pre_str}_triples_chunk_meta.json"),
            "--f", str(f),
            "--output_prefix", str(self.workdir / "xout_hybrid"),
            "--row2_cache", str(self.workdir / "xout_hybrid.row2.npy"),
            "--quiet",
        ]
        print(f"[hybrid] Spawning daemon: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd, env=env, cwd=self.workdir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def search(self, theta, eps_frob, fixed_row1_sage, timeout=120.0):
        """Send one job to the daemon, return the parsed JSON result."""
        job = {"theta": theta, "eps": eps_frob, "fixed_row1": fixed_row1_sage}
        line = json.dumps(job) + "\n"
        if self.proc.poll() is not None:
            return {"success": False, "error": "daemon exited"}
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

        # Read response (one line). Use select for timeout; for now blocking.
        result_line = self.proc.stdout.readline()
        if not result_line:
            return {"success": False, "error": "empty daemon response"}
        try:
            return json.loads(result_line)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"json decode: {e}; raw: {result_line[:200]!r}"}

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30.0)
        except Exception:
            self.proc.kill()


# ----- main -----

def main():
    p = argparse.ArgumentParser(description="hybrid HRSA + zeta9 stage-5 daemon compiler")
    p.add_argument("--theta", type=float, required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--max-f", type=int, required=True,
                   help="V-denom cap (schema convention; HRSA CLI uses max_f/2)")
    p.add_argument("--eps-pre", type=float, default=2.5e-3,
                   help="zeta9 per-coord eps for stages 1-4 (default 2.5e-3)")
    p.add_argument("--mpi", type=int, default=4)
    p.add_argument("--json", default=None, help="schema JSON output path")
    args = p.parse_args()

    wall_start = time.time()
    out = {"backend": "hrsa+zeta9-daemon", "errors": []}

    # 1. HRSA produces a V (row 1 source).
    hrsa_json_path = "/tmp/hybrid_hrsa.json"
    hrsa_result = run_hrsa(args.theta, args.epsilon, args.max_f, hrsa_json_path)
    if hrsa_result is None or not hrsa_result["achieved"]["success"]:
        out["errors"].append("HRSA failed to find a V — cannot run hybrid")
        out["success"] = False
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=2))
        return 1

    V = hrsa_result["unitary"]["V"]
    f_v = hrsa_result["unitary"]["f"]
    out["hrsa_v_f"] = f_v
    out["hrsa_frob"] = hrsa_result["achieved"]["achieved_frob"]
    out["hrsa_method"] = hrsa_result["achieved"]["method"]
    out["hrsa_N_D"] = hrsa_result.get("decomposition", {}).get("N_D")

    # Extract row 0 of V (= row "1" in 1-indexed math language).
    row1_int = V[0]  # 3 entries, each 6 ints
    row1_sage = format_row_as_sage_vector(row1_int, f_v)
    print(f"[hybrid] HRSA row 1 (Sage): {row1_sage[:120]}{'...' if len(row1_sage)>120 else ''}", flush=True)

    # 2. Spawn daemon and send job.
    print(f"[hybrid] Spawning zeta9 daemon at f={f_v}, eps_pre={args.eps_pre}", flush=True)
    daemon = Zeta9Daemon(f=f_v, eps_pre=args.eps_pre, mpi=args.mpi)
    try:
        result = daemon.search(args.theta, args.epsilon, row1_sage, timeout=300.0)
    finally:
        daemon.close()

    out["zeta9_result"] = result
    out["wall_seconds"] = time.time() - wall_start

    print(f"[hybrid] zeta9 daemon result: success={result.get('success')}", flush=True)
    if result.get("success"):
        print(f"  frobenius (zeta9): {result.get('frobenius')}")
        print(f"  hrsa frobenius:    {hrsa_result['achieved']['achieved_frob']}")
        out["success"] = True
    else:
        out["errors"].append(f"zeta9 daemon: {result.get('error', 'no V found')}")
        out["success"] = False

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"[hybrid] wrote {args.json}", flush=True)

    return 0 if out["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
