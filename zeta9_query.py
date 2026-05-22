#!/usr/bin/env python3
"""zeta9_query.py — run zeta9 stage 5 against a previously-built cache.

This is the per-(θ, ε) query stage. Assumes stages 1-4 cache has been built
already by `zeta9_precompute.py` at the same (max-f, eps-pre, mode).

The query ε can be ANY value ≤ eps-pre (the build-time loose ε). Stage 5
filters at runtime via |a| ≤ eps and |prod - target| ≤ eps; the cache may
contain extra Y values that don't satisfy the tighter constraint, but
those are simply rejected by the runtime filter.

Concrete workflow:
    # 1. Build cache once
    zeta9_precompute.py --max-f 2 --eps-pre 0.05 --mode householder --mpi 16

    # 2. Many cheap queries against it
    zeta9_query.py --max-f 2 --eps-pre 0.05 --theta 1.5707963 --epsilon 0.001 \\
                   --mode householder --json /tmp/cell.json
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeta9_compile as zc  # noqa: E402
from zeta9_compile import (  # noqa: E402
    DEFAULT_SAGE_ENV,
    ZETA9_DIR,
    artifact_paths,
    emit_schema_json,
    extract_best_v,
    run_stage5,
)


def main():
    p = argparse.ArgumentParser(
        description="zeta9 stage-5 query against a precomputed cache."
    )
    p.add_argument("--max-f", type=int, required=True,
                   help="u-denominator cap (must match the cache's max-f)")
    p.add_argument("--eps-pre", type=float, required=True,
                   help="cache's eps_pre — used to locate the cache files. "
                        "Must match what was passed to zeta9_precompute.py.")
    p.add_argument("--theta", type=float, required=True)
    p.add_argument("--epsilon", type=float, required=True,
                   help="QUERY Frobenius epsilon. Can be any value ≤ eps-pre.")
    p.add_argument("--mode", choices=["diagonal", "householder"], default="householder")
    p.add_argument("--mpi", type=int, default=4)
    p.add_argument("--workdir", default=str(ZETA9_DIR))
    p.add_argument("--sage-env", default=DEFAULT_SAGE_ENV)
    p.add_argument("--json", default=None, help="schema JSON output path")
    p.add_argument("--check-local-p3k", dest="check_local_p3k",
                   action="store_true", default=False,
                   help="Match the cache's check_local_p3k setting.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        sys.exit(f"workdir not found: {workdir}")
    if not Path(args.sage_env).exists():
        sys.exit(f"sage env not found: {args.sage_env}")

    if args.epsilon > args.eps_pre:
        sys.exit(f"--epsilon ({args.epsilon}) > --eps-pre ({args.eps_pre}); "
                 f"the cache is built for eps_pre and only supports tighter "
                 f"queries. Either rebuild the cache at a looser eps_pre, or "
                 f"tighten your query.")

    f_u = args.max_f
    f_v = 2 * f_u

    arts = artifact_paths(workdir, f_v, args.eps_pre,
                          check_local_p3k=args.check_local_p3k,
                          mode=args.mode)

    # Verify the cache is present.
    required = [arts["stage2"], arts["stage3_prefix"]]
    missing = [str(r) for r in required if not Path(str(r) + ".roots.json").exists()
               and not Path(str(r)).exists()]
    if missing:
        sys.exit(
            f"cache files missing: {missing}\n"
            f"Run zeta9_precompute.py --max-f {f_u} --eps-pre {args.eps_pre} "
            f"--mode {args.mode} first."
        )

    print(f"[zeta9_query] theta={args.theta} epsilon={args.epsilon} "
          f"eps_pre(cache)={args.eps_pre} max_f(u)={f_u} mode={args.mode}")

    t0 = time.time()
    output_prefix = workdir / "xout"
    stage5_t = run_stage5(workdir, f_v, args.eps_pre, args.epsilon, args.theta,
                          args.mpi, args.sage_env, output_prefix, args.dry_run,
                          check_local_p3k=args.check_local_p3k,
                          mode=args.mode)

    V_int, v_f, frob = (None, None, None)
    if not args.dry_run:
        xout = workdir / "xout.npz"
        if xout.exists():
            V_int, v_f, frob = extract_best_v(xout, f_u)

    success = V_int is not None and frob is not None and frob < args.epsilon
    wall = time.time() - t0

    if args.json:
        emit_schema_json(args.json,
            theta=args.theta, epsilon=args.epsilon, max_f=f_u,
            V_int=V_int, v_f=v_f, achieved_frob=frob, success=success,
            wall_seconds=wall, mpi_ranks=args.mpi,
            command_line=sys.argv)

    print(f"[zeta9_query] success={success} frob={frob} stage5={stage5_t:.1f}s "
          f"total={wall:.1f}s")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
