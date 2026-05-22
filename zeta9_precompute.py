#!/usr/bin/env python3
"""zeta9_precompute.py — build zeta9's stages 1-4 cache once, query many.

Stages 1-4 are theta-independent and only stage 1 uses ε (as an upper bound
on the Y-value enumeration). Building the cache at the LOOSEST ε you'll ever
query produces a SUPERSET cache that any tighter-ε query can use via stage 5's
runtime filter. (Stages 2/3/4 don't take --eps at all; their algorithms are
pure functions of the upstream data + f.)

Concrete workflow:
    # Build cache once at f=4 with eps=0.05 (paper's loosest ε for that f-level)
    zeta9_precompute.py --max-f 2 --eps-pre 0.05 --mode householder --mpi 16

    # Then run many tight queries against the same cache
    zeta9_query.py --max-f 2 --eps-pre 0.05 --theta 1.5707 --epsilon 0.001 \\
                   --mode householder --json /tmp/cell_pi2_eps0.001.json
    # ... etc.

Cost:
  - One 30 min - few hour precompute pays for many ~1 min queries.
  - Cache size grows with eps_pre: pick the loosest you'll need, no looser.

Cache lifetime:
  - Cache is keyed on (f_v=2·max-f, eps_pre, mode). Different modes do not
    collide. Same mode at the same (f, eps) → reuse.
  - Files live under ${workdir}/D/. With the Lenore symlink, this is the
    1.8 TB secondary drive.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Reuse the proven implementations from the single-shot wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeta9_compile import (  # noqa: E402
    DEFAULT_SAGE_ENV,
    ZETA9_DIR,
    artifact_paths,
    ensure_precompute,
)


def main():
    p = argparse.ArgumentParser(
        description="Build zeta9 stages 1-4 cache (theta-independent precompute)."
    )
    p.add_argument("--max-f", type=int, required=True,
                   help="u-denominator cap (V denom = 2*max_f)")
    p.add_argument("--eps-pre", type=float, required=True,
                   help="per-coord epsilon for stage 1's Y-enumeration upper "
                        "bound. Pick the LOOSEST ε you intend to query against "
                        "this cache; tighter ε queries will filter at runtime.")
    p.add_argument("--mode", choices=["diagonal", "householder"], default="householder",
                   help="Synthesis mode tagging cache paths (default householder).")
    p.add_argument("--mpi", type=int, default=4,
                   help="MPI ranks for stages 1, 2 (and stage 4 if not patched). "
                        "Defaults to 4; use 16-24 on Lenore for full throughput.")
    p.add_argument("--workdir", default=str(ZETA9_DIR))
    p.add_argument("--sage-env", default=DEFAULT_SAGE_ENV)
    p.add_argument("--check-local-p3k", dest="check_local_p3k",
                   action="store_true", default=False,
                   help="Enable upstream p3k norm screen in stage 1.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        sys.exit(f"workdir not found: {workdir}")
    if not Path(args.sage_env).exists():
        sys.exit(f"sage env not found: {args.sage_env}")

    f_u = args.max_f
    f_v = 2 * f_u  # zeta9 internal --f is V-denom

    arts = artifact_paths(workdir, f_v, args.eps_pre,
                          check_local_p3k=args.check_local_p3k,
                          mode=args.mode)
    print(f"[zeta9_precompute] mode={args.mode} max_f(u)={f_u} f_v={f_v} "
          f"eps_pre={args.eps_pre} mpi={args.mpi} check_local_p3k={args.check_local_p3k}")
    print(f"[zeta9_precompute] cache prefix: {arts['stage2']}")

    t0 = time.time()
    times = ensure_precompute(workdir, f_v, args.eps_pre, args.mpi, args.sage_env,
                              args.dry_run,
                              check_local_p3k=args.check_local_p3k,
                              mode=args.mode)
    wall = time.time() - t0

    print(f"\n[zeta9_precompute] === precompute complete ===")
    for k, v in times.items():
        print(f"  {k:<12}: {v:.1f}s")
    print(f"  TOTAL       : {wall:.1f}s")
    print()
    print(f"[zeta9_precompute] Cache ready at f={f_v}, eps_pre={args.eps_pre}, "
          f"mode={args.mode}.")
    print(f"[zeta9_precompute] Run zeta9_query.py with the same (max-f, eps-pre, "
          f"mode) to query.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
