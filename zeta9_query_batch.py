#!/usr/bin/env python3
"""zeta9_query_batch.py — run zeta9 stage 5 against MANY (theta, eps) queries
in a single mpirun invocation, amortizing Python startup, mpirun spawn, and
the triple-streaming overhead across all queries.

Companion to ``zeta9_query.py`` (which does ONE query per invocation). Use
this when you want the same precomputed cache hit by dozens or hundreds of
theta values — e.g., a paper-grid sweep at fixed epsilon.

V1 constraint: all queries must share the SAME ``eps``. The per-triple
root-enumeration caches inside stage 5 are eps-dependent; sharing eps lets
the inner loop visit each triple once and apply each query's phase filter
without re-enumerating roots. (Lifting this is a future round — e.g., cache
roots at max(eps) and filter per-query.)

Usage:

    # Build queries JSON
    cat > /tmp/q.json <<EOF
    [
      {"id": "00", "theta": 0.0,         "eps": 0.1},
      {"id": "01", "theta": 0.06283185,  "eps": 0.1},
      ...
    ]
    EOF

    zeta9_query_batch.py --max-f 1 --eps-pre 0.1 \\
        --queries /tmp/q.json --output-dir /tmp/batch_out \\
        --mode householder --mpi 4
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeta9_compile import (  # noqa: E402
    DEFAULT_SAGE_ENV,
    ZETA9_DIR,
    artifact_paths,
    stages_env,
)


def main():
    p = argparse.ArgumentParser(
        description="Batched zeta9 stage-5 query against a precomputed cache."
    )
    p.add_argument("--max-f", type=int, required=True,
                   help="u-denominator cap (must match the cache's max-f).")
    p.add_argument("--eps-pre", type=float, required=True,
                   help="cache's eps_pre — used to locate the cache files.")
    p.add_argument("--queries", required=True,
                   help="JSON file with [{id, theta, eps}, ...]. V1 requires "
                        "uniform eps across queries.")
    p.add_argument("--output-dir", required=True,
                   help="output directory for per-query .json/.npz files.")
    p.add_argument("--mode", choices=["diagonal", "householder"], default="householder")
    p.add_argument("--mpi", type=int, default=4)
    p.add_argument("--workdir", default=str(ZETA9_DIR))
    p.add_argument("--sage-env", default=DEFAULT_SAGE_ENV)
    p.add_argument("--check-local-p3k", dest="check_local_p3k",
                   action="store_true", default=False)
    p.add_argument("--max-matches", type=int, default=1000)
    p.add_argument("--triples-chunk-rows", type=int, default=None,
                   help="rows per MPI chunk in stage 5. If omitted, auto-computed "
                        "as nrows / (4*mpi) so all ranks get work (small caches).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.mode != "householder":
        sys.exit("batched stage-5 is currently householder-mode only.")

    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        sys.exit(f"workdir not found: {workdir}")
    if not Path(args.sage_env).exists():
        sys.exit(f"sage env not found: {args.sage_env}")

    queries_path = Path(args.queries).resolve()
    if not queries_path.exists():
        sys.exit(f"queries file not found: {queries_path}")
    with open(queries_path) as fh:
        queries = json.load(fh)
    if not isinstance(queries, list) or not queries:
        sys.exit(f"queries file: expected non-empty JSON list, got {type(queries)}")

    eps_set = sorted({float(q["eps"]) for q in queries})
    if len(eps_set) != 1:
        sys.exit(
            f"batched stage-5 V1 requires uniform eps across queries, got {eps_set}"
        )
    eps = float(eps_set[0])
    if eps > args.eps_pre:
        sys.exit(
            f"query eps ({eps}) > --eps-pre ({args.eps_pre}); cache only "
            f"supports tighter queries. Rebuild cache at looser eps_pre, or "
            f"tighten the queries."
        )

    f_u = args.max_f
    f_v = 2 * f_u
    arts = artifact_paths(workdir, f_v, args.eps_pre,
                          check_local_p3k=args.check_local_p3k,
                          mode=args.mode)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-pick chunk_rows so all MPI ranks get work. Stage 5 dispatches whole
    # chunks per rank (chunk_idx % size != rank → skip); too-large chunks
    # leave most ranks idle on small triple files.
    #
    # CAVEAT: stage 4 meta indexes chunks by chunk_idx using the chunk_rows
    # value it was BUILT with. If we override chunk_rows, the meta is stale,
    # so we skip --chunk_meta_json and fall back to per-chunk locator build.
    # For batch mode this is fine because the per-chunk locator cost is
    # amortized across all queries hitting that chunk.
    use_stage4_meta = True
    try:
        with open(arts["stage2_manifest"]) as fh:
            nrows = int(json.load(fh)["rows_written"])
    except Exception as e:
        sys.exit(f"could not read stage 2 manifest for nrows: {e}")
    if args.triples_chunk_rows is None:
        chunk_rows = max(1, nrows // (4 * args.mpi))
        if chunk_rows < 200000:  # stage 5 default; below this we likely need to override
            use_stage4_meta = False
        print(f"[zeta9_query_batch] nrows={nrows} mpi={args.mpi} -> "
              f"triples_chunk_rows={chunk_rows} (auto, "
              f"{'override' if not use_stage4_meta else 'meta-compatible'})")
    else:
        chunk_rows = int(args.triples_chunk_rows)
        use_stage4_meta = (chunk_rows == 200000)

    py = f"{args.sage_env}/bin/python"
    _no_overs = os.environ.get("ZETA9_NO_OVERSUBSCRIBE", "") in ("1", "true", "yes")
    mpirun = ["mpirun", "-n", str(int(args.mpi))] + ([] if _no_overs else ["--oversubscribe"])

    cmd = mpirun + [py, "zeta9/search_householder_two_rows_streamed_mpi.py",
        "--triples_file", str(arts["stage2"]),
        "--triples_json", str(arts["stage2_manifest"]),
        "--rootdb_prefix", str(arts["stage3_prefix"]),
        "--f", str(f_v),
        "--queries", str(queries_path),
        "--output_dir", str(output_dir),
        "--triples_chunk_rows", str(chunk_rows),
        "--max_matches", str(args.max_matches),
        "--quiet"]
    if use_stage4_meta:
        cmd += ["--chunk_meta_json", str(arts["stage4_meta"])]

    print(f"[zeta9_query_batch] n_queries={len(queries)} eps={eps} "
          f"eps_pre={args.eps_pre} max_f(u)={f_u} mode={args.mode} mpi={args.mpi}")
    print(f"[zeta9_query_batch] output_dir={output_dir}")
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    env = stages_env(args.sage_env)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env, cwd=str(workdir))
    wall = time.time() - t0
    print(f"[zeta9_query_batch] rc={rc} total={wall:.1f}s "
          f"per_query={wall/len(queries):.2f}s")

    summary_path = output_dir / "batch_summary.json"
    if summary_path.exists():
        with open(summary_path) as fh:
            summary = json.load(fh)
        n_succ = sum(1 for q in summary["queries"]
                     if q.get("best_dist_fro") is not None
                     and q["best_dist_fro"] < eps)
        print(f"[zeta9_query_batch] success={n_succ}/{len(queries)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
