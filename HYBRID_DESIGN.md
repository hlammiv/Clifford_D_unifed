# Hybrid HRSA + zeta9-stage-5 design sketch

Goal: HRSA enumerates u_1 candidates; zeta9 stage-5 algebraically completes
u_2 (and u_3 via column-orthogonality). Eliminate per-call Sage startup
overhead via a persistent daemon.

## Current per-call overhead (problem statement)

For each u_1, the current `subprocess.run([mpirun, ..., search_diagonal_matrix_two_rows_streamed_mpi.py, ..., --fixed_row1_sage <u_1>])`:

```
Sage interpreter startup ........ ~5-10 sec
MPI rank initialization (n=14) .. ~1-2 sec
Load triples_file (large bin)  .. ~1-3 sec
Load rootdb (Sage CyclotomicField + ideal cache) ~3-5 sec
Load chunk_meta_json + sidecar .. ~1 sec
Per-call setup TOTAL          ... ~12-20 sec
Actual stage-5 search ........... ~1-2 sec  (the algorithmic work)
Per-call wall time              .. ~15-20 sec
```

For N=100 u_1 candidates from HRSA: ~25 min of pure setup overhead.

## Persistent-daemon design (Option A: stdin/stdout JSON-lines protocol)

### Daemon mode patch to `search_diagonal_matrix_two_rows_streamed_mpi.py`

Add a `--daemon` CLI flag. When set, the script:

1. Initializes ONCE (rank 0 + all MPI ranks):
   - Sage CyclotomicField(9), ZZ, etc.
   - Open triples_file, load chunk_meta, mmap rootdb if possible.
   - Pre-build per-bucket Y-target indices.
2. Enters the main daemon loop:
   ```python
   # Rank 0 reads stdin; broadcasts job to other ranks via MPI.
   # All ranks run the search.  Rank 0 writes JSON result to stdout.

   while True:
       if rank == 0:
           line = sys.stdin.readline()
           if not line:
               job = None  # signal shutdown
           else:
               job = json.loads(line)
       job = comm.bcast(job, root=0)
       if job is None:
           break

       theta   = job["theta"]
       eps     = job["eps"]
       fixed_row1 = job["fixed_row1"]   # Sage-vector string
       # Reuse pre-loaded artifacts; only the per-job state changes.
       result = run_search(theta, eps, fixed_row1)

       if rank == 0:
           sys.stdout.write(json.dumps(result) + "\n")
           sys.stdout.flush()
   ```

3. Result schema (one JSON line per job):
   ```json
   {
     "success": true,
     "frobenius": 0.0123,
     "row_u_coeffs": [[...6 ints...], [...6 ints...], [...6 ints...]],
     "wall_seconds": 1.34
   }
   ```

### Wrapper driver (`hybrid_compile.py`)

```python
import subprocess, json, time
from pathlib import Path

class Zeta9Daemon:
    """Long-lived MPI+Sage process for zeta9 stage-5 row-2 fitting."""
    def __init__(self, *, f, eps_pre, mpi=14, sage_env, workdir):
        env = os.environ.copy()
        env["PATH"] = f"{sage_env}/bin:" + env["PATH"]
        cmd = [
            "mpirun", "-n", str(mpi), "--oversubscribe",
            f"{sage_env}/bin/sage", "-python",
            "zeta9/search_diagonal_matrix_two_rows_streamed_mpi.py",
            "--daemon",
            "--triples_file", f"D/TM_f={f}_eps={eps_pre}",
            "--triples_json", f"D/TM_f={f}_eps={eps_pre}.manifest.json",
            "--rootdb_prefix", f"D/RM_f={f}_eps={eps_pre}_local",
            "--chunk_meta_json", f"D/RM_f={f}_eps={eps_pre}_triples_chunk_meta.json",
            "--f", str(f),
        ]
        self.proc = subprocess.Popen(cmd, env=env, cwd=workdir,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     text=True, bufsize=1)

    def search(self, theta, eps_frob, fixed_row1_sage):
        job = {"theta": theta, "eps": eps_frob, "fixed_row1": fixed_row1_sage}
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        result = json.loads(self.proc.stdout.readline())
        return result

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def hybrid_compile(theta, eps_frob, max_f, *, eps_pre=2.5e-3):
    # 1. Run HRSA to get u_1 candidates
    u_candidates = run_hrsa_get_u_candidates(theta, eps_frob, max_f)
    # ^ extracts x1_cands list from HRSA (needs HRSA mod to expose them; see below)

    # 2. Pre-screen u_1 candidates with cheap Y-feasibility check (optional, big win)
    u_candidates = [u for u in u_candidates if y_norm_equation_feasible(u, max_f)]

    # 3. Spin up daemon ONCE
    daemon = Zeta9Daemon(f=max_f, eps_pre=eps_pre, mpi=14, ...)

    # 4. For each u_1, ask daemon to fit u_2
    best = None
    for u1 in u_candidates:
        row1_str = format_as_sage_vector(u1, max_f)
        result = daemon.search(theta, eps_frob, row1_str)
        if result["success"]:
            if best is None or result["frobenius"] < best["frobenius"]:
                best = {"u1": u1, "row_u_coeffs": result["row_u_coeffs"], **result}
            # optional: stop at first hit / continue collecting / take first under threshold
            if result["frobenius"] < eps_frob * 0.5: break  # good enough

    daemon.close()

    # 5. Build full V from best (u_1, row_2 found by zeta9)
    #    Row 3 from column orthogonality.
    if best is None: return None
    V = construct_V_from_two_rows(best["u1"], best["row_u_coeffs"], max_f, theta)
    return V
```

### HRSA changes needed

To expose u_1 candidates from HRSA's entryEnumeration:

1. Add a `--dump-x1-cands <file>` flag. After enumeration, write x1_cands as
   JSON lines (each: `[a0, a1, a2, a3, a4, a5]`). Don't continue to phase 2
   (the triple-product matching that the hybrid skips).
2. Or expose as a library function: `extern "C" int hrsa_enumerate_u1(double theta, int max_f, int* out_buf)`.

The simpler option: add `--dump-x1-cands` and let `hybrid_compile.py`
read the file. Single-pass HRSA invocation, then daemon-driven completion.

## Implementation phases (~1 week total)

1. **Daemon mode in stage 5** (~1-2 days):
   - Refactor main() to separate init from per-job work.
   - MPI broadcast loop.
   - Test on synthetic input.

2. **`--dump-x1-cands` in HRSA** (~half day):
   - Add CLI flag, write x1_cands to JSON file.
   - Skip phase 2/3 if dumping (fast exit).

3. **`hybrid_compile.py` wrapper** (~1 day):
   - Spawn HRSA with `--dump-x1-cands`.
   - Spawn daemon, drive it, collect results.
   - Construct V, emit schema JSON.

4. **Y-feasibility pre-screen** (~half day, optional):
   - For each u_1, compute `Y_2 = 2·3^{2f} − u_1·ū_1`.
   - Skip if Y_2 is not totally positive in Z[ζ_9 + ζ_9⁻¹]^+.

5. **End-to-end test** (~1 day):
   - Run on θ=0.5, ε=0.05, f=4. Expect: HRSA's u_1 candidates → daemon
     fits u_2 → V → v_validate passes.
   - Compare wall time vs standalone HRSA at same query.

## Risks & open questions

- **MPI broadcasts from stdin**: not all MPI implementations cleanly handle
  rank-0-only stdin reads. May need a barrier + bcast pattern; verify with
  Open MPI 4.1.6.
- **Sage memory leaks**: long-running Sage processes can accumulate memory
  from cached objects. May need periodic restart at N=1000+ jobs.
- **Daemon hang on bad input**: malformed JSON or unrecoverable Sage error
  could hang the daemon. Wrapper should have a per-job timeout (kill+restart
  if exceeded).
- **What does zeta9 actually return for `--fixed_row1`?** Need to confirm
  via test that the existing stage 5 produces sensible output when given a
  fixed row 1. Currently we have artifacts at f=4 norm=2 building, so we
  can test once the precompute finishes.
