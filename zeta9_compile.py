#!/usr/bin/env python3
# zeta9_compile.py — single-shot wrapper around zeta9's 5-stage MPI pipeline.
#
# Mirrors HRSA_tester's role: takes (theta, epsilon, max_f) and produces a
# unified-schema JSON ready for v_validate.  Caches stages 1-4 between runs
# (theta-independent precompute, keyed on (f, eps_pre)).
#
# Usage:
#   zeta9_compile.py --theta T --epsilon EPS --max-f F [--eps-pre EPS_PRE]
#                    [--mpi N] [--workdir DIR] [--json out.json]
#
# Conventions:
#   --max-f is the U-denominator cap (matches HRSA_tester's max_f semantic).
#   V's denominator is 2 * max_f.  The actual unitary.f in the JSON output is
#   what zeta9 actually returned.
#
#   IMPORTANT: zeta9's INTERNAL --f flag is the V-denominator (NOT u-denom),
#   per memory note zeta9_findings_2026-05-07.  This wrapper translates:
#       wrapper --max-f N  →  zeta9 --f 2N
#   so cache files / artifact filenames in zeta9/D/ are labelled with V-denom.
#   A previous version of this wrapper passed --max-f directly as zeta9 --f,
#   producing caches at half the intended denominator and 0% match for any V
#   that HRSA could find at the corresponding u-denom (fixed 2026-05-10).
#
#   --epsilon is the Frobenius distance tolerance ‖V − target‖_F ≤ eps.
#
#   --eps-pre is the per-coordinate vector tolerance used by zeta9 stages
#   1-4 to size the candidate enumeration.  If unset, defaults to eps/2 as
#   a heuristic (collect_targets uses per-coord L2; Frobenius is the joint).
#   Tighter eps_pre = smaller artifacts but risk of missing solutions.
#
#   Sage env path: assumes /home/hlamm/miniforge3/envs/sage (per memory
#   sage_env_setup.md).  Override with $SAGE_ENV.
#
#   MPI rank count: defaults to 4 (fits 15 GB; try.py default is 32).

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# ----- defaults / paths -----
DEFAULT_SAGE_ENV = os.environ.get("SAGE_ENV", "/home/hlamm/miniforge3/envs/sage")
ZETA9_DIR = Path("/home/hlamm/Desktop/efficent_gates/unified/zeta9")
HRSA_DIR = Path("/home/hlamm/Desktop/efficent_gates/unified/hrsa")
DECOMPOSE_TOOL = HRSA_DIR / "decompose_tool"


# ----- Persistent decompose worker (audit decompose_optimization_audit.md) -----
#
# Spawning decompose_tool per-V pays ~15s every call to rebuild the static
# canonical_lookup (953K BFS table). Using --persistent mode keeps the process
# alive: one JSON request per stdin line, one JSON response per stdout line.

class DecomposeWorker:
    """Long-lived decompose_tool process for batched V decomposition.

    Usage:
        with DecomposeWorker() as w:
            for V_int, v_f in batches:
                result = w.decompose(V_int, v_f)
    """
    def __init__(self, decompose_tool=None):
        if decompose_tool is None:
            decompose_tool = DECOMPOSE_TOOL
        self._tool = Path(decompose_tool)
        if not self._tool.exists():
            raise FileNotFoundError(
                f"decompose_tool binary not found at {self._tool}. "
                f"Build it with: cd {HRSA_DIR} && make decompose_tool"
            )
        self._proc = subprocess.Popen(
            [str(self._tool), "--persistent"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def decompose(self, V_int, v_f, *, timeout=120.0):
        """Send one V request, read one response. Same return shape as
        run_hrsa_decompose. `timeout` is enforced via the underlying Popen's
        wait — if the worker hangs, the caller is responsible for catching."""
        payload = {
            "f": int(v_f),
            "V": [[[int(x) for x in V_int[i][j]] for j in range(3)] for i in range(3)],
        }
        line = json.dumps(payload, separators=(",", ":"))
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"decompose worker died (stdin write): {stderr!r}") from e
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"decompose worker died (no response): {stderr!r}")
        try:
            res = json.loads(resp_line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"decompose worker bad JSON: {resp_line[:200]!r}") from e
        # Same shape as run_hrsa_decompose's success branch.
        if not res.get("success", False) and "error" in res:
            return {
                "success": False, "N_D": None, "N_C": None, "N_total": None,
                "sde_chi_initial": None, "sde_chi_final": None,
                "syllables": None, "trailing_clifford": None,
                "decompose_returncode": int(res.get("returncode", 1)),
                "decompose_stderr": res.get("error", ""),
            }
        return {
            "success": bool(res.get("success", False)),
            "N_D": int(res.get("D_count", 0)),
            "N_C": None,
            "N_total": None,
            "sde_chi_initial": int(res.get("sde_chi_initial", 0)),
            "sde_chi_final": int(res.get("sde_chi_final", 0)),
            "syllables": res.get("syllables", []),
            "trailing_clifford": res.get("trailing_clifford", None),
            "decompose_returncode": 0,
        }

    def close(self):
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=5.0)
        except Exception:
            self._proc.kill()
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ----- HRSA exact-synthesis bridge -----
#
# decompose_tool (hrsa/decompose_cli.cpp) is HRSA's exact-synthesis routine
# exposed as a stdin/stdout JSON filter.  Input  schema: {f, V[3][3][6]}.
# Output schema: {success, D_count, sde_chi_initial, sde_chi_final, syllables[],
# trailing_clifford{f, V}}.  D_count is the N_D word length; len(syllables) is
# the syllable count; the trailing_clifford is the residual monomial-Clifford
# factor.  Multiplying syllables * trailing_clifford reproduces V exactly.

def run_hrsa_decompose(V_int, v_f, decompose_tool=None, timeout=120):
    """Invoke decompose_tool on a ringZ9 V matrix.

    Parameters
    ----------
    V_int : 3x3 nested list of length-6 int arrays
        Numerator coefficients on the canonical Z-basis {1, ζ_9, ζ_9², ζ_9³,
        ζ_9⁴, ζ_9⁵} of each V entry. Same shape as ``rows_coeffs_layer_f``
        from zeta9 stage 5 / xout.npz.
    v_f : int
        Common denominator exponent: each entry is sum_i V_int[r][c][i] · ζ_9^i
        divided by 3^v_f.
    decompose_tool : Path-like or None
        Path to the decompose_tool binary; defaults to ``DECOMPOSE_TOOL``.
    timeout : float
        Subprocess timeout in seconds.

    Returns
    -------
    dict with keys:
      - ``success`` (bool): True if the routine reached sde_chi_final == 0
        (monomial Clifford residual).
      - ``N_D`` (int): D_count from the routine (Clifford+D word length).
      - ``N_C`` (None): not currently extracted; trailing_clifford carries
        the monomial Clifford as a matrix, not a word — we leave conversion
        to N_C for downstream consumers.
      - ``N_total`` (None): see above.
      - ``sde_chi_initial`` (int)
      - ``sde_chi_final`` (int)
      - ``syllables`` (list[dict]): one entry per syllable, with keys
        ``a0, a1, a2, eps, delta, has_H``.
      - ``trailing_clifford`` (dict): {f, V} of the residual.
      - ``decompose_returncode`` (int): subprocess exit code.

    Raises
    ------
    FileNotFoundError if decompose_tool is missing.
    subprocess.TimeoutExpired on timeout.
    json.JSONDecodeError on malformed output.
    """
    if decompose_tool is None:
        decompose_tool = DECOMPOSE_TOOL
    tool = Path(decompose_tool)
    if not tool.exists():
        raise FileNotFoundError(
            f"decompose_tool binary not found at {tool}. "
            f"Build it with: cd {HRSA_DIR} && make decompose_tool"
        )

    payload = {
        "f": int(v_f),
        "V": [[[int(x) for x in V_int[i][j]] for j in range(3)] for i in range(3)],
    }
    proc = subprocess.run(
        [str(tool)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        # Tooling error (1=JSON parse, 2=schema/dim).  Surface stderr.
        return {
            "success": False,
            "N_D": None, "N_C": None, "N_total": None,
            "sde_chi_initial": None, "sde_chi_final": None,
            "syllables": None,
            "trailing_clifford": None,
            "decompose_returncode": int(proc.returncode),
            "decompose_stderr": proc.stderr.strip(),
        }
    res = json.loads(proc.stdout)
    return {
        "success": bool(res.get("success", False)),
        "N_D": int(res.get("D_count", 0)),
        "N_C": None,
        "N_total": None,
        "sde_chi_initial": int(res.get("sde_chi_initial", 0)),
        "sde_chi_final": int(res.get("sde_chi_final", 0)),
        "syllables": res.get("syllables", []),
        "trailing_clifford": res.get("trailing_clifford", None),
        "decompose_returncode": 0,
    }


def stages_env(sage_env):
    env = os.environ.copy()
    env["PATH"] = f"{sage_env}/bin:" + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # Ensure the zeta9 package directory is on PYTHONPATH so Numba's cached
    # compiled functions (which pickle the module under its package-qualified
    # name, e.g. ``zeta9.search_diagonal_matrix_two_rows_streamed_mpi``) can be
    # rehydrated when the stage-5 script is invoked as a path rather than as
    # ``-m zeta9.X``.  Without this, mpi worker ranks fail Numba cache load
    # with ``ModuleNotFoundError: No module named 'zeta9'`` on the very first
    # call into a cached kernel.
    extra = str(ZETA9_DIR)
    pp = env.get("PYTHONPATH", "")
    if extra not in pp.split(os.pathsep):
        env["PYTHONPATH"] = extra + (os.pathsep + pp if pp else "")
    return env


def fmt_eps(eps):
    """Format eps for filename use (matches run_matrix's labels)."""
    return f"{eps:g}"


def artifact_paths(workdir, f, eps_pre, check_local_p3k=False, mode="householder",
                   kprime_cap=0):
    """Return dict of stage→artifact path (the sentinel file we check for caching).

    When check_local_p3k=True, stage-1 / downstream caches are tagged with `_p3k`
    so they do NOT collide with unfiltered caches (a p3k-pruned Y1 universe is a
    strict subset and the downstream artifacts depend on which Y's were kept).

    The `mode` arg ('diagonal' or 'householder') tags stage-2+ cache paths so
    cells generated under different modes (different `--norm` and input layouts)
    don't collide. Stage-1 Y1 files are mode-independent (collect_targets
    produces the same Y values regardless — see code audit 2026-05-13).

    Both modes are explicitly tagged (`_hh` for householder, `_diag` for
    diagonal). Untagged paths exist on disk from earlier sessions with the
    wrong norm=2 setup; tagging both prevents collisions with that stale state.

    When ``kprime_cap > 0`` (LOSSY stage-2 prune, feature branch
    kprime-cap-stage2-lossy), stage-2+ cache paths get a `_kp{N}` suffix so the
    pruned TM file does NOT shadow a previously-computed full TM file. Stage 1
    is shared because the prune only affects the cross-join step.
    """
    label = fmt_eps(eps_pre)
    suffix = "_p3k" if check_local_p3k else ""
    mode_tag = "_hh" if mode == "householder" else "_diag"
    kp_tag = f"_kp{int(kprime_cap)}" if kprime_cap and int(kprime_cap) > 0 else ""
    d = workdir / "D"
    return {
        "stage1_u0": d / f"Y1_f={f}_u=0_eps={label}{suffix}.npy",
        "stage1_u1": d / f"Y1_f={f}_u=1_eps={label}{suffix}.npy",
        "stage2":    d / f"TM_f={f}_eps={label}{suffix}{mode_tag}{kp_tag}",
        "stage2_manifest": d / f"TM_f={f}_eps={label}{suffix}{mode_tag}{kp_tag}.manifest.json",
        "stage3":    d / f"RM_f={f}_eps={label}{suffix}{mode_tag}{kp_tag}_local.roots.json",
        "stage3_prefix": d / f"RM_f={f}_eps={label}{suffix}{mode_tag}{kp_tag}_local",
        "stage3_global_prefix": d / f"RM_f={f}_global",
        "stage4_sidecar": d / f"RM_f={f}_eps={label}{suffix}{mode_tag}{kp_tag}_local.phase_sidecar_binned_bins=512.meta.json",
        "stage4_meta": d / f"RM_f={f}_eps={label}{suffix}{mode_tag}{kp_tag}_triples_chunk_meta.json",
        "stage4_meta_prefix": d / f"RM_f={f}_eps={label}{suffix}{mode_tag}{kp_tag}_triples_chunk_meta",
    }


def run_cmd(cmd, env, cwd, label, dry_run=False):
    """Run a subprocess; print and timeit; return wall seconds."""
    print(f"[zeta9_compile] {label}: {' '.join(map(str, cmd))}", flush=True)
    if dry_run:
        return 0.0
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, cwd=cwd).returncode
    dt = time.time() - t0
    print(f"[zeta9_compile] {label} done in {dt:.1f}s (rc={rc})", flush=True)
    if rc != 0:
        raise RuntimeError(f"{label} failed (rc={rc})")
    return dt


def ensure_precompute(workdir, f, eps_pre, mpi, sage_env, dry_run=False,
                      check_local_p3k=False, mode="householder",
                      eps_target=None, lazy_u_values=None,
                      kprime_cap=0):
    """Run stages 1-4 if their artifacts don't exist.

    Mode determines stage-2 norm and input layout:
      - 'diagonal' (norm=1): inputs = (Y1_u=1, Y1_u=0, Y1_u=0). Target row has
        one non-zero entry near 1, others ≈ 0. Use when synthesizing diagonal
        targets directly.
      - 'householder' (norm=2): inputs = (Y1_u=1, Y1_u=1, Y1_u=0). Target row
        has two non-zero entries (Householder construction); after free
        Clifford conjugation recovers R^Z. Cheaper N_D at same ε because the
        lattice has more candidates with Σ|u_i|² = 2 (two ≈ 1 entries) than
        Σ|u_i|² = 1 with the "two near-zero" restriction.

    Returns dict of per-stage wall seconds (0 if cached)."""
    if mode not in ("diagonal", "householder"):
        raise ValueError(f"mode must be 'diagonal' or 'householder', got {mode!r}")
    arts = artifact_paths(workdir, f, eps_pre, check_local_p3k=check_local_p3k,
                          mode=mode, kprime_cap=kprime_cap)
    env = stages_env(sage_env)
    py = f"{sage_env}/bin/python"
    sage_python = [f"{sage_env}/bin/sage", "-python"]
    # MPICH (Lenore) doesn't accept --oversubscribe; OpenMPI (lucia) does.
    # Set ZETA9_NO_OVERSUBSCRIBE=1 in the environment to omit the flag.
    _no_overs = os.environ.get("ZETA9_NO_OVERSUBSCRIBE", "") in ("1", "true", "yes")
    mpirun = ["mpirun", "-n", str(int(mpi))] + ([] if _no_overs else ["--oversubscribe"])
    label = fmt_eps(eps_pre)
    times = {}

    # Round-13 (2026-05-13): mode controls collect_targets --norm AND the
    # input layout for select_triples. Original zeta9 had both setups; the
    # unified wrapper now exposes both via --mode.
    cellnorm = 1 if mode == "diagonal" else 2

    # Stage 1: collect_targets for u=0 and u=1.
    # The Y1 files contain |a|² values in Z[α] for ringZ9 a near u_target.
    # Stage-1 --norm matches the cell's target_sum convention so the Y values
    # are scaled to the right range. (collect_targets uses --norm only for the
    # epsilon-binning sanity bound, not the value content; same Y values
    # appear regardless. But we set it to match for consistency.)
    for u, key in [(0, "stage1_u0"), (1, "stage1_u1")]:
        out = arts[key]
        if out.exists():
            print(f"[zeta9_compile] stage1 (u={u}) cached: {out}")
            times[f"stage1_u{u}"] = 0.0
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = mpirun + [py, "-m", "zeta9.collect_targets",
                        "--f", str(f),
                        "--norm", str(cellnorm),
                        "--u", str(u),
                        "--eps", label,
                        "--eps_bin_width", "0",
                        "--output", str(out.with_suffix("")),
                        "--resume"]
        if check_local_p3k:
            cmd.append("--check_local_p3k")
        times[f"stage1_u{u}"] = run_cmd(cmd, env, workdir, f"stage1 u={u}", dry_run)

    # Stage 2: select_triples_optimized — inputs depend on mode.
    if arts["stage2"].exists():
        print(f"[zeta9_compile] stage2 cached: {arts['stage2']}")
        times["stage2"] = 0.0
    else:
        if mode == "diagonal":
            # (u=1, u=0, u=0): target has one non-zero entry near 1
            inputs2 = str(arts["stage1_u0"])
        else:
            # householder: (u=1, u=1, u=0): two non-zero entries near magnitude 1
            inputs2 = str(arts["stage1_u1"])
        cmd = mpirun + sage_python + ["-m", "zeta9.select_triples_optimized",
            "--inputs1", str(arts["stage1_u1"]),
            "--inputs2", inputs2,
            "--inputs3", str(arts["stage1_u0"]),
            "--f", str(f),
            "--output", str(arts["stage2"]),
            "--norm", str(cellnorm),
            # 2026-05-12: Reverted bucket_cache_entries (was 4) and n_join_buckets
            # (was 32768) — those tight caps dropped cache hit rate to 0.01% and
            # caused massive disk thrashing.  Per perf audit: 64 cache entries +
            # 8192 buckets keeps RAM modest while restoring 5-20× cache hits.
            "--bucket_cache_entries", "64",      # default 16; we have RAM headroom
            "--n_join_buckets", "8192",          # default; was 32768
            "--resume"]
        # 2026-05-24: drop-column-c lossless prune. Writes 6 int64 per row
        # instead of 9, reconstructing Y_c on read. 1.5× final TM file
        # reduction, no behavioural change downstream (readers auto-detect
        # the Z9TC header). On by default; set env ZETA9_NO_DROP_COL_C=1 to
        # keep the legacy 9-int64 format (e.g. when sharing the TM file with
        # non-v2 fitters).
        if os.environ.get("ZETA9_NO_DROP_COL_C", "") not in ("1", "true", "yes"):
            cmd.append("--drop_col_c")
        # K'-cap (2026-05-24, feature branch kprime-cap-stage2-lossy): LOSSY
        # per-(a_group, c_bucket) cap on emitted triples. Default off (byte-
        # identical to legacy). Set --kprime-cap N (recommended 64 for f>=6).
        # The wrapper-driven layout always puts the u=0 input as the smallest
        # → A slot, so the σ_1 band center for Y_a is 0 (kprime_u_a_sq=0.0).
        # This matches the kernel default but we pass it explicitly to make
        # the assumption auditable in stage-2 manifests.
        if kprime_cap and int(kprime_cap) > 0:
            cmd += ["--kprime-cap", str(int(kprime_cap)),
                    "--kprime-u-a-sq", "0.0"]
        times["stage2"] = run_cmd(cmd, env, workdir, "stage2 (select_triples)", dry_run)

    # Stage 3: find_roots_exact_v2
    if arts["stage3"].exists():
        print(f"[zeta9_compile] stage3 cached: {arts['stage3']}")
        times["stage3"] = 0.0
    else:
        cmd = mpirun + sage_python + ["-m", "zeta9.find_roots_exact_v2",
            "--triples_file", str(arts["stage2"]),
            "--triples_json", str(arts["stage2_manifest"]),
            "--no_legacy_fallback",
            "--rootdb_prefix", str(arts["stage3_prefix"]),
            "--global_rootdb_prefix", str(arts["stage3_global_prefix"]),
            "--resume",
            # Dtype dispatch: int32 when f≤15 (cuts sidecar mmap pages ~2× at
            # paper-data f ∈ {4,6,8,10,12,14}; auto-flips to int64 at f≥16).
            # Threshold lives in find_roots_exact_v2._ROOT_DTYPE_F_MAX_INT32.
            "--root_dtype_f", str(int(f))]
        # Lazy-enum pre-filter: skip Y's whose best_frob bound exceeds ε_target.
        # Per audit zeta9_stage3_optimization_audit.md: 31× wall reduction at
        # ε=1e-4, 130× at 5e-5. f here is the zeta9-internal V-denom exponent.
        if eps_target is not None and eps_target > 0:
            cmd += ["--eps_target", repr(float(eps_target)),
                    "--lazy_f", str(int(f))]
            if lazy_u_values is not None:
                cmd += ["--lazy_u_values", ",".join(str(u) for u in lazy_u_values)]
        times["stage3"] = run_cmd(cmd, env, workdir, "stage3 (find_roots)", dry_run)

    # Stage 4a: build_phase_sidecar_binned_mpi
    if arts["stage4_sidecar"].exists():
        print(f"[zeta9_compile] stage4a (sidecar) cached: {arts['stage4_sidecar']}")
        times["stage4a"] = 0.0
    else:
        cmd = mpirun + sage_python + ["-m", "zeta9.build_phase_sidecar_binned_mpi",
            "--rootdb_prefix", str(arts["stage3_prefix"])]
        times["stage4a"] = run_cmd(cmd, env, workdir, "stage4a (sidecar)", dry_run)

    # Stage 4b: build_triple_chunk_metadata — REMOVED 2026-05-20.
    # Bench at f=2 showed stage 5 chunk_meta=None fallback (`_build_locator`
    # on-the-fly) matches precomputed performance (1.34s vs 1.34s) and
    # produces identical output. Stage 4b wasted ~3 min wall + 25 GB disk
    # per cell at production f=4 ε=0.001 (97 min pre-bug-fix, ~3 min after).
    # See audit zeta9_stage4_optimization_audit.md.
    times["stage4b"] = 0.0

    return times


def run_stage5(workdir, f, eps_pre, eps_frob, theta, mpi, sage_env, output_prefix,
               dry_run=False, check_local_p3k=False, mode="householder",
               kprime_cap=0):
    """Run stage 5 for the given theta. Dispatches on mode.

    - ``mode='diagonal'``: invoke ``search_diagonal_matrix_two_rows_streamed_mpi.py``,
      which finds row 1 and row 2 of the diagonal target unitary and reconstructs
      row 3 via cross product. Output schema: ``xout.npz`` with
      ``rows_coeffs_layer_f`` of shape (n, 3, 3, 6).
    - ``mode='householder'``: invoke ``search_householder_two_rows_streamed_mpi.py``
      (Round-13, 2026-05-13), which finds a Householder vector
      ``u = (a_0, a_1, a_2)/3^f`` and builds ``V = X_{(0,1)} . (I - conj(u) (x) u)``
      exactly in ringZ9 at layer 2f. Same output schema so ``extract_best_v``
      works without modification.

    Returns wall seconds."""
    arts = artifact_paths(workdir, f, eps_pre, check_local_p3k=check_local_p3k,
                          mode=mode, kprime_cap=kprime_cap)
    env = stages_env(sage_env)
    py = f"{sage_env}/bin/python"
    # MPICH (Lenore) doesn't accept --oversubscribe; OpenMPI (lucia) does.
    # Set ZETA9_NO_OVERSUBSCRIBE=1 in the environment to omit the flag.
    _no_overs = os.environ.get("ZETA9_NO_OVERSUBSCRIBE", "") in ("1", "true", "yes")
    mpirun = ["mpirun", "-n", str(int(mpi))] + ([] if _no_overs else ["--oversubscribe"])

    # Absolute path: relative path breaks when workdir is outside the source
    # tree (e.g. /mnt/.../validate_workdir). Stages 1-4 use `-m zeta9.X` so
    # they're cwd-independent; stage 5 invokes the script as a file path.
    stage5_diag = str(ZETA9_DIR / "zeta9" / "search_diagonal_matrix_two_rows_streamed_mpi.py")
    stage5_hh   = str(ZETA9_DIR / "zeta9" / "search_householder_two_rows_streamed_mpi.py")

    if mode == "diagonal":
        cmd = mpirun + [py, stage5_diag,
            "--triples_file", str(arts["stage2"]),
            "--triples_json", str(arts["stage2_manifest"]),
            "--rootdb_prefix", str(arts["stage3_prefix"]),
            "--f", str(f),
            "--theta", repr(theta),
            "--eps", repr(eps_frob),
            "--output_prefix", str(output_prefix),
            "--row2_cache", str(output_prefix) + ".row2.npy",
            # --chunk_meta_json removed 2026-05-20: stage 5 fallback is identical perf.
            "--max_matches", "1000",   # Early-exit knob (2026-05-12); avoids redundant 20h scan
            "--quiet"]
        return run_cmd(cmd, env, workdir, "stage5 (diagonal search)", dry_run)
    elif mode == "householder":
        cmd = mpirun + [py, stage5_hh,
            "--triples_file", str(arts["stage2"]),
            "--triples_json", str(arts["stage2_manifest"]),
            "--rootdb_prefix", str(arts["stage3_prefix"]),
            "--f", str(f),
            "--theta", repr(theta),
            "--eps", repr(eps_frob),
            "--output_prefix", str(output_prefix),
            # --chunk_meta_json removed 2026-05-20: fallback is identical perf.
            "--max_matches", "1000",
            "--quiet"]
        return run_cmd(cmd, env, workdir, "stage5 (householder search)", dry_run)
    else:
        raise ValueError(f"mode must be 'diagonal' or 'householder', got {mode!r}")


def run_stage5_batched(workdir, f, eps_pre, eps_frob, thetas, mpi, sage_env,
                       output_dir, dry_run=False, check_local_p3k=False,
                       mode="householder", query_ids=None, kprime_cap=0):
    """Run stage 5 in BATCHED mode (one mpirun, many θ sharing one ε).

    Writes per-query results to ``output_dir/q_<id>.npz`` (and ``.json``).
    Use ``extract_best_v(output_dir + "/q_" + id + ".npz", f)`` to pull each
    cell's best V.

    Amortizes mpirun startup, Numba JIT, sidecar mmap warmup, root enumeration
    (``large_cache``/``get_large_canonical``), Sage init, and triple I/O across
    all queries — typically 2–5× faster per cell than launching ``run_stage5``
    once per θ.

    V1 constraint (from ``search_householder_streamed_batched``): all queries
    must share the same ε. Calling code should group cells by (max_f, eps_pre,
    eps_frob) before invoking.

    Returns wall seconds.
    """
    if not thetas:
        raise ValueError("run_stage5_batched: thetas list is empty")
    if query_ids is not None and len(query_ids) != len(thetas):
        raise ValueError(f"query_ids length {len(query_ids)} != thetas length {len(thetas)}")

    arts = artifact_paths(workdir, f, eps_pre, check_local_p3k=check_local_p3k,
                          mode=mode, kprime_cap=kprime_cap)
    env = stages_env(sage_env)
    py = f"{sage_env}/bin/python"
    _no_overs = os.environ.get("ZETA9_NO_OVERSUBSCRIBE", "") in ("1", "true", "yes")
    mpirun = ["mpirun", "-n", str(int(mpi))] + ([] if _no_overs else ["--oversubscribe"])

    if mode == "diagonal":
        stage5_script = str(ZETA9_DIR / "zeta9" / "search_diagonal_matrix_two_rows_streamed_mpi.py")
    elif mode == "householder":
        stage5_script = str(ZETA9_DIR / "zeta9" / "search_householder_two_rows_streamed_mpi.py")
    else:
        raise ValueError(f"mode must be 'diagonal' or 'householder', got {mode!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if query_ids is None:
        query_ids = [str(i) for i in range(len(thetas))]

    queries = [{"id": str(qid), "theta": float(t), "eps": float(eps_frob)}
               for qid, t in zip(query_ids, thetas)]
    queries_path = str(output_dir / "queries.json")
    with open(queries_path, "w") as fh:
        json.dump(queries, fh)

    cmd = mpirun + [py, stage5_script,
        "--triples_file", str(arts["stage2"]),
        "--triples_json", str(arts["stage2_manifest"]),
        "--rootdb_prefix", str(arts["stage3_prefix"]),
        "--f", str(f),
        "--queries", queries_path,
        "--output_dir", str(output_dir),
        "--max_matches", "1000",
        "--quiet"]
    return run_cmd(cmd, env, workdir,
                   f"stage5 batched ({len(queries)} queries @ eps={eps_frob:g})",
                   dry_run)


def batched_query_npz_path(output_dir, query_id):
    """Path to the per-query stage-5 output npz produced by run_stage5_batched."""
    return str(Path(output_dir) / f"q_{query_id}.npz")


# ----- Householder reconstruction & schema JSON emission -----

def householder_v_from_u(u_coeffs, f):
    """Build V = X_(0,1) * (I - u u†) from u given as ringZ9 coefs.
    Returns 3x3 V as integer numerator coefficients with V denominator 3^(2f).

    u_coeffs: shape (3, 6) — three ringZ9 numerators of u_i (each at denom 3^f).
    """
    # We'll do this exactly using the ringZ9 helpers in v_validate.py.
    # Import them directly.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v_validate import (ringZ9_zero, ringZ9_one, ringZ9_neg, ringZ9_add,
                            ringZ9_sub, ringZ9_mul, ringZ9_conj)

    # Compute u_i * conj(u_j) for all (i,j); each at denom 3^(2f).
    uu = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            uu[i][j] = ringZ9_mul(u_coeffs[i], ringZ9_conj(u_coeffs[j]))

    # I - uu (denominator 3^(2f) for off-diag; diag gets 1·3^(2f) - uu[i][i]).
    f2 = 2 * f
    one_3_2f = [3**f2, 0, 0, 0, 0, 0]  # 1 in canonical, scaled to denom 3^(2f).
    H = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i == j:
                H[i][j] = ringZ9_sub(one_3_2f, uu[i][j])
            else:
                H[i][j] = ringZ9_neg(uu[i][j])

    # V = X_(0,1) · H  : swap rows 0 and 1.
    V = [[None]*3 for _ in range(3)]
    for j in range(3):
        V[0][j] = H[1][j]
        V[1][j] = H[0][j]
        V[2][j] = H[2][j]
    return V, f2


def _build_decomposition_block(*, V_int, v_f, success, do_decompose,
                                errors, decompose_timeout):
    """Return the ``decomposition`` block for the schema JSON.

    If a V is present and ``do_decompose`` is True, calls run_hrsa_decompose
    and packs the result.  On any failure, records into ``errors`` and
    returns the placeholder block (N_D=None, etc.).
    """
    placeholder = {
        "N_D": None, "N_C": None, "N_total": None,
        "sde_chi_initial": None, "sde_chi_final": None,
        "syllables": None,
    }
    if not (success and do_decompose and V_int is not None and v_f is not None):
        return placeholder
    try:
        res = run_hrsa_decompose(V_int, v_f, timeout=decompose_timeout)
    except FileNotFoundError as e:
        errors.append({"stage": "decompose", "error": str(e)})
        return placeholder
    except subprocess.TimeoutExpired:
        errors.append({"stage": "decompose",
                       "error": f"decompose_tool timed out after "
                                f"{decompose_timeout}s"})
        return placeholder
    except Exception as e:  # JSON parse, etc.
        errors.append({"stage": "decompose",
                       "error": f"{type(e).__name__}: {e}"})
        return placeholder
    if res.get("decompose_returncode", 0) != 0:
        errors.append({"stage": "decompose",
                       "error": f"decompose_tool exited rc="
                                f"{res['decompose_returncode']}: "
                                f"{res.get('decompose_stderr', '')}"})
        return placeholder
    return {
        "N_D": res["N_D"],
        "N_C": res["N_C"],
        "N_total": res["N_total"],
        "sde_chi_initial": res["sde_chi_initial"],
        "sde_chi_final": res["sde_chi_final"],
        "syllables": res["syllables"],
        "trailing_clifford": res["trailing_clifford"],
        "decompose_success": res["success"],
    }


def emit_schema_json(out_path, *, theta, epsilon, max_f, V_int, v_f,
                     achieved_frob, success, wall_seconds, mpi_ranks,
                     command_line, errors=None, decompose=True,
                     decompose_timeout=120):
    """Write schema-conformant JSON to out_path.

    When ``decompose=True`` and a V matrix is present, also invoke HRSA's
    ``decompose_tool`` to populate the ``decomposition`` block (N_D, syllables,
    sde_chi_*).  Any decompose failure is appended to ``errors`` but does not
    abort emission (the V matrix is still useful for v_validate).
    """
    if errors is None:
        errors = []
    target_diag = [
        complex(math.cos(-theta/2), math.sin(-theta/2)),
        complex(math.cos( theta/2), math.sin( theta/2)),
        complex(1, 0),
    ]
    target = [[[0.0, 0.0]]*3 for _ in range(3)]
    for i in range(3):
        target[i][i] = [target_diag[i].real, target_diag[i].imag]

    rec = {
        "identification": {
            "backend": "zeta9",
            "version": "zeta9-2026-05",
            "command_line": command_line,
            "host": socket.gethostname(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": "1.0",
        },
        "inputs": {
            "theta": theta,
            "epsilon": epsilon,
            "max_f": max_f,
            "c": None,
            "max_solns": None,
            "max_direct": None,
        },
        "target": {
            "gate": "R_Z_01_theta",
            "convention": "hrsa",
            "matrix": target,
        },
        "achieved": {
            "success": bool(success),
            "achieved_frob": (float(achieved_frob) if success else None),
            "epsilon_passed": bool(success and achieved_frob < epsilon),
            "f_level": (int(v_f) if success else None),
            "method": ("zeta9-householder" if success else "none"),
        },
        "unitary": (None if not success else {
            "ring": "Z[zeta_9, 1/3]",
            "basis": "1, zeta9, zeta9^2, zeta9^3, zeta9^4, zeta9^5",
            "f": int(v_f),
            "V": [[list(map(int, V_int[i][j])) for j in range(3)] for i in range(3)],
        }),
        "decomposition": _build_decomposition_block(
            V_int=V_int, v_f=v_f, success=success,
            do_decompose=decompose, errors=errors,
            decompose_timeout=decompose_timeout),
        "sanity_checks": {
            "unitarity_residual": None,  # not computed here; v_validate does it
            "frobenius_check_passed": bool(success and achieved_frob < epsilon),
            "decompose_roundtrip_passed": None,
        },
        "performance": {
            "wall_seconds": float(wall_seconds),
            "cpu_seconds": float(wall_seconds),
            "max_rss_kb": 0,
            "threads": 1,
            "mpi_ranks": int(mpi_ranks),
        },
        "errors": errors,
    }
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=2)


# ----- Output extraction from xout.npz -----

def extract_best_v(xout_npz_path, f):
    """Load xout.npz from stage 5 and return the best (lowest Frobenius) V.

    Stage 5 emits the FULL 3x3 matrix per result (rows_coeffs_layer_f, shape
    (n, 3, 3, 6)).  Earlier this function expected a u-vector emission from
    a Householder path; that path was never wired and the keys never matched.
    Fixed 2026-05-12: read rows_coeffs_layer_f and dist_fro directly.

    Returns (V_int, v_f, frob) where V_int is a 3x3 nested list of 6-int
    ringZ9 numerator coefs, v_f is the V denominator (= zeta9's --f).
    """
    import numpy as np
    d = np.load(str(xout_npz_path))
    fdists = d.get("dist_fro")
    if fdists is None or fdists.size == 0:
        return None, None, None
    best_idx = int(np.argmin(fdists))
    best_frob = float(fdists[best_idx])

    # rows_coeffs_layer_f: (n, 3, 3, 6) int64.  Pick best, convert to list-of-list-of-list.
    rows = d["rows_coeffs_layer_f"][best_idx]  # shape (3, 3, 6)
    V_int = [[[int(x) for x in rows[i][j]] for j in range(3)] for i in range(3)]
    # The V denominator stored in xout.npz is in d["f"] (a length-1 array).
    v_f = int(d["f"][0]) if "f" in d.files else 2 * f
    return V_int, v_f, best_frob


# ----- main -----

def main():
    p = argparse.ArgumentParser(description="zeta9 single-shot compiler")
    p.add_argument("--theta", type=float, required=True)
    p.add_argument("--epsilon", type=float, required=True,
                   help="Frobenius epsilon ‖V−target‖_F bound")
    p.add_argument("--max-f", type=int, required=True,
                   help="u-denominator cap (V denom = 2*max_f)")
    p.add_argument("--eps-pre", type=float, default=None,
                   help="per-coord eps for stages 1-4 (default: epsilon/2)")
    p.add_argument("--mpi", type=int, default=4)
    p.add_argument("--workdir", default=str(ZETA9_DIR))
    p.add_argument("--sage-env", default=DEFAULT_SAGE_ENV)
    p.add_argument("--json", default=None, help="schema JSON output path")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-local-p3k", dest="check_local_p3k",
                   action="store_true", default=None,
                   help="Enable upstream p3k norm screen in stage 1 (~10x Y "
                        "pruning, per memory zeta9_p3k_screen.md). Caches are "
                        "tagged _p3k and do not collide with unfiltered caches. "
                        "Auto-enabled when epsilon <= 1e-4; pass --no-check-local-p3k "
                        "to override.")
    p.add_argument("--no-check-local-p3k", dest="check_local_p3k",
                   action="store_false",
                   help="Disable p3k screen even at tight epsilon.")
    p.add_argument("--kprime-cap", "--kprime_cap", dest="kprime_cap",
                   type=int, default=0,
                   help="LOSSY stage-2 prune (feature branch kprime-cap-stage2-"
                        "lossy): per (a_group, c_bucket) call, keep at most this "
                        "many triples sorted by smallest |sigma_1(Y_a)|. 0 = "
                        "disabled (default, byte-identical to legacy). "
                        "Recommended 64 for f>=6 builds where the unpruned "
                        "stage-2 cross-join exceeds Lenore's 697 GB. Top-1 Y_a "
                        "per call is provably preserved; ranks 2..K may be "
                        "clipped (~10% miss rate at K'=64 per backlog estimate, "
                        "verify empirically before shipping).")
    p.add_argument("--mode", choices=["diagonal", "householder"], default="householder",
                   help="Synthesis mode (Round-13, 2026-05-13):\n"
                        "  diagonal  : norm=1, inputs (u=1, u=0, u=0). For directly\n"
                        "              synthesizing diagonal R^Z (one non-zero row entry).\n"
                        "  householder (default): norm=2, inputs (u=1, u=1, u=0). Two\n"
                        "              non-zero row entries; after Clifford conjugation\n"
                        "              recovers R^Z. Cheaper N_D at same eps because the\n"
                        "              lattice has more |row|^2 = 2 candidates than\n"
                        "              |row|^2 = 1 with two near-zero entries.\n"
                        "Cache paths tagged with _diag when mode=diagonal so cells\n"
                        "produced under different modes do not collide.")
    args = p.parse_args()

    # Auto-enable p3k screen for tight epsilon (per memory note zeta9_p3k_screen.md).
    # The screen is sound and prunes ~10x Y candidates upstream of stages 2-5.
    if args.check_local_p3k is None:
        args.check_local_p3k = bool(args.epsilon <= 1.0e-4)

    workdir = Path(args.workdir).resolve()
    sage_env = args.sage_env
    eps_pre = args.eps_pre if args.eps_pre is not None else (args.epsilon / 2.0)
    f_u = args.max_f          # u-denominator (HRSA's max_f semantic)
    f_v = 2 * f_u             # V-denominator = zeta9's INTERNAL --f flag

    if not workdir.exists():
        sys.exit(f"workdir not found: {workdir}")
    if not Path(sage_env).exists():
        sys.exit(f"sage env not found: {sage_env}")

    print(f"[zeta9_compile] theta={args.theta} eps_frob={args.epsilon} eps_pre={eps_pre} "
          f"max_f(u-denom)={f_u} zeta9_f(v-denom)={f_v} mpi={args.mpi} "
          f"check_local_p3k={args.check_local_p3k}")

    wall_start = time.time()

    # Stages 1-4 (cached if artifacts exist).  Pass V-denom to zeta9 stages.
    # Lazy-enum: pass Frobenius ε so stage 3 can skip Y's that can't satisfy it.
    # Householder layout u=(1,1,0); diagonal uses different slot values.
    _lazy_u = (1.0, 1.0, 0.0) if args.mode == "householder" else None
    times14 = ensure_precompute(workdir, f_v, eps_pre, args.mpi, sage_env, args.dry_run,
                                check_local_p3k=args.check_local_p3k,
                                mode=args.mode,
                                eps_target=args.epsilon,
                                lazy_u_values=_lazy_u,
                                kprime_cap=args.kprime_cap)

    # Stage 5: V-denom for zeta9 stage 5 args.
    output_prefix = workdir / "xout"
    times["stage5"] = run_stage5(workdir, f_v, eps_pre, args.epsilon, args.theta,
                                 args.mpi, sage_env, output_prefix, args.dry_run,
                                 check_local_p3k=args.check_local_p3k,
                                 mode=args.mode,
                                 kprime_cap=args.kprime_cap)

    # Extract result.  Householder reconstruction uses u-denom.
    V_int, v_f, frob = (None, None, None)
    if not args.dry_run:
        xout = workdir / "xout.npz"
        if xout.exists():
            V_int, v_f, frob = extract_best_v(xout, f_u)

    success = V_int is not None and frob is not None and frob < args.epsilon

    wall = time.time() - wall_start

    if args.json:
        emit_schema_json(args.json,
            theta=args.theta, epsilon=args.epsilon, max_f=f_u,
            V_int=V_int, v_f=v_f, achieved_frob=frob, success=success,
            wall_seconds=wall, mpi_ranks=args.mpi,
            command_line=sys.argv)

    print(f"[zeta9_compile] success={success}  frob={frob}  wall={wall:.1f}s")
    return 0 if success else 1


# stage_5 stash
times = {}

if __name__ == "__main__":
    sys.exit(main())
