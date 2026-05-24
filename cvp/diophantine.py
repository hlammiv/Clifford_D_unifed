"""diophantine.py — Phase 3 of the qutrit Babai-CVP plan.

Given an ``x_1 ∈ Z[ζ_9]`` candidate from Phase 2 (``cvp.babai.babai_x1``)
and the target rotation parameters ``(theta, f, eps)``, find pairs
``(x_2, x_3) ∈ Z[ζ_9]²`` satisfying

* the Householder norm equation
    ``q(x_1) + q(x_2) + q(x_3) = 2 · 3^{2f}``     (exact, in Z)
* principal-embedding proximity:
    ``|σ_1(x_2)/3^f - (-1)| ≤ eps``
    ``|σ_1(x_3)/3^f -   0 | ≤ eps``

The reflector ``u = (x_1, x_2, x_3) / 3^f`` then has ``‖u‖² = 2`` exactly
in the ring and ``H = I - u u*`` is a *unitary* matrix in Z[ζ_9, 1/3]
whose first row is within ``2 eps`` of the target ``R_z(theta)|0⟩``.
Phase 4 reifies this into the full ``H`` and decomposes via canonical_reducer.

Algorithmic outline (Selinger 2012 §6 generalised to Z[ζ_9])
------------------------------------------------------------

1. **Enumerate ``x_3``.** Tiny q-norm, principal embedding ≈ 0. We reuse
   the LLL-reduced rank-6 Minkowski lattice from ``cvp.gram`` and walk its
   shortest non-zero vectors; the candidate count is ``O(eps²·3^{2f})``,
   bounded by a few hundred at f ≤ 6.

2. **For each ``x_3``, solve for ``x_2``.** Let
   ``N := 2·3^{2f} - q(x_1) - q(x_3)``. This is the scalar q-norm target
   for ``x_2``. Together with the embedding constraint
   ``σ_1(x_2 x̄_2) ≈ 3^{2f}`` we get a *2-D* slab of triples
   ``(m_0, m_1, m_2)`` with ``m_0 + 2 m_2 = N`` and
   ``σ_1(M) ≈ 3^{2f}``, parameterised by ``m_1``. The remaining direction
   is bounded by the totally-real positivity of ``M = x_2 x̄_2``.

3. **For each ``(m_0, m_1, m_2)``,** call zeta9's PARI ideal-factorization
   norm-equation solver via the long-running ``_NormEqWorker`` subprocess.
   It returns all ``x_2 ∈ Z[ζ_9]`` with ``x_2 x̄_2 = M``. Filter by
   embedding-proximity to ``-3^f`` and exact q-form constraint.

Wall-time budget
----------------

Aim: < 5 s per ``solve_x2_x3(x_1, ...)`` call at f=6, eps=10⁻³. Dominated
by ``~10²-10³`` PARI calls × 5 ms each, plus a one-time subprocess startup
amortised across all the calls within one ``solve_x2_x3`` invocation.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from fpylll import (
        Enumeration,
        EnumerationError,
        IntegerMatrix,
        LLL,
    )
    from fpylll.fplll.gso import MatGSO
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "fpylll is required for cvp.diophantine. Install with `pip install fpylll`."
    ) from exc

from cvp.gram import q_form
from cvp.babai import minkowski_embedding

__all__ = [
    "enumerate_x3",
    "solve_q_norm",
    "solve_x2_x3",
    "set_sage_env",
    "verify_pair",
    "householder_frobenius",
]

# ---------------------------------------------------------------------------
# Sage subprocess setup
# ---------------------------------------------------------------------------

_DEFAULT_SAGE_ENV = os.environ.get("SAGE_ENV", "/home/hlamm/miniforge3/envs/sage")


def set_sage_env(path: str) -> None:
    """Override the Sage environment used to launch the norm-eq subprocess.

    Defaults to ``$SAGE_ENV`` or ``/home/hlamm/miniforge3/envs/sage`` (per the
    sage_env_setup memory note). Override only if those don't apply on your
    machine.
    """
    global _DEFAULT_SAGE_ENV
    _DEFAULT_SAGE_ENV = str(path)


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# α_r = 2 cos(2π r / 9) for r ∈ {1, 2, 4} — the three real embeddings of
# α = ζ_9 + ζ_9⁻¹ in F = Q(α).
_ALPHA = (
    2.0 * math.cos(2.0 * math.pi * 1 / 9.0),  # ≈ 1.5320888...
    2.0 * math.cos(2.0 * math.pi * 2 / 9.0),  # ≈ 0.3472964...
    2.0 * math.cos(2.0 * math.pi * 4 / 9.0),  # ≈ -1.8793852...
)
# α_r² - 2 = 2 cos(4π r / 9) = α_{2r mod 9 conjugacy class}. We use the raw
# values here so we never depend on any particular convention.
_ALPHA_SQ_M2 = tuple(a * a - 2.0 for a in _ALPHA)

_OMEGA = [
    complex(math.cos(2 * math.pi * j / 9.0), math.sin(2 * math.pi * j / 9.0))
    for j in range(6)
]


def _sigma_1(coefs: Sequence[int]) -> complex:
    """Principal complex embedding ``σ_1(a) = Σ a_j ζ_9^j``."""
    return sum(coefs[j] * _OMEGA[j] for j in range(6))


# ---------------------------------------------------------------------------
# Persistent worker for zeta9 norm-eq solves
# ---------------------------------------------------------------------------


class _NormEqWorker:
    """Long-lived sage-env subprocess wrapping ``solve_single_norm_eq``.

    Usage::

        with _NormEqWorker() as w:
            for Y in many_Y:
                roots = w.solve(Y)
    """

    def __init__(self, sage_env: Optional[str] = None, timeout: float = 120.0):
        self._sage_env = Path(sage_env or _DEFAULT_SAGE_ENV)
        if not self._sage_env.exists():
            raise FileNotFoundError(
                f"Sage env not found at {self._sage_env}. "
                "Set $SAGE_ENV or call cvp.diophantine.set_sage_env(...)."
            )
        self._timeout = float(timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._startup_seconds: Optional[float] = None
        self._n_calls: int = 0
        self._stderr_drainer: Optional[threading.Thread] = None
        self._stderr_buf: list[str] = []
        self._stderr_lock = threading.Lock()

    # -- context-manager glue ------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            return
        py = self._sage_env / "bin" / "python"
        if not py.exists():
            raise FileNotFoundError(f"missing python at {py}")
        # Inherit minimal env: PATH (so Singular etc. resolve) + HOME (Sage
        # writes to ~/.sage). Pass through any explicit user overrides.
        env = {
            "PATH": f"{self._sage_env}/bin:" + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/home/hlamm"),
        }
        # Forward LANG/LC_* to silence locale warnings under Sage.
        for k in ("LANG", "LC_ALL"):
            if k in os.environ:
                env[k] = os.environ[k]

        self._proc = subprocess.Popen(
            [str(py), "-u", "-m", "cvp._zeta9_norm_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(Path(__file__).resolve().parent.parent),  # unified/
            text=True,
            bufsize=1,
        )
        # Drain stderr in the background so a chatty Sage doesn't deadlock us.
        self._stderr_drainer = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_drainer.start()
        # Wait for ready handshake.
        ready = self._read_one()
        if not ready.get("ok") or not ready.get("ready"):
            self.close()
            raise RuntimeError(f"worker failed to start: {ready}")
        self._startup_seconds = float(ready.get("startup_seconds", 0.0))

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                try:
                    self._proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, ValueError):
                    pass
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)
        finally:
            self._proc = None

    # -- internal IO ---------------------------------------------------------

    def _drain_stderr(self) -> None:
        assert self._proc is not None
        try:
            for line in self._proc.stderr:
                with self._stderr_lock:
                    # Keep last few KB so callers can debug; bounded.
                    self._stderr_buf.append(line)
                    if len(self._stderr_buf) > 200:
                        self._stderr_buf.pop(0)
        except Exception:
            pass

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_buf[-20:])

    def _read_one(self) -> dict:
        assert self._proc is not None
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"worker exited unexpectedly. stderr tail:\n{self._stderr_tail()}"
            )
        return json.loads(line)

    # -- public API ----------------------------------------------------------

    def solve(self, Y: Tuple[int, int, int]) -> List[Tuple[int, ...]]:
        """Return all ``x ∈ Z[ζ_9]`` with ``bb_to_real_coeffs(x) == Y``.

        Y must be a 3-tuple of Python ints. Roots include the full unit
        orbit ``± ζ^k`` (matches zeta9's stage-3 default).
        """
        if self._proc is None:
            self.start()
        Y = tuple(int(c) for c in Y)
        if len(Y) != 3:
            raise ValueError(f"Y must be length-3, got {Y!r}")
        assert self._proc is not None
        self._proc.stdin.write(json.dumps({"op": "solve", "Y": list(Y)}) + "\n")
        self._proc.stdin.flush()
        self._n_calls += 1
        resp = self._read_one()
        if not resp.get("ok"):
            raise RuntimeError(
                f"worker error on solve({Y}): {resp.get('err', resp)}\n"
                f"stderr tail:\n{self._stderr_tail()}"
            )
        return [tuple(int(c) for c in r) for r in resp["roots"]]

    @property
    def startup_seconds(self) -> Optional[float]:
        return self._startup_seconds

    @property
    def n_calls(self) -> int:
        return self._n_calls


# ---------------------------------------------------------------------------
# x_3 enumeration: short Minkowski vectors near the origin
# ---------------------------------------------------------------------------

_FPLLL_SCALE = 10 ** 6  # matches cvp.babai


def _unweighted_basis_int() -> Tuple[IntegerMatrix, np.ndarray]:
    """Build the integer-scaled unit-weight Minkowski basis (mirrors babai.py).

    Returns the fpylll ``IntegerMatrix`` plus the unimodular matrix ``U_z``
    such that ``coords @ U_z`` is the ζ-basis coefficient vector. We LLL-reduce
    in-place and return the U so the caller can convert lattice coordinates
    back to ζ-coefficient vectors.
    """
    M = minkowski_embedding()  # 6×6 real
    basis_real = M.T * _FPLLL_SCALE  # rows = image of ζ^j under unit-weight
    rows = [[int(round(x)) for x in row] for row in basis_real]
    B = IntegerMatrix.from_matrix(rows)
    U = IntegerMatrix.identity(6)
    LLL.reduction(B, U=U)
    U_z = np.array([[int(U[i, j]) for j in range(6)] for i in range(6)], dtype=object)
    return B, U_z


def enumerate_x3(
    theta: float,
    f: int,
    eps: float,
    max_q3: Optional[int] = None,
    max_candidates: int = 200,
) -> List[Tuple[int, ...]]:
    """Return candidate ``x_3 ∈ Z[ζ_9]`` close to 0 in σ_1 and small in q.

    Constraints:

    * ``q(x_3) ≤ max_q3`` (defaults to ``ceil(3·eps²·3^{2f}) + 8``);
    * ``|σ_1(x_3)/3^f| ≤ eps``  (per-coord eps; the caller passes
      ``ε_coord``, which is ``ε_frob / (2 c √2)`` per HRSA's contraction
      analysis — at ``c = 1`` that's ``ε_frob / (2 √2) ≈ 0.354 ε_frob``);
    * the zero vector is always returned (a valid x_3 = 0 always satisfies
      both constraints and the algorithm wants it considered).

    Parameters
    ----------
    theta, f : as in ``babai_x1``. ``theta`` is unused here (x_3 target is 0,
        embedding-symmetric) but kept in the signature for call-site symmetry.
    eps : per-coordinate Frobenius tolerance.
    max_q3 : optional explicit q-norm budget. Default = derived from eps and f.
    max_candidates : safety cap on output length.

    Returns
    -------
    list of 6-tuples of int. Ordered ascending by ``|σ_1(x_3)|``.
    """
    if f < 0:
        raise ValueError(f"f must be ≥ 0, got {f}")
    # σ_1 budget: ε · 3^f
    sigma_budget = eps * (3.0 ** f)
    # q-budget: from q(x) ≤ (|σ_1|² + |σ_2|² + |σ_4|²)/3 ≤ 3·ε²·3^{2f} +
    # a generous safety floor for tiny f where the slab is empty otherwise.
    # An additional +8 floor covers the always-included {(0,...,0)} +
    # short-cycle units cases when ε is microscopic.
    if max_q3 is None:
        max_q3 = int(math.ceil(3.0 * eps * eps * (3.0 ** (2 * f)))) + 8
    max_q3 = max(int(max_q3), 1)

    # The zero vector is always a candidate.
    out: List[Tuple[float, Tuple[int, ...]]] = [(0.0, (0, 0, 0, 0, 0, 0))]
    seen: set = {(0, 0, 0, 0, 0, 0)}

    # Build LLL'd unit-weight Minkowski lattice and enumerate short vectors.
    B_unw, U_z = _unweighted_basis_int()
    n = B_unw.nrows

    # Euclidean norm² in the ambient = 3 · q(v) · _FPLLL_SCALE². Set the
    # enumeration radius² to max_q3·3·_FPLLL_SCALE² + a margin.
    max_norm_sq = 3.0 * float(max_q3) * (_FPLLL_SCALE ** 2)

    Mgso = MatGSO(B_unw)
    Mgso.update_gso()
    # Ask for up to ``8 * max_candidates`` short vectors (we filter by σ_1
    # afterwards). fpylll enumerate caps internally; we re-cap via the
    # max_candidates parameter on return.
    enum = Enumeration(Mgso, nr_solutions=max(8 * max_candidates, 64))
    try:
        results = enum.enumerate(0, n, max_norm_sq, 0)
    except EnumerationError:
        results = []

    for _norm_sq, c in results:
        delta_coords = np.array([int(x) for x in c], dtype=object)
        # Lattice point in original basis = c @ U @ B_orig where B_orig is
        # the ζ-coefficient embedding. But our U is built from the LLL of the
        # unit-weight Minkowski lattice, where each *row* of the original
        # basis matrix is the Minkowski image of ζ^j. The corresponding ζ-coef
        # vector is just U.T @ c (the linear combination of ζ-basis vectors).
        delta = tuple(
            int(sum(delta_coords[i] * U_z[i][j] for i in range(n)))
            for j in range(n)
        )
        # Include both ± v.
        for sign in (1, -1):
            cand = tuple(sign * x for x in delta)
            if cand in seen:
                continue
            seen.add(cand)
            qv = q_form(cand)
            if qv > max_q3:
                continue
            s1 = _sigma_1(cand)
            err = abs(s1)
            if err > sigma_budget:
                continue
            out.append((err, cand))

    out.sort(key=lambda x: x[0])
    return [c for _, c in out[:max_candidates]]


# ---------------------------------------------------------------------------
# (m_1, m_2) slab enumeration + worker dispatch
# ---------------------------------------------------------------------------


def _enumerate_m_triples(
    N: int,
    target_sigma1_sq: float,
    sigma1_tol: float,
    *,
    max_triples: int = 2000,
) -> List[Tuple[int, int, int]]:
    """Enumerate integer ``(m_0, m_1, m_2)`` with

    * ``m_0 = N - 2 m_2`` (Householder q-constraint)
    * ``|σ_1(M) - target_sigma1_sq| ≤ sigma1_tol`` where
      ``σ_1(M) = N + α_1 m_1 + (α_1² - 2) m_2``
    * ``σ_2(M) ≥ 0`` and ``σ_4(M) ≥ 0`` (positivity of M = x_2 x̄_2)

    Strategy
    --------
    The σ_1 slab pins ``m_2`` to a ``±sigma1_tol/|α_2|`` interval given
    ``m_1``. We bound ``m_1`` upfront by intersecting the slab with the
    σ_2 ≥ 0 and σ_4 ≥ 0 half-planes; this turns the candidate strip into a
    *compact* line segment whose integer points we iterate. At large ``f``
    the m_1 range stays O(3^f / α_*) (not O(3^{2f})), keeping the sweep
    feasible.

    Returns ``(m_0, m_1, m_2)`` triples sorted ascending by ``|σ_1(M)−target|``.
    """
    if N < 0:
        return []

    alpha_1, alpha_2, alpha_4 = _ALPHA
    asq_m2_1, asq_m2_2, asq_m2_4 = _ALPHA_SQ_M2

    # m_2 slab half-width per m_1.
    radius_m2 = sigma1_tol / abs(asq_m2_1)

    # Substitute the slab *center* m_2 = (T - N - α_1·m_1) / asq_m2_1 into
    # σ_2 ≥ 0 and σ_4 ≥ 0; this gives two linear inequalities on m_1.
    # Each defines a half-line; their intersection (∩ a finite interval
    # from σ_1(M) ≥ 0) is the m_1 range we sweep.
    #
    # σ_r(M) = N + α_r·m_1 + asq_m2_r · m_2  (r ∈ {1,2,4})
    # ⇒ σ_r(M) = N + α_r·m_1 + asq_m2_r · (T - N - α_1 m_1)/asq_m2_1
    #          = (1 - asq_m2_r/asq_m2_1) · N
    #            + asq_m2_r/asq_m2_1 · T
    #            + (α_r - α_1 · asq_m2_r/asq_m2_1) · m_1
    #
    # Solve for m_1 from σ_r(M) ≥ 0 (with slack 0):
    #   a_r · m_1 ≥ -b_r  ⇒  m_1 ≥ -b_r/a_r   if a_r > 0
    #                          m_1 ≤ -b_r/a_r   if a_r < 0
    def _bound(r_idx: int) -> Tuple[Optional[float], Optional[float]]:
        asq_m2_r = _ALPHA_SQ_M2[r_idx]
        alpha_r = _ALPHA[r_idx]
        ratio = asq_m2_r / asq_m2_1
        a_r = alpha_r - alpha_1 * ratio
        b_r = (1.0 - ratio) * N + ratio * target_sigma1_sq
        # Constraint: a_r · m_1 + b_r ≥ -|asq_m2_r| · radius_m2  (slack from
        # m_2 wobble within the σ_1 slab; here we use the σ_r tolerance
        # induced by the slab width — pessimistic linearisation).
        slack = abs(asq_m2_r) * radius_m2
        rhs = -(b_r) - slack
        if abs(a_r) < 1e-12:
            # Constraint independent of m_1: either always satisfied or never.
            if b_r + slack >= 0:
                return -math.inf, math.inf
            return 0.0, -1.0  # empty
        if a_r > 0:
            return rhs / a_r, math.inf
        return -math.inf, rhs / a_r

    bnds = [_bound(idx) for idx in (1, 2)]  # σ_2 and σ_4
    lo = max(b[0] for b in bnds)
    hi = min(b[1] for b in bnds)
    if lo > hi:
        return []
    # Sanity cap: in pathological numerically-near-singular cases the bounds
    # can be very loose. Clamp to ±10·N/|α_2| as an absolute safety net.
    abs_cap = 10 * int(math.ceil(N / max(abs(alpha_2), 1e-12))) + 16
    m1_lo = max(int(math.floor(lo)) - 1, -abs_cap)
    m1_hi = min(int(math.ceil(hi)) + 1, +abs_cap)

    cand: List[Tuple[float, Tuple[int, int, int]]] = []

    for m1 in range(m1_lo, m1_hi + 1):
        m2_center = (target_sigma1_sq - N - alpha_1 * m1) / asq_m2_1
        m2_lo = int(math.floor(m2_center - radius_m2))
        m2_hi = int(math.ceil(m2_center + radius_m2))
        for m2 in range(m2_lo, m2_hi + 1):
            m0 = N - 2 * m2
            sigma1 = m0 + alpha_1 * m1 + alpha_1 * alpha_1 * m2
            sigma2 = m0 + alpha_2 * m1 + alpha_2 * alpha_2 * m2
            sigma4 = m0 + alpha_4 * m1 + alpha_4 * alpha_4 * m2
            if sigma2 < -1e-9 or sigma4 < -1e-9:
                continue
            err = abs(sigma1 - target_sigma1_sq)
            if err > sigma1_tol + 1e-9:
                continue
            cand.append((err, (m0, m1, m2)))
            if len(cand) >= 4 * max_triples:
                break
        if len(cand) >= 4 * max_triples:
            break

    cand.sort(key=lambda x: x[0])
    return [Y for _, Y in cand[:max_triples]]


def solve_q_norm(
    N: int,
    embedding_target: complex,
    eps: float,
    f: int,
    *,
    worker: Optional[_NormEqWorker] = None,
    max_triples: int = 2000,
    max_x2: int = 256,
) -> List[Tuple[int, ...]]:
    """Solve ``q(x_2) = N`` with ``σ_1(x_2)`` close to ``embedding_target``.

    Parameters
    ----------
    N : target q-norm. Must be ≥ 0.
    embedding_target : complex target for ``σ_1(x_2)``. Magnitude squared
        gives ``σ_1(M) = |embedding_target|²``.
    eps : Frobenius tolerance on ``|σ_1(x_2)/3^f - embedding_target/3^f|``.
    f : denominator exponent (used to convert eps → absolute σ_1 tolerance).
    worker : optional pre-warmed :class:`_NormEqWorker`. If None, one is
        spun up for the duration of this call and torn down afterwards
        (paying ~3 s startup); supply your own when calling repeatedly.
    max_triples : upper bound on (m_0, m_1, m_2) triples to try.
    max_x2 : safety cap on returned x_2 count.
    """
    if N < 0:
        return []
    if N == 0:
        # Only x_2 = 0 has q = 0. Check embedding match.
        if abs(0.0 - embedding_target) <= eps * (3.0 ** f):
            return [(0, 0, 0, 0, 0, 0)]
        return []

    abs_target = float(abs(embedding_target))
    # Magnitude constraint: |σ_1(x_2)| ∈ [|target| - ε·3^f, |target| + ε·3^f]
    # (band condition + triangle on the complex distance). Squaring:
    # σ_1(M) = |σ_1(x_2)|² ∈ [(|target|-ε·3^f)², (|target|+ε·3^f)²]
    # which is an asymmetric interval around (|target|² - 0). We treat it
    # via a *centered* slab whose center is the geometric mean of the two
    # endpoints (≈ |target|² for small ε), and half-width is half the
    # gap.
    abs_3f = 3.0 ** f
    band = eps * abs_3f
    lo_sq = max(0.0, (abs_target - band) ** 2)
    hi_sq = (abs_target + band) ** 2
    abs_target_sq = 0.5 * (lo_sq + hi_sq)
    sigma1_tol = 0.5 * (hi_sq - lo_sq) + 1.0

    Ys = _enumerate_m_triples(
        N=int(N),
        target_sigma1_sq=abs_target_sq,
        sigma1_tol=sigma1_tol,
        max_triples=max_triples,
    )

    if not Ys:
        return []

    owns_worker = worker is None
    if owns_worker:
        worker = _NormEqWorker()
        worker.start()

    out: List[Tuple[float, Tuple[int, ...]]] = []
    seen: set = set()
    try:
        for Y in Ys:
            roots = worker.solve(Y)
            for r in roots:
                if r in seen:
                    continue
                # Defensive: filter by exact q(r) == N (zeta9 returns full
                # orbit which includes unit images; q is unit-orbit invariant).
                if q_form(r) != N:
                    continue
                s1 = _sigma_1(r)
                err = abs(s1 - embedding_target)
                if err > eps * (3.0 ** f) + 1e-12:
                    continue
                seen.add(r)
                out.append((err, r))
    finally:
        if owns_worker:
            worker.close()

    out.sort(key=lambda x: x[0])
    return [r for _, r in out[:max_x2]]


# ---------------------------------------------------------------------------
# Top-level Phase 3 entry point
# ---------------------------------------------------------------------------


def householder_frobenius(
    x_1: Tuple[int, ...],
    x_2: Tuple[int, ...],
    x_3: Tuple[int, ...],
    *,
    theta: float,
    f: int,
) -> float:
    """Frobenius residual ``||X_(0,1) H − R_(0,1)^Z(θ)||_F``.

    Uses the canonical theta convention (per memory note ``theta_convention``):
    ``R_(0,1)^Z(θ) = diag(e^{-iθ/2}, e^{+iθ/2}, 1)`` and the Householder
    vector target ``u_target = (e^{+iθ/2}, -1, 0) / sqrt(2)`` (un-normalised
    in our ring: ``u = (x_1, x_2, x_3)/3^f`` with ``||u||²_ring = 2``).

    Returns the Frobenius norm of the residual matrix in the principal
    complex embedding (σ_1). This is the same quantity HRSA reports as
    "Matrix Frobenius distance".
    """
    u = np.array(
        [_sigma_1(x_1), _sigma_1(x_2), _sigma_1(x_3)],
        dtype=np.complex128,
    ) / (3.0 ** f)
    H = np.eye(3, dtype=np.complex128) - np.outer(u, np.conj(u))
    X01 = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.complex128,
    )
    target = np.diag(
        [np.exp(-1j * theta / 2.0), np.exp(1j * theta / 2.0), 1.0 + 0j]
    )
    return float(np.linalg.norm(X01 @ H - target, "fro"))


def verify_pair(
    x_1: Tuple[int, ...],
    x_2: Tuple[int, ...],
    x_3: Tuple[int, ...],
    *,
    f: int,
    theta: Optional[float] = None,
) -> dict:
    """Return a verification record for a (x_1, x_2, x_3) Householder triple.

    Keys::

        "q_sum_ok"          : q(x_1) + q(x_2) + q(x_3) == 2·3^{2f}
        "q_x1", "q_x2", "q_x3"  : individual q-values
        "q_total"           : their sum
        "q_budget"          : 2·3^{2f}
        "sigma1_x1", ...    : σ_1 of each, as Python complex
        "u_norm_sq"         : approximate ``||u||²`` in the σ_1 embedding
                              (should equal 2.0 to numerical precision when
                              the q-sum is exact)
        "frob"              : (only when ``theta`` is provided) Frobenius
                              residual ``||X·H - R_z(θ)||_F`` — see
                              :func:`householder_frobenius`.
    """
    f = int(f)
    q1, q2, q3 = q_form(x_1), q_form(x_2), q_form(x_3)
    budget = 2 * 3 ** (2 * f)
    s1_x1 = _sigma_1(x_1)
    s1_x2 = _sigma_1(x_2)
    s1_x3 = _sigma_1(x_3)
    # σ_1 L²-norm of u; the *ring* norm ||u||² = q_total / 3^{2f}, which
    # equals 2 iff q_sum_ok. The σ_1 norm need NOT equal 2 (the other two
    # embeddings σ_2, σ_4 carry the balance).
    sigma1_u_norm_sq = (
        abs(s1_x1) ** 2 + abs(s1_x2) ** 2 + abs(s1_x3) ** 2
    ) / (3.0 ** (2 * f))
    rec = {
        "q_x1": q1,
        "q_x2": q2,
        "q_x3": q3,
        "q_total": q1 + q2 + q3,
        "q_budget": budget,
        "q_sum_ok": (q1 + q2 + q3) == budget,
        "sigma1_x1": s1_x1,
        "sigma1_x2": s1_x2,
        "sigma1_x3": s1_x3,
        "sigma1_u_norm_sq": sigma1_u_norm_sq,
    }
    if theta is not None:
        rec["frob"] = householder_frobenius(x_1, x_2, x_3, theta=theta, f=f)
    return rec


def solve_x2_x3(
    x_1: Tuple[int, ...],
    theta: float,
    f: int,
    eps: float,
    *,
    max_pairs: int = 100,
    max_x3: int = 64,
    max_m_triples_per_x3: int = 1024,
    worker: Optional[_NormEqWorker] = None,
    c: float = 1.0,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Find ``(x_2, x_3)`` pairs completing the Householder reflector.

    For each ``x_3`` from :func:`enumerate_x3`, computes
    ``N = 2·3^{2f} - q(x_1) - q(x_3)`` and dispatches to
    :func:`solve_q_norm` with ``embedding_target = -3^f`` (the HRSA
    Householder convention: row 1 of the target is ``[e^{iθ/2}, -1, 0]``).

    Each returned ``(x_2, x_3)`` exactly satisfies
    ``q(x_1) + q(x_2) + q(x_3) = 2·3^{2f}``; this is verified via
    :func:`q_form`.

    Parameters
    ----------
    x_1 : 6-tuple of int. Candidate produced by Phase 2 ``babai_x1``.
    theta, f : as elsewhere. ``theta`` does not enter the math here (Phase 4
        consumes it to build the rotation reify), but is passed through for
        consistency with the rest of the pipeline.
    eps : *Frobenius* tolerance on ``||X·H - R_z(θ)||_F``. Internally we
        enumerate candidates within a moderately wider σ_1 band
        (= ``eps`` itself, not the HRSA contraction-tightened
        ``eps/(2 c √2)``) so that more candidates are returned at low f;
        candidates are *ranked by actual Frobenius residual* via
        :func:`householder_frobenius`, so a stricter caller can filter
        after the fact. At the algorithm's working point (f=17-20,
        ε=10⁻⁸) the σ_1 phase resolution dominates and either band gives
        the same set of hits.
    max_pairs : upper bound on returned ``(x_2, x_3)`` pairs.
    max_x3 : upper bound on x_3 candidates explored.
    max_m_triples_per_x3 : per-x_3 cap on (m_0, m_1, m_2) PARI calls.
    worker : optional pre-warmed worker (recommended in tight loops).
    c : Householder contraction factor (HRSA default 1.0).

    Returns
    -------
    list of ``(x_2, x_3)`` tuples, ordered ascending by the **actual**
    Householder Frobenius residual ``||X·H - R_z(θ)||_F``.
    """
    if f < 0:
        raise ValueError(f"f must be ≥ 0, got {f}")
    x_1 = tuple(int(v) for v in x_1)
    if len(x_1) != 6:
        raise ValueError(f"x_1 must be length-6, got {x_1!r}")

    q_budget = 2 * 3 ** (2 * f)
    q1 = q_form(x_1)
    if q1 > q_budget:
        return []

    # Per HRSA's contraction analysis (householder_search.h), bounding
    # each |x_i / 3^f - target_i| ≤ ε/(2 c √2) guarantees the row-1
    # Frobenius residual ‖X·H − R_z(θ)‖_F ≤ ε. We enumerate inside that
    # tighter band so the returned pairs actually pass the Frob test.
    # (A wider band would return more candidates but their Frob would
    # generically exceed ε; the user can always re-call solve_q_norm
    # directly with a different ε if needed.)
    eps_coord = eps / (2.0 * c * math.sqrt(2.0))

    x3_cands = enumerate_x3(theta, f, eps_coord, max_candidates=max_x3)
    if not x3_cands:
        return []

    owns_worker = worker is None
    if owns_worker:
        worker = _NormEqWorker()
        worker.start()

    # x_2 should sit near -3^f (real, negative) in σ_1.
    embedding_target_x2 = complex(-(3.0 ** f), 0.0)

    pairs: List[Tuple[float, Tuple[int, ...], Tuple[int, ...]]] = []
    try:
        for x3 in x3_cands:
            q3 = q_form(x3)
            N = q_budget - q1 - q3
            if N < 0:
                continue
            x2_cands = solve_q_norm(
                N=N,
                embedding_target=embedding_target_x2,
                eps=eps_coord,
                f=f,
                worker=worker,
                max_triples=max_m_triples_per_x3,
            )
            for x2 in x2_cands:
                # Defensive cross-check that the Householder norm equation
                # holds exactly. solve_q_norm already filters by q(x_2) == N
                # so this is a belt-and-suspenders consistency assertion.
                if q_form(x2) + q3 + q1 != q_budget:
                    continue
                frob = householder_frobenius(x_1, x2, x3, theta=theta, f=f)
                pairs.append((frob, x2, x3))
                if len(pairs) >= 4 * max_pairs:
                    break
            if len(pairs) >= 4 * max_pairs:
                break
    finally:
        if owns_worker:
            worker.close()

    pairs.sort(key=lambda x: x[0])
    return [(x2, x3) for _, x2, x3 in pairs[:max_pairs]]


# ---------------------------------------------------------------------------
# Module entry point — sanity smoke test
# ---------------------------------------------------------------------------


def _main():  # pragma: no cover - manual sanity
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test cvp.diophantine.solve_x2_x3."
    )
    parser.add_argument("--theta", type=float, default=math.pi / 2)
    parser.add_argument("--f", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--n-x1", type=int, default=5)
    args = parser.parse_args()

    from cvp import babai_x1

    t0 = time.time()
    x1_cands = babai_x1(args.theta, args.f, n_candidates=args.n_x1, eps=args.eps)
    print(f"|x1_cands| = {len(x1_cands)}  ({time.time()-t0:.2f}s)")

    with _NormEqWorker() as worker:
        print(f"  worker startup = {worker.startup_seconds:.2f}s")
        for k, x1 in enumerate(x1_cands):
            t1 = time.time()
            pairs = solve_x2_x3(
                x_1=x1, theta=args.theta, f=args.f, eps=args.eps,
                max_pairs=10, worker=worker,
            )
            dt = time.time() - t1
            print(
                f"  [{k}] x1=q={q_form(x1)}: |pairs|={len(pairs)} "
                f"({dt:.2f}s, {worker.n_calls} PARI total)"
            )
            for j, (x2, x3) in enumerate(pairs[:3]):
                rec = verify_pair(x1, x2, x3, f=args.f)
                print(
                    f"      ({j}) q_ok={rec['q_sum_ok']} "
                    f"σ_1(x_2)={rec['sigma1_x2']:.3f} "
                    f"σ_1(x_3)={rec['sigma1_x3']:.3f}"
                )


if __name__ == "__main__":  # pragma: no cover
    _main()
