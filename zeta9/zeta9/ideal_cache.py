"""Ideal-first enumeration for the qutrit Clifford+D stage-1 cache.

Background
==========
The lattice-first approach (collect_targets.py) walks (n1, n2) ∈ a polygon
bounding box, builds M = n0 + n1·α + n2·α² ∈ O_F, filters by σ-strip, and
screens for "M is a norm from K" via parity of K/F-inert valuations.

The ideal-first alternative: enumerate elements γ ∈ O_K directly (up to units
of K), compute M = γγ̄ = N_{K/F}(γ) ∈ O_F, then check the σ-strip. Every M
so produced is automatically a K-norm (screen always passes).

For K = Q(ζ_9), CM extension of F = Q(α), α = ζ_9 + ζ_9⁻¹:
- [K:F] = 2, K/F nontrivial automorphism = complex conjugation
- σ_F_r(M) = σ_F_r(γγ̄) = |σ_K_r(γ)|² for the 3 real embeddings of F
  (each real embedding of F lifts to a conjugate pair in K)
- Unit group of K: rank = r1 + r2 - 1 = 2 (Dirichlet); finite torsion 18 (= 2·9)

Algorithm (incremental-generator DFS):
1. One-time: for each rational prime p ≤ √B (B = max ideal norm we want),
   factor (p) in O_K, get prime ideals P_i, compute generators π_{P_i}.
2. DFS-enumerate composite ideals as products of prime ideals (with ordering
   to avoid duplicates), maintaining γ = product of π's incrementally.
3. For each enumerated γ, store (coefficients, log|σ_K_r(γ)| for r in {1,2,4}, N).

At cache-query time, given a cell (f, u, ε, θ):
4. Compute target σ_F_1 = t (= 3^(2f)·u·cos²(θ/2) or similar — comes from the
   row of the target unitary), and σ_F_strip constraints on σ_F_2, σ_F_3.
5. For each cached γ, search the 2-d log-unit lattice for a unit u ∈ O_K* such
   that 2·log|σ_K_r(γ·u)| satisfies σ_F constraints. This is a 2-d closest-
   lattice-point problem (Babai's NP) per cached γ.
6. Output the M = γγ̄ values (in O_F basis) that fit. Format compatible with
   collect_targets' Y1_*.npy.

Design notes
============
- Backing store: cypari2 (direct PARI C wrapping). Sage is 2-5× slower per call;
  python-flint lacks bnfisprincipal / idealprimedec at the level we need.
- Cache file format: binary, packed. Per-ideal record = 6 int32 (γ coefficients
  in integral basis of O_K) + 3 float64 (log|σ_K_r| for r=1,2,4) + 1 int64
  (N_K/Q(γ)). Total 56 bytes per ideal. For 2×10⁸ ideals: ~11 GB.
- Sorted by N_K/Q for range queries.
- Fundamental units precomputed once and stored separately.
- DFS uses prime-ordering to avoid duplicate ideal enumeration.

This module is the BUILD + READ + FIT machinery. The Y1 output path is in
query_ideal_cache.py which uses this.
"""
from __future__ import annotations

import os
import struct
import math
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# K = Q(ζ_9) setup
# ---------------------------------------------------------------------------
# Defining polynomial: Φ_9(x) = x^6 + x^3 + 1
# Embeddings (image of x = ζ_9): ζ_9^k for k ∈ {1, 2, 4, 5, 7, 8}
# Conjugate pairs: (1, 8), (2, 7), (4, 5)
# Real subfield F has 3 real embeddings α → 2cos(2π·k/9) for k ∈ {1, 2, 4}.

K_DEFINING_POLY = "x^6 + x^3 + 1"

# Indices into PARI's 6 complex embeddings (sorted), one per conjugate pair.
# We pick k ∈ {1, 2, 4} — the "positive half" — and use |σ_K_k|² for the
# 3 real embeddings of F.
EMBEDDING_INDICES_K = [0, 1, 3]  # placeholder; verified at runtime via real bracket

# 56 bytes per cached entry:
#   6 × int32  = 24 bytes   (γ coefficients in integral basis of O_K)
#   3 × float64 = 24 bytes  (log|σ_K_r(γ)| for r ∈ {1, 2, 4})
#   1 × int64  = 8 bytes    (N_K/Q(γ); fits int64 for our bounds)
IDEAL_RECORD_DTYPE = np.dtype([
    ("coefs", np.int32, 6),
    ("logsigma", np.float64, 3),
    ("norm", np.int64),
])
assert IDEAL_RECORD_DTYPE.itemsize == 56


CACHE_MAGIC = b"Z9IC"  # "zeta9 ideal cache"
CACHE_VERSION = 1
# Fields: magic (4s), version (B), 3 bytes padding, n_ideals (Q), norm_bound (Q), prime_bound (Q)
# Total: 4 + 1 + 3 + 8 + 8 + 8 = 32 bytes.
CACHE_HEADER_FMT = "<4sBxxxQQQ"
CACHE_HEADER_SIZE = struct.calcsize(CACHE_HEADER_FMT)
assert CACHE_HEADER_SIZE == 32


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

@dataclass
class PrimeEntry:
    """One prime ideal of K above some rational prime, with generator."""
    p: int                # rational prime
    norm: int             # N(P) = p^f_P
    gen_coefs: np.ndarray # generator in integral basis (length 6, int)
    log_sigma: np.ndarray # log|σ_K_r(π_P)| for r ∈ {1, 2, 4} (length 3, float64)


class IdealCacheBuilder:
    """Build the ideal cache by DFS over products of prime ideals.

    Two-stage:
      Stage A: enumerate rational primes p ≤ p_bound, factor (p)O_K, get
               generator + log-embeddings for each prime ideal. Persist this
               prime table.
      Stage B: DFS-enumerate composite ideals up to norm_bound by multiplying
               primes in order. Maintain γ incrementally so we never call
               bnfisprincipal except once per prime ideal in Stage A.

    Use:
        builder = IdealCacheBuilder(norm_bound=2_000_000)
        builder.build_prime_table()
        builder.build_ideal_cache("path/to/cache.bin")

    Performance: for K=Q(ζ_9), N(J) ≤ 2×10⁹ gives ~6×10⁸ principal ideals.
    Per-ideal cost ~15 µs (after incremental-gen optimization). Total ~3-4 hr
    single-threaded; ~30 min if prime-table build is parallelized over disjoint
    prime ranges.
    """

    def __init__(self, norm_bound: int, prime_bound: Optional[int] = None,
                 pari_stack_bytes: int = 1 << 33):
        self.norm_bound = int(norm_bound)
        # Primes p > norm_bound can't appear in any ideal of norm ≤ norm_bound.
        # Tighter: primes p such that N(P|p) ≤ norm_bound. For K = Q(ζ_9),
        # the smallest N(P) for given p is p (when P is split-deg-1, p ≡ 1 mod 9),
        # so prime_bound = norm_bound is the correct upper bound for the SEARCH.
        # In practice most primes contribute deg-6 ideals (p^6 > norm_bound for
        # tiny p) so the effective search space is much smaller.
        self.prime_bound = int(prime_bound or norm_bound)
        self.pari_stack_bytes = pari_stack_bytes

        self._pari = None  # lazy-init
        self._K = None     # bnf object
        self._primes: list[PrimeEntry] = []

    # ---- PARI setup ----

    def _init_pari(self):
        if self._pari is not None:
            return
        import cypari2
        self._pari = cypari2.Pari()
        self._pari.allocatemem(self.pari_stack_bytes)
        # bnfinit: full number-field setup with class group + units.
        # Class number of Q(ζ_9) is 1 (verified), so bnfisprincipal always returns
        # a generator. Regulator small.
        self._K = self._pari.bnfinit(K_DEFINING_POLY)

    # ---- Stage A: prime table ----

    def build_prime_table(self, save_path: Optional[str] = None):
        """Enumerate primes p ≤ self.prime_bound; factor each in K; get
        generator and log-embeddings of each prime ideal. Result stored on
        self._primes (sorted by norm).

        If save_path is given, persist as a .npz for resume / inspection.
        """
        self._init_pari()
        pari = self._pari
        K = self._K

        # Iterate rational primes in order. PARI: primes(n) returns first n
        # primes; for streaming use forprime. We'll use Python's sympy.primerange
        # if available, else PARI.
        try:
            from sympy import primerange
            prime_iter = primerange(2, self.prime_bound + 1)
        except ImportError:
            # Fallback to PARI
            def prime_iter_pari():
                p = pari(1)
                while True:
                    p = pari.nextprime(p + 1)
                    if int(p) > self.prime_bound:
                        return
                    yield int(p)
            prime_iter = prime_iter_pari()

        self._primes = []
        for p in prime_iter:
            # Factor (p) in O_K. Returns list of prime ideals.
            prime_ideals = pari.idealprimedec(K, p)
            for P in prime_ideals:
                # N(P) = p^f_P. PARI: prime ideal has format [p, alpha, e, f, b].
                # We extract f_P (index 4 in PARI's 1-indexed convention).
                # Equivalent: pari.idealnorm(K, P) but more expensive.
                f_P = int(P[3])
                N_P = p ** f_P
                if N_P > self.norm_bound:
                    continue  # this prime ideal is too big to ever appear

                # Get generator of P: bnfisprincipal returns (cl, gen) where
                # cl is the class (zero for class number 1) and gen is in
                # PARI's integral basis as a column vector. PARI's basis is a
                # permutation of standard (1, ω, ..., ω⁵): permute to standard.
                _cl, gen = pari.bnfisprincipal(K, P)
                # PARI_INV_PERM = [0, 2, 4, 1, 3, 5]: standard[j] = pari[PARI_INV_PERM[j]]
                _pari_inv = (0, 2, 4, 1, 3, 5)
                gen_coefs = np.array([int(gen[_pari_inv[j]]) for j in range(6)],
                                     dtype=np.int32)

                # log|σ_K_r(π_P)| for r ∈ {1, 2, 4}.
                # PARI: nfeltembed returns the complex embeddings.
                # The 6 embeddings are in fixed order; we pick indices for the 3
                # conjugate-pair representatives.
                log_sigma = self._compute_log_embeddings_K(gen_coefs)

                self._primes.append(PrimeEntry(
                    p=int(p), norm=int(N_P),
                    gen_coefs=gen_coefs, log_sigma=log_sigma,
                ))

        # Sort by norm so DFS produces ideals in roughly increasing norm order
        # (helpful for diagnostics; not required for correctness).
        self._primes.sort(key=lambda e: e.norm)

        if save_path is not None:
            self._save_prime_table(save_path)

    def _save_prime_table(self, path: str):
        coefs = np.stack([e.gen_coefs for e in self._primes])
        logs = np.stack([e.log_sigma for e in self._primes])
        ps = np.array([e.p for e in self._primes], dtype=np.int64)
        norms = np.array([e.norm for e in self._primes], dtype=np.int64)
        np.savez(path, coefs=coefs, logs=logs, p=ps, norm=norms)

    def load_prime_table(self, path: str):
        data = np.load(path)
        coefs = data["coefs"]
        logs = data["logs"]
        ps = data["p"]
        norms = data["norm"]
        self._primes = [
            PrimeEntry(
                p=int(ps[i]), norm=int(norms[i]),
                gen_coefs=coefs[i], log_sigma=logs[i],
            )
            for i in range(len(ps))
        ]

    def _compute_log_embeddings_K(self, gen_coefs: np.ndarray) -> np.ndarray:
        """Compute log|σ_K_r(γ)| for r ∈ {1, 2, 4} where γ is given in
        integral basis of O_K.

        The integral basis of O_K = Z[ζ_9] is (1, ζ_9, ζ_9², ζ_9³, ζ_9⁴, ζ_9⁵)
        (powers below 6 since Φ_9 has degree 6).

        σ_K_k maps ζ_9 → ζ_9^k for k ∈ {1, 2, 4, 5, 7, 8} (units mod 9).
        We pick k ∈ {1, 2, 4} (one per conjugate pair).
        """
        zeta = np.exp(2j * np.pi / 9)
        # σ_K_k applied to (1, ζ_9, ..., ζ_9^5)
        log_sigma = np.empty(3, dtype=np.float64)
        for idx, k in enumerate([1, 2, 4]):
            # σ_K_k(γ) = Σ_j gen_coefs[j] · ζ_9^{k·j}
            sigma_zeta_k = zeta ** k
            powers = np.array([sigma_zeta_k ** j for j in range(6)])
            val = np.dot(gen_coefs, powers)
            log_sigma[idx] = math.log(abs(val))
        return log_sigma

    # ---- Stage B: DFS over composite ideals ----

    def build_ideal_cache(self, out_path: str, progress_every: int = 100_000):
        """DFS through products of self._primes. For each composite ideal J,
        compute γ = Π π_P (incremental multiplication) and write a cache record.

        Writes to a single binary file with header + packed records.
        """
        if not self._primes:
            raise RuntimeError("Call build_prime_table first.")
        self._init_pari()

        # Setup output file: write placeholder header, then records.
        # We'll come back and patch the header with the actual count.
        n_primes = len(self._primes)
        norm_bound = self.norm_bound
        prime_bound = self.prime_bound

        with open(out_path, "wb") as fh:
            # Write placeholder header
            header = struct.pack(
                CACHE_HEADER_FMT,
                CACHE_MAGIC,
                CACHE_VERSION,
                0,  # n_ideals — patch later
                norm_bound,
                prime_bound,
            )
            fh.write(header)

            # Buffer for batched writes
            buf_size = 65536
            buf = np.empty(buf_size, dtype=IDEAL_RECORD_DTYPE)
            buf_idx = 0
            n_ideals = 0

            for record in self._iter_composite_ideals(progress_every=progress_every):
                buf[buf_idx] = record
                buf_idx += 1
                if buf_idx == buf_size:
                    buf[:buf_idx].tofile(fh)
                    buf_idx = 0
                n_ideals += 1

            # Flush remainder
            if buf_idx > 0:
                buf[:buf_idx].tofile(fh)

            # Patch header
            fh.seek(0)
            header = struct.pack(
                CACHE_HEADER_FMT,
                CACHE_MAGIC,
                CACHE_VERSION,
                n_ideals,
                norm_bound,
                prime_bound,
            )
            fh.write(header)

        return n_ideals

    def _iter_composite_ideals(self, progress_every: int = 100_000) -> Iterator:
        """DFS over multi-sets of prime ideals (with their indices) whose
        product has norm ≤ norm_bound. Yields one (coefs, log_sigma, norm)
        record per composite ideal (including the trivial (1) ideal).
        """
        self._init_pari()
        pari = self._pari
        K = self._K
        n_primes = len(self._primes)
        norm_bound = self.norm_bound

        # Identity element: γ = 1, log|σ_K_r(1)| = 0, N = 1.
        # In integral basis: (1, 0, 0, 0, 0, 0).
        # PARI nfelt for "1": col vector [1, 0, 0, 0, 0, 0].
        id_coefs = np.zeros(6, dtype=np.int32); id_coefs[0] = 1
        id_log = np.zeros(3, dtype=np.float64)

        # Pre-convert prime generators to PARI t_COL for fast nfeltmul.
        # gen_coefs are in STANDARD basis (1, ω, ..., ω⁵); PARI nfeltmul needs
        # them in PARI's internal basis [1, ω³, ω, ω⁴, ω², ω⁵]. Permute.
        # standard_idx → pari_idx: PARI_PERM[i] gives standard_idx for pari_idx i.
        # I.e. pari_coefs[i] = standard_coefs[PARI_PERM[i]].
        _pari_perm = (0, 3, 1, 4, 2, 5)
        _pari_inv = (0, 2, 4, 1, 3, 5)  # standard[j] = pari[_pari_inv[j]]

        def standard_to_pari_col(std_coefs):
            return pari.Col([int(std_coefs[_pari_perm[i]]) for i in range(6)])

        def pari_to_standard(pari_obj):
            """Handle both t_COL (length-6) and t_INT (= rational integer).
            Returns shape (6,) int64 array in standard basis."""
            try:
                # t_COL case: 6 integers in PARI basis
                return np.array(
                    [int(pari_obj[_pari_inv[j]]) for j in range(6)],
                    dtype=np.int64,
                )
            except (TypeError, IndexError):
                # t_INT case: PARI simplified to scalar; γ = integer = std[0]
                return np.array([int(pari_obj), 0, 0, 0, 0, 0],
                                dtype=np.int64)

        prime_gens_pari = [
            standard_to_pari_col(e.gen_coefs) for e in self._primes
        ]

        # DFS frame: (idx, std_coefs, gamma_log, gamma_norm).
        # std_coefs is np.int64[6] in standard basis (1, ω, ..., ω⁵).
        # We constrain ourselves to non-decreasing prime indices to avoid
        # enumerating the same multiset twice.
        progress_count = 0
        id_std = np.zeros(6, dtype=np.int64); id_std[0] = 1
        stack = [(0, id_std, id_log.copy(), 1)]

        # Pre-build IDEAL_RECORD_DTYPE for the yields
        rec_buf = np.empty(1, dtype=IDEAL_RECORD_DTYPE)

        while stack:
            idx, gamma_std, gamma_log, gamma_norm = stack.pop()

            rec_buf[0]["coefs"] = gamma_std.astype(np.int32)
            rec_buf[0]["logsigma"] = gamma_log
            rec_buf[0]["norm"] = gamma_norm
            yield rec_buf[0].copy()

            progress_count += 1
            if progress_every and progress_count % progress_every == 0:
                print(f"  enumerated {progress_count:,} ideals so far...", flush=True)

            # Extend with primes from index idx onward (so multisets are sorted).
            for j in range(idx, n_primes):
                P = self._primes[j]
                new_norm = gamma_norm * P.norm
                if new_norm > norm_bound:
                    # Primes are sorted by norm; further j have norm ≥ P.norm,
                    # so they all blow the budget too. Break.
                    break

                # γ_new = γ · π_P : multiply via PARI (handles the polynomial
                # multiplication + reduction mod K_poly); convert std → pari →
                # multiply → std.
                gamma_pari = standard_to_pari_col(gamma_std)
                new_gamma_pari = pari.nfeltmul(K, gamma_pari, prime_gens_pari[j])
                new_gamma_std = pari_to_standard(new_gamma_pari)
                new_log = gamma_log + P.log_sigma

                stack.append((j, new_gamma_std, new_log, new_norm))


# ---------------------------------------------------------------------------
# Cache reader
# ---------------------------------------------------------------------------

class IdealCacheReader:
    """Memory-mapped reader for the cache file. Provides slice access and
    range queries on N or log_sigma.

    Use:
        cache = IdealCacheReader("path/to/cache.bin")
        print(cache.n_ideals, cache.norm_bound)
        for entry in cache.iter_in_norm_range(1e10, 1e12):
            ...
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            header = fh.read(CACHE_HEADER_SIZE)
        magic, version, n_ideals, norm_bound, prime_bound = struct.unpack(
            CACHE_HEADER_FMT, header,
        )
        if magic != CACHE_MAGIC:
            raise ValueError(f"Bad cache magic: {magic!r}")
        if version != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version: {version}")

        self.n_ideals = int(n_ideals)
        self.norm_bound = int(norm_bound)
        self.prime_bound = int(prime_bound)

        # mmap the records (skip header)
        self._data = np.memmap(
            path, dtype=IDEAL_RECORD_DTYPE, mode="r",
            offset=CACHE_HEADER_SIZE, shape=(self.n_ideals,),
        )

    @property
    def records(self) -> np.ndarray:
        """The full memmap array of records (read-only)."""
        return self._data

    def iter_in_norm_range(self, lo: float, hi: float) -> Iterator:
        """Yield records with lo ≤ N_K/Q(γ) ≤ hi.
        Cache isn't sorted by default; this scans. For sorted cache, see
        sort_by_norm() helper."""
        mask = (self._data["norm"] >= lo) & (self._data["norm"] <= hi)
        for rec in self._data[mask]:
            yield rec


# ---------------------------------------------------------------------------
# σ-fitter — given cell (f, u, ε, θ), find unit translates of cached γ's
# whose M = γγ̄ satisfies the σ-strip constraints.
# ---------------------------------------------------------------------------

class SigmaFitter:
    """Fit cached γ generators to a target σ-strip via unit-orbit translation.

    For each γ, the unit orbit in log-embedding space is {L(γ) + k_1·L(u_1)
    + k_2·L(u_2) : k ∈ Z²} where u_1, u_2 are fundamental units of O_K.

    The constraint: for M = γγ̄ ∈ O_F,
        σ_F_r(M) = exp(2·log|σ_K_r(γ)|)
    must satisfy:
        σ_F_1(M) ∈ [target_t - epsY_hi, target_t + epsY_hi]  (the σ-strip)
        σ_F_2(M) ∈ [0, total_scale]
        σ_F_3(M) ∈ [0, total_scale]

    Equivalently in log-space, with L_r = log|σ_K_r(γ·u)|:
        2L_1 ∈ [log(target_t - epsY_hi), log(target_t + epsY_hi)]
        2L_2 ≤ log(total_scale)
        2L_3 ≤ log(total_scale)

    The first is a NARROW band on L_1; the others are loose upper bounds.

    Search: 2D lattice rounding (Babai's nearest plane) gets candidates near
    the band centerline; we then enumerate a small box around the rounded
    point to catch all integer solutions within the band.
    """

    def __init__(self, fund_unit_logs: np.ndarray, torsion_order: int = 18):
        """fund_unit_logs: shape (2, 3) — log|σ_K_r(u_i)| for r ∈ {1,2,4}, i ∈ {1,2}.
        Must be computed once for K and saved alongside the cache."""
        self.U = np.asarray(fund_unit_logs, dtype=np.float64)  # (2, 3)
        assert self.U.shape == (2, 3)
        self.torsion_order = int(torsion_order)

        # Precompute pseudoinverse for 2D lattice rounding.
        # Constraint sums: 2(L_1 + L_2 + L_3) = log(N_K/Q(γ)) is fixed
        # (units have norm ±1, so this is invariant under multiplication).
        # The unit logs lie in the 2D subspace {(a,b,c) : a+b+c=0}.
        # We project to a 2D coordinate system via any 2 of (L_1 - mean, L_2 - mean, L_3 - mean).
        self.U_pinv = np.linalg.pinv(self.U.T)  # (3, 2)

    def fit(self, gamma_log: np.ndarray, target_log_2L1: float,
            tol_log_2L1: float, max_log_2L2: float, max_log_2L3: float,
            k_radius: int = 3) -> list[tuple[int, int]]:
        """Find integer (k_1, k_2) such that
            L(γ·u) = γ_log + k_1·U[0] + k_2·U[1]
        satisfies:
            2 L_1 in [target_log_2L1 - tol_log_2L1, target_log_2L1 + tol_log_2L1]
            2 L_2 ≤ max_log_2L2
            2 L_3 ≤ max_log_2L3

        Returns list of (k_1, k_2) tuples (may be empty).
        k_radius bounds the integer search box around the rounded center.
        """
        # First constraint: 2 L_1 ≈ target_log_2L1 fixes a 1D LINE in (k_1, k_2)
        # space. Within tolerance: a NARROW BAND.
        # 2 (γ_log[0] + k_1·U[0,0] + k_2·U[1,0]) = target_log_2L1
        # → k_1·U[0,0] + k_2·U[1,0] = (target_log_2L1/2) - γ_log[0]
        rhs = target_log_2L1 * 0.5 - gamma_log[0]
        a, b = self.U[0, 0], self.U[1, 0]

        # Parameterize: pick the closest (k_1, k_2) ∈ Z² to the line a·k_1 + b·k_2 = rhs
        # Center of line: t_center = rhs / (a² + b²) × (a, b)
        denom = a * a + b * b
        if abs(denom) < 1e-30:
            # Both units have 0 effect on σ_K_1 — degenerate; can't satisfy band
            return []
        k1_center = rhs * a / denom
        k2_center = rhs * b / denom

        # Search small integer box around (k1_center, k2_center).
        hits = []
        for dk1 in range(-k_radius, k_radius + 1):
            for dk2 in range(-k_radius, k_radius + 1):
                k1 = int(round(k1_center)) + dk1
                k2 = int(round(k2_center)) + dk2
                # Compute shifted log
                shifted = gamma_log + k1 * self.U[0] + k2 * self.U[1]
                two_L1, two_L2, two_L3 = 2.0 * shifted

                if not (target_log_2L1 - tol_log_2L1 <= two_L1 <= target_log_2L1 + tol_log_2L1):
                    continue
                if two_L2 > max_log_2L2:
                    continue
                if two_L3 > max_log_2L3:
                    continue
                hits.append((k1, k2))

        return hits


# ---------------------------------------------------------------------------
# Utilities: K's fundamental units (precompute once)
# ---------------------------------------------------------------------------

def compute_fundamental_unit_data(pari_stack_bytes: int = 1 << 32) -> dict:
    """Compute fundamental-unit data for K = Q(ζ_9):
      - coefs: (2, 6) int64 — u_i in integral basis (1, ω, ..., ω^5)
      - inv_coefs: (2, 6) int64 — u_i⁻¹ in integral basis
      - logs: (2, 3) float64 — log|σ_K_r(u_i)| for r ∈ {1, 2, 4}

    Called once; result should be cached on disk via save_fund_unit_data().
    """
    import cypari2
    pari = cypari2.Pari()
    pari.allocatemem(pari_stack_bytes)
    K = pari.bnfinit(K_DEFINING_POLY)

    # PARI uses a permuted integral basis. For K = Q(ζ_9) defined by Φ_9 = x^6
    # + x^3 + 1, PARI's basis is [1, ω³, ω, ω⁴, ω², ω⁵] (verified by
    # `nfinit(x^6+x^3+1).zk`). Our code uses the standard basis (1, ω, ω², ω³,
    # ω⁴, ω⁵). Need to permute when crossing the boundary.
    #
    # pari_basis_idx → standard_idx:  PARI_PERM = [0, 3, 1, 4, 2, 5]
    # standard_idx → pari_basis_idx:  PARI_INV_PERM = [0, 2, 4, 1, 3, 5]
    PARI_INV_PERM = [0, 2, 4, 1, 3, 5]

    def pari_basis_to_standard(pari_coefs):
        """pari_coefs (6,) in PARI's integral basis → standard (1, ω, ..., ω⁵)."""
        return np.array([pari_coefs[PARI_INV_PERM[j]] for j in range(6)],
                        dtype=np.int64)

    # Get fundamental units as polmods (e.g. Mod(x+1, K_poly), Mod(x²+1, ...)).
    # cypari2 doesn't expose K.fu as attribute; use member access via eval
    # (one-time cost — runs bnfinit twice but only at cache setup).
    units = pari(f"bnfinit({K_DEFINING_POLY}).fu")

    n_units = len(units)
    if n_units != 2:
        raise RuntimeError(
            f"Expected 2 fundamental units for Q(ζ_9), got {n_units}. "
            f"PARI returned units: {list(units)}"
        )

    out_coefs = np.empty((2, 6), dtype=np.int64)
    out_inv_coefs = np.empty((2, 6), dtype=np.int64)
    out_logs = np.empty((2, 3), dtype=np.float64)

    zeta = np.exp(2j * np.pi / 9)
    for i, u in enumerate(units):
        # u is a polmod. nfalgtobasis returns coefs in PARI's integral basis.
        u_pari_basis = pari.nfalgtobasis(K, u)
        u_pari_coefs = np.array([int(u_pari_basis[j]) for j in range(6)],
                                dtype=np.int64)

        # u⁻¹ via nfeltdiv (works with PARI-basis inputs, returns PARI-basis).
        u_inv_pari = pari.nfeltdiv(K, pari.Col([1, 0, 0, 0, 0, 0]),
                                    u_pari_basis)
        u_inv_pari_coefs = np.array([int(u_inv_pari[j]) for j in range(6)],
                                    dtype=np.int64)

        # Convert both to standard (1, ω, ω², ω³, ω⁴, ω⁵) basis
        u_coefs = pari_basis_to_standard(u_pari_coefs)
        u_inv_coefs = pari_basis_to_standard(u_inv_pari_coefs)
        out_coefs[i] = u_coefs
        out_inv_coefs[i] = u_inv_coefs

        # log|σ_K_r(u)| for r ∈ {1, 2, 4} (computed in standard basis)
        for idx, k in enumerate([1, 2, 4]):
            sigma_zeta = zeta ** k
            powers = np.array([sigma_zeta ** j for j in range(6)])
            val = np.dot(u_coefs, powers)
            out_logs[i, idx] = math.log(abs(val))

    return {
        "coefs": out_coefs,
        "inv_coefs": out_inv_coefs,
        "logs": out_logs,
    }


def save_fund_unit_data(data: dict, path: str):
    """Save fundamental-unit data to a .npz file."""
    np.savez(path, coefs=data["coefs"], inv_coefs=data["inv_coefs"], logs=data["logs"])


def load_fund_unit_data(path: str) -> dict:
    """Load fundamental-unit data from .npz file (or .npy for legacy)."""
    if path.endswith(".npy"):
        # legacy: only logs were saved
        logs = np.load(path)
        return {"coefs": None, "inv_coefs": None, "logs": logs}
    data = np.load(path)
    return {
        "coefs": data["coefs"],
        "inv_coefs": data["inv_coefs"],
        "logs": data["logs"],
    }


# Backward-compat alias
def compute_fundamental_unit_logs(pari_stack_bytes: int = 1 << 32) -> np.ndarray:
    """Returns just the logs (legacy API). For coefs + logs, use
    compute_fundamental_unit_data()."""
    return compute_fundamental_unit_data(pari_stack_bytes)["logs"]


# ---------------------------------------------------------------------------
# Element conversion: γ ∈ O_K → M = γγ̄ ∈ O_F
# See ideal_cache_conv.py for the actual implementations:
#   - gamma_to_M_coefs_python(): Python reference
#   - gamma_to_M_batch(): Numba batched kernel
#   - gamma_times_unit_pow_to_M_batch(): Numba kernel for γ·u_1^k_1·u_2^k_2 → M
#   - _mul_in_OK(), mul_in_OK_batch(): multiplication in O_K
#   - compute_unit_power_table(): precompute u^k for k ∈ [-k_max, k_max]
# ---------------------------------------------------------------------------
