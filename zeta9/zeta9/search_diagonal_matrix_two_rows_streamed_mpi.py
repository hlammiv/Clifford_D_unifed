"""Streamed two-row search for a 3x3 diagonal unitary target over Z[zeta_9, 1/3].

This is a deliberately minimal streamed prototype.

Main design choices
-------------------
* Do NOT globally materialize row-1 candidates.
* Stream row-1 norm triples chunk-by-chunk.
* For every row-1 candidate near (d1,0,0), immediately search row-2
  components near (0,d2,0) using exact orthogonality

      A*conj(U) + B*conj(V) + C*conj(W) = 0,
      U = -conj((B*conj(V) + C*conj(W)) / A).

* Apply exact divisibility by A before forming U.
* Apply exact norm check for U.
* Apply exact cross-product same-layer check before output.
* Print a rank-0 heartbeat from the beginning.

The script expects the same rootdb/binned phase sidecar layout used by
fit_vectors_mpi_sidecar_binned_v2.py.  It imports the reliable helper
functions from that module when available.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import heapq
import json
import math
import os
import re
import sys
import time
from fractions import Fraction
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional

import numpy as np
from mpi4py import MPI

# Reuse the infrastructure from the vector fitter.  This file should be placed
# in the same directory as fit_vectors_mpi_sidecar_binned_v2.py, or run with that
# directory on PYTHONPATH.
try:
    from .fit_vectors_mpi_sidecar_binned_v2 import (  # type: ignore
        _state_paths,
        _read_manifest,
        _read_triple_rows,
        _build_locator,
        _load_chunk_metadata,
        _candidate_desc_from_binned_sidecar,
        _iter_desc_z_and_indices,
        _load_coeff_row,
        coeffs_to_complex_noscale,
        sigma1_from_m012,
    )
except ImportError:
    from fit_vectors_mpi_sidecar_binned_v2 import (  # type: ignore
        _state_paths,
        _read_manifest,
        _read_triple_rows,
        _build_locator,
        _load_chunk_metadata,
        _candidate_desc_from_binned_sidecar,
        _iter_desc_z_and_indices,
        _load_coeff_row,
        coeffs_to_complex_noscale,
        sigma1_from_m012,
    )


# -----------------------------------------------------------------------------
# Z[zeta_9] arithmetic in the basis 1,z,z^2,z^3,z^4,z^5.
# Phi_9(z) = z^6 + z^3 + 1, so z^6 = -z^3 - 1.
# -----------------------------------------------------------------------------

I6 = np.eye(6, dtype=np.int64)


def z9_reduce_power_coeff(power: int) -> np.ndarray:
    """Return coefficients of zeta_9**power in the 6-element basis."""
    p = power % 9
    out = np.zeros(6, dtype=np.int64)
    if p <= 5:
        out[p] = 1
    elif p == 6:
        out[0] = -1
        out[3] = -1
    elif p == 7:
        out[1] = -1
        out[4] = -1
    elif p == 8:
        out[2] = -1
        out[5] = -1
    else:  # impossible after mod 9
        raise AssertionError(p)
    return out


_POWER_TABLE = np.vstack([z9_reduce_power_coeff(k) for k in range(11)]).astype(np.int64)

# Numba acceleration for the Z[zeta_9] inner kernels (2026-05-12).
# Stage 5 search spent 80-90% of inner-loop time in z9_mul (Python double-loop
# with NumPy add).  A Numba @njit kernel is 50-200× faster.  See perf audit memo.
try:
    import numba as _nb
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:
    @_nb.njit(cache=True, inline="always")
    def _z9_mul_nb(a, b, power_table):
        out = np.zeros(6, dtype=np.int64)
        for i in range(6):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(6):
                bj = b[j]
                if bj == 0:
                    continue
                prod = ai * bj
                row = power_table[i + j]
                for k in range(6):
                    out[k] += prod * row[k]
        return out

    @_nb.njit(cache=True, inline="always")
    def _z9_conj_nb(a):
        out = np.empty(6, dtype=np.int64)
        out[0] = a[0] - a[3]
        out[1] = -a[2]
        out[2] = -a[1]
        out[3] = -a[3]
        out[4] = -a[2] + a[5]
        out[5] = -a[1] + a[4]
        return out

    @_nb.njit(cache=True)
    def _z9_divisible_by_int_nb(a, q):
        for k in range(6):
            if a[k] % q != 0:
                return False
        return True

    @_nb.njit(cache=True)
    def _z9_norm_m012_nb(x, power_table):
        # conj
        cx = _z9_conj_nb(x)
        # x * conj(x)
        c = _z9_mul_nb(x, cx, power_table)
        m2 = -c[4]
        m1 = -c[5]
        m0 = c[0] + 2 * c[4]
        # consistency check skipped in hot path; caller can verify if needed
        return m0, m1, m2

    # ------------------------------------------------------------------
    # Action 1 (2026-05-12): batched z9_mul kernels.
    #
    # The stage-5 hot loop builds caches like
    #     BVc = z9_mul(B, conj(V)) for V in roots_v
    #     CWc = z9_mul(C, conj(W)) for W in roots_w
    # one at a time, paying ~1 us per element of Numba JIT-dispatch + wrapper
    # overhead.  When B/C is fixed and V/W ranges over an array, batching the
    # multiplication into a single Numba call amortizes the dispatch cost over
    # the whole batch.  Empirically (microbench /tmp/zeta9_batch_bench.py)
    # the per-element cost drops from ~0.85 us to ~0.05 us, a 15-20× win.
    #
    # Two callable variants:
    #   _z9_mul_batch_nb:  serial; safe to call from MPI ranks (which already
    #                      parallelize at a higher level).
    #   _z9_mul_batch_nb_parallel:  same kernel with nb.prange over the batch
    #                      dimension; reserved for single-rank callers.
    # ------------------------------------------------------------------
    @_nb.njit(cache=True)
    def _z9_mul_batch_nb(A, B_batch, power_table, out):
        """Compute out[k] = A * B_batch[k] for k=0..n-1.

        A: (6,) int64
        B_batch, out: (n, 6) int64
        """
        n = B_batch.shape[0]
        for k in range(n):
            # Inline the scalar kernel; the redundancy is intentional so Numba
            # can vectorize the inner loops without function-call overhead.
            for c in range(6):
                out[k, c] = 0
            for i in range(6):
                ai = A[i]
                if ai == 0:
                    continue
                for j in range(6):
                    bj = B_batch[k, j]
                    if bj == 0:
                        continue
                    prod = ai * bj
                    row = power_table[i + j]
                    for c in range(6):
                        out[k, c] += prod * row[c]

    @_nb.njit(cache=True, parallel=True)
    def _z9_mul_batch_nb_parallel(A, B_batch, power_table, out):
        """Parallel version of _z9_mul_batch_nb, prange over batch dim."""
        n = B_batch.shape[0]
        for k in _nb.prange(n):
            for c in range(6):
                out[k, c] = 0
            for i in range(6):
                ai = A[i]
                if ai == 0:
                    continue
                for j in range(6):
                    bj = B_batch[k, j]
                    if bj == 0:
                        continue
                    prod = ai * bj
                    row = power_table[i + j]
                    for c in range(6):
                        out[k, c] += prod * row[c]

    @_nb.njit(cache=True)
    def _z9_conj_batch_nb(B_batch, out):
        """Compute out[k] = conj(B_batch[k]) for k=0..n-1."""
        n = B_batch.shape[0]
        for k in range(n):
            a0 = B_batch[k, 0]
            a1 = B_batch[k, 1]
            a2 = B_batch[k, 2]
            a3 = B_batch[k, 3]
            a4 = B_batch[k, 4]
            a5 = B_batch[k, 5]
            out[k, 0] = a0 - a3
            out[k, 1] = -a2
            out[k, 2] = -a1
            out[k, 3] = -a3
            out[k, 4] = -a2 + a5
            out[k, 5] = -a1 + a4

    @_nb.njit(cache=True)
    def _z9_mul_conj_batch_nb(A, B_batch, power_table, out):
        """Fused: out[k] = A * conj(B_batch[k]).

        Most common stage-5 pattern: BVc = z9_mul(B, conj(V)) for V in roots_v.
        Saves an intermediate (n, 6) allocation vs separate conj + mul.
        """
        n = B_batch.shape[0]
        for k in range(n):
            # conj(B_batch[k]) inline
            b0 = B_batch[k, 0]
            b1 = B_batch[k, 1]
            b2 = B_batch[k, 2]
            b3 = B_batch[k, 3]
            b4 = B_batch[k, 4]
            b5 = B_batch[k, 5]
            c0 = b0 - b3
            c1 = -b2
            c2 = -b1
            c3 = -b3
            c4 = -b2 + b5
            c5 = -b1 + b4
            # zero out[k]
            for c in range(6):
                out[k, c] = 0
            # multiply A * (c0,c1,c2,c3,c4,c5)
            for i in range(6):
                ai = A[i]
                if ai == 0:
                    continue
                # Unrolled j loop over six conj coefficients.
                cj_vals = (c0, c1, c2, c3, c4, c5)
                for j in range(6):
                    bj = cj_vals[j]
                    if bj == 0:
                        continue
                    prod = ai * bj
                    row = power_table[i + j]
                    for c in range(6):
                        out[k, c] += prod * row[c]

    # ------------------------------------------------------------------
    # 2026-05-12 (afternoon): Numba-ize the now-dominant inner-loop costs
    # that the prior-round (batched z9_mul) migrated the bottleneck to.
    #
    # After the batched-mul work landed, stage-5 inner-loop wall is dominated
    # by:
    #   (A) z9_divide_if_exact_by_a       — ~4.5 us/call Python (float solve +
    #                                       rint + verify-via-z9_mul).
    #   (B) coeff_row_to_complex_scaled +
    #       abs(zu) > eps                 — Python complex embedding + abs.
    #   (C) cross-product divisibility    — z9_sub(z9_mul(B,W), z9_mul(C,V))
    #                                       and three z9_divisible_by_int tests
    #                                       per surviving pair.
    #
    # All three are Numba-friendly (pure integer arithmetic plus a 6x6 float
    # solve for (A)).  The kernels below replicate the Python semantics
    # EXACTLY, including the float-residual tolerance (1e-7) and the exact
    # integer verification step in (A).  They are also fused into a single
    # batched "filter pairs" kernel that processes (BVc, CWc) arrays and
    # emits the surviving (V, W) indices plus the materialized U/Q for each.
    # ------------------------------------------------------------------

    @_nb.njit(cache=True, inline="always")
    def _z9_divide_if_exact_by_a_nb(t, a, inv_ma, power_table, q_out):
        """Numba twin of z9_divide_if_exact_by_a.

        Args:
            t:        (6,) int64       — numerator
            a:        (6,) int64       — divisor (Z[zeta_9] element)
            inv_ma:   (6,6) float64    — precomputed inv(M_left(a))
            power_table: (11, 6) int64 — zeta_9 multiplication reduction table
            q_out:    (6,) int64       — OUT: q if success, untouched otherwise

        Returns:
            True if t = a * q for some q in Z[zeta_9] (q written into q_out);
            False otherwise.
        """
        # q_float = inv_ma @ t.astype(np.float64)
        # max_resid = max(|q_float - rint(q_float)|)
        # If > 1e-7 reject.  Else compute q = rint, verify a * q == t.
        max_resid = 0.0
        for i in range(6):
            s = 0.0
            for j in range(6):
                s += inv_ma[i, j] * float(t[j])
            qi = round(s)  # banker's rounding; matches np.rint for our magnitudes
            r = s - qi
            if r < 0.0:
                r = -r
            if r > max_resid:
                max_resid = r
            q_out[i] = np.int64(qi)
        if max_resid > 1.0e-7:
            return False
        # Verify exact: a * q_out == t (full Z[zeta_9] mul, no shortcuts).
        for c in range(6):
            ac = np.int64(0)
            for i in range(6):
                ai = a[i]
                if ai == 0:
                    continue
                for j in range(6):
                    bj = q_out[j]
                    if bj == 0:
                        continue
                    ac += ai * bj * power_table[i + j, c]
            if ac != t[c]:
                return False
        return True

    @_nb.njit(cache=True)
    def _z9_divide_if_exact_by_a_batch_nb(
        T_batch, a, inv_ma, power_table, Q_out, valid_out
    ):
        """Batched twin of _z9_divide_if_exact_by_a_nb.

        Args:
            T_batch:  (n, 6) int64
            a:        (6,) int64
            inv_ma:   (6,6) float64
            power_table: (11, 6) int64
            Q_out:    (n, 6) int64    — OUT: quotient (valid only where valid_out)
            valid_out:(n,) bool       — OUT: success mask
        """
        n = T_batch.shape[0]
        for k in range(n):
            ok = _z9_divide_if_exact_by_a_nb(
                T_batch[k], a, inv_ma, power_table, Q_out[k]
            )
            valid_out[k] = ok

    # Action B: abs(zu) > eps gate.
    #
    # zu = coeffs_to_complex_noscale(u) / 3^f
    #    = sum_k u[k] * zeta_9^k    /  3^f          (k = 0..5)
    # |zu|^2 = (sum u[k] * cos(2 pi k / 9))^2 + (sum u[k] * sin(2 pi k / 9))^2
    #          / 9^f
    # Gate: |zu| <= eps + 1e-15
    #   <=> |zu|^2 <= (eps + 1e-15)^2 (both >= 0)
    #   <=> |re|^2 + |im|^2 <= (eps + 1e-15)^2 * 9^f
    # We pre-pass (eps_plus_pad)^2 * 9^(f) so the kernel does integer-ish work.

    @_nb.njit(cache=True, inline="always")
    def _z9_abs2_scaled_nb(u, zeta_re, zeta_im):
        """Return |coeffs_to_complex_noscale(u)|^2 as a float.

        zeta_re/zeta_im: length-6 float arrays with cos/sin(2*pi*k/9), k=0..5.
        """
        re = 0.0
        im = 0.0
        for k in range(6):
            uk = float(u[k])
            re += uk * zeta_re[k]
            im += uk * zeta_im[k]
        return re * re + im * im

    @_nb.njit(cache=True)
    def _z9_inner_filter_pairs_nb(
        BVc_arr,         # (n_v, 6) int64
        CWc_arr,         # (n_w, 6) int64
        V_arr,           # (n_v, 6) int64
        W_arr,           # (n_w, 6) int64
        A,               # (6,) int64
        B,               # (6,) int64
        C,               # (6,) int64
        inv_M_A,         # (6,6) float64
        zeta_re,         # (6,) float64
        zeta_im,         # (6,) float64
        abs2_thresh,     # float  — (eps + 1e-15)^2 * 9^f
        scale_int,       # int    — 3^f
        power_table,     # (11, 6) int64
        # Counter outputs (length-1 int64 arrays so Python can read them back):
        ndiv_out,        # (1,) int64 — # pairs passing exact-divide test
        nzu_out,         # (1,) int64 — # pairs passing |zu| gate
        ncross_out,      # (1,) int64 — # pairs passing cross-product div test
        # Match outputs (pre-allocated; capacity = n_v * n_w worst case):
        idx_v_out,       # (cap,) int64
        idx_w_out,       # (cap,) int64
        U_out,           # (cap, 6) int64
    ):
        """Fused inner-loop kernel.

        For each (iv, iw) in n_v x n_w:
          1. T = BVc_arr[iv] + CWc_arr[iw]
          2. solve T = A * Q exactly (in Z[zeta_9]).  If no integer solution,
             reject the pair.
          3. U = -conj(Q).
          4. |zu|^2 gate (abs2_thresh in scaled units; saves the 9^f division).
          5. Compute N1=B*W-C*V, N2=C*U-A*W, N3=A*V-B*U; check each divisible
             by scale_int.
          6. If all checks pass, append (iv, iw, U) to the output arrays.

        Returns the number of surviving pairs in idx_v_out/idx_w_out/U_out.
        Counter-out arrays are also updated so the caller can match the
        Python counters exactly (orthogonality_divisibility_passes,
        u_norm_passes via abs gate, cross_product_layer_passes).
        """
        n_v = BVc_arr.shape[0]
        n_w = CWc_arr.shape[0]
        T = np.empty(6, dtype=np.int64)
        Q = np.empty(6, dtype=np.int64)
        U = np.empty(6, dtype=np.int64)
        N = np.empty(6, dtype=np.int64)
        n_match = np.int64(0)
        ndiv = np.int64(0)
        nzu = np.int64(0)
        ncross = np.int64(0)
        cap = idx_v_out.shape[0]
        for iv in range(n_v):
            BVc = BVc_arr[iv]
            V = V_arr[iv]
            for iw in range(n_w):
                CWc = CWc_arr[iw]
                W = W_arr[iw]
                # T = BVc + CWc
                for c in range(6):
                    T[c] = BVc[c] + CWc[c]
                # Solve T = A * Q exactly.
                ok = _z9_divide_if_exact_by_a_nb(T, A, inv_M_A, power_table, Q)
                if not ok:
                    continue
                ndiv += 1
                # U = -conj(Q)
                # _z9_conj_nb body (inlined to avoid alloc):
                # conj: (q0-q3, -q2, -q1, -q3, -q2+q5, -q1+q4); then negate.
                U[0] = -(Q[0] - Q[3])
                U[1] = Q[2]
                U[2] = Q[1]
                U[3] = Q[3]
                U[4] = -(-Q[2] + Q[5])
                U[5] = -(-Q[1] + Q[4])
                # |zu|^2 <= (eps+1e-15)^2 * 9^f ?
                abs2 = _z9_abs2_scaled_nb(U, zeta_re, zeta_im)
                if abs2 > abs2_thresh:
                    continue
                nzu += 1
                # N1 = B*W - C*V, divisibility-by-scale_int test.
                # Compute each component, early-abort on first non-divisible
                # element.  We must compute all 6 coeffs of N before deciding.
                # Inline z9_mul-and-subtract.
                # ----- N1 = B*W - C*V -----
                for c in range(6):
                    N[c] = 0
                for i in range(6):
                    bi = B[i]
                    if bi != 0:
                        for j in range(6):
                            wj = W[j]
                            if wj != 0:
                                prod = bi * wj
                                row = power_table[i + j]
                                for cc in range(6):
                                    N[cc] += prod * row[cc]
                    ci = C[i]
                    if ci != 0:
                        for j in range(6):
                            vj = V[j]
                            if vj != 0:
                                prod = ci * vj
                                row = power_table[i + j]
                                for cc in range(6):
                                    N[cc] -= prod * row[cc]
                ok1 = True
                for c in range(6):
                    if N[c] % scale_int != 0:
                        ok1 = False
                        break
                if not ok1:
                    continue
                # ----- N2 = C*U - A*W -----
                for c in range(6):
                    N[c] = 0
                for i in range(6):
                    ci = C[i]
                    if ci != 0:
                        for j in range(6):
                            uj = U[j]
                            if uj != 0:
                                prod = ci * uj
                                row = power_table[i + j]
                                for cc in range(6):
                                    N[cc] += prod * row[cc]
                    ai = A[i]
                    if ai != 0:
                        for j in range(6):
                            wj = W[j]
                            if wj != 0:
                                prod = ai * wj
                                row = power_table[i + j]
                                for cc in range(6):
                                    N[cc] -= prod * row[cc]
                ok2 = True
                for c in range(6):
                    if N[c] % scale_int != 0:
                        ok2 = False
                        break
                if not ok2:
                    continue
                # ----- N3 = A*V - B*U -----
                for c in range(6):
                    N[c] = 0
                for i in range(6):
                    ai = A[i]
                    if ai != 0:
                        for j in range(6):
                            vj = V[j]
                            if vj != 0:
                                prod = ai * vj
                                row = power_table[i + j]
                                for cc in range(6):
                                    N[cc] += prod * row[cc]
                    bi = B[i]
                    if bi != 0:
                        for j in range(6):
                            uj = U[j]
                            if uj != 0:
                                prod = bi * uj
                                row = power_table[i + j]
                                for cc in range(6):
                                    N[cc] -= prod * row[cc]
                ok3 = True
                for c in range(6):
                    if N[c] % scale_int != 0:
                        ok3 = False
                        break
                if not ok3:
                    continue
                ncross += 1
                # Survivor; emit (iv, iw, U).
                if n_match < cap:
                    idx_v_out[n_match] = iv
                    idx_w_out[n_match] = iw
                    for c in range(6):
                        U_out[n_match, c] = U[c]
                    n_match += 1
        ndiv_out[0] = ndiv
        nzu_out[0] = nzu
        ncross_out[0] = ncross
        return n_match


# Zeta_9 power tables for the embedding (float) used by the abs-gate kernel.
# zeta_9^k = cos(2*pi*k/9) + i*sin(2*pi*k/9), k = 0..5.
_ZETA9_RE = np.array(
    [math.cos(2.0 * math.pi * k / 9.0) for k in range(6)], dtype=np.float64
)
_ZETA9_IM = np.array(
    [math.sin(2.0 * math.pi * k / 9.0) for k in range(6)], dtype=np.float64
)


# Module-level alias for np.ascontiguousarray (avoids two attribute lookups per
# wrapper call).  This is referenced by every z9_* wrapper below as `_ascont`.
_ascont = np.ascontiguousarray
_int64 = np.int64


def z9_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two Z[zeta_9] elements represented by length-6 int arrays."""
    if _HAVE_NUMBA:
        # Inline contiguity check: when both inputs are already C-contiguous int64
        # arrays (the stage-5 case), skip the ascontiguousarray no-op overhead.
        if type(a) is np.ndarray and a.dtype == _int64 and a.flags.c_contiguous:
            ac = a
        else:
            ac = _ascont(a, dtype=_int64)
        if type(b) is np.ndarray and b.dtype == _int64 and b.flags.c_contiguous:
            bc = b
        else:
            bc = _ascont(b, dtype=_int64)
        return _z9_mul_nb(ac, bc, _POWER_TABLE)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    out = np.zeros(6, dtype=np.int64)
    for i in range(6):
        ai = int(a[i])
        if ai == 0:
            continue
        for j in range(6):
            bj = int(b[j])
            if bj:
                out += ai * bj * _POWER_TABLE[i + j]
    return out


def z9_conj(a: np.ndarray) -> np.ndarray:
    """Complex conjugation in Z[zeta_9]."""
    if _HAVE_NUMBA:
        if type(a) is np.ndarray and a.dtype == _int64 and a.flags.c_contiguous:
            return _z9_conj_nb(a)
        return _z9_conj_nb(_ascont(a, dtype=_int64))
    a = np.asarray(a, dtype=np.int64)
    return np.array(
        [
            int(a[0] - a[3]),
            int(-a[2]),
            int(-a[1]),
            int(-a[3]),
            int(-a[2] + a[5]),
            int(-a[1] + a[4]),
        ],
        dtype=np.int64,
    )


def z9_neg(a: np.ndarray) -> np.ndarray:
    return -np.asarray(a, dtype=np.int64)


def z9_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)


def z9_divisible_by_int(a: np.ndarray, q: int) -> bool:
    if _HAVE_NUMBA:
        if type(a) is np.ndarray and a.dtype == _int64 and a.flags.c_contiguous:
            return bool(_z9_divisible_by_int_nb(a, int(q)))
        return bool(_z9_divisible_by_int_nb(_ascont(a, dtype=_int64), int(q)))
    return bool(np.all(np.asarray(a, dtype=np.int64) % int(q) == 0))


def z9_divide_by_int(a: np.ndarray, q: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    q = int(q)
    if not z9_divisible_by_int(a, q):
        raise ValueError("element is not divisible by requested integer")
    return (a // q).astype(np.int64, copy=False)


def z9_mul_matrix_left(a: np.ndarray) -> np.ndarray:
    """Matrix M such that M @ x == a*x in the basis."""
    a = np.asarray(a, dtype=np.int64)
    M = np.zeros((6, 6), dtype=np.int64)
    for j in range(6):
        M[:, j] = z9_mul(a, I6[j])
    return M


def z9_divide_if_exact_by_a(t: np.ndarray, a: np.ndarray, inv_ma: np.ndarray) -> Optional[np.ndarray]:
    """Return q if a*q == t with q in Z[zeta_9], else None.

    The solve is done in floating point for speed, but the result is accepted
    only after exact integer verification.  This makes the test exact as long as
    the rounded candidate quotient is recovered.  If coefficients become too
    large for reliable double rounding, this function should be replaced by a
    small exact 6x6 rational solver.
    """
    t = np.asarray(t, dtype=np.int64)
    q_float = inv_ma @ t.astype(np.float64)
    q = np.rint(q_float).astype(np.int64)
    if np.max(np.abs(q_float - q.astype(np.float64))) > 1.0e-7:
        return None
    if np.array_equal(z9_mul(a, q), t):
        return q
    return None


def z9_norm_m012(x: np.ndarray) -> tuple[int, int, int]:
    """Return M=(m0,m1,m2) such that x*conj(x)=m0+m1*alpha+m2*alpha^2."""
    if _HAVE_NUMBA:
        if type(x) is np.ndarray and x.dtype == _int64 and x.flags.c_contiguous:
            xc = x
        else:
            xc = _ascont(x, dtype=_int64)
        m0, m1, m2 = _z9_norm_m012_nb(xc, _POWER_TABLE)
        return (int(m0), int(m1), int(m2))
    c = z9_mul(np.asarray(x, dtype=np.int64), z9_conj(x))
    m2 = -int(c[4])
    m1 = -int(c[5])
    m0 = int(c[0]) + 2 * int(c[4])
    if not (
        int(c[1]) == m1 - m2
        and int(c[2]) == -m1 + m2
        and int(c[3]) == 0
    ):
        raise ArithmeticError(f"norm product is not in expected real-subfield form: {c.tolist()}")
    return (m0, m1, m2)


# -----------------------------------------------------------------------------
# Utility and result records
# -----------------------------------------------------------------------------


def parse_complex(s: str) -> complex:
    """Parse a complex command-line value.

    Accepted examples:
        1
        0.5+0.2j
        exp(-0.3j) is not accepted; compute it outside or use --theta.
    """
    s = s.strip().replace("i", "j")
    try:
        v = ast.literal_eval(s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"could not parse complex value {s!r}") from exc
    return complex(v)


def _read_text_or_literal(value: str) -> str:
    """Return file contents if value is an existing path, otherwise value itself."""
    if os.path.exists(value) and os.path.isfile(value):
        with open(value, "r") as fh:
            return fh.read()
    return value


def parse_fixed_row1(value: str) -> np.ndarray:
    """Parse --fixed_row1 as a file path or Python/JSON nested list.

    Accepted shapes are (3,6) and (1,3,6).  Entries are integer coefficients
    of A,B,C in the unscaled numerator layer.
    """
    txt = _read_text_or_literal(value).strip()
    if os.path.exists(value) and (value.endswith(".npy") or value.endswith(".npz")):
        data = np.load(value, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "row1" in data:
                arr = data["row1"]
            elif "row1_coeffs" in data:
                arr = data["row1_coeffs"]
            elif "fixed_row1_coeffs" in data:
                arr = data["fixed_row1_coeffs"]
            else:
                raise argparse.ArgumentTypeError(
                    f"fixed row1 npz must contain row1, row1_coeffs, or fixed_row1_coeffs; keys={list(data.keys())}"
                )
        else:
            arr = data
    else:
        try:
            data = json.loads(txt)
        except Exception:
            data = ast.literal_eval(txt)
        if isinstance(data, dict):
            data = data.get("row1", data.get("fixed_row1"))
        arr = np.asarray(data, dtype=np.int64)
    if arr.shape == (1, 3, 6):
        arr = arr[0]
    if arr.shape != (3, 6):
        raise argparse.ArgumentTypeError(f"fixed row1 must have shape (3,6) or (1,3,6), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.int64)


_SAGE_TERM_TOKEN_RE = re.compile(
    r"[+-]?\s*(?:"
    r"(?:\d+(?:/\d+)?\s*(?:\*\s*)?)?zeta9(?:\s*\^\s*\d+)?"
    r"|"
    r"\d+(?:/\d+)?"
    r")"
)


def _parse_sage_term_token(token: str) -> tuple[Fraction, int]:
    """Parse one Sage term into (coefficient, power)."""
    t = token.strip().replace(" ", "")
    if not t:
        raise argparse.ArgumentTypeError("empty Sage term")

    sign = 1
    if t[0] == "+":
        t = t[1:]
    elif t[0] == "-":
        sign = -1
        t = t[1:]

    if "zeta9" in t:
        before, after = t.split("zeta9", 1)
        if before.endswith("*"):
            before = before[:-1]
        c = Fraction(before) if before else Fraction(1)
        if after.startswith("^"):
            power = int(after[1:])
        elif after == "":
            power = 1
        else:
            raise argparse.ArgumentTypeError(f"could not parse Sage term {token!r}")
        return sign * c, power

    return sign * Fraction(t), 0


def _sage_terms_from_row_text(row_text: str) -> list[tuple[Fraction, int]]:
    """Tokenize a Sage-printed row into polynomial terms.

    This deliberately does not require commas or multi-space column separators.
    In Sage output, a later component may begin immediately after a previous
    constant with only one space, for example ``+ 32/81 1/81*zeta9^5``.
    """
    txt = row_text.replace("ζ", "zeta")
    txt = txt.replace("[", " ").replace("]", " ")
    txt = txt.replace("(", " ").replace(")", " ").replace(",", " ")

    terms: list[tuple[Fraction, int]] = []
    pos = 0
    n = len(txt)
    while pos < n:
        while pos < n and txt[pos].isspace():
            pos += 1
        if pos >= n:
            break
        m = _SAGE_TERM_TOKEN_RE.match(txt, pos)
        if m is None or m.end() == pos:
            raise argparse.ArgumentTypeError(
                f"could not parse Sage row near: {txt[pos:pos+80]!r}"
            )
        terms.append(_parse_sage_term_token(m.group(0)))
        pos = m.end()
    return terms


def _coeffs_from_sage_terms(terms: list[tuple[Fraction, int]], f: int) -> np.ndarray:
    coeff = [Fraction(0) for _ in range(6)]
    for c, power in terms:
        red = z9_reduce_power_coeff(power)
        for k in range(6):
            if int(red[k]):
                coeff[k] += int(red[k]) * c

    scale = 3 ** int(f)
    out = np.zeros(6, dtype=np.int64)
    for k, c in enumerate(coeff):
        val = c * scale
        if val.denominator != 1:
            raise argparse.ArgumentTypeError(
                f"Sage coefficient {c} in basis component {k} is not integral "
                f"after multiplying by 3^{f}; check that --f matches the row layer"
            )
        out[k] = int(val.numerator)
    return out


def parse_fixed_row1_sage(value: str, f: int) -> np.ndarray:
    """Parse Sage's printed one-row Matrix_cyclo_dense form into (3,6).

    This parser accepts Sage's aligned matrix-row format even when the columns
    are separated only by whitespace and the first term of a new column has no
    explicit plus sign.  It uses Sage's usual descending-power polynomial order:
    a new component starts when the zeta power increases relative to the
    previous term, e.g. after a constant term followed by zeta9^5.
    """
    txt = _read_text_or_literal(value)
    terms = _sage_terms_from_row_text(txt)
    if not terms:
        raise argparse.ArgumentTypeError("empty Sage row")

    comps: list[list[tuple[Fraction, int]]] = [[]]
    last_power: Optional[int] = None
    for c, power in terms:
        if comps[-1] and last_power is not None and power > last_power:
            comps.append([])
        comps[-1].append((c, power))
        last_power = power

    comps = [c for c in comps if c]
    if len(comps) != 3:
        # As a fallback, try Sage's visually aligned output with multi-space
        # column separators.  This helps in unusual cases where an entry is not
        # printed in descending powers.
        txt2 = txt.strip()
        if txt2.startswith("[") and txt2.endswith("]"):
            txt2 = txt2[1:-1].strip()
        parts = [p.strip() for p in re.split(r"\s{2,}", txt2) if p.strip()]
        if len(parts) == 3:
            comps = [_sage_terms_from_row_text(p) for p in parts]
        else:
            raise argparse.ArgumentTypeError(
                f"Sage row parser expected 3 components, found {len(comps)}. "
                "Please save the Sage row with str(row) to a text file, or pass "
                "--fixed_row1 as an explicit (3,6) coefficient array."
            )

    rows = [_coeffs_from_sage_terms(comp, f) for comp in comps]
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.int64)


def parse_fixed_row2(value: str) -> np.ndarray:
    """Parse --fixed_row2 as a file path or Python/JSON nested list.

    This accepts the same raw (3,6) format as --fixed_row1.  For .npz/.json
    dictionaries it also recognizes row2-specific keys.
    """
    txt = _read_text_or_literal(value).strip()
    if os.path.exists(value) and (value.endswith(".npy") or value.endswith(".npz")):
        data = np.load(value, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            for key in ("row2", "row2_coeffs", "fixed_row2_coeffs", "row1", "row1_coeffs", "fixed_row1_coeffs"):
                if key in data:
                    arr = data[key]
                    break
            else:
                raise argparse.ArgumentTypeError(
                    f"fixed row2 npz must contain row2, row2_coeffs, fixed_row2_coeffs, or a compatible row1 key; keys={list(data.keys())}"
                )
        else:
            arr = data
    else:
        try:
            data = json.loads(txt)
        except Exception:
            data = ast.literal_eval(txt)
        if isinstance(data, dict):
            data = data.get("row2", data.get("row2_coeffs", data.get("fixed_row2", data.get("fixed_row2_coeffs", data.get("row1", data.get("fixed_row1"))))))
        arr = np.asarray(data, dtype=np.int64)
    if arr.shape == (1, 3, 6):
        arr = arr[0]
    if arr.shape != (3, 6):
        raise argparse.ArgumentTypeError(f"fixed row2 must have shape (3,6) or (1,3,6), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.int64)


def parse_fixed_row2_sage(value: str, f: int) -> np.ndarray:
    """Parse --fixed_row2_sage using the Sage row parser."""
    return parse_fixed_row1_sage(value, f)


@dataclass
class Counters:
    row1_triples_processed: int = 0
    row1_candidates_accepted: int = 0
    row2_pairs_tested: int = 0
    orthogonality_divisibility_passes: int = 0
    u_norm_passes: int = 0
    cross_product_layer_passes: int = 0
    matrices_found: int = 0
    row2_norm_triples: int = 0
    row2_norm_triples_scanned: int = 0
    row2_norm_chunks_scanned: int = 0
    row2_norm_z1_shell_passes: int = 0
    row2_norm_z2_desc_passes: int = 0
    row2_norm_z3_desc_passes: int = 0
    row2_norm_unique_z23_last_chunk: int = 0
    row2_norm_locator_entries_last_chunk: int = 0
    chunks_seen: int = 0
    chunks_processed: int = 0

    def add(self, other: "Counters") -> "Counters":
        for k in self.__dataclass_fields__:
            setattr(self, k, int(getattr(self, k)) + int(getattr(other, k)))
        return self


@dataclass(order=True)
class HeapItem:
    # heapq is a min-heap; priority is negative distance so the worst retained
    # result is at the root.  Do not compare the result dict itself: equal
    # priorities otherwise make heapq try to order dictionaries.
    priority: float
    result: dict = field(compare=False)


def maybe_push_result(heap: list[HeapItem], result: dict, keep: int) -> None:
    item = HeapItem(priority=-float(result["dist_fro"]), result=result)
    if len(heap) < keep:
        heapq.heappush(heap, item)
    elif item.priority > heap[0].priority:
        heapq.heapreplace(heap, item)


def coeff_row_to_complex_scaled(row: np.ndarray, f: int) -> complex:
    return coeffs_to_complex_noscale(row) / float(3 ** f)


def row_complex(row_coeffs: np.ndarray, f: int) -> np.ndarray:
    return np.array([coeff_row_to_complex_scaled(row_coeffs[i], f) for i in range(3)], dtype=np.complex128)


def matrix_frobenius_dist(rows_coeffs: np.ndarray, f: int, diag: np.ndarray) -> float:
    M = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        M[i, :] = row_complex(rows_coeffs[i], f)
    D = np.diag(diag)
    return float(np.linalg.norm(M - D))


def radial_shell_ok(y: tuple[int, int, int], target_abs: float, eps: float, f: int) -> bool:
    sig = sigma1_from_m012(*y)
    if sig < 0.0:
        return False
    r = math.sqrt(max(0.0, sig)) / float(3 ** f)
    return abs(r - target_abs) <= eps + 1.0e-15


def heartbeat(
    *,
    rank: int,
    label: str,
    counters: Counters,
    nrows: int,
    current_row1_dist: float,
    start_time: float,
    force: bool = False,
    last_time_box: list[float],
) -> None:
    now = time.monotonic()
    if not force and now - last_time_box[0] < 10.0:
        return
    last_time_box[0] = now
    if rank != 0:
        return
    elapsed = now - start_time
    cur = "nan" if not math.isfinite(current_row1_dist) else f"{current_row1_dist:.6e}"
    if label.startswith("row2-norm"):
        print(
            f"heartbeat [{label}] elapsed={elapsed:,.1f}s | "
            f"row2 norm triples scanned {counters.row2_norm_triples_scanned:,}/{nrows:,} | "
            f"chunks {counters.row2_norm_chunks_scanned:,} | "
            f"accepted {counters.row2_norm_triples:,} | "
            f"Z1 shell {counters.row2_norm_z1_shell_passes:,} | "
            f"Z2 root-desc {counters.row2_norm_z2_desc_passes:,} | "
            f"Z3 root-desc {counters.row2_norm_z3_desc_passes:,} | "
            f"last unique Z2/Z3 {counters.row2_norm_unique_z23_last_chunk:,} | "
            f"last locator {counters.row2_norm_locator_entries_last_chunk:,}",
            flush=True,
        )
    else:
        print(
            f"heartbeat [{label}] elapsed={elapsed:,.1f}s | "
            f"row1 triples {counters.row1_triples_processed:,}/{nrows:,} | "
            f"row1 cand {counters.row1_candidates_accepted:,} | "
            f"row2 (V,W) tested {counters.row2_pairs_tested:,} | "
            f"orth div {counters.orthogonality_divisibility_passes:,} | "
            f"cross layer {counters.cross_product_layer_passes:,} | "
            f"matrices {counters.matrices_found:,} | "
            f"current row1 dist {cur}",
            flush=True,
        )




def z9_add3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64) + np.asarray(c, dtype=np.int64)


def y_sum_m012(ys: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return tuple(int(sum(y[k] for y in ys)) for k in range(3))  # type: ignore[return-value]


def fixed_two_rows_diagnostic_result(
    *,
    row1: np.ndarray,
    row2: np.ndarray,
    f: int,
    d1: complex,
    d2: complex,
    d3: complex,
    eps: float,
) -> dict:
    """Pure exact-algebra diagnostic for two supplied layer-f rows.

    This bypasses all rootdb/sidecar/triple candidate enumeration.  It checks
    whether the two supplied rows are normalized/orthogonal in the exact integer
    representation, whether the orthogonality formula reproduces U from V,W,
    and whether the cross-product numerators are divisible by 3^f.
    """
    row1 = np.ascontiguousarray(row1, dtype=np.int64)
    row2 = np.ascontiguousarray(row2, dtype=np.int64)
    if row1.shape != (3, 6):
        raise ValueError(f"fixed row1 must have shape (3,6), got {row1.shape}")
    if row2.shape != (3, 6):
        raise ValueError(f"fixed row2 must have shape (3,6), got {row2.shape}")

    A, B, C = row1[0].copy(), row1[1].copy(), row1[2].copy()
    U, V, W = row2[0].copy(), row2[1].copy(), row2[2].copy()
    scale_int = int(3 ** f)
    diag = np.array([d1, d2, d3], dtype=np.complex128)

    row1_Y_t = [z9_norm_m012(A), z9_norm_m012(B), z9_norm_m012(C)]
    row2_Y_t = [z9_norm_m012(U), z9_norm_m012(V), z9_norm_m012(W)]
    zrow1 = row_complex(row1, f)
    zrow2 = row_complex(row2, f)
    row1_target = np.array([d1, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    row2_target = np.array([0.0 + 0.0j, d2, 0.0 + 0.0j], dtype=np.complex128)

    inner_terms = [z9_mul(A, z9_conj(U)), z9_mul(B, z9_conj(V)), z9_mul(C, z9_conj(W))]
    inner = z9_add3(inner_terms[0], inner_terms[1], inner_terms[2])
    exact_orthogonal = bool(np.all(inner == 0))

    T = inner_terms[1] + inner_terms[2]
    formula = {
        "A_is_zero": bool(np.all(A == 0)),
        "A_divides_T": False,
        "formula_U_matches_supplied_U": False,
        "quotient_Q": None,
        "formula_U": None,
    }
    if not np.all(A == 0):
        M_A = z9_mul_matrix_left(A)
        try:
            inv_M_A = np.linalg.inv(M_A.astype(np.float64))
            Q = z9_divide_if_exact_by_a(T, A, inv_M_A)
        except np.linalg.LinAlgError:
            Q = None
        if Q is not None:
            formula_U = z9_neg(z9_conj(Q))
            formula.update({
                "A_divides_T": True,
                "formula_U_matches_supplied_U": bool(np.array_equal(formula_U, U)),
                "quotient_Q": Q.tolist(),
                "formula_U": formula_U.tolist(),
            })

    N1 = z9_sub(z9_mul(B, W), z9_mul(C, V))
    N2 = z9_sub(z9_mul(C, U), z9_mul(A, W))
    N3 = z9_sub(z9_mul(A, V), z9_mul(B, U))
    divs = [z9_divisible_by_int(N1, scale_int), z9_divisible_by_int(N2, scale_int), z9_divisible_by_int(N3, scale_int)]
    cross_layer_ok = bool(all(divs))

    rows = None
    dist_fro = None
    row3 = None
    if cross_layer_ok:
        R31 = z9_conj(z9_divide_by_int(N1, scale_int))
        R32 = z9_conj(z9_divide_by_int(N2, scale_int))
        R33 = z9_conj(z9_divide_by_int(N3, scale_int))
        row3 = np.stack([R31, R32, R33], axis=0).astype(np.int64, copy=False)
        rows = np.stack([row1, row2, row3], axis=0).astype(np.int64, copy=False)
        dist_fro = matrix_frobenius_dist(rows, f, diag)

    return {
        "mode": "fixed_row1_row2_diagnostic",
        "f": int(f),
        "eps": float(eps),
        "diag": [[float(z.real), float(z.imag)] for z in diag],
        "row1_coeffs": row1.tolist(),
        "row2_coeffs": row2.tolist(),
        "row1_Y": [list(y) for y in row1_Y_t],
        "row2_Y": [list(y) for y in row2_Y_t],
        "row1_Y_sum": list(y_sum_m012(row1_Y_t)),
        "row2_Y_sum": list(y_sum_m012(row2_Y_t)),
        "row1_components": [[float(z.real), float(z.imag)] for z in zrow1],
        "row2_components": [[float(z.real), float(z.imag)] for z in zrow2],
        "row1_dist": float(np.linalg.norm(zrow1 - row1_target)),
        "row2_dist": float(np.linalg.norm(zrow2 - row2_target)),
        "row2_u_abs": float(abs(zrow2[0])),
        "row2_v_minus_d2_abs": float(abs(zrow2[1] - d2)),
        "row2_w_abs": float(abs(zrow2[2])),
        "row2_componentwise_eps_ok": bool(
            abs(zrow2[0]) <= eps + 1e-15
            and abs(zrow2[1] - d2) <= eps + 1e-15
            and abs(zrow2[2]) <= eps + 1e-15
        ),
        "orthogonality_terms": [x.tolist() for x in inner_terms],
        "orthogonality_sum": inner.tolist(),
        "exact_orthogonal": exact_orthogonal,
        "formula": formula,
        "cross_product_numerators": [N1.tolist(), N2.tolist(), N3.tolist()],
        "cross_product_divisible_by_3f": divs,
        "cross_product_layer_ok": cross_layer_ok,
        "row3_coeffs": None if row3 is None else row3.tolist(),
        "rows_coeffs_layer_f": None if rows is None else rows.tolist(),
        "dist_fro": None if dist_fro is None else float(dist_fro),
        "row_complex": None if rows is None else [
            [[float(z.real), float(z.imag)] for z in row_complex(rows[i], f)]
            for i in range(3)
        ],
    }


def write_fixed_two_rows_diagnostic(
    *,
    row1: np.ndarray,
    row2: np.ndarray,
    f: int,
    d1: complex,
    d2: complex,
    d3: complex,
    eps: float,
    output_prefix: str,
    rank: int,
) -> Optional[dict]:
    """Run and write the two-fixed-rows diagnostic on rank 0."""
    if rank != 0:
        return None
    result = fixed_two_rows_diagnostic_result(
        row1=row1, row2=row2, f=f, d1=d1, d2=d2, d3=d3, eps=eps
    )
    out_npz = output_prefix + ".npz"
    rows = result.get("rows_coeffs_layer_f")
    np.savez(
        out_npz,
        fixed_row1_coeffs=np.asarray(row1, dtype=np.int64),
        fixed_row2_coeffs=np.asarray(row2, dtype=np.int64),
        rows_coeffs_layer_f=(
            np.asarray(rows, dtype=np.int64).reshape(1, 3, 3, 6)
            if rows is not None else np.empty((0, 3, 3, 6), dtype=np.int64)
        ),
        dist_fro=np.asarray([] if result["dist_fro"] is None else [result["dist_fro"]], dtype=np.float64),
        diag=np.asarray([d1, d2, d3], dtype=np.complex128),
        f=np.asarray([f], dtype=np.int64),
        eps=np.asarray([eps], dtype=np.float64),
    )
    result["results_npz"] = os.path.abspath(out_npz)
    result["results_json"] = os.path.abspath(output_prefix + ".json")
    with open(output_prefix + ".json", "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result

# -----------------------------------------------------------------------------
# Row-2 norm-triple preselection
# -----------------------------------------------------------------------------


def build_or_load_row2_norm_triples(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    d2: complex,
    eps: float,
    n_phase_bins: int,
    triples_chunk_rows: int,
    cache_path: Optional[str],
    rank: int,
    comm,
    start_time: float,
) -> np.ndarray:
    """Return admissible row-2 norm triples (Z1,Z2,Z3) as shape (n,9).

    This stage is intentionally only a norm/root-availability preselection:
    it does not enumerate row-2 vectors (U,V,W).  It requires Z2 and Z3 to
    have sidecar candidates near d2 and 0, respectively, and Z1 to be in the
    small norm shell near 0.  U itself is computed later by exact orthogonality.
    """
    if cache_path and os.path.exists(cache_path):
        if rank == 0:
            arr = np.load(cache_path)
            print(f"loaded row2 norm triples cache {cache_path}: {arr.shape[0]:,} triples", flush=True)
        else:
            arr = None
        arr = comm.bcast(arr, root=0)
        return np.asarray(arr, dtype=np.int64)

    if rank == 0:
        tmeta = _read_manifest(triples_json)
        nrows = int(tmeta["rows_written"])
        paths = _state_paths(rootdb_prefix, n_phase_bins)
        phase_meta = _read_manifest(paths["phase_sidecar_meta"])
        batch_cache = OrderedDict()
        accepted_chunks: list[np.ndarray] = []
        counters = Counters()
        hb = [0.0]

        print("building row2 admissible norm-triple list", flush=True)
        heartbeat(
            rank=rank,
            label="row2-norm-preselect",
            counters=counters,
            nrows=nrows,
            current_row1_dist=math.inf,
            start_time=start_time,
            force=True,
            last_time_box=hb,
        )

        start = 0
        while start < nrows:
            take = min(int(triples_chunk_rows), nrows - start)
            triples = _read_triple_rows(triples_file, start, take)
            counters.row2_norm_triples_scanned += int(triples.shape[0])
            counters.row2_norm_chunks_scanned += 1

            # Locator must cover ALL three component positions of the input
            # triple, not just slots 1-2: row-1-oriented triples have the
            # large component at slot 0, and our row-2 permutation below
            # uses that large component as z2 (the slot-1 lookup target d2).
            needed = np.unique(
                np.vstack([triples[:, 0:3], triples[:, 3:6], triples[:, 6:9]]),
                axis=0,
            )
            counters.row2_norm_unique_z23_last_chunk = int(needed.shape[0])
            locator = _build_locator(needed, paths)
            counters.row2_norm_locator_entries_last_chunk = int(len(locator))

            # Row-2 preselect with implicit permutation.  The upstream
            # select_triples_optimized step builds row-1-oriented triples
            # (large at slot 0).  For row-2 we want triples with large at
            # slot 1; rather than materialize a separate file we test the
            # two permutations of each input triple that put Y_large at
            # slot 1: (Y2, Y1, Y3) and (Y3, Y1, Y2).  Output is stored in
            # the permuted form so downstream consumers see the right layout.
            keep_rows = []
            for tri in triples:
                Y1 = np.asarray(tri[0:3], dtype=np.int64)
                Y2 = np.asarray(tri[3:6], dtype=np.int64)
                Y3 = np.asarray(tri[6:9], dtype=np.int64)
                for z1_a, z2_a, z3_a in ((Y2, Y1, Y3), (Y3, Y1, Y2)):
                    z1 = (int(z1_a[0]), int(z1_a[1]), int(z1_a[2]))
                    z2 = (int(z2_a[0]), int(z2_a[1]), int(z2_a[2]))
                    z3 = (int(z3_a[0]), int(z3_a[1]), int(z3_a[2]))
                    if not radial_shell_ok(z1, 0.0, eps, f):
                        continue
                    counters.row2_norm_z1_shell_passes += 1
                    desc2 = _candidate_desc_from_binned_sidecar(
                        z2, locator, phase_meta, batch_cache, d2, eps, f, profile=None
                    )
                    if desc2 is None:
                        continue
                    counters.row2_norm_z2_desc_passes += 1
                    desc3 = _candidate_desc_from_binned_sidecar(
                        z3, locator, phase_meta, batch_cache, 0.0 + 0.0j, eps, f, profile=None
                    )
                    if desc3 is None:
                        continue
                    counters.row2_norm_z3_desc_passes += 1
                    keep_rows.append(np.concatenate([z1_a, z2_a, z3_a]).astype(np.int64))

            if keep_rows:
                arr = np.vstack(keep_rows).astype(np.int64, copy=False)
                accepted_chunks.append(arr)
                counters.row2_norm_triples += int(arr.shape[0])

            heartbeat(
                rank=rank,
                label="row2-norm-preselect",
                counters=counters,
                nrows=nrows,
                current_row1_dist=math.inf,
                start_time=start_time,
                last_time_box=hb,
            )
            start += take

        heartbeat(
            rank=rank,
            label="row2-norm-preselect",
            counters=counters,
            nrows=nrows,
            current_row1_dist=math.inf,
            start_time=start_time,
            force=True,
            last_time_box=hb,
        )

        if accepted_chunks:
            out = np.vstack(accepted_chunks).astype(np.int64, copy=False)
            before_unique = int(out.shape[0])
            out = np.unique(out, axis=0)
            print(
                f"row2 norm preselect unique: {before_unique:,} accepted rows -> {out.shape[0]:,} unique rows",
                flush=True,
            )
        else:
            out = np.empty((0, 9), dtype=np.int64)

        if cache_path:
            tmp = cache_path + ".tmp.npy"
            np.save(tmp, out)
            os.replace(tmp, cache_path)
            print(f"wrote row2 norm triples cache {cache_path}: {out.shape[0]:,} triples", flush=True)
        else:
            print(f"row2 admissible norm triples: {out.shape[0]:,}", flush=True)
    else:
        out = None

    out = comm.bcast(out, root=0)
    return np.asarray(out, dtype=np.int64)


# -----------------------------------------------------------------------------
# Main streamed search
# -----------------------------------------------------------------------------


def iter_roots_for_desc(desc, phase_meta, batch_cache, f: int, eps: float):
    """Yield (coeffs, z_scaled) for all exact-filtered roots in a desc."""
    if desc is None:
        return
    for base, idx_local, zvals in _iter_desc_z_and_indices(desc, phase_meta, batch_cache, f, eps):
        for k, z in enumerate(zvals):
            abs_idx = int(base + idx_local[k])
            coeffs = _load_coeff_row(desc[0], abs_idx, phase_meta, batch_cache)
            yield np.asarray(coeffs, dtype=np.int64), complex(z)


def materialize_roots_for_desc(desc, phase_meta, batch_cache, f: int, eps: float):
    """Action 1 (2026-05-12): materialize a descriptor's exact-filtered roots
    into (coeffs_arr, z_arr) arrays so a batched z9_mul can be applied across
    the whole descriptor in one Numba call.

    Returns (coeffs, zvals) where coeffs is (n, 6) int64 C-contiguous and
    zvals is (n,) complex128.  Returns (empty, empty) if desc is None.
    """
    if desc is None:
        return (
            np.empty((0, 6), dtype=np.int64),
            np.empty((0,), dtype=np.complex128),
        )
    coeffs_list: list[np.ndarray] = []
    z_list: list[complex] = []
    for base, idx_local, zvals in _iter_desc_z_and_indices(desc, phase_meta, batch_cache, f, eps):
        for k, z in enumerate(zvals):
            abs_idx = int(base + idx_local[k])
            coeffs = _load_coeff_row(desc[0], abs_idx, phase_meta, batch_cache)
            coeffs_list.append(np.asarray(coeffs, dtype=np.int64))
            z_list.append(complex(z))
    if not coeffs_list:
        return (
            np.empty((0, 6), dtype=np.int64),
            np.empty((0,), dtype=np.complex128),
        )
    coeffs_arr = np.ascontiguousarray(np.stack(coeffs_list, axis=0), dtype=np.int64)
    z_arr = np.asarray(z_list, dtype=np.complex128)
    return coeffs_arr, z_arr


def search_diagonal_two_rows_streamed(
    *,
    triples_file: str,
    triples_json: str,
    rootdb_prefix: str,
    f: int,
    d1: complex,
    d2: complex,
    d3: complex,
    eps: float,
    output_prefix: str,
    triples_chunk_rows: int = 200000,
    n_phase_bins: int = 512,
    chunk_meta_json: Optional[str] = None,
    row2_cache: Optional[str] = None,
    keep: int = 16,
    disable_y_pruning: bool = False,
    fixed_row1_coeffs: Optional[np.ndarray] = None,
    fixed_row2_coeffs: Optional[np.ndarray] = None,
    verbose: bool = True,
    max_matches: int = 0,
) -> Optional[dict]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = time.monotonic()
    hb = [0.0]

    local_counters = Counters()
    current_row1_dist = math.inf

    if rank == 0:
        tmeta = _read_manifest(triples_json)
        nrows = int(tmeta["rows_written"])
        file_size = os.path.getsize(triples_file)
        expected = nrows * 9 * np.dtype(np.int64).itemsize
        if file_size != expected:
            raise RuntimeError(f"Triple file size mismatch: expected {expected}, got {file_size}")
        paths = _state_paths(rootdb_prefix, n_phase_bins)
        if not os.path.exists(paths["exact_roots_index_meta"]):
            raise FileNotFoundError(f"Rootdb index not found for prefix {rootdb_prefix}")
        if not os.path.exists(paths["phase_sidecar_meta"]):
            raise FileNotFoundError(
                f"Binned phase sidecar not found. Run the sidecar builder first for {rootdb_prefix}"
            )
        phase_meta = _read_manifest(paths["phase_sidecar_meta"])
        chunk_meta = _read_manifest(chunk_meta_json) if chunk_meta_json else None
        print(
            f"streamed two-row diagonal search | rows={nrows:,} ranks={size} f={f} eps={eps:g}",
            flush=True,
        )
    else:
        nrows = None
        paths = None
        phase_meta = None
        chunk_meta = None

    nrows = comm.bcast(nrows, root=0)
    paths = comm.bcast(paths, root=0)
    phase_meta = comm.bcast(phase_meta, root=0)
    chunk_meta = comm.bcast(chunk_meta, root=0)

    heartbeat(
        rank=rank,
        label="startup",
        counters=local_counters,
        nrows=nrows,
        current_row1_dist=current_row1_dist,
        start_time=start_time,
        force=True,
        last_time_box=hb,
    )

    if fixed_row2_coeffs is not None:
        if fixed_row1_coeffs is None:
            raise ValueError("--fixed_row2/--fixed_row2_sage requires --fixed_row1 or --fixed_row1_sage.")
        return write_fixed_two_rows_diagnostic(
            row1=np.asarray(fixed_row1_coeffs, dtype=np.int64),
            row2=np.asarray(fixed_row2_coeffs, dtype=np.int64),
            f=f,
            d1=d1,
            d2=d2,
            d3=d3,
            eps=eps,
            output_prefix=output_prefix,
            rank=rank,
        )

    row2_triples = build_or_load_row2_norm_triples(
        triples_file=triples_file,
        triples_json=triples_json,
        rootdb_prefix=rootdb_prefix,
        f=f,
        d2=d2,
        eps=eps,
        n_phase_bins=n_phase_bins,
        triples_chunk_rows=triples_chunk_rows,
        cache_path=row2_cache,
        rank=rank,
        comm=comm,
        start_time=start_time,
    )

    # Build a locator for all Z2/Z3 values used by row2 roots.  This is a small
    # reusable cache, not a row-vector materialization.
    if row2_triples.shape[0] > 0:
        row2_needed = np.unique(
            np.vstack([row2_triples[:, 3:6], row2_triples[:, 6:9]]),
            axis=0,
        )
        row2_locator = _build_locator(row2_needed, paths)
    else:
        row2_locator = {}

    if rank == 0:
        print(
            f"row2 norm triples ready: {row2_triples.shape[0]:,}; "
            f"unique Z2/Z3 locator entries={len(row2_locator):,}",
            flush=True,
        )

    batch_cache = OrderedDict()
    heap: list[HeapItem] = []
    diag = np.array([d1, d2, d3], dtype=np.complex128)
    scale_int = int(3 ** f)

    # ------------------------------------------------------------------
    # Fixed-row1 test mode.  This bypasses row-1 triple streaming and
    # directly completes the supplied row with the same exact row-2 filters.
    # ------------------------------------------------------------------
    if fixed_row1_coeffs is not None:
        row1 = np.asarray(fixed_row1_coeffs, dtype=np.int64)
        if row1.shape != (3, 6):
            raise ValueError(f"fixed_row1_coeffs must have shape (3,6), got {row1.shape}")
        A, B, C = row1[0].copy(), row1[1].copy(), row1[2].copy()
        y1, y2, y3 = z9_norm_m012(A), z9_norm_m012(B), z9_norm_m012(C)
        zrow1 = row_complex(row1, f)
        current_row1_dist = float(np.linalg.norm(zrow1 - np.array([d1, 0.0 + 0.0j, 0.0 + 0.0j])))
        if rank == 0:
            print(
                "fixed row1 mode | "
                f"row1_dist={current_row1_dist:.12e} | "
                f"row1_Y={[list(y1), list(y2), list(y3)]}",
                flush=True,
            )
        local_counters.row1_candidates_accepted = 1

        if np.all(A == 0):
            raise ValueError("fixed row1 has A=0; orthogonality formula divides by A")
        M_A = z9_mul_matrix_left(A)
        try:
            inv_M_A = np.linalg.inv(M_A.astype(np.float64))
        except np.linalg.LinAlgError as exc:
            raise ValueError("fixed row1 A multiplication matrix is singular") from exc

        row2_desc_cache: dict[tuple[tuple[int, int, int], complex], object] = {}

        def get_row2_desc_fixed(y: tuple[int, int, int], target: complex):
            key = (y, target)
            if key not in row2_desc_cache:
                row2_desc_cache[key] = _candidate_desc_from_binned_sidecar(
                    y,
                    row2_locator,
                    phase_meta,
                    batch_cache,
                    target,
                    eps,
                    f,
                    profile=None,
                    disable_y_pruning=disable_y_pruning,
                )
            return row2_desc_cache[key]

        heartbeat(
            rank=rank,
            label="fixed-row1-search",
            counters=local_counters,
            nrows=int(row2_triples.shape[0]),
            current_row1_dist=current_row1_dist,
            start_time=start_time,
            force=True,
            last_time_box=hb,
        )

        # Action 3 (2026-05-12): pre-bind raw Numba kernels and POWER_TABLE for
        # the hot loop.  This bypasses the z9_mul / z9_conj / z9_divisible_by_int
        # Python wrappers (each of which does an isinstance check + ascontig +
        # global lookup, ~0.25 us / call).  Stage-5 inner loop does 5 muls,
        # 4 conjs, and 3 divisibility checks per (V,W) — pre-binding saves on
        # the order of 2-3 us per iteration of the innermost loop.
        if _HAVE_NUMBA:
            _mul_nb = _z9_mul_nb
            _conj_nb = _z9_conj_nb
            _div_nb = _z9_divisible_by_int_nb
            _mulconj_batch = _z9_mul_conj_batch_nb
            _mul_batch = _z9_mul_batch_nb
            _filter_pairs = _z9_inner_filter_pairs_nb
            _PT = _POWER_TABLE
            _ZRE = _ZETA9_RE
            _ZIM = _ZETA9_IM
            # |zu|^2 <= (eps + 1e-15)^2 * 9^f  (abs2 in unscaled units)
            _abs2_thresh = float((eps + 1.0e-15) ** 2 * (9 ** f))
            # Reusable inner-buffers for the fused kernel; per-iz resize.
            _ndiv_box = np.zeros(1, dtype=np.int64)
            _nzu_box = np.zeros(1, dtype=np.int64)
            _ncross_box = np.zeros(1, dtype=np.int64)
        for iz in range(rank, int(row2_triples.shape[0]), size):
            ztri = row2_triples[iz]
            local_counters.row1_triples_processed += 1  # reused as row2 norm triples visited in fixed mode
            Z1 = (int(ztri[0]), int(ztri[1]), int(ztri[2]))
            Z2 = (int(ztri[3]), int(ztri[4]), int(ztri[5]))
            Z3 = (int(ztri[6]), int(ztri[7]), int(ztri[8]))
            desc_v = get_row2_desc_fixed(Z2, d2)
            if desc_v is None:
                continue
            desc_w = get_row2_desc_fixed(Z3, 0.0 + 0.0j)
            if desc_w is None:
                continue

            # 2026-05-12 (Actions A+B+C, post-prior-round): fused Numba kernel.
            # Materialize V's/W's, precompute BVc/CWc via batched mul,
            # then run the single fused inner-filter that does:
            #   T = BVc + CWc; solve T = A*Q (exact div); U = -conj(Q);
            #   |zu|^2 <= thresh gate; N1/N2/N3 cross-product divisibility.
            # The kernel returns surviving (iv, iw, U) and updates counters.
            # Falls back to legacy Python path when Numba is unavailable.
            if _HAVE_NUMBA:
                V_arr, _ = materialize_roots_for_desc(desc_v, phase_meta, batch_cache, f, eps)
                W_arr, _ = materialize_roots_for_desc(desc_w, phase_meta, batch_cache, f, eps)
                n_v = V_arr.shape[0]
                n_w = W_arr.shape[0]
                if n_v == 0 or n_w == 0:
                    continue
                BVc_arr = np.empty((n_v, 6), dtype=np.int64)
                CWc_arr = np.empty((n_w, 6), dtype=np.int64)
                _mulconj_batch(B, V_arr, _PT, BVc_arr)
                _mulconj_batch(C, W_arr, _PT, CWc_arr)
                cap = n_v * n_w
                # row2_pairs_tested: we cover the full Cartesian product.
                local_counters.row2_pairs_tested += cap
                idx_v_out = np.empty(cap, dtype=np.int64)
                idx_w_out = np.empty(cap, dtype=np.int64)
                U_out_arr = np.empty((cap, 6), dtype=np.int64)
                _ndiv_box[0] = 0
                _nzu_box[0] = 0
                _ncross_box[0] = 0
                n_match = _filter_pairs(
                    BVc_arr, CWc_arr, V_arr, W_arr, A, B, C, inv_M_A,
                    _ZRE, _ZIM, _abs2_thresh, scale_int, _PT,
                    _ndiv_box, _nzu_box, _ncross_box,
                    idx_v_out, idx_w_out, U_out_arr,
                )
                local_counters.orthogonality_divisibility_passes += int(_ndiv_box[0])
                # u_norm_passes was incremented once per Q-divide-success
                # in the prior version; nzu_box is # pairs that passed
                # *both* divide AND |zu| gate.  Match the prior counter
                # semantics: u_norm_passes counted divide-successes.
                local_counters.u_norm_passes += int(_ndiv_box[0])
                local_counters.cross_product_layer_passes += int(_ncross_box[0])
                for k in range(n_match):
                    iv = int(idx_v_out[k])
                    iw = int(idx_w_out[k])
                    V = V_arr[iv]
                    W = W_arr[iw]
                    U = U_out_arr[k]
                    # Recompute N1/N2/N3 once per surviving pair (rare).
                    N1 = _mul_nb(B, W, _PT) - _mul_nb(C, V, _PT)
                    N2 = _mul_nb(C, U, _PT) - _mul_nb(A, W, _PT)
                    N3 = _mul_nb(A, V, _PT) - _mul_nb(B, U, _PT)
                    R31 = z9_conj(z9_divide_by_int(N1, scale_int))
                    R32 = z9_conj(z9_divide_by_int(N2, scale_int))
                    R33 = z9_conj(z9_divide_by_int(N3, scale_int))
                    rows = np.stack(
                        [
                            np.stack([A, B, C], axis=0),
                            np.stack([U, V, W], axis=0),
                            np.stack([R31, R32, R33], axis=0),
                        ],
                        axis=0,
                    ).astype(np.int64, copy=False)
                    dist_fro = matrix_frobenius_dist(rows, f, diag)
                    local_counters.matrices_found += 1
                    result = {
                        "dist_fro": float(dist_fro),
                        "row1_dist": float(current_row1_dist),
                        "row1_Y": [list(y1), list(y2), list(y3)],
                        "row2_Y": [list(Z1), list(Z2), list(Z3)],
                        "rows_coeffs_layer_f": rows.tolist(),
                        "row_complex": [
                            [[float(z.real), float(z.imag)] for z in row_complex(rows[i], f)]
                            for i in range(3)
                        ],
                    }
                    maybe_push_result(heap, result, keep)
                heartbeat(
                    rank=rank,
                    label="fixed-row1-search",
                    counters=local_counters,
                    nrows=int(row2_triples.shape[0]),
                    current_row1_dist=current_row1_dist,
                    start_time=start_time,
                    last_time_box=hb,
                )
                continue

            # Legacy (no-Numba) path retained for correctness fallback.
            B_contrib_cache: dict[bytes, np.ndarray] = {}
            C_contrib_cache: dict[bytes, np.ndarray] = {}
            for V, zv in iter_roots_for_desc(desc_v, phase_meta, batch_cache, f, eps):
                key_v = V.tobytes()
                BVc = B_contrib_cache.get(key_v)
                if BVc is None:
                    BVc = z9_mul(B, z9_conj(V))
                    B_contrib_cache[key_v] = BVc
                for W, zw in iter_roots_for_desc(desc_w, phase_meta, batch_cache, f, eps):
                    local_counters.row2_pairs_tested += 1
                    key_w = W.tobytes()
                    CWc = C_contrib_cache.get(key_w)
                    if CWc is None:
                        CWc = z9_mul(C, z9_conj(W))
                        C_contrib_cache[key_w] = CWc
                    T = BVc + CWc
                    Q = z9_divide_if_exact_by_a(T, A, inv_M_A)
                    if Q is None:
                        continue
                    local_counters.orthogonality_divisibility_passes += 1
                    U = z9_neg(z9_conj(Q))
                    # Disabled 2026-05-12: see comment in Action-1 batched path.
                    local_counters.u_norm_passes += 1
                    zu = coeff_row_to_complex_scaled(U, f)
                    if abs(zu) > eps + 1.0e-15:
                        continue
                    N1 = z9_sub(z9_mul(B, W), z9_mul(C, V))
                    N2 = z9_sub(z9_mul(C, U), z9_mul(A, W))
                    N3 = z9_sub(z9_mul(A, V), z9_mul(B, U))
                    if not (
                        z9_divisible_by_int(N1, scale_int)
                        and z9_divisible_by_int(N2, scale_int)
                        and z9_divisible_by_int(N3, scale_int)
                    ):
                        continue
                    local_counters.cross_product_layer_passes += 1
                    R31 = z9_conj(z9_divide_by_int(N1, scale_int))
                    R32 = z9_conj(z9_divide_by_int(N2, scale_int))
                    R33 = z9_conj(z9_divide_by_int(N3, scale_int))
                    rows = np.stack(
                        [
                            np.stack([A, B, C], axis=0),
                            np.stack([U, V, W], axis=0),
                            np.stack([R31, R32, R33], axis=0),
                        ],
                        axis=0,
                    ).astype(np.int64, copy=False)
                    dist_fro = matrix_frobenius_dist(rows, f, diag)
                    local_counters.matrices_found += 1
                    result = {
                        "dist_fro": float(dist_fro),
                        "row1_dist": float(current_row1_dist),
                        "row1_Y": [list(y1), list(y2), list(y3)],
                        "row2_Y": [list(Z1), list(Z2), list(Z3)],
                        "rows_coeffs_layer_f": rows.tolist(),
                        "row_complex": [
                            [[float(z.real), float(z.imag)] for z in row_complex(rows[i], f)]
                            for i in range(3)
                        ],
                    }
                    maybe_push_result(heap, result, keep)
                    heartbeat(
                        rank=rank,
                        label="fixed-row1-search",
                        counters=local_counters,
                        nrows=int(row2_triples.shape[0]),
                        current_row1_dist=current_row1_dist,
                        start_time=start_time,
                        last_time_box=hb,
                    )
            heartbeat(
                rank=rank,
                label="fixed-row1-search",
                counters=local_counters,
                nrows=int(row2_triples.shape[0]),
                current_row1_dist=current_row1_dist,
                start_time=start_time,
                last_time_box=hb,
            )

        local_results = [item.result for item in heap]
        payload = {"rank": rank, "counters": asdict(local_counters), "results": local_results}
        gathered = comm.gather(payload, root=0)
        if rank != 0:
            return None
        total = Counters()
        final_heap: list[HeapItem] = []
        by_rank = {}
        for part in gathered:
            c = Counters(**part["counters"])
            total.add(c)
            by_rank[str(part["rank"])] = part["counters"]
            for r in part["results"]:
                maybe_push_result(final_heap, r, keep)
        final_results = [item.result for item in sorted(final_heap, key=lambda x: -x.priority)]
        final_results.sort(key=lambda r: float(r["dist_fro"]))
        out_npz = output_prefix + ".npz"
        if final_results:
            rows_arr = np.asarray([r["rows_coeffs_layer_f"] for r in final_results], dtype=np.int64)
            dist_arr = np.asarray([r["dist_fro"] for r in final_results], dtype=np.float64)
        else:
            rows_arr = np.empty((0, 3, 3, 6), dtype=np.int64)
            dist_arr = np.empty((0,), dtype=np.float64)
        np.savez(
            out_npz,
            rows_coeffs_layer_f=rows_arr,
            dist_fro=dist_arr,
            fixed_row1_coeffs=row1,
            fixed_row1_Y=np.asarray([y1, y2, y3], dtype=np.int64),
            row1_dist=np.asarray([current_row1_dist], dtype=np.float64),
            diag=np.asarray(diag, dtype=np.complex128),
            f=np.asarray([f], dtype=np.int64),
            eps=np.asarray([eps], dtype=np.float64),
        )
        summary = {
            "mode": "fixed_row1",
            "results_npz": os.path.abspath(out_npz),
            "results_json": os.path.abspath(output_prefix + ".json"),
            "profile_json": os.path.abspath(output_prefix + ".profile.json"),
            "n_results": len(final_results),
            "best_dist_fro": None if not final_results else float(final_results[0]["dist_fro"]),
            "row1_dist": float(current_row1_dist),
            "fixed_row1_coeffs": row1.tolist(),
            "fixed_row1_Y": [list(y1), list(y2), list(y3)],
            "f": int(f),
            "eps": float(eps),
            "diag": [[float(z.real), float(z.imag)] for z in diag],
            "triples_file": os.path.abspath(triples_file),
            "triples_json": os.path.abspath(triples_json),
            "rootdb_prefix": os.path.abspath(rootdb_prefix),
            "mpi_ranks": int(size),
            "n_phase_bins": int(n_phase_bins),
            "row2_norm_triples": int(row2_triples.shape[0]),
            "counters_sum": asdict(total),
            "top_results": final_results,
        }
        with open(output_prefix + ".json", "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        with open(output_prefix + ".profile.json", "w") as fh:
            json.dump({"by_rank": by_rank, "sum": asdict(total)}, fh, indent=2, sort_keys=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    n_chunks = (int(nrows) + int(triples_chunk_rows) - 1) // int(triples_chunk_rows)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * int(triples_chunk_rows)
        take = min(int(triples_chunk_rows), int(nrows) - start)
        local_counters.chunks_seen += 1

        # Simple streamed MPI partitioning over row-1 chunks.
        if chunk_idx % size != rank:
            continue

        triples = _read_triple_rows(triples_file, start, take)
        local_counters.chunks_processed += 1

        if verbose and rank == 0:
            print(f"row1 search chunk rows {start:,}..{start + take - 1:,} of {nrows:,}", flush=True)

        if chunk_meta is not None:
            rec = chunk_meta["chunks"][chunk_idx]
            _needed, locator = _load_chunk_metadata(rec["path"])
        else:
            needed = np.unique(triples.reshape((-1, 3)), axis=0)
            locator = _build_locator(needed, paths)

        desc_cache: dict[tuple[tuple[int, int, int], int], object] = {}

        def get_desc(y: tuple[int, int, int], coord: int, target: complex):
            key = (y, coord)
            if key not in desc_cache:
                desc_cache[key] = _candidate_desc_from_binned_sidecar(
                    y,
                    locator,
                    phase_meta,
                    batch_cache,
                    target,
                    eps,
                    f,
                    profile=None,
                    disable_y_pruning=disable_y_pruning,
                )
            return desc_cache[key]

        row2_desc_cache: dict[tuple[tuple[int, int, int], complex], object] = {}

        def get_row2_desc(y: tuple[int, int, int], target: complex):
            key = (y, target)
            if key not in row2_desc_cache:
                row2_desc_cache[key] = _candidate_desc_from_binned_sidecar(
                    y,
                    row2_locator,
                    phase_meta,
                    batch_cache,
                    target,
                    eps,
                    f,
                    profile=None,
                    disable_y_pruning=disable_y_pruning,
                )
            return row2_desc_cache[key]

        # Action 3 (2026-05-12): pre-bind raw Numba kernels (see fixed-row1 loop
        # above for rationale).  Hot inner loop performs ~10 z9 ops per (V, W)
        # pair; bypassing the Python wrappers saves ~2-3 us / pair.
        if _HAVE_NUMBA:
            _mul_nb = _z9_mul_nb
            _conj_nb = _z9_conj_nb
            _div_nb = _z9_divisible_by_int_nb
            _mulconj_batch = _z9_mul_conj_batch_nb
            _filter_pairs = _z9_inner_filter_pairs_nb
            _PT = _POWER_TABLE
            _ZRE = _ZETA9_RE
            _ZIM = _ZETA9_IM
            _abs2_thresh = float((eps + 1.0e-15) ** 2 * (9 ** f))
            _ndiv_box = np.zeros(1, dtype=np.int64)
            _nzu_box = np.zeros(1, dtype=np.int64)
            _ncross_box = np.zeros(1, dtype=np.int64)
        # Action 1 (2026-05-12): per-chunk caches for materialized V/W root arrays.
        # Keyed on id(desc), which is stable across get_row2_desc() because that
        # function caches its results in row2_desc_cache.
        mat_v_cache: dict[int, np.ndarray] = {}
        mat_w_cache: dict[int, np.ndarray] = {}
        for tri in triples:
            if max_matches > 0 and local_counters.matrices_found >= max_matches:
                break  # Early exit (--max_matches reached).
            local_counters.row1_triples_processed += 1
            y1 = (int(tri[0]), int(tri[1]), int(tri[2]))
            y2 = (int(tri[3]), int(tri[4]), int(tri[5]))
            y3 = (int(tri[6]), int(tri[7]), int(tri[8]))

            desc_a = get_desc(y1, 0, d1)
            if desc_a is None:
                heartbeat(
                    rank=rank,
                    label="main-search",
                    counters=local_counters,
                    nrows=nrows,
                    current_row1_dist=current_row1_dist,
                    start_time=start_time,
                    last_time_box=hb,
                )
                continue
            desc_b = get_desc(y2, 1, 0.0 + 0.0j)
            if desc_b is None:
                continue
            desc_c = get_desc(y3, 2, 0.0 + 0.0j)
            if desc_c is None:
                continue

            for A, za in iter_roots_for_desc(desc_a, phase_meta, batch_cache, f, eps):
                da2 = abs(za - d1) ** 2
                if np.all(A == 0):
                    continue
                M_A = z9_mul_matrix_left(A)
                try:
                    inv_M_A = np.linalg.inv(M_A.astype(np.float64))
                except np.linalg.LinAlgError:
                    continue

                for B, zb in iter_roots_for_desc(desc_b, phase_meta, batch_cache, f, eps):
                    dab2 = da2 + abs(zb) ** 2
                    for C, zc in iter_roots_for_desc(desc_c, phase_meta, batch_cache, f, eps):
                        row1_dist = math.sqrt(dab2 + abs(zc) ** 2)
                        current_row1_dist = row1_dist
                        local_counters.row1_candidates_accepted += 1

                        B_contrib_cache: dict[bytes, np.ndarray] = {}
                        C_contrib_cache: dict[bytes, np.ndarray] = {}

                        # Search compatible row-2 roots by V,W only. U is exact.
                        for ztri in row2_triples:
                            Z1 = (int(ztri[0]), int(ztri[1]), int(ztri[2]))
                            Z2 = (int(ztri[3]), int(ztri[4]), int(ztri[5]))
                            Z3 = (int(ztri[6]), int(ztri[7]), int(ztri[8]))

                            desc_v = get_row2_desc(Z2, d2)
                            if desc_v is None:
                                continue
                            desc_w = get_row2_desc(Z3, 0.0 + 0.0j)
                            if desc_w is None:
                                continue

                            if _HAVE_NUMBA:
                                # 2026-05-12 (Actions A+B+C): fused inner-filter kernel.
                                # Materialize V/W once (id(desc)-cached), batched mul,
                                # then a single Numba call does the divide / |zu| / cross
                                # product divisibility filter, returning survivors only.
                                V_arr = mat_v_cache.get(id(desc_v))
                                if V_arr is None:
                                    V_arr, _ = materialize_roots_for_desc(
                                        desc_v, phase_meta, batch_cache, f, eps
                                    )
                                    mat_v_cache[id(desc_v)] = V_arr
                                W_arr = mat_w_cache.get(id(desc_w))
                                if W_arr is None:
                                    W_arr, _ = materialize_roots_for_desc(
                                        desc_w, phase_meta, batch_cache, f, eps
                                    )
                                    mat_w_cache[id(desc_w)] = W_arr
                                n_v = V_arr.shape[0]
                                n_w = W_arr.shape[0]
                                if n_v == 0 or n_w == 0:
                                    continue
                                BVc_arr = np.empty((n_v, 6), dtype=np.int64)
                                CWc_arr = np.empty((n_w, 6), dtype=np.int64)
                                _mulconj_batch(B, V_arr, _PT, BVc_arr)
                                _mulconj_batch(C, W_arr, _PT, CWc_arr)

                                cap = n_v * n_w
                                local_counters.row2_pairs_tested += cap
                                idx_v_out = np.empty(cap, dtype=np.int64)
                                idx_w_out = np.empty(cap, dtype=np.int64)
                                U_out_arr = np.empty((cap, 6), dtype=np.int64)
                                _ndiv_box[0] = 0
                                _nzu_box[0] = 0
                                _ncross_box[0] = 0
                                n_match = _filter_pairs(
                                    BVc_arr, CWc_arr, V_arr, W_arr, A, B, C, inv_M_A,
                                    _ZRE, _ZIM, _abs2_thresh, scale_int, _PT,
                                    _ndiv_box, _nzu_box, _ncross_box,
                                    idx_v_out, idx_w_out, U_out_arr,
                                )
                                local_counters.orthogonality_divisibility_passes += int(_ndiv_box[0])
                                local_counters.u_norm_passes += int(_ndiv_box[0])
                                local_counters.cross_product_layer_passes += int(_ncross_box[0])
                                for k in range(n_match):
                                    iv = int(idx_v_out[k])
                                    iw = int(idx_w_out[k])
                                    V = V_arr[iv]
                                    W = W_arr[iw]
                                    U = U_out_arr[k]
                                    N1 = _mul_nb(B, W, _PT) - _mul_nb(C, V, _PT)
                                    N2 = _mul_nb(C, U, _PT) - _mul_nb(A, W, _PT)
                                    N3 = _mul_nb(A, V, _PT) - _mul_nb(B, U, _PT)
                                    R31 = z9_conj(z9_divide_by_int(N1, scale_int))
                                    R32 = z9_conj(z9_divide_by_int(N2, scale_int))
                                    R33 = z9_conj(z9_divide_by_int(N3, scale_int))
                                    rows = np.stack(
                                        [
                                            np.stack([A, B, C], axis=0),
                                            np.stack([U, V, W], axis=0),
                                            np.stack([R31, R32, R33], axis=0),
                                        ],
                                        axis=0,
                                    ).astype(np.int64, copy=False)
                                    dist_fro = matrix_frobenius_dist(rows, f, diag)
                                    local_counters.matrices_found += 1
                                    result = {
                                        "dist_fro": float(dist_fro),
                                        "row1_dist": float(row1_dist),
                                        "row1_Y": [list(y1), list(y2), list(y3)],
                                        "row2_Y": [list(Z1), list(Z2), list(Z3)],
                                        "rows_coeffs_layer_f": rows.tolist(),
                                        "row_complex": [
                                            [
                                                [float(z.real), float(z.imag)]
                                                for z in row_complex(rows[i], f)
                                            ]
                                            for i in range(3)
                                        ],
                                    }
                                    maybe_push_result(heap, result, keep)
                                    heartbeat(
                                        rank=rank,
                                        label="main-search",
                                        counters=local_counters,
                                        nrows=nrows,
                                        current_row1_dist=current_row1_dist,
                                        start_time=start_time,
                                        last_time_box=hb,
                                    )
                                continue  # skip the legacy per-V/W path

                            # Legacy (no-Numba) path
                            for V, zv in iter_roots_for_desc(desc_v, phase_meta, batch_cache, f, eps):
                                key_v = V.tobytes()
                                BVc = B_contrib_cache.get(key_v)
                                if BVc is None:
                                    BVc = z9_mul(B, z9_conj(V))
                                    B_contrib_cache[key_v] = BVc

                                for W, zw in iter_roots_for_desc(desc_w, phase_meta, batch_cache, f, eps):
                                    local_counters.row2_pairs_tested += 1
                                    key_w = W.tobytes()
                                    CWc = C_contrib_cache.get(key_w)
                                    if CWc is None:
                                        CWc = z9_mul(C, z9_conj(W))
                                        C_contrib_cache[key_w] = CWc

                                    T = BVc + CWc
                                    Q = z9_divide_if_exact_by_a(T, A, inv_M_A)
                                    if Q is None:
                                        continue
                                    local_counters.orthogonality_divisibility_passes += 1

                                    U = z9_neg(z9_conj(Q))
                                    # Disabled 2026-05-12: see comment in main-search loop above.
                                    # if z9_norm_m012(U) != Z1:
                                    #     continue
                                    local_counters.u_norm_passes += 1

                                    zu = coeff_row_to_complex_scaled(U, f)
                                    if abs(zu) > eps + 1.0e-15:
                                        continue

                                    # Cross-product numerators for row1 x row2.
                                    N1 = z9_sub(z9_mul(B, W), z9_mul(C, V))
                                    N2 = z9_sub(z9_mul(C, U), z9_mul(A, W))
                                    N3 = z9_sub(z9_mul(A, V), z9_mul(B, U))
                                    if not (
                                        z9_divisible_by_int(N1, scale_int)
                                        and z9_divisible_by_int(N2, scale_int)
                                        and z9_divisible_by_int(N3, scale_int)
                                    ):
                                        continue

                                    local_counters.cross_product_layer_passes += 1

                                    R31 = z9_conj(z9_divide_by_int(N1, scale_int))
                                    R32 = z9_conj(z9_divide_by_int(N2, scale_int))
                                    R33 = z9_conj(z9_divide_by_int(N3, scale_int))

                                    rows = np.stack(
                                        [
                                            np.stack([A, B, C], axis=0),
                                            np.stack([U, V, W], axis=0),
                                            np.stack([R31, R32, R33], axis=0),
                                        ],
                                        axis=0,
                                    ).astype(np.int64, copy=False)

                                    dist_fro = matrix_frobenius_dist(rows, f, diag)
                                    local_counters.matrices_found += 1

                                    result = {
                                        "dist_fro": float(dist_fro),
                                        "row1_dist": float(row1_dist),
                                        "row1_Y": [list(y1), list(y2), list(y3)],
                                        "row2_Y": [list(Z1), list(Z2), list(Z3)],
                                        "rows_coeffs_layer_f": rows.tolist(),
                                        "row_complex": [
                                            [
                                                [float(z.real), float(z.imag)]
                                                for z in row_complex(rows[i], f)
                                            ]
                                            for i in range(3)
                                        ],
                                    }
                                    maybe_push_result(heap, result, keep)

                                    heartbeat(
                                        rank=rank,
                                        label="main-search",
                                        counters=local_counters,
                                        nrows=nrows,
                                        current_row1_dist=current_row1_dist,
                                        start_time=start_time,
                                        last_time_box=hb,
                                    )

                        heartbeat(
                            rank=rank,
                            label="main-search",
                            counters=local_counters,
                            nrows=nrows,
                            current_row1_dist=current_row1_dist,
                            start_time=start_time,
                            last_time_box=hb,
                        )

        heartbeat(
            rank=rank,
            label="main-search",
            counters=local_counters,
            nrows=nrows,
            current_row1_dist=current_row1_dist,
            start_time=start_time,
            force=(rank == 0),
            last_time_box=hb,
        )

    local_results = [item.result for item in heap]
    payload = {
        "rank": rank,
        "counters": asdict(local_counters),
        "results": local_results,
    }
    gathered = comm.gather(payload, root=0)

    if rank != 0:
        return None

    total = Counters()
    final_heap: list[HeapItem] = []
    by_rank = {}
    for part in gathered:
        c = Counters(**part["counters"])
        total.add(c)
        by_rank[str(part["rank"])] = part["counters"]
        for r in part["results"]:
            maybe_push_result(final_heap, r, keep)

    final_results = [item.result for item in sorted(final_heap, key=lambda x: -x.priority)]
    final_results.sort(key=lambda r: float(r["dist_fro"]))

    out_npz = output_prefix + ".npz"
    if final_results:
        rows_arr = np.asarray([r["rows_coeffs_layer_f"] for r in final_results], dtype=np.int64)
        dist_arr = np.asarray([r["dist_fro"] for r in final_results], dtype=np.float64)
    else:
        rows_arr = np.empty((0, 3, 3, 6), dtype=np.int64)
        dist_arr = np.empty((0,), dtype=np.float64)
    np.savez(
        out_npz,
        rows_coeffs_layer_f=rows_arr,
        dist_fro=dist_arr,
        diag=np.asarray(diag, dtype=np.complex128),
        f=np.asarray([f], dtype=np.int64),
        eps=np.asarray([eps], dtype=np.float64),
    )

    summary = {
        "results_npz": os.path.abspath(out_npz),
        "results_json": os.path.abspath(output_prefix + ".json"),
        "profile_json": os.path.abspath(output_prefix + ".profile.json"),
        "n_results": len(final_results),
        "best_dist_fro": None if not final_results else float(final_results[0]["dist_fro"]),
        "f": int(f),
        "eps": float(eps),
        "diag": [[float(z.real), float(z.imag)] for z in diag],
        "triples_file": os.path.abspath(triples_file),
        "triples_json": os.path.abspath(triples_json),
        "rootdb_prefix": os.path.abspath(rootdb_prefix),
        "mpi_ranks": int(size),
        "n_phase_bins": int(n_phase_bins),
        "row2_norm_triples": int(row2_triples.shape[0]),
        "counters_sum": asdict(total),
        "top_results": final_results,
    }

    with open(output_prefix + ".json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with open(output_prefix + ".profile.json", "w") as fh:
        json.dump({"by_rank": by_rank, "sum": asdict(total)}, fh, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _do_search(
    *,
    args,
    theta: Optional[float],
    eps: float,
    d1_arg,
    d2_arg,
    d3_arg,
    fixed_row1_coeffs: Optional[np.ndarray],
    fixed_row2_coeffs: Optional[np.ndarray],
    verbose: bool,
) -> Optional[dict]:
    """Run a single search with explicit per-job parameters.

    Centralizes the theta/d1/d2/d3 resolution and the call to
    search_diagonal_two_rows_streamed so it can be shared between the
    one-shot CLI path and the daemon loop.
    """
    if theta is not None:
        d1 = complex(np.exp(-0.5j * float(theta)))
        d2 = complex(np.exp(0.5j * float(theta)))
        d3 = 1.0 + 0.0j
    else:
        if d1_arg is None or d2_arg is None:
            raise SystemExit("Either provide --theta, or provide both --d1 and --d2.")
        d1 = complex(d1_arg)
        d2 = complex(d2_arg)
        d3 = complex(d3_arg)

    return search_diagonal_two_rows_streamed(
        triples_file=args.triples_file,
        triples_json=args.triples_json,
        rootdb_prefix=args.rootdb_prefix,
        f=args.f,
        d1=d1,
        d2=d2,
        d3=d3,
        eps=eps,
        output_prefix=args.output_prefix,
        triples_chunk_rows=args.triples_chunk_rows,
        n_phase_bins=args.n_phase_bins,
        chunk_meta_json=args.chunk_meta_json,
        row2_cache=args.row2_cache,
        keep=args.keep,
        disable_y_pruning=args.disable_y_pruning,
        fixed_row1_coeffs=fixed_row1_coeffs,
        fixed_row2_coeffs=fixed_row2_coeffs,
        verbose=verbose,
        max_matches=args.max_matches,
    )


def _summary_to_daemon_result(summary: Optional[dict], wall_seconds: float) -> dict:
    """Map the rich summary dict from search_diagonal_two_rows_streamed to the
    minimal one-line daemon protocol."""
    if not summary:
        return {"success": False, "n_results": 0, "wall_seconds": float(wall_seconds)}
    n_results = int(summary.get("n_results", 0) or 0)
    if n_results <= 0:
        return {"success": False, "n_results": 0, "wall_seconds": float(wall_seconds)}
    top = summary.get("top_results") or []
    best = top[0] if top else None
    row_u_coeffs = None
    if best is not None:
        rows = best.get("rows_coeffs_layer_f")
        if rows is not None:
            # rows is a (3,3,6) nested list; row_u is the second row.
            try:
                row_u_coeffs = [list(map(int, rows[1][k])) for k in range(3)]
            except Exception:
                row_u_coeffs = None
    return {
        "success": True,
        "frobenius": (
            float(summary["best_dist_fro"])
            if summary.get("best_dist_fro") is not None
            else None
        ),
        "row_u_coeffs": row_u_coeffs,
        "n_results": n_results,
        "wall_seconds": float(wall_seconds),
    }


def _daemon_loop(args) -> None:
    """Read JSON jobs from stdin (rank 0), broadcast to all ranks, and run
    repeated searches without re-paying Sage/MPI/import startup.

    Stdout (rank 0) emits exactly one line of JSON per job, so the parent
    process can parse it line-by-line. All other prints from the search
    pipeline are redirected to stderr so they cannot corrupt the protocol.
    Loop terminates cleanly when rank 0 sees EOF on stdin.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    real_stdout = sys.stdout

    if rank == 0:
        # Daemon banner goes to stderr so stdout stays a clean JSON stream.
        print(
            f"[daemon] ready ranks={comm.Get_size()} f={args.f} "
            f"triples={args.triples_file} rootdb={args.rootdb_prefix}",
            file=sys.stderr,
            flush=True,
        )

    while True:
        # Rank 0 reads one line of JSON from stdin, others wait at bcast.
        job = None
        if rank == 0:
            try:
                line = sys.stdin.readline()
            except Exception as exc:
                print(f"[daemon] stdin read error: {exc!r}", file=sys.stderr, flush=True)
                line = ""
            if not line:
                # EOF -> sentinel None tells all ranks to exit.
                job = None
            else:
                line = line.strip()
                if not line:
                    job = {"__skip__": True}
                else:
                    try:
                        parsed = json.loads(line)
                        if not isinstance(parsed, dict):
                            job = {"__error__": f"job must be a JSON object, got {type(parsed).__name__}"}
                        else:
                            job = parsed
                    except Exception as exc:
                        job = {"__error__": f"invalid JSON: {exc}"}

        job = comm.bcast(job, root=0)

        if job is None:
            if rank == 0:
                print("[daemon] EOF on stdin, shutting down", file=sys.stderr, flush=True)
            break

        if rank == 0 and job.get("__skip__"):
            continue

        if rank == 0 and "__error__" in job:
            print(
                json.dumps({"success": False, "error": job["__error__"], "wall_seconds": 0.0}),
                file=real_stdout,
                flush=True,
            )
            continue

        # Build per-job parameters. Required: theta and eps. Optional:
        # fixed_row1 (Sage one-row string).
        t0 = time.monotonic()
        per_job_error: Optional[str] = None
        theta_val: Optional[float] = None
        eps_val: Optional[float] = None
        fixed_row1_coeffs: Optional[np.ndarray] = None
        try:
            if "theta" not in job or "eps" not in job:
                raise ValueError("job missing required field 'theta' or 'eps'")
            theta_val = float(job["theta"])
            eps_val = float(job["eps"])
            row1_str = job.get("fixed_row1")
            if row1_str is not None and str(row1_str).strip():
                fixed_row1_coeffs = parse_fixed_row1_sage(str(row1_str), args.f)
        except Exception as exc:
            per_job_error = f"job parse error: {exc}"

        if per_job_error is not None:
            wall = time.monotonic() - t0
            if rank == 0:
                print(
                    json.dumps({"success": False, "error": per_job_error, "wall_seconds": float(wall)}),
                    file=real_stdout,
                    flush=True,
                )
            # All ranks loop back together.
            continue

        # Run the search. Redirect stdout to stderr for the duration so any
        # internal prints (heartbeats, summary dump) do not corrupt the
        # protocol; restore real stdout afterwards on rank 0 for the JSON line.
        summary: Optional[dict] = None
        run_error: Optional[str] = None
        try:
            with contextlib.redirect_stdout(sys.stderr):
                summary = _do_search(
                    args=args,
                    theta=theta_val,
                    eps=eps_val,
                    d1_arg=None,
                    d2_arg=None,
                    d3_arg="1",
                    fixed_row1_coeffs=fixed_row1_coeffs,
                    fixed_row2_coeffs=None,
                    verbose=not args.quiet,
                )
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"

        wall = time.monotonic() - t0

        if rank == 0:
            if run_error is not None:
                out = {"success": False, "error": run_error, "wall_seconds": float(wall)}
            else:
                out = _summary_to_daemon_result(summary, wall)
            print(json.dumps(out), file=real_stdout, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Streamed exact two-row search for a diagonal 3x3 target over Z[zeta9,1/3]."
    )
    p.add_argument("--triples_file", required=True)
    p.add_argument("--triples_json", required=True)
    p.add_argument("--rootdb_prefix", required=True)
    p.add_argument("--f", type=int, required=True)
    p.add_argument("--eps", type=float, required=False, default=None,
                   help="Required in non-daemon mode. In --daemon mode, eps is supplied per job via stdin.")
    p.add_argument("--output_prefix", required=True)
    p.add_argument("--d1", type=parse_complex, default=None, help="First diagonal entry, e.g. '0.7-0.2j'.")
    p.add_argument("--d2", type=parse_complex, default=None, help="Second diagonal entry, e.g. '0.7+0.2j'.")
    p.add_argument("--d3", type=parse_complex, default="1", help="Third diagonal entry. Default: 1.")
    p.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Convenience option: set d1=exp(-0.5j*theta), d2=exp(0.5j*theta), d3=1.",
    )
    p.add_argument("--triples_chunk_rows", type=int, default=200000)
    p.add_argument("--n_phase_bins", type=int, default=512)
    p.add_argument("--chunk_meta_json", default=None)
    p.add_argument("--row2_cache", default=None, help="Optional .npy cache for admissible row2 norm triples.")
    p.add_argument("--fixed_row1", default=None, help="Fixed row 1 as a (3,6) JSON/Python list, .json, or .npy file.")
    p.add_argument("--fixed_row1_sage", default=None, help="Fixed row 1 in Sage printed one-row matrix form, or a text file containing it.")
    p.add_argument("--fixed_row2", default=None, help="Fixed row 2 as a (3,6) JSON/Python list, .json, or .npy file. Requires fixed row 1.")
    p.add_argument("--fixed_row2_sage", default=None, help="Fixed row 2 in Sage printed one-row matrix form, or a text file containing it. Requires fixed row 1.")
    p.add_argument("--keep", type=int, default=16)
    p.add_argument("--disable_y_pruning", action="store_true")
    p.add_argument("--max_matches", type=int, default=0,
                   help="If > 0, break out of the row1 loop once this many matrices "
                        "have been pushed onto the heap.  Lets short sanity runs finish "
                        "in minutes instead of scanning all row1 triples (which is "
                        "redundant once the heap is full and saturated).")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--daemon",
        action="store_true",
        help=(
            "Persistent daemon mode: initialize once, then loop reading "
            "newline-delimited JSON jobs from stdin and writing one-line "
            "JSON results to stdout. Exits cleanly on EOF. Reuses the same "
            "(triples_file, rootdb_prefix, f) for every job; theta, eps, "
            "and optional fixed_row1 (Sage string) are read per job."
        ),
    )
    args = p.parse_args()

    if args.daemon:
        # In daemon mode, theta/eps/d1/d2/fixed_row* on the CLI are ignored;
        # those come per-job from stdin. Reject the two-fixed-rows diagnostic
        # mode entirely (it is a one-shot diagnostic and does not match the
        # daemon protocol).
        if args.fixed_row2 is not None or args.fixed_row2_sage is not None:
            raise SystemExit("--fixed_row2/--fixed_row2_sage are not supported in --daemon mode.")
        _daemon_loop(args)
        return

    # ---- one-shot (non-daemon) path: behavior must match the original ----
    if args.eps is None:
        raise SystemExit("--eps is required (only optional in --daemon mode).")

    if args.fixed_row1 is not None and args.fixed_row1_sage is not None:
        raise SystemExit("Use only one of --fixed_row1 and --fixed_row1_sage.")
    if args.fixed_row2 is not None and args.fixed_row2_sage is not None:
        raise SystemExit("Use only one of --fixed_row2 and --fixed_row2_sage.")
    fixed_row1_coeffs = None
    if args.fixed_row1 is not None:
        fixed_row1_coeffs = parse_fixed_row1(args.fixed_row1)
    elif args.fixed_row1_sage is not None:
        fixed_row1_coeffs = parse_fixed_row1_sage(args.fixed_row1_sage, args.f)

    fixed_row2_coeffs = None
    if args.fixed_row2 is not None:
        fixed_row2_coeffs = parse_fixed_row2(args.fixed_row2)
    elif args.fixed_row2_sage is not None:
        fixed_row2_coeffs = parse_fixed_row2_sage(args.fixed_row2_sage, args.f)
    if fixed_row2_coeffs is not None and fixed_row1_coeffs is None:
        raise SystemExit("--fixed_row2/--fixed_row2_sage requires --fixed_row1 or --fixed_row1_sage.")

    _do_search(
        args=args,
        theta=args.theta,
        eps=args.eps,
        d1_arg=args.d1,
        d2_arg=args.d2,
        d3_arg=args.d3,
        fixed_row1_coeffs=fixed_row1_coeffs,
        fixed_row2_coeffs=fixed_row2_coeffs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
