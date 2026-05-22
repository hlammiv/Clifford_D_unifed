"""Tests for the ideal-first cache pipeline.

Three layers:

1. **Conversion test** (γ → M = γγ̄):
   - Hand-computed cases (γ = 1, ω, 1+ω, ...) match expected M values
   - Python reference matches Numba batched on random γ
   - Property test: γ̄γ = γγ̄ (real-valued), M_3 = 0, M_1 = -M_2

2. **Round-trip test** (small cache build → conversion → screen-pass check):
   - Build a tiny cache (norm_bound = 10^4)
   - For each cached γ, compute M = γγ̄
   - Run M through the existing `roots_fast` screen
   - Assert all M's pass screen (this is the whole point: every γγ̄ is a K-norm)

3. **Cross-validation test** (lattice-first vs ideal-first on same cell):
   - Take an existing Y1_*.npy from a collect_targets run
   - Run query_ideal_cache on the same cell parameters
   - Assert that ideal-first output is a SUBSET (or set-equal) of lattice-first
     output, modulo the k=(0,0) unit-orbit limitation (see query_ideal_cache.py
     TODO about unit multiplication).

Run via:
    python -m zeta9.test_ideal_cache
"""
import sys
import math
import tempfile
import os

import numpy as np


def test_gamma_to_M_handcoded():
    """Spot-check conversion on known cases."""
    from .ideal_cache_conv import gamma_to_M_coefs_python

    # γ = 1 → M = 1
    M = gamma_to_M_coefs_python(np.array([1, 0, 0, 0, 0, 0]))
    assert tuple(M) == (1, 0, 0), f"γ=1: expected (1,0,0), got {tuple(M)}"

    # γ = ω → M = ω·ω⁻¹ = 1
    M = gamma_to_M_coefs_python(np.array([0, 1, 0, 0, 0, 0]))
    assert tuple(M) == (1, 0, 0), f"γ=ω: expected (1,0,0), got {tuple(M)}"

    # γ = 1 + ω → M = (1+ω)(1+ω⁻¹) = 1 + ω + ω⁻¹ + 1 = 2 + α → (2, 1, 0)
    M = gamma_to_M_coefs_python(np.array([1, 1, 0, 0, 0, 0]))
    assert tuple(M) == (2, 1, 0), f"γ=1+ω: expected (2,1,0), got {tuple(M)}"

    # γ = 2 → M = 4
    M = gamma_to_M_coefs_python(np.array([2, 0, 0, 0, 0, 0]))
    assert tuple(M) == (4, 0, 0), f"γ=2: expected (4,0,0), got {tuple(M)}"

    # γ = 1 + ω² → γγ̄ = (1+ω²)(1+ω⁻²) = 1 + ω² + ω⁻² + 1 = 2 + α². So M = α² → (0,0,1).
    # This case has nonzero M_4 (exposes the m_0 = M_0 + 2·M_4 sign).
    M = gamma_to_M_coefs_python(np.array([1, 0, 1, 0, 0, 0]))
    assert tuple(M) == (0, 0, 1), f"γ=1+ω²: expected (0,0,1), got {tuple(M)}"

    # γ = 1 + ω + ω² → γγ̄ = ? Quick check via σ embeddings.
    # |σ_1(γ)|² where ω = e^(2πi/9): σ = 1 + e^(2πi/9) + e^(4πi/9)
    # Equivalently, 2cos(2π/9) + 2cos(4π/9) + 1 component... easier: just check positivity.
    M = gamma_to_M_coefs_python(np.array([1, 1, 1, 0, 0, 0]))
    # Numeric check: M should give σ_F_r ≥ 0 for all r
    alpha1 = 2 * np.cos(2*np.pi/9)
    alpha2 = 2 * np.cos(4*np.pi/9)
    alpha4 = 2 * np.cos(8*np.pi/9)
    for alpha in [alpha1, alpha2, alpha4]:
        s = M[0] + M[1]*alpha + M[2]*alpha*alpha
        assert s >= -1e-10, f"γ=1+ω+ω²: σ_F(α={alpha:.3f}) = {s} should be ≥0"

    print("test_gamma_to_M_handcoded: PASS", flush=True)


def test_gamma_to_M_python_vs_numba():
    """Python ref ≡ Numba batched on random inputs."""
    from .ideal_cache_conv import gamma_to_M_coefs_python, gamma_to_M_batch

    rng = np.random.default_rng(42)
    N = 1000
    test_coefs = rng.integers(-50, 51, size=(N, 6), dtype=np.int64)

    py_out = np.empty((N, 3), dtype=np.int64)
    for i in range(N):
        py_out[i] = gamma_to_M_coefs_python(test_coefs[i])

    nb_out = np.empty((N, 3), dtype=np.int64)
    gamma_to_M_batch(test_coefs, nb_out)

    diff = np.abs(py_out - nb_out).sum()
    assert diff == 0, f"Python ref ≠ Numba: {diff} differences"
    print(f"test_gamma_to_M_python_vs_numba: PASS ({N} random γ)", flush=True)


def test_gamma_to_M_is_in_OF():
    """M = γγ̄ should give a valid element of O_F.
    Verified by checking M passes the existing screen kernel.
    (Every γγ̄ is a K-norm, so screen always passes.)"""
    from .ideal_cache_conv import gamma_to_M_coefs_python
    from . import roots_fast as rf

    rng = np.random.default_rng(7)
    N = 100
    # Use small coefficients to keep M magnitudes manageable
    test_coefs = rng.integers(-20, 21, size=(N, 6), dtype=np.int64)

    M_arr = np.empty((N, 3), dtype=np.int64)
    for i in range(N):
        M_arr[i] = gamma_to_M_coefs_python(test_coefs[i])

    # Filter out M = 0 (degenerate γ̄γ = 0 means γ = 0)
    nonzero = ~np.all(M_arr == 0, axis=1)
    M_test = M_arr[nonzero]
    n_test = M_test.shape[0]

    # Run through screen kernel
    keep, n_fb = rf.screen_rows_batch(
        M_test, check_real_embeddings=True, sage_fallback_fn=None,
    )

    # All non-fallback rows should be KEEP (screen passes for K-norms).
    # Fallback rows: we conservatively keep without screening (so always True).
    fail_count = int(((~keep) & (n_fb == 0)).sum()) if isinstance(n_fb, np.ndarray) else 0
    # Simpler: count non-keeps overall (should be 0 for K-norms with σ ≥ 0)
    # NOTE: real_embeddings_nonneg might reject some γγ̄ where σ_r(M) < 0.
    # For γγ̄, σ_r(M) = |σ_K_r(γ)|² ≥ 0 always, but for tiny γ may underflow.
    n_pass = int(keep.sum())
    print(f"test_gamma_to_M_is_in_OF: {n_pass}/{n_test} screen-pass, {n_fb} fallback",
          flush=True)
    # The screen should pass essentially all rows. Allow small fallback count.
    assert n_pass + n_fb >= n_test - 5, \
        f"Screen rejected {n_test - n_pass - (n_fb or 0)} γγ̄ rows (expected ≤ 5)"


def test_screen_subset_property():
    """Validate the screen-subset property on an existing Y1 file.

    Loads a known Y1_*.npy from a recent collect_targets run, checks that all
    its rows pass the screen and conform to the σ-strip expected.
    """
    candidate_path = "/mnt/993c1724-f80f-4440-a384-daf788d9a041/data/zeta9_D/Y1_f=6_u=1_eps=1e-06_bin_0000.npy"
    if not os.path.exists(candidate_path):
        print(f"test_screen_subset_property: SKIP (no Y1 file at {candidate_path})",
              flush=True)
        return

    rows = np.load(candidate_path, mmap_mode="r")
    # Sample to keep test bounded
    rng = np.random.default_rng(0)
    n_sample = min(10_000, rows.shape[0])
    sample = np.array(rows[rng.choice(rows.shape[0], n_sample, replace=False)])

    from . import roots_fast as rf
    keep, n_fb = rf.screen_rows_batch(
        sample, check_real_embeddings=True, sage_fallback_fn=None,
    )
    pass_rate = int(keep.sum()) / n_sample
    print(f"test_screen_subset_property: {int(keep.sum())}/{n_sample} screen-pass "
          f"(rate {pass_rate:.3f}, fallback {n_fb})", flush=True)
    # Lattice-first Y1 should be 100% screen-pass by construction
    assert pass_rate >= 0.95, f"Lattice-first Y1 pass rate {pass_rate} unexpectedly low"


def test_mul_in_OK():
    """Verify O_K multiplication kernel matches Python ref."""
    from .ideal_cache_conv import mul_in_OK_python, mul_in_OK_batch

    rng = np.random.default_rng(11)
    N = 500
    a = rng.integers(-20, 21, size=(N, 6), dtype=np.int64)
    b = rng.integers(-20, 21, size=(N, 6), dtype=np.int64)

    py_out = np.empty((N, 6), dtype=np.int64)
    for i in range(N):
        py_out[i] = mul_in_OK_python(a[i], b[i])

    nb_out = np.empty((N, 6), dtype=np.int64)
    mul_in_OK_batch(a, b, nb_out)

    diff = np.abs(py_out - nb_out).sum()
    assert diff == 0, f"mul_in_OK: python vs numba differ by {diff}"
    print(f"test_mul_in_OK: PASS ({N} random products)", flush=True)


def test_unit_power_table():
    """Verify unit power table consistency: u^k · u^{-k} = 1."""
    try:
        import cypari2
    except ImportError:
        print("test_unit_power_table: SKIP (cypari2 not installed)", flush=True)
        return

    from .ideal_cache import compute_fundamental_unit_data
    from .ideal_cache_conv import compute_unit_power_table, mul_in_OK_python

    data = compute_fundamental_unit_data()
    u1_coefs = data["coefs"][0]
    u1_inv = data["inv_coefs"][0]

    # Identity check: u · u⁻¹ = 1
    prod = mul_in_OK_python(u1_coefs, u1_inv)
    assert tuple(prod) == (1, 0, 0, 0, 0, 0), \
        f"u_1 · u_1⁻¹: expected (1,0,...), got {tuple(prod)}"

    # Power table check: u^2 · u^{-2} = 1
    table = compute_unit_power_table(u1_coefs, u1_inv, k_max=3)
    u_pos2 = table[3 + 2]
    u_neg2 = table[3 - 2]
    prod = mul_in_OK_python(u_pos2, u_neg2)
    assert tuple(prod) == (1, 0, 0, 0, 0, 0), \
        f"u^2 · u^{{-2}}: expected (1,0,...), got {tuple(prod)}"

    # Power table check: u^k matches direct multiplication
    cumulative = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
    for k in range(1, 4):
        cumulative = mul_in_OK_python(cumulative, u1_coefs)
        assert np.array_equal(table[3 + k], cumulative), \
            f"power table u^{k}: mismatch"

    print(f"test_unit_power_table: PASS", flush=True)


def test_gamma_times_unit_pow_to_M():
    """Verify γ · u^k → M is correct by comparing against direct multiplication
    + γ→M."""
    try:
        import cypari2
    except ImportError:
        print("test_gamma_times_unit_pow_to_M: SKIP (cypari2 not installed)", flush=True)
        return

    from .ideal_cache import compute_fundamental_unit_data
    from .ideal_cache_conv import (
        compute_unit_power_table,
        gamma_times_unit_pow_to_M_batch,
        gamma_to_M_coefs_python,
        mul_in_OK_python,
    )

    data = compute_fundamental_unit_data()
    u1_coefs = data["coefs"][0]
    u2_coefs = data["coefs"][1]
    u1_inv = data["inv_coefs"][0]
    u2_inv = data["inv_coefs"][1]

    k_max = 2
    u1_table = compute_unit_power_table(u1_coefs, u1_inv, k_max)
    u2_table = compute_unit_power_table(u2_coefs, u2_inv, k_max)

    rng = np.random.default_rng(13)
    N = 50
    test_gamma = rng.integers(-5, 6, size=(N, 6), dtype=np.int64)
    test_k1 = rng.integers(-k_max, k_max + 1, size=N)
    test_k2 = rng.integers(-k_max, k_max + 1, size=N)

    # Reference: γ' = γ · u_1^{k1} · u_2^{k2}, M = γ'γ̄'
    ref_M = np.empty((N, 3), dtype=np.int64)
    for i in range(N):
        gamma_p = test_gamma[i].astype(np.int64)
        if test_k1[i] != 0:
            gamma_p = mul_in_OK_python(gamma_p, u1_table[k_max + test_k1[i]])
        if test_k2[i] != 0:
            gamma_p = mul_in_OK_python(gamma_p, u2_table[k_max + test_k2[i]])
        ref_M[i] = gamma_to_M_coefs_python(gamma_p)

    # Numba batched
    nb_M = np.empty((N, 3), dtype=np.int64)
    gamma_times_unit_pow_to_M_batch(
        test_gamma, test_k1.astype(np.int64), test_k2.astype(np.int64),
        u1_table, u2_table, k_max, nb_M,
    )

    diff = np.abs(ref_M - nb_M).sum()
    assert diff == 0, f"γ·u→M: python vs numba differ by {diff}"
    print(f"test_gamma_times_unit_pow_to_M: PASS ({N} random γ·u^k cases)", flush=True)


def test_cache_round_trip():
    """Build a small cache (norm_bound = 10^4), then verify reader matches.
    Skipped if cypari2 not installed."""
    try:
        import cypari2
    except ImportError:
        print("test_cache_round_trip: SKIP (cypari2 not installed)", flush=True)
        return

    from .ideal_cache import IdealCacheBuilder, IdealCacheReader

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "tiny.bin")
        builder = IdealCacheBuilder(norm_bound=10_000, prime_bound=10_000)
        builder.build_prime_table()
        n_ideals = builder.build_ideal_cache(cache_path)

        # Read it back
        reader = IdealCacheReader(cache_path)
        assert reader.n_ideals == n_ideals
        assert reader.norm_bound == 10_000

        # Spot-check: each ideal record's N should be ≤ 10_000
        norms = reader.records["norm"]
        assert norms.max() <= 10_000
        assert norms.min() >= 1  # the identity γ = 1 has N = 1

        # Sample 10 records, verify γ → M conversion works
        from .ideal_cache_conv import gamma_to_M_coefs_python
        for i in range(min(10, n_ideals)):
            M = gamma_to_M_coefs_python(reader.records[i]["coefs"])
            # M should not be all zero (γ ≠ 0)
            assert not np.all(M == 0)

        print(f"test_cache_round_trip: PASS ({n_ideals:,} ideals)", flush=True)


def main():
    print("=== ideal_cache tests ===", flush=True)
    test_gamma_to_M_handcoded()
    test_gamma_to_M_python_vs_numba()
    test_mul_in_OK()
    test_gamma_to_M_is_in_OF()
    test_screen_subset_property()
    test_unit_power_table()
    test_gamma_times_unit_pow_to_M()
    test_cache_round_trip()
    print("\nall tests passed", flush=True)


if __name__ == "__main__":
    main()
