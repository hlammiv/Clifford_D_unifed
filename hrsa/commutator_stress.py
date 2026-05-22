"""Stress-test factor_commutator across a wide eps range.

Validates that the SU(3) group-commutator factorization keeps contracting at
tight eps needed for deep Solovay-Kitaev recursion. At SK depth k, the residual
shrinks like eps_0^{2^k} -- so for eps_0 = 0.3 and depth k = 4 we already need
factor_commutator to work cleanly at eps ~ 1e-8.

For each target eps we generate n_trials random near-identity unitaries
U = exp(i * eps * H_hat) with H_hat a unit-Frobenius Hermitian traceless
Gell-Mann sum, run factor_commutator, and tally:
  - |A - I|_F, |B - I|_F       (SK requires both ~ sqrt(eps), const ~ 1.4)
  - residual = |[A,B] - U|_F
  - contraction ratio residual / eps (must be << 1 for SK to recurse)
  - failures (ratio >= 1)

Pure numpy + scipy. Run as a script for the full table.
"""
import json
import os
import numpy as np

from su3_commutator import group_commutator, factor_commutator
from su3_lie import GELL_MANN, matrix_exp_su3


I3 = np.eye(3, dtype=complex)


def random_near_identity(eps, rng):
    """Random U in SU(3) with |U - I|_F approximately eps (small-eps regime).

    Build H = sum_a c_a lambda_a with c ~ N(0, I_8) rescaled so |H|_F = eps,
    then return U = exp(i H). For small eps, |U - I|_F approx |H|_F = eps.
    """
    c = rng.standard_normal(8)
    # |H|_F^2 = sum_a c_a^2 * |lambda_a|_F^2.  All Gell-Mann matrices have
    # |lambda_a|_F^2 = 2, so |H|_F = sqrt(2) * |c|_2.  Scale c so |H|_F = eps.
    c = c * (eps / (np.sqrt(2.0) * np.linalg.norm(c)))
    U = matrix_exp_su3(c)
    return U


def stress_test_at_eps(eps, n_trials, rng):
    """Run n_trials factor_commutator trials at target eps; return summary dict."""
    eps_actual = np.empty(n_trials, dtype=float)
    A_dist = np.empty(n_trials, dtype=float)
    B_dist = np.empty(n_trials, dtype=float)
    residuals = np.empty(n_trials, dtype=float)
    ratios = np.empty(n_trials, dtype=float)
    failures = 0

    for k in range(n_trials):
        U = random_near_identity(eps, rng)
        E = U  # E is the near-identity unitary we want to factor.
        eps_k = float(np.linalg.norm(E - I3, ord="fro"))
        eps_actual[k] = eps_k

        A, B, residual = factor_commutator(E)
        A_dist[k] = float(np.linalg.norm(A - I3, ord="fro"))
        B_dist[k] = float(np.linalg.norm(B - I3, ord="fro"))
        residuals[k] = residual
        # Use eps_target (not eps_actual) so ratios at the smallest eps stay
        # interpretable even if logm/optimizer noise floors eps_actual.
        ratios[k] = residual / eps if eps > 0 else float("inf")
        if residual >= eps:
            failures += 1

    sqrt_eps = np.sqrt(eps) if eps > 0 else 1.0
    return {
        "eps_target": float(eps),
        "eps_actual_med": float(np.median(eps_actual)),
        "A_minus_I_F_med": float(np.median(A_dist)),
        "A_minus_I_F_max": float(np.max(A_dist)),
        "A_over_sqrt_eps_med": float(np.median(A_dist) / sqrt_eps),
        "B_minus_I_F_med": float(np.median(B_dist)),
        "residual_med": float(np.median(residuals)),
        "residual_max": float(np.max(residuals)),
        "contraction_ratio_med": float(np.median(ratios)),
        "contraction_ratio_max": float(np.max(ratios)),
        "failures": int(failures),
        "n_trials": int(n_trials),
    }


def run_full_sweep(n_trials_per_eps=10, eps_values=None, rng_seed=0):
    """Sweep factor_commutator across eps_values; return list of stress dicts."""
    if eps_values is None:
        eps_values = [0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    rng = np.random.default_rng(rng_seed)
    results = []
    for eps in eps_values:
        results.append(stress_test_at_eps(eps, n_trials_per_eps, rng))
    return results


def _print_table(results):
    header = (
        f"{'eps_tgt':>10s} {'eps_act':>10s} {'<A-I>/vEps':>12s} "
        f"{'<A-I>_F':>10s} {'<resid>':>11s} {'<ratio>':>11s} "
        f"{'max_ratio':>11s} {'fails':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['eps_target']:>10.2e} "
            f"{r['eps_actual_med']:>10.2e} "
            f"{r['A_over_sqrt_eps_med']:>12.3f} "
            f"{r['A_minus_I_F_med']:>10.3e} "
            f"{r['residual_med']:>11.2e} "
            f"{r['contraction_ratio_med']:>11.2e} "
            f"{r['contraction_ratio_max']:>11.2e} "
            f"{r['failures']:>6d}"
        )


def smallest_contracting_eps(results, ratio_threshold=0.5):
    """Return the smallest eps where the median contraction ratio < threshold."""
    ok = [r for r in results if r["contraction_ratio_med"] < ratio_threshold
          and r["failures"] == 0]
    if not ok:
        return None
    return min(r["eps_target"] for r in ok)


def _diagnose_failure(eps, rng_seed, out_path):
    """Re-run one failing trial with verbose=True; dump (A, B, residual) to JSON."""
    rng = np.random.default_rng(rng_seed)
    U = random_near_identity(eps, rng)
    print(f"\n--- Diagnosing eps = {eps:.2e} (seed={rng_seed}) ---")
    print(f"  |U - I|_F = {np.linalg.norm(U - I3, ord='fro'):.3e}")
    A, B, residual = factor_commutator(U, verbose=True)
    print(f"  final residual = {residual:.3e}")
    print(f"  ratio          = {residual / eps:.3e}")
    print(f"  |A - I|_F      = {np.linalg.norm(A - I3, ord='fro'):.3e}")
    print(f"  |B - I|_F      = {np.linalg.norm(B - I3, ord='fro'):.3e}")
    # Write the case to disk for offline inspection.
    payload = {
        "eps": float(eps),
        "rng_seed": int(rng_seed),
        "U_real": U.real.tolist(),
        "U_imag": U.imag.tolist(),
        "A_real": A.real.tolist(),
        "A_imag": A.imag.tolist(),
        "B_real": B.real.tolist(),
        "B_imag": B.imag.tolist(),
        "residual": float(residual),
        "ratio": float(residual / eps),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  dumped failing case to {out_path}")


if __name__ == "__main__":
    print("=" * 86)
    print("factor_commutator stress test: sweeping eps from 0.5 down to 1e-9")
    print("=" * 86)

    results = run_full_sweep(n_trials_per_eps=10, rng_seed=0)
    _print_table(results)

    print()
    floor = smallest_contracting_eps(results, ratio_threshold=0.5)
    if floor is None:
        print("WARNING: factor_commutator did not contract (ratio < 0.5) at any tested eps.")
    else:
        print(f"Smallest eps with median contraction ratio < 0.5 (and zero failures): "
              f"{floor:.2e}")

    # Show all eps with failures.
    failing = [r for r in results if r["failures"] > 0]
    if failing:
        print("\nEPS with failures (ratio >= 1 in at least one trial):")
        for r in failing:
            print(f"  eps = {r['eps_target']:.2e}: "
                  f"{r['failures']}/{r['n_trials']} trials failed, "
                  f"max_ratio = {r['contraction_ratio_max']:.2e}")
        # Diagnose the worst case (smallest eps with failures).
        worst = min(failing, key=lambda r: r["eps_target"])
        dump_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"commutator_stress_failure_eps_{worst['eps_target']:.0e}.json",
        )
        _diagnose_failure(worst["eps_target"], rng_seed=999, out_path=dump_path)
    else:
        print("\nNo trial failures (residual < eps_target) anywhere in the sweep.")

    # SK depth implication.
    if floor is not None:
        # SK at base eps_0 -> depth k achieves eps_0^{2^k}.  Find largest k
        # with eps_0^{2^k} >= floor for a reasonable eps_0 = 0.3.
        eps_0 = 0.3
        k_max = 0
        while eps_0 ** (2 ** (k_max + 1)) >= floor:
            k_max += 1
            if k_max > 30:
                break
        print(f"\nSK implication: with base eps_0 = {eps_0}, factor_commutator "
              f"contracts down to eps = {floor:.2e}, supporting SK depth k <= {k_max} "
              f"(residual at depth k = {eps_0 ** (2 ** k_max):.2e}).")
