"""extrapolate.py -- best-of-K T-cost slope under explicit assumptions.

Model (all per log3(1/eps) = L):
  mean T  : mu(L) = a + 10.0 L  (Nick's fit; HRSA's own staircase is ~12/L)
  sd      : s(L)  = 12.5 + 0.68 L (cross-angle model; this study finds within-angle sd ~= model, +0..10%)
  tail    : Gaussian, E[min of K] = mu - e(K) s   (verified: empirical gain/sd = 0.84, 1.50, 2.00, 2.44,
            2.76, 3.11 at K = 3..1000 vs Gaussian 0.85, 1.54, 2.04, 2.51, 2.88, 3.24)
Scenarios
  A  unlimited pool: K candidates always available  ->  slope = 10 - 0.68 e(K)
  B  pool-limited staircase: denominator levels spaced dL (=2.43 measured, HRSA f=2->3, pool_size.py) in L;
     at the lowest feasible level the pool is N = Nmax^(1-x), x in [0,1) the position within the
     level (Nmax ~ 1.5e4 measured, frob criterion, k3=1).  The compiler may instead go up one level
     (pool effectively unlimited) paying +10 dL in mean.  Cost = min of the two options.
Slope is fitted over L in [5, 25] (eps 4e-3 .. 1e-12) after averaging the periodic staircase.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

xs = np.linspace(-9, 9, 4001)


def e_of(N):
    """E[max of N iid std normals], continuous in N >= 1."""
    if N <= 1:
        return 0.0
    pdf = N * norm.pdf(xs) * norm.cdf(xs) ** (N - 1)
    return float(np.trapezoid(xs * pdf, xs))


def s(L):
    return 12.5 + 0.68 * L


def slope(K, scenario, dL=2.43, Nmax=1.5e4, grid=4000):
    Ls = np.linspace(5, 25, grid)
    eK = e_of(K)
    T = []
    for L in Ls:
        mu = 10.0 * L
        if scenario == "A":
            T.append(mu - eK * s(L))
            continue
        x = (L / dL) % 1.0
        N = Nmax ** (1 - x)
        stay = mu - e_of(min(K, N)) * s(L)
        jump = mu + 10.0 * dL - eK * s(L)
        T.append(min(stay, jump))
    T = np.array(T)
    b = np.polyfit(Ls, T, 1)[0]
    return b, float(np.mean(10.0 * Ls - T))


if __name__ == "__main__":
    print(f"{'K':>5} {'e(K)':>6} {'A slope':>8} | {'B slope (Nmax=1.5e4)':>18} {'B (Nmax=1e3)':>13} {'B (Nmax=1e5)':>13} | mean gain A/B[T]")
    for K in [1, 3, 10, 30, 100, 1000]:
        a, ga = slope(K, "A", grid=400)
        b, gb = slope(K, "B")
        b3, _ = slope(K, "B", Nmax=1e3)
        b5, _ = slope(K, "B", Nmax=1e5)
        print(f"{K:>5} {e_of(K):6.3f} {a:8.2f} | {b:18.2f} {b3:13.2f} {b5:13.2f} | {ga:6.1f}/{gb:6.1f}")
    # sensitivity to the sd slope (0.68 model vs 1.16 measured over HRSA f=2->3 only)
    for ds in (0.5, 0.68, 0.78, 1.16):
        print(f"sd slope {ds}: K=10 -> {10 - ds * e_of(10):.2f}, K=100 -> {10 - ds * e_of(100):.2f}")
