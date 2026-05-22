"""
Microbench + validation: compare Sage I.factor() vs PARI nf.idealfactor()
on principal_ideal_factorization for a number of (m0,m1,m2) triples.

Run via:
    ~/miniforge3/envs/sage/bin/sage -python zeta9_pari_bench.py

Expected output: per-call ms for both paths and a speedup ratio. Audit
predicts 1.5-3x; if <1.2x, the boundary prid->Sage-ideal conversion is
likely eating the gain and the change should not be shipped without
caching that conversion.

Was created during PARI-direct work 2026-05-12 (see PARI_DIRECT_RESULTS.md).
"""
import sys
import time
import numpy as np

# Make zeta9 importable from any cwd
HERE = "/home/hlamm/Desktop/efficent_gates/unified/zeta9"
sys.path.insert(0, HERE)

from sage.all import (
    CyclotomicField, NumberField, QQ, ZZ, polygen, factor, pari
)


def build_fields():
    x = polygen(QQ, "x")
    F = NumberField(x**3 - 3*x + 1, names=("alpha",))
    alpha = F.gen()
    K = CyclotomicField(9)
    z = K.gen()
    emb = F.hom([z + z**(-1)], K)
    return F, alpha, K, z, emb


F, alpha, K, z, emb = build_fields()
nfF = F.pari_nf()


def factor_sage(M):
    """Old path: I = F.ideal(M); fac = I.factor()."""
    I = F.ideal(M)
    return list(I.factor())


def factor_pari_direct(M):
    """New path: nfF.idealfactor(M_as_pari) -> list of (Sage prime ideal, exp)."""
    M_pari = pari(M)
    fac_mat = nfF.idealfactor(M_pari)
    n = int(fac_mat.matsize()[0])
    out = []
    for i in range(n):
        prime_pari = fac_mat[i, 0]   # PARI prid: [p, a, e, f, b]
        exp = int(fac_mat[i, 1])
        p_rat = ZZ(prime_pari[0])
        gen2_alg = nfF.nfbasistoalg(prime_pari[1])   # t_POLMOD in y
        gen2_F = F(gen2_alg)
        Pideal = F.ideal([p_rat, gen2_F])
        out.append((Pideal, exp))
    return out


def normalize_sage_factors(fac):
    """Canonical (norm, sorted-gen-coeff-tuples, exp) for cheap compare."""
    out = []
    for P, e in fac:
        nrm = ZZ(P.norm())
        gens = tuple(
            tuple(int(c) for c in g.polynomial().list())
            for g in P.gens()
        )
        out.append((int(nrm), gens, int(e)))
    out.sort()
    return out


def factors_equivalent(fac1, fac2):
    """True iff fac1 and fac2 contain the same multiset of (ideal, exp)."""
    if len(fac1) != len(fac2):
        return False
    rest = list(fac2)
    for P1, e1 in fac1:
        n1 = ZZ(P1.norm()); e1 = int(e1)
        match_idx = None
        for j, (P2, e2) in enumerate(rest):
            if int(e2) != e1:
                continue
            if ZZ(P2.norm()) != n1:
                continue
            if P1 == P2:
                match_idx = j
                break
        if match_idx is None:
            return False
        rest.pop(match_idx)
    return len(rest) == 0


# ---------------------------------------------------------------------------
# Sample inputs
# ---------------------------------------------------------------------------
y_path = HERE + "/D/Y1_f=2_u=1_eps=0.025.npy"
Ys = np.load(y_path)
print(f"Loaded {len(Ys)} Y triples from {y_path}")
print(f"shape: {Ys.shape}, dtype: {Ys.dtype}")
print(f"first few: {Ys[:5]}")

sample = []
for row in Ys:
    m0, m1, m2 = int(row[0]), int(row[1]), int(row[2])
    if m0 == 0 and m1 == 0 and m2 == 0:
        continue
    sample.append((m0, m1, m2))
    if len(sample) >= 8:
        break

print(f"\nSample triples: {sample}")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
print("\n=== VALIDATION ===")
all_match = True
for trip in sample:
    m0, m1, m2 = trip
    M = F(m0 + m1 * alpha + m2 * alpha**2)
    f1 = factor_sage(M)
    f2 = factor_pari_direct(M)
    eq = factors_equivalent(f1, f2)
    n1 = normalize_sage_factors(f1)
    n2 = normalize_sage_factors(f2)
    norm_eq = (n1 == n2)
    print(f"  Y={trip}: sage={len(f1)} primes, pari={len(f2)} primes, "
          f"ideal_eq={eq}, normalized_eq={norm_eq}")
    if not eq:
        print(f"    sage:  {n1}")
        print(f"    pari:  {n2}")
        all_match = False
print(f"All match: {all_match}")

# ---------------------------------------------------------------------------
# Microbench
# ---------------------------------------------------------------------------
print("\n=== MICROBENCH ===")
N = 4
# Warmup (also primes any one-shot Sage/PARI init costs)
for trip in sample:
    M = F(trip[0] + trip[1]*alpha + trip[2]*alpha**2)
    factor_sage(M)
    factor_pari_direct(M)

t0 = time.perf_counter()
for _ in range(N):
    for trip in sample:
        M = F(trip[0] + trip[1]*alpha + trip[2]*alpha**2)
        factor_sage(M)
t_sage = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N):
    for trip in sample:
        M = F(trip[0] + trip[1]*alpha + trip[2]*alpha**2)
        factor_pari_direct(M)
t_pari = time.perf_counter() - t0

per_call_sage = t_sage / (N * len(sample)) * 1000
per_call_pari = t_pari / (N * len(sample)) * 1000
ratio = t_sage / t_pari if t_pari > 0 else float("inf")
print(f"  sage:           {t_sage*1000:.1f} ms total, {per_call_sage:.2f} ms/call")
print(f"  pari_direct:    {t_pari*1000:.1f} ms total, {per_call_pari:.2f} ms/call")
print(f"  speedup ratio:  {ratio:.2f}x")
