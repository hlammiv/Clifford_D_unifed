"""
Offline builder for the inert-deg-1 prime generator table used by roots_fast.

For each rational prime p ≡ 8 mod 9 up to BOUND:
  - factor the principal ideal (p) in O_F as P_1 · P_2 · P_3
  - find a principal generator π_i = a_i + b_i α + c_i α² for each P_i (norm = p)
  - record (p, a_i, b_i, c_i) for i in {0,1,2}

The three P_i above p are K/F-inert (residue field of K above each P_i has degree 2),
so for screening we need v_{P_i}(M) for each.

Output: /home/hlamm/.../zeta9/zeta9/_deg1_inert_table.npy
  shape (N, 4) int64, columns: (p, a, b, c)
  sorted by p, contiguous within each p (3 rows per p).
"""
import sys, os, time
import numpy as np
from sage.all import (
    NumberField, polygen, QQ, ZZ, Primes, Mod,
)

BOUND = int(os.environ.get("DEG1_TABLE_BOUND", 1000000))
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "zeta9", "_deg1_inert_table.npy",
)


def build():
    x = polygen(QQ, "x")
    F = NumberField(x**3 - 3*x + 1, names=("alpha",))
    alpha = F.gen()
    OF = F.ring_of_integers()

    rows = []
    t0 = time.time()
    last_log = t0
    n_seen = 0
    p = 2
    while p <= BOUND:
        n_seen += 1
        if p == 3:
            p = int(ZZ(p).next_prime())
            continue
        if p % 9 != 8:
            p = int(ZZ(p).next_prime())
            continue
        # Inert-deg-1 prime: factor (p)
        I = F.ideal(p)
        fac = I.factor()
        # Expected: 3 primes, each with exponent 1
        if len(fac) != 3 or any(e != 1 for _, e in fac):
            raise RuntimeError(f"unexpected factorization of ({p}) in F: {fac}")
        for P_ideal, e in fac:
            # Each P_ideal should be principal (F has class number 1)
            if not P_ideal.is_principal():
                raise RuntimeError(f"non-principal prime above {p}: {P_ideal}")
            pi = P_ideal.gens_reduced()[0]  # the generator (an element of O_F)
            # Get coefficients in basis (1, α, α²)
            coeffs = pi.polynomial().list()
            # Pad to 3 if needed
            while len(coeffs) < 3:
                coeffs.append(0)
            a, b, c = int(coeffs[0]), int(coeffs[1]), int(coeffs[2])
            # Sanity check: norm should equal p
            if abs(pi.norm()) != p:
                raise RuntimeError(f"bad generator for prime above {p}: pi={pi} norm={pi.norm()}")
            rows.append((p, a, b, c))
        if time.time() - last_log > 5.0:
            last_log = time.time()
            print(f"  p={p} ({n_seen} primes seen, {len(rows)//3} inert-deg-1), elapsed={time.time()-t0:.0f}s",
                  flush=True)
        p = int(ZZ(p).next_prime())

    out = np.array(rows, dtype=np.int64)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, out)
    print(f"wrote {OUT_PATH}: shape={out.shape} (covers all inert-deg-1 primes p≡8 mod 9, p ≤ {BOUND})", flush=True)
    print(f"elapsed: {time.time()-t0:.0f}s")
    # Quick stats
    primes_in_table = sorted(set(int(r[0]) for r in rows))
    print(f"  {len(primes_in_table)} primes (each yields 3 generators)")
    print(f"  smallest: {primes_in_table[0]}, largest: {primes_in_table[-1]}")


if __name__ == "__main__":
    build()
