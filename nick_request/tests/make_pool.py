"""Synthetic exact Householder vectors at level f: c in Z[zeta9]^3,
sum_i c_i conj(c_i) == 2*3^f exactly (=> V = X01(I - conj(c)c^T/3^f) exactly unitary)."""
import itertools, pickle, sys, random
sys.path.insert(0, "/home/hlamm/Desktop/efficent_gates/zeta9")
from zeta9.tools import mul as _m, conj as _z9_conj
_z9_mul = lambda a, b: list(_m(tuple(a), tuple(b)))

def norm(c):
    n = _z9_mul(list(c), _z9_conj(list(c)))
    return n[0] if all(v == 0 for v in n[1:]) else None

def pool(f, nmax=400, seed=1):
    base = pickle.load(open("by_norm.pkl", "rb"))
    elems = {}
    for n, L in base.items():
        for c in L:
            elems.setdefault(n, set()).add(tuple(c))
    # close under pairwise products and scaling by 3 up to norm 2*3^f
    T = 2 * 3 ** f
    items = [(n, c) for n, L in elems.items() for c in L]
    for _ in range(2):
        new = []
        for (n1, a), (n2, b) in itertools.product(items, items):
            if n1 * n2 <= T:
                new.append((n1 * n2, tuple(_z9_mul(list(a), list(b)))))
        for n, c in items:
            if 9 * n <= T:
                new.append((9 * n, tuple(3 * v for v in c)))
        items = list({(n, c) for n, c in items + new})
    items.append((0, (0,) * 6))
    byn = {}
    for n, c in items:
        byn.setdefault(n, []).append(c)
    rng = random.Random(seed)
    out = set()
    norms = sorted(byn)
    tries = 0
    while len(out) < nmax and tries < 200000:
        tries += 1
        n0 = rng.choice(norms); n1 = rng.choice(norms)
        n2 = T - n0 - n1
        if n2 not in byn:
            continue
        c = (rng.choice(byn[n0]), rng.choice(byn[n1]), rng.choice(byn[n2]))
        out.add(c)
    return sorted(out)

if __name__ == "__main__":
    for f in (2, 4):
        P = pool(f)
        assert all(sum(norm(ci) for ci in c) == 2 * 3 ** f for c in P)
        print(f, len(P))
        pickle.dump(P, open(f"pool_f{f}.pkl", "wb"))
