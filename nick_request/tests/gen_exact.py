"""Brute-force Householder vectors c in Z[zeta9]^3 with sum c_i conj(c_i) == 2*3^f exactly."""
import itertools, sys, pickle
sys.path.insert(0, "/home/hlamm/Desktop/efficent_gates/zeta9")
from zeta9.tools import mul as _m, conj as _z9_conj
_z9_mul = lambda a, b: list(_m(tuple(a), tuple(b)))
R = range(-2, 3)
by_norm = {}
for c in itertools.product(R, repeat=6):
    n = _z9_mul(list(c), _z9_conj(list(c)))
    if all(v == 0 for v in n[1:]) and n[0] > 0:
        by_norm.setdefault(n[0], []).append(c)
print({k: len(v) for k, v in sorted(by_norm.items())})
pickle.dump(by_norm, open("by_norm.pkl", "wb"))
