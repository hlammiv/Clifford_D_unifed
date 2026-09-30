"""raw_lb_1anc.py -- dedup-free lower-bound check (1 ancilla): enumerate ALL rotation products of
length <= 3 (80^3 = 512000 at length 3, no class merging) from E0 and from R E0, hash with the
left-Clifford invariant (a necessary condition for Clifford equivalence) and intersect.
Empty intersection for all (a,b) with a+b <= 6  =>  no Clifford+T circuit with T-count <= 6 exists
(1 clean ancilla, any final 2-qutrit Clifford correction)."""
import sys, time, numpy as np
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/r_from_d')
from mitm_core import *
t0 = time.time()
def raw_levels(M0):
    L = [M0[None]]; hs = [H(M0[None])[0]]
    for d in (1, 2):
        prev = L[-1]; new = np.concatenate([apply_rot(prev, r) for r in range(80)])
        L.append(new); hs.append(H(new)[0])
    h3, err = CH(L[2]); assert err.max() < 1e-3
    hs.append(h3.ravel())
    return [np.unique(h) for h in hs]
tgt = {'R': [1, 1, -1]}
A = raw_levels(embed_all(e0)); print('E0 raw level class-hash counts', [len(h) for h in A], f'{time.time()-t0:.0f}s', flush=True)
B = raw_levels(embed_all(lambda a: np.kron(np.diag([1, 1, -1]), np.eye(3)) @ e0()))
print('RE0 raw level counts', [len(h) for h in B], f'{time.time()-t0:.0f}s')
for a in range(4):
    for b in range(4):
        n = np.intersect1d(A[a], B[b]).size
        print(f'  a={a} b={b} (t={a+b}): {n} meets')
