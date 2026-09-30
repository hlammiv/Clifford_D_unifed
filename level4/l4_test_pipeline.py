"""Sanity test: decompose the known 8-T circuit into T-rotations and check the MITM invariant
matches at every split s (forward s rotations from E0, backward 8-s rotations from target)."""
import numpy as np
from l4core import *
C = lambda A: np.kron(A, I3); Tg = lambda A: np.kron(I3, A)
def SUM():
    M = np.zeros((9, 9))
    for x in range(3):
        for y in range(3): M[3*x+(y+x) % 3, 3*x+y] = 1
    return M
def perm(p):
    M = np.zeros((3, 3)); [M.__setitem__((p[j], j), 1) for j in range(3)]; return M
t01, t02 = perm([1, 0, 2]), perm([2, 1, 0])
Xd = X1.conj().T
def g(M): return ('C', M)
def tg(dag): return ('T', 1, dag)      # T on target qutrit (index 1)
P9 = [g(Tg(Xd)), tg(False), g(Tg(X1))]
c2x = [g(Tg(H1)), g(C(X1)), g(Tg(X1))] + (P9+[g(SUM())])*3 + [g(C(Xd)), g(Tg(Xd)), g(Tg(H1.conj().T))]
def dag(lst):
    out = []
    for it in reversed(lst):
        out.append(('C', it[1].conj().T) if it[0] == 'C' else ('T', it[1], not it[2]))
    return out
blk = c2x + [g(Tg(t01)), tg(True), g(Tg(t01))] + dag(c2x) + [g(Tg(t01)), tg(False), g(Tg(t01))]
circ = [g(Tg(t02))] + blk + [g(Tg(t02))]
U = np.eye(9, dtype=complex); K = np.eye(9, dtype=complex); rots = []
for it in circ:
    if it[0] == 'C':
        U = it[1] @ U; K = it[1] @ K
    else:
        Tm = Tg(T1.conj().T if it[2] else T1)
        U = Tm @ U
        rots.append(K.conj().T @ Tm @ K)
print("T-count", len(rots))
print("U E0 = E0 diag(1,1,z^7):", np.allclose(U @ E0, E0 @ np.diag([1, 1, z**7])))
# check each rotation is f(P) for some Pauli up to Clifford-phase: invariants
t = len(rots)
Jt = E0 @ np.diag([1, 1, z**7])
for s in range(t+1):
    Jf = E0.copy()
    for R in rots[:s]: Jf = R @ Jf
    Jb = K.conj().T @ U @ E0
    for R in reversed(rots[s:]): pass
    Jb = Jt.copy()
    Jb = K.conj().T @ Jb   # clifford, irrelevant
    for R in reversed(rots[s:]): Jb = R.conj().T @ Jb
    hf = inv_hash_int(Jf[None]); hb = inv_hash_int(Jb[None])
    print(s, hf == hb, np.allclose(Jf, Jb))
# identify rotation Paulis
for R in rots:
    best = [i for i in range(80) if any(np.allclose(R, ph*ROTS[i]@D) for ph in [1] for D in [np.eye(9)])]
    print(best, end=' ')
print()
