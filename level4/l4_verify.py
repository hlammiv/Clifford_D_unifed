"""l4_verify.py -- exact Clifford-equivalence test for MITM invariant hits.
Given fwd seq F and bwd seq B (rotation indices), find 2-qutrit Clifford K, 1-qutrit Clifford G, phase:
   K J_f = e^{i phi} J_b G .  If found, U = R_B... K R_F... implements L G with t = |F|+|B| T gates."""
import numpy as np, itertools, sys, pickle, os
from l4core import *

def key(U, paulis):
    """symplectic image signature: which Pauli (index mod phase) each generator maps to."""
    out = []
    for P in paulis:
        Q = U @ P @ U.conj().T
        for i, R in enumerate(P2):
            t = np.trace(R.conj().T @ Q)/9
            if abs(abs(t)-1) < 1e-9: out.append(i); break
        else: raise ValueError
    return tuple(out)

GENS_P = [P2[P2lab.index(l)] for l in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]]
def clifford_reps():
    fn = "/home/hlamm/Desktop/efficent_gates/unified/level4/cliff2_reps.npy"
    if os.path.exists(fn): return np.load(fn)
    S1 = np.diag([1, 1, w]); SUMm = np.zeros((9, 9))
    for x in range(3):
        for y in range(3): SUMm[3*x+(y+x) % 3, 3*x+y] = 1
    gens = [np.kron(H1, I3), np.kron(I3, H1), np.kron(S1, I3), np.kron(I3, S1), SUMm]
    reps = {key(np.eye(9), GENS_P): np.eye(9, dtype=complex)}
    frontier = list(reps.items())
    while frontier:
        nf = []
        for k, U in frontier:
            for g in gens:
                V = g @ U; kk = key(V, GENS_P)
                if kk not in reps: reps[kk] = V; nf.append((kk, V))
        frontier = nf
        print("cliff reps", len(reps), flush=True)
    arr = np.array(list(reps.values())); np.save(fn, arr); return arr

def cliff1():
    S1 = np.diag([1, 1, w]); out = {}
    fr = [np.eye(3, dtype=complex)]
    def k1(U):
        v = U.flatten(); i = np.argmax(abs(v) > 1e-9); v = v/(v[i]/abs(v[i])); return tuple(np.round(v, 8))
    out[k1(fr[0])] = fr[0]
    while fr:
        nf = []
        for U in fr:
            for g in (H1, S1, X1):
                V = g @ U; kk = k1(V)
                if kk not in out: out[kk] = V; nf.append(V)
        fr = nf
    return np.array(list(out.values()))

def build(F, B, L):
    Jf = E0.copy()
    for j in F: Jf = ROTS[j] @ Jf
    Jb = E0 @ L
    for j in B: Jb = ROTS_DAG[j] @ Jb
    return Jf, Jb

def equivalent(Jf, Jb, KR, G1):
    """returns (K, G, phase) or None."""
    A = np.einsum('sij,jk->sik', KR, Jf).reshape(len(KR), -1)          # S x 27
    Bs = []; meta = []
    for gi, G in enumerate(G1):
        JbG = Jb @ G
        for qi, Q in enumerate(P2):
            Bs.append((Q.conj().T @ JbG).reshape(-1)); meta.append((gi, qi))
    Bs = np.array(Bs)
    for c in range(0, len(A), 4000):
        ov = np.abs(A[c:c+4000].conj() @ Bs.T)
        idx = np.argwhere(ov > 3-1e-8)
        if len(idx):
            s, b = idx[0]; gi, qi = meta[b]
            K = P2[qi] @ KR[c+s]
            M = K @ Jf; ph = np.vdot(Jb @ G1[gi], M)/3
            return K, G1[gi], ph
    return None

if __name__ == "__main__":
    KR = clifford_reps(); G1 = cliff1()
    print(len(KR), "2-qutrit symplectic reps;", len(G1), "1-qutrit Cliffords mod phase")
