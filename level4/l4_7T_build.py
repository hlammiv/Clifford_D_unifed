"""Build an explicit gate list for the 7-T solution found by l4_orbits.py 4 4:
   fwd seq (2,11,20,0) from E0  ~  bwd seq (2,11,20) from E0 diag(1,1,z)
=> W = R2 R11 R20 K R0 R20 R11 R2  with  W E0 = ph E0 L G.   Writes l4_7T_gates.json."""
import numpy as np, json
from l4core import *
from l4_orbits import equiv, signatures

S1 = np.diag([1, 1, w]); SUMm = np.zeros((9, 9))
for x in range(3):
    for y in range(3): SUMm[3*x+(y+x) % 3, 3*x+y] = 1
GEN = {"H0": np.kron(H1, I3), "H1": np.kron(I3, H1), "S0": np.kron(S1, I3), "S1": np.kron(I3, S1),
       "SUM01": SUMm, "X0": np.kron(X1, I3), "X1": np.kron(I3, X1), "Z0": np.kron(Z1, I3), "Z1": np.kron(I3, Z1)}
def ekey(U):
    v = U.flatten(); i = np.argmax(abs(v) > 1e-9); v = v/(v[i]/abs(v[i])); return tuple(np.round(v, 6))
# BFS over the full 2-qutrit Clifford group mod phase (4.2M) is big; instead BFS over symplectic
# classes (key by Pauli images) and fix the Pauli part afterwards with X/Z words.
from l4_verify import key, GENS_P
def cliff_words():
    reps = {key(np.eye(9), GENS_P): (np.eye(9, dtype=complex), [])}
    fr = list(reps.keys())
    while fr:
        nf = []
        for k in fr:
            U, wd = reps[k]
            for g in ("H0", "H1", "S0", "S1", "SUM01"):
                V = GEN[g] @ U; kk = key(V, GENS_P)
                if kk not in reps: reps[kk] = (V, wd+[g]); nf.append(kk)
        fr = nf
    return reps
REPS = cliff_words()
def word_for(C):
    """time-ordered Clifford word equal to C up to global phase."""
    U, wd = REPS[key(C, GENS_P)]
    for a1, b1, a2, b2 in itertools.product(range(3), repeat=4):
        Pw = ["X0"]*a1 + ["Z0"]*b1 + ["X1"]*a2 + ["Z1"]*b2      # applied after U
        M = U
        for g in Pw: M = GEN[g] @ M
        t = np.vdot(M, C)/9
        if abs(abs(t)-1) < 1e-9: return wd + Pw
    raise ValueError
def mat(word):
    M = np.eye(9, dtype=complex)
    for g in word: M = GEN[g] @ M
    return M
def rot_word(ri, dag=False):
    """word for R(P) (or R(P)^dag) = C T0 C^dag with C Z0 C^dag = P exactly."""
    P = P2[ROT_IDX[ri]]
    from l4_verify import clifford_reps
    for (U, wd) in REPS.values():
        Zi = U @ GEN["Z0"] @ U.conj().T
        t = np.vdot(P, Zi)/9
        if abs(abs(t)-1) < 1e-9: break
    # fix phase with a Pauli Q: Q Zi Q^dag = P
    for q in range(81):
        C = P2[q] @ U
        if np.allclose(C @ GEN["Z0"] @ C.conj().T, P):
            wc = word_for(C); wcd = word_for(C.conj().T)
            return wcd + (["Tdg0"] if dag else ["T0"]) + wc
    raise ValueError

if __name__ == "__main__":
    import itertools
    L = np.diag([1, 1, z])
    Jf = E0.copy()
    for j in (2, 11, 20, 0): Jf = ROTS[j] @ Jf
    Jb = E0 @ L
    for j in (2, 11, 20): Jb = ROTS_DAG[j] @ Jb
    vf, cf, _ = signatures(Jf[None]); vb, cb, _ = signatures(Jb[None])
    K, G, ph = equiv(Jf, vf[0], cf[0], Jb, vb[0], cb[0])
    print("K Jf = ph Jb G :", np.allclose(K @ Jf, ph*Jb @ G))
    word = []
    # prepend G^{-1} on system (qutrit 0)
    Ginv = np.kron(G.conj().T, I3); word += word_for(Ginv)
    for j in (2, 11, 20, 0): word += rot_word(j)
    word += word_for(K)
    for j in (20, 11, 2): word += rot_word(j)
    GEN["T0"] = np.kron(T1, I3); GEN["Tdg0"] = np.kron(T1.conj().T, I3)
    U = mat(word)
    M = E0.conj().T @ U @ E0
    print("ancilla returned to |0>:", np.allclose(U @ E0, E0 @ M), " logical op:\n", np.round(M/M[0, 0], 4))
    print("T-count:", sum(g in ("T0", "Tdg0") for g in word), " total gates:", len(word))
    json.dump(word, open("l4_7T_gates.json", "w"))
