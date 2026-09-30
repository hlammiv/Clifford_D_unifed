"""Reconstruct a MITM hit into an explicit circuit and verify it (sigma_1 embedding, i.e. true numbers)."""
import numpy as np, itertools
from mitm_core import PAULIS, pauli_mat, rot_mat, e0, zeta

P1 = [pauli_mat(s, p, 1) for s, p in PAULIS]
R1 = [rot_mat(s, p, 1) for s, p in PAULIS]
IDX = {}   # (a1,b1,a2,b2) -> index
for k, (a1, b1, a2, b2) in enumerate([t for t in itertools.product(range(3), repeat=4) if t != (0, 0, 0, 0)]):
    IDX[(a1, b1, a2, b2)] = k
def sq_index(r):   # index of P^2 (R_P^{-1} = R_{P^2})
    key = [k for k, v in IDX.items() if v == r][0]
    return IDX[tuple((2*x) % 3 for x in key)]

def prod(seq, M):
    for r in seq: M = R1[r] @ M
    return M

GEN = {'X1': (1, 0, 0, 0), 'Z1': (0, 1, 0, 0), 'X2': (0, 0, 1, 0), 'Z2': (0, 0, 0, 1)}
def find_clifford(M2, M1):
    """find 2-qutrit Clifford C with C M2 = lam M1 (9x3 isometries); returns C or None"""
    w = zeta(3)
    K1 = {g: M1.conj().T @ P1[IDX[v]] @ M1 for g, v in GEN.items()}
    allQ = [(k, s) for k in range(80) for s in range(3)]
    KQ = {(k, s): w**s * (M2.conj().T @ P1[k] @ M2) for (k, s) in allQ}
    cands = {g: [q for q in allQ if np.allclose(KQ[q], K1[g], atol=1e-9)] for g in GEN}
    rng = np.random.default_rng(1)
    for choice in itertools.product(*[cands[g] for g in GEN]):
        phi = {g: w**s * P1[k] for g, (k, s) in zip(GEN, choice)}
        Y = rng.normal(size=(9, 9)) + 1j*rng.normal(size=(9, 9)); Cp = np.zeros((9, 9), complex)
        for a1, b1, a2, b2 in itertools.product(range(3), repeat=4):
            P = np.linalg.matrix_power(P1[IDX[(1,0,0,0)]], a1) @ np.linalg.matrix_power(P1[IDX[(0,1,0,0)]], b1) @ \
                np.linalg.matrix_power(P1[IDX[(0,0,1,0)]], a2) @ np.linalg.matrix_power(P1[IDX[(0,0,0,1)]], b2)
            F = np.linalg.matrix_power(phi['X1'], a1) @ np.linalg.matrix_power(phi['Z1'], b1) @ \
                np.linalg.matrix_power(phi['X2'], a2) @ np.linalg.matrix_power(phi['Z2'], b2)
            Cp += F @ Y @ np.linalg.inv(P)
        if np.linalg.norm(Cp) < 1e-6: continue
        # Cp P Cp^-1 = phi(P)  ->  C = Cp^-1 satisfies C^dag P C = phi(P)
        Cp /= abs(np.linalg.det(Cp))**(1/9)
        if not np.allclose(Cp @ Cp.conj().T, np.eye(9), atol=1e-8): continue
        C = Cp.conj().T
        V = C @ M2; lam = np.vdot(M1.ravel(), V.ravel()) / 3
        if abs(abs(lam)-1) < 1e-8 and np.allclose(V, lam*M1, atol=1e-9):
            return C / lam
    return None

def is_clifford(C):
    Cd = C.conj().T
    for v in GEN.values():
        Q = C @ P1[IDX[v]] @ Cd
        if not any(np.allclose(Q, zeta(3)**s * P1[k], atol=1e-9) for k in range(80) for s in range(3)): return False
    return True
