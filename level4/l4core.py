"""l4core.py -- shared machinery for the level-4 T-count search (2 qutrits: system a, ancilla b).

A Clifford+T circuit with t T-gates on (a,b) can be written U = K * R(P_t)...R(P_1), K Clifford,
R(P) = f(P) the 'T-rotation' about a 2-qutrit Pauli P (eigenvalues 1,w,w^2 -> 1,z,z^-1).
Condition: U E0 = e^{i phi} E0 L  (E0: psi -> psi (x) |0>),  L level-4 diagonal (x Clifford).
Modulo left Clifford K and right logical Clifford G, this is an orbit problem on 9x3 isometries J.
Clifford-invariant used for matching: multiset of |<chi| A (x) P |chi>|^2, chi = Choi state of J
(A: 9 one-qutrit Paulis on reference, P: 81 two-qutrit Paulis).
"""
import numpy as np, itertools, hashlib

z = np.exp(2j*np.pi/9); w = z**3
I3 = np.eye(3)
X1 = np.roll(I3, 1, axis=0)
Z1 = np.diag([1, w, w**2])
H1 = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(-3+0j)
T1 = np.diag([1, z, z**8])
mp = np.linalg.matrix_power

P1 = [mp(X1, a) @ mp(Z1, b) for a in range(3) for b in range(3)]          # 9 one-qutrit Paulis
P2 = [np.kron(mp(X1, a1) @ mp(Z1, b1), mp(X1, a2) @ mp(Z1, b2))
      for a1 in range(3) for b1 in range(3) for a2 in range(3) for b2 in range(3)]  # 81
P2lab = [(a1, b1, a2, b2) for a1 in range(3) for b1 in range(3) for a2 in range(3) for b2 in range(3)]

def fpauli(P):
    """T-rotation f(P): eigenvalue 1->1, w->z, w^2->z^-1. P must have spectrum in {1,w,w^2}."""
    d = P.shape[0]
    fv = {0: 1, 1: z, 2: z**8}
    out = np.zeros((d, d), complex)
    for m in range(3):
        Pi = sum(w**(-m*k) * mp(P, k) for k in range(3))/3
        out += fv[m]*Pi
    return out

def normalize_pauli(P):
    """Multiply by a power of w so that the spectrum is {1,w,w^2} exactly (check)."""
    ev = np.linalg.eigvals(P)
    ok = all(min(abs(e-w**k) for k in range(3)) < 1e-9 for e in ev)
    assert ok, ev
    return P

ROT_IDX = list(range(1, 81))
ROTS = np.array([fpauli(P2[i]) for i in ROT_IDX])        # 80 x 9 x 9
ROTS_DAG = np.conj(np.transpose(ROTS, (0, 2, 1)))

def symp(l1, l2):
    a1, b1, a2, b2 = l1; c1, d1, c2, d2 = l2
    return (a1*d1 - b1*c1 + a2*d2 - b2*c2) % 3

def line_id(i):
    """Index of the line <P> (P and P^2 share it)."""
    a = P2lab[i]; b = tuple((2*x) % 3 for x in a)
    return min(i, P2lab.index(b))

LINE = np.array([line_id(i) for i in ROT_IDX])
COMM = np.array([[symp(P2lab[i], P2lab[j]) == 0 for j in ROT_IDX] for i in ROT_IDX])

def allowed_next(prev):
    """Rotation indices (0..79) allowed after rotation prev (0..79, or -1): drop same line
    (R(P)R(P)=R(P^2)C, R(P)R(P^2)=1) and enforce ordering inside commuting pairs."""
    if prev < 0: return np.arange(80)
    ok = []
    for j in range(80):
        if LINE[j] == LINE[prev]: continue
        if COMM[prev, j] and j < prev: continue
        ok.append(j)
    return np.array(ok)

ALLOWED = [allowed_next(p) for p in range(80)]

E0 = np.zeros((9, 3), complex)
for x in range(3): E0[3*x+0, x] = 1

P2arr = np.array(P2)          # 81x9x9
P1arr = np.array(P1)          # 9x3x3
P1conj = np.conj(P1arr)

def invariant_values(J):
    """J: (N,9,3). Returns (N,729) array of |Tr(A^dag J^dag P J)|^2/9."""
    PJ = np.einsum('qij,njk->nqik', P2arr, J, optimize=True)            # N,81,9,3
    M = np.einsum('nia,nqib->nqab', np.conj(J), PJ, optimize=True)      # N,81,3,3
    c = np.einsum('Aab,nqab->nqA', P1conj, M, optimize=True)/3.0         # Tr(A^dag M)/3
    v = (c.real**2 + c.imag**2).reshape(len(J), -1)
    return v

def inv_hash(J, q=1e5):
    v = invariant_values(J)
    v = np.sort(np.rint(v*q).astype(np.int64), axis=1)
    return [hashlib.blake2b(row.tobytes(), digest_size=8).digest() for row in v]

def inv_hash_int(J, q=1e5):
    v = invariant_values(J)
    v = np.sort(np.rint(v*q).astype(np.int64), axis=1)
    # 64-bit polynomial hash, vectorized
    h = np.zeros(len(J), dtype=np.uint64)
    mult = np.uint64(1099511628211)
    with np.errstate(over='ignore'):
        for k in range(v.shape[1]):
            h = (h ^ v[:, k].astype(np.uint64)) * mult
    return h
