"""meascore.py -- shared machinery for the MEASUREMENT-model T-count search (system + 1 clean ancilla).

Model.  A circuit is W = K f(P_t)...f(P_1) (K Clifford, f(P) = Clifford-conjugated T rotation, t = T-count),
applied to psi (x) |0>, followed by a measurement of the ancilla in a stabilizer basis, i.e. of a
2-qutrit Pauli Q (K and the stabilizer basis change are absorbed), then an outcome-dependent
single-qutrit Clifford on the system.  With V = f(P_t)...f(P_1) E0 (9x3) and Pi_m(Q) the eigenprojectors,
the gadget implements target G deterministically iff for one Q and some fixed Clifford C' (the
"right" Clifford, i.e. the circuit implements G C'):

        Pi_m(Q) V  =  c_m  J_m  G C'     for every m,   J_m a Clifford isometry into eigenspace m

(c_m may be 0).  Mapping Q -> Z_anc with a Clifford K_Q, block m is B_m = (I (x) <m|) K_Q V (3x3), and the
condition is  B_m / |c_m|  in  Cliff1 . G . C'.  This is invariant under V -> e^{i phi} K V G' (K 2-qutrit
Clifford, G' 1-qutrit Clifford), so it suffices to test one representative per orbit (as in l4_orbits).
Necessary prefilter: V^dag Q V is a multiple of I.
RUS bookkeeping: a block in Cliff1 . C' (i.e. the identity up to Clifford) is a clean failure: undo the
Clifford and repeat.  p_succ = sum of |c_m|^2 over blocks in Cliff1.G.C'.
"""
import sys, numpy as np, numba as nb
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/level4')
from l4core import z, w, P1, P2, P2lab, ROTS, E0, ALLOWED, LINE
from l4_verify import clifford_reps, cliff1

KR = clifford_reps()
PERM = np.load('/home/hlamm/Desktop/efficent_gates/unified/level4/cliff2_perm.npy')
G1 = cliff1()                                   # 216 single-qutrit Cliffords (mod phase)
assert len(G1) == 216
P1A = np.array(P1)                              # 9x3x3
P2A = np.array(P2)                              # 81x9x9
X1 = np.roll(np.eye(3), 1, axis=0); Z1 = np.diag([1, w, w*w])

# one Pauli per line <Q> (Q^2 = Q^dag gives the same measurement)
QLINES = sorted(set(int(LINE[i]) for i in range(80)))       # indices into P2 (1..80)
assert len(QLINES) == 40
ZANC = P2lab.index((0, 0, 0, 1))                            # I (x) Z on the ancilla (minor index)
KQ = np.zeros((40, 9, 9), complex)
for n, q in enumerate(QLINES):
    k = int(np.nonzero(PERM[:, ZANC] == q)[0][0])           # K^dag Z_anc K = Q (up to phase)
    K = KR[k]
    M = K @ P2[q] @ K.conj().T
    t = np.trace(P2[ZANC].conj().T @ M)/9
    assert abs(abs(t)-1) < 1e-9
    KQ[n] = K
QSRC = np.zeros((40, 9), np.int64); QPH = np.zeros((40, 9), complex)
for n, q in enumerate(QLINES):
    for y in range(9):
        x = int(np.argmax(np.abs(P2[q][y]) > 0.5)); QSRC[n, y] = x; QPH[n, y] = P2[q][y, x]


@nb.njit(cache=True)
def is_cliff1(Y, P1A, tol=1e-7):
    """Y 3x3 unitary (any phase): True iff Y X Y^dag and Y Z Y^dag are Paulis up to phase."""
    X = np.zeros((3, 3), np.complex128); Zm = np.zeros((3, 3), np.complex128)
    w_ = np.exp(2j*np.pi/3)
    for j in range(3):
        X[(j+1) % 3, j] = 1.0; Zm[j, j] = w_**j
    for g in range(2):
        A = X if g == 0 else Zm
        M = Y @ A @ np.conj(Y.T)
        ok = False
        for k in range(9):
            t = 0j
            for i in range(3):
                for j in range(3):
                    t += np.conj(P1A[k, i, j])*M[i, j]
            if abs(abs(t)/3 - 1) < tol: ok = True; break
        if not ok: return False
    return True


@nb.njit(cache=True)
def transfer_sig(Y, P1A):
    """sorted 81 values |Tr(P_i Y P_j Y^dag)|^2/9 : invariant of the double coset Cliff1.Y.Cliff1"""
    out = np.zeros(81)
    for j in range(9):
        M = Y @ P1A[j] @ np.conj(Y.T)
        for i in range(9):
            t = 0j
            for a in range(3):
                for b in range(3):
                    t += np.conj(P1A[i, a, b])*M[a, b]
            out[9*i+j] = (t.real**2 + t.imag**2)/9
    return np.sort(out)


@nb.njit(cache=True)
def test_one(V, QSRC, QPH, KQ, P1A, sigI, sigG, out_q, tol=1e-7):
    """Prefilter for one 9x3 isometry V.  For every Q line with V^dag Q V ~ I and all non-zero blocks of
    double-coset type I or G, with at least one of type G, write q into out_q.  Returns #written."""
    nw = 0
    for n in range(40):
        # scalar test M = V^dag Q V
        M = np.zeros((3, 3), np.complex128)
        bad = False
        for i in range(3):
            for j in range(3):
                s = 0j
                for y in range(9):
                    s += np.conj(V[y, i])*QPH[n, y]*V[QSRC[n, y], j]
                M[i, j] = s
                if i != j and abs(s) > tol: bad = True; break
            if bad: break
        if bad: continue
        if abs(M[0, 0]-M[1, 1]) > tol or abs(M[0, 0]-M[2, 2]) > tol: continue
        W = KQ[n] @ V
        nG = 0; ok = True
        for m in range(3):
            B = np.zeros((3, 3), np.complex128)
            p = 0.0
            for i in range(3):
                for j in range(3):
                    B[i, j] = W[3*i+m, j]; p += abs(B[i, j])**2
            p /= 3
            if p < 1e-10: continue
            B /= np.sqrt(p)
            s = transfer_sig(B, P1A)
            if np.max(np.abs(s-sigG)) < 1e-6: nG += 1
            elif np.max(np.abs(s-sigI)) < 1e-6: pass
            else: ok = False; break
        if ok and nG > 0:
            out_q[nw] = n; nw += 1
    return nw


@nb.njit(cache=True)
def scan_batch(Vs, prevs, children, ROTS, ALLOWED_M, NALLOW, QSRC, QPH, KQ, P1A, sigI, sigG):
    """Test every V in Vs and (if children) every allowed child f(P_r) V.
    Returns hits array rows (index, r or -1, q)."""
    N = Vs.shape[0]
    hits = np.zeros((200000, 3), np.int64); nh = 0
    out_q = np.zeros(40, np.int64)
    for i in range(N):
        k = test_one(Vs[i], QSRC, QPH, KQ, P1A, sigI, sigG, out_q)
        for a in range(k):
            if nh < hits.shape[0]:
                hits[nh, 0] = i; hits[nh, 1] = -1; hits[nh, 2] = out_q[a]; nh += 1
        if children:
            pv = prevs[i]
            nr = 80 if pv < 0 else NALLOW[pv]
            for jj in range(nr):
                r = jj if pv < 0 else ALLOWED_M[pv, jj]
                C = ROTS[r] @ Vs[i]
                k = test_one(C, QSRC, QPH, KQ, P1A, sigI, sigG, out_q)
                for a in range(k):
                    if nh < hits.shape[0]:
                        hits[nh, 0] = i; hits[nh, 1] = r; hits[nh, 2] = out_q[a]; nh += 1
    return hits[:nh]


NALLOW = np.array([len(a) for a in ALLOWED], np.int64)
ALLOWED_M = np.full((80, 80), -1, np.int64)
for p, a in enumerate(ALLOWED): ALLOWED_M[p, :len(a)] = a


def sig(G): return transfer_sig(np.asarray(G, complex), P1A)


def blocks(V, n):
    W = KQ[n] @ V
    out = []
    for m in range(3):
        B = np.array([[W[3*i+m, j] for j in range(3)] for i in range(3)])
        p = float(np.sum(np.abs(B)**2)/3)
        out.append((p, B/np.sqrt(p) if p > 1e-10 else None))
    return out


def full_check(V, n, G):
    """exact-up-to-float full test with the 216 right Cliffords.  Returns (best p_succ, C' index, labels)."""
    bl = blocks(V, n); best = (0.0, None, None)
    Gd = np.conj(np.asarray(G, complex).T)
    for ci, Cp in enumerate(G1):
        Cpd = np.conj(Cp.T); ps = 0.0; lab = []; ok = True
        for p, B in bl:
            if B is None: lab.append('0'); continue
            Y = B @ Cpd
            if is_cliff1(Y @ Gd, P1A): ps += p; lab.append('G')
            elif is_cliff1(Y, P1A): lab.append('I')
            else: ok = False; break
        if ok and ps > best[0] + 1e-9: best = (ps, ci, lab)
    return best
