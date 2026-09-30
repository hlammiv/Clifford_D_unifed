"""l4_orbits.py -- EXACT orbit BFS for the level-4 T-count problem (system + 1 ancilla).

State: isometry J (9x3) = R(P_d)...R(P_1) E0, considered modulo  J ~ e^{i phi} K J G
(K any 2-qutrit Clifford, G any 1-qutrit Clifford).  orbit(R(P) J) over all 80 P depends only on
orbit(J), so a BFS over orbit representatives is exhaustive.  Dedup uses a Clifford-invariant
bucket hash + an EXACT equivalence test (symplectic prefilter + explicit K, G search), so no orbit is
ever wrongly merged (rounding can only create duplicate orbits, never lose one).
The level-4 target has T-count t (1 ancilla) iff orbit(E0 L) first appears at depth t.
Bidirectional: also BFS from E0 L with R^dag, and match orbits exactly (MITM).
"""
import numpy as np, time, sys, os
from l4core import *
from l4_verify import clifford_reps, cliff1

KR = clifford_reps(); G1 = cliff1()
Q = float(os.environ.get("L4Q", "1e5"))

def pauli_index(M):
    """M: (N,9,9) each proportional to a 2-qutrit Pauli X^a Z^b. Return index into P2lab."""
    N = len(M)
    r = np.argmax(np.abs(M[:, :, 0]) > 0.5, axis=1)        # X-shift of |00>: r = 3*a1 + a2
    a1, a2 = r // 3, r % 3
    # D = X^{-a} M is diagonal ~ Z^b ; read phases at |01> and |10> relative to |00>
    idx = np.arange(N)
    col0 = M[idx, r, 0]
    r01 = 3*a1 + (a2+1) % 3; r10 = 3*((a1+1) % 3) + a2
    p01 = M[idx, r01, 1]/col0; p10 = M[idx, r10, 3]/col0
    b2 = np.rint(np.angle(p01)/(2*np.pi/3)).astype(int) % 3
    b1 = np.rint(np.angle(p10)/(2*np.pi/3)).astype(int) % 3
    return ((a1*3 + b1)*3 + a2)*3 + b2

def perm_table():
    fn = "/home/hlamm/Desktop/efficent_gates/unified/level4/cliff2_perm.npy"
    if os.path.exists(fn): return np.load(fn)
    out = np.zeros((len(KR), 81), np.int16)
    for i in range(81):
        M = np.einsum('sji,jk,skl->sil', np.conj(KR), P2[i], KR)       # K^dag P K
        out[:, i] = pauli_index(M)
    np.save(fn, out); return out

PERM = perm_table()
assert all(PERM[0] == np.arange(81))

MARGIN = [1.0]
def cvals(J):
    """(N,81,9) |c(A,P)|^2 quantized ints. Tracks the minimum distance of v*Q to a rounding
    boundary (MARGIN); if MARGIN >> float error (~1e-9 here), every hash/filter is stable, so
    Clifford-equivalent states always get identical quantized invariants (no missed matches)."""
    v = invariant_values(J).reshape(len(J), 81, 9) * Q
    r = np.rint(v)
    MARGIN[0] = min(MARGIN[0], float(np.min(0.5 - np.abs(v - r))))
    return r.astype(np.int64)

def mix(a, axis):
    """hash sorted vectors along axis -> uint64"""
    a = np.sort(a, axis=axis)
    h = np.zeros(np.delete(a.shape, axis), dtype=np.uint64)
    m = np.uint64(1099511628211)
    with np.errstate(over='ignore'):
        for k in range(a.shape[axis]):
            h = (h ^ np.take(a, k, axis=axis).astype(np.uint64)) * m + np.uint64(k+7)
    return h

def signatures(J):
    """returns (v (N,81,9) ints, colsig (N,81) uint64, bucket hash (N,))"""
    v = cvals(J)
    colsig = mix(v, 2)                 # N,81
    rowsig = mix(v, 1)                 # N,9  (per A, over P)
    b = mix(colsig.view(np.int64), 1) ^ (mix(rowsig.view(np.int64), 1) * np.uint64(31))
    return v, colsig, b

def _p1index(M):
    for i, A in enumerate(P1):
        t = np.trace(A.conj().T @ M)/3
        if abs(abs(t)-1) < 1e-9: return i
    raise ValueError
GPERM = np.array([[_p1index(G @ A @ G.conj().T) for A in P1] for G in G1])   # 216 x 9
P2conj = np.conj(P2arr)

def equiv(J1, v1, sig1, J2, v2, sig2):
    """exact test: exists Clifford K, G, phase with K J1 = e^{i phi} J2 G.  Returns (K, G, phase) or None."""
    ok = np.all(sig1[PERM] == sig2[None, :], axis=1)
    S = np.nonzero(ok)[0]
    if len(S) == 0: return None
    A = v1[PERM[S]]                                  # nS,81,9 : v1[perm_S(P), A]
    Bg = v2[:, GPERM]                                # 81,216,9 -> v2[P, gperm(A)]
    Bg = np.transpose(Bg, (1, 0, 2))                 # 216,81,9
    for si in range(len(S)):
        gm = np.nonzero(np.all(A[si][None] == Bg, axis=(1, 2)))[0]
        if len(gm) == 0: continue
        X = KR[S[si]] @ J1                           # 9x3
        Ms = np.einsum('ij,gjk->gik', J2, G1[gm])    # nG,9,3
        Y = np.einsum('gik,lk->gil', Ms, np.conj(X)) # M X^dag
        co = np.abs(np.einsum('qij,gij->gq', P2conj, Y))   # |Tr(Q^dag Y)|
        hit = np.argwhere(co > 3-1e-7)
        if len(hit):
            g, q = hit[0]
            K = P2[q] @ KR[S[si]]
            G = G1[gm[g]]
            ph = np.vdot(J2 @ G, K @ J1)/3
            assert np.allclose(K @ J1, ph*(J2 @ G))
            return K, G, ph
    return None

class OrbitStore:
    def __init__(self): self.buckets = {}; self.reps = []   # rep: (J, colsig, depth, seq)
    def add(self, J, v, colsig, b, depth, seq):
        lst = self.buckets.setdefault(int(b), [])
        for k in lst:
            r = self.reps[k]
            if equiv(J, v, colsig, r[0], r[1], r[2]) is not None: return False
        lst.append(len(self.reps)); self.reps.append((J, v, colsig, depth, seq)); return True
    def find(self, J, v, colsig, b):
        for k in self.buckets.get(int(b), []):
            r = self.reps[k]
            e = equiv(J, v, colsig, r[0], r[1], r[2])
            if e is not None: return k, e
        return None

def bfs(J0, rot, maxd, label, maxorbits=10**6, t0=None):
    t0 = t0 or time.time()
    st = OrbitStore()
    v, cs, b = signatures(J0[None]); st.add(J0, v[0], cs[0], b[0], 0, ())
    frontier = [0]
    for d in range(1, maxd+1):
        new = []
        for k in frontier:
            J = st.reps[k][0]; seq = st.reps[k][4]
            C = np.einsum('pij,jk->pik', rot, J)
            v, cs, bb = signatures(C)
            for p in range(80):
                if st.add(C[p], v[p], cs[p], bb[p], d, seq+(p,)): new.append(len(st.reps)-1)
        frontier = new
        print(f"[{label}] depth {d}: {len(new)} new orbits (total {len(st.reps)})  [{time.time()-t0:.0f}s]", flush=True)
        if len(st.reps) > maxorbits or not new: break
    return st

if __name__ == "__main__":
    import pickle
    DF = int(sys.argv[1]); DB = int(sys.argv[2])
    t0 = time.time()
    fw = bfs(E0.astype(complex), ROTS, DF, "fwd", t0=t0)
    results = {}
    for k in (1, 2):
        L = np.diag([1, 1, z**k]); JT = E0 @ L
        v, cs, b = signatures(JT[None])
        hit = fw.find(JT, v[0], cs[0], b[0])
        print(f"target k={k}: in forward orbits? ", None if hit is None else (fw.reps[hit[0]][3], fw.reps[hit[0]][4]))
        bw = bfs(JT, ROTS_DAG, DB, f"bwd k={k}", t0=t0)
        best = None
        for (J, v, c, d, seq) in bw.reps:
            _, _, bb = signatures(J[None])
            h = fw.find(J, v, c, bb[0])
            if h is not None:
                tt = d + fw.reps[h[0]][3]
                if best is None or tt < best[0]: best = (tt, fw.reps[h[0]][4], seq, h[1])
        print(f"target k={k}: MITM exact min t over fwd<= {DF}, bwd<= {DB}: ",
              None if best is None else best[:3], f"[{time.time()-t0:.0f}s]", flush=True)
        results[k] = best
        if k == 1 and best is None and DB == DF:
            pass
    pickle.dump({k: (None if r is None else (r[0], r[1], r[2], r[3][0], r[3][1], r[3][2])) for k, r in results.items()},
                open(f"orbits_result_{DF}_{DB}.pkl", "wb"))
    print("min rounding margin over all states hashed:", MARGIN[0], " (Q =", Q, ")")
    print("orbit counts fwd by depth:", np.bincount([r[3] for r in fw.reps]).tolist())
