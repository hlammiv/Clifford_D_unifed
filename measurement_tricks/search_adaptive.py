"""search_adaptive.py -- ADAPTIVE (one mid-circuit measurement) search, system + 2 clean ancillas, T-count <= 3.
Stage 1: any 3-qutrit circuit with t1 in {1,2} T-gates (exact orbit reps, two_ancilla/out/fwd_3.pkl), then a
measurement of one 3-qutrit Pauli Q.  Each outcome m leaves a 9x3 isometry W_m on two qutrits.
Stage 2 (may depend on m): t2 <= 3 - t1 more T-rotations on those two qutrits, then a terminal stabilizer
measurement of the remaining ancilla (meascore.test_one / full_check).  We call the gadget feasible if EVERY
outcome m of stage 1 is completable (we do not even require a common right Clifford -> generous, so 'none'
is a valid lower bound for this class).  The Pauli Q is mapped to Z on qutrit 2 by an explicit Clifford."""
import sys, os, pickle, numpy as np, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meascore import test_one, full_check, QSRC, QPH, KQ, P1A, sig, ROTS, ALLOWED
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla')
import c3core as cc
z = np.exp(2j*np.pi/9)
fw = pickle.load(open('/home/hlamm/Desktop/efficent_gates/unified/two_ancilla/out/fwd_3.pkl', 'rb'))
PM, DBL = cc.PM, cc.DBL
Z2 = cc.gate1(np.diag([1, cc.w, cc.w**2]), 2)

def to_z2(k):
    """3-qutrit Clifford K with K P_k K^dag = phase * Z_2 (BFS on conjugation images)."""
    start = PM[k]; seen = {tuple(np.round(start.ravel(), 5))}; fr = [(start, np.eye(27, dtype=complex))]
    while fr:
        nf = []
        for M, K in fr:
            t = np.trace(Z2.conj().T @ M)/27
            if abs(abs(t)-1) < 1e-9: return K
            for g in cc.CLGENS:
                N = g @ M @ g.conj().T; key = tuple(np.round(N.ravel(), 5))
                if key not in seen: seen.add(key); nf.append((N, g @ K))
        fr = nf
KCACHE = {}
def completable(W, G, sI, sG, t2):
    """can the 9x3 isometry W reach G with <= t2 more rotations + terminal measurement?"""
    out = np.zeros(40, np.int64); lev = [(W, -1)]
    for d in range(t2+1):
        for (V, pv) in lev:
            k = test_one(V, QSRC, QPH, KQ, P1A, sI, sG, out)
            for a in range(k):
                if full_check(V, out[a], G)[0] > 1-1e-9: return d
        if d < t2:
            lev = [(ROTS[r] @ V, r) for (V, pv) in lev for r in (range(80) if pv < 0 else ALLOWED[pv])]
    return None

if __name__ == '__main__':
    tg = {'R': np.diag([1, 1, -1]).astype(complex), 'L': np.diag([1, 1, z])}
    sI = sig(np.eye(3))
    # control: L 4T split as stage 1 = C2X (3 T) + measure nothing useful... use t1=1: T on idle qutrit 2 then
    # measure it (outcome-independent) -> stage 2 must find L with 3 T? No: positive control instead = known
    # 4T R gadget with t1=1 (first rotation) and t2=3, allowing t2=3 here.
    for name in (sys.argv[1:] or ['R', 'L']):
        G = tg[name]; sG = sig(G); feasible = []
        for i, M in enumerate(fw['M']):
            t1 = fw['depth'][i]
            if t1 not in (1, 2): continue
            V = np.asarray(M)
            for k in range(1, 729):
                if DBL[k] < k: continue
                Mq = V.conj().T @ PM[k] @ V
                if np.abs(Mq - Mq[0, 0]*np.eye(3)).max() > 1e-7: continue
                if k not in KCACHE: KCACHE[k] = to_z2(k)
                Y = KCACHE[k] @ V; res = []
                for m in range(3):
                    Wm = np.array([[Y[9*x0+3*x1+m, j] for j in range(3)] for x0 in range(3) for x1 in range(3)])
                    p = np.linalg.norm(Wm)**2/3
                    if p < 1e-10: continue
                    res.append(completable(Wm/np.sqrt(p), G, sI, sG, 3-t1))
                if all(r is not None for r in res):
                    feasible.append((t1, fw['seq'][i], k, res))
        print(f'{name}: adaptive one-mid-measurement gadgets with total T <= 3: {feasible if feasible else "NONE"}', flush=True)
    # positive control (generosity check): allow total 4 for R with t1 = 1  -> expect feasible
    G = tg['R']; sG = sig(G); cnt = 0
    for i, M in enumerate(fw['M']):
        if fw['depth'][i] != 1: continue
        V = np.asarray(M)
        for k in range(1, 729):
            if DBL[k] < k: continue
            Mq = V.conj().T @ PM[k] @ V
            if np.abs(Mq - Mq[0, 0]*np.eye(3)).max() > 1e-7: continue
            if k not in KCACHE: KCACHE[k] = to_z2(k)
            Y = KCACHE[k] @ V; res = []
            for m in range(3):
                Wm = np.array([[Y[9*x0+3*x1+m, j] for j in range(3)] for x0 in range(3) for x1 in range(3)])
                p = np.linalg.norm(Wm)**2/3
                if p < 1e-10: continue
                res.append(completable(Wm/np.sqrt(p), G, sI, sG, 3))
            if all(r is not None for r in res): cnt += 1
    print('positive control (R, t1=1, t2<=3, total 4): feasible (orbit,Q) pairs =', cnt)
