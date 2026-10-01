"""search1.py -- EXHAUSTIVE measurement-model search, system + 1 clean ancilla, T-count <= 3 + DESC.
Every circuit with t <= 6 T-gates is (up to left 2-qutrit Clifford / right 1-qutrit Clifford, which the
test is invariant under) <= 3 raw rotations applied to an exact orbit representative of depth <= 3.
All such elements are tested (meascore.test_one prefilter, then full_check).
usage: python3 search1.py TARGET(R|L) [NPROC]"""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meascore import *
HERE = os.path.dirname(os.path.abspath(__file__))
TGT = sys.argv[1]; NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
G = {'R': np.diag([1, 1, -1]).astype(complex), 'L': np.diag([1, 1, z])}[TGT]
sI, sG = sig(np.eye(3)), sig(G)

def get_reps():
    fn = os.path.join(HERE, 'reps_d3.pkl')
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    os.chdir('/home/hlamm/Desktop/efficent_gates/unified/level4')
    import l4_orbits
    st = l4_orbits.bfs(E0.astype(complex), ROTS, 3, 'fwd')
    reps = [(J, d, seq) for (J, v, c, d, seq) in st.reps]
    pickle.dump(reps, open(fn, 'wb')); return reps

def job(k):
    J, d, seq = REPS[k]; res = []
    h = scan_batch(J[None], np.array([-1]), True, ROTS, ALLOWED_M, NALLOW, QSRC, QPH, KQ, P1A, sI, sG)
    for (_, r, q) in h: res.append((d + (r >= 0), seq + ((r,) if r >= 0 else ()), (), int(q)))
    L1 = np.einsum('pij,jk->pik', ROTS, J)
    L2 = []; pv = []; par = []
    for a in range(80):
        for b in ALLOWED[a]:
            L2.append(ROTS[b] @ L1[a]); pv.append(b); par.append((a, b))
    L2 = np.array(L2)
    h = scan_batch(L2, np.array(pv), True, ROTS, ALLOWED_M, NALLOW, QSRC, QPH, KQ, P1A, sI, sG)
    for (i, r, q) in h:
        path = par[i] + ((r,) if r >= 0 else ())
        res.append((d + len(path), seq, path, int(q)))
    # full check of every prefilter hit
    out = []
    for (dep, sq, path, q) in res:
        V = J.copy()
        for r in path: V = ROTS[r] @ V
        if not path:
            V = E0.copy()
            for r in sq: V = ROTS[r] @ V
        ps, ci, lab = full_check(V, q, G)
        if ps > 1e-9: out.append((dep, ps, sq, path, q, ci, lab))
    return k, len(res), out

REPS = get_reps()
if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    print(f'target {TGT}: {len(REPS)} orbit reps (depth<=3), depth counts', np.bincount([r[1] for r in REPS]).tolist(), flush=True)
    allhits = []; npre = 0
    with Pool(NPROC) as pool:
        for k, n, out in pool.imap_unordered(job, range(len(REPS))):
            npre += n; allhits += out
    best = {}
    for (dep, ps, sq, path, q, ci, lab) in allhits:
        if dep not in best or ps > best[dep][1]: best[dep] = (dep, ps, sq, path, q, ci, lab)
    print(f'prefilter hits {npre}, full hits {len(allhits)}  [{time.time()-t0:.0f}s]')
    for dep in sorted(best):
        nd = sum(1 for h in allhits if h[0] == dep and h[1] > 1-1e-9)
        print(f'  T-count {dep}: best p_succ {best[dep][1]:.6f} (deterministic hits: {nd})  example {best[dep]}')
    if not best: print('  NO hit (no outcome ever yields the target) for t <= 6')
    pickle.dump(allhits, open(os.path.join(HERE, f'search1_{TGT}.pkl'), 'wb'))
