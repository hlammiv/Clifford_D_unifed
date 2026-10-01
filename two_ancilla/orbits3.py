"""orbits3.py -- EXACT orbit BFS + MITM for diag-type gates with TWO clean ancillas (3 qutrits).

State J = f(P_d)...f(P_1) E00 (27x3), taken modulo  J ~ e^{i phi} C J G  (C 3-qutrit Clifford,
G 1-qutrit Clifford).  orbit(f(P) J) over all 728 P depends only on orbit(J)  (C^dag f(P) C =
f(w^c P') = f(P') x Clifford commuting with it), so BFS over orbit representatives is exhaustive.
Dedup: invariant hash buckets + the exact test c3core.equiv (never merges inequivalent states).
Since f(P)^{-1} = f(P^2), the backward BFS from E00 L uses the same 728 generators.
T-count <= a+b with 2 ancillas  <=>  some depth<=a fwd orbit == some depth<=b bwd orbit.
usage: python3 orbits3.py DEPTH NPROC
"""
import numpy as np, sys, time, pickle, os
from multiprocessing import Pool
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla')
import c3core as cc
from c3core import ROT, E00, qlabels, inv_hash, equiv, z

OUT = '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla/out'
os.makedirs(OUT, exist_ok=True)
_G = {}

def _bucket_job(args):
    """args: (old_reps list of (M, L), cand list of (parent, rot)).  Returns list of indices of cand
    that are new orbits (each inequivalent to all old reps and to each other), and #tests."""
    olds, cands = args
    par = _G['par']
    reps = [(M, L) for (M, L) in olds]; new = []
    for ci, (p, r) in enumerate(cands):
        M = ROT[r] @ par[p]
        L = qlabels(M[None])[0]
        hit = False
        for (Mr, Lr) in reps:
            if equiv(M, Mr) is not None: hit = True; break
        if not hit: reps.append((M, L)); new.append(ci)
    return new, cc.MARGIN[0]

class Store:
    def __init__(self): self.M = []; self.L = []; self.h = []; self.depth = []; self.seq = []; self.buckets = {}
    def add(self, M, L, h, d, seq):
        self.buckets.setdefault(int(h), []).append(len(self.M))
        self.M.append(M); self.L.append(L.astype(np.int64)); self.h.append(int(h)); self.depth.append(d); self.seq.append(seq)

def bfs(J0, depth, label, nproc, t0):
    st = Store()
    L0 = qlabels(J0[None]); st.add(J0, L0[0], inv_hash(L0)[0], 0, ())
    frontier = [0]
    for d in range(1, depth+1):
        par = np.array([st.M[k] for k in frontier])
        # hashes of all children
        hs = np.zeros((len(par), 728), np.uint64)
        for i in range(len(par)):
            C = np.einsum('rij,jk->rik', ROT, par[i])
            for c0 in range(0, 728, 104):
                hs[i, c0:c0+104] = inv_hash(qlabels(C[c0:c0+104]))
        groups = {}
        for i in range(len(par)):
            for r in range(728): groups.setdefault(int(hs[i, r]), []).append((i, r))
        jobs = []; keys = []
        for h, lst in groups.items():
            olds = [(st.M[k], st.L[k]) for k in st.buckets.get(h, [])]
            jobs.append((olds, lst)); keys.append(h)
        _G['par'] = par
        order = np.argsort([-len(j[1])*(1+len(j[0])) for j in jobs])
        with Pool(nproc) as pool:
            res = pool.map(_bucket_job, [jobs[i] for i in order], chunksize=1)
        newf = []
        for i, (newidx, marg) in zip(order, res):
            cc.MARGIN[0] = min(cc.MARGIN[0], marg)
            for ci in newidx:
                p, r = jobs[i][1][ci]
                M = ROT[r] @ par[p]; L = qlabels(M[None])[0]
                st.add(M, L, keys[i], d, st.seq[frontier[p]] + (r,))
                newf.append(len(st.M)-1)
        frontier = newf
        print(f"[{label}] depth {d}: {len(groups)} child hash-buckets, {len(newf)} new orbits "
              f"(total {len(st.M)})  [{time.time()-t0:.0f}s]  margin {cc.MARGIN[0]:.2e}", flush=True)
    return st

def meet_job(args):
    (Mb, Lb), olds = args
    for (k, Mf) in olds:
        r = equiv(Mf, Mb)
        if r is not None: return k, r
    return None

def mitm(fw, bw, nproc):
    jobs = []; idx = []
    for j in range(len(bw.M)):
        ks = fw.buckets.get(bw.h[j], [])
        if ks: jobs.append(((bw.M[j], bw.L[j]), [(k, fw.M[k]) for k in ks])); idx.append(j)
    with Pool(nproc) as pool: res = pool.map(meet_job, jobs, chunksize=1)
    hits = []
    for j, r in zip(idx, res):
        if r is not None:
            k, (C, G, ph) = r
            hits.append((fw.depth[k] + bw.depth[j], fw.seq[k], bw.seq[j], C, G, ph))
    return len(jobs), sorted(hits, key=lambda x: x[0])

if __name__ == "__main__":
    DEP = int(sys.argv[1]); NP = int(sys.argv[2]); t0 = time.time()
    fw = bfs(E00.copy(), DEP, "fwd", NP, t0)
    pickle.dump(dict(M=fw.M, h=fw.h, depth=fw.depth, seq=fw.seq), open(f"{OUT}/fwd_{DEP}.pkl", "wb"))
    T1 = np.diag([1, z, z**8]); H1 = cc.H1
    TGT = {"L4k1": np.diag([1, 1, z]), "R": np.diag([1, 1, -1]),
           "ctl_T": T1, "ctl_THTHT": T1 @ H1 @ T1 @ H1 @ T1}
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else ["L4k1", "R"]
    for name in names:
        JT = E00 @ TGT[name]
        bw = bfs(JT, DEP, f"bwd {name}", NP, t0)
        pickle.dump(dict(M=bw.M, h=bw.h, depth=bw.depth, seq=bw.seq), open(f"{OUT}/bwd_{name}_{DEP}.pkl", "wb"))
        nb, hits = mitm(fw, bw, NP)
        print(f"== {name}: {nb} bwd orbits share a hash bucket with fwd orbits; exact meets: {len(hits)}", flush=True)
        if hits:
            print(f"   min t = {hits[0][0]}; fwd seq {hits[0][1]} bwd seq {hits[0][2]}", flush=True)
            pickle.dump(hits, open(f"{OUT}/hits_{name}_{DEP}.pkl", "wb"))
        else:
            print(f"   NO exact meet: T-count > {2*DEP} with two clean ancillas", flush=True)
        print(f"   [{time.time()-t0:.0f}s] margin {cc.MARGIN[0]:.2e} equiv stats {cc.STATS}", flush=True)
