"""mitm3.py -- same exact MITM as mitm_core.py but for system + TWO clean ancillas (3 qutrits, 728 rotations).
usage: python3 mitm3.py <depth_store>   (searches T-count <= 2*(depth_store+1))"""
import sys, time, pickle, numpy as np
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/r_from_d')
from mitm_core import (paulisn, rot_mat, EMB, OMEGA_E, BINV, HCOEF, SCALE_D, hash_batch, children_hash,
                       embed_all, meet, path)
P3 = paulisn(3); NR = len(P3)
ROT3 = np.array([[rot_mat(s, p, a) for a in EMB] for (s, p) in P3])
PSRC3 = np.array([s for s, p in P3]); PPH3 = np.array([p for s, p in P3])
sc = float(9**SCALE_D)
H3 = lambda Ms: hash_batch(Ms, PSRC3, PPH3, OMEGA_E, BINV, sc, HCOEF)
CH3 = lambda Ms: children_hash(Ms, ROT3, PSRC3, PPH3, OMEGA_E, BINV, sc, HCOEF)
def e00():
    E = np.zeros((27, 3), complex)
    for x in range(3): E[9*x, x] = 1
    return E
def bfs3(M0, depth_store, tag):
    h0, _ = H3(M0[None]); levels = [dict(h=h0, par=np.array([-1]), rot=np.array([-1]))]; Ms = [M0[None]]
    seen = h0.copy()
    for d in range(1, depth_store+2):
        t0 = time.time(); prev = Ms[-1]; hs, err = CH3(prev); assert err.max() < 1e-3
        uniq, idx = np.unique(hs.ravel(), return_index=True); new = ~np.isin(uniq, seen)
        uniq, idx = uniq[new], idx[new]; par, rr = idx // NR, idx % NR
        levels.append(dict(h=uniq, par=par, rot=rr)); seen = np.union1d(seen, uniq)
        print(f'  [{tag}] level {d}: {len(uniq)} new classes ({hs.size} children, {time.time()-t0:.1f}s)', flush=True)
        if d <= depth_store:
            Ms.append(np.stack([np.einsum('eyx,exj->eyj', ROT3[r], prev[p]) for p, r in zip(par, rr)]))
    return levels
if __name__ == '__main__':
    ds = int(sys.argv[1]); t0 = time.time()
    la = bfs3(embed_all(lambda a: e00()), ds, 'E00')
    targets = {'R': [1, 1, -1]}
    for name, dg in targets.items():
        lb = bfs3(embed_all(lambda a: np.kron(np.diag(dg), np.eye(9)) @ e00()), ds, name)
        best, hits = meet(la, lb)
        print(name, 'min meet t =', best, 'hits', hits[:3], f'({time.time()-t0:.0f}s)')
        pickle.dump((la, lb), open(f'levels3_{name}_{ds}.pkl', 'wb'))
