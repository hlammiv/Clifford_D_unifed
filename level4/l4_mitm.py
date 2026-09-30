"""l4_mitm.py -- meet-in-the-middle T-count search, 2 qutrits (system + 1 ancilla |0>).
forward:  J = R_s..R_1 E0 ;  backward: J = R_{s+1}^dag .. R_t^dag E0 L.
Match of Clifford-orbit invariant (left 2-qutrit Clifford, right 1-qutrit Clifford, phase) at
depths (s, s') => candidate with t = s + s'. No match for all s+s' <= t  =>  T-count > t (1 ancilla).
usage: python3 l4_mitm.py DF DB     (max forward depth, max backward depth)
"""
import sys, time, numpy as np
from l4core import *

def expand(J, last, seqs, rot):
    """Apply every allowed rotation to every state. rot: (80,9,9)."""
    outJ, outL, outS = [], [], []
    for j in range(80):
        if last is None:
            sel = np.arange(len(J))
        else:
            sel = np.nonzero(np.array([j in AL_SET[p] for p in range(80)])[last])[0]
        if len(sel) == 0: continue
        outJ.append(np.einsum('ij,njk->nik', rot[j], J[sel]))
        outL.append(np.full(len(sel), j, np.int16))
        outS.append(np.hstack([seqs[sel], np.full((len(sel), 1), j, np.int16)]))
    return np.concatenate(outJ), np.concatenate(outL), np.concatenate(outS)

AL_SET = [set(a.tolist()) for a in ALLOWED]

def hashes(J, chunk=20000):
    return np.concatenate([inv_hash_int(J[i:i+chunk]) for i in range(0, len(J), chunk)])

def run(DF, DB, targets):
    t0 = time.time()
    fwd = {}   # hash -> (depth, seq)
    J = E0[None].copy(); last = None; seqs = np.zeros((1, 0), np.int16)
    for d in range(DF+1):
        if d > 0:
            J, last, seqs = expand(J, last, seqs, ROTS)
        h = hashes(J)
        for k, hh in enumerate(h):
            if hh not in fwd: fwd[int(hh)] = (d, seqs[k])
        print(f"fwd depth {d}: {len(J)} states, {len(fwd)} distinct inv so far  [{time.time()-t0:.0f}s]", flush=True)
    best = None
    for name, L in targets.items():
        J = (E0 @ L)[None].copy(); last = None; seqs = np.zeros((1, 0), np.int16)
        for d in range(DB+1):
            if d > 0:
                J, last, seqs = expand(J, last, seqs, ROTS_DAG)
            h = hashes(J)
            hits = [(fwd[int(hh)][0] + d, fwd[int(hh)][1], seqs[k]) for k, hh in enumerate(h) if int(hh) in fwd]
            nb = len(set(h.tolist()))
            if hits:
                m = min(x[0] for x in hits)
                print(f"  [{name}] bwd depth {d}: {len(J)} states ({nb} inv); {len(hits)} invariant hits, min t = {m}", flush=True)
                for x in hits:
                    if x[0] == m:
                        print("     e.g. fwd", x[1].tolist(), "bwd", x[2].tolist()); break
                if best is None or m < best[0]: best = (m, name, [x for x in hits if x[0] == m][:50])
            else:
                print(f"  [{name}] bwd depth {d}: {len(J)} states ({nb} inv); no hits  [{time.time()-t0:.0f}s]", flush=True)
    return best

if __name__ == "__main__":
    DF, DB = int(sys.argv[1]), int(sys.argv[2])
    targets = {f"diag(1,1,z^{k})": np.diag([1, 1, z**k]) for k in (1, 2)}
    best = run(DF, DB, targets)
    print("BEST:", None if best is None else best[:2])
    if best is not None:
        np.save(f"mitm_hits_{DF}_{DB}.npy", np.array([np.concatenate([h[1], [-1], h[2]]) for h in best[2]], dtype=object), allow_pickle=True)
