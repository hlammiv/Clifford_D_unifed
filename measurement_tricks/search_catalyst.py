"""search_catalyst.py -- catalyst search: qutrits (0 system, 1 catalyst |c>, 2 clean ancilla |0>), T-count <= 2
(raw enumeration of all products of <= 2 of the 728 T-rotations), terminal stabilizer measurement of any
3-qutrit Pauli Q.  Success (deterministic): for every outcome m the 9x3 block W_m (on the two remaining qutrits,
in a Clifford frame) is left-2-qutrit-Clifford equivalent (right 1-qutrit Clifford allowed, generous) to
  psi -> G psi (x) |c>      (target G applied, catalyst returned, up to a Clifford).
Equivalence: l4_orbits invariant bucket + exact l4_orbits.equiv.  Also records outcomes that succeed at all."""
import sys, os, numpy as np, numba as nb, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla')
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/level4')
import c3core as cc
os.chdir('/home/hlamm/Desktop/efficent_gates/unified/level4')
import l4_orbits as lo
from search_adaptive import to_z2
z = np.exp(2j*np.pi/9); w = z**3
SRC = cc.SRC.astype(np.int64); PH = cc.PH.astype(np.complex128); ROT = cc.ROT; DBL = cc.DBL

@nb.njit(cache=True)
def scalar_qs(V, SRC, PH, DBL, out):
    n = 0
    for k in range(1, 729):
        if DBL[k] < k: continue
        bad = False; d0 = 0j
        for i in range(3):
            for j in range(3):
                s = 0j
                for y in range(27): s += np.conj(V[y, i])*PH[k, y]*V[SRC[k, y], j]
                if i != j:
                    if abs(s) > 1e-7: bad = True; break
                elif i == 0: d0 = s
                elif abs(s-d0) > 1e-7: bad = True; break
            if bad: break
        if not bad: out[n] = k; n += 1
    return n

KC = {}
def blocks(V, k):
    if k not in KC: KC[k] = to_z2(k)
    Y = KC[k] @ V; out = []
    for m in range(3):
        W = np.array([[Y[9*a+3*b+m, j] for j in range(3)] for a in range(3) for b in range(3)])
        p = np.linalg.norm(W)**2/3
        out.append(None if p < 1e-10 else W/np.sqrt(p))
    return out

def run(cname, c, targets):
    c = np.asarray(c, complex); c /= np.linalg.norm(c)
    E = np.zeros((27, 3), complex)
    for x in range(3):
        for y in range(3): E[9*x+3*y, x] = c[y]
    tg = {}
    for nm, G in targets.items():
        J = np.kron(G, c[:, None]); v, cs, b = lo.signatures(J[None]); tg[nm] = (J, v[0], cs[0], b[0])
    res = {nm: {'det': [], 'any': 0} for nm in targets}
    out = np.zeros(400, np.int64)
    elems = [((), E)] + [((r,), ROT[r] @ E) for r in range(728)]
    lev1 = elems[1:]
    def test(seq, V):
        n = scalar_qs(V, SRC, PH, DBL, out)
        for k in out[:n]:
            bl = [W for W in blocks(V, int(k)) if W is not None]
            v, cs, b = lo.signatures(np.array(bl))
            for nm, (J, vt, cst, bt) in tg.items():
                okm = [b[i] == bt and lo.equiv(bl[i], v[i], cs[i], J, vt, cst) is not None for i in range(len(bl))]
                if any(okm): res[nm]['any'] += 1
                if all(okm): res[nm]['det'].append((seq, int(k)))
    t0 = time.time()
    for seq, V in elems: test(seq, V)
    for (s1, V1) in (lev1 if DEPTH >= 2 else []):
        for r in range(728):
            if r == s1[0] or DBL[r+1] == s1[0]+1: continue
            test(s1 + (r,), ROT[r] @ V1)
    for nm in targets:
        print(f"catalyst {cname:22s} target {nm}: t<={DEPTH} deterministic hits {res[nm]['det'][:3] or 'NONE'}; "
              f"(Q,outcome-sets with >=1 successful outcome: {res[nm]['any']})  [{time.time()-t0:.0f}s]", flush=True)

DEPTH = int(os.environ.get('CAT_DEPTH', '2'))
if __name__ == '__main__':
    T = {'R': np.diag([1, 1, -1]).astype(complex), 'L': np.diag([1, 1, z]), 'T(control)': np.diag([1, z, z**8])}
    cats = {'Strange (|1>-|2>)/sqrt2': [0, 1, -1], 'T|+>': [1, z, z**8], 'L|+> (level-4 state)': [1, 1, z],
            'R|+>': [1, 1, -1], 'Norrell (-1,2,-1)': [-1, 2, -1], '|0> (control, no catalyst)': [1, 0, 0]}
    which = sys.argv[1:] or list(cats)
    for nm in which: run(nm, cats[nm], T)
