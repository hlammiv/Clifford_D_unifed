"""search2.py -- EXHAUSTIVE measurement-model search with TWO clean ancillas, T-count <= 3.
Elements: exact orbit representatives (mod left 3-qutrit Clifford, right 1-qutrit Clifford) of all
isometries f(P_t)...f(P_1) E00, t <= 3, from two_ancilla/out/fwd_3.pkl (125 orbits).
Measurement: Pauli Q1 (728), then an outcome-ADAPTIVE commuting Pauli Q2 (any stabilizer measurement of
two ancillas, adaptive or not, is of this form up to Clifford), then a Clifford on the system.
Deterministic hit for target G iff  exists Q1, and per outcome m1 some Q2(m1), and one common right
Clifford C' such that every non-zero final block (in a Clifford logical basis of the joint eigenspace)
lies in Cliff1 G C'.  We also record any outcome that yields G at all (for RUS)."""
import sys, os, pickle, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meascore import is_cliff1, P1A, G1, sig
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla')
import c3core as cc
z = np.exp(2j*np.pi/9); w = z**3
PM, SFORM, LABS, ADD, DBL = cc.PM, cc.SFORM, cc.LABS, cc.ADD, cc.DBL
fw = pickle.load(open('/home/hlamm/Desktop/efficent_gates/unified/two_ancilla/out/fwd_3.pkl', 'rb'))
TOL = 1e-7

def proj(k, m):
    P = PM[k]; return (np.eye(27) + w**(-m)*P + w**(-2*m)*P @ P)/3   # eigenvalue w^m of P

def scalar(V, k):
    M = V.conj().T @ PM[k] @ V
    return np.abs(M - M[0, 0]*np.eye(3)).max() < TOL

def lines(cands):
    seen = set(); out = []
    for k in cands:
        if k == 0 or k in seen: continue
        seen.add(k); seen.add(int(DBL[k])); out.append(k)
    return out

def logical_basis(Pi, k1, k2):
    """Clifford isometry (27x3) onto range(Pi), Pi = joint eigenprojector of <P_k1, P_k2>."""
    grp = {0}
    for a in range(3):
        for b in range(3):
            v = (LABS[k1]*a + LABS[k2]*b) % 3; grp.add(cc.vec2idx(v))
    comm = [k for k in range(1, 729) if SFORM[k, k1] == 0 and SFORM[k, k2] == 0 and k not in grp]
    for zb in comm:
        for xb in comm:
            if SFORM[zb, xb] != 0:
                Zb = Pi @ PM[zb] @ Pi
                ev, U = np.linalg.eig(Zb + 5*(np.eye(27)-Pi))
                i = int(np.argmin(np.abs(ev - 5) < 1e-6)) if False else None
                # pick an eigenvector of Zb inside range(Pi)
                sel = [j for j in range(27) if abs(ev[j]-5) > 1e-6]
                v0 = U[:, sel[0]]; v0 = v0/np.linalg.norm(v0)
                J = np.stack([v0, PM[xb] @ v0, PM[xb] @ PM[xb] @ v0], axis=1)
                assert np.allclose(J.conj().T @ J, np.eye(3), atol=1e-9)
                return J
    raise RuntimeError

def valid_set(Y, G):
    """set of right-Clifford indices c with Y C'^dag in Cliff1 G ; Y 3x3 unitary"""
    Gd = G.conj().T
    return {c for c, Cp in enumerate(G1) if is_cliff1(Y @ Cp.conj().T @ Gd, P1A)}

def analyse(V, G, sG):
    best = None; anyG = False
    for k1 in lines(range(1, 729)):
        if not scalar(V, k1): continue
        S_total = None; okall = True
        for m1 in range(3):
            V1 = proj(k1, m1) @ V
            p1 = np.linalg.norm(V1)**2/3
            if p1 < 1e-10: continue
            V1 = V1/np.sqrt(p1)
            Sm1 = set()
            for k2 in lines([k for k in range(1, 729) if SFORM[k, k1] == 0 and k not in (k1, int(DBL[k1]))]):
                if not scalar(V1, k2): continue
                Sk2 = None
                for m2 in range(3):
                    Pi = proj(k1, m1) @ proj(k2, m2)
                    Y = Pi @ V1; p2 = np.linalg.norm(Y)**2/3
                    if p2 < 1e-10: continue
                    J = logical_basis(Pi, k1, k2)
                    B = J.conj().T @ Y / np.sqrt(p2)
                    if np.abs(sig(B) - sG).max() < 1e-6: anyG = True
                    vs = valid_set(B, G) if np.abs(sig(B) - sG).max() < 1e-6 else set()
                    Sk2 = vs if Sk2 is None else (Sk2 & vs)
                    if not Sk2: break
                if Sk2: Sm1 |= Sk2
            if not Sm1: okall = False; break
            S_total = Sm1 if S_total is None else (S_total & Sm1)
            if not S_total: okall = False; break
        if okall and S_total: best = (k1, sorted(S_total)[:3]); break
    return best, anyG

if __name__ == '__main__':
    tg = {'R': np.diag([1, 1, -1]).astype(complex), 'L': np.diag([1, 1, z])}
    # positive control: R 4T one-ancilla gadget embedded with an idle 2nd ancilla is found by search1; here
    # control that the 2-ancilla machinery finds L = c2x;T;measure embedded on qutrits (0,1), qutrit 2 idle.
    os.chdir('/home/hlamm/Desktop/efficent_gates/unified/level4')
    exec(open("l4_from_8T.py").read().split("# 2)")[0])
    U = np.kron(np.kron(np.eye(3), np.diag([1, z, z**8])) @ c2x, np.eye(3))
    print('control L4 (4T, embedded):', analyse(U @ cc.E00, tg['L'], sig(tg['L'])))
    for name in ('R', 'L'):
        G = tg[name]; sG = sig(G); found = []; anyG = []
        for i, M in enumerate(fw['M']):
            b, a = analyse(np.asarray(M), G, sG)
            if b: found.append((fw['depth'][i], fw['seq'][i], b))
            if a: anyG.append(fw['depth'][i])
        print(f'{name}: {len(fw["M"])} orbits (depth <= 3, counts {np.bincount(fw["depth"]).tolist()}): '
              f'deterministic hits {found}; orbits with ANY outcome = {name}: {anyG}', flush=True)
