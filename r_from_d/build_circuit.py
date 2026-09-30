"""build_circuit.py -- turn a MITM hit (rotation sequences sa, sb + Clifford C) into an explicit gate list over
{H, S, X, Z, T, Tdg on qutrit 0 or 1, SUM01 (|x,y>->|x,x+y>), SUM10} and write it to a JSON file.
Qutrit 0 = system, qutrit 1 = ancilla (starts and ends in |0>).
usage: python3 build_circuit.py <levels.pkl> <diag e.g. '1,1,-1'> <out.json>"""
import sys, json, pickle, itertools, numpy as np
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/r_from_d')
from mitm_core import e0, zeta, PAULIS, pauli_mat
from mitm_verify import prod, find_clifford, R1, P1, IDX, is_clifford
from mitm_core import path, meet

z = zeta(1); w = z**3; I3 = np.eye(3)
G1 = {'H': np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(-3+0j),
      'S': np.diag([1, 1, w]), 'X': np.roll(I3, 1, axis=0), 'Z': np.diag([1, w, w*w]),
      'T': np.diag([1, z, z**8]), 'Tdg': np.diag([1, z**8, z])}
def SUM(c):
    M = np.zeros((9, 9))
    for x0, x1 in itertools.product(range(3), repeat=2):
        if c == 0: M[3*x0+(x1+x0) % 3, 3*x0+x1] = 1
        else:      M[3*((x0+x1) % 3)+x1, 3*x0+x1] = 1
    return M
def gate(g):
    name, q = g
    if name == 'SUM01': return SUM(0)
    if name == 'SUM10': return SUM(1)
    return np.kron(G1[name], I3) if q == 0 else np.kron(I3, G1[name])
def word_mat(word):
    U = np.eye(9, dtype=complex)
    for g in word: U = gate(g) @ U
    return U

def pauli_id(Q):
    """Q = phase * X^a1 Z^b1 (x) X^a2 Z^b2 -> ((a1,b1,a2,b2), phase)"""
    col0 = np.argmax(abs(Q[:, 0])); a1, a2 = divmod(col0, 3)
    r = lambda x: Q[(3*((x[0]+a1) % 3)+(x[1]+a2) % 3), 3*x[0]+x[1]] / Q[col0, 0]
    lg = lambda c: int(round(np.angle(c)/(2*np.pi/3))) % 3
    b1, b2 = lg(r((1, 0))), lg(r((0, 1)))
    key = (a1, b1, a2, b2)
    P = np.eye(9) if key == (0, 0, 0, 0) else P1[IDX[key]]
    ph = Q[col0, 0] / P[col0, 0]
    assert np.allclose(Q, ph*P, atol=1e-9)
    return key, ph

BASIS = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
def sympl_key(U):
    return tuple(pauli_id(U @ P1[IDX[v]] @ U.conj().T)[0] for v in BASIS)

GENS = [('H', 0), ('H', 1), ('S', 0), ('S', 1), ('SUM01', None), ('SUM10', None)]
def sympl_table():
    start = np.eye(9, dtype=complex); tab = {sympl_key(start): []}; fr = [([], start)]
    while fr:
        nf = []
        for wd, U in fr:
            for g in GENS:
                V = gate(g) @ U; k = sympl_key(V)
                if k not in tab: tab[k] = wd + [g]; nf.append((wd+[g], V))
        fr = nf
    return tab

def clifford_word(C, tab):
    """C = lam * word ; returns (word, lam)"""
    wd = tab[sympl_key(C)]; W = word_mat(wd)
    key, ph = pauli_id(np.linalg.inv(W) @ C)          # W^-1 C = ph * Pauli
    a1, b1, a2, b2 = key
    pw = [('Z', 0)]*b1 + [('X', 0)]*a1 + [('Z', 1)]*b2 + [('X', 1)]*a2   # X^a Z^b : Z first in time
    full = pw + wd
    lam = (C @ np.linalg.inv(word_mat(full)))[0, 0]
    assert np.allclose(C, lam*word_mat(full), atol=1e-9)
    return full, lam

def main(pkl, diag, out):
    la, lb = pickle.load(open(pkl, 'rb'))
    best, hits = meet(la, lb)
    a, b, h = hits[0]; sa = path(la, a, h); sb = path(lb, b, h)
    D = np.diag([complex(eval(x.replace('z', 'zeta(1)'))) for x in diag.split(',')])
    E = e0(); DE = np.kron(D, I3) @ E
    C = find_clifford(prod(sb, DE), prod(sa, E)); assert C is not None and is_clifford(C)
    # list of non-Clifford steps: (rotation index, dagger?) with Cliffords in between
    steps = [('R', r, False) for r in sa] + [('C', np.linalg.inv(C))] + [('R', r, True) for r in reversed(sb)]
    tab = sympl_table(); print('symplectic table size', len(tab))
    Z0 = np.kron(G1['Z'], I3)
    # V words mapping Z0 -> P exactly (fix omega phase with X0 powers)
    def v_for(r):
        for wd in tab.values():
            V = word_mat(wd); Q = V @ Z0 @ V.conj().T
            key, ph = pauli_id(Q)
            if key != (0, 0, 0, 0) and np.allclose(Q, ph*P1[IDX[key]]) and IDX[key] == r:
                s = int(round(np.angle(ph)/(2*np.pi/3))) % 3
                for k in range(3):
                    wd2 = [('X', 0)]*k + wd; V2 = word_mat(wd2)
                    if np.allclose(V2 @ Z0 @ V2.conj().T, P1[r]): return wd2
        raise RuntimeError
    gates = []
    for st in steps:
        if st[0] == 'C':
            wd, lam = clifford_word(st[1], tab); gates += [list(g) for g in wd]
        else:
            _, r, dag = st; V = v_for(r); Vd_word = None
            # V^dag as a word: invert via its own Clifford decomposition
            Vd, _ = clifford_word(word_mat(V).conj().T, tab)
            gates += [list(g) for g in Vd] + [['Tdg' if dag else 'T', 0]] + [list(g) for g in V]
            assert np.allclose(word_mat(V) @ np.kron(G1['Tdg' if dag else 'T'], I3) @ word_mat(V).conj().T,
                               R1[r].conj().T if dag else R1[r], atol=1e-9)
    json.dump(dict(target=diag, gates=gates, sa=sa, sb=sb), open(out, 'w'))
    print('T-count', sum(g[0] in ('T', 'Tdg') for g in gates), 'total gates', len(gates), '->', out)

if __name__ == '__main__':
    main(*sys.argv[1:4])
