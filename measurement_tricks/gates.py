"""gates.py -- named 2-qutrit gates (qutrit 0 = system, qutrit 1 = ancilla, index 3*x0 + x1) and helpers to
turn a T-rotation f(P) into an explicit Clifford+T word.  Conventions as in r_from_d/verify_R7T.py:
H = (1/sqrt(-3))[w^{jk}], S = diag(1,1,w), X|j>=|j+1>, Z = diag(1,w,w^2), T = diag(1,z,z^8), z = e^{2 pi i/9}."""
import itertools, numpy as np
z = np.exp(2j*np.pi/9); w = z**3; I3 = np.eye(3)
G1Q = {'H': np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(-3+0j),
       'S': np.diag([1, 1, w]), 'X': np.roll(I3, 1, axis=0), 'Z': np.diag([1, w, w*w]),
       'T': np.diag([1, z, z**8])}
for k in list(G1Q): G1Q[k+'dg'] = G1Q[k].conj().T
def SUM(c):
    M = np.zeros((9, 9))
    for x0, x1 in itertools.product(range(3), repeat=2):
        y0, y1 = (x0, (x1+x0) % 3) if c == 0 else ((x0+x1) % 3, x1)
        M[3*y0+y1, 3*x0+x1] = 1
    return M
G2Q = {'SUM01': SUM(0), 'SUM10': SUM(1)}
G2Q['SUM01dg'] = G2Q['SUM01'].T; G2Q['SUM10dg'] = G2Q['SUM10'].T
def gate(g):
    if g in G2Q: return G2Q[g]
    name, q = g[:-1], int(g[-1])
    return np.kron(G1Q[name], I3) if q == 0 else np.kron(I3, G1Q[name])
def circuit_matrix(gl):
    U = np.eye(9, dtype=complex)
    for g in gl: U = gate(g) @ U
    return U
def tcount(gl): return sum(1 for g in gl if g[:-1] in ('T', 'Tdg'))
def inv(g):
    if g in G2Q: return g[:-2] if g.endswith('dg') else g+'dg'
    name, q = g[:-1], g[-1]
    return (name[:-2] if name.endswith('dg') else name+'dg') + q
CLIFF_GENS = ['H0', 'H1', 'S0', 'S1', 'X0', 'X1', 'Z0', 'Z1', 'SUM01', 'SUM10']

def _key(M):
    v = M.ravel(); i = int(np.argmax(np.abs(v) > 0.5)); return (i, tuple(np.round(v / v[i] * 0 + v, 6)))

def rotation_word(P):
    """Clifford word K (list, time order) with K (I x Z) K^dag == P exactly; f(P) = K . T1 . K^dag."""
    Z1 = gate('Z1'); start = (Z1, [])
    seen = {tuple(np.round(Z1.ravel(), 6))}; frontier = [start]
    while frontier:
        nf = []
        for M, word in frontier:
            if np.allclose(M, P): return word
            for g in CLIFF_GENS:
                Gm = gate(g); N = Gm @ M @ Gm.conj().T; k = tuple(np.round(N.ravel(), 6))
                if k not in seen: seen.add(k); nf.append((N, word + [g]))
        frontier = nf
    raise RuntimeError('no word')

def rot_gates(P):
    K = rotation_word(P)
    return [inv(g) for g in reversed(K)] + ['T1'] + K

def simplify(gl):
    out = []
    for g in gl:
        if out and out[-1] == inv(g): out.pop()
        else: out.append(g)
    return out

ORDER = {'H': 4, 'S': 3, 'X': 3, 'Z': 3, 'T': 9, 'SUM01': 3, 'SUM10': 3}
def _base(g):
    if g in G2Q: return (g[:5], 'q', -1 if g.endswith('dg') else 1)
    name, q = g[:-1], g[-1]
    return (name[:-2], q, -1) if name.endswith('dg') else (name, q, 1)
def peephole(gl):
    """collapse runs of the same Clifford generator (T untouched) to the shortest power"""
    out = []; i = 0
    while i < len(gl):
        b, q, s = _base(gl[i])
        if b == 'T': out.append(gl[i]); i += 1; continue
        j = i; tot = 0
        while j < len(gl) and _base(gl[j])[:2] == (b, q): tot += _base(gl[j])[2]; j += 1
        o = ORDER[b]; tot %= o
        nm = (lambda dg: (b + ('dg' if dg else '')) if q == 'q' else (b + ('dg' if dg else '') + q))
        if tot <= o // 2: out += [nm(False)]*tot
        else: out += [nm(True)]*(o - tot)
        i = j
    return out
