"""l4_7T_verify.py -- STANDALONE exact check: diag(1,1,zeta) with T-count 7 and one clean |0> ancilla.

Circuit (qutrit 0 = system, qutrit 1 = ancilla, time order left->right):
   C2X  = H1, X0, X1, [X1^dag, T1, X1, SUM01] x3, X0^dag, X1^dag, H1^dag      (3 T : |2>-controlled X)
   full = C2X ; T1 ; C2X^dag                                                   (3 + 1 + 3 = 7 T)
Mechanism (compute / phase / uncompute): C2X writes [x==2] into the ancilla, T on the ancilla gives
zeta^{t([x==2])} = zeta^{[x==2]}, C2X^dag uncomputes.  diag(1,1,zeta) has exponent sum 1 (not 0 mod 3),
i.e. it is level 4 and not in single-qutrit Clifford+T (det obstruction), so the ancilla is essential.
Also re-verifies the (different) 7-T word found by the exact orbit MITM (l4_7T_gates.json)."""
import numpy as np, json, os
z = np.exp(2j*np.pi/9); w = z**3; I3 = np.eye(3)
H = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(-3+0j)
X = np.roll(I3, 1, axis=0); Z = np.diag([1, w, w**2]); S = np.diag([1, 1, w]); T = np.diag([1, z, z**8])
dg = lambda A: A.conj().T
SUM = np.zeros((9, 9))
for x in range(3):
    for y in range(3): SUM[3*x+(y+x) % 3, 3*x+y] = 1
one = {"H": H, "Hdg": dg(H), "X": X, "Xdg": dg(X), "Z": Z, "S": S, "Sdg": dg(S), "T": T, "Tdg": dg(T)}
def gate(name):
    if name in ("SUM01",): return SUM
    if name == "SUMdg01": return dg(SUM)
    base, q = name[:-1], int(name[-1])
    return np.kron(one[base], I3) if q == 0 else np.kron(I3, one[base])
CLIFF_OK = {"H", "Hdg", "X", "Xdg", "Z", "S", "Sdg", "SUM", "SUMdg"}
def inverse(word):
    inv = {"H": "Hdg", "Hdg": "H", "X": "Xdg", "Xdg": "X", "T": "Tdg", "Tdg": "T", "S": "Sdg", "Sdg": "S"}
    out = []
    for g in reversed(word):
        if g.startswith("SUM"): out.append("SUMdg01" if g == "SUM01" else "SUM01")
        elif g[:-1] == "Z": out += [g, g]          # Z^dag = Z^2
        else: out.append(inv[g[:-1]] + g[-1])
    return out
def unitary(word):
    U = np.eye(9, dtype=complex)
    for g in word: U = gate(g) @ U
    return U
def tcount(word): return sum(g[:-1] in ("T", "Tdg") for g in word)
def check(word, label):
    assert all(g[:-1] in CLIFF_OK | {"T", "Tdg"} or g.startswith("SUM") for g in word)
    U = unitary(word)
    rng = np.random.default_rng(7); errs = []
    for _ in range(200):
        psi = rng.normal(size=3)+1j*rng.normal(size=3); psi /= np.linalg.norm(psi)
        out = U @ np.kron(psi, [1, 0, 0])
        tgt = np.kron(np.diag([1, 1, z]) @ psi, [1, 0, 0])
        ph = np.vdot(tgt, out); errs.append(np.linalg.norm(out - ph*tgt)); assert abs(abs(ph)-1) < 1e-12
    print(f"{label}: T-count {tcount(word)}, {len(word)} gates, max |U(psi,0) - e^(i phi) (L psi,0)| = {max(errs):.2e}")
    assert max(errs) < 1e-12
C2X = ["H1", "X0", "X1"] + ["Xdg1", "T1", "X1", "SUM01"]*3 + ["Xdg0", "Xdg1", "Hdg1"]
ctrl = np.kron(np.diag([1, 1, 0]), I3) + np.kron(np.diag([0, 0, 1]), X)
ph = np.vdot(ctrl, unitary(C2X))/9
print("C2X word == |2>-controlled X exactly:", np.allclose(unitary(C2X), ph*ctrl), "phase", np.round(ph, 12), "T-count", tcount(C2X))
full = C2X + ["T1"] + inverse(C2X)
print("gate list:", " ".join(full))
check(full, "compute/phase/uncompute")
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l4_7T_gates.json")
if os.path.exists(fn):
    wd = [g.replace("T0", "T0") for g in json.load(open(fn))]
    wd = [{"Tdg0": "Tdg0"}.get(g, g) for g in wd]
    check(wd, "MITM-found word (l4_7T_gates.json)")
