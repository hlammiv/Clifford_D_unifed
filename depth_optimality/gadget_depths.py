"""gadget_depths.py — T-count / T-depth of the four gadgets (+ an ancilla-parallel L4)."""
import json, sys
from pathlib import Path
import numpy as np
U_ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(U_ / "compiler"), str(Path(__file__).resolve().parent)]
import cd_to_ct as cc
from tdepth import Gate, parse, normalise, wire_depth, rot_depth, mergeable_pairs, tcount

def report(name, gl):
    gs = normalise(gl)
    print(f"{name:42s} T={tcount(gs):2d}  wire-depth={wire_depth(gs):2d}  rot-depth={rot_depth(gs):2d}  mergeable={mergeable_pairs(gs)}")
    return gs

L4u = cc._l4_word(2, +1)
Ru = cc._R_WORD
lm = json.load(open(U_ / "measurement_tricks/l_meas_4T.json"))["gates"] + [Gate("M", (1,), basis="F")]
rm = json.load(open(U_ / "measurement_tricks/r_meas_4T.json"))["gates"] + [Gate("M", (1,), basis="Z")]
report("L4 unitary (C2X;T;C2X^dag), 1 anc", L4u)
report("R unitary (R_7T_compact), 1 anc", Ru)
report("L4 measurement (C2X;T;meas F), 1 anc", lm)
report("R measurement (r_meas_4T), 1 anc", rm)
# print rotation axes for inspection
from tdepth import rotations
for name, gl in (("R unitary", Ru), ("R meas", rm), ("L4 unitary", L4u), ("L4 meas", lm)):
    nodes, n, _ = rotations(normalise(gl))
    print(name, [("".join(map(str, a)), k) for k, a, s, _ in nodes])

# ---- L4 with 2 extra ancillas: the 3 T's of C2X are a phase polynomial on (x, y) -> depth 1
# C2X = H1 X0 X1 [Xdg1 T1 X1 SUM01]x3 Xdg0 Xdg1 Hdg1 ; the bracket applies prod_k T(y + k x' )  (x' = x+1)
# Parallel version: copy y+kx' into fresh ancillas 2,3 by SUMs, apply the 3 T's simultaneously, uncopy.
def c2x_par():
    g = ["H1", "X0", "X1", "Xdg1"]                 # now ancilla holds y' ; T acts on y'
    # anc2 <- y' + x', anc3 <- y' + 2x'
    g += [("SUM12"), ("SUM02"), ("SUM13"), ("SUM03"), ("SUM03")]
    g += ["T1", "T2", "T3"]
    g += [("SUMdg03"), ("SUMdg03"), ("SUMdg13"), ("SUMdg02"), ("SUMdg12")]
    g += ["X1", "Xdg0", "Xdg1", "Hdg1"]
    return g
def inv(word):
    out = []
    for w in reversed(word):
        if w.startswith("SUMdg"): out.append("SUM" + w[5:])
        elif w.startswith("SUM"): out.append("SUMdg" + w[3:])
        else:
            b, q = w[:-1], w[-1]
            out.append((b[:-2] if b.endswith("dg") else b + "dg") + q)
    return out
def mat(gl, nq):
    U = np.eye(3 ** nq, dtype=complex)
    for g in normalise(gl):
        U = embed(g, nq) @ U
    return U
def embed(g, nq):
    k = len(g.qubits); others = [q for q in range(nq) if q not in g.qubits]
    perm = list(g.qubits) + others
    full = np.kron(g.U, np.eye(3 ** (nq - k)))
    full = full.reshape([3] * (2 * nq))
    inv_p = np.argsort(perm)
    full = full.transpose(list(inv_p) + [nq + i for i in inv_p])
    return full.reshape(3 ** nq, 3 ** nq)
# check: parallel C2X / L4 against the serial ones on |psi>|0..0>
def act(gl, nq, psi):
    st = np.zeros(3 ** nq, dtype=complex); st[:3 ** (nq - 0)] = 0
    full = np.kron(psi, np.eye(3 ** (nq - 2 if psi.size == 9 else nq - 1))[:, 0])
    return mat(gl, nq) @ full
rng = np.random.default_rng(1)
L4par = c2x_par() + ["T1"] + inv(c2x_par())
err1 = err2 = 0
for _ in range(20):
    p9 = rng.normal(size=9) + 1j * rng.normal(size=9); p9 /= np.linalg.norm(p9)
    a = act(c2x_par(), 4, p9); b = np.kron(cc.C2X and mat(cc.C2X, 2) @ p9, np.eye(9)[:, 0])
    err1 = max(err1, np.abs(a - np.vdot(b, a) * b).max())
    p3 = rng.normal(size=3) + 1j * rng.normal(size=3); p3 /= np.linalg.norm(p3)
    a = act(L4par, 4, p3); b = np.kron(np.diag([1, 1, cc.Z9]) @ p3, np.eye(27)[:, 0])
    err2 = max(err2, np.abs(a - np.vdot(b, a) * b).max())
print("parallel C2X == C2X (anc 2,3 clean): err %.1e" % err1)
print("L4 with 3 ancillas == diag(1,1,zeta) (x) |000>: err %.1e" % err2)
report("L4 unitary, 3 anc (parallel C2X)", L4par)
report("L4 measurement, 3 anc (parallel C2X)", c2x_par() + ["T1", Gate("M", (1,), basis="F")])
