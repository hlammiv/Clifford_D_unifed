"""Rebuild Yeh-van de Wetering arXiv:2204.00552 Cor. 4 at k=1 (= diag(1,1,zeta9) on the control qutrit)
numerically from its lemmas and tally the T-count.  Base blocks and their T-counts:
  |2>-ctrl X^{+-1}       : 3 T   (GRVY arXiv:2202.09235 Lemma, explicit circuit re-verified below)
  |2>-ctrl tau_(01/02/12): 15 T  (GRVY 2202.09235 Lemma 'tcd12'; = 5 controlled-X of 3T)
Wires: 0 = system (control), 1 = 'target' of zeta*I (borrowed in effect), 2 = borrowed ancilla (Lemma 11)."""
import numpy as np, itertools
z = np.exp(2j*np.pi/9); w = z**3; I3 = np.eye(3)
X = np.roll(I3, 1, axis=0); Xd = X.T; Zg = np.diag([1, w, w*w]); Zd = Zg.conj()
T = np.diag([1, z, z**8]); Td = T.conj()
H = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(3)
def perm(p): M = np.zeros((3, 3)); [M.__setitem__((p[i], i), 1) for i in range(3)]; return M
X01, X02, X12 = perm([1, 0, 2]), perm([2, 1, 0]), perm([0, 2, 1])
N = 3
def op1(U, q):
    mats = [I3]*N; mats[q] = U; out = mats[0]
    for m in mats[1:]: out = np.kron(out, m)
    return out
def ctrl(U, ctrls, tgt):
    """ctrls: dict wire->value; applies U on tgt iff all controls match."""
    D = 3**N; M = np.zeros((D, D), complex)
    for idx in itertools.product(range(3), repeat=N):
        fire = all(idx[c] == v for c, v in ctrls.items())
        col = np.ravel_multi_index(idx, [3]*N)
        for o in range(3):
            amp = U[o, idx[tgt]] if fire else (1.0 if o == idx[tgt] else 0.0)
            if amp != 0:
                j = list(idx); j[tgt] = o; M[np.ravel_multi_index(j, [3]*N), col] += amp
    return M
def eq_up_to_phase(A, B):
    ph = np.vdot(B, A)/np.vdot(B, B); return abs(abs(ph)-1) < 1e-10 and np.allclose(A, ph*B), ph
prod = lambda *Ms: __import__('functools').reduce(lambda a, b: b @ a, Ms)   # time order left->right
tally = {}
# (0) explicit 3-T |2>-ctrl X (same word as l4_7T_verify.py), 2-qutrit check
SUM = np.zeros((9, 9))
for a in range(3):
    for b in range(3): SUM[3*a+(a+b) % 3, 3*a+b] = 1
k2 = lambda U, q: np.kron(U, I3) if q == 0 else np.kron(I3, U)
word = [k2(H, 1), k2(X, 0), k2(X, 1)] + [k2(Xd, 1), k2(T, 1), k2(X, 1), SUM]*3 + [k2(Xd, 0), k2(Xd, 1), k2(H.conj().T, 1)]
C2X2 = np.kron(np.diag([1, 1, 0]), I3) + np.kron(np.diag([0, 0, 1]), X)
print("[0] 3-T |2>-ctrl X word exact (up to phase):", eq_up_to_phase(prod(*word), C2X2)[0])
# (1) Lemma 9 / Eq.(12),(24), k=1: X01 T^dag X01 . C2-X . X01 T X01 . C2-X^dag  on wire 1, control wire 0
# Eq.(24) read in the orientation that yields |2>-ctrl zeta*S_plain^dag (= their S^dag, since their S := zeta^-1 diag(1,1,w))
L9 = prod(op1(X01 @ T @ X01, 1), ctrl(X, {0: 2}, 1), op1(X01 @ Td @ X01, 1), ctrl(Xd, {0: 2}, 1))
# what single-qutrit gate is applied when control=2?
blk = L9.reshape(3, 3, 3, 3, 3, 3)[2, :, 0, 2, :, 0]
print("[1] Lemma 9 block (ctrl=2):", np.round(np.angle(np.diag(blk))/(2*np.pi/9)) % 9, "(zeta exponents); T-count 1+3+1+3 = 8")
Sd_z = blk                                  # = |2>-ctrl (zeta^a S^dag-type), exact
# (2) Lemma 1 / Eq.(14) with V = X01: search control values of the SW pattern giving |22>-ctrl X01 (ctrls 0,1 ; tgt 2)
tgt = ctrl(X01, {0: 2, 1: 2}, 2); found = None
for a, b, c, s1, s2 in itertools.product(range(3), range(3), range(3), [X, Xd], [X, Xd]):
    for c1, c2, c3 in itertools.product([0, 1], repeat=3):
        M = prod(ctrl(X01, {c1: a}, 2), ctrl(s1, {0: 2}, 1), ctrl(X01, {c2: b}, 2), ctrl(s2, {0: 2}, 1), ctrl(X01, {c3: c}, 2))
        if eq_up_to_phase(M, tgt)[0]: found = (a, b, c, c1, c2, c3); break
    if found: break
print("[2] Lemma 1 (V=X01) -> |22>-ctrl X01 with 3 C-X01 + 2 C-X+-1:", found is not None, found, " T =", 3*15+2*3)
TccX01 = 3*15 + 2*3
# (3) Lemma 3 / Eq.(17): |22>-ctrl X+1 from 2 |2>-ctrl X+-1 and 2 |22>-ctrl X12
tgt = ctrl(X, {0: 2, 1: 2}, 2); ok17 = False
for sA, sB, ca, cb in itertools.product([X, Xd], [X, Xd], [0, 1], [0, 1]):
    M = prod(ctrl(sA, {ca: 2}, 2), ctrl(X12, {0: 2, 1: 2}, 2), ctrl(sB, {cb: 2}, 2), ctrl(X12, {0: 2, 1: 2}, 2))
    if eq_up_to_phase(M, tgt)[0]: ok17 = True; break
TccX = 2*3 + 2*TccX01
print("[3] Lemma 3 |22>-ctrl X+1 identity holds:", ok17, " T =", TccX, "(|22>-ctrl Z^-1 = H-conjugate, same T)")
# (4) Lemma 11 / Eq.(26) k=1: |2>-ctrl Z(0,1) on wire 1 via two |22>-ctrl Z^-1 on borrowed wire 2 + X02
CCZd = ctrl(Zd, {0: 2, 1: 2}, 2)
L11 = prod(CCZd, op1(X02, 2), CCZd, op1(X02, 2))
print("[4] Lemma 11 = |2>-ctrl diag(1,1,w) on wire1 (any wire-2 state):",
      eq_up_to_phase(L11, ctrl(np.diag([1, 1, w]), {0: 2}, 1))[0], " T =", 2*TccX)
# (5) Cor. 4, k=1: Lemma 11 . Lemma 9  ==> diag(1,1,zeta^?) (x) I (x) I
full = L11 @ L9
d = np.diag(full); offd = np.abs(full - np.diag(d)).max()
ex = np.round(np.angle(d/d[0])/(2*np.pi/9)).astype(int) % 9
print("[5] Cor 4 k=1 product diagonal:", offd < 1e-10, " zeta-exponents (sys,wire1,wire2) pattern:",
      ex.reshape(3, 9)[:, 0], "; constant over wires 1,2:", np.allclose(d.reshape(3, 9)/d.reshape(3, 9)[:, :1], 1))
print("    TOTAL T-count (Cor 4, k=1) =", 2*TccX + 8, " using 2 extra qutrits (ζI target + borrowed ancilla), both returned")
print("    diag exponents grid (rows=sys, cols=wire1*3+wire2):\n", ex.reshape(3, 9))
