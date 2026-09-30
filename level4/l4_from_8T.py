"""l4_from_8T.py — verify: the level-4 gate diag(1,1,zeta^7) (= diag(1,1,zeta) x Clifford)
is implemented EXACTLY by Clifford+T with T-count 8 and one clean |0> ancilla.

Built from Glaudell-Ross-van de Wetering-Yeh (arXiv:2202.09235) circuits, rebuilt from
their tikz figures (tcxp1, tcspdagphase, tczwwphase):
  |2>-ctrl X        : 3 T  (H, X, [P9, SUM]x3 , X^dag, H^dag ;  P9 = X T X^dag)
  |2>-ctrl zeta S^dag : C2(X) . t01 . T^dag . t01 . C2(X^dag) . t01 . T . t01   (8 T)
  |2>-ctrl zeta^7 Z(1,1) : t02 . C2(zeta S^dag) . t02                           (8 T)
Applied with target (ancilla) in |0>, Z(1,1)|0> = |0>, so the controlled global phase
zeta^7 kicks back onto the control: diag(1,1,zeta^7) (x) |0>.  Every gate used is
Clifford except T / T^dag (P9 is Clifford-conjugate of T), so T-count = 8.
"""
import itertools
import numpy as np

z = np.exp(2j*np.pi/9); w = z**3; I3 = np.eye(3)
H = np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(-3+0j)
X = np.roll(I3, 1, axis=0)                        # |j> -> |j+1>
T = np.diag([1, z, z**8])
P9 = X @ T @ X.conj().T
def perm(p):
    M = np.zeros((3, 3)); [M.__setitem__((p[j], j), 1) for j in range(3)]; return M
t01, t02 = perm([1, 0, 2]), perm([2, 1, 0])
C = lambda A: np.kron(A, I3)                      # on control (qutrit 0)
Tg = lambda A: np.kron(I3, A)                     # on target (qutrit 1)
def SUM(sign=1):                                  # |x,y> -> |x, y + sign*x>
    M = np.zeros((9, 9))
    for x in range(3):
        for y in range(3): M[3*x+(y+sign*x) % 3, 3*x+y] = 1
    return M
def ctrl2(U):                                     # exact |2>-controlled U (reference)
    return np.kron(np.diag([1, 1, 0]), I3) + np.kron(np.diag([0, 0, 1]), U)
def seq(*gates):                                  # time order left -> right
    M = np.eye(9, dtype=complex)
    for g in gates: M = g @ M
    return M

# 1) find the tcxp1 wiring (SUM direction / sign, X vs X^dag placement) that gives C2(X) with 3 T
target = ctrl2(X); c2x = None
for s, (a, b) in itertools.product((1, -1), [(X, X.conj().T), (X.conj().T, X)]):
    M = seq(Tg(H), C(a), Tg(a), Tg(P9), SUM(s), Tg(P9), SUM(s), Tg(P9), SUM(s), C(b), Tg(b), Tg(H.conj().T))
    for ph in [z**k for k in range(9)] + [-z**k for k in range(9)]:
        if np.allclose(M*ph, target):
            c2x = M*ph; print(f"C2(X) found: SUM sign {s}, global phase fix {np.round(ph,3)} (3 T)"); break
    if c2x is not None: break
assert c2x is not None, "could not reproduce |2>-ctrl X"
c2xd = c2x.conj().T                               # |2>-ctrl X^dag, 3 T

# 2) |2>-ctrl (zeta S^dag), 8 T
M = seq(c2x, Tg(t01), Tg(T.conj().T), Tg(t01), c2xd, Tg(t01), Tg(T), Tg(t01))
S = np.diag([1, 1, w])
cands = {f"zeta^{k} S^dag": ctrl2(z**k * S.conj().T) for k in range(9)}
cands.update({f"zeta^{k} S^dag (S=diag(1,w,1))": ctrl2(z**k*np.diag([1, w, 1]).conj().T) for k in range(9)})
hit = [k for k, v in cands.items() if np.allclose(M, v)]
print("8-T block equals |2>-ctrl of:", hit)

# 3) |2>-ctrl (zeta^7 Z(1,1)) = t02 . block . t02 (up to which phase?)
M2 = seq(Tg(t02), M, Tg(t02))
Z11 = np.diag([1, w, w])
hit2 = [k for k in range(9) if np.allclose(M2, ctrl2(z**k*Z11))]
print("t02-conjugated block equals |2>-ctrl zeta^k Z(1,1) for k =", hit2)

# 4) end-to-end: ancilla |0>  ->  diag(1,1,zeta^k) on control, ancilla returned to |0>
rng = np.random.default_rng(0)
for _ in range(3):
    psi = rng.normal(size=3)+1j*rng.normal(size=3); psi /= np.linalg.norm(psi)
    out = M2 @ np.kron(psi, [1, 0, 0])
    k = hit2[0]
    print("  exact diag(1,1,zeta^%d) (x) |0>:" % k, np.allclose(out, np.kron(np.diag([1, 1, z**k]) @ psi, [1, 0, 0])))
k = hit2[0]
print(f"diag(1,1,zeta^{k}) = diag(1,1,zeta) * diag(1,1,zeta^{(k-1)%9}); second factor Clifford: {(k-1)%3==0}")
print(f"level of diag(1,1,zeta^{k}): {'3' if k%3==0 else '4'} (exponent sum {k} mod 3 = {k%3})")
