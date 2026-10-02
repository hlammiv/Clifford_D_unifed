"""ct_sde.py — sde_chi (chi = 1 - zeta_9) of single-qutrit C+T operators vs T-count, exact ring arithmetic.
Random canonical rotation sequences T(P_t)...T(P_1) (consecutive axes distinct, T or T^dag)."""
import itertools, sys, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hrsa"))
import canonical_reducer as cr
H = cr.gate_H(); S = cr.gate_Dcyclo(0, 0, 3); T = cr.gate_Dcyclo(0, 1, 8); Td = cr.gate_Dcyclo(0, 8, 1)
mm = cr._fast_matmul; ct = cr._fast_conjugate_transpose
def num(M):
    B = np.array([np.exp(2j * np.pi * k / 9) for k in range(6)])
    return np.array([[np.dot([int(c) for c in z.num.coefs], B) / 3 ** z.denom_pow3 for z in r] for r in M])
def sde(M): return max(cr.sde_chi_full(z) for r in M for z in r)
I = cr.gate_Dcyclo(0, 0, 0)
w = np.exp(2j*np.pi/3); Xn = np.roll(np.eye(3), 1, 0); Zn = np.diag([1, w, w*w])
axes = [Zn, Xn, Xn @ Zn, Xn @ Zn @ Zn]
conj = {}
for L in range(0, 6):
    for word in itertools.product("HS", repeat=L):
        C = I
        for g in word: C = mm(H if g == "H" else S, C)
        Q = num(C) @ Zn @ num(C).conj().T
        for a, P in enumerate(axes):
            if a not in conj and abs(abs(np.vdot(P, Q)) - 3) < 1e-8: conj[a] = C
    if len(conj) == 4: break
rots = {(a, s): mm(mm(conj[a], T if s == 1 else Td), ct(conj[a])) for a in range(4) for s in (1, -1)}
print("sde of the 8 rotations:", {k: sde(v) for k, v in rots.items()})
random.seed(1)
for t in range(1, 25, 2):
    vals = []
    for _ in range(40):
        U = I; last = None
        for _ in range(t):
            a = random.choice([x for x in range(4) if x != last]); last = a
            U = mm(rots[(a, random.choice((1, -1)))], U)
        vals.append(sde(U))
    print(f"t={t:2d}  sde_chi min {min(vals)}  mean {np.mean(vals):.2f}  max {max(vals)}", flush=True)
