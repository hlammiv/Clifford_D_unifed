"""controls.py -- positive controls for meascore.test_one / full_check."""
import sys, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meascore import *
os.chdir('/home/hlamm/Desktop/efficent_gates/unified/level4')
exec(open("l4_from_8T.py").read().split("# 2)")[0])
R = np.diag([1, 1, -1]).astype(complex); L = np.diag([1, 1, z])
sR, sL, sI = sig(R), sig(L), sig(np.eye(3))
print('sig distinct R/L/I:', np.abs(sR-sI).max() > 1e-3, np.abs(sL-sI).max() > 1e-3, np.abs(sR-sL).max() > 1e-3)
U = np.kron(I3, np.diag([1, z, z**8])) @ c2x
V = U @ E0
out = np.zeros(40, np.int64)
k = test_one(V, QSRC, QPH, KQ, P1A, sI, sL, out); print('L4 meas control: hits', k, [full_check(V, n, L) for n in out[:k]][:2])
# 7T unitary R
from importlib import util
CIRC = open('/home/hlamm/Desktop/efficent_gates/unified/r_from_d/verify_R7T.py').read().split('CIRCUIT = (')[1].split(')')[0]
CIRC = eval('(' + CIRC + ')')
Gm = {'H': H, 'S': np.diag([1, 1, w]), 'X': X, 'Z': np.diag([1, w, w*w]), 'T': np.diag([1, z, z**8]), 'Tdg': np.diag([1, z**8, z])}
U = np.eye(9, dtype=complex)
for g in CIRC.split():
    if g == 'SUM01': M = SUM(1)
    elif g == 'SUM10':
        M = np.zeros((9, 9))
        for a in range(3):
            for b in range(3): M[3*((a+b) % 3)+b, 3*a+b] = 1
    else: M = np.kron(Gm[g[:-1]], I3) if g[-1] == '0' else np.kron(I3, Gm[g[:-1]])
    U = M @ U
V = U @ E0
k = test_one(V, QSRC, QPH, KQ, P1A, sI, sR, out); print('R 7T control: hits', k, full_check(V, out[0], R) if k else None)
