"""build_r4.py -- turn the 4-T measurement-model hit for R found by search1.py into an explicit gate list
(r_meas_4T.json).  Rotations (in order) f(P) about the 2-qutrit Paulis P (system (x) ancilla), then the
ancilla is measured in the computational basis."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gates import *
from meascore import P2, P2lab, ROTS, E0
SEQ = [2, 14, 4, 24]                    # ROTS indices (P2 index = r+1), from search1_R.pkl (C' = identity)
gl = []
for r in SEQ:
    P = P2[r+1]; g = rot_gates(P)
    assert np.allclose(circuit_matrix(g), ROTS[r]), 'rotation word mismatch'
    gl += g
gl = peephole(simplify(peephole(gl)))
U = circuit_matrix(gl)
V = U @ E0
Vref = E0.copy()
for r in SEQ: Vref = ROTS[r] @ Vref
assert np.allclose(V, Vref)
R = np.diag([1, 1, -1])
corr = {}
for m in range(3):
    amp = np.array([V[3*x+m, x] for x in range(3)])
    ratio = amp / amp[0] / np.diag(R)                 # should be omega powers
    c = [int(np.rint(np.angle(r)/(2*np.pi/3))) % 3 for r in ratio]
    assert np.allclose(ratio, [w**k for k in c]), ratio
    corr[m] = [(-k) % 3 for k in c]                    # correction diag(w^{-c(x)})
json.dump({'target': 'R = diag(1,1,-1)', 'qutrits': '0 = system, 1 = clean ancilla |0>',
           'gates': gl, 'T_count': tcount(gl),
           'rotation_paulis_(a1,b1,a2,b2)=X^a1Z^b1(x)X^a2Z^b2': [P2lab[r+1] for r in SEQ],
           'measure': 'ancilla in computational basis, outcome m',
           'correction_on_system': {str(m): f'diag(w^{c[0]}, w^{c[1]}, w^{c[2]})' for m, c in corr.items()},
           'correction_exponents': {str(m): c for m, c in corr.items()}},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r_meas_4T.json'), 'w'), indent=1)
print('T-count', tcount(gl), 'gates', len(gl)); print(' '.join(gl)); print('corrections', corr)
