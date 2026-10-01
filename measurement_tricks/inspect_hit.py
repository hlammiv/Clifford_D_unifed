import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meascore import *
hits = pickle.load(open('search1_R.pkl', 'rb'))
h4 = [h for h in hits if h[0] == 4 and h[1] > 1-1e-9]
print(len(h4))
R = np.diag([1, 1, -1]).astype(complex)
for h in h4[:3]:
    dep, ps, sq, path, q, ci, lab = h
    V = E0.copy()
    for r in list(sq) + list(path): V = ROTS[r] @ V
    print('seq', list(sq)+list(path), 'Paulis', [P2lab[r+1] for r in list(sq)+list(path)], 'Q', P2lab[QLINES[q]])
    for p, B in blocks(V, q):
        Y = B @ G1[ci].conj().T @ R.conj().T
        print('  p=%.4f' % p, 'corr Clifford:', is_cliff1(Y, P1A)); print(np.round(Y / Y.ravel()[np.argmax(np.abs(Y.ravel()) > 1e-6)], 3))
