import numpy as np, time, sys
sys.path.insert(0, '/home/hlamm/Desktop/efficent_gates/unified/two_ancilla')
from c3core import *
rng = np.random.default_rng(7)
def rand_state(d):
    M = E00.copy()
    for _ in range(d): M = ROT[rng.integers(728)] @ M
    return M
# check (X^a Z^b)^3 = I, f(P) unitary, clifford_from_images on identity
assert all(np.allclose(np.linalg.matrix_power(PM[k], 3), np.eye(D)) for k in range(0, 729, 37))
U = clifford_from_images(STD); print("id-images unitary:", np.allclose(U.conj().T @ U, np.eye(D)))
t0 = time.time(); npos = 0
for trial in range(30):
    d = trial % 5
    M1 = rand_state(d); C = random_clifford(rng); G = CL1[rng.integers(216)]
    M2 = np.exp(2j*np.pi*rng.random()) * C @ M1 @ G
    r = equiv(M1, M2)
    assert r is not None, ("POSITIVE CONTROL FAILED", trial, d)
    Cf, Gf, ph = r
    assert np.allclose(Cf.conj().T @ Cf, np.eye(D)) and np.allclose(Cf @ M1 @ Gf, ph*M2)
    # Cf must be Clifford: maps Paulis to Paulis
    for k in (1, 3, 9, 27, 81, 243):
        Y = Cf @ PM[k] @ Cf.conj().T
        assert np.max(np.abs(np.einsum('pij,ij->p', PM.conj(), Y)))/27 > 1-1e-9
    npos += 1
print(f"positive controls passed: {npos}  ({time.time()-t0:.1f}s, stats {STATS})")
# negative controls: complex conjugate (anti-symplectic label relabelling) and other random states with equal hash
t0 = time.time(); nneg = 0; nhash = 0
for trial in range(30):
    d = 1 + trial % 4
    M1 = rand_state(d)
    for M2 in (np.conj(M1), rand_state(d), ROT[rng.integers(728)] @ M1):
        same_hash = inv_hash(qlabels(M1[None]))[0] == inv_hash(qlabels(M2[None]))[0]
        r = equiv(M1, M2)
        if r is not None:
            Cf, Gf, ph = r; assert np.allclose(Cf @ M1 @ Gf, ph*M2)
        else:
            nneg += 1; nhash += same_hash
print(f"negative (non-equivalent) outcomes: {nneg}, of which same invariant hash: {nhash} ({time.time()-t0:.1f}s)")
print("margin", MARGIN[0])
