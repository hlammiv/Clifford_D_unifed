"""verify_R7T.py -- standalone check: R = diag(1,1,-1) on a qutrit with ONE clean ancilla and T-count 7.
Qutrits: 0 = system, 1 = ancilla (input |0>, output |0>).  Gates in time order (left -> right):
  H = (1/sqrt(-3)) [w^{jk}],  S = diag(1,1,w),  X|j>=|j+1>,  Z = diag(1,w,w^2),  T = diag(1,z,z^8),  Tdg = T^dag,
  SUM01 |x0,x1> -> |x0, x1+x0>,   SUM10 |x0,x1> -> |x0+x1, x1>,   z = e^{2 pi i/9}, w = z^3.
Found by exact meet-in-the-middle search (mitm_core.py); minimal over 1-ancilla circuits (see raw_lb_1anc.py)."""
import itertools, numpy as np
CIRCUIT = ("H0 H0 SUM10 SUM01 SUM01 H0 T0 Z0 Z0 SUM01 S1 S1 SUM10 S0 SUM01 SUM01 T0 X0 H0 SUM01 S1 SUM01 H0 S1 "
           "SUM01 H0 T0 Z0 X0 Z1 X1 X1 SUM10 S0 SUM01 SUM01 H0 SUM10 Tdg0 SUM10 H1 S1 SUM01 H0 SUM01 H0 Tdg0 Z0 "
           "X1 H1 S0 SUM01 SUM01 S1 S1 SUM10 SUM01 Tdg0 H0 SUM01 SUM01 SUM10 Tdg0")
z = np.exp(2j*np.pi/9); w = z**3; I3 = np.eye(3)
G = {'H': np.array([[w**(j*k) for k in range(3)] for j in range(3)])/np.sqrt(-3+0j),
     'S': np.diag([1, 1, w]), 'X': np.roll(I3, 1, axis=0), 'Z': np.diag([1, w, w*w]),
     'T': np.diag([1, z, z**8]), 'Tdg': np.diag([1, z**8, z])}
def SUM(c):
    M = np.zeros((9, 9))
    for x0, x1 in itertools.product(range(3), repeat=2):
        M[(3*x0+(x1+x0) % 3) if c == 0 else (3*((x0+x1) % 3)+x1), 3*x0+x1] = 1
    return M
def run(CIRCUIT, R, label):
    U = np.eye(9, dtype=complex); tcount = 0
    for g in CIRCUIT.split():
        if g.startswith('SUM'): M = SUM(0 if g == 'SUM01' else 1)
        else:
            name, q = g[:-1], int(g[-1]); M = np.kron(G[name], I3) if q == 0 else np.kron(I3, G[name])
            tcount += name in ('T', 'Tdg')
        U = M @ U
    assert np.allclose(U @ U.conj().T, np.eye(9), atol=1e-13)
    ket0 = np.array([1, 0, 0])
    lam = (U @ np.kron([1, 0, 0], ket0))[0]          # global phase
    print(f'{label}: T-count = {tcount},  total gates = {len(CIRCUIT.split())},  global phase = {lam:.12f} (|.|={abs(lam):.15f})')
    err = np.abs(U @ np.kron(I3, ket0[:, None]) - lam*np.kron(R, ket0[:, None])).max()
    print(f'max |U (I (x) |0>) - lam (R (x) |0>)| over basis inputs = {err:.2e}')
    rng = np.random.default_rng(7); worst = 0
    for _ in range(1000):
        psi = rng.normal(size=3)+1j*rng.normal(size=3); psi /= np.linalg.norm(psi)
        worst = max(worst, np.abs(U @ np.kron(psi, ket0) - lam*np.kron(R @ psi, ket0)).max())
    print(f'1000 random states: max deviation = {worst:.2e}   ->', 'PASS' if worst < 1e-12 and err < 1e-12 else 'FAIL')

run(CIRCUIT, np.diag([1, 1, -1]), 'R = diag(1,1,-1)')
# bonus: level-4 diagonal diag(1,1,zeta) with one clean ancilla, also 7 T (previous best 8)
CIRCUIT_D4 = "H0 H0 SUM10 SUM01 SUM01 H0 T0 Z1 Z1 SUM10 S0 S0 SUM01 S1 SUM01 T0 SUM01 SUM10 SUM01 SUM01 T0 Z0 X0 X0 Z1 X1 SUM01 SUM01 SUM10 SUM10 H1 SUM10 SUM01 Tdg0 H0 H0 SUM10 SUM01 H0 SUM01 SUM10 Tdg0 SUM10 SUM01 SUM10 SUM10 Tdg0 Z0 Z1 S0 SUM01 SUM10 S0 S0 SUM10 Tdg0 H0 SUM01 SUM01 SUM10"
run(CIRCUIT_D4, np.diag([1, 1, z]), 'diag(1,1,zeta)')
